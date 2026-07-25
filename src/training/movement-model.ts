import type { ReplayManifest } from "../capture/replay.js";
import type { TrajectoryAction, TrajectorySpan } from "../capture/trajectory.js";

// ---------------------------------------------------------------------------
// Local-movement learning subsystem — in-process model backend.
//
// This module closes the objective-#2(c)/(d) gap: the training runner already
// emits *external* command plans for real Apple-Silicon runtimes (mlx/axolotl),
// but nothing in the repo could actually learn from a movement dataset and
// infer movements without that hardware. This provides a pluggable backend
// interface plus a deterministic, dependency-free Markov backend that:
//   (c) post-trains on a recorded movement dataset and reproduces the recorded
//       movements, and
//   (d) generalizes to new-but-related movements via variable-order backoff.
//
// The backend is fully deterministic (seeded PRNG, alphabetical tie-breaks) so
// it validates in the cloud/CI on synthetic data, while `createMovementModelBackend`
// leaves a documented seam ("local-native") for a real on-device small model.
// ---------------------------------------------------------------------------

/** A canonical, discrete movement symbol, e.g. `"tap:submit"` or `"swipe:up"`. */
export type MovementToken = string;

/** Sentinels marking sequence boundaries in the learned model. */
export const MOVEMENT_START_TOKEN = "␂start" as const;
export const MOVEMENT_END_TOKEN = "␃end" as const;

export type MovementSequence = {
  id: string;
  tokens: MovementToken[];
};

export type MovementDataset = {
  sequences: MovementSequence[];
};

export type MovementTrainOptions = {
  /** Maximum Markov context length. Defaults to 2 (trigram with backoff). */
  order?: number;
};

export type MovementPrediction = {
  token: MovementToken;
  probability: number;
  /** Context length actually used after backoff (0 = unconditional). */
  order: number;
  alternatives: { token: MovementToken; probability: number }[];
};

export type MovementGenerateOptions = {
  prefix?: MovementToken[];
  maxLength: number;
  /** When set, sample stochastically with this seed; otherwise greedy argmax. */
  seed?: number;
  /** Softmax-style sharpening for sampling; ignored when greedy. */
  temperature?: number;
};

export type MovementFidelity = {
  transitions: number;
  correct: number;
  accuracy: number;
};

export type SerializedMovementPolicy = {
  version: 1;
  backendId: string;
  order: number;
  vocabulary: MovementToken[];
  /** context-key -> [token, count][] */
  contexts: Array<{ context: string; counts: Array<[MovementToken, number]> }>;
};

export interface MovementPolicy {
  readonly backendId: string;
  readonly order: number;
  readonly vocabulary: MovementToken[];
  /** Most-likely next token given recent context (with backoff), or undefined if the model is empty. */
  predictNext(context: MovementToken[]): MovementPrediction | undefined;
  /** Roll out a movement sequence from an optional prefix. */
  generate(options: MovementGenerateOptions): MovementToken[];
  /** Top-1 reproduction fidelity against a known sequence. */
  score(sequence: MovementSequence): MovementFidelity;
  serialize(): SerializedMovementPolicy;
}

export interface MovementModelBackend {
  readonly id: string;
  /** Whether this backend can train in the current environment (cloud/CI vs on-device). */
  readonly available: boolean;
  train(dataset: MovementDataset, options?: MovementTrainOptions): MovementPolicy;
}

// ---------------------------------------------------------------------------
// Tokenization — turn captured movement actions into discrete symbols.
// ---------------------------------------------------------------------------

export type MovementActionLike = {
  tool: string;
  summary: string;
  metadata?: Record<string, unknown>;
};

const CONTEXT_SEPARATOR = "␟";

function slug(value: string): string {
  return value
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 40);
}

function metadataString(metadata: Record<string, unknown> | undefined, key: string): string | undefined {
  const value = metadata?.[key];
  return typeof value === "string" && value.trim().length > 0 ? value : undefined;
}

/**
 * Map a captured movement action to a canonical token. Prefers structured
 * gesture metadata (kind/direction/target) and falls back to tool + summary so
 * every action produces a stable, learnable symbol.
 */
export function tokenizeMovementAction(action: MovementActionLike): MovementToken {
  const verb = metadataString(action.metadata, "gesture") ?? action.tool;
  const object =
    metadataString(action.metadata, "target") ??
    metadataString(action.metadata, "direction") ??
    action.summary;
  const verbSlug = slug(verb) || "act";
  const objectSlug = slug(object) || "unknown";
  return `${verbSlug}:${objectSlug}`;
}

function sortByTs<T extends { ts: number }>(items: readonly T[]): T[] {
  return [...items].sort((a, b) => a.ts - b.ts);
}

/** Build a dataset (one sequence per trajectory) from trajectory spans. */
export function datasetFromTrajectories(trajectories: readonly TrajectorySpan[]): MovementDataset {
  const sequences = trajectories.map((trajectory) => ({
    id: trajectory.id,
    tokens: sortByTs<TrajectoryAction>(trajectory.actions).map(tokenizeMovementAction),
  }));
  return { sequences: sequences.filter((sequence) => sequence.tokens.length > 0) };
}

/** Build a dataset (one sequence per manifest) from replay manifests. */
export function datasetFromReplayManifests(manifests: readonly ReplayManifest[]): MovementDataset {
  const sequences = manifests.map((manifest) => {
    const actions = manifest.events.flatMap((event) =>
      event.kind === "action" ? [{ tool: event.tool, summary: event.summary, ts: event.ts }] : [],
    );
    return {
      id: manifest.sessionId,
      tokens: sortByTs(actions).map((action) => tokenizeMovementAction(action)),
    };
  });
  return { sequences: sequences.filter((sequence) => sequence.tokens.length > 0) };
}

// ---------------------------------------------------------------------------
// Deterministic seeded PRNG (mulberry32) — reproducible sampling in tests.
// ---------------------------------------------------------------------------

export function createSeededRng(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// ---------------------------------------------------------------------------
// Markov backend — variable-order n-gram with backoff.
// ---------------------------------------------------------------------------

class MarkovMovementPolicy implements MovementPolicy {
  readonly backendId: string;
  readonly order: number;
  readonly vocabulary: MovementToken[];
  private readonly contexts: Map<string, Map<MovementToken, number>>;

  constructor(params: {
    backendId: string;
    order: number;
    vocabulary: MovementToken[];
    contexts: Map<string, Map<MovementToken, number>>;
  }) {
    this.backendId = params.backendId;
    this.order = params.order;
    this.vocabulary = params.vocabulary;
    this.contexts = params.contexts;
  }

  private distributionFor(context: MovementToken[]): {
    counts: Map<MovementToken, number>;
    orderUsed: number;
  } | undefined {
    const maxContext = Math.min(this.order, context.length);
    for (let k = maxContext; k >= 0; k -= 1) {
      const suffix = context.slice(context.length - k);
      const counts = this.contexts.get(suffix.join(CONTEXT_SEPARATOR));
      if (counts && counts.size > 0) {
        return { counts, orderUsed: k };
      }
    }
    return undefined;
  }

  private predictInternal(context: MovementToken[]): {
    prediction: MovementPrediction;
    orderUsed: number;
  } | undefined {
    const found = this.distributionFor(context);
    if (!found) {
      return undefined;
    }
    const total = [...found.counts.values()].reduce((sum, count) => sum + count, 0);
    // Deterministic ordering: highest probability first, alphabetical tie-break.
    const ranked = [...found.counts.entries()]
      .map(([token, count]) => ({ token, probability: count / total }))
      .sort((a, b) => (b.probability - a.probability) || (a.token < b.token ? -1 : 1));
    const [top, ...rest] = ranked;
    if (!top) {
      return undefined;
    }
    return {
      orderUsed: found.orderUsed,
      prediction: {
        token: top.token,
        probability: top.probability,
        order: found.orderUsed,
        alternatives: rest.slice(0, 4),
      },
    };
  }

  predictNext(context: MovementToken[]): MovementPrediction | undefined {
    const result = this.predictInternal(context.filter((token) => token !== MOVEMENT_START_TOKEN));
    if (!result || result.prediction.token === MOVEMENT_END_TOKEN) {
      return undefined;
    }
    return result.prediction;
  }

  generate(options: MovementGenerateOptions): MovementToken[] {
    const rng = options.seed !== undefined ? createSeededRng(options.seed) : undefined;
    const temperature = options.temperature ?? 1;
    const output: MovementToken[] = [...(options.prefix ?? [])];
    // Pad the working context with START sentinels so the first step matches how
    // the model was trained (start-of-sequence conditioning).
    const working: MovementToken[] = [
      ...Array.from({ length: this.order }, () => MOVEMENT_START_TOKEN),
      ...output,
    ];

    while (output.length < options.maxLength) {
      const context = working.slice(working.length - this.order);
      const found = this.distributionFor(context);
      if (!found) {
        break;
      }
      const next = rng
        ? this.sample(found.counts, rng, temperature)
        : this.argmax(found.counts);
      if (next === undefined || next === MOVEMENT_END_TOKEN) {
        break;
      }
      output.push(next);
      working.push(next);
    }
    return output;
  }

  private argmax(counts: Map<MovementToken, number>): MovementToken | undefined {
    let best: MovementToken | undefined;
    let bestCount = -1;
    for (const [token, count] of counts) {
      if (count > bestCount || (count === bestCount && best !== undefined && token < best)) {
        best = token;
        bestCount = count;
      }
    }
    return best;
  }

  private sample(counts: Map<MovementToken, number>, rng: () => number, temperature: number): MovementToken | undefined {
    const entries = [...counts.entries()].sort((a, b) => (a[0] < b[0] ? -1 : 1));
    const weights = entries.map(([, count]) => Math.pow(count, 1 / Math.max(temperature, 1e-6)));
    const total = weights.reduce((sum, weight) => sum + weight, 0);
    if (total <= 0) {
      return undefined;
    }
    let threshold = rng() * total;
    for (let i = 0; i < entries.length; i += 1) {
      threshold -= weights[i];
      if (threshold <= 0) {
        return entries[i][0];
      }
    }
    return entries[entries.length - 1]?.[0];
  }

  score(sequence: MovementSequence): MovementFidelity {
    const padded = [
      ...Array.from({ length: this.order }, () => MOVEMENT_START_TOKEN),
      ...sequence.tokens,
      MOVEMENT_END_TOKEN,
    ];
    let transitions = 0;
    let correct = 0;
    for (let i = this.order; i < padded.length; i += 1) {
      const context = padded.slice(i - this.order, i);
      const result = this.predictInternal(context);
      transitions += 1;
      if (result && result.prediction.token === padded[i]) {
        correct += 1;
      }
    }
    return {
      transitions,
      correct,
      accuracy: transitions === 0 ? 1 : correct / transitions,
    };
  }

  serialize(): SerializedMovementPolicy {
    return {
      version: 1,
      backendId: this.backendId,
      order: this.order,
      vocabulary: [...this.vocabulary],
      contexts: [...this.contexts.entries()].map(([context, counts]) => ({
        context,
        counts: [...counts.entries()],
      })),
    };
  }
}

export class MarkovMovementBackend implements MovementModelBackend {
  readonly id = "markov";
  readonly available = true;

  train(dataset: MovementDataset, options: MovementTrainOptions = {}): MovementPolicy {
    const order = Math.max(1, Math.floor(options.order ?? 2));
    const contexts = new Map<string, Map<MovementToken, number>>();
    const vocabulary = new Set<MovementToken>();

    for (const sequence of dataset.sequences) {
      const padded = [
        ...Array.from({ length: order }, () => MOVEMENT_START_TOKEN),
        ...sequence.tokens,
        MOVEMENT_END_TOKEN,
      ];
      for (const token of sequence.tokens) {
        vocabulary.add(token);
      }
      for (let i = order; i < padded.length; i += 1) {
        const target = padded[i];
        // Record this transition at every context length 0..order (variable order).
        for (let k = 0; k <= order; k += 1) {
          const context = padded.slice(i - k, i);
          const key = context.join(CONTEXT_SEPARATOR);
          let counts = contexts.get(key);
          if (!counts) {
            counts = new Map<MovementToken, number>();
            contexts.set(key, counts);
          }
          counts.set(target, (counts.get(target) ?? 0) + 1);
        }
      }
    }

    return new MarkovMovementPolicy({
      backendId: this.id,
      order,
      vocabulary: [...vocabulary].sort(),
      contexts,
    });
  }
}

/** Rehydrate a policy previously produced by `MovementPolicy.serialize()`. */
export function loadMovementPolicy(serialized: SerializedMovementPolicy): MovementPolicy {
  const contexts = new Map<string, Map<MovementToken, number>>();
  for (const entry of serialized.contexts) {
    contexts.set(entry.context, new Map(entry.counts));
  }
  return new MarkovMovementPolicy({
    backendId: serialized.backendId,
    order: serialized.order,
    vocabulary: [...serialized.vocabulary],
    contexts,
  });
}

// ---------------------------------------------------------------------------
// Backend registry — pluggable seam for a real on-device model.
// ---------------------------------------------------------------------------

export type MovementBackendKind = "markov" | "local-native";

/**
 * Documented seam for a real on-device small model (e.g. an MLX/ONNX policy).
 * It reports `available: false` and refuses to train in the cloud, so callers
 * can detect the missing runtime and fall back to `markov` without special-casing.
 */
export class LocalNativeMovementBackend implements MovementModelBackend {
  readonly id = "local-native";
  readonly available = false;

  train(): MovementPolicy {
    throw new Error(
      "local-native movement backend requires an on-device runtime; run bee-agent locally or use the 'markov' backend in cloud/CI",
    );
  }
}

export function createMovementModelBackend(kind: MovementBackendKind = "markov"): MovementModelBackend {
  switch (kind) {
    case "markov":
      return new MarkovMovementBackend();
    case "local-native":
      return new LocalNativeMovementBackend();
  }
}

// ---------------------------------------------------------------------------
// Generalization eval harness — measure replay fidelity on held-out data.
// ---------------------------------------------------------------------------

export type MovementSequenceReport = {
  id: string;
  fidelity: MovementFidelity;
  /** Fraction of tokens in the sequence that exist in the training vocabulary. */
  vocabularyCoverage: number;
};

export type MovementGeneralizationReport = {
  sequences: MovementSequenceReport[];
  transitions: number;
  correct: number;
  /** Aggregate top-1 next-movement accuracy across all held-out transitions. */
  accuracy: number;
  /** Aggregate fraction of held-out tokens the model has seen before. */
  vocabularyCoverage: number;
};

export function evaluateGeneralization(
  policy: MovementPolicy,
  heldOut: MovementDataset,
): MovementGeneralizationReport {
  const vocabulary = new Set(policy.vocabulary);
  const sequences: MovementSequenceReport[] = [];
  let transitions = 0;
  let correct = 0;
  let coveredTokens = 0;
  let totalTokens = 0;

  for (const sequence of heldOut.sequences) {
    const fidelity = policy.score(sequence);
    const covered = sequence.tokens.filter((token) => vocabulary.has(token)).length;
    transitions += fidelity.transitions;
    correct += fidelity.correct;
    coveredTokens += covered;
    totalTokens += sequence.tokens.length;
    sequences.push({
      id: sequence.id,
      fidelity,
      vocabularyCoverage: sequence.tokens.length === 0 ? 1 : covered / sequence.tokens.length,
    });
  }

  return {
    sequences,
    transitions,
    correct,
    accuracy: transitions === 0 ? 1 : correct / transitions,
    vocabularyCoverage: totalTokens === 0 ? 1 : coveredTokens / totalTokens,
  };
}
