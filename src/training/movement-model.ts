import type { ReplayManifest, ReplayTimelineEvent } from "../capture/replay.js";
import type { TrajectorySpan } from "../capture/trajectory.js";

/**
 * Local-movement learning: in-process model backend.
 *
 * This module realizes the "post-train a local model on the recorded dataset to
 * repeat the movements, and generalize to related movements" piece of the
 * movement subsystem WITHOUT any OS or GPU dependency, so it trains and infers
 * deterministically in the cloud/CI. The {@link MovementModelBackend} interface
 * is the seam: {@link MarkovMovementBackend} is the always-available, fully
 * deterministic mock; a real on-device small model (via the Apple-Silicon
 * runner) can be dropped in behind the same interface later. A trained model is
 * a plain JSON {@link MovementModelArtifact}, so persistence is trivial.
 */

export type MovementToken = string;

/** Sentinel tokens used for sequence boundaries. They never appear in a dataset. */
export const MOVEMENT_START_TOKEN = "<bos>";
export const MOVEMENT_END_TOKEN = "<eos>";


export type MovementSequence = {
  id: string;
  tokens: MovementToken[];
};

export type MovementDataset = {
  sequences: MovementSequence[];
};

export type MovementTrainOptions = {
  /** N-gram order (context window = order - 1). Defaults to the backend order. */
  order?: number;
};

export type MovementModelArtifact = {
  version: 1;
  backend: string;
  order: number;
  vocabulary: MovementToken[];
  /**
   * Backoff transition counts. Keyed by the JSON-encoded context array (context
   * length 0..order-1); value maps next-token -> observed count.
   */
  transitions: Record<string, Record<MovementToken, number>>;
  sequenceCount: number;
  transitionCount: number;
};

export type MovementCandidate = {
  token: MovementToken;
  probability: number;
};

export type MovementPrediction = {
  /** The argmax next token, or `null` when the model predicts end-of-sequence. */
  token: MovementToken | null;
  probability: number;
  /** How many context tokens were actually used (after backoff); -1 if unknown. */
  backoffOrder: number;
  /** Full next-token distribution at the chosen backoff level, most-likely first. */
  candidates: MovementCandidate[];
};

export type MovementGenerateOptions = {
  maxSteps?: number;
};

export interface MovementModelInference {
  readonly backend: string;
  /** Predict the next movement token given a (possibly novel) context prefix. */
  predictNext(context: MovementToken[]): MovementPrediction;
  /**
   * Generate a continuation for `prompt`. With an empty prompt the model
   * reproduces the most-likely recorded movement; with a related-but-unseen
   * prefix it generalizes via backoff. The returned tokens exclude the prompt.
   */
  generate(prompt: MovementToken[], options?: MovementGenerateOptions): MovementToken[];
}

export interface MovementModelBackend {
  readonly id: string;
  train(dataset: MovementDataset, options?: MovementTrainOptions): MovementModelArtifact;
  createInference(artifact: MovementModelArtifact): MovementModelInference;
}

function contextKey(context: MovementToken[]): string {
  // JSON-encode the array so keys are unambiguous regardless of token contents
  // (no separator can collide with a token, and different lengths never alias).
  return JSON.stringify(context);
}

/**
 * Deterministic n-gram backoff model. Ties are broken lexicographically, so the
 * same dataset always yields the same artifact and the same predictions — which
 * is exactly what makes it testable in the cloud.
 */
export class MarkovMovementBackend implements MovementModelBackend {
  readonly id = "markov-ngram";

  constructor(private readonly defaultOrder = 3) {}

  train(dataset: MovementDataset, options?: MovementTrainOptions): MovementModelArtifact {
    const order = Math.max(1, options?.order ?? this.defaultOrder);
    const transitions: Record<string, Record<MovementToken, number>> = {};
    const vocabulary = new Set<MovementToken>();
    let sequenceCount = 0;
    let transitionCount = 0;

    for (const sequence of dataset.sequences) {
      sequenceCount += 1;
      for (const token of sequence.tokens) {
        vocabulary.add(token);
      }
      const padded = [
        ...Array<MovementToken>(order - 1).fill(MOVEMENT_START_TOKEN),
        ...sequence.tokens,
        MOVEMENT_END_TOKEN,
      ];
      for (let i = order - 1; i < padded.length; i += 1) {
        const target = padded[i]!;
        transitionCount += 1;
        for (let k = 0; k <= order - 1; k += 1) {
          const context = padded.slice(i - k, i);
          const key = contextKey(context);
          const dist = (transitions[key] ??= {});
          dist[target] = (dist[target] ?? 0) + 1;
        }
      }
    }

    return {
      version: 1,
      backend: this.id,
      order,
      vocabulary: [...vocabulary].sort(),
      transitions,
      sequenceCount,
      transitionCount,
    };
  }

  createInference(artifact: MovementModelArtifact): MovementModelInference {
    return new MarkovMovementInference(artifact);
  }
}

class MarkovMovementInference implements MovementModelInference {
  readonly backend: string;

  constructor(private readonly artifact: MovementModelArtifact) {
    this.backend = artifact.backend;
  }

  predictNext(context: MovementToken[]): MovementPrediction {
    const maxK = Math.min(this.artifact.order - 1, context.length);
    for (let k = maxK; k >= 0; k -= 1) {
      const window = context.slice(context.length - k);
      const dist = this.artifact.transitions[contextKey(window)];
      if (!dist) {
        continue;
      }
      const entries = Object.entries(dist);
      let total = 0;
      for (const [, count] of entries) {
        total += count;
      }
      entries.sort((a, b) => (b[1] - a[1]) || (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0));
      const candidates: MovementCandidate[] = entries.map(([token, count]) => ({
        token,
        probability: count / total,
      }));
      const best = candidates[0]!;
      return {
        token: best.token === MOVEMENT_END_TOKEN ? null : best.token,
        probability: best.probability,
        backoffOrder: k,
        candidates,
      };
    }
    return { token: null, probability: 0, backoffOrder: -1, candidates: [] };
  }

  generate(prompt: MovementToken[], options?: MovementGenerateOptions): MovementToken[] {
    const maxSteps = options?.maxSteps ?? 128;
    const window: MovementToken[] = [
      ...Array<MovementToken>(this.artifact.order - 1).fill(MOVEMENT_START_TOKEN),
      ...prompt,
    ];
    const generated: MovementToken[] = [];
    for (let step = 0; step < maxSteps; step += 1) {
      const prediction = this.predictNext(window);
      if (prediction.token === null) {
        break;
      }
      generated.push(prediction.token);
      window.push(prediction.token);
    }
    return generated;
  }
}

// ---------------------------------------------------------------------------
// Tokenizers: turn recorded movement data into learnable token sequences.
// ---------------------------------------------------------------------------

export type TokenizeOptions = {
  /** Include observation events as context tokens (default: actions only). */
  includeObservations?: boolean;
};

export function actionEventToToken(
  event: Extract<ReplayTimelineEvent, { kind: "action" }>,
): MovementToken {
  return `act:${event.tool}:${event.summary}`;
}

export function observationEventToToken(
  event: Extract<ReplayTimelineEvent, { kind: "observation" }>,
): MovementToken {
  return `obs:${event.source}:${event.summary}`;
}

export function replayEventsToTokens(
  events: ReplayTimelineEvent[],
  options?: TokenizeOptions,
): MovementToken[] {
  return events.flatMap((event) => {
    if (event.kind === "action") {
      return [actionEventToToken(event)];
    }
    if (event.kind === "observation" && options?.includeObservations) {
      return [observationEventToToken(event)];
    }
    return [];
  });
}

export function replayManifestToSequence(
  manifest: Pick<ReplayManifest, "sessionId" | "events">,
  options?: TokenizeOptions & { id?: string },
): MovementSequence {
  return {
    id: options?.id ?? manifest.sessionId,
    tokens: replayEventsToTokens(manifest.events, options),
  };
}

export function trajectoryToMovementSequence(
  span: TrajectorySpan,
  options?: TokenizeOptions,
): MovementSequence {
  const actionTokens = [...span.actions]
    .sort((a, b) => a.ts - b.ts)
    .map((action) => `act:${action.tool}:${action.summary}`);
  if (!options?.includeObservations) {
    return { id: span.id, tokens: actionTokens };
  }
  const merged = [
    ...span.observations.map((observation) => ({
      ts: observation.ts,
      token: `obs:${observation.source}:${observation.summary}`,
    })),
    ...span.actions.map((action) => ({
      ts: action.ts,
      token: `act:${action.tool}:${action.summary}`,
    })),
  ].sort((a, b) => a.ts - b.ts);
  return { id: span.id, tokens: merged.map((entry) => entry.token) };
}

export function datasetFromReplayManifests(
  manifests: Array<Pick<ReplayManifest, "sessionId" | "events">>,
  options?: TokenizeOptions,
): MovementDataset {
  return {
    sequences: manifests
      .map((manifest, index) =>
        replayManifestToSequence(manifest, { ...options, id: `${manifest.sessionId}#${index}` }),
      )
      .filter((sequence) => sequence.tokens.length > 0),
  };
}

// ---------------------------------------------------------------------------
// Synthetic event streams: validate capture->dataset->train->replay without OS.
// ---------------------------------------------------------------------------

/** Small deterministic PRNG (mulberry32) so synthetic data is reproducible. */
function createRng(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export type SyntheticMovementOptions = {
  sequenceCount: number;
  /** Reusable movement "motifs" (each a short ordered burst of tokens). */
  motifs?: MovementToken[][];
  /** How many motifs to concatenate per sequence. */
  motifsPerSequence?: number;
  seed?: number;
};

const DEFAULT_MOTIFS: MovementToken[][] = [
  ["act:device:tapped Search", "act:device:typed into Search", "act:device:tapped Result"],
  ["act:device:scrolled down", "act:device:scrolled down", "act:device:tapped Item"],
  ["act:os:focused Editor", "act:os:ran build", "act:os:focused Terminal"],
  ["act:device:swiped left", "act:device:tapped Confirm"],
];

/**
 * Build a structured synthetic dataset: sequences are concatenations of
 * repeated motifs, so an n-gram model provably has learnable structure. Used to
 * validate the capture->dataset->train->replay round-trip in tests.
 */
export function generateSyntheticMovementDataset(
  options: SyntheticMovementOptions,
): MovementDataset {
  const motifs = options.motifs ?? DEFAULT_MOTIFS;
  const motifsPerSequence = Math.max(1, options.motifsPerSequence ?? 2);
  const rng = createRng(options.seed ?? 1);
  const sequences: MovementSequence[] = [];
  for (let s = 0; s < options.sequenceCount; s += 1) {
    const tokens: MovementToken[] = [];
    for (let m = 0; m < motifsPerSequence; m += 1) {
      const motif = motifs[Math.floor(rng() * motifs.length)] ?? motifs[0]!;
      tokens.push(...motif);
    }
    sequences.push({ id: `synthetic-${s}`, tokens });
  }
  return { sequences };
}

// ---------------------------------------------------------------------------
// Generalization / fidelity eval harness.
// ---------------------------------------------------------------------------

export type SequenceFidelity = {
  id: string;
  total: number;
  matched: number;
  accuracy: number;
};

export type FidelityReport = {
  total: number;
  matched: number;
  accuracy: number;
  perSequence: SequenceFidelity[];
};

/**
 * Teacher-forced next-token accuracy: at each position, feed the true prefix and
 * check whether the model's argmax matches the actual next movement. Run against
 * held-out (but related) sequences, this measures generalization fidelity.
 */
export function evaluateNextTokenFidelity(
  inference: MovementModelInference,
  sequences: MovementSequence[],
): FidelityReport {
  const perSequence: SequenceFidelity[] = [];
  let totalAll = 0;
  let matchedAll = 0;
  for (const sequence of sequences) {
    let total = 0;
    let matched = 0;
    for (let i = 0; i < sequence.tokens.length; i += 1) {
      const prefix = sequence.tokens.slice(0, i);
      const prediction = inference.predictNext(prefix);
      total += 1;
      if (prediction.token === sequence.tokens[i]) {
        matched += 1;
      }
    }
    totalAll += total;
    matchedAll += matched;
    perSequence.push({
      id: sequence.id,
      total,
      matched,
      accuracy: total === 0 ? 0 : matched / total,
    });
  }
  return {
    total: totalAll,
    matched: matchedAll,
    accuracy: totalAll === 0 ? 0 : matchedAll / totalAll,
    perSequence,
  };
}

/** Convenience: train + immediately return an inference handle. */
export function trainMovementModel(
  dataset: MovementDataset,
  backend: MovementModelBackend = new MarkovMovementBackend(),
  options?: MovementTrainOptions,
): { artifact: MovementModelArtifact; inference: MovementModelInference } {
  const artifact = backend.train(dataset, options);
  return { artifact, inference: backend.createInference(artifact) };
}
