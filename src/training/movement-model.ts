// Local-movement learning subsystem: a pluggable, in-process model backend that
// learns from a recorded movement dataset and infers next movements — including
// generalizing to new-but-related movements (self-evolution objective 2c/2d).
//
// The training runner (`runner.ts`) only builds external launch plans (mlx /
// axolotl) that execute on the user's Apple-silicon machine. Those cannot run in
// the cloud/CI, so this module provides a *deterministic, dependency-free* model
// backend that trains and infers entirely in-process. Real on-device backends
// plug in behind the same `MovementModelBackend` interface later; this mock is
// the seam that lets the whole capture → dataset → train → infer → eval loop be
// exercised and tested without any real OS input or GPU.

import type { ReplayManifest } from "../capture/replay.js";
import type { TrajectorySpan } from "../capture/trajectory.js";

/** A single normalized movement/action token, e.g. `"mouse:move-left"`. */
export type MovementToken = string;

/** One recorded trajectory reduced to an ordered sequence of movement tokens. */
export type MovementSequence = {
  id: string;
  sessionId?: string;
  tokens: MovementToken[];
  reward?: number;
};

/** A replayable movement dataset: the training input for a model backend. */
export type MovementDataset = {
  version: 1;
  sequences: MovementSequence[];
};

/** A ranked next-token guess plus the full distribution it was drawn from. */
export type MovementPrediction = {
  /** Best next token, or `undefined` when the model has no signal at all. */
  token: MovementToken | undefined;
  /** Probability mass assigned to `token` within the resolved context. */
  probability: number;
  /** Context length (n-gram order) that produced the prediction; 0 = unigram. */
  order: number;
  /** Full ranked distribution for the resolved context (descending prob). */
  distribution: Array<{ token: MovementToken; probability: number }>;
};

/** Backend-agnostic, JSON-serializable trained model. */
export type MovementModelArtifact = {
  version: 1;
  backendId: string;
  maxOrder: number;
  vocabSize: number;
  sequenceCount: number;
  tokenCount: number;
  /** Backend-specific parameters; opaque to callers, consumed by `load()`. */
  params: unknown;
};

export type MovementTrainOptions = {
  /** Highest n-gram order to model (context length). Default 3. */
  maxOrder?: number;
};

export type MovementPredictOptions = {
  /** Cap the backoff order used for this prediction (e.g. 0 = unigram only). */
  maxOrder?: number;
};

/** A loaded, ready-to-infer model. */
export interface MovementModelInference {
  readonly backendId: string;
  readonly maxOrder: number;
  predictNext(context: MovementToken[], options?: MovementPredictOptions): MovementPrediction;
  /** Roll the model forward from `seed`, stopping at a natural end or `steps`. */
  generate(seed: MovementToken[], steps: number): MovementToken[];
}

/** A pluggable movement-model implementation (mock, mlx, onnx, …). */
export interface MovementModelBackend {
  readonly id: string;
  train(dataset: MovementDataset, options?: MovementTrainOptions): MovementModelArtifact;
  load(artifact: MovementModelArtifact): MovementModelInference;
}

// ---------------------------------------------------------------------------
// Reserved tokens & separators
// ---------------------------------------------------------------------------

/** Left-padding marker; never emitted as a prediction. */
const START = "START";
/** End-of-sequence marker; may be predicted to signal a natural stop. */
export const MOVEMENT_END = "END";
const CONTEXT_SEP = "";

// ---------------------------------------------------------------------------
// Tokenization: turn recorded actions into movement tokens
// ---------------------------------------------------------------------------

/**
 * Normalize a recorded action into a stable movement token. Keeps the tool and
 * a coarse, lowercased slug of the summary so related movements collapse to the
 * same token (the basis for generalization) without over-fragmenting the vocab.
 */
export function movementTokenFromAction(action: { tool: string; summary: string }): MovementToken {
  const tool = action.tool.trim().toLowerCase().replace(/\s+/g, "-");
  const slug = action.summary
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .split("-")
    .slice(0, 3)
    .join("-");
  return slug ? `${tool}:${slug}` : tool;
}

/** Build a movement dataset from reviewed trajectory spans (actions only). */
export function buildMovementDatasetFromTrajectories(trajectories: TrajectorySpan[]): MovementDataset {
  const sequences = trajectories.map<MovementSequence>((trajectory) => ({
    id: trajectory.id,
    sessionId: trajectory.sessionId,
    tokens: [...trajectory.actions]
      .sort((a, b) => a.ts - b.ts)
      .map((action) => movementTokenFromAction(action)),
    ...(typeof trajectory.outcome?.reward === "number" ? { reward: trajectory.outcome.reward } : {}),
  }));
  return { version: 1, sequences: sequences.filter((sequence) => sequence.tokens.length > 0) };
}

/** Build a movement dataset from replay manifests (action-kind events only). */
export function buildMovementDatasetFromReplays(replays: ReplayManifest[]): MovementDataset {
  const sequences = replays.map<MovementSequence>((replay) => ({
    id: replay.trajectoryIds.join("+") || replay.sessionId,
    sessionId: replay.sessionId,
    tokens: replay.events
      .filter((event): event is Extract<ReplayManifest["events"][number], { kind: "action" }> => event.kind === "action")
      .sort((a, b) => a.ts - b.ts)
      .map((event) => movementTokenFromAction(event)),
  }));
  return { version: 1, sequences: sequences.filter((sequence) => sequence.tokens.length > 0) };
}

// ---------------------------------------------------------------------------
// Deterministic mock backend: variable-order Markov with n-gram backoff
// ---------------------------------------------------------------------------

type MarkovGram = { order: number; entries: Array<[string, Array<[string, number]>]> };

type MarkovParams = {
  maxOrder: number;
  vocab: string[];
  grams: MarkovGram[];
};

function contextKey(tokens: MovementToken[]): string {
  return tokens.join(CONTEXT_SEP);
}

/**
 * The built-in deterministic backend. It counts token transitions at every
 * order up to `maxOrder`, then predicts by trying the longest matching context
 * and backing off to shorter suffixes (finally the unigram). Backoff is exactly
 * what lets it generalize: an unseen full context still resolves through a
 * shorter suffix it *has* seen. Training and inference are fully deterministic
 * (argmax with a lexical tie-break), so cloud/CI runs are reproducible.
 */
export class MarkovMovementBackend implements MovementModelBackend {
  readonly id = "markov-mock";

  train(dataset: MovementDataset, options: MovementTrainOptions = {}): MovementModelArtifact {
    const maxOrder = Math.max(0, Math.floor(options.maxOrder ?? 3));
    // counts[order] : Map<contextKey, Map<token, count>>
    const counts: Array<Map<string, Map<string, number>>> = Array.from(
      { length: maxOrder + 1 },
      () => new Map<string, Map<string, number>>(),
    );
    const vocab = new Set<string>();
    let tokenCount = 0;

    for (const sequence of dataset.sequences) {
      if (sequence.tokens.length === 0) {
        continue;
      }
      // Pad front with START markers and append END so start- and end-of-move
      // behavior is learnable.
      const padded = [...Array.from({ length: maxOrder }, () => START), ...sequence.tokens, MOVEMENT_END];
      for (const token of sequence.tokens) {
        vocab.add(token);
      }
      tokenCount += sequence.tokens.length;

      for (let i = maxOrder; i < padded.length; i += 1) {
        const next = padded[i]!;
        for (let order = 0; order <= maxOrder; order += 1) {
          const context = padded.slice(i - order, i);
          const key = contextKey(context);
          const bucket = counts[order]!;
          let row = bucket.get(key);
          if (!row) {
            row = new Map<string, number>();
            bucket.set(key, row);
          }
          row.set(next, (row.get(next) ?? 0) + 1);
        }
      }
    }

    const grams: MarkovGram[] = counts.map((bucket, order) => ({
      order,
      entries: [...bucket.entries()].map(([key, row]) => [key, [...row.entries()]] as [string, Array<[string, number]>]),
    }));

    const params: MarkovParams = { maxOrder, vocab: [...vocab].sort(), grams };
    return {
      version: 1,
      backendId: this.id,
      maxOrder,
      vocabSize: vocab.size,
      sequenceCount: dataset.sequences.length,
      tokenCount,
      params,
    };
  }

  load(artifact: MovementModelArtifact): MovementModelInference {
    if (artifact.backendId !== this.id) {
      throw new Error(`MarkovMovementBackend cannot load artifact from backend "${artifact.backendId}"`);
    }
    return new MarkovMovementInference(artifact.params as MarkovParams);
  }
}

class MarkovMovementInference implements MovementModelInference {
  readonly backendId = "markov-mock";
  readonly maxOrder: number;
  private readonly grams: Array<Map<string, Map<string, number>>>;

  constructor(params: MarkovParams) {
    this.maxOrder = params.maxOrder;
    this.grams = params.grams
      .slice()
      .sort((a, b) => a.order - b.order)
      .map((gram) => new Map(gram.entries.map(([key, row]) => [key, new Map(row)])));
  }

  predictNext(context: MovementToken[], options: MovementPredictOptions = {}): MovementPrediction {
    const cap = Math.min(this.maxOrder, options.maxOrder ?? this.maxOrder);
    // Left-pad with START so short/empty contexts resolve against the same
    // padded keys the model was trained on.
    const padded = [...Array.from({ length: this.maxOrder }, () => START), ...context];
    for (let order = cap; order >= 0; order -= 1) {
      const bucket = this.grams[order];
      if (!bucket) {
        continue;
      }
      const key = contextKey(padded.slice(padded.length - order));
      const row = bucket.get(key);
      if (row && row.size > 0) {
        return toPrediction(row, order);
      }
    }
    return { token: undefined, probability: 0, order: 0, distribution: [] };
  }

  generate(seed: MovementToken[], steps: number): MovementToken[] {
    const out: MovementToken[] = [];
    const context = [...seed];
    for (let i = 0; i < steps; i += 1) {
      const prediction = this.predictNext(context);
      if (!prediction.token || prediction.token === MOVEMENT_END) {
        break;
      }
      out.push(prediction.token);
      context.push(prediction.token);
    }
    return out;
  }
}

function toPrediction(row: Map<string, number>, order: number): MovementPrediction {
  const total = [...row.values()].reduce((sum, count) => sum + count, 0);
  const distribution = [...row.entries()]
    .filter(([token]) => token !== START)
    .map(([token, count]) => ({ token, probability: count / total }))
    // Deterministic ranking: probability desc, then token asc for stable ties.
    .sort((a, b) => (b.probability - a.probability) || (a.token < b.token ? -1 : a.token > b.token ? 1 : 0));
  const best = distribution[0];
  return {
    token: best?.token,
    probability: best?.probability ?? 0,
    order,
    distribution,
  };
}

// ---------------------------------------------------------------------------
// Pluggable backend registry
// ---------------------------------------------------------------------------

const backends = new Map<string, MovementModelBackend>();

/** Register (or replace) a movement-model backend by id. */
export function registerMovementBackend(backend: MovementModelBackend): void {
  backends.set(backend.id, backend);
}

/** Resolve a registered backend by id, or throw if unknown. */
export function getMovementBackend(id: string): MovementModelBackend {
  const backend = backends.get(id);
  if (!backend) {
    throw new Error(`unknown movement backend "${id}" (registered: ${[...backends.keys()].join(", ") || "none"})`);
  }
  return backend;
}

/** List the ids of all registered backends. */
export function listMovementBackends(): string[] {
  return [...backends.keys()];
}

// The deterministic mock is always available.
registerMovementBackend(new MarkovMovementBackend());

/** Convenience: train with a registered backend (defaults to the mock). */
export function trainMovementModel(
  dataset: MovementDataset,
  options: MovementTrainOptions & { backendId?: string } = {},
): { artifact: MovementModelArtifact; model: MovementModelInference } {
  const backend = getMovementBackend(options.backendId ?? "markov-mock");
  const artifact = backend.train(dataset, options);
  return { artifact, model: backend.load(artifact) };
}

// ---------------------------------------------------------------------------
// Synthetic movement-stream generator (no real OS input required)
// ---------------------------------------------------------------------------

/** Seeded PRNG (mulberry32) so synthetic datasets are fully reproducible. */
function mulberry32(seed: number): () => number {
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
  /** Number of trajectories to synthesize. */
  sequenceCount: number;
  seed?: number;
  minLength?: number;
  maxLength?: number;
  /** 0..1 chance a step deviates from the dominant transition (learnable noise). */
  noise?: number;
};

/**
 * Generate structured, learnable movement sequences from a small hidden Markov
 * process (a UI-navigation grammar: focus → move* → click → confirm). Because
 * transitions are near-deterministic with a little noise, a trained model should
 * predict held-out sequences from the same grammar well above a unigram
 * baseline — giving the generalization eval a real signal without any real
 * mouse/keyboard capture.
 */
export function generateSyntheticMovementSequences(options: SyntheticMovementOptions): MovementDataset {
  const rand = mulberry32(options.seed ?? 1);
  const minLength = Math.max(2, options.minLength ?? 4);
  const maxLength = Math.max(minLength, options.maxLength ?? 8);
  const noise = Math.min(1, Math.max(0, options.noise ?? 0.1));

  // Dominant next-token per state; a small ordered alternative set is used when
  // noise fires, so deviations are still drawn from a learnable distribution.
  const grammar: Record<string, { main: MovementToken; alt: MovementToken[] }> = {
    focus: { main: "mouse:move-right", alt: ["mouse:move-down"] },
    "mouse:move-right": { main: "mouse:move-down", alt: ["mouse:move-right", "mouse:click"] },
    "mouse:move-down": { main: "mouse:click", alt: ["mouse:move-left", "mouse:move-down"] },
    "mouse:move-left": { main: "mouse:click", alt: ["mouse:move-up"] },
    "mouse:move-up": { main: "mouse:move-right", alt: ["mouse:click"] },
    "mouse:click": { main: "key:confirm", alt: ["mouse:move-right"] },
    "key:confirm": { main: MOVEMENT_END, alt: ["focus"] },
  };

  const sequences: MovementSequence[] = [];
  for (let i = 0; i < options.sequenceCount; i += 1) {
    const target = minLength + Math.floor(rand() * (maxLength - minLength + 1));
    const tokens: MovementToken[] = [];
    let state = "focus";
    tokens.push(state);
    while (tokens.length < target) {
      const rule = grammar[state];
      if (!rule) {
        break;
      }
      let next: MovementToken;
      if (rand() < noise && rule.alt.length > 0) {
        next = rule.alt[Math.floor(rand() * rule.alt.length)]!;
      } else {
        next = rule.main;
      }
      if (next === MOVEMENT_END) {
        // Only stop early once we've met the minimum length.
        if (tokens.length >= minLength) {
          break;
        }
        next = rule.alt[0] ?? "focus";
      }
      tokens.push(next);
      state = next;
    }
    sequences.push({ id: `synthetic-${i}`, tokens });
  }
  return { version: 1, sequences };
}

// ---------------------------------------------------------------------------
// Generalization eval harness
// ---------------------------------------------------------------------------

export type NextTokenEvalResult = {
  total: number;
  correct: number;
  accuracy: number;
};

export type EvalOptions = {
  /** Skip scoring positions with fewer than this many context tokens. Default 0. */
  minContext?: number;
  /** Cap the backoff order the model may use (0 = unigram baseline). */
  maxOrder?: number;
};

/**
 * Top-1 next-token accuracy over held-out sequences. For each position `i`, the
 * model sees `tokens[0..i-1]` and must predict `tokens[i]`.
 */
export function evaluateNextTokenAccuracy(
  model: MovementModelInference,
  sequences: MovementSequence[],
  options: EvalOptions = {},
): NextTokenEvalResult {
  const minContext = Math.max(0, options.minContext ?? 0);
  let total = 0;
  let correct = 0;
  for (const sequence of sequences) {
    for (let i = minContext; i < sequence.tokens.length; i += 1) {
      const context = sequence.tokens.slice(0, i);
      const prediction = model.predictNext(context, options.maxOrder !== undefined ? { maxOrder: options.maxOrder } : undefined);
      total += 1;
      if (prediction.token === sequence.tokens[i]) {
        correct += 1;
      }
    }
  }
  return { total, correct, accuracy: total === 0 ? 0 : correct / total };
}

export type GeneralizationReport = {
  trainSequences: number;
  evalSequences: number;
  /** Accuracy of the full-order model on held-out data. */
  modelAccuracy: number;
  /** Accuracy of an order-0 (unigram) baseline on the same held-out data. */
  baselineAccuracy: number;
  /** modelAccuracy − baselineAccuracy; > 0 means the model generalized. */
  lift: number;
  artifact: MovementModelArtifact;
};

/**
 * Train on the first `trainRatio` of a dataset's sequences and measure next-token
 * accuracy on the held-out remainder against a unigram baseline. Positive `lift`
 * is direct evidence the model learned transition structure that transfers to
 * new-but-related sequences.
 */
export function evaluateGeneralization(
  dataset: MovementDataset,
  options: MovementTrainOptions & { backendId?: string; trainRatio?: number } = {},
): GeneralizationReport {
  const ratio = Math.min(0.95, Math.max(0.05, options.trainRatio ?? 0.7));
  const splitAt = Math.max(1, Math.floor(dataset.sequences.length * ratio));
  const trainSequences = dataset.sequences.slice(0, splitAt);
  const evalSequences = dataset.sequences.slice(splitAt);
  const holdout = evalSequences.length > 0 ? evalSequences : trainSequences;

  const { artifact, model } = trainMovementModel(
    { version: 1, sequences: trainSequences },
    options,
  );
  const modelResult = evaluateNextTokenAccuracy(model, holdout);
  const baselineResult = evaluateNextTokenAccuracy(model, holdout, { maxOrder: 0 });

  return {
    trainSequences: trainSequences.length,
    evalSequences: holdout.length,
    modelAccuracy: modelResult.accuracy,
    baselineAccuracy: baselineResult.accuracy,
    lift: modelResult.accuracy - baselineResult.accuracy,
    artifact,
  };
}
