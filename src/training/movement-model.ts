// Local-movement learning subsystem — pluggable model backend.
//
// This module turns recorded movement trajectories into a training dataset,
// defines a *pluggable* model-backend interface, and ships a deterministic
// in-process backend (`MarkovMovementBackend`) so the whole
// capture → dataset → train → infer → replay loop is exercisable in the cloud
// with NO access to the user's real machine.
//
// The real on-device pipeline (mlx / axolotl launch scripts) lives in
// `runner.ts`; this file is the seam a small local/open model plugs into. A
// real backend implements the same `MovementModelBackend` interface — train on
// the dataset, snapshot its weights, and predict the next movement token — so
// the rest of bee-agent (replay, eval, generalization) is backend-agnostic.

import type { DeviceCaptureInput, DeviceGestureKind } from "../capture/device-adapter.js";
import type { TrajectoryAction, TrajectorySpan } from "../capture/trajectory.js";

/** Boundary tokens framing every recorded movement sequence. */
export const MOVEMENT_START_TOKEN = "⟨start⟩";
export const MOVEMENT_STOP_TOKEN = "⟨stop⟩";

/** A single movement in a sequence, canonicalized to a stable string token. */
export type MovementToken = string;

export type MovementSequence = {
  /** Stable id — usually the source trajectory id. */
  id: string;
  /** Ordered movement tokens, boundary tokens excluded. */
  tokens: MovementToken[];
};

export type MovementDataset = {
  version: 1;
  /** Sorted, de-duplicated set of every token seen across all sequences. */
  vocabulary: MovementToken[];
  sequences: MovementSequence[];
};

/**
 * Canonicalize a gesture into a movement token. Direction and a coarse target
 * label are folded in so the model learns e.g. `tap:Submit` distinct from
 * `tap:Cancel`, while free-form value text is dropped (privacy + generalization).
 */
export function tokenizeGesture(gesture: NonNullable<DeviceCaptureInput["gesture"]>): MovementToken {
  return movementToken(gesture.kind, gesture.direction ?? gesture.target);
}

/**
 * Canonicalize a recorded trajectory action into a movement token. Device
 * gestures encode their gesture kind in `metadata.gesture`; other tools fall
 * back to the tool name plus a normalized target/verb from the summary.
 */
export function tokenizeAction(action: TrajectoryAction): MovementToken {
  const metadata = action.metadata ?? {};
  const gesture = typeof metadata.gesture === "string" ? (metadata.gesture as DeviceGestureKind) : undefined;
  if (gesture) {
    const qualifier =
      (typeof metadata.direction === "string" ? metadata.direction : undefined) ??
      (typeof metadata.target === "string" ? metadata.target : undefined);
    return movementToken(gesture, qualifier);
  }
  return movementToken(action.tool, firstWord(action.summary));
}

function movementToken(kind: string, qualifier: string | undefined): MovementToken {
  const head = slug(kind);
  const tail = qualifier ? slug(qualifier) : "";
  return tail ? `${head}:${tail}` : head;
}

function slug(value: string): string {
  return value.trim().toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "");
}

function firstWord(summary: string): string | undefined {
  const match = summary.trim().split(/\s+/)[0];
  return match && match.length > 0 ? match : undefined;
}

/** Build a movement sequence from a single trajectory's ordered actions. */
export function sequenceFromTrajectory(trajectory: TrajectorySpan): MovementSequence {
  const tokens = [...trajectory.actions]
    .sort((a, b) => a.ts - b.ts)
    .map(tokenizeAction);
  return { id: trajectory.id, tokens };
}

/** Assemble a dataset from trajectories (empty sequences are dropped). */
export function buildMovementDataset(trajectories: TrajectorySpan[]): MovementDataset {
  return datasetFromSequences(trajectories.map(sequenceFromTrajectory));
}

/** Assemble a dataset directly from pre-tokenized sequences. */
export function datasetFromSequences(sequences: MovementSequence[]): MovementDataset {
  const kept = sequences.filter((sequence) => sequence.tokens.length > 0);
  const vocabulary = new Set<MovementToken>();
  for (const sequence of kept) {
    for (const token of sequence.tokens) {
      vocabulary.add(token);
    }
  }
  return {
    version: 1,
    vocabulary: [...vocabulary].sort(),
    sequences: kept,
  };
}

// ---------------------------------------------------------------------------
// Pluggable backend interface
// ---------------------------------------------------------------------------

export type MovementCandidate = {
  token: MovementToken;
  probability: number;
};

export type MovementPrediction = {
  /** Argmax next token (may be the stop token, signalling end of movement). */
  token: MovementToken;
  probability: number;
  /** Ranked candidates (highest probability first), including `token`. */
  candidates: MovementCandidate[];
  /**
   * Context length actually used after backoff. `order` === requested context
   * length means an exact match; a smaller value means the model generalized
   * from a shorter, more common prefix.
   */
  order: number;
};

/** Opaque, serializable model weights — a real backend fills in its own shape. */
export type MovementModelSnapshot = {
  backendId: string;
  version: 1;
  vocabulary: MovementToken[];
  payload: unknown;
};

export type PredictOptions = {
  /** Cap on ranked candidates returned (default: all). */
  topK?: number;
};

export type GenerateOptions = {
  /** Hard cap on generated tokens (excluding the stop token). */
  maxSteps?: number;
};

/** A trained model instance — the inference surface used by replay + eval. */
export interface TrainedMovementModel {
  readonly backendId: string;
  readonly vocabulary: readonly MovementToken[];
  /** Predict the movement following `context` (most-recent token last). */
  predictNext(context: MovementToken[], options?: PredictOptions): MovementPrediction;
  /** Deterministically roll out a full movement sequence from `seed`. */
  generate(seed: MovementToken[], options?: GenerateOptions): MovementToken[];
  /** Serialize for persistence / handing to a replay engine. */
  snapshot(): MovementModelSnapshot;
}

export type TrainOptions = {
  /** Max Markov context order for backends that use it (default: 3). */
  maxOrder?: number;
};

/** The seam a local/open model plugs into. Backends are interchangeable. */
export interface MovementModelBackend {
  readonly id: string;
  train(dataset: MovementDataset, options?: TrainOptions): Promise<TrainedMovementModel>;
  load(snapshot: MovementModelSnapshot): TrainedMovementModel;
}

// ---------------------------------------------------------------------------
// Deterministic reference backend: variable-order Markov with stupid backoff
// ---------------------------------------------------------------------------

const STUPID_BACKOFF_DISCOUNT = 0.4;

type ContextCounts = Record<string, Record<MovementToken, number>>;

type MarkovPayload = {
  maxOrder: number;
  /** counts[order] maps a joined context of length `order` → token counts. */
  counts: ContextCounts[];
};

/**
 * A fully deterministic n-gram backend. It learns transition counts up to
 * `maxOrder` and predicts via stupid backoff: try the longest matching
 * context, else drop the oldest token and retry. This means it (a) reproduces
 * recorded movement paths exactly (argmax on a seen context) and (b)
 * *generalizes* to unseen prefixes by backing off to shorter, more common
 * sub-contexts — the core "repeat + generalize" requirement, with zero
 * randomness so cloud tests are reproducible.
 */
export class MarkovMovementBackend implements MovementModelBackend {
  readonly id = "markov-backoff";

  async train(dataset: MovementDataset, options: TrainOptions = {}): Promise<TrainedMovementModel> {
    const maxOrder = Math.max(1, options.maxOrder ?? 3);
    const counts: ContextCounts[] = Array.from({ length: maxOrder + 1 }, () => ({}));

    for (const sequence of dataset.sequences) {
      const framed = [MOVEMENT_START_TOKEN, ...sequence.tokens, MOVEMENT_STOP_TOKEN];
      for (let i = 1; i < framed.length; i += 1) {
        const next = framed[i];
        for (let order = 0; order <= maxOrder; order += 1) {
          if (i - order < 0) {
            break;
          }
          const context = framed.slice(i - order, i);
          const key = contextKey(context);
          const bucket = (counts[order][key] ??= {});
          bucket[next] = (bucket[next] ?? 0) + 1;
        }
      }
    }

    const snapshot: MovementModelSnapshot = {
      backendId: this.id,
      version: 1,
      vocabulary: [...dataset.vocabulary],
      payload: { maxOrder, counts } satisfies MarkovPayload,
    };
    return this.load(snapshot);
  }

  load(snapshot: MovementModelSnapshot): TrainedMovementModel {
    if (snapshot.backendId !== this.id) {
      throw new Error(`snapshot backend "${snapshot.backendId}" is not "${this.id}"`);
    }
    return new MarkovMovementModel(snapshot);
  }
}

class MarkovMovementModel implements TrainedMovementModel {
  readonly backendId: string;
  readonly vocabulary: readonly MovementToken[];
  private readonly payload: MarkovPayload;

  constructor(private readonly snapshotData: MovementModelSnapshot) {
    this.backendId = snapshotData.backendId;
    this.vocabulary = snapshotData.vocabulary;
    this.payload = snapshotData.payload as MarkovPayload;
  }

  predictNext(context: MovementToken[], options: PredictOptions = {}): MovementPrediction {
    const { counts, maxOrder } = this.payload;
    const startOrder = Math.min(maxOrder, context.length);

    for (let order = startOrder; order >= 0; order -= 1) {
      const key = contextKey(order === 0 ? [] : context.slice(context.length - order));
      const bucket = counts[order]?.[key];
      if (!bucket) {
        continue;
      }
      const total = Object.values(bucket).reduce((sum, count) => sum + count, 0);
      if (total === 0) {
        continue;
      }
      // Stupid-backoff weight: exact matches score 1.0, each dropped token
      // multiplies by the discount so shorter (more general) contexts rank
      // below exact ones when both exist at prediction time.
      const backoffWeight = STUPID_BACKOFF_DISCOUNT ** (startOrder - order);
      const ranked = Object.entries(bucket)
        .map(([token, count]) => ({ token, probability: (count / total) * backoffWeight }))
        .sort(compareCandidates);
      const limited = options.topK ? ranked.slice(0, options.topK) : ranked;
      const best = ranked[0];
      return {
        token: best.token,
        probability: best.probability,
        candidates: limited,
        order,
      };
    }

    // Unseen even at order 0 (empty model): fall back to the stop token.
    return {
      token: MOVEMENT_STOP_TOKEN,
      probability: 0,
      candidates: [{ token: MOVEMENT_STOP_TOKEN, probability: 0 }],
      order: 0,
    };
  }

  generate(seed: MovementToken[], options: GenerateOptions = {}): MovementToken[] {
    const maxSteps = options.maxSteps ?? 64;
    const history = [MOVEMENT_START_TOKEN, ...seed];
    const produced: MovementToken[] = [...seed];
    for (let step = 0; step < maxSteps; step += 1) {
      const prediction = this.predictNext(history);
      if (prediction.token === MOVEMENT_STOP_TOKEN) {
        break;
      }
      produced.push(prediction.token);
      history.push(prediction.token);
    }
    return produced;
  }

  snapshot(): MovementModelSnapshot {
    return this.snapshotData;
  }
}

function compareCandidates(a: MovementCandidate, b: MovementCandidate): number {
  if (b.probability !== a.probability) {
    return b.probability - a.probability;
  }
  // Deterministic tie-break so argmax and ranking are stable across runs.
  return a.token < b.token ? -1 : a.token > b.token ? 1 : 0;
}

function contextKey(context: MovementToken[]): string {
  return context.join("");
}

// ---------------------------------------------------------------------------
// Synthetic event-stream generator (validates the loop without real OS input)
// ---------------------------------------------------------------------------

export type SyntheticMovementScenario = {
  /** Ordered movement tokens the "user" tends to perform. */
  pattern: MovementToken[];
  /** How many noisy variants to emit. */
  count: number;
  /** Probability [0,1] a step is dropped in a variant (default 0). */
  dropRate?: number;
};

/**
 * Deterministically synthesize movement sequences from scenarios using a
 * seeded PRNG — no `Math.random`, so datasets are byte-stable and tests are
 * reproducible. Used to validate capture→dataset→train→replay round-trips in
 * environments with no real input devices.
 */
export function generateSyntheticMovementSequences(
  scenarios: SyntheticMovementScenario[],
  seed = 1,
): MovementSequence[] {
  const rng = mulberry32(seed >>> 0);
  const sequences: MovementSequence[] = [];
  let index = 0;
  for (const scenario of scenarios) {
    const dropRate = scenario.dropRate ?? 0;
    for (let variant = 0; variant < scenario.count; variant += 1) {
      const tokens = scenario.pattern.filter(() => rng() >= dropRate);
      const kept = tokens.length > 0 ? tokens : [...scenario.pattern];
      sequences.push({ id: `synthetic-${index}`, tokens: kept });
      index += 1;
    }
  }
  return sequences;
}

function mulberry32(seed: number): () => number {
  let state = seed;
  return () => {
    state |= 0;
    state = (state + 0x6d2b79f5) | 0;
    let t = Math.imul(state ^ (state >>> 15), 1 | state);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// ---------------------------------------------------------------------------
// Generalization eval harness
// ---------------------------------------------------------------------------

export type MovementEvalReport = {
  /** Held-out (context → next) prediction steps evaluated. */
  steps: number;
  /** Fraction where the argmax token matched the held-out next token. */
  top1Accuracy: number;
  /** Fraction where the true token appeared in the top-K candidates. */
  topKAccuracy: number;
  /** Fraction of steps answered by backing off below the full context order. */
  backoffRate: number;
};

/**
 * Measure how well a trained model predicts *held-out* movement sequences —
 * the concrete test of generalization to "new but related movements". Each
 * held-out sequence is walked token by token; at each step the model predicts
 * from the true prefix and is scored against the true continuation.
 */
export function evaluateMovementModel(
  model: TrainedMovementModel,
  heldOut: MovementSequence[],
  options: { topK?: number } = {},
): MovementEvalReport {
  const topK = options.topK ?? 3;
  let steps = 0;
  let top1 = 0;
  let topKHits = 0;
  let backoffs = 0;

  for (const sequence of heldOut) {
    const framed = [...sequence.tokens, MOVEMENT_STOP_TOKEN];
    const context: MovementToken[] = [];
    for (const trueNext of framed) {
      const prediction = model.predictNext(context, { topK });
      steps += 1;
      if (prediction.token === trueNext) {
        top1 += 1;
      }
      if (prediction.candidates.some((candidate) => candidate.token === trueNext)) {
        topKHits += 1;
      }
      if (prediction.order < Math.min(context.length, maxOrderOf(model))) {
        backoffs += 1;
      }
      if (trueNext !== MOVEMENT_STOP_TOKEN) {
        context.push(trueNext);
      }
    }
  }

  return {
    steps,
    top1Accuracy: steps === 0 ? 0 : top1 / steps,
    topKAccuracy: steps === 0 ? 0 : topKHits / steps,
    backoffRate: steps === 0 ? 0 : backoffs / steps,
  };
}

function maxOrderOf(model: TrainedMovementModel): number {
  const payload = model.snapshot().payload as Partial<MarkovPayload> | undefined;
  return typeof payload?.maxOrder === "number" ? payload.maxOrder : 0;
}
