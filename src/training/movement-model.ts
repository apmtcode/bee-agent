import type { TrajectoryAction, TrajectorySpan } from "../capture/trajectory.js";
import type { ReplayTimelineEvent } from "../capture/replay.js";
import type { ExportedReplayManifest } from "./export-manifest.js";

/**
 * In-process, deterministic movement-learning model.
 *
 * Objective #2 of the self-evolution charter asks bee-agent to (c) post-train a
 * local model on recorded movement datasets so it can repeat the recorded
 * movements and (d) generalize to new-but-related movements. The production path
 * shells out to on-device runtimes (mlx / axolotl — see {@link ./runner.ts}),
 * which cannot run in the cloud/CI. This module provides a *pluggable* backend
 * abstraction plus a fully in-process, deterministic reference backend so the
 * capture → dataset → train → infer loop can be validated end-to-end without any
 * real OS input or GPU. Swap {@link NgramMovementModelBackend} for a real
 * on-device small-model backend by implementing {@link MovementModelBackend}.
 */

/** A single movement, encoded as an opaque string token (e.g. `device:tap:Submit`). */
export type MovementToken = string;

/** Sentinel marking the start of a movement sequence (never emitted). */
export const MOVEMENT_BEGIN = "<bos>";
/** Sentinel marking the end of a movement sequence (terminates generation). */
export const MOVEMENT_END = "<eos>";

/** An ordered movement sequence for a single trajectory / session. */
export type MovementSequence = {
  id: string;
  tokens: MovementToken[];
};

export type MovementCandidate = {
  token: MovementToken;
  probability: number;
};

export type MovementPrediction = {
  token: MovementToken;
  probability: number;
  /** Number of preceding context tokens the prediction actually used (backoff depth). */
  contextOrder: number;
  /** Ranked alternatives at the chosen backoff order (deterministic ordering). */
  candidates: MovementCandidate[];
};

/** A trained model ready for inference. Serializable so it can be persisted. */
export interface TrainedMovementModel {
  readonly backendId: string;
  readonly vocabulary: MovementToken[];
  /** Predict the most likely next movement given a context prefix. */
  predictNext(context: MovementToken[]): MovementPrediction | undefined;
  /** Roll out a sequence of movements starting from an optional seed prefix. */
  generate(seed?: MovementToken[], maxSteps?: number): MovementToken[];
  /** Structured snapshot for persistence / inspection. */
  serialize(): SerializedMovementModel;
}

export type SerializedMovementModel = {
  backendId: string;
  order: number;
  vocabulary: MovementToken[];
  /** contextOrder -> joined-context -> nextToken -> count */
  transitions: Record<number, Record<string, Record<MovementToken, number>>>;
};

/** Pluggable training backend. Implement this for a real on-device model. */
export interface MovementModelBackend {
  readonly id: string;
  train(sequences: MovementSequence[], options?: MovementTrainOptions): TrainedMovementModel;
}

export type MovementTrainOptions = {
  /** Maximum n-gram order (context length + 1). Defaults to 3. */
  order?: number;
};

const CONTEXT_SEPARATOR = "␟"; // unit separator; safe against token contents

/**
 * Deterministic n-gram backend with stupid-backoff.
 *
 * - **Repeat** (objective 2c): a context seen during training predicts the exact
 *   recorded next movement at the highest available order.
 * - **Generalize** (objective 2d): an unseen full context backs off to shorter
 *   contexts (down to the unigram marginal), yielding a plausible related
 *   movement instead of failing.
 *
 * No randomness or wall-clock is used: ties break by count desc then token asc,
 * so training + inference are byte-for-byte reproducible across runs.
 */
export class NgramMovementModelBackend implements MovementModelBackend {
  readonly id = "ngram-backoff";

  constructor(private readonly defaultOrder = 3) {}

  train(sequences: MovementSequence[], options?: MovementTrainOptions): TrainedMovementModel {
    const order = Math.max(1, options?.order ?? this.defaultOrder);
    // contextOrder (0..order-1) -> joined context -> next token -> count
    const transitions = new Map<number, Map<string, Map<MovementToken, number>>>();
    const vocabulary = new Set<MovementToken>();

    for (let k = 0; k < order; k += 1) {
      transitions.set(k, new Map());
    }

    for (const sequence of sequences) {
      const padded = [MOVEMENT_BEGIN, ...sequence.tokens, MOVEMENT_END];
      for (let i = 1; i < padded.length; i += 1) {
        const target = padded[i]!;
        vocabulary.add(target);
        const maxContext = Math.min(order - 1, i);
        for (let k = 0; k <= maxContext; k += 1) {
          const context = padded.slice(i - k, i);
          const key = context.join(CONTEXT_SEPARATOR);
          const byContext = transitions.get(k)!;
          let counts = byContext.get(key);
          if (!counts) {
            counts = new Map();
            byContext.set(key, counts);
          }
          counts.set(target, (counts.get(target) ?? 0) + 1);
        }
      }
    }

    vocabulary.delete(MOVEMENT_BEGIN);
    return new NgramTrainedModel(this.id, order, transitions, [...vocabulary].sort());
  }
}

class NgramTrainedModel implements TrainedMovementModel {
  constructor(
    readonly backendId: string,
    private readonly order: number,
    private readonly transitions: Map<number, Map<string, Map<MovementToken, number>>>,
    readonly vocabulary: MovementToken[],
  ) {}

  predictNext(context: MovementToken[]): MovementPrediction | undefined {
    const maxOrder = Math.min(this.order - 1, context.length);
    for (let k = maxOrder; k >= 0; k -= 1) {
      const key = context.slice(context.length - k).join(CONTEXT_SEPARATOR);
      const counts = this.transitions.get(k)?.get(key);
      if (!counts || counts.size === 0) {
        continue;
      }
      const total = [...counts.values()].reduce((sum, value) => sum + value, 0);
      const candidates: MovementCandidate[] = [...counts.entries()]
        .map(([token, count]) => ({ token, probability: count / total }))
        .sort((a, b) => {
          if (b.probability !== a.probability) {
            return b.probability - a.probability;
          }
          return a.token < b.token ? -1 : a.token > b.token ? 1 : 0;
        });
      const best = candidates[0]!;
      return { token: best.token, probability: best.probability, contextOrder: k, candidates };
    }
    return undefined;
  }

  generate(seed: MovementToken[] = [], maxSteps = 64): MovementToken[] {
    const result: MovementToken[] = [];
    const context: MovementToken[] = [MOVEMENT_BEGIN, ...seed];
    for (let step = 0; step < maxSteps; step += 1) {
      const prediction = this.predictNext(context);
      if (!prediction || prediction.token === MOVEMENT_END) {
        break;
      }
      result.push(prediction.token);
      context.push(prediction.token);
    }
    return result;
  }

  serialize(): SerializedMovementModel {
    const transitions: Record<number, Record<string, Record<MovementToken, number>>> = {};
    for (const [k, byContext] of this.transitions) {
      const contextRecord: Record<string, Record<MovementToken, number>> = {};
      for (const [key, counts] of byContext) {
        contextRecord[key] = Object.fromEntries(counts);
      }
      transitions[k] = contextRecord;
    }
    return { backendId: this.backendId, order: this.order, vocabulary: [...this.vocabulary], transitions };
  }
}

// ---------------------------------------------------------------------------
// Tokenization: turn captured movements into model tokens.
// ---------------------------------------------------------------------------

function metadataString(metadata: Record<string, unknown> | undefined, key: string): string | undefined {
  const value = metadata?.[key];
  return typeof value === "string" && value.length > 0 ? value : undefined;
}

/** Encode a rich trajectory action (gesture/target/direction) as a movement token. */
export function tokenizeAction(action: TrajectoryAction): MovementToken {
  const parts: string[] = [action.tool];
  const gesture = metadataString(action.metadata, "gesture");
  const direction = metadataString(action.metadata, "direction");
  const target = metadataString(action.metadata, "target");
  if (gesture) parts.push(gesture);
  if (direction) parts.push(direction);
  if (target) parts.push(target);
  if (parts.length === 1) {
    parts.push(action.summary);
  }
  return parts.join(":");
}

/** Build movement sequences from trajectory spans (uses per-action metadata). */
export function sequencesFromTrajectories(trajectories: TrajectorySpan[]): MovementSequence[] {
  return trajectories
    .map((trajectory) => ({
      id: trajectory.id,
      tokens: [...trajectory.actions]
        .sort((a, b) => a.ts - b.ts)
        .map((action) => tokenizeAction(action)),
    }))
    .filter((sequence) => sequence.tokens.length > 0);
}

/** Build movement sequences from replay-timeline action events, grouped by trajectory. */
export function sequencesFromReplayEvents(events: ReplayTimelineEvent[]): MovementSequence[] {
  const byTrajectory = new Map<string, { ts: number; token: MovementToken }[]>();
  for (const event of events) {
    if (event.kind !== "action") {
      continue;
    }
    const list = byTrajectory.get(event.trajectoryId) ?? [];
    list.push({ ts: event.ts, token: `${event.tool}:${event.summary}` });
    byTrajectory.set(event.trajectoryId, list);
  }
  return [...byTrajectory.entries()]
    .map(([id, entries]) => ({
      id,
      tokens: entries.sort((a, b) => a.ts - b.ts).map((entry) => entry.token),
    }))
    .filter((sequence) => sequence.tokens.length > 0);
}

/** Build movement sequences from an exported reviewed dataset's replay manifests. */
export function sequencesFromReplayManifests(replays: ExportedReplayManifest[]): MovementSequence[] {
  return replays.flatMap((replay) => sequencesFromReplayEvents(replay.events as ReplayTimelineEvent[]));
}

// ---------------------------------------------------------------------------
// Generalization eval harness.
// ---------------------------------------------------------------------------

export type MovementEvalResult = {
  sequences: number;
  predictions: number;
  correct: number;
  /** Fraction of next-token predictions that matched the held-out ground truth. */
  accuracy: number;
  /** contextOrder -> how many predictions used that backoff depth. */
  backoffHistogram: Record<number, number>;
};

/**
 * Measure replay fidelity: for each position in each sequence, ask the model to
 * predict the next movement from the true prefix and compare to ground truth.
 * Run on training sequences to check *repeat* fidelity; run on held-out related
 * sequences to check *generalization*.
 */
export function evaluateMovementModel(
  model: TrainedMovementModel,
  sequences: MovementSequence[],
): MovementEvalResult {
  let predictions = 0;
  let correct = 0;
  const backoffHistogram: Record<number, number> = {};

  for (const sequence of sequences) {
    const context: MovementToken[] = [MOVEMENT_BEGIN];
    for (const expected of sequence.tokens) {
      const prediction = model.predictNext(context);
      predictions += 1;
      if (prediction) {
        backoffHistogram[prediction.contextOrder] = (backoffHistogram[prediction.contextOrder] ?? 0) + 1;
        if (prediction.token === expected) {
          correct += 1;
        }
      }
      context.push(expected);
    }
  }

  return {
    sequences: sequences.length,
    predictions,
    correct,
    accuracy: predictions === 0 ? 0 : correct / predictions,
    backoffHistogram,
  };
}

// ---------------------------------------------------------------------------
// Synthetic event-stream generator (deterministic; no real OS input required).
// ---------------------------------------------------------------------------

export type SyntheticMovementOptions = {
  /** Ordered movement templates to sample from. */
  templates: MovementToken[][];
  /** Number of sequences to generate. */
  count: number;
  /** Deterministic seed (no Math.random is used). */
  seed: number;
  /** Chance [0,1] of perturbing a template into a new-but-related variant. */
  perturbationRate?: number;
};

/**
 * Deterministic pseudo-random stream. Seeded LCG — reproducible across runs and
 * safe in environments where Math.random / Date.now are unavailable.
 */
function createLcg(seed: number): () => number {
  let state = (seed >>> 0) || 0x9e3779b9;
  return () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 0x100000000;
  };
}

/**
 * Generate synthetic movement sequences from a small template grammar. Emits a
 * mix of verbatim templates (for repeat evaluation) and lightly perturbed
 * variants — a dropped or duplicated step (for generalization evaluation) — so
 * the whole capture→train→infer loop can be validated without real OS input.
 */
export function generateSyntheticMovementSequences(options: SyntheticMovementOptions): MovementSequence[] {
  const next = createLcg(options.seed);
  const perturbationRate = options.perturbationRate ?? 0;
  const sequences: MovementSequence[] = [];

  for (let index = 0; index < options.count; index += 1) {
    const template = options.templates[Math.floor(next() * options.templates.length)] ?? [];
    let tokens = [...template];
    if (perturbationRate > 0 && tokens.length > 1 && next() < perturbationRate) {
      const position = Math.floor(next() * tokens.length);
      if (next() < 0.5) {
        // drop a step
        tokens = [...tokens.slice(0, position), ...tokens.slice(position + 1)];
      } else {
        // duplicate a step
        tokens = [...tokens.slice(0, position), tokens[position]!, ...tokens.slice(position)];
      }
    }
    sequences.push({ id: `synthetic-${index}`, tokens });
  }

  return sequences;
}
