import type { ReplayManifest, ReplayTimelineEvent } from "../capture/replay.js";
import type { TrajectorySpan } from "../capture/trajectory.js";

/**
 * Local-movement learning: model backend.
 *
 * This module implements the "post-train a local model to repeat recorded
 * movements, and generalize to new-but-related movements" piece of the
 * movement-learning subsystem (standing objective #2, parts c + d).
 *
 * The heavy on-device backends (MLX / Axolotl) are launched externally by the
 * runner, and cannot run in the cloud. To keep the *pipeline* code testable
 * everywhere, the model backend is expressed as a pluggable interface with a
 * dependency-free, fully deterministic default backend: an order-k Markov chain
 * with stupid-backoff. It genuinely learns from a movement dataset (transition
 * frequencies), reproduces recorded sequences (objective c), and generalizes to
 * unseen prefixes via backoff to shorter matched suffixes (objective d).
 *
 * Everything here is pure (no fs, no clock, no RNG) so it is trivially testable
 * and can be wired into the training runner / execution service later.
 */

/** A single normalized movement token — the structural shape of one movement. */
export type MovementToken = string;

/** Marks the start of a sequence so first-move prediction is learnable. */
export const MOVEMENT_START_TOKEN: MovementToken = "<start>";
/** Marks the end of a sequence so the model can learn to stop. */
export const MOVEMENT_END_TOKEN: MovementToken = "<end>";

/** One recorded movement sequence (e.g. derived from a single trajectory). */
export type MovementSequence = {
  id: string;
  tokens: MovementToken[];
};

/** A dataset of movement sequences the backend trains on. */
export type MovementDataset = {
  version: 1;
  sequences: MovementSequence[];
};

/** A single ranked candidate for the next movement. */
export type MovementCandidate = {
  token: MovementToken;
  /** Normalized probability in [0, 1] across the returned candidates. */
  probability: number;
  /** The context order (n-gram length) that produced this prediction. */
  backoffOrder: number;
};

export type MovementPrediction = {
  token: MovementToken;
  probability: number;
  backoffOrder: number;
  candidates: MovementCandidate[];
};

/** A model that has been trained on a {@link MovementDataset}. */
export interface TrainedMovementModel {
  readonly backendId: string;
  /** Highest context order the model was trained with. */
  readonly order: number;
  /**
   * Predict the next movement given the trailing context. Returns `undefined`
   * only for an empty vocabulary (untrained). Context may be any length; the
   * model uses at most its own `order` trailing tokens.
   */
  predictNext(context: MovementToken[]): MovementPrediction | undefined;
  /**
   * Roll out a full movement sequence from a seed prefix, greedily choosing the
   * most likely next token until {@link MOVEMENT_END_TOKEN} or `maxSteps`.
   * The returned tokens exclude the seed and the terminal end marker.
   */
  generate(seed: MovementToken[], maxSteps?: number): MovementToken[];
  /** Serialize to a plain JSON-safe object (for persistence / transport). */
  toJSON(): SerializedMovementModel;
}

/** Options accepted by a backend's train step. */
export type MovementTrainOptions = {
  /** Max n-gram context order (default 2 = trigram-ish with backoff). */
  order?: number;
};

/** A pluggable model backend — swap the deterministic default for MLX later. */
export interface MovementModelBackend {
  readonly id: string;
  train(dataset: MovementDataset, options?: MovementTrainOptions): Promise<TrainedMovementModel>;
}

export type SerializedMovementModel = {
  version: 1;
  backendId: string;
  order: number;
  /** context-key -> (nextToken -> count). Context keys join tokens with "". */
  transitions: Record<string, Record<MovementToken, number>>;
  vocabulary: MovementToken[];
};

const CONTEXT_SEPARATOR = "";

function contextKey(tokens: MovementToken[]): string {
  return tokens.join(CONTEXT_SEPARATOR);
}

/**
 * Order-k Markov model with stupid-backoff. Deterministic: ties break by count
 * desc then token lexical asc, so identical datasets always yield identical
 * predictions (critical for reproducible cloud tests).
 */
export class MarkovMovementModel implements TrainedMovementModel {
  readonly backendId: string;
  readonly order: number;
  private readonly transitions: Map<string, Map<MovementToken, number>>;
  private readonly vocabulary: Set<MovementToken>;

  constructor(params: {
    backendId: string;
    order: number;
    transitions: Map<string, Map<MovementToken, number>>;
    vocabulary: Set<MovementToken>;
  }) {
    this.backendId = params.backendId;
    this.order = params.order;
    this.transitions = params.transitions;
    this.vocabulary = params.vocabulary;
  }

  predictNext(context: MovementToken[]): MovementPrediction | undefined {
    if (this.vocabulary.size === 0) {
      return undefined;
    }
    // Stupid-backoff: try the longest available context first, shrink on miss.
    for (let order = Math.min(this.order, context.length); order >= 0; order -= 1) {
      const key = contextKey(order === 0 ? [] : context.slice(context.length - order));
      const counts = this.transitions.get(key);
      if (!counts || counts.size === 0) {
        continue;
      }
      const candidates = rankCandidates(counts, order);
      const best = candidates[0];
      if (best) {
        return { token: best.token, probability: best.probability, backoffOrder: order, candidates };
      }
    }
    return undefined;
  }

  generate(seed: MovementToken[], maxSteps = 64): MovementToken[] {
    const produced: MovementToken[] = [];
    const context = [...seed];
    for (let step = 0; step < maxSteps; step += 1) {
      const prediction = this.predictNext(context);
      if (!prediction || prediction.token === MOVEMENT_END_TOKEN) {
        break;
      }
      produced.push(prediction.token);
      context.push(prediction.token);
    }
    return produced;
  }

  toJSON(): SerializedMovementModel {
    const transitions: Record<string, Record<MovementToken, number>> = {};
    for (const [key, counts] of this.transitions) {
      const entry: Record<MovementToken, number> = {};
      for (const [token, count] of counts) {
        entry[token] = count;
      }
      transitions[key] = entry;
    }
    return {
      version: 1,
      backendId: this.backendId,
      order: this.order,
      transitions,
      vocabulary: [...this.vocabulary].sort(),
    };
  }

  static fromJSON(serialized: SerializedMovementModel): MarkovMovementModel {
    const transitions = new Map<string, Map<MovementToken, number>>();
    for (const [key, counts] of Object.entries(serialized.transitions)) {
      transitions.set(key, new Map(Object.entries(counts)));
    }
    return new MarkovMovementModel({
      backendId: serialized.backendId,
      order: serialized.order,
      transitions,
      vocabulary: new Set(serialized.vocabulary),
    });
  }
}

function rankCandidates(counts: Map<MovementToken, number>, order: number): MovementCandidate[] {
  const total = [...counts.values()].reduce((sum, count) => sum + count, 0);
  return [...counts.entries()]
    .map(([token, count]) => ({ token, probability: total > 0 ? count / total : 0, backoffOrder: order, count }))
    .sort((a, b) => (b.count - a.count) || (a.token < b.token ? -1 : a.token > b.token ? 1 : 0))
    .map(({ token, probability, backoffOrder }) => ({ token, probability, backoffOrder }));
}

/**
 * The deterministic default backend. Learns an order-k Markov model with all
 * lower-order contexts (down to unigram) counted, enabling stupid-backoff.
 */
export class MarkovMovementBackend implements MovementModelBackend {
  readonly id = "markov-local";

  async train(dataset: MovementDataset, options: MovementTrainOptions = {}): Promise<MarkovMovementModel> {
    const order = Math.max(0, Math.trunc(options.order ?? 2));
    const transitions = new Map<string, Map<MovementToken, number>>();
    const vocabulary = new Set<MovementToken>();

    for (const sequence of dataset.sequences) {
      // Frame each sequence with start/end so the model learns entry + stopping.
      const framed = [MOVEMENT_START_TOKEN, ...sequence.tokens, MOVEMENT_END_TOKEN];
      for (let i = 1; i < framed.length; i += 1) {
        const next = framed[i]!;
        vocabulary.add(next);
        // Record the transition at every context order from 0..order.
        for (let k = 0; k <= order; k += 1) {
          if (i - k < 0) {
            break;
          }
          const key = contextKey(framed.slice(i - k, i));
          const counts = transitions.get(key) ?? new Map<MovementToken, number>();
          counts.set(next, (counts.get(next) ?? 0) + 1);
          transitions.set(key, counts);
        }
      }
    }

    return new MarkovMovementModel({ backendId: this.id, order, transitions, vocabulary });
  }
}

/**
 * Normalize a replay timeline event into a structural movement token. The exact
 * argument (window title, tap target, file path) is intentionally dropped so
 * that "tapped Submit" and "tapped Cancel" collapse to the same token family —
 * this is what lets the model *generalize* across related movements rather than
 * memorizing literals.
 */
export function tokenizeReplayEvent(event: ReplayTimelineEvent): MovementToken {
  switch (event.kind) {
    case "action":
      return `action:${event.tool}:${firstWord(event.summary)}`;
    case "observation":
      return `observation:${event.source}:${firstWord(event.summary)}`;
    case "transcript":
      return `transcript:${event.role}`;
  }
}

/**
 * Normalize a trajectory action/observation directly (without a replay
 * manifest). Mirrors {@link tokenizeReplayEvent} so both entry points agree.
 */
export function tokenizeTrajectory(trajectory: TrajectorySpan): MovementToken[] {
  const events = [
    ...trajectory.observations.map((observation) => ({
      ts: observation.ts,
      token: `observation:${observation.source}:${firstWord(observation.summary)}`,
    })),
    ...trajectory.actions.map((action) => ({
      ts: action.ts,
      token: `action:${action.tool}:${firstWord(action.summary)}`,
    })),
  ];
  return events.sort((a, b) => a.ts - b.ts).map((event) => event.token);
}

export type DatasetFromReplaysOptions = {
  /** Only include these event kinds (default: actions only — the movements). */
  include?: Array<ReplayTimelineEvent["kind"]>;
};

/**
 * Build a training dataset from replay manifests (the reviewed-export replay
 * format). By default only `action` events are kept, since those are the
 * movements to be reproduced; pass `include` to add observations/transcript for
 * richer context conditioning.
 */
export function datasetFromReplayManifests(
  manifests: ReplayManifest[],
  options: DatasetFromReplaysOptions = {},
): MovementDataset {
  const include = new Set<ReplayTimelineEvent["kind"]>(options.include ?? ["action"]);
  const sequences: MovementSequence[] = manifests.map((manifest, index) => ({
    id: manifest.trajectoryIds.join("+") || `${manifest.sessionId}#${index}`,
    tokens: manifest.events
      .filter((event) => include.has(event.kind))
      .map((event) => tokenizeReplayEvent(event)),
  }));
  return { version: 1, sequences: sequences.filter((sequence) => sequence.tokens.length > 0) };
}

export function datasetFromTrajectories(trajectories: TrajectorySpan[]): MovementDataset {
  const sequences: MovementSequence[] = trajectories
    .map((trajectory) => ({ id: trajectory.id, tokens: tokenizeTrajectory(trajectory) }))
    .filter((sequence) => sequence.tokens.length > 0);
  return { version: 1, sequences };
}

export type MovementEvalResult = {
  /** Number of next-token predictions scored. */
  predictions: number;
  /** Top-1 next-token accuracy in [0, 1]. */
  accuracy: number;
  /**
   * Fraction of predictions that required backoff below full order — a proxy
   * for how much *generalization* (vs. exact recall) the eval exercised.
   */
  generalizationRate: number;
  /** Mean probability the model assigned to the correct token. */
  meanConfidence: number;
};

/**
 * Generalization eval harness: replay held-out sequences through the model one
 * step at a time and measure how well it predicts the true next movement. Held-
 * out sequences that were never trained on exercise the backoff path, so a high
 * accuracy here means the model generalizes rather than memorizes.
 */
export function evaluateMovementModel(
  model: TrainedMovementModel,
  heldOut: MovementSequence[],
): MovementEvalResult {
  let predictions = 0;
  let correct = 0;
  let backedOff = 0;
  let confidenceSum = 0;

  for (const sequence of heldOut) {
    const framed = [MOVEMENT_START_TOKEN, ...sequence.tokens, MOVEMENT_END_TOKEN];
    for (let i = 1; i < framed.length; i += 1) {
      const context = framed.slice(0, i);
      const expected = framed[i]!;
      const prediction = model.predictNext(context);
      predictions += 1;
      if (!prediction) {
        continue;
      }
      if (prediction.backoffOrder < Math.min(model.order, context.length)) {
        backedOff += 1;
      }
      const matched = prediction.candidates.find((candidate) => candidate.token === expected);
      if (matched) {
        confidenceSum += matched.probability;
      }
      if (prediction.token === expected) {
        correct += 1;
      }
    }
  }

  return {
    predictions,
    accuracy: predictions > 0 ? correct / predictions : 0,
    generalizationRate: predictions > 0 ? backedOff / predictions : 0,
    meanConfidence: predictions > 0 ? confidenceSum / predictions : 0,
  };
}

/**
 * Deterministic synthetic movement-sequence generator. Given a set of movement
 * "motifs" (ordered token templates), it expands each into `repeats` concrete
 * sequences by rotating through variant suffixes. No RNG — a `seed` integer just
 * offsets the rotation so callers can produce disjoint train/held-out splits
 * (e.g. `seed: 0` for training, `seed: 1` for related-but-unseen eval data).
 * This lets the whole capture→dataset→train→eval pipeline be validated without
 * any real OS input.
 */
export type MovementMotif = {
  id: string;
  base: MovementToken[];
  /** Optional interchangeable trailing movements, rotated per repeat. */
  variants?: MovementToken[];
};

export function synthesizeMovementSequences(params: {
  motifs: MovementMotif[];
  repeats?: number;
  seed?: number;
}): MovementSequence[] {
  const repeats = Math.max(1, Math.trunc(params.repeats ?? 1));
  const seed = Math.trunc(params.seed ?? 0);
  const sequences: MovementSequence[] = [];
  for (const motif of params.motifs) {
    for (let r = 0; r < repeats; r += 1) {
      const variants = motif.variants ?? [];
      const tokens = [...motif.base];
      if (variants.length > 0) {
        const index = Math.abs(seed + r) % variants.length;
        tokens.push(variants[index]!);
      }
      sequences.push({ id: `${motif.id}#${seed}-${r}`, tokens });
    }
  }
  return sequences;
}

function firstWord(summary: string): string {
  const trimmed = summary.trim();
  if (trimmed.length === 0) {
    return "unknown";
  }
  const [word] = trimmed.split(/\s+/);
  return (word ?? "unknown").toLowerCase();
}
