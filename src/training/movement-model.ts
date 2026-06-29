import type { ReplayManifest } from "../capture/replay.js";
import type { TrajectorySpan } from "../capture/trajectory.js";

/**
 * Pluggable local movement-model backend.
 *
 * Standing objective #2(c)/(d): bee-agent must be able to post-train a *local*
 * model on a recorded movement dataset so it can repeat the recorded movements,
 * and generalize to new-but-related movements. The on-device training itself
 * (mlx / axolotl) is emitted as a launch plan by {@link LocalAppleSiliconTrainingRunner}
 * and only runs on the user's real machine. This module provides the in-process,
 * fully-deterministic seam that lets bee-agent (a) define what a movement model
 * *is*, (b) train and run one in the cloud/CI for validation, and (c) swap in a
 * real on-device small-model backend without touching call sites.
 *
 * The reference backend here ({@link MarkovMovementBackend}) is a variable-order
 * Markov model with stupid-backoff. It is intentionally tiny and deterministic
 * so capture -> dataset -> train -> replay round-trips can be asserted in tests
 * without any OS input or GPU. Backoff is what gives it generalization: a prefix
 * it has never seen end-to-end is still predictable as long as its *suffix* was
 * observed during training.
 */

/** A single canonicalized movement/action drawn from a recorded trajectory. */
export type MovementEvent = {
  /** Tool/surface that produced the action (e.g. "device", "browser", "os"). */
  tool: string;
  /** Gesture/verb when present in the action metadata (tap, swipe, type, ...). */
  gesture?: string;
  /** UI target the action acted on (button label, field, element). */
  target?: string;
  /** Direction for directional gestures (up/down/left/right). */
  direction?: string;
};

/** An ordered sequence of movements captured within one trajectory. */
export type MovementSequence = {
  trajectoryId: string;
  events: MovementEvent[];
};

/** A replayable dataset of movement sequences ready for local training. */
export type MovementDataset = {
  version: 1;
  sequences: MovementSequence[];
};

export type MovementTrainingConfig = {
  /** Maximum context length the model conditions on (n-gram order). */
  maxOrder: number;
};

export const DEFAULT_MOVEMENT_TRAINING_CONFIG: MovementTrainingConfig = {
  maxOrder: 3,
};

/** A scored next-movement prediction. */
export type MovementPrediction = {
  event: MovementEvent;
  /** Conditional probability of this event given the matched context. */
  probability: number;
  /** Context length (in tokens) the prediction actually conditioned on. */
  order: number;
};

/** Serialized, replayable snapshot of a trained model (backend-agnostic shape). */
export type MovementModelSnapshot = {
  backend: string;
  maxOrder: number;
  /** context-token -> (next-token -> count). Empty context key is "". */
  transitions: Record<string, Record<string, number>>;
  /** Canonical vocabulary so tokens can be re-hydrated into MovementEvents. */
  vocabulary: Record<string, MovementEvent>;
};

/** A model produced by a backend's `train()`; deterministic given the dataset. */
export interface TrainedMovementModel {
  readonly backend: string;
  /** Most likely next movement given a context, or undefined if none learned. */
  predictNext(context: MovementEvent[]): MovementPrediction | undefined;
  /** Greedily roll out `steps` movements from a seed prefix. */
  generate(seed: MovementEvent[], steps: number): MovementEvent[];
  /** Round-trippable snapshot (e.g. to persist or ship to a device). */
  serialize(): MovementModelSnapshot;
}

/** Pluggable local-model backend. Real on-device backends implement this too. */
export interface MovementModelBackend {
  readonly name: string;
  train(dataset: MovementDataset, config?: MovementTrainingConfig): Promise<TrainedMovementModel>;
}

const TOKEN_FIELD_SEPARATOR = "|";

/** Canonical, stable string identity for a movement (the model's "token"). */
export function tokenizeMovement(event: MovementEvent): string {
  return [
    event.tool,
    event.gesture ?? "",
    event.target ?? "",
    event.direction ?? "",
  ].join(TOKEN_FIELD_SEPARATOR);
}

function detokenizeMovement(token: string): MovementEvent {
  const [tool = "", gesture = "", target = "", direction = ""] = token.split(TOKEN_FIELD_SEPARATOR);
  const event: MovementEvent = { tool };
  if (gesture) event.gesture = gesture;
  if (target) event.target = target;
  if (direction) event.direction = direction;
  return event;
}

function contextKey(tokens: string[]): string {
  return tokens.join(" ␟ ");
}

/**
 * Reference deterministic backend: a variable-order Markov model with
 * stupid-backoff. No randomness, no I/O — training and inference are pure
 * functions of the dataset, so they are safe to run (and assert) in the cloud.
 */
export class MarkovMovementBackend implements MovementModelBackend {
  readonly name = "markov-backoff";

  async train(dataset: MovementDataset, config?: MovementTrainingConfig): Promise<TrainedMovementModel> {
    const maxOrder = Math.max(1, config?.maxOrder ?? DEFAULT_MOVEMENT_TRAINING_CONFIG.maxOrder);
    const transitions: Record<string, Record<string, number>> = {};
    const vocabulary: Record<string, MovementEvent> = {};

    for (const sequence of dataset.sequences) {
      const tokens = sequence.events.map((event) => {
        const token = tokenizeMovement(event);
        if (!vocabulary[token]) {
          vocabulary[token] = detokenizeMovement(token);
        }
        return token;
      });

      for (let index = 0; index < tokens.length; index += 1) {
        const next = tokens[index];
        // Record counts for every context order 0..maxOrder ending at `index`.
        for (let order = 0; order <= maxOrder; order += 1) {
          if (index - order < 0) {
            break;
          }
          const context = tokens.slice(index - order, index);
          const key = contextKey(context);
          const bucket = (transitions[key] ??= {});
          bucket[next] = (bucket[next] ?? 0) + 1;
        }
      }
    }

    return new MarkovMovementModel({ backend: this.name, maxOrder, transitions, vocabulary });
  }
}

/** Re-hydrate a trained Markov model from a serialized snapshot. */
export function restoreMovementModel(snapshot: MovementModelSnapshot): TrainedMovementModel {
  return new MarkovMovementModel(snapshot);
}

class MarkovMovementModel implements TrainedMovementModel {
  readonly backend: string;
  private readonly maxOrder: number;
  private readonly transitions: Record<string, Record<string, number>>;
  private readonly vocabulary: Record<string, MovementEvent>;

  constructor(snapshot: MovementModelSnapshot) {
    this.backend = snapshot.backend;
    this.maxOrder = snapshot.maxOrder;
    this.transitions = snapshot.transitions;
    this.vocabulary = snapshot.vocabulary;
  }

  predictNext(context: MovementEvent[]): MovementPrediction | undefined {
    const contextTokens = context.map(tokenizeMovement);
    // Stupid-backoff: try the longest context first, shorten until a hit.
    for (let order = Math.min(this.maxOrder, contextTokens.length); order >= 0; order -= 1) {
      const key = contextKey(contextTokens.slice(contextTokens.length - order));
      const bucket = this.transitions[key];
      if (!bucket) {
        continue;
      }
      const best = pickBest(bucket);
      if (!best) {
        continue;
      }
      const total = Object.values(bucket).reduce((sum, count) => sum + count, 0);
      const event = this.vocabulary[best.token] ?? detokenizeMovement(best.token);
      return { event, probability: best.count / total, order };
    }
    return undefined;
  }

  generate(seed: MovementEvent[], steps: number): MovementEvent[] {
    const produced: MovementEvent[] = [];
    const context = [...seed];
    for (let step = 0; step < steps; step += 1) {
      const prediction = this.predictNext(context);
      if (!prediction) {
        break;
      }
      produced.push(prediction.event);
      context.push(prediction.event);
    }
    return produced;
  }

  serialize(): MovementModelSnapshot {
    return {
      backend: this.backend,
      maxOrder: this.maxOrder,
      transitions: this.transitions,
      vocabulary: this.vocabulary,
    };
  }
}

/** Deterministic argmax over a count bucket; ties broken by token string. */
function pickBest(bucket: Record<string, number>): { token: string; count: number } | undefined {
  let best: { token: string; count: number } | undefined;
  for (const [token, count] of Object.entries(bucket)) {
    if (!best || count > best.count || (count === best.count && token < best.token)) {
      best = { token, count };
    }
  }
  return best;
}

/** Extract a canonical MovementEvent from a recorded trajectory action. */
function actionToMovement(action: TrajectorySpan["actions"][number]): MovementEvent {
  const metadata = action.metadata ?? {};
  const event: MovementEvent = { tool: action.tool };
  const gesture = readString(metadata.gesture);
  const target = readString(metadata.target);
  const direction = readString(metadata.direction);
  if (gesture) event.gesture = gesture;
  if (target) event.target = target;
  if (direction) event.direction = direction;
  return event;
}

function readString(value: unknown): string | undefined {
  return typeof value === "string" && value.length > 0 ? value : undefined;
}

/**
 * Build a movement dataset from recorded trajectory spans. Uses each
 * trajectory's reviewed/redacted actions when present (so only consented,
 * review-approved movements feed training) and falls back to raw actions.
 */
export function buildMovementDataset(trajectories: TrajectorySpan[]): MovementDataset {
  const sequences: MovementSequence[] = [];
  for (const trajectory of trajectories) {
    const reviewedActions = trajectory.review?.redactedActions;
    const events = reviewedActions
      ? reviewedActions.map<MovementEvent>((action) => ({ tool: action.tool }))
      : trajectory.actions
          .slice()
          .sort((a, b) => a.ts - b.ts)
          .map(actionToMovement);
    if (events.length > 0) {
      sequences.push({ trajectoryId: trajectory.id, events });
    }
  }
  return { version: 1, sequences };
}

/** Build a movement dataset from one or more replay manifests. */
export function buildMovementDatasetFromReplays(replays: ReplayManifest[]): MovementDataset {
  const byTrajectory = new Map<string, MovementEvent[]>();
  for (const replay of replays) {
    for (const event of replay.events) {
      if (event.kind !== "action") {
        continue;
      }
      const list = byTrajectory.get(event.trajectoryId) ?? [];
      list.push({ tool: event.tool });
      byTrajectory.set(event.trajectoryId, list);
    }
  }
  const sequences: MovementSequence[] = [];
  for (const [trajectoryId, events] of byTrajectory) {
    if (events.length > 0) {
      sequences.push({ trajectoryId, events });
    }
  }
  return { version: 1, sequences };
}

export type MovementEvalResult = {
  /** Number of (context -> next) predictions scored. */
  predictions: number;
  /** Predictions whose argmax matched the held-out next movement. */
  correct: number;
  /** correct / predictions (0 when there is nothing to predict). */
  accuracy: number;
};

/**
 * Generalization eval harness: for each sequence, walk every prefix and ask the
 * model to predict the next movement, scoring against ground truth. Run on
 * held-out (but related) sequences this measures generalization, not memory.
 */
export function evaluateNextActionAccuracy(
  model: TrainedMovementModel,
  sequences: MovementSequence[],
): MovementEvalResult {
  let predictions = 0;
  let correct = 0;
  for (const sequence of sequences) {
    for (let index = 1; index < sequence.events.length; index += 1) {
      const context = sequence.events.slice(0, index);
      const expected = sequence.events[index];
      const prediction = model.predictNext(context);
      predictions += 1;
      if (prediction && tokenizeMovement(prediction.event) === tokenizeMovement(expected)) {
        correct += 1;
      }
    }
  }
  return {
    predictions,
    correct,
    accuracy: predictions === 0 ? 0 : correct / predictions,
  };
}

/**
 * Deterministic, seedable synthetic movement-stream generator. Lets the engine
 * validate capture -> dataset -> train -> replay round-trips in the cloud with
 * no real OS input. Sequences are drawn from a small grammar of recurring
 * motifs so a trained model has structure to learn and generalize over.
 */
export function generateSyntheticMovementDataset(params: {
  seed: number;
  sequenceCount: number;
  minLength?: number;
  maxLength?: number;
}): MovementDataset {
  const rng = mulberry32(params.seed >>> 0);
  const minLength = Math.max(2, params.minLength ?? 4);
  const maxLength = Math.max(minLength, params.maxLength ?? 8);

  // Motifs: an action tends to be followed by a characteristic next action,
  // giving the Markov model learnable, generalizable structure.
  const motifs: MovementEvent[][] = [
    [
      { tool: "device", gesture: "tap", target: "search" },
      { tool: "device", gesture: "type", target: "query" },
      { tool: "device", gesture: "tap", target: "result" },
    ],
    [
      { tool: "browser", gesture: "scroll", direction: "down" },
      { tool: "browser", gesture: "tap", target: "link" },
      { tool: "browser", gesture: "scroll", direction: "up" },
    ],
    [
      { tool: "os", gesture: "shortcut", target: "copy" },
      { tool: "os", gesture: "shortcut", target: "switch-window" },
      { tool: "os", gesture: "shortcut", target: "paste" },
    ],
  ];

  const sequences: MovementSequence[] = [];
  for (let i = 0; i < params.sequenceCount; i += 1) {
    const targetLength = minLength + Math.floor(rng() * (maxLength - minLength + 1));
    const events: MovementEvent[] = [];
    while (events.length < targetLength) {
      const motif = motifs[Math.floor(rng() * motifs.length)] ?? motifs[0];
      for (const event of motif) {
        if (events.length >= targetLength) {
          break;
        }
        events.push({ ...event });
      }
    }
    sequences.push({ trajectoryId: `synthetic-${params.seed}-${i}`, events });
  }
  return { version: 1, sequences };
}

/** Tiny deterministic PRNG (mulberry32). Avoids Math.random for reproducibility. */
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
