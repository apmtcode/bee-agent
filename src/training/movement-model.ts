import type { ReplayTimelineEvent } from "../capture/replay.js";
import type { TrajectorySpan } from "../capture/trajectory.js";

/**
 * Local-movement learning: pluggable model backend.
 *
 * A recorded "movement" is a sequence of observation/action tokens captured
 * from the local computer (see `src/capture`). A movement model is trained on
 * those sequences so bee-agent can (a) repeat recorded movements and (b)
 * generalize to new-but-related movements.
 *
 * This module provides:
 *   - a compact `MovementDataset` derived from trajectories / replay timelines,
 *   - a `LocalMovementModelBackend` interface (the pluggable seam), and
 *   - `MarkovMovementBackend`: a deterministic, dependency-free, in-process
 *     backend that trains and infers entirely in the cloud/CI (no external
 *     mlx/axolotl process required). It is a smoothed variable-order Markov
 *     next-action predictor: unseen high-order contexts back off to lower-order
 *     statistics, which is what lets it generalize to related movements. It also
 *     learns an end-of-movement signal so replay rollouts terminate naturally.
 *
 * A real on-device small model (e.g. an MLX/GGUF policy) can implement the same
 * `LocalMovementModelBackend` interface later; the deterministic backend keeps
 * the whole capture -> dataset -> train -> infer -> replay pipeline testable
 * without a GPU.
 */

export type MovementToken =
  | { kind: "observation"; label: string }
  | { kind: "action"; label: string };

export type MovementSequence = {
  trajectoryId: string;
  tokens: MovementToken[];
};

export type MovementDataset = {
  version: 1;
  sequences: MovementSequence[];
};

export type MovementTrainOptions = {
  /** Maximum context length, in prior actions, the model keys on. Default 2. */
  order?: number;
};

export type MovementCandidate = {
  action: string;
  probability: number;
  count: number;
};

export type MovementPrediction = {
  /** Highest-probability next action, or undefined if nothing was learned. */
  action?: string;
  /** Probability mass of the chosen action (0..1). */
  confidence: number;
  /** Context length actually used after backoff (`-1` when nothing matched). */
  contextOrderUsed: number;
  /** True when backoff to a shorter context was required (generalization). */
  backedOff: boolean;
  /** Learned probability that the movement ends after this context (0..1). */
  terminalProbability: number;
  /** True when ending is at least as likely as any next action. */
  terminal: boolean;
  /** Candidate next actions (terminal excluded), sorted by probability then name. */
  candidates: MovementCandidate[];
};

export type SerializedMovementModel = {
  version: 1;
  backendId: string;
  order: number;
  actionVocabulary: string[];
  transitions: Record<string, Record<string, number>>;
};

export interface MovementModel {
  readonly backendId: string;
  readonly order: number;
  readonly actionVocabulary: readonly string[];
  /** Predict the next action given the movement context so far. */
  predictNext(context: MovementToken[]): MovementPrediction;
  serialize(): SerializedMovementModel;
}

export interface LocalMovementModelBackend {
  readonly id: string;
  train(dataset: MovementDataset, options?: MovementTrainOptions): MovementModel;
  restore(serialized: SerializedMovementModel): MovementModel;
}

const CONTEXT_SEPARATOR = "␟";
/** Reserved end-of-movement target; learned so replay rollouts terminate. */
const END_OF_MOVEMENT = "";
const DEFAULT_ORDER = 2;

/** Build a movement dataset from reviewed/captured trajectory spans. */
export function buildMovementDatasetFromTrajectories(trajectories: TrajectorySpan[]): MovementDataset {
  return {
    version: 1,
    sequences: trajectories.map((trajectory) => ({
      trajectoryId: trajectory.id,
      tokens: trajectorySpanToTokens(trajectory),
    })),
  };
}

/** Convert a single trajectory span into a time-ordered movement token stream. */
export function trajectorySpanToTokens(trajectory: TrajectorySpan): MovementToken[] {
  const tokens: Array<{ ts: number; order: number; token: MovementToken }> = [];
  for (const observation of trajectory.observations) {
    tokens.push({ ts: observation.ts, order: 0, token: { kind: "observation", label: observation.source } });
  }
  for (const action of trajectory.actions) {
    tokens.push({ ts: action.ts, order: 1, token: { kind: "action", label: action.tool } });
  }
  tokens.sort((a, b) => (a.ts !== b.ts ? a.ts - b.ts : a.order - b.order));
  return tokens.map((entry) => entry.token);
}

/** Build a movement sequence from a replay timeline (observations + actions). */
export function replayEventsToMovementSequence(
  trajectoryId: string,
  events: ReplayTimelineEvent[],
): MovementSequence {
  const tokens: MovementToken[] = [];
  for (const event of events) {
    if (event.kind === "observation") {
      tokens.push({ kind: "observation", label: event.source });
    } else if (event.kind === "action") {
      tokens.push({ kind: "action", label: event.tool });
    }
  }
  return { trajectoryId, tokens };
}

/** Deterministic, in-process variable-order Markov backend over action tokens. */
export class MarkovMovementBackend implements LocalMovementModelBackend {
  readonly id = "markov";

  train(dataset: MovementDataset, options?: MovementTrainOptions): MovementModel {
    const order = normalizeOrder(options?.order);
    const transitions = new Map<string, Map<string, number>>();
    const vocabulary = new Set<string>();

    for (const sequence of dataset.sequences) {
      const actions = sequence.tokens.filter((token) => token.kind === "action").map((token) => token.label);
      // Position `actions.length` records the end-of-movement target so the
      // model learns where recorded movements stop.
      for (let i = 0; i <= actions.length; i += 1) {
        const target = i < actions.length ? actions[i]! : END_OF_MOVEMENT;
        if (target !== END_OF_MOVEMENT) {
          vocabulary.add(target);
        }
        for (let k = 0; k <= order && k <= i; k += 1) {
          const key = actions.slice(i - k, i).join(CONTEXT_SEPARATOR);
          increment(transitions, key, target);
        }
      }
    }

    return new MarkovMovementModel(this.id, order, transitions, sortedVocabulary(vocabulary));
  }

  restore(serialized: SerializedMovementModel): MovementModel {
    const transitions = new Map<string, Map<string, number>>();
    for (const [key, distribution] of Object.entries(serialized.transitions)) {
      transitions.set(key, new Map(Object.entries(distribution)));
    }
    return new MarkovMovementModel(
      serialized.backendId,
      serialized.order,
      transitions,
      [...serialized.actionVocabulary],
    );
  }
}

class MarkovMovementModel implements MovementModel {
  constructor(
    readonly backendId: string,
    readonly order: number,
    private readonly transitions: Map<string, Map<string, number>>,
    readonly actionVocabulary: readonly string[],
  ) {}

  predictNext(context: MovementToken[]): MovementPrediction {
    const actions = context.filter((token) => token.kind === "action").map((token) => token.label);
    const maxK = Math.min(this.order, actions.length);
    for (let k = maxK; k >= 0; k -= 1) {
      const key = actions.slice(actions.length - k).join(CONTEXT_SEPARATOR);
      const distribution = this.transitions.get(key);
      if (distribution && distribution.size > 0) {
        return buildPrediction(distribution, k, k < maxK);
      }
    }
    return {
      action: undefined,
      confidence: 0,
      contextOrderUsed: -1,
      backedOff: maxK > 0,
      terminalProbability: 0,
      terminal: false,
      candidates: [],
    };
  }

  serialize(): SerializedMovementModel {
    const transitions: Record<string, Record<string, number>> = {};
    for (const [key, distribution] of this.transitions) {
      transitions[key] = Object.fromEntries([...distribution.entries()].sort(([a], [b]) => compareStrings(a, b)));
    }
    return {
      version: 1,
      backendId: this.backendId,
      order: this.order,
      actionVocabulary: [...this.actionVocabulary],
      transitions,
    };
  }
}

export type GenerateMovementOptions = {
  maxSteps?: number;
  /** Stop once a prediction's confidence drops below this floor. Default 0. */
  minConfidence?: number;
};

export type GeneratedMovementStep = {
  action: string;
  confidence: number;
  contextOrderUsed: number;
  backedOff: boolean;
};

/**
 * Roll out a movement sequence from a seed context by repeatedly predicting the
 * next action and feeding it back in. With a seed drawn from a trained
 * trajectory this replays recorded movements; with a novel seed it generalizes.
 * Stops at the learned end-of-movement signal, at `maxSteps`, or when confidence
 * falls below `minConfidence`.
 */
export function generateMovementSequence(
  model: MovementModel,
  seed: MovementToken[],
  options?: GenerateMovementOptions,
): GeneratedMovementStep[] {
  const maxSteps = Math.max(0, options?.maxSteps ?? 16);
  const minConfidence = options?.minConfidence ?? 0;
  const context: MovementToken[] = [...seed];
  const steps: GeneratedMovementStep[] = [];
  for (let i = 0; i < maxSteps; i += 1) {
    const prediction = model.predictNext(context);
    if (prediction.terminal || !prediction.action || prediction.confidence < minConfidence) {
      break;
    }
    steps.push({
      action: prediction.action,
      confidence: prediction.confidence,
      contextOrderUsed: prediction.contextOrderUsed,
      backedOff: prediction.backedOff,
    });
    context.push({ kind: "action", label: prediction.action });
  }
  return steps;
}

export type NextActionAccuracy = {
  evaluated: number;
  correct: number;
  accuracy: number;
  /** Fraction of predictions that required backoff (generalization pressure). */
  backoffRate: number;
};

/**
 * Teacher-forcing next-action accuracy over a (typically held-out) dataset:
 * for each action, predict from its true prefix and check the argmax. Useful as
 * a generalization signal when the eval sequences differ from the training set.
 */
export function evaluateNextActionAccuracy(model: MovementModel, dataset: MovementDataset): NextActionAccuracy {
  let evaluated = 0;
  let correct = 0;
  let backoff = 0;
  for (const sequence of dataset.sequences) {
    const prefix: MovementToken[] = [];
    for (const token of sequence.tokens) {
      if (token.kind === "action") {
        const prediction = model.predictNext(prefix);
        evaluated += 1;
        // Rank the next action independent of the stop decision: `action` is
        // undefined when the model favors terminating, but here an action does
        // follow, so score against the top-ranked candidate.
        if (prediction.candidates[0]?.action === token.label) {
          correct += 1;
        }
        if (prediction.backedOff) {
          backoff += 1;
        }
      }
      prefix.push(token);
    }
  }
  return {
    evaluated,
    correct,
    accuracy: evaluated === 0 ? 0 : correct / evaluated,
    backoffRate: evaluated === 0 ? 0 : backoff / evaluated,
  };
}

export function createMarkovMovementBackend(): MarkovMovementBackend {
  return new MarkovMovementBackend();
}

function buildPrediction(
  distribution: Map<string, number>,
  contextOrderUsed: number,
  backedOff: boolean,
): MovementPrediction {
  let total = 0;
  let terminalCount = 0;
  for (const [target, count] of distribution) {
    total += count;
    if (target === END_OF_MOVEMENT) {
      terminalCount = count;
    }
  }
  const candidates: MovementCandidate[] = [...distribution.entries()]
    .filter(([action]) => action !== END_OF_MOVEMENT)
    .map(([action, count]) => ({ action, count, probability: total === 0 ? 0 : count / total }))
    .sort((a, b) => (b.count !== a.count ? b.count - a.count : compareStrings(a.action, b.action)));
  const top = candidates[0];
  const terminalProbability = total === 0 ? 0 : terminalCount / total;
  const terminal = terminalCount > 0 && terminalCount >= (top?.count ?? 0);
  return {
    action: terminal ? undefined : top?.action,
    confidence: terminal ? terminalProbability : top?.probability ?? 0,
    contextOrderUsed,
    backedOff,
    terminalProbability,
    terminal,
    candidates,
  };
}

function increment(transitions: Map<string, Map<string, number>>, key: string, target: string): void {
  let distribution = transitions.get(key);
  if (!distribution) {
    distribution = new Map<string, number>();
    transitions.set(key, distribution);
  }
  distribution.set(target, (distribution.get(target) ?? 0) + 1);
}

function normalizeOrder(order: number | undefined): number {
  if (order === undefined || !Number.isFinite(order)) {
    return DEFAULT_ORDER;
  }
  return Math.max(0, Math.floor(order));
}

function sortedVocabulary(vocabulary: Set<string>): string[] {
  return [...vocabulary].sort(compareStrings);
}

function compareStrings(a: string, b: string): number {
  return a < b ? -1 : a > b ? 1 : 0;
}
