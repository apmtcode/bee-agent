import type { TrajectorySpan } from "../capture/trajectory.js";
import type { ReplayManifest, ReplayTimelineEvent } from "../capture/replay.js";

/**
 * In-process, cloud-safe movement-learning backend.
 *
 * Objective 2 of the self-evolution charter asks bee-agent to (c) post-train a
 * local model on a recorded movement dataset so it can repeat the movements, and
 * (d) generalize to new-but-related movements. The {@link LocalAppleSiliconTrainingRunner}
 * already emits shell commands for *real* on-device trainers (mlx/axolotl), but
 * that pipeline cannot run — or be tested — in the cloud.
 *
 * This module supplies the missing seam: a pluggable {@link MovementModelBackend}
 * plus a deterministic reference backend ({@link MarkovMovementBackend}) that
 * trains and predicts entirely in-process with no external dependencies and no
 * randomness. Real on-device backends implement the same interface; tests and
 * cloud runs use the mock. The learned model is a plain JSON object, so it can be
 * persisted with the existing `writeJsonAtomic` helper as a genuine artifact.
 */

/** Field separator inside a context key; chosen to never collide with token text. */
const CONTEXT_DELIMITER = "";

/** Synthetic action token appended to every sequence so the model learns to stop. */
export const MOVEMENT_STOP_TOKEN = "<stop>";

export type MovementEventKind = "observation" | "action";

/** A single tokenized step in a movement sequence. */
export type MovementStep = {
  kind: MovementEventKind;
  /** Stable vocabulary token, e.g. `act:mouse.move:drag-window`. */
  token: string;
  ts: number;
};

/** One recorded trajectory rendered as an ordered token stream. */
export type MovementSequence = {
  trajectoryId: string;
  sessionId?: string;
  steps: MovementStep[];
};

/** A dataset of tokenized movement sequences ready for training. */
export type MovementDataset = {
  version: 1;
  /** Backoff context length used when building/consuming the model. */
  order: number;
  sequences: MovementSequence[];
  /** Sorted, de-duplicated vocabulary across all steps. */
  vocabulary: string[];
  /** Subset of the vocabulary that represents actions (prediction targets). */
  actionTokens: string[];
};

export type MovementTrainingConfig = {
  /** Max context length for the highest-order transition table. */
  order?: number;
};

/** Serializable trained model. Deterministic backends emit pure JSON. */
export type MovementModel = {
  version: 1;
  backend: string;
  order: number;
  vocabulary: string[];
  actionTokens: string[];
  /**
   * context-key -> (next-action-token -> count). The empty-string key holds the
   * unigram action prior; longer keys hold higher-order transitions. Predictions
   * back off from the longest available context toward the prior.
   */
  contexts: Record<string, Record<string, number>>;
  /** Number of sequences the model was trained on (diagnostics only). */
  sequenceCount: number;
};

export type MovementPredictionSource = "exact" | "backoff" | "prior" | "unknown";

export type MovementPrediction = {
  /** Most likely next action token, or `undefined` when the model is empty. */
  token?: string;
  /** Probability mass of {@link token} within the matched context distribution. */
  confidence: number;
  /** How the prediction was resolved: exact context, backed-off, or global prior. */
  source: MovementPredictionSource;
  /** Length of the context that produced the prediction. */
  matchedOrder: number;
  /** Ranked candidate tokens (deterministic order), best first. */
  candidates: Array<{ token: string; probability: number }>;
};

/**
 * Pluggable local-model interface. Swap {@link MarkovMovementBackend} for a real
 * on-device backend (mlx/gguf, etc.) without touching callers.
 */
export interface MovementModelBackend {
  readonly name: string;
  train(dataset: MovementDataset, config?: MovementTrainingConfig): MovementModel | Promise<MovementModel>;
  predict(model: MovementModel, context: readonly string[]): MovementPrediction;
}

function slug(value: string): string {
  const normalized = value
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
  return normalized.length > 48 ? normalized.slice(0, 48).replace(/-+$/g, "") : normalized;
}

/** Tokenize an observation event into a stable vocabulary token. */
export function observationToken(source: string, summary: string): string {
  return `obs:${slug(source)}:${slug(summary)}`;
}

/** Tokenize an action event into a stable vocabulary token. */
export function actionToken(tool: string, summary: string): string {
  return `act:${slug(tool)}:${slug(summary)}`;
}

/** True when a token represents an action (a valid prediction target). */
export function isActionToken(token: string): boolean {
  return token.startsWith("act:") || token === MOVEMENT_STOP_TOKEN;
}

/** Render a captured trajectory as an ordered, tokenized movement sequence. */
export function movementSequenceFromTrajectory(trajectory: TrajectorySpan): MovementSequence {
  const steps: MovementStep[] = [
    ...trajectory.observations.map<MovementStep>((observation) => ({
      kind: "observation",
      token: observationToken(observation.source, observation.summary),
      ts: observation.ts,
    })),
    ...trajectory.actions.map<MovementStep>((action) => ({
      kind: "action",
      token: actionToken(action.tool, action.summary),
      ts: action.ts,
    })),
  ].sort((a, b) => (a.ts !== b.ts ? a.ts - b.ts : kindOrder(a.kind) - kindOrder(b.kind)));

  return { trajectoryId: trajectory.id, sessionId: trajectory.sessionId, steps };
}

/** Render a replay manifest's timeline into a tokenized movement sequence. */
export function movementSequenceFromReplay(replay: ReplayManifest): MovementSequence {
  const steps: MovementStep[] = replay.events.flatMap((event) => tokenizeReplayEvent(event));
  const trajectoryId = replay.trajectoryIds[0] ?? replay.sessionId;
  return { trajectoryId, sessionId: replay.sessionId, steps };
}

function tokenizeReplayEvent(event: ReplayTimelineEvent): MovementStep[] {
  switch (event.kind) {
    case "observation":
      return [{ kind: "observation", token: observationToken(event.source, event.summary), ts: event.ts }];
    case "action":
      return [{ kind: "action", token: actionToken(event.tool, event.summary), ts: event.ts }];
    case "transcript":
      // Transcript turns are context, not movements; model them as observations.
      return [{ kind: "observation", token: observationToken(`transcript.${event.role}`, event.content), ts: event.ts }];
  }
}

function kindOrder(kind: MovementEventKind): number {
  return kind === "observation" ? 0 : 1;
}

/** Assemble a training-ready dataset from tokenized movement sequences. */
export function buildMovementDataset(
  sequences: MovementSequence[],
  options: { order?: number } = {},
): MovementDataset {
  const order = Math.max(1, options.order ?? 3);
  const vocabulary = new Set<string>();
  const actionTokens = new Set<string>();
  for (const sequence of sequences) {
    for (const step of sequence.steps) {
      vocabulary.add(step.token);
      if (step.kind === "action") {
        actionTokens.add(step.token);
      }
    }
  }
  actionTokens.add(MOVEMENT_STOP_TOKEN);
  vocabulary.add(MOVEMENT_STOP_TOKEN);
  return {
    version: 1,
    order,
    sequences,
    vocabulary: [...vocabulary].sort(),
    actionTokens: [...actionTokens].sort(),
  };
}

/** Convenience: build a dataset straight from captured trajectory spans. */
export function movementDatasetFromTrajectories(
  trajectories: TrajectorySpan[],
  options: { order?: number } = {},
): MovementDataset {
  return buildMovementDataset(trajectories.map(movementSequenceFromTrajectory), options);
}

function contextKey(tokens: readonly string[]): string {
  return tokens.join(CONTEXT_DELIMITER);
}

/**
 * Deterministic variable-order Markov backend.
 *
 * Training records, for every action in every sequence, the distribution of that
 * action across each backoff context length 0..order (0 = global prior). At
 * inference time it matches the longest available context, backing off toward the
 * prior — so contexts seen verbatim in training reproduce the recorded movement
 * exactly (objective 2c), while unseen-but-related contexts fall back to shared
 * shorter contexts and still predict a plausible next movement (objective 2d).
 */
export class MarkovMovementBackend implements MovementModelBackend {
  readonly name = "markov-mock";

  train(dataset: MovementDataset, config: MovementTrainingConfig = {}): MovementModel {
    const order = Math.max(1, config.order ?? dataset.order);
    const contexts: Record<string, Record<string, number>> = {};

    const record = (key: string, target: string): void => {
      const distribution = (contexts[key] ??= {});
      distribution[target] = (distribution[target] ?? 0) + 1;
    };

    for (const sequence of dataset.sequences) {
      const tokens = sequence.steps.map((step) => step.token);
      // Append the stop token so the model learns where a movement sequence ends.
      const withStop = [...tokens, MOVEMENT_STOP_TOKEN];
      for (let i = 0; i < withStop.length; i += 1) {
        const target = withStop[i]!;
        if (!isActionToken(target)) {
          continue;
        }
        const maxContext = Math.min(order, i);
        for (let k = 0; k <= maxContext; k += 1) {
          record(contextKey(withStop.slice(i - k, i)), target);
        }
      }
    }

    return {
      version: 1,
      backend: this.name,
      order,
      vocabulary: [...dataset.vocabulary],
      actionTokens: [...dataset.actionTokens],
      contexts,
      sequenceCount: dataset.sequences.length,
    };
  }

  predict(model: MovementModel, context: readonly string[]): MovementPrediction {
    const maxContext = Math.min(model.order, context.length);
    for (let k = maxContext; k >= 0; k -= 1) {
      const key = contextKey(context.slice(context.length - k));
      const distribution = model.contexts[key];
      if (!distribution) {
        continue;
      }
      const ranked = rankDistribution(distribution);
      if (ranked.length === 0) {
        continue;
      }
      const source: MovementPredictionSource = k === maxContext && k > 0 ? "exact" : k === 0 ? "prior" : "backoff";
      return {
        token: ranked[0]!.token,
        confidence: ranked[0]!.probability,
        source,
        matchedOrder: k,
        candidates: ranked,
      };
    }
    return { token: undefined, confidence: 0, source: "unknown", matchedOrder: 0, candidates: [] };
  }
}

function rankDistribution(distribution: Record<string, number>): Array<{ token: string; probability: number }> {
  const total = Object.values(distribution).reduce((sum, count) => sum + count, 0);
  if (total === 0) {
    return [];
  }
  return Object.entries(distribution)
    .map(([token, count]) => ({ token, probability: count / total }))
    // Highest probability first; deterministic lexicographic tie-break.
    .sort((a, b) => (b.probability !== a.probability ? b.probability - a.probability : a.token < b.token ? -1 : 1));
}

export type MovementGenerationResult = {
  /** Predicted action tokens in order, excluding the trailing stop token. */
  tokens: string[];
  /** True when generation ended on the model's stop token rather than the cap. */
  stopped: boolean;
  steps: Array<{ token: string; source: MovementPredictionSource; confidence: number }>;
};

/**
 * Greedily roll out a movement sequence from a seed context. Used both to replay
 * a recorded movement (seed with its opening context) and to generalize (seed
 * with a related-but-unseen context). Deterministic: always takes the argmax.
 */
export function generateMovementSequence(
  backend: MovementModelBackend,
  model: MovementModel,
  seed: readonly string[] = [],
  options: { maxSteps?: number } = {},
): MovementGenerationResult {
  const maxSteps = Math.max(1, options.maxSteps ?? 64);
  const context = [...seed];
  const tokens: string[] = [];
  const steps: MovementGenerationResult["steps"] = [];
  for (let i = 0; i < maxSteps; i += 1) {
    const prediction = backend.predict(model, context);
    if (!prediction.token || prediction.token === MOVEMENT_STOP_TOKEN) {
      return { tokens, stopped: Boolean(prediction.token), steps };
    }
    tokens.push(prediction.token);
    steps.push({ token: prediction.token, source: prediction.source, confidence: prediction.confidence });
    context.push(prediction.token);
  }
  return { tokens, stopped: false, steps };
}

export type ReplayFidelityReport = {
  /** Fraction of recorded actions the model's top prediction reproduced. */
  accuracy: number;
  totalActions: number;
  correct: number;
  perSequence: Array<{ trajectoryId: string; totalActions: number; correct: number }>;
};

/**
 * Teacher-forced replay fidelity: walk each recorded sequence and, at every
 * action, check whether the model's top prediction (given the true preceding
 * context) matches the recorded action. Measures objective 2c — repeating the
 * recorded movements.
 */
export function evaluateReplayFidelity(
  backend: MovementModelBackend,
  model: MovementModel,
  sequences: MovementSequence[],
): ReplayFidelityReport {
  let totalActions = 0;
  let correct = 0;
  const perSequence: ReplayFidelityReport["perSequence"] = [];

  for (const sequence of sequences) {
    const tokens = sequence.steps.map((step) => step.token);
    let seqTotal = 0;
    let seqCorrect = 0;
    for (let i = 0; i < tokens.length; i += 1) {
      const target = tokens[i]!;
      if (!isActionToken(target)) {
        continue;
      }
      seqTotal += 1;
      const prediction = backend.predict(model, tokens.slice(0, i));
      if (prediction.token === target) {
        seqCorrect += 1;
      }
    }
    totalActions += seqTotal;
    correct += seqCorrect;
    perSequence.push({ trajectoryId: sequence.trajectoryId, totalActions: seqTotal, correct: seqCorrect });
  }

  return {
    accuracy: totalActions === 0 ? 1 : correct / totalActions,
    totalActions,
    correct,
    perSequence,
  };
}

export type GeneralizationReport = {
  /** Next-action accuracy on held-out sequences (objective 2d). */
  accuracy: number;
  totalActions: number;
  correct: number;
  trainSequenceCount: number;
  holdoutSequenceCount: number;
  /** Share of correct holdout predictions resolved via context backoff. */
  backoffShare: number;
};

/**
 * Deterministic hold-out generalization eval. Splits sequences (last
 * `holdoutCount` become the test set), trains on the rest, and measures
 * teacher-forced next-action accuracy on the held-out sequences the model never
 * trained on. Because the held-out contexts are unseen at full order, correct
 * predictions come from shared shorter contexts — i.e. real generalization.
 */
export function evaluateGeneralization(
  backend: MovementModelBackend,
  dataset: MovementDataset,
  options: { holdoutCount?: number } = {},
): GeneralizationReport {
  const holdoutCount = Math.max(1, Math.min(options.holdoutCount ?? 1, Math.max(0, dataset.sequences.length - 1)));
  const splitIndex = dataset.sequences.length - holdoutCount;
  const trainSequences = dataset.sequences.slice(0, splitIndex);
  const holdoutSequences = dataset.sequences.slice(splitIndex);

  const trainDataset = buildMovementDataset(trainSequences, { order: dataset.order });
  const model = syncTrain(backend, trainDataset);

  let totalActions = 0;
  let correct = 0;
  let backoffCorrect = 0;
  for (const sequence of holdoutSequences) {
    const tokens = sequence.steps.map((step) => step.token);
    for (let i = 0; i < tokens.length; i += 1) {
      const target = tokens[i]!;
      if (!isActionToken(target)) {
        continue;
      }
      totalActions += 1;
      const prediction = backend.predict(model, tokens.slice(0, i));
      if (prediction.token === target) {
        correct += 1;
        if (prediction.source === "backoff" || prediction.source === "prior") {
          backoffCorrect += 1;
        }
      }
    }
  }

  return {
    accuracy: totalActions === 0 ? 0 : correct / totalActions,
    totalActions,
    correct,
    trainSequenceCount: trainSequences.length,
    holdoutSequenceCount: holdoutSequences.length,
    backoffShare: correct === 0 ? 0 : backoffCorrect / correct,
  };
}

/**
 * Synchronously realize a backend's trained model. The reference backend trains
 * synchronously; this guards the async interface seam without forcing eval
 * helpers to be async for the common in-process case.
 */
function syncTrain(backend: MovementModelBackend, dataset: MovementDataset): MovementModel {
  const model = backend.train(dataset);
  if (model instanceof Promise) {
    throw new Error("evaluateGeneralization requires a synchronous backend; await the model and call the eval directly");
  }
  return model;
}
