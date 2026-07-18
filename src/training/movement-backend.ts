import type { ReplayManifest, ReplayTimelineEvent } from "../capture/replay.js";
import type { TrajectorySpan } from "../capture/trajectory.js";

/**
 * Pluggable local-movement model backend.
 *
 * bee-agent's on-device training runner ({@link LocalAppleSiliconTrainingRunner})
 * emits command plans for real Apple-Silicon runtimes (mlx / axolotl) that can
 * only execute on the user's machine. That leaves the cloud self-evolution
 * engine unable to exercise the actual *learn -> repeat -> generalize* loop that
 * objective #2 of the movement subsystem calls for.
 *
 * This module introduces a backend-agnostic contract for that loop:
 *   - {@link buildMovementDataset} turns a reviewed replay/trajectory timeline
 *     into a supervised next-action dataset.
 *   - {@link MovementModelBackend} trains a model from that dataset and predicts
 *     the next movement given a context window.
 *   - {@link MockMovementBackend} is a deterministic, dependency-free reference
 *     backend (a stupid-backoff n-gram over event tokens) so the whole pipeline
 *     is testable in CI without a GPU, a real model, or OS input.
 *
 * A real on-device backend (small local model + on-device fine-tune) implements
 * the same interface and is swapped in when bee-agent runs locally.
 */

export type MovementToken = string;

export type MovementAction = {
  tool: string;
  summary: string;
};

/** One supervised decision point: recent context -> the action that followed. */
export type MovementSample = {
  /** Most-recent-last window of tokenized events preceding the action. */
  context: MovementToken[];
  action: MovementAction;
};

export type MovementDataset = {
  version: 1;
  contextWindow: number;
  samples: MovementSample[];
  /** Sorted unique tokens seen — the seam a real backend uses for embeddings. */
  vocabulary: MovementToken[];
};

export type MovementTokenizer = {
  observation: (event: Extract<ReplayTimelineEvent, { kind: "observation" }>) => MovementToken;
  transcript: (event: Extract<ReplayTimelineEvent, { kind: "transcript" }>) => MovementToken;
  action: (event: Extract<ReplayTimelineEvent, { kind: "action" }>) => MovementToken;
};

/**
 * Default tokenizer. Deliberately *coarse* on context (source/tool only) so the
 * model can generalize across trajectories that differ only in surface detail,
 * while the predicted {@link MovementAction} keeps the full summary so replays
 * stay faithful.
 */
export const DEFAULT_MOVEMENT_TOKENIZER: MovementTokenizer = {
  observation: (event) => `obs:${event.source}`,
  transcript: (event) => `msg:${event.role}`,
  action: (event) => `act:${event.tool}`,
};

export type BuildMovementDatasetOptions = {
  contextWindow?: number;
  tokenizer?: MovementTokenizer;
};

const DEFAULT_CONTEXT_WINDOW = 4;

/** Build a next-action dataset from ordered replay timelines. */
export function buildMovementDataset(
  replays: ReplayManifest[],
  options: BuildMovementDatasetOptions = {},
): MovementDataset {
  const contextWindow = Math.max(0, options.contextWindow ?? DEFAULT_CONTEXT_WINDOW);
  const tokenizer = options.tokenizer ?? DEFAULT_MOVEMENT_TOKENIZER;
  const samples: MovementSample[] = [];
  const vocabulary = new Set<MovementToken>();

  for (const replay of replays) {
    const window: MovementToken[] = [];
    for (const event of replay.events) {
      const token = tokenizeEvent(event, tokenizer);
      vocabulary.add(token);
      if (event.kind === "action") {
        samples.push({
          context: window.slice(-contextWindow),
          action: { tool: event.tool, summary: event.summary },
        });
      }
      window.push(token);
    }
  }

  return {
    version: 1,
    contextWindow,
    samples,
    vocabulary: [...vocabulary].sort(),
  };
}

/** Convenience: build a dataset straight from trajectory spans (no transcript). */
export function buildMovementDatasetFromTrajectories(
  trajectories: TrajectorySpan[],
  options: BuildMovementDatasetOptions = {},
): MovementDataset {
  const replays: ReplayManifest[] = trajectories.map((trajectory) => ({
    version: 1,
    sessionId: trajectory.sessionId,
    trajectoryIds: [trajectory.id],
    eventCount: trajectory.observations.length + trajectory.actions.length,
    events: [
      ...trajectory.observations.map<ReplayTimelineEvent>((observation) => ({
        kind: "observation",
        ts: observation.ts,
        trajectoryId: trajectory.id,
        source: observation.source,
        summary: observation.summary,
      })),
      ...trajectory.actions.map<ReplayTimelineEvent>((action) => ({
        kind: "action",
        ts: action.ts,
        trajectoryId: trajectory.id,
        tool: action.tool,
        summary: action.summary,
      })),
    ].sort((a, b) => a.ts - b.ts),
  }));
  return buildMovementDataset(replays, options);
}

function tokenizeEvent(event: ReplayTimelineEvent, tokenizer: MovementTokenizer): MovementToken {
  switch (event.kind) {
    case "observation":
      return tokenizer.observation(event);
    case "transcript":
      return tokenizer.transcript(event);
    case "action":
      return tokenizer.action(event);
  }
}

export type MovementPrediction = {
  action: MovementAction;
  /** How many context tokens matched (backoff length); 0 == prior only. */
  matchedContextLength: number;
  /** Fraction of observations at the matched context that took this action. */
  confidence: number;
};

/** A trained model artifact. Serializable so it can be persisted as JSON. */
export type TrainedMovementModel = {
  backend: string;
  version: 1;
  contextWindow: number;
  vocabulary: MovementToken[];
  sampleCount: number;
  /** Backend-specific learned parameters. */
  parameters: unknown;
};

export type MovementTrainConfig = {
  /** Max context length the backend conditions on (defaults to dataset window). */
  contextWindow?: number;
};

export interface MovementModelBackend {
  readonly id: string;
  train(dataset: MovementDataset, config?: MovementTrainConfig): Promise<TrainedMovementModel>;
  infer(model: TrainedMovementModel, context: MovementToken[]): Promise<MovementPrediction | undefined>;
}

type ActionCounts = Record<string, { action: MovementAction; count: number }>;
type BackoffTable = Record<string, ActionCounts>;

type MockModelParameters = {
  /** contextKey (n-gram of tokens, "" == unconditional) -> action counts. */
  table: BackoffTable;
};

/**
 * Deterministic reference backend: a stupid-backoff n-gram over event tokens.
 *
 * Training records, for every context length 0..N, how often each action
 * followed that exact token suffix. Inference tries the longest context first
 * and backs off to shorter suffixes (and finally the unconditional prior),
 * which is what lets it *generalize* to a new-but-related context whose full
 * prefix was never seen. Ties break by action key sort, so it is fully
 * deterministic — no clock, no RNG, identical model for identical input.
 */
export class MockMovementBackend implements MovementModelBackend {
  readonly id = "mock-ngram";

  async train(dataset: MovementDataset, config: MovementTrainConfig = {}): Promise<TrainedMovementModel> {
    const contextWindow = Math.max(0, config.contextWindow ?? dataset.contextWindow);
    const table: BackoffTable = {};

    for (const sample of dataset.samples) {
      const trimmed = sample.context.slice(-contextWindow);
      for (let length = 0; length <= trimmed.length; length += 1) {
        const key = contextKey(trimmed.slice(trimmed.length - length));
        const counts = (table[key] ??= {});
        const actionKey = actionToKey(sample.action);
        const entry = (counts[actionKey] ??= { action: sample.action, count: 0 });
        entry.count += 1;
      }
    }

    return {
      backend: this.id,
      version: 1,
      contextWindow,
      vocabulary: dataset.vocabulary,
      sampleCount: dataset.samples.length,
      parameters: { table } satisfies MockModelParameters,
    };
  }

  async infer(
    model: TrainedMovementModel,
    context: MovementToken[],
  ): Promise<MovementPrediction | undefined> {
    const { table } = model.parameters as MockModelParameters;
    const trimmed = context.slice(-model.contextWindow);
    for (let length = trimmed.length; length >= 0; length -= 1) {
      const key = contextKey(trimmed.slice(trimmed.length - length));
      const counts = table[key];
      if (!counts) {
        continue;
      }
      const best = argmaxAction(counts);
      if (!best) {
        continue;
      }
      const total = Object.values(counts).reduce((sum, entry) => sum + entry.count, 0);
      return {
        action: best.action,
        matchedContextLength: length,
        confidence: total > 0 ? best.count / total : 0,
      };
    }
    return undefined;
  }
}

export type MovementRolloutStep = MovementPrediction & { step: number };

export type MovementRolloutOptions = {
  maxSteps?: number;
  tokenizer?: MovementTokenizer;
  /** Stop when a predicted action repeats the previous one (loop guard). */
  stopOnRepeat?: boolean;
};

/**
 * Autoregressively roll the model out from a seed context, feeding each
 * predicted action back in as the next token. This is how a trained model
 * "repeats the recorded movements" (objective #2c) and, off a novel seed,
 * attempts a related-but-new sequence (objective #2d).
 */
export async function rolloutMovements(
  backend: MovementModelBackend,
  model: TrainedMovementModel,
  seedContext: MovementToken[],
  options: MovementRolloutOptions = {},
): Promise<MovementRolloutStep[]> {
  const maxSteps = Math.max(0, options.maxSteps ?? 16);
  const tokenizer = options.tokenizer ?? DEFAULT_MOVEMENT_TOKENIZER;
  const context = [...seedContext];
  const steps: MovementRolloutStep[] = [];
  let previousActionKey: string | undefined;

  for (let step = 0; step < maxSteps; step += 1) {
    const prediction = await backend.infer(model, context);
    if (!prediction) {
      break;
    }
    const actionKey = actionToKey(prediction.action);
    if (options.stopOnRepeat && actionKey === previousActionKey) {
      break;
    }
    steps.push({ ...prediction, step });
    previousActionKey = actionKey;
    context.push(tokenizer.action({
      kind: "action",
      ts: step,
      trajectoryId: "rollout",
      tool: prediction.action.tool,
      summary: prediction.action.summary,
    }));
  }

  return steps;
}

export type MovementFidelityReport = {
  total: number;
  matched: number;
  accuracy: number;
  /** Average backoff length used across predictions (higher == more specific). */
  averageMatchedContextLength: number;
};

/**
 * Held-out fidelity eval: for each sample, does the model predict the recorded
 * action? Seeds the generalization eval harness the roadmap calls for.
 */
export async function evaluateMovementFidelity(
  backend: MovementModelBackend,
  model: TrainedMovementModel,
  dataset: MovementDataset,
): Promise<MovementFidelityReport> {
  let matched = 0;
  let contextLengthSum = 0;
  for (const sample of dataset.samples) {
    const prediction = await backend.infer(model, sample.context);
    if (prediction && actionToKey(prediction.action) === actionToKey(sample.action)) {
      matched += 1;
    }
    contextLengthSum += prediction?.matchedContextLength ?? 0;
  }
  const total = dataset.samples.length;
  return {
    total,
    matched,
    accuracy: total > 0 ? matched / total : 0,
    averageMatchedContextLength: total > 0 ? contextLengthSum / total : 0,
  };
}

function contextKey(tokens: MovementToken[]): string {
  return tokens.join("␟");
}

function actionToKey(action: MovementAction): string {
  return `${action.tool}␟${action.summary}`;
}

function argmaxAction(counts: ActionCounts): { action: MovementAction; count: number } | undefined {
  let best: { action: MovementAction; count: number } | undefined;
  let bestKey = "";
  for (const [key, entry] of Object.entries(counts)) {
    if (!best || entry.count > best.count || (entry.count === best.count && key < bestKey)) {
      best = entry;
      bestKey = key;
    }
  }
  return best;
}
