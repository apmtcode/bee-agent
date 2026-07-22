import type { ReviewedExportManifest } from "./export-manifest.js";

/**
 * Pluggable local-model training backend for the movement-learning subsystem.
 *
 * The reviewed export manifest (see {@link ReviewedExportManifest}) carries
 * `replays` — ordered timelines of transcript / observation / action events that
 * describe recorded movements. This module turns those timelines into a
 * next-action prediction dataset and defines the backend seam that trains a
 * local model on it.
 *
 * bee-agent runs in the cloud with no access to a real machine, so the real
 * on-device backends (mlx / axolotl, wired up by {@link LocalAppleSiliconTrainingRunner})
 * cannot execute here. This file provides a fully in-process, deterministic
 * {@link MockNearestNeighborBackend} that *does* train and infer, so the whole
 * capture → dataset → train → infer → evaluate loop can be validated in CI with
 * synthetic event streams. Swap in a real backend by implementing
 * {@link LocalModelBackend} and registering it.
 */

export const MOVEMENT_DATASET_VERSION = 1 as const;
export const SERIALIZED_MOVEMENT_MODEL_VERSION = 1 as const;

/** Default number of preceding timeline events kept as an action's context. */
export const DEFAULT_CONTEXT_WINDOW = 16;

export type MovementContextEvent = {
  kind: "transcript" | "observation" | "action";
  /** Normalized human-readable text: transcript content, observation/action summary. */
  text: string;
  /** Provenance token: transcript role, observation source, or prior action's tool. */
  source: string;
};

export type MovementAction = {
  tool: string;
  summary: string;
};

export type MovementSample = {
  trajectoryId: string;
  sessionId: string;
  /** Index of this action within its trajectory's action sequence (0-based). */
  stepIndex: number;
  context: MovementContextEvent[];
  action: MovementAction;
};

export type MovementDataset = {
  version: typeof MOVEMENT_DATASET_VERSION;
  sampleCount: number;
  samples: MovementSample[];
};

export type ExtractMovementDatasetOptions = {
  /** Max preceding events retained as context for each action. Default 16. */
  contextWindow?: number;
};

type ManifestReplay = ReviewedExportManifest["replays"][number];
type ManifestReplayEvent = ManifestReplay["events"][number];

/**
 * Derive a next-action prediction dataset from a reviewed export manifest.
 *
 * Each `action` event in a replay timeline becomes one training sample whose
 * context is the window of preceding events (within the same session replay).
 * Prior actions are themselves added to the running context, so the model can
 * learn multi-step sequences.
 */
export function extractMovementDataset(
  manifest: ReviewedExportManifest,
  options: ExtractMovementDatasetOptions = {},
): MovementDataset {
  const contextWindow = Math.max(0, options.contextWindow ?? DEFAULT_CONTEXT_WINDOW);
  const samples: MovementSample[] = [];

  for (const replay of manifest.replays) {
    const orderedEvents = [...replay.events].sort((a, b) => a.ts - b.ts);
    const rollingContext: MovementContextEvent[] = [];
    const stepIndexByTrajectory = new Map<string, number>();

    for (const event of orderedEvents) {
      if (event.kind === "action") {
        const trajectoryId = event.trajectoryId;
        const stepIndex = stepIndexByTrajectory.get(trajectoryId) ?? 0;
        stepIndexByTrajectory.set(trajectoryId, stepIndex + 1);
        samples.push({
          trajectoryId,
          sessionId: replay.sessionId,
          stepIndex,
          context: rollingContext.slice(-contextWindow),
          action: { tool: event.tool, summary: event.summary },
        });
      }
      pushContext(rollingContext, toContextEvent(event), contextWindow);
    }
  }

  return { version: MOVEMENT_DATASET_VERSION, sampleCount: samples.length, samples };
}

function toContextEvent(event: ManifestReplayEvent): MovementContextEvent {
  switch (event.kind) {
    case "transcript":
      return { kind: "transcript", text: event.content, source: event.role };
    case "observation":
      return { kind: "observation", text: event.summary, source: event.source };
    case "action":
      return { kind: "action", text: event.summary, source: event.tool };
  }
}

function pushContext(target: MovementContextEvent[], event: MovementContextEvent, window: number): void {
  target.push(event);
  // window === 0 means "no context"; keep the array empty in that case.
  const limit = window === 0 ? 0 : window;
  while (target.length > limit) {
    target.shift();
  }
}

// ---------------------------------------------------------------------------
// Backend seam
// ---------------------------------------------------------------------------

export type TrainMovementModelOptions = {
  /** Opaque hyper-parameters forwarded to the backend. */
  hyperparameters?: Record<string, number | string | boolean>;
};

export type ActionPrediction = {
  tool: string;
  summary: string;
  /** Model confidence in [0, 1]. */
  confidence: number;
  /** True when the incoming context matched a training sample exactly. */
  exact: boolean;
  /** Trajectory the nearest training sample came from, when applicable. */
  neighborTrajectoryId?: string;
  /** Step index of the nearest training sample within its trajectory. */
  neighborStepIndex?: number;
};

export type SerializedMovementModel = {
  version: typeof SERIALIZED_MOVEMENT_MODEL_VERSION;
  backendId: string;
  sampleCount: number;
  payload: unknown;
};

export interface TrainedMovementModel {
  readonly backendId: string;
  /** Predict the next action for a context, or `undefined` for an empty model. */
  predict(context: MovementContextEvent[]): ActionPrediction | undefined;
  serialize(): SerializedMovementModel;
}

export interface LocalModelBackend {
  readonly id: string;
  train(dataset: MovementDataset, options?: TrainMovementModelOptions): Promise<TrainedMovementModel>;
  restore(serialized: SerializedMovementModel): TrainedMovementModel;
}

// ---------------------------------------------------------------------------
// Deterministic mock backend (nearest-neighbor over context token sets)
// ---------------------------------------------------------------------------

type MockSample = {
  trajectoryId: string;
  sessionId: string;
  stepIndex: number;
  tokens: string[];
  action: MovementAction;
};

type MockPayload = {
  samples: MockSample[];
};

/**
 * Deterministic, dependency-free backend used for cloud/CI validation and as a
 * fallback when no on-device model is available.
 *
 * Training memorizes every sample's context as a token multiset. Inference:
 *  - an identical context token set → exact replay (confidence 1).
 *  - otherwise → the highest Jaccard-similarity neighbor (confidence = the
 *    similarity), which is how it *generalizes* to new-but-related movements.
 *
 * All ties break deterministically (lower stepIndex, then trajectoryId), so the
 * same dataset always yields the same model and predictions — no randomness.
 */
export class MockNearestNeighborBackend implements LocalModelBackend {
  readonly id = "mock-nearest-neighbor";

  async train(dataset: MovementDataset): Promise<TrainedMovementModel> {
    const samples: MockSample[] = dataset.samples.map((sample) => ({
      trajectoryId: sample.trajectoryId,
      sessionId: sample.sessionId,
      stepIndex: sample.stepIndex,
      tokens: tokenizeContext(sample.context),
      action: { ...sample.action },
    }));
    return new MockTrainedModel(this.id, { samples });
  }

  restore(serialized: SerializedMovementModel): TrainedMovementModel {
    if (serialized.backendId !== this.id) {
      throw new Error(`backend ${this.id} cannot restore a model from backend ${serialized.backendId}`);
    }
    const payload = serialized.payload as MockPayload;
    return new MockTrainedModel(this.id, { samples: payload.samples.map((sample) => ({ ...sample })) });
  }
}

class MockTrainedModel implements TrainedMovementModel {
  constructor(
    readonly backendId: string,
    private readonly payload: MockPayload,
  ) {}

  predict(context: MovementContextEvent[]): ActionPrediction | undefined {
    if (this.payload.samples.length === 0) {
      return undefined;
    }
    const queryTokens = new Set(tokenizeContext(context));

    let best: MockSample | undefined;
    let bestScore = -1;
    let bestExact = false;
    for (const sample of this.payload.samples) {
      const sampleTokens = new Set(sample.tokens);
      const score = jaccard(queryTokens, sampleTokens);
      const exact = score === 1 || (queryTokens.size === 0 && sampleTokens.size === 0);
      if (isBetterCandidate(score, sample, bestScore, best)) {
        best = sample;
        bestScore = score;
        bestExact = exact;
      }
    }

    if (!best) {
      return undefined;
    }
    return {
      tool: best.action.tool,
      summary: best.action.summary,
      confidence: roundConfidence(Math.max(0, bestScore)),
      exact: bestExact,
      neighborTrajectoryId: best.trajectoryId,
      neighborStepIndex: best.stepIndex,
    };
  }

  serialize(): SerializedMovementModel {
    return {
      version: SERIALIZED_MOVEMENT_MODEL_VERSION,
      backendId: this.backendId,
      sampleCount: this.payload.samples.length,
      payload: { samples: this.payload.samples.map((sample) => ({ ...sample, tokens: [...sample.tokens] })) },
    };
  }
}

function isBetterCandidate(
  score: number,
  sample: MockSample,
  bestScore: number,
  best: MockSample | undefined,
): boolean {
  if (!best) {
    return true;
  }
  if (score !== bestScore) {
    return score > bestScore;
  }
  // Deterministic tie-break: prefer earlier step, then lexicographically smaller trajectory id.
  if (sample.stepIndex !== best.stepIndex) {
    return sample.stepIndex < best.stepIndex;
  }
  return sample.trajectoryId < best.trajectoryId;
}

// ---------------------------------------------------------------------------
// Tokenization + similarity
// ---------------------------------------------------------------------------

export function tokenizeContext(context: MovementContextEvent[]): string[] {
  const tokens: string[] = [];
  for (const event of context) {
    tokens.push(`kind:${event.kind}`);
    for (const token of tokenizeText(event.source)) {
      tokens.push(`src:${token}`);
    }
    for (const token of tokenizeText(event.text)) {
      tokens.push(token);
    }
  }
  return tokens;
}

function tokenizeText(value: string): string[] {
  return value
    .toLowerCase()
    .split(/[^a-z0-9]+/)
    .filter((token) => token.length > 0);
}

function jaccard(a: Set<string>, b: Set<string>): number {
  if (a.size === 0 && b.size === 0) {
    return 1;
  }
  if (a.size === 0 || b.size === 0) {
    return 0;
  }
  let intersection = 0;
  for (const token of a) {
    if (b.has(token)) {
      intersection += 1;
    }
  }
  const union = a.size + b.size - intersection;
  return union === 0 ? 0 : intersection / union;
}

function roundConfidence(value: number): number {
  return Math.round(value * 1_000_000) / 1_000_000;
}

// ---------------------------------------------------------------------------
// Backend registry (pluggability)
// ---------------------------------------------------------------------------

export class LocalModelBackendRegistry {
  private readonly backends = new Map<string, LocalModelBackend>();

  register(backend: LocalModelBackend): this {
    this.backends.set(backend.id, backend);
    return this;
  }

  get(id: string): LocalModelBackend | undefined {
    return this.backends.get(id);
  }

  require(id: string): LocalModelBackend {
    const backend = this.backends.get(id);
    if (!backend) {
      throw new Error(`no local-model backend registered with id "${id}"`);
    }
    return backend;
  }

  list(): LocalModelBackend[] {
    return [...this.backends.values()];
  }
}

/** Registry seeded with the deterministic mock backend (always safe in the cloud). */
export function createDefaultBackendRegistry(): LocalModelBackendRegistry {
  return new LocalModelBackendRegistry().register(new MockNearestNeighborBackend());
}

// ---------------------------------------------------------------------------
// Generalization eval harness
// ---------------------------------------------------------------------------

export type MovementEvalReport = {
  total: number;
  exactMatches: number;
  toolMatches: number;
  summaryMatches: number;
  exactMatchRate: number;
  toolAccuracy: number;
  summaryAccuracy: number;
  meanConfidence: number;
  predictions: Array<{
    trajectoryId: string;
    stepIndex: number;
    expectedTool: string;
    predictedTool?: string;
    toolCorrect: boolean;
    summaryCorrect: boolean;
    confidence: number;
    exact: boolean;
  }>;
};

/**
 * Measure how well a trained model reproduces the actions of a held-out set of
 * samples. Feed it samples the model was NOT trained on to measure
 * generalization; feed it the training samples to confirm exact replay.
 */
export function evaluateMovementModel(
  model: TrainedMovementModel,
  samples: MovementSample[],
): MovementEvalReport {
  let exactMatches = 0;
  let toolMatches = 0;
  let summaryMatches = 0;
  let confidenceSum = 0;
  const predictions: MovementEvalReport["predictions"] = [];

  for (const sample of samples) {
    const prediction = model.predict(sample.context);
    const toolCorrect = prediction?.tool === sample.action.tool;
    const summaryCorrect = prediction?.summary === sample.action.summary;
    if (toolCorrect) {
      toolMatches += 1;
    }
    if (summaryCorrect) {
      summaryMatches += 1;
    }
    if (prediction?.exact) {
      exactMatches += 1;
    }
    confidenceSum += prediction?.confidence ?? 0;
    predictions.push({
      trajectoryId: sample.trajectoryId,
      stepIndex: sample.stepIndex,
      expectedTool: sample.action.tool,
      predictedTool: prediction?.tool,
      toolCorrect,
      summaryCorrect,
      confidence: prediction?.confidence ?? 0,
      exact: prediction?.exact ?? false,
    });
  }

  const total = samples.length;
  return {
    total,
    exactMatches,
    toolMatches,
    summaryMatches,
    exactMatchRate: total === 0 ? 0 : exactMatches / total,
    toolAccuracy: total === 0 ? 0 : toolMatches / total,
    summaryAccuracy: total === 0 ? 0 : summaryMatches / total,
    meanConfidence: total === 0 ? 0 : confidenceSum / total,
    predictions,
  };
}

/**
 * Deterministically partition a dataset into train / holdout splits without any
 * randomness (every `holdoutEvery`-th sample goes to the holdout set). Useful
 * for a reproducible generalization eval in CI.
 */
export function splitMovementDataset(
  dataset: MovementDataset,
  holdoutEvery = 4,
): { train: MovementSample[]; holdout: MovementSample[] } {
  const train: MovementSample[] = [];
  const holdout: MovementSample[] = [];
  const stride = Math.max(2, Math.floor(holdoutEvery));
  dataset.samples.forEach((sample, index) => {
    if ((index + 1) % stride === 0) {
      holdout.push(sample);
    } else {
      train.push(sample);
    }
  });
  return { train, holdout };
}
