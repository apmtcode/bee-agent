import type { ExportedReplayManifest, ReviewedExportManifest } from "./export-manifest.js";

/**
 * Local-movement learning: model layer.
 *
 * This module closes objective #2(c)/(d) of the self-evolution charter — post-train
 * a local model to *repeat* recorded movements and *generalize* to new-but-related
 * ones — in a form that is fully deterministic and testable in the cloud.
 *
 * The design is deliberately backend-pluggable: {@link MovementModelBackend} is the
 * seam. `DeterministicMarkovBackend` is the reference (mock) backend — a small,
 * serializable, on-device-friendly model with context backoff for generalization.
 * A real small local model (e.g. an MLX/GGUF policy) can implement the same
 * interface later and drop straight into {@link MovementModelTrainer} without any
 * call-site changes.
 */

export type MovementActionLabel = {
  tool: string;
  summary: string;
};

/**
 * A single supervised movement example: the context leading up to an action, and
 * the action that followed. Derived from replayable timeline events, so the schema
 * stays aligned with the capture/replay pipeline.
 */
export type MovementStep = {
  /** Recent action tools that preceded this action, oldest-first (sliding window). */
  contextTools: string[];
  /** Most recent observation source (window/app/screen context), if any. */
  observationSource?: string;
  action: MovementActionLabel;
  trajectoryId: string;
  ts: number;
};

export type MovementDataset = {
  version: 1;
  contextWindow: number;
  steps: MovementStep[];
};

export type MovementModelBackendId = string;

export type TrainedMovementModel = {
  version: 1;
  backend: MovementModelBackendId;
  contextWindow: number;
  stepCount: number;
  /** Backend-specific serialized parameters. Opaque to callers; JSON-safe. */
  parameters: unknown;
};

export type MovementPredictionInput = {
  contextTools: string[];
  observationSource?: string;
};

/** How a prediction was resolved — the observable signal of generalization. */
export type MovementPredictionBackoff = "exact" | "partial" | "observation" | "prior" | "none";

export type MovementPrediction = {
  action: MovementActionLabel;
  /** Probability of the chosen action within the matched distribution (0..1). */
  confidence: number;
  /** Length of the context prefix that actually matched (0 = none/backed off fully). */
  matchedContextLength: number;
  backoff: MovementPredictionBackoff;
  candidates: Array<{ action: MovementActionLabel; probability: number }>;
};

export type MovementTrainOptions = {
  contextWindow?: number;
};

export interface MovementModelBackend {
  readonly id: MovementModelBackendId;
  train(dataset: MovementDataset, options?: MovementTrainOptions): TrainedMovementModel;
  predict(model: TrainedMovementModel, input: MovementPredictionInput): MovementPrediction | undefined;
}

export const DEFAULT_MOVEMENT_CONTEXT_WINDOW = 3;

/**
 * Build a movement dataset from replayable timeline events. Walks each replay in
 * timeline order, maintaining a sliding window of recent action tools and the most
 * recent observation source; every `action` event becomes a supervised step whose
 * context is the state *before* that action.
 */
export function buildMovementDataset(
  replays: ReadonlyArray<Pick<ExportedReplayManifest, "events">>,
  options: MovementTrainOptions = {},
): MovementDataset {
  const contextWindow = normalizeContextWindow(options.contextWindow);
  const steps: MovementStep[] = [];

  for (const replay of replays) {
    const events = [...replay.events].sort((a, b) => a.ts - b.ts);
    const recentTools: string[] = [];
    let observationSource: string | undefined;

    for (const event of events) {
      if (event.kind === "observation") {
        observationSource = event.source;
        continue;
      }
      if (event.kind !== "action") {
        continue;
      }
      steps.push({
        contextTools: recentTools.slice(-contextWindow),
        ...(observationSource !== undefined ? { observationSource } : {}),
        action: { tool: event.tool, summary: event.summary },
        trajectoryId: event.trajectoryId,
        ts: event.ts,
      });
      recentTools.push(event.tool);
      if (recentTools.length > contextWindow) {
        recentTools.shift();
      }
    }
  }

  return { version: 1, contextWindow, steps };
}

type FrequencyTable = Record<string, number>;

type MarkovParameters = {
  /** context-signature -> action-key -> count, for every context length 0..window. */
  contexts: Record<string, FrequencyTable>;
  /** observation-source -> action-key -> count. */
  observations: Record<string, FrequencyTable>;
  /** action-key -> count, the unconditioned prior. */
  prior: FrequencyTable;
  /** action-key -> label, so predictions can rehydrate the full action. */
  labels: Record<string, MovementActionLabel>;
};

/**
 * Deterministic n-gram (Markov) backend with context backoff.
 *
 * - `train` counts how often each action follows each context prefix (lengths
 *   0..contextWindow), plus an observation-conditioned table and a global prior.
 * - `predict` tries the longest context first (`exact` repeat of a learned
 *   movement), then progressively shorter prefixes (`partial` — this is the
 *   generalization path for new-but-related contexts), then the observation
 *   source alone, then the global prior. Ties break lexicographically, so results
 *   are fully reproducible across machines and runs.
 *
 * It is small, JSON-serializable, and on-device friendly — a legitimate reference
 * backend and the deterministic mock that keeps cloud/CI tests hermetic.
 */
export class DeterministicMarkovBackend implements MovementModelBackend {
  readonly id: MovementModelBackendId = "deterministic-markov";

  train(dataset: MovementDataset, options: MovementTrainOptions = {}): TrainedMovementModel {
    const contextWindow = normalizeContextWindow(options.contextWindow ?? dataset.contextWindow);
    const parameters: MarkovParameters = {
      contexts: {},
      observations: {},
      prior: {},
      labels: {},
    };

    for (const step of dataset.steps) {
      const actionKey = actionKeyOf(step.action);
      parameters.labels[actionKey] = { tool: step.action.tool, summary: step.action.summary };

      const tools = step.contextTools.slice(-contextWindow);
      for (let length = 0; length <= tools.length; length += 1) {
        const signature = contextSignature(tools.slice(tools.length - length));
        increment(tableFor(parameters.contexts, signature), actionKey);
      }
      if (step.observationSource !== undefined) {
        increment(tableFor(parameters.observations, step.observationSource), actionKey);
      }
      increment(parameters.prior, actionKey);
    }

    return {
      version: 1,
      backend: this.id,
      contextWindow,
      stepCount: dataset.steps.length,
      parameters,
    };
  }

  predict(model: TrainedMovementModel, input: MovementPredictionInput): MovementPrediction | undefined {
    if (model.backend !== this.id) {
      throw new Error(`DeterministicMarkovBackend cannot read model from backend "${model.backend}"`);
    }
    const parameters = model.parameters as MarkovParameters;
    const tools = input.contextTools.slice(-model.contextWindow);

    for (let length = tools.length; length >= 0; length -= 1) {
      const signature = contextSignature(tools.slice(tools.length - length));
      const table = parameters.contexts[signature];
      if (table && hasEntries(table)) {
        return resolvePrediction(parameters, table, length, length === tools.length ? "exact" : "partial");
      }
    }

    if (input.observationSource !== undefined) {
      const table = parameters.observations[input.observationSource];
      if (table && hasEntries(table)) {
        return resolvePrediction(parameters, table, 0, "observation");
      }
    }

    if (hasEntries(parameters.prior)) {
      return resolvePrediction(parameters, parameters.prior, 0, "prior");
    }

    return undefined;
  }
}

export type MovementModelTrainerOptions = {
  contextWindow?: number;
};

/**
 * Ties the pluggable backend to the reviewed-export dataset. Cloud runs use the
 * deterministic backend; a local run can pass a real on-device backend.
 */
export class MovementModelTrainer {
  constructor(
    private readonly backend: MovementModelBackend = new DeterministicMarkovBackend(),
    private readonly options: MovementModelTrainerOptions = {},
  ) {}

  buildDataset(
    replays: ReadonlyArray<Pick<ExportedReplayManifest, "events">>,
    options: MovementTrainOptions = {},
  ): MovementDataset {
    return buildMovementDataset(replays, { contextWindow: this.resolveContextWindow(options.contextWindow) });
  }

  train(
    replays: ReadonlyArray<Pick<ExportedReplayManifest, "events">>,
    options: MovementTrainOptions = {},
  ): TrainedMovementModel {
    const contextWindow = this.resolveContextWindow(options.contextWindow);
    const dataset = this.buildDataset(replays, { contextWindow });
    return this.backend.train(dataset, { contextWindow });
  }

  trainFromExport(manifest: ReviewedExportManifest, options: MovementTrainOptions = {}): TrainedMovementModel {
    return this.train(manifest.replays, options);
  }

  predict(model: TrainedMovementModel, input: MovementPredictionInput): MovementPrediction | undefined {
    return this.backend.predict(model, input);
  }

  private resolveContextWindow(override?: number): number {
    return normalizeContextWindow(override ?? this.options.contextWindow);
  }
}

export type MovementEvaluationResult = {
  total: number;
  correct: number;
  accuracy: number;
  /** Correct predictions that required backing off — i.e. genuine generalization. */
  generalizedCorrect: number;
  backoffBreakdown: Record<MovementPredictionBackoff, number>;
};

/**
 * Generalization eval harness: measure how often the model reproduces the correct
 * action on held-out steps, and how many of those correct calls came from a
 * backed-off (generalized) match rather than an exact context repeat.
 */
export function evaluateMovementModel(
  backend: MovementModelBackend,
  model: TrainedMovementModel,
  heldOut: ReadonlyArray<MovementStep>,
): MovementEvaluationResult {
  const backoffBreakdown: Record<MovementPredictionBackoff, number> = {
    exact: 0,
    partial: 0,
    observation: 0,
    prior: 0,
    none: 0,
  };
  let correct = 0;
  let generalizedCorrect = 0;

  for (const step of heldOut) {
    const prediction = backend.predict(model, {
      contextTools: step.contextTools,
      ...(step.observationSource !== undefined ? { observationSource: step.observationSource } : {}),
    });
    if (!prediction) {
      backoffBreakdown.none += 1;
      continue;
    }
    backoffBreakdown[prediction.backoff] += 1;
    if (actionKeyOf(prediction.action) === actionKeyOf(step.action)) {
      correct += 1;
      if (prediction.backoff !== "exact") {
        generalizedCorrect += 1;
      }
    }
  }

  const total = heldOut.length;
  return {
    total,
    correct,
    accuracy: total === 0 ? 0 : correct / total,
    generalizedCorrect,
    backoffBreakdown,
  };
}

function resolvePrediction(
  parameters: MarkovParameters,
  table: FrequencyTable,
  matchedContextLength: number,
  backoff: MovementPredictionBackoff,
): MovementPrediction {
  const totalCount = Object.values(table).reduce((sum, count) => sum + count, 0);
  const candidates = Object.entries(table)
    .map(([actionKey, count]) => ({
      action: parameters.labels[actionKey] ?? decodeActionKey(actionKey),
      probability: count / totalCount,
      actionKey,
    }))
    .sort((a, b) => {
      if (b.probability !== a.probability) {
        return b.probability - a.probability;
      }
      return a.actionKey < b.actionKey ? -1 : a.actionKey > b.actionKey ? 1 : 0;
    });

  const best = candidates[0];
  return {
    action: best.action,
    confidence: best.probability,
    matchedContextLength,
    backoff,
    candidates: candidates.map(({ action, probability }) => ({ action, probability })),
  };
}

function normalizeContextWindow(value: number | undefined): number {
  if (value === undefined || !Number.isFinite(value)) {
    return DEFAULT_MOVEMENT_CONTEXT_WINDOW;
  }
  return Math.max(0, Math.floor(value));
}

function contextSignature(tools: string[]): string {
  // JSON encoding makes ["ab"] and ["a","b"] distinct signatures and is escape-safe
  // for arbitrary tool strings.
  return JSON.stringify(tools);
}

function actionKeyOf(action: MovementActionLabel): string {
  return JSON.stringify([action.tool, action.summary]);
}

function decodeActionKey(actionKey: string): MovementActionLabel {
  try {
    const parsed = JSON.parse(actionKey) as unknown;
    if (Array.isArray(parsed) && typeof parsed[0] === "string" && typeof parsed[1] === "string") {
      return { tool: parsed[0], summary: parsed[1] };
    }
  } catch {
    // fall through to the raw-key fallback below
  }
  return { tool: actionKey, summary: "" };
}

function tableFor(container: Record<string, FrequencyTable>, key: string): FrequencyTable {
  const existing = container[key];
  if (existing) {
    return existing;
  }
  const created: FrequencyTable = {};
  container[key] = created;
  return created;
}

function increment(table: FrequencyTable, key: string): void {
  table[key] = (table[key] ?? 0) + 1;
}

function hasEntries(table: FrequencyTable): boolean {
  for (const key in table) {
    if (Object.prototype.hasOwnProperty.call(table, key)) {
      return true;
    }
  }
  return false;
}
