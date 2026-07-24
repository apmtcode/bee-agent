import type { TrajectorySpan } from "../capture/trajectory.js";

/**
 * Movement-model backend: the pluggable local-model seam for standing
 * objective #2 (post-train a local model on recorded movements and generalize
 * to new but related movements).
 *
 * This module defines the normalized policy schema (context -> action), a
 * dataset abstraction derived from recorded {@link TrajectorySpan}s, a backend
 * interface, and a deterministic in-process backend that runs entirely in the
 * cloud/CI (no OS input, no external model). A real on-device backend (e.g. an
 * MLX-served small model) implements the same {@link MovementModelBackend}
 * interface and is swapped in when bee-agent runs locally — see
 * {@link LocalAppleSiliconTrainingRunner} for the real-training launch path.
 */

// --------------------------------------------------------------------------
// Normalized movement schema (policy input/output)
// --------------------------------------------------------------------------

/** The observed state immediately before a movement action is taken. */
export type MovementContext = {
  appId: string;
  screenTitle?: string;
  goal?: string;
  recentTools?: string[];
};

/** A single movement the model can be asked to repeat or generalize. */
export type MovementAction = {
  tool: string;
  gesture?: string;
  target?: string;
  direction?: string;
  valueSummary?: string;
  summary: string;
};

export type MovementSample = {
  context: MovementContext;
  action: MovementAction;
};

export type MovementDataset = {
  version: 1;
  samples: MovementSample[];
};

export type MovementPredictionSource = "exact" | "generalized" | "fallback";

export type MovementPrediction = {
  action: MovementAction;
  /** 0..1 model confidence in the predicted action. */
  confidence: number;
  source: MovementPredictionSource;
  /** Context similarity used for the match (1 for exact, 0 for fallback). */
  similarity: number;
  matchedContextKey?: string;
};

export type MovementModelDescriptor = {
  backendId: string;
  sampleCount: number;
  contextCount: number;
  generalizationThreshold: number;
  contexts: { key: string; features: string[]; actionCount: number }[];
};

export interface TrainedMovementModel {
  readonly backendId: string;
  readonly sampleCount: number;
  readonly contextCount: number;
  predict(context: MovementContext): MovementPrediction | undefined;
  describe(): MovementModelDescriptor;
}

export type TrainMovementModelOptions = {
  /** Minimum context similarity (0..1) to accept a generalized prediction. */
  generalizationThreshold?: number;
};

export interface MovementModelBackend {
  readonly id: string;
  train(dataset: MovementDataset, options?: TrainMovementModelOptions): Promise<TrainedMovementModel>;
}

// --------------------------------------------------------------------------
// Dataset derivation from recorded trajectories
// --------------------------------------------------------------------------

/**
 * Pair every recorded action with the state observed just before it, producing
 * a supervised (context -> action) dataset. The context for an action is the
 * most recent preceding observation in the same trajectory; the trajectory's
 * outcome summary (if any) is carried as the goal signal.
 */
export function deriveMovementDataset(trajectories: readonly TrajectorySpan[]): MovementDataset {
  const samples: MovementSample[] = [];

  for (const trajectory of trajectories) {
    const observations = [...trajectory.observations].sort((a, b) => a.ts - b.ts);
    const goal = trajectory.outcome?.summary;
    const recentTools: string[] = [];

    for (const action of [...trajectory.actions].sort((a, b) => a.ts - b.ts)) {
      const priorObservation = latestBefore(observations, action.ts);
      const context: MovementContext = {
        appId: contextAppId(priorObservation),
        ...(contextScreen(priorObservation) ? { screenTitle: contextScreen(priorObservation) } : {}),
        ...(goal ? { goal } : {}),
        ...(recentTools.length ? { recentTools: [...recentTools] } : {}),
      };

      const metadata = action.metadata ?? {};
      samples.push({
        context,
        action: {
          tool: action.tool,
          ...(asString(metadata.gesture) ? { gesture: asString(metadata.gesture) } : {}),
          ...(asString(metadata.target) ? { target: asString(metadata.target) } : {}),
          ...(asString(metadata.direction) ? { direction: asString(metadata.direction) } : {}),
          ...(asString(metadata.valueSummary) ? { valueSummary: asString(metadata.valueSummary) } : {}),
          summary: action.summary,
        },
      });

      recentTools.push(action.tool);
      if (recentTools.length > 3) {
        recentTools.shift();
      }
    }
  }

  return { version: 1, samples };
}

function latestBefore<T extends { ts: number }>(items: readonly T[], ts: number): T | undefined {
  let match: T | undefined;
  for (const item of items) {
    if (item.ts <= ts) {
      match = item;
    } else {
      break;
    }
  }
  return match;
}

function contextAppId(observation: { metadata?: Record<string, unknown>; source?: string } | undefined): string {
  const metadata = observation?.metadata ?? {};
  return (
    asString(metadata.appId) ??
    asString(metadata.appName) ??
    observation?.source ??
    "unknown"
  );
}

function contextScreen(
  observation: { metadata?: Record<string, unknown> } | undefined,
): string | undefined {
  const metadata = observation?.metadata ?? {};
  return asString(metadata.screenTitle) ?? asString(metadata.windowTitle);
}

function asString(value: unknown): string | undefined {
  return typeof value === "string" && value.length > 0 ? value : undefined;
}

// --------------------------------------------------------------------------
// Deterministic in-process backend
// --------------------------------------------------------------------------

const APP_FEATURE_WEIGHT = 5;
const DEFAULT_GENERALIZATION_THRESHOLD = 0.34;

type ActionBucket = { action: MovementAction; count: number; actionKey: string };

type ContextEntry = {
  key: string;
  features: string[];
  featureSet: Set<string>;
  actions: Map<string, ActionBucket>;
  total: number;
};

/**
 * A frequency/similarity policy over the movement schema. It memorizes the
 * most common action per context (repeat known movements exactly) and, for an
 * unseen context, transfers the action from the most similar known context
 * (perform new but related movements). Fully deterministic — the same dataset
 * always yields the same model and predictions.
 */
export class InProcessMovementModelBackend implements MovementModelBackend {
  readonly id = "in-process-frequency-v1";

  async train(
    dataset: MovementDataset,
    options: TrainMovementModelOptions = {},
  ): Promise<TrainedMovementModel> {
    const threshold = clamp01(options.generalizationThreshold ?? DEFAULT_GENERALIZATION_THRESHOLD);
    const contexts = new Map<string, ContextEntry>();

    for (const sample of dataset.samples) {
      const key = contextKey(sample.context);
      let entry = contexts.get(key);
      if (!entry) {
        const features = contextFeatures(sample.context);
        entry = { key, features, featureSet: new Set(features), actions: new Map(), total: 0 };
        contexts.set(key, entry);
      }

      const actionKey = actionSignature(sample.action);
      const bucket = entry.actions.get(actionKey);
      if (bucket) {
        bucket.count += 1;
      } else {
        entry.actions.set(actionKey, { action: sample.action, count: 1, actionKey });
      }
      entry.total += 1;
    }

    return new FrequencyMovementModel(this.id, dataset.samples.length, [...contexts.values()], threshold);
  }
}

class FrequencyMovementModel implements TrainedMovementModel {
  private readonly globalTop: ActionBucket | undefined;
  private readonly globalTotal: number;

  constructor(
    readonly backendId: string,
    readonly sampleCount: number,
    private readonly contexts: ContextEntry[],
    private readonly threshold: number,
  ) {
    const totals = new Map<string, ActionBucket>();
    let globalTotal = 0;
    for (const entry of contexts) {
      for (const bucket of entry.actions.values()) {
        const existing = totals.get(bucket.actionKey);
        if (existing) {
          existing.count += bucket.count;
        } else {
          totals.set(bucket.actionKey, { ...bucket });
        }
        globalTotal += bucket.count;
      }
    }
    this.globalTotal = globalTotal;
    this.globalTop = pickTopBucket([...totals.values()]);
  }

  get contextCount(): number {
    return this.contexts.length;
  }

  predict(context: MovementContext): MovementPrediction | undefined {
    const key = contextKey(context);
    const exact = this.contexts.find((entry) => entry.key === key);
    if (exact) {
      const top = pickTopBucket([...exact.actions.values()]);
      if (top) {
        return {
          action: top.action,
          confidence: top.count / exact.total,
          source: "exact",
          similarity: 1,
          matchedContextKey: exact.key,
        };
      }
    }

    const queryFeatures = new Set(contextFeatures(context));
    let best: { entry: ContextEntry; similarity: number } | undefined;
    for (const entry of this.contexts) {
      if (entry.key === key) {
        continue;
      }
      const similarity = weightedJaccard(queryFeatures, entry.featureSet);
      if (!best || similarity > best.similarity || (similarity === best.similarity && entry.key < best.entry.key)) {
        best = { entry, similarity };
      }
    }

    if (best && best.similarity >= this.threshold) {
      const top = pickTopBucket([...best.entry.actions.values()]);
      if (top) {
        return {
          action: top.action,
          confidence: best.similarity * (top.count / best.entry.total),
          source: "generalized",
          similarity: best.similarity,
          matchedContextKey: best.entry.key,
        };
      }
    }

    if (this.globalTop && this.globalTotal > 0) {
      return {
        action: this.globalTop.action,
        confidence: this.globalTop.count / this.globalTotal,
        source: "fallback",
        similarity: 0,
      };
    }

    return undefined;
  }

  describe(): MovementModelDescriptor {
    return {
      backendId: this.backendId,
      sampleCount: this.sampleCount,
      contextCount: this.contexts.length,
      generalizationThreshold: this.threshold,
      contexts: this.contexts.map((entry) => ({
        key: entry.key,
        features: [...entry.features],
        actionCount: entry.actions.size,
      })),
    };
  }
}

function pickTopBucket(buckets: ActionBucket[]): ActionBucket | undefined {
  let top: ActionBucket | undefined;
  for (const bucket of buckets) {
    if (!top || bucket.count > top.count || (bucket.count === top.count && bucket.actionKey < top.actionKey)) {
      top = bucket;
    }
  }
  return top;
}

function contextKey(context: MovementContext): string {
  return JSON.stringify({
    appId: normalizeToken(context.appId),
    screenTitle: context.screenTitle ? normalizeToken(context.screenTitle) : "",
    goal: context.goal ? normalizeToken(context.goal) : "",
  });
}

function contextFeatures(context: MovementContext): string[] {
  const features: string[] = [`app:${normalizeToken(context.appId)}`];
  for (const token of tokenize(context.screenTitle)) {
    features.push(`screen:${token}`);
  }
  for (const token of tokenize(context.goal)) {
    features.push(`goal:${token}`);
  }
  for (const tool of context.recentTools ?? []) {
    features.push(`tool:${normalizeToken(tool)}`);
  }
  return [...new Set(features)].sort();
}

function actionSignature(action: MovementAction): string {
  return [action.tool, action.gesture ?? "", action.target ?? "", action.direction ?? ""]
    .map(normalizeToken)
    .join("|");
}

/**
 * Jaccard similarity with the `app:` feature weighted more heavily, so that
 * transferring a movement pattern across screens of the *same* app scores far
 * higher than across unrelated apps.
 */
function weightedJaccard(a: Set<string>, b: Set<string>): number {
  let intersectionWeight = 0;
  let unionWeight = 0;
  const seen = new Set<string>();
  for (const feature of a) {
    seen.add(feature);
    const weight = featureWeight(feature);
    unionWeight += weight;
    if (b.has(feature)) {
      intersectionWeight += weight;
    }
  }
  for (const feature of b) {
    if (seen.has(feature)) {
      continue;
    }
    unionWeight += featureWeight(feature);
  }
  return unionWeight === 0 ? 0 : intersectionWeight / unionWeight;
}

function featureWeight(feature: string): number {
  return feature.startsWith("app:") ? APP_FEATURE_WEIGHT : 1;
}

function tokenize(value: string | undefined): string[] {
  if (!value) {
    return [];
  }
  return [
    ...new Set(
      value
        .toLowerCase()
        .split(/[^a-z0-9]+/)
        .map((token) => token.trim())
        .filter((token) => token.length > 0),
    ),
  ];
}

function normalizeToken(value: string): string {
  return value.trim().toLowerCase();
}

function clamp01(value: number): number {
  if (Number.isNaN(value)) {
    return DEFAULT_GENERALIZATION_THRESHOLD;
  }
  return Math.min(1, Math.max(0, value));
}

// --------------------------------------------------------------------------
// Generalization eval harness
// --------------------------------------------------------------------------

export type MovementEvalResult = {
  total: number;
  predicted: number;
  toolMatches: number;
  gestureMatches: number;
  exact: number;
  generalized: number;
  fallback: number;
  /** Fraction of held-out samples whose predicted tool matched. */
  toolFidelity: number;
  /** Fraction whose predicted gesture matched (undefined treated as equal). */
  gestureFidelity: number;
};

/** Measure replay/generalization fidelity on held-out (context -> action) pairs. */
export function evaluateMovementModel(
  model: TrainedMovementModel,
  heldOut: readonly MovementSample[],
): MovementEvalResult {
  let predicted = 0;
  let toolMatches = 0;
  let gestureMatches = 0;
  let exact = 0;
  let generalized = 0;
  let fallback = 0;

  for (const sample of heldOut) {
    const prediction = model.predict(sample.context);
    if (!prediction) {
      continue;
    }
    predicted += 1;
    if (prediction.source === "exact") {
      exact += 1;
    } else if (prediction.source === "generalized") {
      generalized += 1;
    } else {
      fallback += 1;
    }
    if (normalizeToken(prediction.action.tool) === normalizeToken(sample.action.tool)) {
      toolMatches += 1;
    }
    if ((prediction.action.gesture ?? "") === (sample.action.gesture ?? "")) {
      gestureMatches += 1;
    }
  }

  const total = heldOut.length;
  return {
    total,
    predicted,
    toolMatches,
    gestureMatches,
    exact,
    generalized,
    fallback,
    toolFidelity: total === 0 ? 0 : toolMatches / total,
    gestureFidelity: total === 0 ? 0 : gestureMatches / total,
  };
}
