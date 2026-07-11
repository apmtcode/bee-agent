import type { ReplayTimelineEvent } from "../capture/replay.js";
import type { TrajectorySpan } from "../capture/trajectory.js";

/**
 * Movement-model backend: the seam between a reviewed movement dataset and a
 * *trained policy* that can (a) repeat the recorded movements and (b) generalize
 * to new-but-related movements.
 *
 * The real on-device training/inference (objective 2c/2d) runs when bee-agent is
 * launched on the user's machine — see `LocalAppleSiliconTrainingRunner`, which
 * emits MLX/axolotl launch scripts. That path cannot run in the cloud/CI. This
 * module provides the *pluggable interface* plus a deterministic reference
 * backend (`DeterministicMovementBackend`) so the whole capture → dataset →
 * train → infer → generalize loop is exercisable with synthetic event streams,
 * and so a real small local model can be dropped in behind `MovementModelBackend`
 * later without touching call sites.
 */

const DEFAULT_WINDOW_SIZE = 3;
const NOOP_ACTION: MovementActionLabel = { tool: "noop", summary: "no learned movement" };

export type MovementActionLabel = {
  tool: string;
  summary: string;
  gesture?: string;
  target?: string;
  direction?: string;
};

export type MovementContext = {
  /** Application / surface the movement happens in (e.g. an appId or event source). */
  app: string;
  /** Latest observation summary the actor was reacting to. */
  observation: string;
  /** Ordered tokens of the most recent prior actions (oldest → newest). */
  recentActions: string[];
};

export type MovementExample = {
  context: MovementContext;
  action: MovementActionLabel;
};

export type MovementDataset = {
  version: 1;
  windowSize: number;
  examples: MovementExample[];
};

export type MovementTrainingConfig = {
  /** Drop learned context entries supported by fewer than this many examples. */
  minContextSupport?: number;
};

export type MovementModelEntry = {
  key: string;
  features: string[];
  support: number;
  actions: Array<{ key: string; label: MovementActionLabel; count: number }>;
};

export type TrainedMovementModel = {
  version: 1;
  backendId: string;
  windowSize: number;
  trainedExampleCount: number;
  entries: MovementModelEntry[];
};

export type MovementPrediction = {
  action: MovementActionLabel;
  /** 0..1 — how strongly the model favours this action for the context. */
  confidence: number;
  /** True when no exact context was learned and a nearest-neighbour was used. */
  generalized: boolean;
  matchedKey?: string;
};

export interface MovementModelBackend {
  readonly id: string;
  train(dataset: MovementDataset, config?: MovementTrainingConfig): Promise<TrainedMovementModel>;
  predict(model: TrainedMovementModel, context: MovementContext): MovementPrediction;
}

// ---------------------------------------------------------------------------
// Dataset construction (capture → dataset)
// ---------------------------------------------------------------------------

/**
 * Build a supervised movement dataset from reviewed trajectory spans. Each
 * `action` becomes one `(context → action)` example whose context is the running
 * app/observation plus a sliding window of the preceding actions.
 */
export function buildMovementDataset(
  trajectories: TrajectorySpan[],
  options: { windowSize?: number } = {},
): MovementDataset {
  const windowSize = Math.max(0, options.windowSize ?? DEFAULT_WINDOW_SIZE);
  const examples: MovementExample[] = [];

  for (const trajectory of trajectories) {
    const timeline = [
      ...trajectory.observations.map((observation) => ({ ts: observation.ts, kind: "observation" as const, observation })),
      ...trajectory.actions.map((action) => ({ ts: action.ts, kind: "action" as const, action })),
    ].sort((a, b) => (a.ts !== b.ts ? a.ts - b.ts : kindRank(a.kind) - kindRank(b.kind)));

    let app = "unknown";
    let observation = "";
    const recent: string[] = [];

    for (const entry of timeline) {
      if (entry.kind === "observation") {
        app = observationApp(entry.observation) ?? app;
        observation = entry.observation.summary;
        continue;
      }
      const label = actionLabelFromMetadata(entry.action.tool, entry.action.summary, entry.action.metadata);
      examples.push({
        context: { app, observation, recentActions: [...recent] },
        action: label,
      });
      recent.push(actionToken(label));
      if (windowSize > 0 && recent.length > windowSize) {
        recent.shift();
      }
    }
  }

  return { version: 1, windowSize, examples };
}

/**
 * Build a dataset from a flat replay timeline (e.g. an exported replay manifest).
 * Replay events carry only `tool` + `summary` for actions, so gesture structure
 * is not recovered here — use {@link buildMovementDataset} when trajectory
 * metadata is available.
 */
export function buildMovementDatasetFromReplay(
  events: ReplayTimelineEvent[],
  options: { windowSize?: number } = {},
): MovementDataset {
  const windowSize = Math.max(0, options.windowSize ?? DEFAULT_WINDOW_SIZE);
  const ordered = [...events].sort((a, b) => (a.ts !== b.ts ? a.ts - b.ts : kindRank(a.kind) - kindRank(b.kind)));
  const examples: MovementExample[] = [];

  let app = "unknown";
  let observation = "";
  const recent: string[] = [];

  for (const event of ordered) {
    if (event.kind === "observation") {
      app = event.source || app;
      observation = event.summary;
      continue;
    }
    if (event.kind !== "action") {
      continue;
    }
    const label: MovementActionLabel = { tool: event.tool, summary: event.summary };
    examples.push({ context: { app, observation, recentActions: [...recent] }, action: label });
    recent.push(actionToken(label));
    if (windowSize > 0 && recent.length > windowSize) {
      recent.shift();
    }
  }

  return { version: 1, windowSize, examples };
}

// ---------------------------------------------------------------------------
// Deterministic reference backend (train → infer → generalize)
// ---------------------------------------------------------------------------

/**
 * A dependency-free, fully deterministic movement policy. It learns a frequency
 * table `contextKey → action distribution`; inference returns the modal action
 * for an exactly-seen context, and otherwise falls back to the most feature-
 * similar learned context (Jaccard overlap) to generalize to new-but-related
 * movements. Ties are broken by first-seen order, so training and inference are
 * reproducible — a hard requirement for cloud/CI validation.
 */
export class DeterministicMovementBackend implements MovementModelBackend {
  readonly id = "deterministic-frequency";

  async train(dataset: MovementDataset, config: MovementTrainingConfig = {}): Promise<TrainedMovementModel> {
    const minSupport = Math.max(1, config.minContextSupport ?? 1);
    const order: string[] = [];
    const table = new Map<string, { features: string[]; support: number; actions: Map<string, { label: MovementActionLabel; count: number }> }>();

    for (const example of dataset.examples) {
      const key = contextKey(example.context);
      let entry = table.get(key);
      if (!entry) {
        entry = { features: contextFeatures(example.context), support: 0, actions: new Map() };
        table.set(key, entry);
        order.push(key);
      }
      entry.support += 1;
      const aKey = actionKey(example.action);
      const action = entry.actions.get(aKey);
      if (action) {
        action.count += 1;
      } else {
        entry.actions.set(aKey, { label: example.action, count: 1 });
      }
    }

    const entries: MovementModelEntry[] = [];
    for (const key of order) {
      const entry = table.get(key);
      if (!entry || entry.support < minSupport) {
        continue;
      }
      const actions = [...entry.actions.entries()]
        .map(([aKey, value]) => ({ key: aKey, label: value.label, count: value.count }))
        .sort((a, b) => b.count - a.count); // stable: preserves first-seen order on ties
      entries.push({ key, features: entry.features, support: entry.support, actions });
    }

    return {
      version: 1,
      backendId: this.id,
      windowSize: dataset.windowSize,
      trainedExampleCount: dataset.examples.length,
      entries,
    };
  }

  predict(model: TrainedMovementModel, context: MovementContext): MovementPrediction {
    const key = contextKey(context);
    const exact = model.entries.find((entry) => entry.key === key);
    if (exact && exact.actions.length > 0) {
      const top = exact.actions[0];
      const total = exact.actions.reduce((sum, action) => sum + action.count, 0);
      return { action: top.label, confidence: total > 0 ? top.count / total : 0, generalized: false, matchedKey: key };
    }

    const features = contextFeatures(context);
    let best: { entry: MovementModelEntry; score: number } | undefined;
    for (const entry of model.entries) {
      const score = jaccard(features, entry.features);
      if (score > 0 && (!best || score > best.score)) {
        best = { entry, score };
      }
    }

    if (best && best.entry.actions.length > 0) {
      const top = best.entry.actions[0];
      const total = best.entry.actions.reduce((sum, action) => sum + action.count, 0);
      const purity = total > 0 ? top.count / total : 0;
      return {
        action: top.label,
        confidence: best.score * purity,
        generalized: true,
        matchedKey: best.entry.key,
      };
    }

    return { action: NOOP_ACTION, confidence: 0, generalized: true };
  }
}

/**
 * Roll the policy forward from a seed context, feeding each predicted action back
 * into the recent-action window. Demonstrates the "repeat the recorded movements"
 * objective and, on unseen seeds, the generalization objective.
 */
export function rolloutMovements(
  backend: MovementModelBackend,
  model: TrainedMovementModel,
  seed: MovementContext,
  steps: number,
): MovementPrediction[] {
  const predictions: MovementPrediction[] = [];
  const recent = [...seed.recentActions];
  for (let step = 0; step < steps; step += 1) {
    const prediction = backend.predict(model, { app: seed.app, observation: seed.observation, recentActions: [...recent] });
    predictions.push(prediction);
    recent.push(actionToken(prediction.action));
    if (model.windowSize > 0 && recent.length > model.windowSize) {
      recent.shift();
    }
  }
  return predictions;
}

// ---------------------------------------------------------------------------
// Generalization eval harness
// ---------------------------------------------------------------------------

export type MovementEvalResult = {
  total: number;
  exactActionMatches: number;
  toolMatches: number;
  generalizedPredictions: number;
  generalizedCorrect: number;
  accuracy: number;
  toolAccuracy: number;
};

/**
 * Measure replay fidelity on a held-out dataset: for each example predict from
 * its context and compare against the true action. Reports overall accuracy plus
 * how well the model does specifically on generalized (nearest-neighbour)
 * predictions — the signal for objective 2(d).
 */
export function evaluateMovementModel(
  backend: MovementModelBackend,
  model: TrainedMovementModel,
  heldOut: MovementDataset,
): MovementEvalResult {
  let exact = 0;
  let toolMatches = 0;
  let generalized = 0;
  let generalizedCorrect = 0;

  for (const example of heldOut.examples) {
    const prediction = backend.predict(model, example.context);
    const isExact = actionKey(prediction.action) === actionKey(example.action);
    if (isExact) {
      exact += 1;
    }
    if (prediction.action.tool === example.action.tool) {
      toolMatches += 1;
    }
    if (prediction.generalized) {
      generalized += 1;
      if (isExact) {
        generalizedCorrect += 1;
      }
    }
  }

  const total = heldOut.examples.length;
  return {
    total,
    exactActionMatches: exact,
    toolMatches,
    generalizedPredictions: generalized,
    generalizedCorrect,
    accuracy: total > 0 ? exact / total : 0,
    toolAccuracy: total > 0 ? toolMatches / total : 0,
  };
}

// ---------------------------------------------------------------------------
// Pluggable backend registry
// ---------------------------------------------------------------------------

export type MovementBackendFactory = () => MovementModelBackend;

/**
 * The default backend registry. `deterministic-frequency` is the cloud-safe
 * reference implementation. A real on-device model (e.g. an MLX-trained small
 * policy) registers here under its own id — the seam the runner's launch scripts
 * ultimately feed.
 */
export function defaultMovementBackends(): Record<string, MovementBackendFactory> {
  return {
    "deterministic-frequency": () => new DeterministicMovementBackend(),
  };
}

export function resolveMovementBackend(
  id: string,
  backends: Record<string, MovementBackendFactory> = defaultMovementBackends(),
): MovementModelBackend {
  const factory = backends[id];
  if (!factory) {
    throw new Error(`Unknown movement model backend: ${id}`);
  }
  return factory();
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

function kindRank(kind: "observation" | "action" | "transcript"): number {
  return kind === "observation" ? 0 : kind === "transcript" ? 1 : 2;
}

function observationApp(observation: { source: string; metadata?: Record<string, unknown> }): string | undefined {
  const appName = observation.metadata?.["appName"];
  if (typeof appName === "string" && appName.trim().length > 0) {
    return appName.trim();
  }
  return observation.source || undefined;
}

function actionLabelFromMetadata(
  tool: string,
  summary: string,
  metadata: Record<string, unknown> | undefined,
): MovementActionLabel {
  const label: MovementActionLabel = { tool, summary };
  const gesture = metadata?.["gesture"];
  const target = metadata?.["target"];
  const direction = metadata?.["direction"];
  if (typeof gesture === "string") {
    label.gesture = gesture;
  }
  if (typeof target === "string") {
    label.target = target;
  }
  if (typeof direction === "string") {
    label.direction = direction;
  }
  return label;
}

function actionToken(label: MovementActionLabel): string {
  return actionKey(label);
}

function actionKey(label: MovementActionLabel): string {
  return [label.tool, label.gesture ?? "", label.direction ?? "", label.target ?? "", label.summary]
    .map((part) => part.trim().toLowerCase())
    .join("|");
}

function contextKey(context: MovementContext): string {
  return [normalize(context.app), normalize(context.observation), context.recentActions.map(normalize).join(">")].join("::");
}

function contextFeatures(context: MovementContext): string[] {
  const features = new Set<string>();
  for (const token of tokenize(context.app)) {
    features.add(`app:${token}`);
  }
  for (const token of tokenize(context.observation)) {
    features.add(`obs:${token}`);
  }
  for (const action of context.recentActions) {
    features.add(`act:${normalize(action)}`);
  }
  return [...features];
}

function tokenize(value: string): string[] {
  return normalize(value)
    .split(/[^a-z0-9]+/)
    .filter((token) => token.length > 0);
}

function normalize(value: string): string {
  return value.trim().toLowerCase();
}

function jaccard(a: string[], b: string[]): number {
  if (a.length === 0 && b.length === 0) {
    return 0;
  }
  const setA = new Set(a);
  const setB = new Set(b);
  let intersection = 0;
  for (const item of setA) {
    if (setB.has(item)) {
      intersection += 1;
    }
  }
  const union = setA.size + setB.size - intersection;
  return union === 0 ? 0 : intersection / union;
}
