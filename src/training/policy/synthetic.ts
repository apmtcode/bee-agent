/**
 * Deterministic synthetic movement-trajectory generator + generalization eval
 * harness. Lets the train -> predict -> generalize loop be validated in the cloud
 * with no real OS input (standing objective #2: "use synthetic/simulated event
 * streams to validate your code"). No RNG, no clock — flows are pure functions of
 * their parameters so tests are reproducible.
 */

import { buildTrajectorySpan, type TrajectorySpan } from "../../capture/trajectory.js";
import { buildMovementDataset, labelAction } from "./dataset.js";
import { actionKey, type MovementModelBackend, type MovementPredictionMethod } from "./model.js";
import type { NgramModelState } from "./ngram-backend.js";
import type { TrainedMovementModel } from "./model.js";

/** One step of a parametric flow. `{v}` in any string is replaced by the variant. */
export type SyntheticFlowStep = {
  /** Observation summary the agent sees before acting. */
  observe: string;
  /** Optional observation metadata (values may contain `{v}`). */
  observeMeta?: Record<string, string>;
  /** The action taken in response. */
  act: {
    tool: string;
    gesture?: string;
    target?: string;
    direction?: string;
    summary?: string;
  };
};

export type SyntheticFlow = {
  name: string;
  steps: SyntheticFlowStep[];
};

export type GenerateSyntheticParams = {
  /** Variant tokens substituted for `{v}` (e.g. ["compose", "reply", "forward"]). */
  variants: string[];
  sessionPrefix?: string;
  /** Base timestamp; each event advances by `stepMs`. */
  baseTs?: number;
  stepMs?: number;
  /** Optional outcome reward attached to every generated trajectory. */
  reward?: number;
};

function substitute(text: string, variant: string): string {
  return text.replaceAll("{v}", variant);
}

/**
 * Generate one trajectory per variant by substituting `{v}`. Trajectories share
 * structure but differ in a token, so holding out a variant tests generalization.
 */
export function generateSyntheticTrajectories(
  flow: SyntheticFlow,
  params: GenerateSyntheticParams,
): TrajectorySpan[] {
  const prefix = params.sessionPrefix ?? flow.name;
  const baseTs = params.baseTs ?? 1_700_000_000_000;
  const stepMs = params.stepMs ?? 1_000;

  return params.variants.map((variant, variantIndex) => {
    let ts = baseTs + variantIndex * flow.steps.length * stepMs * 2;
    const observations = [] as TrajectorySpan["observations"];
    const actions = [] as TrajectorySpan["actions"];

    for (const step of flow.steps) {
      const observationMeta: Record<string, string> = {};
      for (const [key, value] of Object.entries(step.observeMeta ?? {})) {
        observationMeta[key] = substitute(value, variant);
      }
      observations.push({
        kind: "observation",
        source: observationMeta.source ?? "synthetic",
        summary: substitute(step.observe, variant),
        ts,
        metadata: observationMeta,
      });
      ts += stepMs;

      const gesture = step.act.gesture;
      const target = step.act.target ? substitute(step.act.target, variant) : undefined;
      const direction = step.act.direction;
      actions.push({
        kind: "action",
        tool: step.act.tool,
        summary: step.act.summary ? substitute(step.act.summary, variant) : `${gesture ?? "act"} ${target ?? ""}`.trim(),
        ts,
        metadata: {
          ...(gesture ? { gesture } : {}),
          ...(target ? { target } : {}),
          ...(direction ? { direction } : {}),
        },
      });
      ts += stepMs;
    }

    const span = buildTrajectorySpan({
      id: `${prefix}-${variant}`,
      sessionId: `${prefix}-session`,
      captureTier: "full",
      observations,
      actions,
      ...(params.reward !== undefined
        ? { outcome: { status: "success", summary: `completed ${flow.name}`, reward: params.reward } }
        : {}),
    });
    // buildTrajectorySpan stamps createdAt from the wall clock; overwrite it with a
    // value derived from the flow's timestamps so generated trajectories are fully
    // deterministic (equal across calls) for reproducible tests and datasets.
    return { ...span, createdAt: new Date(baseTs + variantIndex).toISOString() };
  });
}

/** A canonical multi-step flow useful across tests and demos. */
export function mailComposeFlow(): SyntheticFlow {
  return {
    name: "mail-{v}",
    steps: [
      // Step 1's observation is variant-distinct (you see the affordance before
      // tapping it), so exact recall is unambiguous per variant.
      { observe: "mail inbox {v} option focused", observeMeta: { source: "os", event: "focus-changed", appName: "mail", target: "{v}" }, act: { tool: "device", gesture: "tap", target: "{v}" } },
      { observe: "{v} editor open", observeMeta: { source: "device", appName: "mail", screenTitle: "{v}" }, act: { tool: "device", gesture: "type", target: "body" } },
      // Step 3's action is shared across every variant — the generalization target.
      { observe: "draft ready in {v}", observeMeta: { source: "device", appName: "mail" }, act: { tool: "device", gesture: "tap", target: "send" } },
    ],
  };
}

export type MovementEvalMethodBreakdown = Record<MovementPredictionMethod, { total: number; correct: number }>;

export type MovementEvalResult = {
  total: number;
  correct: number;
  accuracy: number;
  byMethod: MovementEvalMethodBreakdown;
};

/**
 * Measure next-action prediction accuracy on held-out trajectories, broken down by
 * how each prediction was made (exact vs generalized vs prior). The generalized
 * bucket is the signal that the model performs new-but-related movements.
 */
export function evaluateMovementModel(
  backend: MovementModelBackend<NgramModelState>,
  model: TrainedMovementModel<NgramModelState>,
  heldOut: TrajectorySpan[],
): MovementEvalResult {
  const dataset = buildMovementDataset(heldOut);
  const byMethod: MovementEvalMethodBreakdown = {
    exact: { total: 0, correct: 0 },
    generalized: { total: 0, correct: 0 },
    prior: { total: 0, correct: 0 },
    none: { total: 0, correct: 0 },
  };
  let correct = 0;

  for (const example of dataset.examples) {
    const prediction = backend.predict(model, example.context);
    const isCorrect = prediction.action !== undefined && actionKey(prediction.action) === actionKey(example.action);
    byMethod[prediction.method].total += 1;
    if (isCorrect) {
      byMethod[prediction.method].correct += 1;
      correct += 1;
    }
  }

  const total = dataset.examples.length;
  return {
    total,
    correct,
    accuracy: total > 0 ? correct / total : 0,
    byMethod,
  };
}

// Re-exported for convenience so `labelAction` is reachable from the policy barrel
// path used by callers building custom eval sets.
export { labelAction };
