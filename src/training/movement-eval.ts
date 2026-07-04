import type {
  MovementContext,
  MovementPolicyModel,
  MovementStep,
  MovementTrajectory,
} from "./movement-policy.js";
import type { DeviceGestureKind, DevicePlatform } from "../capture/device-adapter.js";

/**
 * Synthetic-data generation and a fidelity eval harness for the movement policy.
 *
 * Because the engine runs in the cloud with no OS access, we validate the
 * capture→dataset→policy→replay loop against deterministically-generated
 * "families" of related movements. The generalization eval measures whether a
 * policy fit on some members of a family can reproduce held-out members by
 * re-parameterizing — the core requirement of objective (d) "generalize to
 * perform new but related movements".
 */

export type SyntheticMovementSpec = {
  /** Prefix for the goal label; each target produces `${goalPrefix} ${target}`. */
  goalPrefix: string;
  appId: string;
  platform?: DevicePlatform;
  /** The parameter that varies across the family (each becomes one trajectory). */
  targets: string[];
  /** Gesture sequence to perform per target; the target/value fills placeholders. */
  gestures?: DeviceGestureKind[];
  /** Optional per-target value (e.g. text to type); defaults to `focus ${target}`. */
  valueFor?: (target: string) => string;
};

/**
 * Generate a family of related movement trajectories that differ only by their
 * varying target/value. Fully deterministic (index-based timestamps, no RNG).
 */
export function generateSyntheticMovementFamily(spec: SyntheticMovementSpec): MovementTrajectory[] {
  const gestures = spec.gestures ?? ["tap", "type"];
  const valueFor = spec.valueFor ?? ((target: string) => `focus ${target}`);

  return spec.targets.map((target, index) => {
    const steps: MovementStep[] = gestures.map((gesture, stepIndex) => ({
      gesture,
      appId: spec.appId,
      target,
      ts: index * 1000 + stepIndex,
      ...(gesture === "type" ? { valueSummary: valueFor(target) } : {}),
      ...(gesture === "swipe" || gesture === "scroll" ? { direction: "down" as const } : {}),
    }));

    return {
      id: `${spec.appId}:${slug(spec.goalPrefix)}:${slug(target)}`,
      goal: `${spec.goalPrefix} ${target}`,
      appId: spec.appId,
      ...(spec.platform ? { platform: spec.platform } : {}),
      steps,
    };
  });
}

export type MovementEvalCase = {
  context: MovementContext;
  expected: MovementStep[];
};

export type MovementEvalCaseResult = {
  matchedTrajectoryId: string | null;
  generalized: boolean;
  /** Fraction of expected steps reproduced in order, in [0,1]. */
  fidelity: number;
  exact: boolean;
};

export type MovementEvalReport = {
  caseCount: number;
  /** Mean per-case step fidelity in [0,1]. */
  meanFidelity: number;
  /** Fraction of cases reproduced exactly. */
  exactMatchRate: number;
  results: MovementEvalCaseResult[];
};

/** Run a policy over labelled cases and score how faithfully it reproduces them. */
export function evaluateMovementPolicy(
  model: MovementPolicyModel,
  cases: MovementEvalCase[],
): MovementEvalReport {
  const results = cases.map<MovementEvalCaseResult>((evalCase) => {
    const prediction = model.predict(evalCase.context);
    const fidelity = stepSequenceFidelity(prediction.steps, evalCase.expected);
    return {
      matchedTrajectoryId: prediction.matchedTrajectoryId,
      generalized: prediction.generalized,
      fidelity,
      exact: fidelity === 1 && prediction.steps.length === evalCase.expected.length,
    };
  });

  const caseCount = results.length;
  const meanFidelity =
    caseCount === 0 ? 0 : results.reduce((sum, result) => sum + result.fidelity, 0) / caseCount;
  const exactMatchRate =
    caseCount === 0 ? 0 : results.filter((result) => result.exact).length / caseCount;

  return { caseCount, meanFidelity, exactMatchRate, results };
}

/**
 * Held-out generalization eval: fit on some family members, then require the
 * policy to reproduce the *held-out* members purely by re-parameterizing the
 * varying target/value. Deterministic split by index (`holdOutEvery`).
 */
export function heldOutGeneralizationCases(
  family: MovementTrajectory[],
  options: { holdOutEvery?: number } = {},
): { train: MovementTrajectory[]; cases: MovementEvalCase[] } {
  const holdOutEvery = options.holdOutEvery ?? 2;
  const train: MovementTrajectory[] = [];
  const cases: MovementEvalCase[] = [];

  family.forEach((trajectory, index) => {
    if (holdOutEvery > 0 && index % holdOutEvery === 0 && family.length > 1) {
      const target = trajectory.steps.find((step) => step.target !== undefined)?.target;
      const valueSummary = trajectory.steps.find((step) => step.valueSummary !== undefined)?.valueSummary;
      const direction = trajectory.steps.find((step) => step.direction !== undefined)?.direction;
      cases.push({
        context: {
          goal: trajectory.goal,
          appId: trajectory.appId,
          parameters: {
            ...(target !== undefined ? { target } : {}),
            ...(valueSummary !== undefined ? { valueSummary } : {}),
            ...(direction !== undefined ? { direction } : {}),
          },
        },
        expected: trajectory.steps,
      });
    } else {
      train.push(trajectory);
    }
  });

  // Guarantee a non-empty training set even for tiny families.
  if (train.length === 0 && family.length > 0) {
    train.push(family[family.length - 1]);
  }

  return { train, cases };
}

/**
 * Ordered step-level fidelity: fraction of expected steps whose gesture and
 * parameters match the prediction at the same position.
 */
export function stepSequenceFidelity(predicted: MovementStep[], expected: MovementStep[]): number {
  if (expected.length === 0) {
    return predicted.length === 0 ? 1 : 0;
  }
  let matched = 0;
  for (let index = 0; index < expected.length; index += 1) {
    const predictedStep = predicted[index];
    if (predictedStep && stepsEqual(predictedStep, expected[index])) {
      matched += 1;
    }
  }
  return matched / expected.length;
}

function stepsEqual(a: MovementStep, b: MovementStep): boolean {
  return (
    a.gesture === b.gesture &&
    a.appId === b.appId &&
    a.target === b.target &&
    a.direction === b.direction &&
    a.valueSummary === b.valueSummary
  );
}

function slug(value: string): string {
  return value
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
}
