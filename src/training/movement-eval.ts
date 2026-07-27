import type { ReplayManifest, ReplayTimelineEvent } from "../capture/replay.js";
import {
  MOVEMENT_BOS,
  MOVEMENT_EOS,
  type MovementDataset,
  type MovementSequence,
  type MovementToken,
  type TrainedMovementModel,
} from "./movement-model.js";

/**
 * Synthetic event-stream generator + generalization eval harness.
 *
 * Lets the local-movement learning loop be validated with zero real OS access:
 * generate deterministic replay manifests from a small workflow grammar, train
 * on one split, and measure next-movement accuracy on a *held-out* split whose
 * orderings were never trained on. The `fallbackRate` isolates how much of that
 * accuracy came from the model generalizing (backing off to a shorter observed
 * context) rather than replaying a memorized full context.
 */

export type SyntheticMovementStep = {
  tool: string;
  summary: string;
};

export type SyntheticWorkflow = {
  id: string;
  /** Ordered step keys into the step catalog. */
  steps: string[];
};

export type BuildSyntheticMovementReplaysParams = {
  /** Catalog of atomic movements keyed by step id — the shared vocabulary. */
  steps: Record<string, SyntheticMovementStep>;
  workflows: SyntheticWorkflow[];
  sessionIdPrefix?: string;
  /** Deterministic base timestamp; incremented per event. */
  startTs?: number;
  stepIntervalMs?: number;
};

/**
 * Build deterministic replay manifests (one per workflow) from a workflow
 * grammar. No randomness or wall-clock — reproducible in cloud/CI.
 */
export function buildSyntheticMovementReplays(params: BuildSyntheticMovementReplaysParams): ReplayManifest[] {
  const sessionPrefix = params.sessionIdPrefix ?? "synthetic";
  const startTs = params.startTs ?? 1_700_000_000_000;
  const interval = params.stepIntervalMs ?? 1_000;

  return params.workflows.map((workflow, workflowIndex) => {
    const sessionId = `${sessionPrefix}-${workflow.id}`;
    const base = startTs + workflowIndex * 1_000_000;
    const events: ReplayTimelineEvent[] = workflow.steps.map((stepKey, stepIndex) => {
      const step = params.steps[stepKey];
      if (!step) {
        throw new Error(`unknown synthetic step "${stepKey}" in workflow "${workflow.id}"`);
      }
      return {
        kind: "action",
        ts: base + stepIndex * interval,
        trajectoryId: workflow.id,
        tool: step.tool,
        summary: step.summary,
      };
    });

    return {
      version: 1,
      sessionId,
      trajectoryIds: [workflow.id],
      eventCount: events.length,
      events,
    };
  });
}

export type MovementEvalResult = {
  sequencesEvaluated: number;
  /** Total next-token predictions attempted (skips leading BOS). */
  predictions: number;
  /** Exact next-token matches. */
  correct: number;
  nextTokenAccuracy: number;
  /** Fraction of correct predictions that required back-off — the generalization signal. */
  fallbackRate: number;
  /** Contexts for which the model produced no prediction at all. */
  unpredicted: number;
};

/**
 * Measure replay fidelity on held-out sequences: for each position, predict the
 * next token from its preceding context and compare to ground truth.
 */
export function evaluateMovementModel(
  model: TrainedMovementModel,
  heldOut: MovementSequence[] | MovementDataset,
): MovementEvalResult {
  const sequences = Array.isArray(heldOut) ? heldOut : heldOut.sequences;
  let predictions = 0;
  let correct = 0;
  let fallbackCorrect = 0;
  let unpredicted = 0;

  for (const sequence of sequences) {
    const tokens = sequence.tokens;
    for (let index = 1; index < tokens.length; index += 1) {
      const context = tokens.slice(0, index);
      const actual = tokens[index];
      predictions += 1;
      const prediction = model.predictNext(context);
      if (!prediction) {
        unpredicted += 1;
        continue;
      }
      if (prediction.token === actual) {
        correct += 1;
        if (prediction.fallback) {
          fallbackCorrect += 1;
        }
      }
    }
  }

  return {
    sequencesEvaluated: sequences.length,
    predictions,
    correct,
    nextTokenAccuracy: predictions > 0 ? correct / predictions : 0,
    fallbackRate: correct > 0 ? fallbackCorrect / correct : 0,
    unpredicted,
  };
}

/**
 * Roll the model out from BOS and report how faithfully it reproduces a target
 * movement sequence (exact match + longest common prefix length).
 */
export function scoreRolloutFidelity(
  model: TrainedMovementModel,
  target: MovementToken[],
  maxSteps = 64,
): { exactMatch: boolean; commonPrefix: number; expectedLength: number } {
  const expected = stripBoundaries(target);
  const generated = model.generate([MOVEMENT_BOS], maxSteps);
  let commonPrefix = 0;
  while (
    commonPrefix < expected.length &&
    commonPrefix < generated.length &&
    expected[commonPrefix] === generated[commonPrefix]
  ) {
    commonPrefix += 1;
  }
  return {
    exactMatch: commonPrefix === expected.length && generated.length === expected.length,
    commonPrefix,
    expectedLength: expected.length,
  };
}

function stripBoundaries(tokens: MovementToken[]): MovementToken[] {
  return tokens.filter((token) => token !== MOVEMENT_BOS && token !== MOVEMENT_EOS);
}
