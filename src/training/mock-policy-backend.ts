import type {
  MovementContext,
  MovementPolicyBackend,
  MovementPolicyModel,
  MovementPrediction,
  MovementStep,
  MovementTrajectory,
} from "./movement-policy.js";

/**
 * Deterministic, dependency-free movement policy used for cloud/CI validation
 * and as the reference implementation of {@link MovementPolicyBackend}.
 *
 * Strategy: retrieve the recorded trajectory whose goal is most similar to the
 * requested goal (token Jaccard similarity, deterministic id tie-break), replay
 * its steps to *repeat* the movement, and substitute `context.parameters` into
 * the relevant fields to *generalize* to a new-but-related target/value.
 *
 * A real on-device small model implements the same interface; nothing else in
 * the subsystem changes when it is swapped in.
 */

export type MockMovementBackendOptions = {
  /** Backend id surfaced on predictions. Default `"mock-nearest-neighbor"`. */
  id?: string;
  /** Below this similarity a lookup returns an empty (no-op) prediction. Default `0`. */
  minSimilarity?: number;
};

export class MockMovementPolicyBackend implements MovementPolicyBackend {
  readonly id: string;
  private readonly minSimilarity: number;

  constructor(options: MockMovementBackendOptions = {}) {
    this.id = options.id ?? "mock-nearest-neighbor";
    this.minSimilarity = options.minSimilarity ?? 0;
  }

  fit(dataset: MovementTrajectory[]): MovementPolicyModel {
    return new MockMovementPolicyModel(this.id, dataset, this.minSimilarity);
  }
}

class MockMovementPolicyModel implements MovementPolicyModel {
  readonly trajectoryCount: number;
  private readonly candidates: readonly MovementTrajectory[];

  constructor(
    readonly backendId: string,
    dataset: MovementTrajectory[],
    private readonly minSimilarity: number,
  ) {
    // Sort by id so that, among equally-similar candidates, the smallest id
    // wins deterministically regardless of input ordering.
    this.candidates = [...dataset].sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0));
    this.trajectoryCount = this.candidates.length;
  }

  predict(context: MovementContext): MovementPrediction {
    const goalTokens = tokenize(context.goal);
    let best: MovementTrajectory | undefined;
    let bestScore = -1;

    for (const candidate of this.candidates) {
      if (context.appId && candidate.appId !== context.appId) {
        continue;
      }
      const score = jaccard(goalTokens, tokenize(candidate.goal));
      if (score > bestScore) {
        best = candidate;
        bestScore = score;
      }
    }

    if (!best || bestScore < this.minSimilarity) {
      return {
        steps: [],
        matchedTrajectoryId: null,
        confidence: 0,
        generalized: false,
        backendId: this.backendId,
      };
    }

    const { steps, generalized } = applyParameters(best.steps, context.parameters);
    return {
      steps,
      matchedTrajectoryId: best.id,
      confidence: roundConfidence(bestScore),
      generalized,
      backendId: this.backendId,
    };
  }
}

function applyParameters(
  steps: readonly MovementStep[],
  parameters: MovementContext["parameters"],
): { steps: MovementStep[]; generalized: boolean } {
  if (!parameters) {
    return { steps: steps.map((step) => ({ ...step })), generalized: false };
  }

  let generalized = false;
  const rewritten = steps.map((step) => {
    const next: MovementStep = { ...step };
    // Only substitute a field the step actually exercises, so we don't invent
    // gestures the recording never made.
    if (parameters.target !== undefined && step.target !== undefined && parameters.target !== step.target) {
      next.target = parameters.target;
      generalized = true;
    }
    if (
      parameters.valueSummary !== undefined &&
      (step.valueSummary !== undefined || step.gesture === "type") &&
      parameters.valueSummary !== step.valueSummary
    ) {
      next.valueSummary = parameters.valueSummary;
      generalized = true;
    }
    if (
      parameters.direction !== undefined &&
      step.direction !== undefined &&
      parameters.direction !== step.direction
    ) {
      next.direction = parameters.direction;
      generalized = true;
    }
    return next;
  });

  return { steps: rewritten, generalized };
}

export function tokenize(value: string): Set<string> {
  return new Set(
    value
      .toLowerCase()
      .split(/[^a-z0-9]+/)
      .filter((token) => token.length > 0),
  );
}

export function jaccard(a: Set<string>, b: Set<string>): number {
  if (a.size === 0 && b.size === 0) {
    return 1;
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

function roundConfidence(score: number): number {
  return Math.round(score * 1e6) / 1e6;
}
