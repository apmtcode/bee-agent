import type { ExportedReplayManifest, ReviewedExportManifest } from "./export-manifest.js";

/**
 * Pluggable local-model inference layer for the movement-learning subsystem.
 *
 * The rest of the subsystem covers capture -> reviewed dataset -> replay ->
 * training-plan generation. This module adds the missing inference/policy side:
 * loading a "trained" policy from a reviewed movement dataset and predicting a
 * movement sequence for a goal so bee-agent can (c) *repeat* a recorded
 * movement and (d) *generalize* to a new-but-related one.
 *
 * The backend is an interface so a real on-device small model (mlx / axolotl,
 * see runner.ts) can be dropped in later. The `DeterministicMockMovementBackend`
 * shipped here needs no OS access and no real model, so it validates the whole
 * capture -> dataset -> policy -> predict loop in the cloud / CI.
 */

/** A single movement step a policy predicts (mirrors a trajectory action). */
export type PredictedMovementStep = {
  tool: string;
  summary: string;
  /** 0..1 confidence for this individual step. */
  confidence: number;
  target?: string;
};

/** One labelled training example derived from a reviewed replay. */
export type MovementExample = {
  trajectoryId: string;
  /** Natural-language label of what the movement accomplishes. */
  goal: string;
  steps: PredictedMovementStep[];
};

export type MovementInferenceRequest = {
  /** Description of the desired (possibly new) movement. */
  goal: string;
  /** Optional recent observation summaries for context. */
  context?: string[];
  /** Cap on predicted steps (defaults to the matched example length). */
  maxSteps?: number;
};

export type MovementInferenceResult = {
  backendId: string;
  policyId: string;
  goal: string;
  steps: PredictedMovementStep[];
  /** How the prediction was produced. */
  strategy: "repeat" | "generalize" | "empty";
  /** true when no recorded movement matched closely and steps were adapted. */
  generalized: boolean;
  /** Similarity (0..1) of the goal to the best-matching recorded movement. */
  matchScore: number;
  matchedTrajectoryId?: string;
};

export type LoadMovementPolicyParams = {
  policyId: string;
  examples: MovementExample[];
  /**
   * Goal-similarity at/above which a recorded movement is repeated verbatim
   * rather than generalized. Default 0.6.
   */
  repeatThreshold?: number;
};

/** A loaded, ready-to-infer movement policy. */
export interface MovementPolicy {
  readonly id: string;
  readonly backendId: string;
  readonly exampleCount: number;
  predict(request: MovementInferenceRequest): MovementInferenceResult;
}

/** Pluggable backend that turns a reviewed dataset into an inferable policy. */
export interface MovementModelBackend {
  readonly id: string;
  /** Matching training runtime, if this backend pairs with one (see runner.ts). */
  readonly runtime?: string;
  loadPolicy(params: LoadMovementPolicyParams): Promise<MovementPolicy>;
}

const STOP_TOKENS = new Set([
  "a", "an", "the", "to", "on", "in", "of", "into", "onto", "for", "and",
  "with", "at", "active", "device",
]);

export function tokenizeGoal(text: string): string[] {
  return text
    .toLowerCase()
    .split(/[^a-z0-9]+/)
    .filter((token) => token.length > 0 && !STOP_TOKENS.has(token));
}

/** Jaccard similarity over meaningful goal tokens (deterministic, 0..1). */
export function goalSimilarity(a: string, b: string): number {
  const left = new Set(tokenizeGoal(a));
  const right = new Set(tokenizeGoal(b));
  if (left.size === 0 && right.size === 0) {
    return 1;
  }
  let intersection = 0;
  for (const token of left) {
    if (right.has(token)) {
      intersection += 1;
    }
  }
  const union = left.size + right.size - intersection;
  return union === 0 ? 0 : intersection / union;
}

/**
 * Build a movement dataset from a reviewed export's replays. Each trajectory's
 * `action` events become ordered steps; its `observation` summaries form the
 * goal label. Replays with no actions are skipped (nothing to repeat).
 */
export function buildMovementDataset(replays: ExportedReplayManifest[]): MovementExample[] {
  const examples: MovementExample[] = [];
  for (const replay of replays) {
    const byTrajectory = new Map<string, { observations: string[]; steps: PredictedMovementStep[] }>();
    const ordered = [...replay.events].sort((a, b) => a.ts - b.ts);
    for (const event of ordered) {
      if (event.kind === "transcript") {
        continue;
      }
      const bucket = byTrajectory.get(event.trajectoryId) ?? { observations: [], steps: [] };
      if (event.kind === "observation") {
        bucket.observations.push(event.summary);
      } else {
        bucket.steps.push({ tool: event.tool, summary: event.summary, confidence: 1 });
      }
      byTrajectory.set(event.trajectoryId, bucket);
    }
    for (const [trajectoryId, bucket] of byTrajectory) {
      if (bucket.steps.length === 0) {
        continue;
      }
      const goal = bucket.observations.length > 0 ? bucket.observations.join("; ") : bucket.steps.map((step) => step.summary).join("; ");
      examples.push({ trajectoryId, goal, steps: bucket.steps });
    }
  }
  return examples;
}

/** Convenience: dataset straight from a full reviewed export manifest. */
export function datasetFromReviewedExport(manifest: ReviewedExportManifest): MovementExample[] {
  return buildMovementDataset(manifest.replays);
}

/**
 * Deterministic, no-OS reference backend. It "learns" by indexing the dataset
 * and infers by nearest-neighbour goal matching:
 *  - match score >= repeatThreshold  -> repeat the recorded steps verbatim.
 *  - otherwise                       -> generalize by substituting the novel
 *    goal entity into the best match's step summaries/targets (transfer of a
 *    learned pattern to a related target, e.g. "open Safari" -> "open Chrome").
 * Fully deterministic given the same dataset + request, so it is testable.
 */
export class DeterministicMockMovementBackend implements MovementModelBackend {
  readonly id = "mock-deterministic";
  readonly runtime = "mock";

  async loadPolicy(params: LoadMovementPolicyParams): Promise<MovementPolicy> {
    const repeatThreshold = params.repeatThreshold ?? 0.6;
    const examples = params.examples;
    const backendId = this.id;

    return {
      id: params.policyId,
      backendId,
      exampleCount: examples.length,
      predict(request: MovementInferenceRequest): MovementInferenceResult {
        if (examples.length === 0) {
          return {
            backendId,
            policyId: params.policyId,
            goal: request.goal,
            steps: [],
            strategy: "empty",
            generalized: false,
            matchScore: 0,
          };
        }

        let best = examples[0]!;
        let bestScore = goalSimilarity(request.goal, best.goal);
        for (const example of examples.slice(1)) {
          const score = goalSimilarity(request.goal, example.goal);
          // Deterministic tie-break: prefer the higher score, else the
          // lexicographically smaller trajectoryId for stable output.
          if (score > bestScore || (score === bestScore && example.trajectoryId < best.trajectoryId)) {
            best = example;
            bestScore = score;
          }
        }

        const limit = request.maxSteps ?? best.steps.length;
        const generalized = bestScore < repeatThreshold;
        const rawSteps = best.steps.slice(0, Math.max(0, limit));

        if (!generalized) {
          return {
            backendId,
            policyId: params.policyId,
            goal: request.goal,
            steps: rawSteps.map((step) => ({ ...step })),
            strategy: "repeat",
            generalized: false,
            matchScore: bestScore,
            matchedTrajectoryId: best.trajectoryId,
          };
        }

        const substitution = deriveEntitySubstitution(request.goal, best.goal);
        const confidenceScale = 0.5 + bestScore / 2;
        const steps = rawSteps.map((step) => applySubstitution(step, substitution, confidenceScale));
        return {
          backendId,
          policyId: params.policyId,
          goal: request.goal,
          steps,
          strategy: "generalize",
          generalized: true,
          matchScore: bestScore,
          matchedTrajectoryId: best.trajectoryId,
        };
      },
    };
  }
}

type EntitySubstitution = { from?: string; to?: string };

/**
 * Identify the entity being swapped: the token unique to the matched example's
 * goal (the old target) and the token unique to the request goal (the new
 * target). Deterministic — first unique token on each side by original order.
 */
function deriveEntitySubstitution(requestGoal: string, exampleGoal: string): EntitySubstitution {
  const requestTokens = tokenizeGoal(requestGoal);
  const exampleTokens = tokenizeGoal(exampleGoal);
  const requestSet = new Set(requestTokens);
  const exampleSet = new Set(exampleTokens);
  const from = exampleTokens.find((token) => !requestSet.has(token));
  const to = requestTokens.find((token) => !exampleSet.has(token));
  return { from, to };
}

function applySubstitution(
  step: PredictedMovementStep,
  substitution: EntitySubstitution,
  confidenceScale: number,
): PredictedMovementStep {
  const next: PredictedMovementStep = {
    tool: step.tool,
    summary: substituteToken(step.summary, substitution),
    confidence: roundConfidence(step.confidence * confidenceScale),
  };
  if (step.target !== undefined) {
    next.target = substituteToken(step.target, substitution);
  }
  return next;
}

function substituteToken(text: string, substitution: EntitySubstitution): string {
  if (!substitution.from || !substitution.to) {
    return text;
  }
  // Case-insensitive, word-boundary replacement of the old entity token.
  const pattern = new RegExp(`\\b${escapeRegExp(substitution.from)}\\b`, "gi");
  return text.replace(pattern, substitution.to);
}

function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function roundConfidence(value: number): number {
  return Math.round(value * 1000) / 1000;
}
