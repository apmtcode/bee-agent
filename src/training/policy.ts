import type { TrajectorySpan } from "../capture/trajectory.js";
import type { ExportedReplayManifest, ReviewedExportManifest } from "./export-manifest.js";

/**
 * Movement-policy inference layer (standing objective #2c/#2d).
 *
 * The capture → schema → dataset → training-plan pipeline already produces a
 * reviewed dataset of movement trajectories. This module closes the loop: it
 * turns that dataset into a *policy* that can (c) reproduce recorded movements
 * for a known goal and (d) generalize to new-but-related goals by substituting
 * the changed entity into the closest recorded movement.
 *
 * The model backend is pluggable. A real on-device backend (e.g. an MLX-served
 * small model) implements {@link MovementPolicyBackend}; the deterministic
 * {@link RetrievalMovementBackend} below is the mock that runs in the cloud/CI
 * with no OS or model dependency, and is also a genuinely useful nearest-neighbour
 * baseline.
 */

/** A single predicted or recorded movement step. */
export type MovementStep = {
  /** The tool/effector that performs the movement (e.g. "device", "ui", "keyboard"). */
  tool: string;
  /** Human-readable description of the movement (e.g. "tapped Compose"). */
  summary: string;
  /** Optional parameterizable slot (the entity the movement acts on). */
  target?: string;
};

/** One training example: a goal/context and the movement sequence that satisfied it. */
export type MovementTrainingExample = {
  trajectoryId: string;
  /** Natural-language goal, derived from the trajectory's observations. */
  goal: string;
  /** Normalized context tokens used for similarity matching. */
  contextTokens: string[];
  /** The recorded movement sequence. */
  steps: MovementStep[];
  outcomeStatus?: "success" | "failure" | "aborted";
  reward?: number;
};

export type MovementPredictionRequest = {
  /** The goal to accomplish. */
  goal: string;
  /** Optional recent observation summaries that further condition the prediction. */
  recentSummaries?: string[];
  /** Cap the number of predicted steps. */
  maxSteps?: number;
};

export type MovementSubstitution = {
  from: string;
  to: string;
};

export type MovementPrediction = {
  /** Name of the backend that produced this prediction. */
  backend: string;
  /** The trajectory whose movements were used as the template, if any. */
  matchedTrajectoryId?: string;
  /** Similarity of the request goal to the matched example, in [0, 1]. */
  similarity: number;
  /** True when the prediction is a generalization (not a verbatim reproduction). */
  generalized: boolean;
  /** Entity substitutions applied when generalizing. */
  substitutions: MovementSubstitution[];
  /** The predicted movement sequence. */
  steps: MovementStep[];
};

/**
 * Pluggable movement-policy backend. Real backends fine-tune a local model in
 * {@link fit}; the mock indexes examples for retrieval. Both answer {@link predict}.
 */
export interface MovementPolicyBackend {
  readonly name: string;
  fit(examples: MovementTrainingExample[]): void | Promise<void>;
  predict(request: MovementPredictionRequest): MovementPrediction | Promise<MovementPrediction>;
}

const STOP_WORDS = new Set([
  "a",
  "an",
  "the",
  "to",
  "of",
  "in",
  "on",
  "at",
  "for",
  "and",
  "or",
  "with",
  "into",
  "onto",
  "from",
  "then",
  "this",
  "that",
  "it",
  "is",
  "was",
  "active",
]);

/** Lowercase, split on non-alphanumerics, drop stop-words and single chars. */
export function tokenizeMovementText(text: string): string[] {
  return text
    .toLowerCase()
    .split(/[^a-z0-9]+/)
    .filter((token) => token.length > 1 && !STOP_WORDS.has(token));
}

function uniqueTokens(tokens: string[]): string[] {
  const seen = new Set<string>();
  const out: string[] = [];
  for (const token of tokens) {
    if (!seen.has(token)) {
      seen.add(token);
      out.push(token);
    }
  }
  return out;
}

function jaccardSimilarity(a: Set<string>, b: Set<string>): number {
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

function replaceWholeWord(text: string, from: string, to: string): { text: string; changed: boolean } {
  const pattern = new RegExp(`\\b${escapeRegExp(from)}\\b`, "gi");
  let changed = false;
  const next = text.replace(pattern, () => {
    changed = true;
    return to;
  });
  return { text: next, changed };
}

function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

/**
 * Deterministic nearest-neighbour movement policy. Reproduces recorded movements
 * on an exact goal match, and generalizes to related goals by substituting the
 * changed entity token(s) into the closest matching trajectory's steps.
 */
export class RetrievalMovementBackend implements MovementPolicyBackend {
  readonly name = "retrieval-mock";
  private examples: MovementTrainingExample[] = [];

  fit(examples: MovementTrainingExample[]): void {
    // Prefer higher-reward, successful, movement-bearing examples when ties occur.
    this.examples = examples.filter((example) => example.steps.length > 0);
  }

  predict(request: MovementPredictionRequest): MovementPrediction {
    const queryTokens = uniqueTokens([
      ...tokenizeMovementText(request.goal),
      ...(request.recentSummaries ?? []).flatMap((summary) => tokenizeMovementText(summary)),
    ]);
    const querySet = new Set(queryTokens);

    if (this.examples.length === 0) {
      return { backend: this.name, similarity: 0, generalized: false, substitutions: [], steps: [] };
    }

    let best: { example: MovementTrainingExample; similarity: number } | undefined;
    for (const example of this.examples) {
      const similarity = jaccardSimilarity(querySet, new Set(example.contextTokens));
      if (best === undefined || outranks(similarity, example, best.similarity, best.example)) {
        best = { example, similarity };
      }
    }
    if (best === undefined) {
      return { backend: this.name, similarity: 0, generalized: false, substitutions: [], steps: [] };
    }

    const maxSteps = request.maxSteps ?? best.example.steps.length;
    const exampleTokens = best.example.contextTokens;
    const exampleSet = new Set(exampleTokens);

    // Exact goal match → reproduce recorded movements verbatim (objective 2c).
    if (best.similarity >= 1) {
      return {
        backend: this.name,
        matchedTrajectoryId: best.example.trajectoryId,
        similarity: best.similarity,
        generalized: false,
        substitutions: [],
        steps: best.example.steps.slice(0, maxSteps).map((step) => ({ ...step })),
      };
    }

    // Related goal → generalize by substituting the changed entity (objective 2d).
    const addedTokens = queryTokens.filter((token) => !exampleSet.has(token));
    const droppedTokens = exampleTokens.filter((token) => !querySet.has(token));
    const substitutions = buildSubstitutions(droppedTokens, addedTokens);

    const steps = best.example.steps.slice(0, maxSteps).map((step) => applySubstitutions(step, substitutions));
    const appliedSubstitutions = substitutions.filter((substitution) =>
      steps.some((step) => step.summary.includes(substitution.to) || step.target === substitution.to),
    );

    return {
      backend: this.name,
      matchedTrajectoryId: best.example.trajectoryId,
      similarity: best.similarity,
      generalized: true,
      substitutions: appliedSubstitutions,
      steps,
    };
  }
}

function outranks(
  similarity: number,
  example: MovementTrainingExample,
  bestSimilarity: number,
  bestExample: MovementTrainingExample,
): boolean {
  if (similarity !== bestSimilarity) {
    return similarity > bestSimilarity;
  }
  const reward = example.reward ?? 0;
  const bestReward = bestExample.reward ?? 0;
  if (reward !== bestReward) {
    return reward > bestReward;
  }
  const success = example.outcomeStatus === "success" ? 1 : 0;
  const bestSuccess = bestExample.outcomeStatus === "success" ? 1 : 0;
  return success > bestSuccess;
}

function buildSubstitutions(droppedTokens: string[], addedTokens: string[]): MovementSubstitution[] {
  if (addedTokens.length === 0 || droppedTokens.length === 0) {
    return [];
  }
  return droppedTokens.map((from, index) => ({ from, to: addedTokens[index] ?? addedTokens[0]! }));
}

function applySubstitutions(step: MovementStep, substitutions: MovementSubstitution[]): MovementStep {
  let summary = step.summary;
  let target = step.target;
  for (const substitution of substitutions) {
    summary = replaceWholeWord(summary, substitution.from, substitution.to).text;
    if (target !== undefined) {
      target = replaceWholeWord(target, substitution.from, substitution.to).text;
    }
  }
  return target === undefined ? { tool: step.tool, summary } : { tool: step.tool, summary, target };
}

/** Build movement training examples directly from recorded trajectory spans. */
export function buildMovementExamplesFromTrajectories(trajectories: TrajectorySpan[]): MovementTrainingExample[] {
  return trajectories.flatMap((trajectory) => {
    const observations = trajectory.review?.redactedObservations ?? trajectory.observations;
    const actions = trajectory.review?.redactedActions ?? trajectory.actions;
    if (actions.length === 0) {
      return [];
    }
    const goal = observations.map((observation) => observation.summary).join(" ").trim() || trajectory.id;
    const steps = [...actions]
      .sort((a, b) => a.ts - b.ts)
      .map((action) => {
        const metadata =
          "metadata" in action && action.metadata && typeof action.metadata === "object"
            ? (action.metadata as Record<string, unknown>)
            : undefined;
        return toStep(action.tool, action.summary, metadata);
      });
    return [
      {
        trajectoryId: trajectory.id,
        goal,
        contextTokens: uniqueTokens(tokenizeMovementText(goal)),
        steps,
        outcomeStatus: trajectory.outcome?.status,
        reward: trajectory.outcome?.reward,
      },
    ];
  });
}

/** Build movement training examples from an exported reviewed dataset manifest. */
export function buildMovementExamplesFromManifest(manifest: ReviewedExportManifest): MovementTrainingExample[] {
  const outcomeByTrajectory = new Map(
    manifest.trajectories.map((trajectory) => [trajectory.id, trajectory] as const),
  );
  return manifest.replays.flatMap((replay) => buildExamplesFromReplay(replay, outcomeByTrajectory));
}

function buildExamplesFromReplay(
  replay: ExportedReplayManifest,
  outcomeByTrajectory: Map<string, { outcomeStatus?: "success" | "failure" | "aborted"; reward?: number }>,
): MovementTrainingExample[] {
  const examples: MovementTrainingExample[] = [];
  for (const trajectoryId of replay.trajectoryIds) {
    const events = replay.events
      .filter((event) => "trajectoryId" in event && event.trajectoryId === trajectoryId)
      .sort((a, b) => a.ts - b.ts);
    const observations = events.filter(
      (event): event is Extract<typeof event, { kind: "observation" }> => event.kind === "observation",
    );
    const actions = events.filter(
      (event): event is Extract<typeof event, { kind: "action" }> => event.kind === "action",
    );
    if (actions.length === 0) {
      continue;
    }
    const goal = observations.map((event) => event.summary).join(" ").trim() || trajectoryId;
    const outcome = outcomeByTrajectory.get(trajectoryId);
    examples.push({
      trajectoryId,
      goal,
      contextTokens: uniqueTokens(tokenizeMovementText(goal)),
      steps: actions.map((event) => toStep(event.tool, event.summary)),
      outcomeStatus: outcome?.outcomeStatus,
      reward: outcome?.reward,
    });
  }
  return examples;
}

function toStep(tool: string, summary: string, metadata?: Record<string, unknown>): MovementStep {
  const target = metadata && typeof metadata.target === "string" ? metadata.target : undefined;
  return target === undefined ? { tool, summary } : { tool, summary, target };
}

export type MovementPolicyEvaluation = {
  /** Number of held-out examples scored. */
  count: number;
  /** Fraction of examples whose predicted tool sequence exactly matches. */
  exactSequenceMatch: number;
  /** Mean per-step tool accuracy across all examples. */
  toolAccuracy: number;
  /** Fraction of predictions that were generalizations rather than reproductions. */
  generalizedRate: number;
};

/**
 * Generalization eval harness: score a fitted backend against held-out examples.
 * Measures how faithfully predicted movements match the recorded ground truth.
 */
export async function evaluateMovementPolicy(
  backend: MovementPolicyBackend,
  heldOut: MovementTrainingExample[],
): Promise<MovementPolicyEvaluation> {
  if (heldOut.length === 0) {
    return { count: 0, exactSequenceMatch: 0, toolAccuracy: 0, generalizedRate: 0 };
  }
  let exactMatches = 0;
  let toolAccuracySum = 0;
  let generalizedCount = 0;
  for (const example of heldOut) {
    const prediction = await backend.predict({ goal: example.goal, maxSteps: example.steps.length });
    if (prediction.generalized) {
      generalizedCount += 1;
    }
    const predictedTools = prediction.steps.map((step) => step.tool);
    const expectedTools = example.steps.map((step) => step.tool);
    if (sequencesEqual(predictedTools, expectedTools)) {
      exactMatches += 1;
    }
    toolAccuracySum += perStepToolAccuracy(predictedTools, expectedTools);
  }
  return {
    count: heldOut.length,
    exactSequenceMatch: exactMatches / heldOut.length,
    toolAccuracy: toolAccuracySum / heldOut.length,
    generalizedRate: generalizedCount / heldOut.length,
  };
}

function sequencesEqual(a: string[], b: string[]): boolean {
  return a.length === b.length && a.every((value, index) => value === b[index]);
}

function perStepToolAccuracy(predicted: string[], expected: string[]): number {
  if (expected.length === 0) {
    return predicted.length === 0 ? 1 : 0;
  }
  let matches = 0;
  for (let index = 0; index < expected.length; index += 1) {
    if (predicted[index] === expected[index]) {
      matches += 1;
    }
  }
  return matches / expected.length;
}

/**
 * High-level engine tying a pluggable backend to the dataset sources. Fit from
 * raw trajectories or an exported manifest, then predict/rollout movements.
 */
export class MovementPolicyEngine {
  constructor(private readonly backend: MovementPolicyBackend = new RetrievalMovementBackend()) {}

  get backendName(): string {
    return this.backend.name;
  }

  async fitFromTrajectories(trajectories: TrajectorySpan[]): Promise<MovementTrainingExample[]> {
    const examples = buildMovementExamplesFromTrajectories(trajectories);
    await this.backend.fit(examples);
    return examples;
  }

  async fitFromManifest(manifest: ReviewedExportManifest): Promise<MovementTrainingExample[]> {
    const examples = buildMovementExamplesFromManifest(manifest);
    await this.backend.fit(examples);
    return examples;
  }

  async fit(examples: MovementTrainingExample[]): Promise<void> {
    await this.backend.fit(examples);
  }

  async predict(request: MovementPredictionRequest): Promise<MovementPrediction> {
    return await this.backend.predict(request);
  }

  /** Predict the movement sequence for a goal (alias for predict with a goal string). */
  async rollout(goal: string, maxSteps?: number): Promise<MovementPrediction> {
    return await this.backend.predict(maxSteps === undefined ? { goal } : { goal, maxSteps });
  }
}
