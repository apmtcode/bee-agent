import type { TrajectorySpan } from "../capture/trajectory.js";

/**
 * Movement-policy inference subsystem.
 *
 * This is the "repeat + generalize" half of the local-movement learning
 * objective. The capture/training pipeline records reviewed trajectories and
 * emits launch plans for an external on-device trainer (mlx / axolotl). This
 * module provides the complementary *inference* surface: a pluggable policy
 * backend that learns an observation -> next-action mapping from reviewed
 * trajectories, replays recorded movements when it sees a familiar
 * observation, and generalizes to novel-but-related observations via feature
 * similarity.
 *
 * The default backend is a deterministic nearest-neighbour model over
 * bag-of-token observation features. It needs no external model weights, so it
 * runs identically in the cloud (tests / CI) and on-device. Real on-device
 * models plug in behind the same {@link MovementPolicyBackend} interface, and
 * models serialize to {@link MovementPolicyModelSnapshot} so a trained policy
 * is portable and replayable.
 */

export type MovementObservationInput = {
  source: string;
  summary: string;
  metadata?: Record<string, unknown>;
};

export type MovementActionLabel = {
  tool: string;
  summary: string;
};

export type PolicyTrainingExample = {
  observation: MovementObservationInput;
  action: MovementActionLabel;
  trajectoryId?: string;
};

export type MovementPredictionSource = "exact" | "generalized" | "fallback";

export type MovementPolicyPrediction = {
  action: MovementActionLabel;
  /** Cosine similarity to the matched training observation, in [0, 1]. */
  confidence: number;
  /** Index into the model's stored entries, or -1 for a fallback prediction. */
  matchedEntryIndex: number;
  matchedTrajectoryId?: string;
  source: MovementPredictionSource;
};

export type MovementPolicyModelSnapshot = {
  version: 1;
  backend: string;
  fallbackAction: MovementActionLabel;
  entries: Array<{
    vector: Record<string, number>;
    observation: MovementObservationInput;
    action: MovementActionLabel;
    trajectoryId?: string;
  }>;
};

export interface MovementPolicyModel {
  readonly backend: string;
  readonly exampleCount: number;
  predict(observation: MovementObservationInput): MovementPolicyPrediction;
  toJSON(): MovementPolicyModelSnapshot;
}

export interface MovementPolicyBackend {
  readonly name: string;
  train(examples: PolicyTrainingExample[]): MovementPolicyModel;
  restore(snapshot: MovementPolicyModelSnapshot): MovementPolicyModel;
}

/**
 * Similarities at or above this threshold count as a replay of a recorded
 * movement ("exact"); anything strictly above zero but below it is a
 * generalization ("generalized"); zero overlap falls back to the modal action.
 */
export const EXACT_MATCH_THRESHOLD = 0.9999;

const NO_ACTION: MovementActionLabel = { tool: "noop", summary: "no action available" };

function tokenize(text: string): string[] {
  return text
    .toLowerCase()
    .split(/[^a-z0-9]+/u)
    .filter((token) => token.length > 0);
}

/** Bag-of-token features for an observation, L2-normalized to a unit vector. */
export function observationFeatureVector(observation: MovementObservationInput): Record<string, number> {
  const counts = new Map<string, number>();
  for (const token of tokenize(`${observation.source} ${observation.summary}`)) {
    counts.set(token, (counts.get(token) ?? 0) + 1);
  }
  let norm = 0;
  for (const value of counts.values()) {
    norm += value * value;
  }
  const magnitude = Math.sqrt(norm);
  const vector: Record<string, number> = {};
  if (magnitude === 0) {
    return vector;
  }
  for (const [token, value] of counts) {
    vector[token] = value / magnitude;
  }
  return vector;
}

/** Dot product of two L2-normalized sparse vectors == cosine similarity. */
function cosineSimilarity(a: Record<string, number>, b: Record<string, number>): number {
  const [small, large] = Object.keys(a).length <= Object.keys(b).length ? [a, b] : [b, a];
  let dot = 0;
  for (const [token, value] of Object.entries(small)) {
    const other = large[token];
    if (other !== undefined) {
      dot += value * other;
    }
  }
  if (dot < 0) {
    return 0;
  }
  return dot > 1 ? 1 : dot;
}

function modalAction(examples: PolicyTrainingExample[]): MovementActionLabel {
  const counts = new Map<string, { action: MovementActionLabel; count: number }>();
  for (const example of examples) {
    const existing = counts.get(example.action.tool);
    if (existing) {
      existing.count += 1;
    } else {
      counts.set(example.action.tool, { action: example.action, count: 1 });
    }
  }
  let best: { action: MovementActionLabel; count: number } | undefined;
  for (const entry of counts.values()) {
    if (!best || entry.count > best.count) {
      best = entry;
    }
  }
  return best?.action ?? NO_ACTION;
}

class NearestNeighborPolicyModel implements MovementPolicyModel {
  constructor(
    readonly backend: string,
    private readonly entries: MovementPolicyModelSnapshot["entries"],
    private readonly fallbackAction: MovementActionLabel,
  ) {}

  get exampleCount(): number {
    return this.entries.length;
  }

  predict(observation: MovementObservationInput): MovementPolicyPrediction {
    const query = observationFeatureVector(observation);
    let bestIndex = -1;
    let bestScore = 0;
    for (let index = 0; index < this.entries.length; index += 1) {
      const score = cosineSimilarity(query, this.entries[index]!.vector);
      if (score > bestScore) {
        bestScore = score;
        bestIndex = index;
      }
    }

    if (bestIndex === -1 || bestScore <= 0) {
      return {
        action: this.fallbackAction,
        confidence: 0,
        matchedEntryIndex: -1,
        source: "fallback",
      };
    }

    const entry = this.entries[bestIndex]!;
    return {
      action: entry.action,
      confidence: bestScore,
      matchedEntryIndex: bestIndex,
      ...(entry.trajectoryId ? { matchedTrajectoryId: entry.trajectoryId } : {}),
      source: bestScore >= EXACT_MATCH_THRESHOLD ? "exact" : "generalized",
    };
  }

  toJSON(): MovementPolicyModelSnapshot {
    return {
      version: 1,
      backend: this.backend,
      fallbackAction: this.fallbackAction,
      entries: this.entries.map((entry) => ({
        vector: { ...entry.vector },
        observation: { ...entry.observation },
        action: { ...entry.action },
        ...(entry.trajectoryId ? { trajectoryId: entry.trajectoryId } : {}),
      })),
    };
  }
}

/**
 * Deterministic local backend. Retrieves the most similar recorded observation
 * and repeats its action; generalizes across novel-but-related observations via
 * shared tokens. No external weights, so it behaves identically in cloud tests
 * and on-device.
 */
export class NearestNeighborPolicyBackend implements MovementPolicyBackend {
  readonly name = "nearest-neighbor";

  train(examples: PolicyTrainingExample[]): MovementPolicyModel {
    const entries = examples.map((example) => ({
      vector: observationFeatureVector(example.observation),
      observation: { ...example.observation },
      action: { ...example.action },
      ...(example.trajectoryId ? { trajectoryId: example.trajectoryId } : {}),
    }));
    return new NearestNeighborPolicyModel(this.name, entries, modalAction(examples));
  }

  restore(snapshot: MovementPolicyModelSnapshot): MovementPolicyModel {
    return new NearestNeighborPolicyModel(snapshot.backend, snapshot.entries, snapshot.fallbackAction);
  }
}

function reviewedObservations(trajectory: TrajectorySpan): MovementObservationInput[] {
  if (trajectory.review?.redactedObservations) {
    return trajectory.review.redactedObservations.map((observation) => ({
      source: observation.source,
      summary: observation.summary,
    }));
  }
  return trajectory.observations.map((observation) => ({
    source: observation.source,
    summary: observation.summary,
    ...(observation.metadata ? { metadata: observation.metadata } : {}),
  }));
}

type TimedObservation = MovementObservationInput & { ts: number };
type TimedAction = MovementActionLabel & { ts: number };

function reviewedTimedObservations(trajectory: TrajectorySpan): TimedObservation[] {
  if (trajectory.review?.redactedObservations) {
    return trajectory.review.redactedObservations.map((observation) => ({
      ts: observation.ts,
      source: observation.source,
      summary: observation.summary,
    }));
  }
  return trajectory.observations.map((observation) => ({
    ts: observation.ts,
    source: observation.source,
    summary: observation.summary,
    ...(observation.metadata ? { metadata: observation.metadata } : {}),
  }));
}

function reviewedTimedActions(trajectory: TrajectorySpan): TimedAction[] {
  if (trajectory.review?.redactedActions) {
    return trajectory.review.redactedActions.map((action) => ({
      ts: action.ts,
      tool: action.tool,
      summary: action.summary,
    }));
  }
  return trajectory.actions.map((action) => ({
    ts: action.ts,
    tool: action.tool,
    summary: action.summary,
  }));
}

/**
 * Pair each action with the most recent observation at or before it (the first
 * observation if none precede). This is the movement-prediction training signal:
 * "given what was observed, take this next action".
 */
export function buildPolicyExamplesFromTrajectory(trajectory: TrajectorySpan): PolicyTrainingExample[] {
  const observations = reviewedTimedObservations(trajectory).sort((a, b) => a.ts - b.ts);
  const actions = reviewedTimedActions(trajectory).sort((a, b) => a.ts - b.ts);
  if (observations.length === 0 || actions.length === 0) {
    return [];
  }

  return actions.map((action) => {
    let context = observations[0]!;
    for (const observation of observations) {
      if (observation.ts <= action.ts) {
        context = observation;
      } else {
        break;
      }
    }
    return {
      observation: { source: context.source, summary: context.summary },
      action: { tool: action.tool, summary: action.summary },
      trajectoryId: trajectory.id,
    };
  });
}

export function buildPolicyExamples(trajectories: TrajectorySpan[]): PolicyTrainingExample[] {
  return trajectories.flatMap((trajectory) => buildPolicyExamplesFromTrajectory(trajectory));
}

export type MovementPolicyRunnerOptions = {
  backend?: MovementPolicyBackend;
};

/**
 * High-level façade: trains a movement policy from reviewed trajectories and
 * runs inference over an observation stream to produce a predicted movement
 * sequence. The backend is pluggable; the default is deterministic and local.
 */
export class MovementPolicyRunner {
  private readonly backend: MovementPolicyBackend;

  constructor(options: MovementPolicyRunnerOptions = {}) {
    this.backend = options.backend ?? new NearestNeighborPolicyBackend();
  }

  get backendName(): string {
    return this.backend.name;
  }

  trainFromTrajectories(trajectories: TrajectorySpan[]): MovementPolicyModel {
    return this.backend.train(buildPolicyExamples(trajectories));
  }

  trainFromExamples(examples: PolicyTrainingExample[]): MovementPolicyModel {
    return this.backend.train(examples);
  }

  restore(snapshot: MovementPolicyModelSnapshot): MovementPolicyModel {
    return this.backend.restore(snapshot);
  }

  predictSequence(
    model: MovementPolicyModel,
    observations: MovementObservationInput[],
  ): MovementPolicyPrediction[] {
    return observations.map((observation) => model.predict(observation));
  }
}

export type PolicyEvaluation = {
  total: number;
  toolMatches: number;
  exactMatches: number;
  generalizedMatches: number;
  fallbacks: number;
  /** Fraction of held-out examples whose predicted tool matched ground truth. */
  toolAccuracy: number;
  /** Tool accuracy restricted to predictions the model made by generalizing. */
  generalizationAccuracy: number;
  averageConfidence: number;
  perTool: Record<string, { total: number; matched: number }>;
};

/**
 * Generalization eval harness: measures replay fidelity of a trained policy on
 * held-out examples — how often it recovers the correct next movement, and how
 * well it does so specifically when it has to generalize (novel observations).
 */
export function evaluateMovementPolicy(
  model: MovementPolicyModel,
  examples: PolicyTrainingExample[],
): PolicyEvaluation {
  const perTool: Record<string, { total: number; matched: number }> = {};
  let toolMatches = 0;
  let exactMatches = 0;
  let generalizedMatches = 0;
  let fallbacks = 0;
  let generalizedTotal = 0;
  let generalizedCorrect = 0;
  let confidenceSum = 0;

  for (const example of examples) {
    const prediction = model.predict(example.observation);
    const matched = prediction.action.tool === example.action.tool;
    confidenceSum += prediction.confidence;

    const bucket = (perTool[example.action.tool] ??= { total: 0, matched: 0 });
    bucket.total += 1;
    if (matched) {
      bucket.matched += 1;
      toolMatches += 1;
    }

    switch (prediction.source) {
      case "exact":
        exactMatches += 1;
        break;
      case "generalized":
        generalizedMatches += 1;
        generalizedTotal += 1;
        if (matched) {
          generalizedCorrect += 1;
        }
        break;
      case "fallback":
        fallbacks += 1;
        break;
    }
  }

  const total = examples.length;
  return {
    total,
    toolMatches,
    exactMatches,
    generalizedMatches,
    fallbacks,
    toolAccuracy: total === 0 ? 0 : toolMatches / total,
    generalizationAccuracy: generalizedTotal === 0 ? 0 : generalizedCorrect / generalizedTotal,
    averageConfidence: total === 0 ? 0 : confidenceSum / total,
    perTool,
  };
}

// --- Synthetic trajectory generator (deterministic, for cloud/CI validation) ---

/** Small deterministic PRNG (mulberry32) — no global Math.random dependency. */
function mulberry32(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) | 0;
    let t = Math.imul(state ^ (state >>> 15), 1 | state);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export type SyntheticMovementPattern = {
  /** The tool a well-behaved policy should learn to emit for this pattern. */
  tool: string;
  /** Observation-source keyword (shared across the pattern's variants). */
  source: string;
  /** Observation-summary template; `{target}` is filled with a target noun. */
  summaryTemplate: string;
};

export const DEFAULT_MOVEMENT_PATTERNS: SyntheticMovementPattern[] = [
  { tool: "click", source: "ui.pointer", summaryTemplate: "click the {target} button" },
  { tool: "type", source: "ui.keyboard", summaryTemplate: "type text into the {target} field" },
  { tool: "scroll", source: "ui.pointer", summaryTemplate: "scroll to reveal the {target} panel" },
  { tool: "drag", source: "ui.pointer", summaryTemplate: "drag the {target} handle" },
  { tool: "open", source: "ui.window", summaryTemplate: "open the {target} window" },
];

export type SyntheticTrajectoryOptions = {
  count: number;
  seed?: number;
  actionsPerTrajectory?: number;
  patterns?: SyntheticMovementPattern[];
  /** Target nouns spliced into summaries; drives train/held-out splits. */
  targets?: string[];
  sessionId?: string;
  captureTier?: TrajectorySpan["captureTier"];
  startTs?: number;
};

/**
 * Generate deterministic, related synthetic trajectories to validate the
 * capture -> examples -> train -> infer round-trip without any real OS input.
 * Observations for the same pattern share tokens, so a policy trained on one
 * set of `targets` should generalize to held-out targets of the same pattern.
 */
export function generateSyntheticTrajectories(options: SyntheticTrajectoryOptions): TrajectorySpan[] {
  const patterns = options.patterns ?? DEFAULT_MOVEMENT_PATTERNS;
  const targets = options.targets ?? ["save", "submit", "cancel", "search", "profile", "settings"];
  const actionsPerTrajectory = Math.max(1, options.actionsPerTrajectory ?? 3);
  const sessionId = options.sessionId ?? "synthetic-session";
  const captureTier = options.captureTier ?? "operator";
  const startTs = options.startTs ?? 1_000_000;
  const random = mulberry32(options.seed ?? 1);

  const trajectories: TrajectorySpan[] = [];
  for (let index = 0; index < options.count; index += 1) {
    const observations: TrajectorySpan["observations"] = [];
    const actions: TrajectorySpan["actions"] = [];
    let ts = startTs + index * 10_000;

    for (let step = 0; step < actionsPerTrajectory; step += 1) {
      const pattern = patterns[Math.floor(random() * patterns.length)]!;
      const target = targets[Math.floor(random() * targets.length)]!;
      observations.push({
        kind: "observation",
        source: pattern.source,
        summary: pattern.summaryTemplate.replace("{target}", target),
        ts,
      });
      ts += 1;
      actions.push({
        kind: "action",
        tool: pattern.tool,
        summary: `${pattern.tool} ${target}`,
        ts,
      });
      ts += 1;
    }

    trajectories.push({
      id: `synthetic-${index}`,
      sessionId,
      createdAt: new Date(startTs + index * 10_000).toISOString(),
      captureTier,
      observations,
      actions,
      outcome: { status: "success", summary: "synthetic movement replay" },
    });
  }

  return trajectories;
}
