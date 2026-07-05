import type { ReplayManifest, ReplayTimelineEvent } from "../capture/replay.js";
import type { TrajectorySpan } from "../capture/trajectory.js";

/**
 * Local-movement learning: an in-process, pluggable model backend that trains on
 * recorded movement trajectories and predicts the next action for a given context.
 *
 * This is the cloud/CI-runnable half of standing objective #2 (parts c + d): it
 * post-trains a model on the reviewed movement dataset so it can (c) reproduce the
 * recorded movements and (d) generalize to new-but-related movements. The heavy,
 * on-device runtimes (MLX / axolotl, see `runner.ts`) stay behind the same
 * {@link MovementModelBackend} seam; this statistical backend is the deterministic
 * default that lets the whole pipeline — capture → dataset → train → infer → eval —
 * run end-to-end without a real GPU or real OS input.
 */

/** The state observable to the model at one decision point in a trajectory. */
export type MovementContext = {
  /** Summary of the most recent observation preceding the action (the "screen"). */
  observation?: string;
  /** Source channel of that observation (e.g. "screen", "window", "clipboard"). */
  observationSource?: string;
  /** Tool of the action immediately before this one (action→action transition). */
  previousAction?: string;
};

/** One supervised (context → action) pair derived from a recorded trajectory. */
export type MovementStep = {
  context: MovementContext;
  /** Canonical action token, `"<tool>::<summary>"`. */
  action: string;
  tool: string;
  summary: string;
  ts: number;
};

/** An ordered sequence of steps for a single trajectory. */
export type MovementSample = {
  trajectoryId: string;
  sessionId?: string;
  steps: MovementStep[];
};

/** The replayable, model-ready dataset. */
export type MovementDataset = {
  version: 1;
  samples: MovementSample[];
};

export type MovementTrainOptions = {
  /** Drop tokens shorter than this from the generalization index. Default 2. */
  minTokenLength?: number;
};

export type MovementPredictionStrategy = "exact" | "similar" | "transition" | "prior";

export type MovementPrediction = {
  tool: string;
  summary: string;
  action: string;
  /** 0..1 — share of evidence mass behind the chosen action. */
  confidence: number;
  strategy: MovementPredictionStrategy;
};

/** A trained model instance. Deterministic given the same dataset. */
export interface MovementModel {
  readonly backend: string;
  /** Predict the next action for a context, or undefined if the model is empty. */
  predict(context: MovementContext): MovementPrediction | undefined;
  /**
   * Roll out a full action sequence from an ordered list of observation contexts,
   * threading each predicted action forward as the `previousAction` of the next
   * step — this is the "repeat / generalize the recorded movement" inference path.
   */
  rollout(observations: Array<Pick<MovementContext, "observation" | "observationSource">>): MovementPrediction[];
  /** Serialize to a plain-JSON snapshot for persistence / transport. */
  snapshot(): MovementModelSnapshot;
}

/** A trainer + (de)serializer. Real on-device backends implement the same shape. */
export interface MovementModelBackend {
  readonly name: string;
  train(dataset: MovementDataset, options?: MovementTrainOptions): MovementModel;
  load(snapshot: MovementModelSnapshot): MovementModel;
}

// ---------------------------------------------------------------------------
// Dataset construction
// ---------------------------------------------------------------------------

/**
 * Turn recorded trajectories into (context → action) training samples. For each
 * action, the context is the most recent observation before it plus the tool of
 * the previous action, so the model learns both stimulus→response and the motor
 * chain of a movement.
 */
export function buildMovementDatasetFromTrajectories(trajectories: TrajectorySpan[]): MovementDataset {
  const samples = trajectories.map((trajectory) => {
    const observations = (trajectory.review?.redactedObservations ?? trajectory.observations)
      .map((observation) => ({ ts: observation.ts, source: observation.source, summary: observation.summary }))
      .sort((a, b) => a.ts - b.ts);
    const actions = (trajectory.review?.redactedActions ?? trajectory.actions)
      .map((action) => ({ ts: action.ts, tool: action.tool, summary: action.summary }))
      .sort((a, b) => a.ts - b.ts);
    return { trajectoryId: trajectory.id, sessionId: trajectory.sessionId, steps: buildSteps(observations, actions) };
  });
  return { version: 1, samples };
}

/** Build the same dataset from replay manifests (already-sorted timelines). */
export function buildMovementDatasetFromReplays(replays: ReplayManifest[]): MovementDataset {
  const samples: MovementSample[] = [];
  for (const replay of replays) {
    const byTrajectory = new Map<string, { observations: ObservationSlice[]; actions: ActionSlice[] }>();
    for (const event of replay.events) {
      if (event.kind === "observation") {
        getBucket(byTrajectory, event.trajectoryId).observations.push({
          ts: event.ts,
          source: event.source,
          summary: event.summary,
        });
      } else if (event.kind === "action") {
        getBucket(byTrajectory, event.trajectoryId).actions.push({
          ts: event.ts,
          tool: event.tool,
          summary: event.summary,
        });
      }
    }
    for (const [trajectoryId, bucket] of [...byTrajectory.entries()].sort((a, b) => compareStrings(a[0], b[0]))) {
      samples.push({
        trajectoryId,
        sessionId: replay.sessionId,
        steps: buildSteps(
          bucket.observations.sort((a, b) => a.ts - b.ts),
          bucket.actions.sort((a, b) => a.ts - b.ts),
        ),
      });
    }
  }
  return { version: 1, samples };
}

type ObservationSlice = { ts: number; source: string; summary: string };
type ActionSlice = { ts: number; tool: string; summary: string };

function getBucket(
  map: Map<string, { observations: ObservationSlice[]; actions: ActionSlice[] }>,
  key: string,
): { observations: ObservationSlice[]; actions: ActionSlice[] } {
  let bucket = map.get(key);
  if (!bucket) {
    bucket = { observations: [], actions: [] };
    map.set(key, bucket);
  }
  return bucket;
}

function buildSteps(observations: ObservationSlice[], actions: ActionSlice[]): MovementStep[] {
  const steps: MovementStep[] = [];
  let previousAction: string | undefined;
  for (const action of actions) {
    const observation = latestBefore(observations, action.ts);
    steps.push({
      context: {
        ...(observation ? { observation: observation.summary, observationSource: observation.source } : {}),
        ...(previousAction ? { previousAction } : {}),
      },
      action: actionToken(action.tool, action.summary),
      tool: action.tool,
      summary: action.summary,
      ts: action.ts,
    });
    previousAction = action.tool;
  }
  return steps;
}

function latestBefore(observations: ObservationSlice[], ts: number): ObservationSlice | undefined {
  let match: ObservationSlice | undefined;
  for (const observation of observations) {
    if (observation.ts <= ts) {
      match = observation;
    } else {
      break;
    }
  }
  return match;
}

export function actionToken(tool: string, summary: string): string {
  return `${tool}::${summary}`;
}

// ---------------------------------------------------------------------------
// Snapshot format
// ---------------------------------------------------------------------------

type CountEntries = Array<[string, Array<[string, number]>]>;

export type MovementModelSnapshot = {
  version: 1;
  backend: string;
  minTokenLength: number;
  /** observation key → action → count (exact-context memory). */
  contextActions: CountEntries;
  /** previous-action tool → action → count (motor-chain transitions). */
  transitions: CountEntries;
  /** observation token → action → count (generalization index). */
  tokenActions: CountEntries;
  /** action → global count (prior). */
  priors: Array<[string, number]>;
  /** action token → its {tool, summary} parts (for reconstruction). */
  actionParts: Array<[string, { tool: string; summary: string }]>;
};

// ---------------------------------------------------------------------------
// Statistical backend (deterministic default)
// ---------------------------------------------------------------------------

const STOPWORDS = new Set([
  "the", "a", "an", "of", "to", "in", "on", "at", "is", "and", "or", "for", "with", "by", "this", "that",
]);

export const STATISTICAL_MOVEMENT_BACKEND_NAME = "statistical-ngram";

/**
 * A backoff n-gram / token-affinity model. Reproduces recorded movements via exact
 * context memory and generalizes to related ones via shared-token affinity, falling
 * back to action→action transitions and finally the global action prior.
 */
export class StatisticalMovementBackend implements MovementModelBackend {
  readonly name = STATISTICAL_MOVEMENT_BACKEND_NAME;

  train(dataset: MovementDataset, options?: MovementTrainOptions): MovementModel {
    const minTokenLength = options?.minTokenLength ?? 2;
    const contextActions = new NestedCounter();
    const transitions = new NestedCounter();
    const tokenActions = new NestedCounter();
    const priors = new Map<string, number>();
    const actionParts = new Map<string, { tool: string; summary: string }>();

    for (const sample of dataset.samples) {
      for (const step of sample.steps) {
        actionParts.set(step.action, { tool: step.tool, summary: step.summary });
        priors.set(step.action, (priors.get(step.action) ?? 0) + 1);
        if (step.context.observation !== undefined) {
          contextActions.add(normalizeKey(step.context.observation), step.action);
          for (const token of tokenize(step.context.observation, minTokenLength)) {
            tokenActions.add(token, step.action);
          }
        }
        if (step.context.previousAction !== undefined) {
          transitions.add(normalizeKey(step.context.previousAction), step.action);
        }
      }
    }

    return new StatisticalMovementModel({
      backend: this.name,
      minTokenLength,
      contextActions,
      transitions,
      tokenActions,
      priors,
      actionParts,
    });
  }

  load(snapshot: MovementModelSnapshot): MovementModel {
    return new StatisticalMovementModel({
      backend: snapshot.backend,
      minTokenLength: snapshot.minTokenLength,
      contextActions: NestedCounter.fromEntries(snapshot.contextActions),
      transitions: NestedCounter.fromEntries(snapshot.transitions),
      tokenActions: NestedCounter.fromEntries(snapshot.tokenActions),
      priors: new Map(snapshot.priors),
      actionParts: new Map(snapshot.actionParts),
    });
  }
}

type StatisticalModelState = {
  backend: string;
  minTokenLength: number;
  contextActions: NestedCounter;
  transitions: NestedCounter;
  tokenActions: NestedCounter;
  priors: Map<string, number>;
  actionParts: Map<string, { tool: string; summary: string }>;
};

class StatisticalMovementModel implements MovementModel {
  readonly backend: string;

  constructor(private readonly state: StatisticalModelState) {
    this.backend = state.backend;
  }

  predict(context: MovementContext): MovementPrediction | undefined {
    if (context.observation !== undefined) {
      const exact = this.state.contextActions.argmax(normalizeKey(context.observation));
      if (exact) {
        return this.toPrediction(exact.key, exact.share, "exact");
      }
      const similar = this.predictBySimilarity(context.observation);
      if (similar) {
        return similar;
      }
    }
    if (context.previousAction !== undefined) {
      const transition = this.state.transitions.argmax(normalizeKey(context.previousAction));
      if (transition) {
        return this.toPrediction(transition.key, transition.share, "transition");
      }
    }
    const prior = argmaxMap(this.state.priors);
    if (prior) {
      return this.toPrediction(prior.key, prior.share, "prior");
    }
    return undefined;
  }

  rollout(
    observations: Array<Pick<MovementContext, "observation" | "observationSource">>,
  ): MovementPrediction[] {
    const predictions: MovementPrediction[] = [];
    let previousAction: string | undefined;
    for (const observation of observations) {
      const prediction = this.predict({
        ...(observation.observation !== undefined ? { observation: observation.observation } : {}),
        ...(observation.observationSource !== undefined
          ? { observationSource: observation.observationSource }
          : {}),
        ...(previousAction ? { previousAction } : {}),
      });
      if (!prediction) {
        break;
      }
      predictions.push(prediction);
      previousAction = prediction.tool;
    }
    return predictions;
  }

  snapshot(): MovementModelSnapshot {
    return {
      version: 1,
      backend: this.state.backend,
      minTokenLength: this.state.minTokenLength,
      contextActions: this.state.contextActions.toEntries(),
      transitions: this.state.transitions.toEntries(),
      tokenActions: this.state.tokenActions.toEntries(),
      priors: [...this.state.priors.entries()].sort((a, b) => compareStrings(a[0], b[0])),
      actionParts: [...this.state.actionParts.entries()].sort((a, b) => compareStrings(a[0], b[0])),
    };
  }

  private predictBySimilarity(observation: string): MovementPrediction | undefined {
    const tokens = tokenize(observation, this.state.minTokenLength);
    if (tokens.length === 0) {
      return undefined;
    }
    const scores = new Map<string, number>();
    for (const token of tokens) {
      const perAction = this.state.tokenActions.get(token);
      if (!perAction) {
        continue;
      }
      let tokenTotal = 0;
      for (const count of perAction.values()) {
        tokenTotal += count;
      }
      // Inverse-frequency weight so a token shared by every context (low signal)
      // counts less than a token that distinguishes one movement family.
      const weight = 1 / tokenTotal;
      for (const [action, count] of perAction) {
        scores.set(action, (scores.get(action) ?? 0) + count * weight);
      }
    }
    const best = argmaxMap(scores);
    if (!best) {
      return undefined;
    }
    return this.toPrediction(best.key, best.share, "similar");
  }

  private toPrediction(action: string, confidence: number, strategy: MovementPredictionStrategy): MovementPrediction {
    const parts = this.state.actionParts.get(action) ?? splitActionToken(action);
    return { tool: parts.tool, summary: parts.summary, action, confidence, strategy };
  }
}

// ---------------------------------------------------------------------------
// Evaluation harness (generalization / replay fidelity)
// ---------------------------------------------------------------------------

export type MovementEvalReport = {
  trajectoryCount: number;
  stepCount: number;
  /** Steps where the predicted action token equals the ground-truth action token. */
  actionMatched: number;
  /** Steps where only the predicted tool equals the ground-truth tool (looser). */
  toolMatched: number;
  /** actionMatched / stepCount. */
  fidelity: number;
  /** toolMatched / stepCount. */
  toolFidelity: number;
  byStrategy: Record<MovementPredictionStrategy, number>;
};

/**
 * Measure how faithfully a trained model reproduces a held-out (but related)
 * dataset — the generalization eval. Feeds each recorded context back to the model
 * and compares its prediction to the recorded action.
 */
export function evaluateMovementModel(model: MovementModel, heldOut: MovementDataset): MovementEvalReport {
  let stepCount = 0;
  let actionMatched = 0;
  let toolMatched = 0;
  const byStrategy: Record<MovementPredictionStrategy, number> = {
    exact: 0,
    similar: 0,
    transition: 0,
    prior: 0,
  };
  for (const sample of heldOut.samples) {
    for (const step of sample.steps) {
      stepCount += 1;
      const prediction = model.predict(step.context);
      if (!prediction) {
        continue;
      }
      byStrategy[prediction.strategy] += 1;
      if (prediction.action === step.action) {
        actionMatched += 1;
      }
      if (prediction.tool === step.tool) {
        toolMatched += 1;
      }
    }
  }
  return {
    trajectoryCount: heldOut.samples.length,
    stepCount,
    actionMatched,
    toolMatched,
    fidelity: stepCount === 0 ? 0 : actionMatched / stepCount,
    toolFidelity: stepCount === 0 ? 0 : toolMatched / stepCount,
    byStrategy,
  };
}

// ---------------------------------------------------------------------------
// Internals
// ---------------------------------------------------------------------------

/** A Map<string, Map<string, number>> with deterministic argmax + serialization. */
class NestedCounter {
  private readonly rows = new Map<string, Map<string, number>>();

  add(key: string, inner: string): void {
    let row = this.rows.get(key);
    if (!row) {
      row = new Map<string, number>();
      this.rows.set(key, row);
    }
    row.set(inner, (row.get(inner) ?? 0) + 1);
  }

  get(key: string): Map<string, number> | undefined {
    return this.rows.get(key);
  }

  argmax(key: string): { key: string; share: number } | undefined {
    const row = this.rows.get(key);
    if (!row) {
      return undefined;
    }
    return argmaxMap(row);
  }

  toEntries(): CountEntries {
    return [...this.rows.entries()]
      .sort((a, b) => compareStrings(a[0], b[0]))
      .map(([key, row]) => [
        key,
        [...row.entries()].sort((a, b) => compareStrings(a[0], b[0])) as Array<[string, number]>,
      ]);
  }

  static fromEntries(entries: CountEntries): NestedCounter {
    const counter = new NestedCounter();
    for (const [key, row] of entries) {
      const inner = new Map<string, number>(row);
      counter.rows.set(key, inner);
    }
    return counter;
  }
}

/** Deterministic argmax: highest count wins; ties broken by lowest key. */
function argmaxMap(counts: Map<string, number>): { key: string; share: number } | undefined {
  let total = 0;
  for (const count of counts.values()) {
    total += count;
  }
  if (total === 0) {
    return undefined;
  }
  let bestKey: string | undefined;
  let bestCount = -1;
  for (const [key, count] of [...counts.entries()].sort((a, b) => compareStrings(a[0], b[0]))) {
    if (count > bestCount) {
      bestCount = count;
      bestKey = key;
    }
  }
  if (bestKey === undefined) {
    return undefined;
  }
  return { key: bestKey, share: bestCount / total };
}

function tokenize(value: string, minTokenLength: number): string[] {
  return value
    .toLowerCase()
    .split(/[^a-z0-9]+/)
    .filter((token) => token.length >= minTokenLength && !STOPWORDS.has(token));
}

function normalizeKey(value: string): string {
  return value.trim().toLowerCase();
}

function splitActionToken(action: string): { tool: string; summary: string } {
  const index = action.indexOf("::");
  if (index === -1) {
    return { tool: action, summary: "" };
  }
  return { tool: action.slice(0, index), summary: action.slice(index + 2) };
}

function compareStrings(a: string, b: string): number {
  if (a < b) {
    return -1;
  }
  if (a > b) {
    return 1;
  }
  return 0;
}

/** Re-export for callers that only need to read replay event kinds. */
export type { ReplayTimelineEvent };
