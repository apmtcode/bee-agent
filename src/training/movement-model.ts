// Local-movement learning subsystem — pluggable model backend (objective #2c/#2d).
//
// The rest of the movement subsystem records movements (capture/*), gives them a
// schema (capture/trajectory), a dataset format (training/exporter), and a replay
// engine (capture/replay). This module supplies the missing piece: a *model* that
// (c) learns from the recorded movement dataset so it can repeat recorded
// movements, and (d) generalizes to perform new-but-related movements.
//
// The backend is pluggable: `MovementModelBackend` is the seam. In the cloud (and
// in CI) we cannot touch the real OS or train a heavyweight model, so this file
// ships `MarkovMovementBackend` — a fully deterministic, dependency-free learned
// backend that trains in-process and runs on synthetic event streams. A real
// on-device small-model backend (e.g. the mlx/axolotl launch plan produced by
// `LocalAppleSiliconTrainingRunner`) can implement the same interface without any
// caller change.

import type { TrajectorySpan } from "../capture/trajectory.js";
import type { ReplayManifest, ReplayTimelineEvent } from "../capture/replay.js";

/** One recorded movement — the atom the model learns to reproduce/generalize. */
export type MovementStep = {
  ts: number;
  /** The gesture/tool: e.g. "tap", "swipe", "scroll", "type", "shortcut", or a tool name. */
  gesture: string;
  /** UI/element the movement targets, if known (e.g. "submit-button"). */
  target?: string;
  /** Directional gestures (swipe/scroll). */
  direction?: string;
  /** Free-text summary carried through for reconstruction/debugging. */
  summary?: string;
};

/** An ordered run of movements sharing a context (app/screen). */
export type MovementSequence = {
  id: string;
  /** Context the sequence happened in (used as a generalization key). */
  context: string;
  steps: MovementStep[];
};

export type MovementDataset = {
  version: 1;
  sequences: MovementSequence[];
};

/** The prefix the model conditions on when predicting the next movement. */
export type MovementContext = {
  context: string;
  /** Movements observed so far in the current sequence (most recent last). */
  history: MovementStep[];
};

export type MovementPrediction = {
  step: MovementStep;
  /** Normalized confidence in [0,1]. */
  confidence: number;
  /** Which back-off order produced this prediction (see MarkovMovementBackend). */
  order: number;
  /** True when the prediction came from a broader context than requested (generalization). */
  generalized: boolean;
};

export type TrainedMovementModelStats = {
  backend: string;
  sequenceCount: number;
  stepCount: number;
  distinctTokens: number;
  /** Highest context order the model actually retained. */
  maxOrder: number;
};

/**
 * A trained model. `predictNext` is the primitive (one-step, teacher-forced);
 * `generate` rolls it out to reproduce or extend a movement.
 */
export interface TrainedMovementModel {
  readonly stats: TrainedMovementModelStats;
  predictNext(context: MovementContext): MovementPrediction | undefined;
  generate(seed: MovementContext, maxSteps: number): MovementStep[];
}

export type MovementTrainOptions = {
  /** Context window (n-gram order). Higher = more faithful reproduction, less generalization. */
  order?: number;
};

/** The pluggable seam. Any backend (mock, on-device small model, remote) implements this. */
export interface MovementModelBackend {
  readonly name: string;
  train(dataset: MovementDataset, options?: MovementTrainOptions): Promise<TrainedMovementModel>;
}

// ---------------------------------------------------------------------------
// Token helpers — canonical string encoding of a movement, so the model works
// over a discrete vocabulary regardless of how the step was captured.
// ---------------------------------------------------------------------------

// Sorts lexicographically last so a tie at a broad back-off context prefers a
// real movement over "sequence ends here" — that tie is exactly where
// generalization to an unseen-but-related prefix happens.
const TERMINAL_TOKEN = "￿end";

export function movementToken(step: MovementStep): string {
  return [step.gesture, step.target ?? "", step.direction ?? ""].join("");
}

function contextKey(context: string, tokens: string[]): string {
  return `${context}${tokens.join("")}`;
}

// ---------------------------------------------------------------------------
// Dataset construction from the existing capture artifacts.
// ---------------------------------------------------------------------------

/** Build a movement dataset from reviewed/recorded trajectory spans. */
export function buildMovementDatasetFromTrajectories(trajectories: TrajectorySpan[]): MovementDataset {
  const sequences: MovementSequence[] = [];
  for (const trajectory of trajectories) {
    const actions = trajectory.review?.redactedActions
      ? trajectory.review.redactedActions.map((action) => ({ tool: action.tool, summary: action.summary, ts: action.ts, metadata: undefined as Record<string, unknown> | undefined }))
      : trajectory.actions.map((action) => ({ tool: action.tool, summary: action.summary, ts: action.ts, metadata: action.metadata }));
    if (actions.length === 0) {
      continue;
    }
    const steps = actions
      .slice()
      .sort((a, b) => a.ts - b.ts)
      .map((action) => stepFromAction(action.tool, action.summary, action.ts, action.metadata));
    sequences.push({
      id: trajectory.id,
      context: inferContext(trajectory),
      steps,
    });
  }
  return { version: 1, sequences };
}

/** Build a movement dataset from a replay manifest's action timeline. */
export function buildMovementDatasetFromReplay(manifest: ReplayManifest): MovementDataset {
  const byTrajectory = new Map<string, MovementStep[]>();
  for (const event of manifest.events) {
    if (event.kind !== "action") {
      continue;
    }
    const actionEvent = event as Extract<ReplayTimelineEvent, { kind: "action" }>;
    const list = byTrajectory.get(actionEvent.trajectoryId) ?? [];
    list.push(stepFromAction(actionEvent.tool, actionEvent.summary, actionEvent.ts, undefined));
    byTrajectory.set(actionEvent.trajectoryId, list);
  }
  const sequences: MovementSequence[] = [];
  for (const [trajectoryId, steps] of byTrajectory) {
    if (steps.length === 0) {
      continue;
    }
    sequences.push({ id: trajectoryId, context: manifest.sessionId, steps });
  }
  return { version: 1, sequences };
}

function stepFromAction(
  tool: string,
  summary: string,
  ts: number,
  metadata: Record<string, unknown> | undefined,
): MovementStep {
  const gesture = typeof metadata?.gesture === "string" ? metadata.gesture : tool;
  const target = typeof metadata?.target === "string" ? metadata.target : undefined;
  const direction = typeof metadata?.direction === "string" ? metadata.direction : undefined;
  return {
    ts,
    gesture,
    ...(target ? { target } : {}),
    ...(direction ? { direction } : {}),
    summary,
  };
}

function inferContext(trajectory: TrajectorySpan): string {
  for (const observation of trajectory.observations) {
    const appName = observation.metadata?.appName;
    if (typeof appName === "string" && appName.length > 0) {
      return appName;
    }
  }
  const first = trajectory.observations[0];
  return first?.source ?? "unknown";
}

// ---------------------------------------------------------------------------
// MarkovMovementBackend — the deterministic learned backend.
//
// Trains a variable-order Markov model: for every order o in [0, order] it
// records how often each token follows each length-o context (scoped to the
// sequence's context string). Prediction uses stupid back-off — it tries the
// highest order whose context was seen and falls back to shorter contexts,
// which is exactly what lets the model GENERALIZE to a new-but-related movement
// whose exact prefix was never recorded. All tie-breaking is deterministic
// (higher count first, then lexicographic token), so training and inference are
// reproducible in the cloud with no RNG.
// ---------------------------------------------------------------------------

export class MarkovMovementBackend implements MovementModelBackend {
  readonly name = "markov";

  async train(dataset: MovementDataset, options?: MovementTrainOptions): Promise<TrainedMovementModel> {
    const order = Math.max(1, Math.min(options?.order ?? 2, 8));
    // order -> contextKey -> token -> count
    const tables: Map<number, Map<string, Map<string, number>>> = new Map();
    for (let o = 0; o <= order; o += 1) {
      tables.set(o, new Map());
    }
    const stepByToken = new Map<string, MovementStep>();
    const vocabulary = new Set<string>();
    let stepCount = 0;

    for (const sequence of dataset.sequences) {
      const tokens = sequence.steps.map((step) => {
        const token = movementToken(step);
        if (!stepByToken.has(token)) {
          stepByToken.set(token, step);
        }
        vocabulary.add(token);
        return token;
      });
      // Append a terminal so the model learns where sequences end.
      const withEnd = [...tokens, TERMINAL_TOKEN];
      for (let i = 0; i < withEnd.length; i += 1) {
        stepCount += i < tokens.length ? 1 : 0;
        const nextToken = withEnd[i];
        for (let o = 0; o <= order; o += 1) {
          if (i - o < 0) {
            continue;
          }
          const prev = withEnd.slice(i - o, i);
          const key = contextKey(sequence.context, prev);
          const table = tables.get(o)!;
          const counts = table.get(key) ?? new Map<string, number>();
          counts.set(nextToken, (counts.get(nextToken) ?? 0) + 1);
          table.set(key, counts);
        }
      }
    }

    const stats: TrainedMovementModelStats = {
      backend: this.name,
      sequenceCount: dataset.sequences.length,
      stepCount,
      distinctTokens: vocabulary.size,
      maxOrder: order,
    };
    return new MarkovMovementModel(order, tables, stepByToken, stats);
  }
}

class MarkovMovementModel implements TrainedMovementModel {
  constructor(
    private readonly order: number,
    private readonly tables: Map<number, Map<string, Map<string, number>>>,
    private readonly stepByToken: Map<string, MovementStep>,
    readonly stats: TrainedMovementModelStats,
  ) {}

  predictNext(context: MovementContext): MovementPrediction | undefined {
    const historyTokens = context.history.map((step) => movementToken(step));
    for (let o = Math.min(this.order, historyTokens.length); o >= 0; o -= 1) {
      const prev = historyTokens.slice(historyTokens.length - o, historyTokens.length);
      const table = this.tables.get(o);
      if (!table) {
        continue;
      }
      const counts = table.get(contextKey(context.context, prev));
      if (!counts || counts.size === 0) {
        continue;
      }
      const best = pickBest(counts);
      if (!best || best.token === TERMINAL_TOKEN) {
        // A confident "sequence ends here" prediction is still a valid answer at
        // the exact-context order, but for callers `generate` handles termination.
        if (best?.token === TERMINAL_TOKEN) {
          return undefined;
        }
        continue;
      }
      const step = this.stepByToken.get(best.token);
      if (!step) {
        continue;
      }
      return {
        step,
        confidence: best.total === 0 ? 0 : best.count / best.total,
        order: o,
        generalized: o < Math.min(this.order, historyTokens.length),
      };
    }
    return undefined;
  }

  generate(seed: MovementContext, maxSteps: number): MovementStep[] {
    const generated: MovementStep[] = [];
    const history = [...seed.history];
    for (let i = 0; i < maxSteps; i += 1) {
      const prediction = this.predictNext({ context: seed.context, history });
      if (!prediction) {
        break;
      }
      generated.push(prediction.step);
      history.push(prediction.step);
    }
    return generated;
  }
}

function pickBest(counts: Map<string, number>): { token: string; count: number; total: number } | undefined {
  let total = 0;
  let best: { token: string; count: number } | undefined;
  for (const [token, count] of counts) {
    total += count;
    if (!best || count > best.count || (count === best.count && token < best.token)) {
      best = { token, count };
    }
  }
  return best ? { token: best.token, count: best.count, total } : undefined;
}
