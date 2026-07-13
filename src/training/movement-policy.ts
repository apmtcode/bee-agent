import type { ReplayManifest } from "../capture/replay.js";
import type { TrajectorySpan } from "../capture/trajectory.js";

/**
 * Local-movement policy learning.
 *
 * The rest of the training subsystem ({@link ../training/runner.ts}) emits
 * launch scripts that hand a reviewed movement dataset to a real on-device
 * trainer (MLX / axolotl on Apple Silicon). That path cannot run in the cloud
 * CI where this engine evolves, so the "generalize to new but related
 * movements" objective was previously unvalidated.
 *
 * This module closes that gap with a *pluggable* backend interface plus a
 * deterministic, dependency-free in-process backend (an n-gram / Markov model
 * with context backoff). It lets the full capture -> dataset -> train -> infer
 * -> replay loop be exercised end-to-end against synthetic event streams, and
 * documents the exact seam a real small local model would implement.
 */

/**
 * Canonical, comparable token for a single recorded movement (an action the
 * operator performed). Built from a replay/trajectory action as `tool:summary`.
 */
export type MovementActionToken = string;

/**
 * The situation a movement was performed in. Kept deliberately small and
 * string-valued so it is trivially serializable and hashable.
 */
export type MovementContext = {
  /** App / surface the movement happened on (e.g. `mail`, `browser:github`). */
  app: string;
  /** Screen / view within the app, when known (e.g. `inbox`, `compose`). */
  screen?: string;
  /** The immediately preceding movement token, when known. */
  previousAction?: MovementActionToken;
};

/** One (context -> action) training example. */
export type MovementSample = {
  context: MovementContext;
  action: MovementActionToken;
};

/** How much generalization the model needed to produce a prediction. */
export type MovementBackoffLevel = "exact" | "app" | "global";

export type MovementPrediction = {
  action: MovementActionToken;
  /** Share of observations at the matched backoff level, in `[0, 1]`. */
  confidence: number;
  /** Which context specificity produced the prediction. */
  backoffLevel: MovementBackoffLevel;
};

/**
 * A serializable trained model. Structure is backend-specific but must round
 * trip through JSON so it can be persisted next to the reviewed export.
 */
export type MovementPolicyModel = {
  backendId: string;
  version: 1;
  sampleCount: number;
  [key: string]: unknown;
};

export type TrainMovementPolicyOptions = {
  /**
   * Ignore contexts observed fewer than this many times at the exact level, so
   * a single noisy sample does not overrule a well-supported backoff. Defaults
   * to 1 (keep everything).
   */
  minExactSupport?: number;
};

/**
 * The seam a real local model plugs into. A backend fits a model from reviewed
 * movement samples and predicts the next movement for a (possibly unseen)
 * context. Implementations MUST be deterministic given identical input so
 * training runs are reproducible and testable.
 */
export interface MovementPolicyBackend {
  /** Stable identifier stamped into produced models (e.g. `markov`, `mlx-lora`). */
  readonly id: string;
  train(samples: MovementSample[], options?: TrainMovementPolicyOptions): MovementPolicyModel;
  predict(model: MovementPolicyModel, context: MovementContext): MovementPrediction | undefined;
}

const KEY_SEP = "";

export function movementActionToken(tool: string, summary: string): MovementActionToken {
  return `${tool.trim()}:${summary.trim()}`;
}

function exactKey(context: MovementContext): string {
  return [context.app, context.screen ?? "", context.previousAction ?? ""].join(KEY_SEP);
}

function appKey(context: MovementContext): string {
  return [context.app, context.previousAction ?? ""].join(KEY_SEP);
}

function prevKey(context: MovementContext): string {
  return context.previousAction ?? "";
}

type CountTable = Record<string, Record<string, number>>;

type MarkovModel = MovementPolicyModel & {
  exact: CountTable;
  app: CountTable;
  prev: CountTable;
  unconditional: Record<string, number>;
};

function increment(table: CountTable, key: string, action: MovementActionToken): void {
  const bucket = (table[key] ??= {});
  bucket[action] = (bucket[action] ?? 0) + 1;
}

/**
 * Pick the highest-count action from a distribution. Ties break by
 * lexicographic action order so the result is fully deterministic.
 */
function topAction(
  distribution: Record<string, number> | undefined,
): { action: MovementActionToken; confidence: number } | undefined {
  if (!distribution) {
    return undefined;
  }
  let total = 0;
  let best: string | undefined;
  let bestCount = -1;
  for (const [action, count] of Object.entries(distribution)) {
    total += count;
    if (count > bestCount || (count === bestCount && best !== undefined && action < best)) {
      best = action;
      bestCount = count;
    }
  }
  if (best === undefined || total === 0) {
    return undefined;
  }
  return { action: best, confidence: bestCount / total };
}

/**
 * Deterministic in-process movement backend. Learns next-movement
 * distributions at three context specificities and predicts with backoff:
 * exact `(app, screen, previousAction)` -> `(app, previousAction)` ->
 * `(previousAction)` / unconditional. The backoff is what lets it generalize
 * to *new but related* movements — an unseen screen in a known app still
 * resolves through the app-level table.
 */
export class MarkovMovementPolicyBackend implements MovementPolicyBackend {
  readonly id = "markov";

  train(samples: MovementSample[], options: TrainMovementPolicyOptions = {}): MovementPolicyModel {
    const minExactSupport = Math.max(1, options.minExactSupport ?? 1);
    const exact: CountTable = {};
    const app: CountTable = {};
    const prev: CountTable = {};
    const unconditional: Record<string, number> = {};

    for (const sample of samples) {
      increment(exact, exactKey(sample.context), sample.action);
      increment(app, appKey(sample.context), sample.action);
      increment(prev, prevKey(sample.context), sample.action);
      unconditional[sample.action] = (unconditional[sample.action] ?? 0) + 1;
    }

    if (minExactSupport > 1) {
      for (const [key, bucket] of Object.entries(exact)) {
        const total = Object.values(bucket).reduce((sum, count) => sum + count, 0);
        if (total < minExactSupport) {
          delete exact[key];
        }
      }
    }

    const model: MarkovModel = {
      backendId: this.id,
      version: 1,
      sampleCount: samples.length,
      exact,
      app,
      prev,
      unconditional,
    };
    return model;
  }

  predict(model: MovementPolicyModel, context: MovementContext): MovementPrediction | undefined {
    const markov = model as MarkovModel;

    const exactHit = topAction(markov.exact?.[exactKey(context)]);
    if (exactHit) {
      return { ...exactHit, backoffLevel: "exact" };
    }
    const appHit = topAction(markov.app?.[appKey(context)]);
    if (appHit) {
      return { ...appHit, backoffLevel: "app" };
    }
    const prevHit = topAction(markov.prev?.[prevKey(context)]);
    if (prevHit) {
      return { ...prevHit, backoffLevel: "global" };
    }
    const unconditionalHit = topAction(markov.unconditional);
    if (unconditionalHit) {
      return { ...unconditionalHit, backoffLevel: "global" };
    }
    return undefined;
  }
}

const BACKEND_FACTORIES: Record<string, () => MovementPolicyBackend> = {
  markov: () => new MarkovMovementPolicyBackend(),
};

/**
 * Resolve a pluggable backend by id. Unknown ids throw with the list of known
 * backends. A real on-device backend (e.g. `mlx-lora`) registers here and
 * implements {@link MovementPolicyBackend} against the same sample/model
 * contracts this module already validates in CI.
 */
export function createMovementPolicyBackend(id = "markov"): MovementPolicyBackend {
  const factory = BACKEND_FACTORIES[id];
  if (!factory) {
    throw new Error(
      `Unknown movement policy backend "${id}". Known backends: ${Object.keys(BACKEND_FACTORIES).join(", ")}`,
    );
  }
  return factory();
}

/**
 * Convert a session {@link ReplayManifest} into ordered (context -> action)
 * samples. Observations set the current app/screen context; each action
 * becomes a target whose context carries the most recent observation and the
 * previously performed action.
 */
export function buildMovementSamplesFromReplay(manifest: ReplayManifest): MovementSample[] {
  const samples: MovementSample[] = [];
  let app = manifest.sessionId;
  let screen: string | undefined;
  let previousAction: MovementActionToken | undefined;

  for (const event of manifest.events) {
    if (event.kind === "observation") {
      app = event.source;
      screen = event.summary;
      continue;
    }
    if (event.kind === "action") {
      const action = movementActionToken(event.tool, event.summary);
      samples.push({
        context: {
          app,
          ...(screen !== undefined ? { screen } : {}),
          ...(previousAction !== undefined ? { previousAction } : {}),
        },
        action,
      });
      previousAction = action;
    }
  }
  return samples;
}

/**
 * Convert reviewed {@link TrajectorySpan}s into samples. Prefers redacted
 * review data when present so only export-approved movements are learned.
 */
export function buildMovementSamplesFromTrajectories(spans: TrajectorySpan[]): MovementSample[] {
  const samples: MovementSample[] = [];
  for (const span of spans) {
    const observations = (span.review?.redactedObservations ?? span.observations).map((observation) => ({
      ts: observation.ts,
      source: observation.source,
      summary: observation.summary,
    }));
    const actions = (span.review?.redactedActions ?? span.actions).map((action) => ({
      ts: action.ts,
      tool: action.tool,
      summary: action.summary,
    }));

    let app = span.sessionId;
    let screen: string | undefined;
    let previousAction: MovementActionToken | undefined;
    let observationCursor = 0;
    const sortedObservations = [...observations].sort((a, b) => a.ts - b.ts);

    for (const action of [...actions].sort((a, b) => a.ts - b.ts)) {
      while (observationCursor < sortedObservations.length && sortedObservations[observationCursor]!.ts <= action.ts) {
        const observation = sortedObservations[observationCursor]!;
        app = observation.source;
        screen = observation.summary;
        observationCursor += 1;
      }
      const token = movementActionToken(action.tool, action.summary);
      samples.push({
        context: {
          app,
          ...(screen !== undefined ? { screen } : {}),
          ...(previousAction !== undefined ? { previousAction } : {}),
        },
        action: token,
      });
      previousAction = token;
    }
  }
  return samples;
}

export type MovementRolloutStep = {
  context: MovementContext;
  prediction: MovementPrediction;
};

/**
 * Autonomously generate a movement sequence from a starting context by feeding
 * each prediction back in as the next `previousAction`. This is the inference
 * side of the objective: replaying learned movements and extending them into
 * new-but-related situations. Stops early when the model has no prediction.
 */
export function rolloutMovementPolicy(
  backend: MovementPolicyBackend,
  model: MovementPolicyModel,
  start: MovementContext,
  steps: number,
): MovementRolloutStep[] {
  const rollout: MovementRolloutStep[] = [];
  let context: MovementContext = { ...start };
  for (let index = 0; index < steps; index += 1) {
    const prediction = backend.predict(model, context);
    if (!prediction) {
      break;
    }
    rollout.push({ context, prediction });
    context = { ...context, previousAction: prediction.action };
  }
  return rollout;
}

export type MovementPolicyEvaluation = {
  total: number;
  correct: number;
  accuracy: number;
  /** Fraction of correct predictions that required each backoff level. */
  backoffBreakdown: Record<MovementBackoffLevel, number>;
  /** Samples for which the model produced no prediction at all. */
  abstained: number;
};

/**
 * Generalization eval harness: score a trained model against held-out samples
 * (ideally drawn from related-but-unseen contexts). Reports accuracy plus how
 * much backoff correct answers needed, so regressions in generalization are
 * measurable rather than anecdotal.
 */
export function evaluateMovementPolicy(
  backend: MovementPolicyBackend,
  model: MovementPolicyModel,
  heldOut: MovementSample[],
): MovementPolicyEvaluation {
  const backoffBreakdown: Record<MovementBackoffLevel, number> = { exact: 0, app: 0, global: 0 };
  let correct = 0;
  let abstained = 0;

  for (const sample of heldOut) {
    const prediction = backend.predict(model, sample.context);
    if (!prediction) {
      abstained += 1;
      continue;
    }
    if (prediction.action === sample.action) {
      correct += 1;
      backoffBreakdown[prediction.backoffLevel] += 1;
    }
  }

  return {
    total: heldOut.length,
    correct,
    accuracy: heldOut.length === 0 ? 0 : correct / heldOut.length,
    backoffBreakdown,
    abstained,
  };
}
