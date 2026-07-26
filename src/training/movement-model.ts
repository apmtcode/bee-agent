import type { TrajectorySpan } from "../capture/trajectory.js";
import type { ReplayManifest, ReplayTimelineEvent } from "../capture/replay.js";

/**
 * Pluggable local-movement model backend (standing objective #2c/#2d).
 *
 * The training runner ({@link ../training/runner.ts}) emits command plans for
 * external on-device toolchains (MLX / Axolotl). That covers *how* a real model
 * gets trained on the user's machine, but bee-agent had no *in-process* model it
 * could actually train and query to (c) repeat recorded movements and (d)
 * generalize to related-but-new movements. This module provides that seam:
 *
 *   - {@link MovementModelBackend} — the pluggable interface. A real on-device
 *     small model plugs in here behind the same contract.
 *   - {@link NgramMovementBackend} — a deterministic, dependency-free mock
 *     backend (Markov n-gram policy with tool-level backoff) so the whole
 *     capture → dataset → train → infer → replay loop is exercisable and
 *     testable in the cloud with synthetic event streams.
 *
 * The model is intentionally simple and fully deterministic (no Date/random) so
 * it can serve as the reference/mock implementation and as a fast baseline for
 * the generalization eval harness.
 */

export type MovementObservationEvent = {
  kind: "observation";
  source: string;
  summary: string;
};

export type MovementActionEvent = {
  kind: "action";
  tool: string;
  summary: string;
};

export type MovementEvent = MovementObservationEvent | MovementActionEvent;

/** An ordered stream of movement events derived from a single recording. */
export type MovementSequence = {
  id: string;
  events: MovementEvent[];
};

export type MovementPredictionSource = "exact" | "generalized" | "prior";

export type MovementPrediction = {
  action: MovementActionEvent;
  /** Fraction of the matched context's distribution that chose this action. */
  confidence: number;
  /**
   * How the prediction was reached:
   *  - `exact`: the specific context (tool + summary) was seen in training →
   *    reproduces the recorded movement.
   *  - `generalized`: only the abstract context (tool sequence, ignoring the
   *    instance-specific summary) matched → a related-but-new movement.
   *  - `prior`: no context matched; fell back to the most frequent action.
   */
  source: MovementPredictionSource;
  /** Length of the context window that produced the match (0 for prior). */
  matchedOrder: number;
};

export type MovementPolicyStats = {
  backendId: string;
  order: number;
  sequenceCount: number;
  actionCount: number;
  /** Distinct specific-context keys learned (a rough model-size proxy). */
  specificContextCount: number;
  genericContextCount: number;
};

export interface MovementPolicy {
  readonly stats: MovementPolicyStats;
  /** Predict the next action given the trailing context of events. */
  predict(context: MovementEvent[]): MovementPrediction | undefined;
}

export type MovementTrainOptions = {
  /** Maximum context window length (n-gram order). Default 3. */
  order?: number;
};

export interface MovementModelBackend {
  readonly id: string;
  train(sequences: MovementSequence[], options?: MovementTrainOptions): MovementPolicy;
}

// --- adapters: existing recorded formats → movement sequences ----------------

/** Interleave a trajectory's observations and actions into a time-ordered stream. */
export function sequenceFromTrajectory(trajectory: TrajectorySpan): MovementSequence {
  const events: Array<{ ts: number; event: MovementEvent }> = [
    ...trajectory.observations.map((observation) => ({
      ts: observation.ts,
      event: { kind: "observation", source: observation.source, summary: observation.summary } as MovementEvent,
    })),
    ...trajectory.actions.map((action) => ({
      ts: action.ts,
      event: { kind: "action", tool: action.tool, summary: action.summary } as MovementEvent,
    })),
  ];
  events.sort((a, b) => {
    if (a.ts !== b.ts) {
      return a.ts - b.ts;
    }
    // Observations precede actions at the same timestamp (matches replay ordering).
    return a.event.kind === b.event.kind ? 0 : a.event.kind === "observation" ? -1 : 1;
  });
  return { id: trajectory.id, events: events.map((entry) => entry.event) };
}

/** Derive a movement sequence from a replay manifest (drops transcript events). */
export function sequenceFromReplayManifest(manifest: ReplayManifest): MovementSequence {
  const events = manifest.events.flatMap((event) => movementEventFromReplay(event));
  return { id: manifest.sessionId, events };
}

function movementEventFromReplay(event: ReplayTimelineEvent): MovementEvent[] {
  switch (event.kind) {
    case "observation":
      return [{ kind: "observation", source: event.source, summary: event.summary }];
    case "action":
      return [{ kind: "action", tool: event.tool, summary: event.summary }];
    case "transcript":
      return [];
  }
}

// --- feature keys ------------------------------------------------------------

function specificKey(event: MovementEvent): string {
  return event.kind === "action"
    ? `a:${event.tool}::${event.summary}`
    : `o:${event.source}::${event.summary}`;
}

function genericKey(event: MovementEvent): string {
  return event.kind === "action" ? `a:${event.tool}` : `o:${event.source}`;
}

function contextKey(events: MovementEvent[], key: (event: MovementEvent) => string): string {
  return events.map(key).join(" | ");
}

function actionKey(action: MovementActionEvent): string {
  return `${action.tool}::${action.summary}`;
}

// --- deterministic n-gram backend -------------------------------------------

type Distribution = Map<string, { action: MovementActionEvent; count: number }>;

type NgramModel = {
  order: number;
  /** contextLen (1..order) → contextKey → distribution over next action. */
  specific: Map<number, Map<string, Distribution>>;
  generic: Map<number, Map<string, Distribution>>;
  prior: Distribution;
  sequenceCount: number;
  actionCount: number;
};

function addToDistribution(distribution: Distribution, action: MovementActionEvent): void {
  const key = actionKey(action);
  const existing = distribution.get(key);
  if (existing) {
    existing.count += 1;
  } else {
    distribution.set(key, { action, count: 1 });
  }
}

function record(
  levels: Map<number, Map<string, Distribution>>,
  contextLen: number,
  key: string,
  action: MovementActionEvent,
): void {
  let byContext = levels.get(contextLen);
  if (!byContext) {
    byContext = new Map();
    levels.set(contextLen, byContext);
  }
  let distribution = byContext.get(key);
  if (!distribution) {
    distribution = new Map();
    byContext.set(key, distribution);
  }
  addToDistribution(distribution, action);
}

/**
 * Pick the highest-count action in a distribution deterministically. Ties break
 * by action key (lexicographic) so results never depend on insertion order.
 */
function argmax(distribution: Distribution): { action: MovementActionEvent; count: number; total: number } | undefined {
  let best: { key: string; action: MovementActionEvent; count: number } | undefined;
  let total = 0;
  for (const [key, entry] of distribution) {
    total += entry.count;
    if (!best || entry.count > best.count || (entry.count === best.count && key < best.key)) {
      best = { key, action: entry.action, count: entry.count };
    }
  }
  return best ? { action: best.action, count: best.count, total } : undefined;
}

export class NgramMovementBackend implements MovementModelBackend {
  readonly id = "ngram-mock";

  train(sequences: MovementSequence[], options: MovementTrainOptions = {}): MovementPolicy {
    const order = Math.max(1, Math.floor(options.order ?? 3));
    const model: NgramModel = {
      order,
      specific: new Map(),
      generic: new Map(),
      prior: new Map(),
      sequenceCount: sequences.length,
      actionCount: 0,
    };

    for (const sequence of sequences) {
      const events = sequence.events;
      for (let i = 0; i < events.length; i += 1) {
        const current = events[i];
        if (current.kind !== "action") {
          continue;
        }
        model.actionCount += 1;
        addToDistribution(model.prior, current);
        for (let len = 1; len <= order; len += 1) {
          const start = i - len;
          if (start < 0) {
            break;
          }
          const window = events.slice(start, i);
          record(model.specific, len, contextKey(window, specificKey), current);
          record(model.generic, len, contextKey(window, genericKey), current);
        }
      }
    }

    return new NgramMovementPolicy(this.id, model);
  }
}

class NgramMovementPolicy implements MovementPolicy {
  constructor(
    private readonly backendId: string,
    private readonly model: NgramModel,
  ) {}

  get stats(): MovementPolicyStats {
    return {
      backendId: this.backendId,
      order: this.model.order,
      sequenceCount: this.model.sequenceCount,
      actionCount: this.model.actionCount,
      specificContextCount: countContexts(this.model.specific),
      genericContextCount: countContexts(this.model.generic),
    };
  }

  predict(context: MovementEvent[]): MovementPrediction | undefined {
    const maxLen = Math.min(this.model.order, context.length);

    // 1. Exact (instance-specific) match — reproduces the recorded movement.
    for (let len = maxLen; len >= 1; len -= 1) {
      const window = context.slice(context.length - len);
      const hit = this.model.specific.get(len)?.get(contextKey(window, specificKey));
      const best = hit && argmax(hit);
      if (best) {
        return { action: best.action, confidence: best.count / best.total, source: "exact", matchedOrder: len };
      }
    }

    // 2. Generalized match — abstract tool/source context, novel instance.
    for (let len = maxLen; len >= 1; len -= 1) {
      const window = context.slice(context.length - len);
      const hit = this.model.generic.get(len)?.get(contextKey(window, genericKey));
      const best = hit && argmax(hit);
      if (best) {
        return { action: best.action, confidence: best.count / best.total, source: "generalized", matchedOrder: len };
      }
    }

    // 3. Prior — most frequent action overall.
    const prior = argmax(this.model.prior);
    if (prior) {
      return { action: prior.action, confidence: prior.count / prior.total, source: "prior", matchedOrder: 0 };
    }
    return undefined;
  }
}

function countContexts(levels: Map<number, Map<string, Distribution>>): number {
  let total = 0;
  for (const byContext of levels.values()) {
    total += byContext.size;
  }
  return total;
}

// --- rollout / replay driver -------------------------------------------------

export type MovementRolloutOptions = {
  maxSteps?: number;
  /** Stop once a prediction's confidence drops below this threshold. */
  minConfidence?: number;
  /**
   * Keep rolling out even when the policy falls back to the global `prior`
   * (i.e. has no genuine context match). Default false: a `prior` prediction
   * marks the natural end of a learned movement, so the rollout stops there
   * instead of emitting the most-frequent action indefinitely.
   */
  allowPrior?: boolean;
};

export type MovementRolloutStep = {
  action: MovementActionEvent;
  confidence: number;
  source: MovementPredictionSource;
};

/**
 * Greedily roll out a policy from a seed context to reproduce (or generalize) a
 * movement sequence — the in-process analogue of "replay the recorded
 * movements". Each predicted action is appended to the running context so the
 * next prediction conditions on it.
 */
export function rolloutMovement(
  policy: MovementPolicy,
  seed: MovementEvent[],
  options: MovementRolloutOptions = {},
): MovementRolloutStep[] {
  const maxSteps = Math.max(0, Math.floor(options.maxSteps ?? 32));
  const minConfidence = options.minConfidence ?? 0;
  const allowPrior = options.allowPrior ?? false;
  const context = [...seed];
  const steps: MovementRolloutStep[] = [];
  for (let step = 0; step < maxSteps; step += 1) {
    const prediction = policy.predict(context);
    if (!prediction || prediction.confidence < minConfidence) {
      break;
    }
    if (prediction.source === "prior" && !allowPrior) {
      break;
    }
    steps.push({ action: prediction.action, confidence: prediction.confidence, source: prediction.source });
    context.push(prediction.action);
  }
  return steps;
}
