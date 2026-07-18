import type { TrajectorySpan } from "../capture/trajectory.js";

/**
 * Movement-policy backend — the inference half of the local-movement learning
 * subsystem (standing objective #2c/#2d). Capture/replay/export already turn
 * recorded movements into reviewed trajectories; this is the pluggable seam that
 * *learns a policy* from them and predicts the next movement, so bee-agent can
 *   (c) repeat recorded movements, and
 *   (d) generalise to new-but-related movements.
 *
 * The interface is intentionally backend-agnostic: the default
 * {@link MarkovMovementBackend} is a deterministic, dependency-free
 * variable-order Markov model that runs anywhere (cloud/CI included), while a
 * real on-device small model can implement the same contract without touching
 * callers. Snapshots are the serialisable "model file" seam for persistence and
 * for swapping a trained on-device model in later.
 */

export type MovementPredictionSource = "recall" | "backoff" | "prior";

export type MovementPrediction = {
  /** Predicted next action tool token. */
  tool: string;
  /** Representative summary for that tool (most frequent seen in training). */
  summary: string;
  /** Empirical probability of this tool given the matched context, 0..1. */
  confidence: number;
  /** Context length (in prior actions) the prediction was drawn from. */
  order: number;
  /**
   * How the prediction was derived:
   * - `recall`: matched the full available context (repeats a recorded move).
   * - `backoff`: matched a shorter prefix (generalises across related moves).
   * - `prior`: no context matched; used the global action frequency.
   */
  source: MovementPredictionSource;
};

export type MovementContext = {
  /** Recent action tool tokens, oldest → newest. */
  history: string[];
};

export type MovementPolicyBackendInfo = {
  id: string;
  kind: string;
  maxOrder: number;
  trainedTrajectories: number;
  trainedActions: number;
  vocabulary: number;
};

/** Serialisable model state — the on-device "model file" seam. */
export type MovementPolicySnapshot = {
  version: 1;
  kind: string;
  id: string;
  maxOrder: number;
  trainedTrajectories: number;
  trainedActions: number;
  /** order → contextKey → tool → count */
  transitions: Record<string, Record<string, Record<string, number>>>;
  /** tool → summary → count */
  summaries: Record<string, Record<string, number>>;
};

export interface MovementPolicyBackend {
  readonly id: string;
  /** Learn a policy from reviewed trajectories. Additive over prior fits. */
  fit(trajectories: TrajectorySpan[]): void;
  /** Predict the single next movement for a context, or undefined if untrained. */
  predict(context: MovementContext): MovementPrediction | undefined;
  /** Greedily roll out up to `steps` predicted movements (deterministic). */
  rollout(context: MovementContext, steps: number): MovementPrediction[];
  info(): MovementPolicyBackendInfo;
  snapshot(): MovementPolicySnapshot;
}

const CONTEXT_SEPARATOR = "";

export type MarkovMovementBackendOptions = {
  /** Longest action-context the model conditions on (default 3). */
  maxOrder?: number;
  /** Stable id for the backend instance (default "markov"). */
  id?: string;
};

/**
 * Deterministic variable-order Markov movement policy with Katz-style backoff.
 *
 * Training counts, for every order k in 0..maxOrder, how often each tool follows
 * each length-k action prefix. Prediction takes the highest order whose prefix
 * matches the tail of the context and returns the most frequent continuation;
 * when the full context is unseen it backs off to shorter prefixes, and finally
 * to the global unigram prior — that backoff is exactly what lets it generalise
 * to related-but-unseen movement sequences. Ties break lexicographically so the
 * same inputs always yield the same output.
 */
export class MarkovMovementBackend implements MovementPolicyBackend {
  readonly id: string;
  private readonly maxOrder: number;
  private trainedTrajectories = 0;
  private trainedActions = 0;
  /** order -> contextKey -> tool -> count */
  private readonly transitions = new Map<number, Map<string, Map<string, number>>>();
  /** tool -> summary -> count */
  private readonly summaries = new Map<string, Map<string, number>>();

  constructor(options: MarkovMovementBackendOptions = {}) {
    this.id = options.id ?? "markov";
    this.maxOrder = Math.max(1, Math.floor(options.maxOrder ?? 3));
  }

  fit(trajectories: TrajectorySpan[]): void {
    for (const trajectory of trajectories) {
      const actions = reviewedActionSequence(trajectory);
      if (actions.length === 0) {
        continue;
      }
      this.trainedTrajectories += 1;
      const tokens = actions.map((action) => action.tool);
      for (const action of actions) {
        this.trainedActions += 1;
        this.observeSummary(action.tool, action.summary);
      }
      for (let index = 0; index < tokens.length; index += 1) {
        const next = tokens[index]!;
        for (let order = 0; order <= this.maxOrder; order += 1) {
          if (index - order < 0) {
            break;
          }
          const contextTokens = tokens.slice(index - order, index);
          this.observeTransition(order, contextTokens, next);
        }
      }
    }
  }

  predict(context: MovementContext): MovementPrediction | undefined {
    if (this.trainedActions === 0) {
      return undefined;
    }
    const history = context.history;
    const maxUsable = Math.min(this.maxOrder, history.length);
    for (let order = maxUsable; order >= 0; order -= 1) {
      const contextTokens = order === 0 ? [] : history.slice(history.length - order);
      const distribution = this.transitions.get(order)?.get(contextKey(contextTokens));
      const best = argmax(distribution);
      if (!best) {
        continue;
      }
      const total = sumCounts(distribution!);
      const source: MovementPredictionSource =
        order === 0 ? "prior" : order === maxUsable ? "recall" : "backoff";
      return {
        tool: best.tool,
        summary: this.representativeSummary(best.tool),
        confidence: total === 0 ? 0 : best.count / total,
        order,
        source,
      };
    }
    return undefined;
  }

  rollout(context: MovementContext, steps: number): MovementPrediction[] {
    const predictions: MovementPrediction[] = [];
    const history = [...context.history];
    for (let step = 0; step < Math.max(0, Math.floor(steps)); step += 1) {
      const prediction = this.predict({ history });
      if (!prediction) {
        break;
      }
      predictions.push(prediction);
      history.push(prediction.tool);
    }
    return predictions;
  }

  info(): MovementPolicyBackendInfo {
    return {
      id: this.id,
      kind: "markov",
      maxOrder: this.maxOrder,
      trainedTrajectories: this.trainedTrajectories,
      trainedActions: this.trainedActions,
      vocabulary: this.summaries.size,
    };
  }

  snapshot(): MovementPolicySnapshot {
    const transitions: MovementPolicySnapshot["transitions"] = {};
    for (const [order, contexts] of this.transitions) {
      const contextRecord: Record<string, Record<string, number>> = {};
      for (const [key, tools] of contexts) {
        contextRecord[key] = Object.fromEntries(tools);
      }
      transitions[String(order)] = contextRecord;
    }
    const summaries: MovementPolicySnapshot["summaries"] = {};
    for (const [tool, counts] of this.summaries) {
      summaries[tool] = Object.fromEntries(counts);
    }
    return {
      version: 1,
      kind: "markov",
      id: this.id,
      maxOrder: this.maxOrder,
      trainedTrajectories: this.trainedTrajectories,
      trainedActions: this.trainedActions,
      transitions,
      summaries,
    };
  }

  private observeTransition(order: number, contextTokens: string[], next: string): void {
    let contexts = this.transitions.get(order);
    if (!contexts) {
      contexts = new Map();
      this.transitions.set(order, contexts);
    }
    const key = contextKey(contextTokens);
    let tools = contexts.get(key);
    if (!tools) {
      tools = new Map();
      contexts.set(key, tools);
    }
    tools.set(next, (tools.get(next) ?? 0) + 1);
  }

  private observeSummary(tool: string, summary: string): void {
    let counts = this.summaries.get(tool);
    if (!counts) {
      counts = new Map();
      this.summaries.set(tool, counts);
    }
    counts.set(summary, (counts.get(summary) ?? 0) + 1);
  }

  private representativeSummary(tool: string): string {
    const best = argmax(this.summaries.get(tool));
    return best ? best.tool : tool;
  }
}

/** Rehydrate a backend from a persisted snapshot (the on-device model file seam). */
export function restoreMarkovMovementBackend(snapshot: MovementPolicySnapshot): MarkovMovementBackend {
  if (snapshot.kind !== "markov") {
    throw new Error(`unsupported movement policy snapshot kind: ${snapshot.kind}`);
  }
  const backend = new MarkovMovementBackend({ id: snapshot.id, maxOrder: snapshot.maxOrder });
  // Reach into the instance via a controlled restore to avoid re-deriving counts.
  const internal = backend as unknown as {
    trainedTrajectories: number;
    trainedActions: number;
    transitions: Map<number, Map<string, Map<string, number>>>;
    summaries: Map<string, Map<string, number>>;
  };
  internal.trainedTrajectories = snapshot.trainedTrajectories;
  internal.trainedActions = snapshot.trainedActions;
  for (const [orderKey, contexts] of Object.entries(snapshot.transitions)) {
    const order = Number(orderKey);
    const contextMap = new Map<string, Map<string, number>>();
    for (const [key, tools] of Object.entries(contexts)) {
      contextMap.set(key, new Map(Object.entries(tools)));
    }
    internal.transitions.set(order, contextMap);
  }
  for (const [tool, counts] of Object.entries(snapshot.summaries)) {
    internal.summaries.set(tool, new Map(Object.entries(counts)));
  }
  return backend;
}

function reviewedActionSequence(trajectory: TrajectorySpan): { tool: string; summary: string }[] {
  const actions = trajectory.review?.redactedActions
    ? trajectory.review.redactedActions.map((action) => ({ ts: action.ts, tool: action.tool, summary: action.summary }))
    : trajectory.actions.map((action) => ({ ts: action.ts, tool: action.tool, summary: action.summary }));
  return [...actions].sort((a, b) => a.ts - b.ts).map(({ tool, summary }) => ({ tool, summary }));
}

function contextKey(tokens: string[]): string {
  return tokens.join(CONTEXT_SEPARATOR);
}

function argmax(distribution: Map<string, number> | undefined): { tool: string; count: number } | undefined {
  if (!distribution || distribution.size === 0) {
    return undefined;
  }
  let bestTool: string | undefined;
  let bestCount = -1;
  for (const [tool, count] of distribution) {
    if (count > bestCount || (count === bestCount && (bestTool === undefined || tool < bestTool))) {
      bestTool = tool;
      bestCount = count;
    }
  }
  return bestTool === undefined ? undefined : { tool: bestTool, count: bestCount };
}

function sumCounts(distribution: Map<string, number>): number {
  let total = 0;
  for (const count of distribution.values()) {
    total += count;
  }
  return total;
}
