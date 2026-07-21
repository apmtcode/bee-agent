import type { TrajectorySpan } from "../capture/trajectory.js";

/**
 * In-process movement-policy learning subsystem.
 *
 * This is the "post-train a local model on the recorded dataset, then repeat
 * and generalize the recorded movements" half of the local-movement objective.
 * Unlike {@link LocalAppleSiliconTrainingRunner} (which emits MLX/axolotl launch
 * scripts for real on-device training), everything here runs deterministically
 * in-process so it is exercised by cloud/CI tests against synthetic event
 * streams. The model backend is pluggable via {@link registerMovementBackend}
 * so a real on-device small model can be dropped in behind the same interface.
 */

/** A context under which a movement action is taken. */
export type MovementContext = {
  /** Coarse setting the movement happened in (app id, screen, or observation source). */
  contextTag: string;
  /** Ordered labels of the actions leading up to this point, oldest first. */
  recentActions: string[];
};

/** One supervised training example: given a context, this action was taken next. */
export type MovementSample = {
  context: MovementContext;
  action: string;
};

/** A learned distribution over next-action labels for one backoff key. */
export type MovementDistribution = {
  total: number;
  counts: Record<string, number>;
};

/** A serialized, replayable movement policy — the trained "model". */
export type MovementPolicy = {
  version: 1;
  backend: string;
  /** Highest recent-action window length the model conditions on. */
  order: number;
  /** Size of vocabulary of distinct action labels seen in training. */
  actions: string[];
  sampleCount: number;
  /** Backoff-keyed transition tables (see {@link buildContextKeys}). */
  transitions: Record<string, MovementDistribution>;
};

export type MovementBackoffLevel = "exact" | "recent" | "context" | "global" | "none";

/** The result of asking a policy what to do next in a given context. */
export type MovementPrediction = {
  /** Most-likely next action, or undefined if the policy is empty. */
  action: string | undefined;
  /** Probability of {@link action} under the distribution that was consulted. */
  confidence: number;
  /**
   * Which backoff level answered: "exact"/"recent" means the exact recorded
   * sequence was memorized; "context"/"global" means the policy generalized
   * from broader statistics to a related-but-unseen situation.
   */
  backoff: MovementBackoffLevel;
  /** Full ranked candidate list (deterministic: prob desc, then label asc). */
  candidates: { action: string; probability: number }[];
};

export type MovementPolicyTrainOptions = {
  /** Recent-action window length to condition on (>=0). Defaults to 2. */
  order?: number;
};

/** Pluggable backend contract: train a policy from samples, infer next action. */
export interface MovementPolicyBackend {
  readonly name: string;
  train(samples: MovementSample[], options?: MovementPolicyTrainOptions): MovementPolicy;
  predict(policy: MovementPolicy, context: MovementContext): MovementPrediction;
}

const GLOBAL_KEY = "*";

/**
 * Build the ordered list of backoff context keys for a context, longest match
 * first. Training increments counts at every key so each fallback distribution
 * exists; prediction walks the same list and stops at the first populated one.
 */
export function buildContextKeys(context: MovementContext, order: number): { key: string; level: MovementBackoffLevel }[] {
  const keys: { key: string; level: MovementBackoffLevel }[] = [];
  const tag = encode(context.contextTag);
  const window = context.recentActions.slice(-order).map(encode);
  for (let length = Math.min(order, window.length); length >= 1; length -= 1) {
    const seq = window.slice(window.length - length).join(",");
    keys.push({ key: `L${length}|${tag}|${seq}`, level: length === order ? "exact" : "recent" });
  }
  keys.push({ key: `C|${tag}`, level: "context" });
  keys.push({ key: GLOBAL_KEY, level: "global" });
  return keys;
}

/**
 * A deterministic reference backend: a variable-order Markov model over action
 * labels with Katz-style backoff. Memorizes exact recorded sequences and
 * generalizes to unseen-but-related contexts by falling back to context-tag
 * and global priors. No randomness — ties broken lexically — so tests are
 * reproducible and the model doubles as a self-documenting dataset baseline.
 */
export class MarkovMovementBackend implements MovementPolicyBackend {
  readonly name = "markov";

  train(samples: MovementSample[], options: MovementPolicyTrainOptions = {}): MovementPolicy {
    const order = Math.max(0, options.order ?? 2);
    const transitions: Record<string, MovementDistribution> = {};
    const vocabulary = new Set<string>();

    for (const sample of samples) {
      const action = sample.action;
      vocabulary.add(action);
      for (const { key } of buildContextKeys(sample.context, order)) {
        const distribution = (transitions[key] ??= { total: 0, counts: {} });
        distribution.counts[action] = (distribution.counts[action] ?? 0) + 1;
        distribution.total += 1;
      }
    }

    return {
      version: 1,
      backend: this.name,
      order,
      actions: [...vocabulary].sort(),
      sampleCount: samples.length,
      transitions,
    };
  }

  predict(policy: MovementPolicy, context: MovementContext): MovementPrediction {
    for (const { key, level } of buildContextKeys(context, policy.order)) {
      const distribution = policy.transitions[key];
      if (distribution && distribution.total > 0) {
        return rankDistribution(distribution, level);
      }
    }
    return { action: undefined, confidence: 0, backoff: "none", candidates: [] };
  }
}

function rankDistribution(distribution: MovementDistribution, level: MovementBackoffLevel): MovementPrediction {
  const candidates = Object.entries(distribution.counts)
    .map(([action, count]) => ({ action, probability: count / distribution.total }))
    .sort((a, b) => (b.probability - a.probability) || (a.action < b.action ? -1 : a.action > b.action ? 1 : 0));
  const top = candidates[0];
  return {
    action: top?.action,
    confidence: top?.probability ?? 0,
    backoff: top ? level : "none",
    candidates,
  };
}

function encode(value: string): string {
  return value.replaceAll("\\", "\\\\").replaceAll("|", "\\p").replaceAll(",", "\\c");
}

const backendRegistry = new Map<string, () => MovementPolicyBackend>();

/** Register (or override) a named, pluggable movement-policy backend factory. */
export function registerMovementBackend(name: string, factory: () => MovementPolicyBackend): void {
  backendRegistry.set(name, factory);
}

/** Instantiate a registered backend by name. Throws on unknown backends. */
export function createMovementBackend(name: string): MovementPolicyBackend {
  const factory = backendRegistry.get(name);
  if (!factory) {
    throw new Error(`unknown movement backend: ${name} (registered: ${[...backendRegistry.keys()].join(", ") || "none"})`);
  }
  return factory();
}

/** Names of all registered backends, sorted. */
export function listMovementBackends(): string[] {
  return [...backendRegistry.keys()].sort();
}

registerMovementBackend("markov", () => new MarkovMovementBackend());

export type ExtractMovementSamplesOptions = {
  /**
   * Derive the context tag for a trajectory. Defaults to the summary of the
   * trajectory's first observation (falling back to its capture tier), which is
   * stable across the reviewed/redacted export shape.
   */
  contextTagOf?: (trajectory: TrajectorySpan) => string;
  /** Derive a canonical action label. Defaults to the action summary. */
  labelOf?: (action: TrajectorySpan["actions"][number]) => string;
  /** Recent-action window fed into each sample's context. Defaults to 2. */
  order?: number;
};

/**
 * Flatten reviewed movement trajectories into ordered supervised samples. Each
 * trajectory becomes a sequence: sample i predicts action i from the context
 * tag plus the labels of the preceding `order` actions. This is the bridge from
 * the recorded dataset ({@link TrajectorySpan}) to the training interface.
 */
export function extractMovementSamples(
  trajectories: TrajectorySpan[],
  options: ExtractMovementSamplesOptions = {},
): MovementSample[] {
  const order = Math.max(0, options.order ?? 2);
  const contextTagOf = options.contextTagOf ?? defaultContextTag;
  const labelOf = options.labelOf ?? ((action) => action.summary);
  const samples: MovementSample[] = [];

  for (const trajectory of trajectories) {
    const tag = contextTagOf(trajectory);
    const actions = [...trajectory.actions].sort((a, b) => a.ts - b.ts);
    const history: string[] = [];
    for (const action of actions) {
      samples.push({
        context: { contextTag: tag, recentActions: history.slice(-order) },
        action: labelOf(action),
      });
      history.push(labelOf(action));
    }
  }
  return samples;
}

function defaultContextTag(trajectory: TrajectorySpan): string {
  return trajectory.observations[0]?.summary ?? `tier:${trajectory.captureTier}`;
}

/**
 * Roll a policy forward from a starting context, greedily predicting a sequence
 * of actions — the "repeat/continue the recorded movement" replay path.
 */
export function rolloutMovementPolicy(
  backend: MovementPolicyBackend,
  policy: MovementPolicy,
  start: MovementContext,
  steps: number,
): { actions: string[]; predictions: MovementPrediction[] } {
  const history = [...start.recentActions];
  const actions: string[] = [];
  const predictions: MovementPrediction[] = [];
  for (let step = 0; step < steps; step += 1) {
    const prediction = backend.predict(policy, {
      contextTag: start.contextTag,
      recentActions: history.slice(-policy.order),
    });
    predictions.push(prediction);
    if (!prediction.action) {
      break;
    }
    actions.push(prediction.action);
    history.push(prediction.action);
  }
  return { actions, predictions };
}

export type MovementEvaluation = {
  /** Number of predictions scored. */
  total: number;
  /** Predictions whose top-1 action matched the recorded next action. */
  correct: number;
  /** correct / total, or 0 when there is nothing to score. */
  accuracy: number;
  /** Correct predictions that came from a memorized exact/recent sequence. */
  memorized: number;
  /** Correct predictions that required generalizing via context/global backoff. */
  generalized: number;
};

/**
 * Generalization eval harness: score a trained policy's top-1 next-action
 * accuracy against held-out trajectories, splitting hits into memorized vs.
 * generalized so replay fidelity and generalization can be tracked separately.
 */
export function evaluateMovementPolicy(
  backend: MovementPolicyBackend,
  policy: MovementPolicy,
  heldOut: MovementSample[],
): MovementEvaluation {
  let correct = 0;
  let memorized = 0;
  let generalized = 0;
  for (const sample of heldOut) {
    const prediction = backend.predict(policy, sample.context);
    if (prediction.action === sample.action) {
      correct += 1;
      if (prediction.backoff === "exact" || prediction.backoff === "recent") {
        memorized += 1;
      } else {
        generalized += 1;
      }
    }
  }
  const total = heldOut.length;
  return { total, correct, accuracy: total === 0 ? 0 : correct / total, memorized, generalized };
}
