import type { ReplayManifest, ReplayTimelineEvent } from "../capture/replay.js";
import type { TrajectorySpan } from "../capture/trajectory.js";

/**
 * Local-movement learning subsystem — pluggable model backend.
 *
 * This module implements pieces (c) and (d) of the movement objective in a
 * cloud-safe, deterministic way: post-train a *local* model on a recorded
 * movement dataset so it can (c) repeat the recorded movements and
 * (d) generalize to new-but-related movements.
 *
 * The real on-device training (mlx/axolotl, see {@link ./runner.ts}) executes
 * when the user runs bee-agent locally. Here we provide the backend *interface*
 * plus a deterministic in-process backend (an order-k Markov model with
 * stupid-backoff) that requires no GPU and no randomness, so the whole
 * capture → dataset → train → infer → generalize loop is exercisable in tests.
 * Additional backends (a real small local model) plug in via
 * {@link MovementModelRegistry} without touching call sites.
 */

/** A discrete, replayable movement symbol (e.g. `device:tapped submit`). */
export type MovementToken = string;

/** Start-of-sequence sentinel. Never collides with a real action token. */
export const MOVEMENT_START: MovementToken = "start";
/** End-of-sequence sentinel emitted after the last recorded action. */
export const MOVEMENT_END: MovementToken = "end";

/** One training example: an ordered run of movement tokens for a trajectory. */
export type MovementSample = {
  trajectoryId: string;
  tokens: MovementToken[];
};

export type MovementPrediction = {
  token: MovementToken;
  /** Empirical probability of `token` given the matched context (0..1). */
  confidence: number;
  /** Length of the context that produced the prediction (backoff depth). */
  order: number;
};

export type MovementModelMetrics = {
  backendId: string;
  order: number;
  sampleCount: number;
  tokenCount: number;
  vocabularySize: number;
};

export type MovementPolicySnapshot = {
  version: 1;
  backendId: string;
  order: number;
  metrics: MovementModelMetrics;
  transitions: Array<{ context: string; nextCounts: Array<[MovementToken, number]> }>;
};

/** A trained, serializable movement policy. */
export interface MovementPolicy {
  readonly backendId: string;
  readonly order: number;
  readonly metrics: MovementModelMetrics;
  /** Best next token given the recent context, or undefined if the model is empty. */
  predictNext(context: MovementToken[]): MovementPrediction | undefined;
  /** Roll the policy forward from `seed` until END or `maxSteps` real tokens. */
  generate(seed?: MovementToken[], maxSteps?: number): MovementToken[];
  toJSON(): MovementPolicySnapshot;
}

export type TrainMovementOptions = {
  /** Maximum context length for the highest-order model (default 3). */
  order?: number;
};

/** A pluggable local-model backend. Real backends implement the same shape. */
export interface MovementModelBackend {
  readonly id: string;
  train(samples: MovementSample[], options?: TrainMovementOptions): Promise<MovementPolicy>;
}

const CONTEXT_SEPARATOR = "";
const DEFAULT_ORDER = 3;

/**
 * Deterministic order-k Markov backend with stupid-backoff.
 *
 * Given identical input it always produces identical predictions (ties broken
 * by count, then token order), so it is safe to run in CI where
 * `Math.random`/`Date.now` are unavailable. High-order contexts reproduce
 * recorded runs exactly (repeat); backoff to shorter contexts yields plausible
 * continuations for unseen prefixes (generalize).
 */
export class MarkovMovementBackend implements MovementModelBackend {
  readonly id = "markov";

  async train(samples: MovementSample[], options?: TrainMovementOptions): Promise<MovementPolicy> {
    const order = Math.max(1, Math.trunc(options?.order ?? DEFAULT_ORDER));
    // context string -> (next token -> count)
    const transitions = new Map<string, Map<MovementToken, number>>();
    const vocabulary = new Set<MovementToken>();
    let tokenCount = 0;

    for (const sample of samples) {
      const framed = [MOVEMENT_START, ...sample.tokens, MOVEMENT_END];
      for (let i = 1; i < framed.length; i += 1) {
        const next = framed[i]!;
        if (next !== MOVEMENT_START) {
          vocabulary.add(next);
          if (next !== MOVEMENT_END) {
            tokenCount += 1;
          }
        }
        // Record this transition at every context length 1..order (backoff table).
        for (let k = 1; k <= order; k += 1) {
          if (i - k < 0) {
            break;
          }
          const context = framed.slice(i - k, i).join(CONTEXT_SEPARATOR);
          let counts = transitions.get(context);
          if (!counts) {
            counts = new Map<MovementToken, number>();
            transitions.set(context, counts);
          }
          counts.set(next, (counts.get(next) ?? 0) + 1);
        }
      }
    }

    const metrics: MovementModelMetrics = {
      backendId: this.id,
      order,
      sampleCount: samples.length,
      tokenCount,
      vocabularySize: vocabulary.size,
    };

    return new MarkovMovementPolicy(this.id, order, transitions, metrics);
  }
}

class MarkovMovementPolicy implements MovementPolicy {
  constructor(
    readonly backendId: string,
    readonly order: number,
    private readonly transitions: Map<string, Map<MovementToken, number>>,
    readonly metrics: MovementModelMetrics,
  ) {}

  predictNext(context: MovementToken[]): MovementPrediction | undefined {
    // Stupid-backoff: try the longest available context, shorten on a miss.
    const bounded = context.slice(-this.order);
    for (let k = bounded.length; k >= 1; k -= 1) {
      const key = bounded.slice(bounded.length - k).join(CONTEXT_SEPARATOR);
      const counts = this.transitions.get(key);
      if (counts && counts.size > 0) {
        return pickBest(counts, k);
      }
    }
    return undefined;
  }

  generate(seed: MovementToken[] = [], maxSteps = 64): MovementToken[] {
    const emitted: MovementToken[] = [];
    const context: MovementToken[] = [MOVEMENT_START, ...seed];
    const output: MovementToken[] = [...seed];
    for (let step = 0; step < maxSteps; step += 1) {
      const prediction = this.predictNext(context);
      if (!prediction || prediction.token === MOVEMENT_END) {
        break;
      }
      emitted.push(prediction.token);
      output.push(prediction.token);
      context.push(prediction.token);
    }
    return [...seed, ...emitted];
  }

  toJSON(): MovementPolicySnapshot {
    const transitions = [...this.transitions.entries()]
      .map(([context, counts]) => ({
        context,
        nextCounts: [...counts.entries()].sort(compareCountEntries),
      }))
      .sort((a, b) => (a.context < b.context ? -1 : a.context > b.context ? 1 : 0));
    return {
      version: 1,
      backendId: this.backendId,
      order: this.order,
      metrics: this.metrics,
      transitions,
    };
  }
}

function pickBest(counts: Map<MovementToken, number>, order: number): MovementPrediction {
  let total = 0;
  let best: MovementToken | undefined;
  let bestCount = -1;
  for (const [token, count] of counts) {
    total += count;
    if (count > bestCount || (count === bestCount && best !== undefined && token < best)) {
      best = token;
      bestCount = count;
    }
  }
  return { token: best!, confidence: total > 0 ? bestCount / total : 0, order };
}

function compareCountEntries(a: [MovementToken, number], b: [MovementToken, number]): number {
  if (a[1] !== b[1]) {
    return b[1] - a[1];
  }
  return a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0;
}

/** Registry so backends are swappable without touching call sites. */
export class MovementModelRegistry {
  private readonly backends = new Map<string, MovementModelBackend>();

  register(backend: MovementModelBackend): this {
    this.backends.set(backend.id, backend);
    return this;
  }

  has(id: string): boolean {
    return this.backends.has(id);
  }

  get(id: string): MovementModelBackend {
    const backend = this.backends.get(id);
    if (!backend) {
      throw new Error(`unknown movement-model backend: ${id}`);
    }
    return backend;
  }

  list(): string[] {
    return [...this.backends.keys()].sort();
  }
}

/** Registry seeded with the deterministic in-process backend. */
export function createDefaultMovementModelRegistry(): MovementModelRegistry {
  return new MovementModelRegistry().register(new MarkovMovementBackend());
}

/** Encode a recorded action into a stable, comparable movement token. */
export function encodeActionToken(tool: string, summary: string): MovementToken {
  const normalizedTool = tool.trim().toLowerCase();
  const normalizedSummary = summary.trim().replace(/\s+/g, " ").toLowerCase();
  return `${normalizedTool}:${normalizedSummary}`;
}

/** Build training samples from trajectory action streams. */
export function extractMovementSamplesFromTrajectories(trajectories: TrajectorySpan[]): MovementSample[] {
  return trajectories
    .map((trajectory) => ({
      trajectoryId: trajectory.id,
      tokens: [...trajectory.actions]
        .sort((a, b) => a.ts - b.ts)
        .map((action) => encodeActionToken(action.tool, action.summary)),
    }))
    .filter((sample) => sample.tokens.length > 0);
}

type ReplayLike = Pick<ReplayManifest, "trajectoryIds" | "events">;

/** Build training samples from replay manifests (action events, per trajectory). */
export function extractMovementSamplesFromReplays(replays: ReplayLike[]): MovementSample[] {
  const byTrajectory = new Map<string, Array<Extract<ReplayTimelineEvent, { kind: "action" }>>>();
  for (const replay of replays) {
    for (const event of replay.events) {
      if (event.kind !== "action") {
        continue;
      }
      const bucket = byTrajectory.get(event.trajectoryId);
      if (bucket) {
        bucket.push(event);
      } else {
        byTrajectory.set(event.trajectoryId, [event]);
      }
    }
  }
  return [...byTrajectory.entries()]
    .map(([trajectoryId, events]) => ({
      trajectoryId,
      tokens: events.sort((a, b) => a.ts - b.ts).map((event) => encodeActionToken(event.tool, event.summary)),
    }))
    .filter((sample) => sample.tokens.length > 0);
}

export type MovementEvaluation = {
  sampleCount: number;
  /** Teacher-forced next-token accuracy across all held-out transitions. */
  nextTokenAccuracy: number;
  /** Fraction of held-out samples the policy reproduces exactly from its seed. */
  exactReplayRate: number;
};

/**
 * Generalization eval harness: measure how well a trained policy predicts and
 * reproduces held-out (but related) movement samples. `seedLength` tokens of
 * each sample are given as context; the rest must be predicted.
 */
export function evaluateMovementPolicy(
  policy: MovementPolicy,
  heldOut: MovementSample[],
  seedLength = 1,
): MovementEvaluation {
  let correct = 0;
  let total = 0;
  let exact = 0;
  let evaluated = 0;

  for (const sample of heldOut) {
    if (sample.tokens.length === 0) {
      continue;
    }
    evaluated += 1;

    // Teacher-forced accuracy: predict each token from its true prefix.
    for (let i = 0; i < sample.tokens.length; i += 1) {
      const context = [MOVEMENT_START, ...sample.tokens.slice(0, i)];
      const prediction = policy.predictNext(context);
      total += 1;
      if (prediction?.token === sample.tokens[i]) {
        correct += 1;
      }
    }

    // Free-running replay from a short seed.
    const seed = sample.tokens.slice(0, Math.min(seedLength, sample.tokens.length));
    const generated = policy.generate(seed, sample.tokens.length + 4);
    if (tokensEqual(generated, sample.tokens)) {
      exact += 1;
    }
  }

  return {
    sampleCount: evaluated,
    nextTokenAccuracy: total > 0 ? correct / total : 0,
    exactReplayRate: evaluated > 0 ? exact / evaluated : 0,
  };
}

function tokensEqual(a: MovementToken[], b: MovementToken[]): boolean {
  if (a.length !== b.length) {
    return false;
  }
  return a.every((token, index) => token === b[index]);
}
