import type { DeviceGestureKind } from "../capture/device-adapter.js";

/**
 * Local-movement learning: pluggable model backend.
 *
 * The training subsystem's {@link LocalAppleSiliconTrainingRunner} emits launch
 * scripts for *external* on-device trainers (mlx / axolotl). Those cannot run in
 * the cloud, so this module adds an **in-process, pluggable model backend** that
 * can actually (c) post-train on a recorded movement dataset and (d) generalize
 * to new-but-related movements — fully exercisable with synthetic data in CI.
 *
 * The shipped {@link DeterministicNgramBackend} is a variable-order Markov model
 * with stupid-backoff: it learns next-movement distributions from the dataset and
 * generalizes to unseen contexts by backing off to shorter contexts. It is 100%
 * deterministic (no RNG, stable tie-breaks) so tests are reproducible. Register a
 * real small on-device model behind the same {@link MovementModelBackend} seam.
 */

// ---------------------------------------------------------------------------
// Movement event schema (the tokens the model learns over)
// ---------------------------------------------------------------------------

export type MovementDirection = "up" | "down" | "left" | "right";

/** A single normalized movement/UI action — the atom of the dataset. */
export type MovementStep = {
  gesture: DeviceGestureKind;
  target?: string;
  direction?: MovementDirection;
  appId?: string;
};

/** An ordered run of movement steps captured from one task. */
export type MovementSequence = {
  id: string;
  appId: string;
  steps: MovementStep[];
};

/** A compact discrete label for one step, e.g. `tap:submit-button` or `swipe:down`. */
export type MovementToken = string;

/** Sentinel tokens marking sequence boundaries so the model can learn starts/ends. */
export const MOVEMENT_START_TOKEN = "start";
export const MOVEMENT_END_TOKEN = "end";

const CONTEXT_SEPARATOR = "";

export function tokenizeStep(step: MovementStep): MovementToken {
  const parts: string[] = [step.gesture];
  if (step.target) {
    parts.push(step.target);
  }
  if (step.direction) {
    parts.push(step.direction);
  }
  return parts.join(":");
}

export function tokenizeSequence(sequence: MovementSequence): MovementToken[] {
  return sequence.steps.map(tokenizeStep);
}

// ---------------------------------------------------------------------------
// Backend interface (the pluggable seam)
// ---------------------------------------------------------------------------

export type MovementModelConfig = {
  /** Maximum context length (n-gram order). Higher = more specific, less general. */
  order: number;
};

export const DEFAULT_MOVEMENT_MODEL_CONFIG: MovementModelConfig = { order: 3 };

export type MovementDistributionEntry = {
  token: MovementToken;
  probability: number;
};

export type MovementPrediction = {
  /** Most-likely next token, or `null` when the model predicts sequence end / has no data. */
  token: MovementToken | null;
  /** Probability mass of the top-1 token (0 when unknown). */
  confidence: number;
  /** Full next-token distribution, sorted by probability desc then token asc. */
  distribution: MovementDistributionEntry[];
  /**
   * The context order that actually produced the prediction. Equals the full
   * requested order for memorized contexts; a smaller value means the model
   * *generalized* by backing off to a shorter context (-1 = empty model).
   */
  backoffOrder: number;
};

/** Serializable snapshot so a trained model can be persisted and reloaded. */
export type MovementModelSnapshot = {
  version: 1;
  backend: string;
  config: MovementModelConfig;
  vocabulary: MovementToken[];
  /** contextKey -> (nextToken -> count) */
  counts: Record<string, Record<MovementToken, number>>;
};

export type MovementGenerateOptions = {
  /** Prior tokens to condition on (defaults to a fresh sequence start). */
  seed?: MovementToken[];
  /** Hard cap on generated steps (excludes boundary sentinels). */
  maxSteps?: number;
};

export interface TrainedMovementModel {
  readonly backend: string;
  readonly config: MovementModelConfig;
  readonly vocabulary: MovementToken[];
  predictNext(context: MovementToken[]): MovementPrediction;
  /** Roll out a new movement sequence (generalization). Deterministic: always top-1. */
  generate(options?: MovementGenerateOptions): MovementToken[];
  serialize(): MovementModelSnapshot;
}

export interface MovementModelBackend {
  readonly name: string;
  train(dataset: MovementSequence[], config?: Partial<MovementModelConfig>): TrainedMovementModel;
  load(snapshot: MovementModelSnapshot): TrainedMovementModel;
}

// ---------------------------------------------------------------------------
// Deterministic variable-order n-gram backend (the CI-safe default)
// ---------------------------------------------------------------------------

export const DETERMINISTIC_NGRAM_BACKEND = "deterministic-ngram";

function contextKey(tokens: MovementToken[]): string {
  return tokens.join(CONTEXT_SEPARATOR);
}

function normalizeConfig(config?: Partial<MovementModelConfig>): MovementModelConfig {
  const order = Math.max(1, Math.trunc(config?.order ?? DEFAULT_MOVEMENT_MODEL_CONFIG.order));
  return { order };
}

class NgramMovementModel implements TrainedMovementModel {
  readonly backend = DETERMINISTIC_NGRAM_BACKEND;

  constructor(
    readonly config: MovementModelConfig,
    private readonly counts: Map<string, Map<MovementToken, number>>,
    readonly vocabulary: MovementToken[],
  ) {}

  predictNext(context: MovementToken[]): MovementPrediction {
    const trimmed = context.slice(-this.config.order);
    for (let k = trimmed.length; k >= 0; k -= 1) {
      const key = contextKey(trimmed.slice(trimmed.length - k));
      const bucket = this.counts.get(key);
      if (bucket && bucket.size > 0) {
        return this.buildPrediction(bucket, k);
      }
    }
    return { token: null, confidence: 0, distribution: [], backoffOrder: -1 };
  }

  private buildPrediction(bucket: Map<MovementToken, number>, backoffOrder: number): MovementPrediction {
    let total = 0;
    for (const count of bucket.values()) {
      total += count;
    }
    const distribution = [...bucket.entries()]
      .map(([token, count]) => ({ token, probability: count / total }))
      .sort((a, b) => (b.probability - a.probability) || (a.token < b.token ? -1 : a.token > b.token ? 1 : 0));

    const top = distribution[0];
    const isEnd = top.token === MOVEMENT_END_TOKEN;
    const visible = distribution.filter((entry) => entry.token !== MOVEMENT_END_TOKEN);
    return {
      token: isEnd ? null : top.token,
      confidence: top.probability,
      distribution: visible,
      backoffOrder,
    };
  }

  generate(options: MovementGenerateOptions = {}): MovementToken[] {
    const maxSteps = options.maxSteps ?? 64;
    const context: MovementToken[] = options.seed ? [...options.seed] : [MOVEMENT_START_TOKEN];
    const output: MovementToken[] = [];

    for (let step = 0; step < maxSteps; step += 1) {
      const trimmed = context.slice(-this.config.order);
      const next = this.rawNext(trimmed);
      if (next === null || next === MOVEMENT_END_TOKEN) {
        break;
      }
      output.push(next);
      context.push(next);
    }
    return output;
  }

  /** Like predictNext but returns the raw top-1 token including the END sentinel. */
  private rawNext(context: MovementToken[]): MovementToken | null {
    for (let k = context.length; k >= 0; k -= 1) {
      const bucket = this.counts.get(contextKey(context.slice(context.length - k)));
      if (bucket && bucket.size > 0) {
        return [...bucket.entries()].sort(
          (a, b) => (b[1] - a[1]) || (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0),
        )[0][0];
      }
    }
    return null;
  }

  serialize(): MovementModelSnapshot {
    const counts: Record<string, Record<MovementToken, number>> = {};
    for (const [key, bucket] of this.counts.entries()) {
      counts[key] = Object.fromEntries(bucket.entries());
    }
    return {
      version: 1,
      backend: this.backend,
      config: this.config,
      vocabulary: [...this.vocabulary],
      counts,
    };
  }
}

export class DeterministicNgramBackend implements MovementModelBackend {
  readonly name = DETERMINISTIC_NGRAM_BACKEND;

  train(dataset: MovementSequence[], config?: Partial<MovementModelConfig>): TrainedMovementModel {
    const resolved = normalizeConfig(config);
    const counts = new Map<string, Map<MovementToken, number>>();
    const vocab = new Set<MovementToken>();

    for (const sequence of dataset) {
      const tokens = [MOVEMENT_START_TOKEN, ...tokenizeSequence(sequence), MOVEMENT_END_TOKEN];
      for (const token of tokens) {
        if (token !== MOVEMENT_START_TOKEN && token !== MOVEMENT_END_TOKEN) {
          vocab.add(token);
        }
      }
      for (let i = 1; i < tokens.length; i += 1) {
        const next = tokens[i];
        const maxContext = Math.min(resolved.order, i);
        for (let k = 0; k <= maxContext; k += 1) {
          const key = contextKey(tokens.slice(i - k, i));
          const bucket = counts.get(key) ?? new Map<MovementToken, number>();
          bucket.set(next, (bucket.get(next) ?? 0) + 1);
          counts.set(key, bucket);
        }
      }
    }

    return new NgramMovementModel(resolved, counts, [...vocab].sort());
  }

  load(snapshot: MovementModelSnapshot): TrainedMovementModel {
    const counts = new Map<string, Map<MovementToken, number>>();
    for (const [key, bucket] of Object.entries(snapshot.counts)) {
      counts.set(key, new Map(Object.entries(bucket)));
    }
    return new NgramMovementModel(normalizeConfig(snapshot.config), counts, [...snapshot.vocabulary]);
  }
}

// ---------------------------------------------------------------------------
// Backend registry (pluggability)
// ---------------------------------------------------------------------------

const registry = new Map<string, MovementModelBackend>();

export function registerMovementBackend(backend: MovementModelBackend): void {
  registry.set(backend.name, backend);
}

export function getMovementBackend(name: string): MovementModelBackend | undefined {
  return registry.get(name);
}

export function listMovementBackends(): string[] {
  return [...registry.keys()].sort();
}

registerMovementBackend(new DeterministicNgramBackend());
