/**
 * Pluggable local-movement model backend.
 *
 * This is the learning half of the local-movement subsystem (objective 2c/2d):
 * given a dataset of recorded movement sequences (derived from approved capture
 * trajectories), a backend trains a model that can (a) *repeat* recorded
 * movements and (b) *generalize* to new-but-related movements.
 *
 * bee-agent runs in the cloud and cannot execute a real on-device trainer
 * (MLX / axolotl). So the default backend here is a fully in-process,
 * deterministic variable-order Markov model with stupid-backoff — it genuinely
 * learns transition structure from the dataset and generalizes via backoff,
 * yet needs no GPU, no network, and no randomness. Real on-device backends plug
 * in behind the same {@link MovementModelBackend} interface; the registry
 * exposes documented seams (`mlx`, `axolotl`) that fail loudly in environments
 * that lack the native runtime instead of pretending to train.
 */

import type { TrajectoryAction, TrajectorySpan } from "../capture/trajectory.js";

/** A single normalized movement primitive (e.g. `device:swipe:down`). */
export type MovementToken = string;

/** An ordered run of movement tokens produced by one trajectory. */
export type MovementSequence = {
  trajectoryId: string;
  tokens: MovementToken[];
};

export type MovementDataset = {
  version: 1;
  sequences: MovementSequence[];
};

export type MovementPrediction = {
  /** Predicted next token, or `undefined` when the model has no information. */
  token: MovementToken | undefined;
  /** Estimated probability of {@link token} under the model. */
  probability: number;
  /** Context order actually used after backoff (0 = unigram/prior). */
  order: number;
  /** True when a lower-than-requested order was used (i.e. generalization). */
  backedOff: boolean;
};

export type SerializedMovementModel = {
  backend: string;
  order: number;
  vocabulary: MovementToken[];
  /** Per-order transition counts keyed by `context -> {token: count}`. */
  counts: Record<string, Record<MovementToken, number>>;
};

/** A trained model. Deterministic backends serialize/replay without a runtime. */
export interface TrainedMovementModel {
  readonly backend: string;
  readonly order: number;
  readonly vocabulary: readonly MovementToken[];
  /** Predict the most likely next token given a (possibly long) context. */
  predictNext(context: MovementToken[]): MovementPrediction;
  /** Roll the model forward from a seed, greedily, up to `maxSteps` tokens. */
  generate(seed: MovementToken[], maxSteps: number): MovementToken[];
  /** Mean per-token log-probability of a sequence (higher = better fit). */
  score(sequence: MovementToken[]): number;
  serialize(): SerializedMovementModel;
}

export type MovementTrainingOptions = {
  /** Maximum Markov order (context length). Clamped to >= 1. */
  order?: number;
};

/** The seam every model backend implements — local, mock, or real on-device. */
export interface MovementModelBackend {
  readonly name: string;
  train(dataset: MovementDataset, options?: MovementTrainingOptions): TrainedMovementModel;
}

const SEQUENCE_START: MovementToken = "<s>";

/**
 * Normalize a captured action into a stable movement token. Structured gesture
 * metadata (kind/direction/target) is preferred so that semantically identical
 * movements collapse to the same token regardless of summary wording; free-text
 * summaries fall back to a slug so nothing is silently dropped.
 */
export function tokenizeAction(action: Pick<TrajectoryAction, "tool" | "summary" | "metadata">): MovementToken {
  const tool = slug(action.tool) || "action";
  const metadata = action.metadata ?? {};
  const gesture = typeof metadata.gesture === "string" ? slug(metadata.gesture) : "";
  const direction = typeof metadata.direction === "string" ? slug(metadata.direction) : "";
  const key = typeof metadata.key === "string" ? slug(metadata.key) : "";
  const parts = [tool];
  if (gesture) {
    parts.push(gesture);
  }
  if (direction) {
    parts.push(direction);
  }
  if (key) {
    parts.push(key);
  }
  if (parts.length === 1) {
    const summary = slug(action.summary);
    if (summary) {
      parts.push(summary);
    }
  }
  return parts.join(":");
}

/** Build a training dataset from trajectory spans (uses redacted actions when reviewed). */
export function buildMovementDataset(trajectories: TrajectorySpan[]): MovementDataset {
  const sequences: MovementSequence[] = [];
  for (const trajectory of trajectories) {
    const redacted = trajectory.review?.redactedActions;
    const source: Array<Pick<TrajectoryAction, "tool" | "summary" | "metadata">> = redacted
      ? redacted.map((action) => ({ tool: action.tool, summary: action.summary }))
      : trajectory.actions;
    const tokens = source.map((action) => tokenizeAction(action)).filter((token) => token.length > 0);
    if (tokens.length > 0) {
      sequences.push({ trajectoryId: trajectory.id, tokens });
    }
  }
  return { version: 1, sequences };
}

class MarkovMovementModel implements TrainedMovementModel {
  readonly backend = "markov";

  constructor(
    readonly order: number,
    private readonly counts: Map<string, Map<MovementToken, number>>,
    readonly vocabulary: readonly MovementToken[],
  ) {}

  predictNext(context: MovementToken[]): MovementPrediction {
    for (let used = Math.min(this.order, context.length); used >= 0; used -= 1) {
      const key = contextKey(context.slice(context.length - used));
      const table = this.counts.get(key);
      if (!table || table.size === 0) {
        continue;
      }
      let total = 0;
      let best: MovementToken | undefined;
      let bestCount = -1;
      // Sort keys so ties resolve deterministically (lexicographic).
      for (const token of [...table.keys()].sort()) {
        const count = table.get(token) ?? 0;
        total += count;
        if (count > bestCount) {
          bestCount = count;
          best = token;
        }
      }
      if (best !== undefined && total > 0) {
        return {
          token: best,
          probability: bestCount / total,
          order: used,
          backedOff: used < Math.min(this.order, context.length),
        };
      }
    }
    return { token: undefined, probability: 0, order: 0, backedOff: context.length > 0 };
  }

  generate(seed: MovementToken[], maxSteps: number): MovementToken[] {
    const produced: MovementToken[] = [];
    const context = [SEQUENCE_START, ...seed];
    for (let step = 0; step < Math.max(0, maxSteps); step += 1) {
      const prediction = this.predictNext(context);
      if (prediction.token === undefined) {
        break;
      }
      produced.push(prediction.token);
      context.push(prediction.token);
    }
    return produced;
  }

  score(sequence: MovementToken[]): number {
    if (sequence.length === 0) {
      return 0;
    }
    const context = [SEQUENCE_START];
    let logSum = 0;
    const floor = 1 / (this.vocabulary.length + 1);
    for (const token of sequence) {
      const probability = this.probabilityOf(context, token);
      logSum += Math.log(Math.max(probability, floor * 1e-3));
      context.push(token);
    }
    return logSum / sequence.length;
  }

  private probabilityOf(context: MovementToken[], token: MovementToken): number {
    const backoff = 0.4;
    for (let used = Math.min(this.order, context.length); used >= 0; used -= 1) {
      const key = contextKey(context.slice(context.length - used));
      const table = this.counts.get(key);
      if (!table || table.size === 0) {
        continue;
      }
      let total = 0;
      for (const count of table.values()) {
        total += count;
      }
      const count = table.get(token) ?? 0;
      const discount = Math.pow(backoff, Math.min(this.order, context.length) - used);
      if (count > 0) {
        return (count / total) * discount;
      }
    }
    return 0;
  }

  serialize(): SerializedMovementModel {
    const counts: Record<string, Record<MovementToken, number>> = {};
    for (const [key, table] of this.counts) {
      counts[key] = Object.fromEntries([...table.entries()].sort((a, b) => a[0].localeCompare(b[0])));
    }
    return {
      backend: this.backend,
      order: this.order,
      vocabulary: [...this.vocabulary].sort(),
      counts,
    };
  }
}

/**
 * Deterministic, dependency-free variable-order Markov backend with
 * stupid-backoff. This is the default mock/local backend: it learns real
 * transition structure and generalizes to unseen contexts via backoff, so
 * capture→dataset→train→infer round-trips are fully testable in the cloud.
 */
export class MarkovMovementBackend implements MovementModelBackend {
  readonly name = "markov";

  train(dataset: MovementDataset, options: MovementTrainingOptions = {}): TrainedMovementModel {
    const order = Math.max(1, Math.floor(options.order ?? 2));
    const counts = new Map<string, Map<MovementToken, number>>();
    const vocabulary = new Set<MovementToken>();
    for (const sequence of dataset.sequences) {
      const padded = [SEQUENCE_START, ...sequence.tokens];
      for (let i = 1; i < padded.length; i += 1) {
        const token = padded[i];
        vocabulary.add(token);
        for (let used = 0; used <= order; used += 1) {
          if (i - used < 0) {
            break;
          }
          const context = padded.slice(i - used, i);
          const key = contextKey(context);
          let table = counts.get(key);
          if (!table) {
            table = new Map<MovementToken, number>();
            counts.set(key, table);
          }
          table.set(token, (table.get(token) ?? 0) + 1);
        }
      }
    }
    return new MarkovMovementModel(order, counts, [...vocabulary].sort());
  }
}

/**
 * A documented seam for a real on-device backend. bee-agent cannot run native
 * training in the cloud, so these throw a clear, actionable error rather than
 * silently degrading. When the user runs bee-agent locally with the runtime
 * installed, a real implementation is registered under the same name.
 */
export class OnDeviceMovementBackendStub implements MovementModelBackend {
  constructor(readonly name: "mlx" | "axolotl") {}

  train(): TrainedMovementModel {
    throw new Error(
      `movement backend "${this.name}" requires a local on-device runtime and is unavailable in this environment; ` +
        `use the "markov" backend for in-process training, or run bee-agent locally with ${this.name} installed`,
    );
  }
}

/** Registry mapping backend name → backend, making the model backend pluggable. */
export class MovementModelRegistry {
  private readonly backends = new Map<string, MovementModelBackend>();

  constructor() {
    this.register(new MarkovMovementBackend());
    this.register(new OnDeviceMovementBackendStub("mlx"));
    this.register(new OnDeviceMovementBackendStub("axolotl"));
  }

  register(backend: MovementModelBackend): void {
    this.backends.set(backend.name, backend);
  }

  has(name: string): boolean {
    return this.backends.has(name);
  }

  list(): string[] {
    return [...this.backends.keys()].sort();
  }

  get(name: string): MovementModelBackend {
    const backend = this.backends.get(name);
    if (!backend) {
      throw new Error(`unknown movement model backend "${name}"; available: ${this.list().join(", ")}`);
    }
    return backend;
  }

  train(name: string, dataset: MovementDataset, options?: MovementTrainingOptions): TrainedMovementModel {
    return this.get(name).train(dataset, options);
  }
}

function contextKey(context: MovementToken[]): string {
  return context.length === 0 ? "" : context.join("");
}

function slug(value: string | undefined): string {
  if (!value) {
    return "";
  }
  return value
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 48);
}
