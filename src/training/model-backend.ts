import type { ReplayManifest } from "../capture/replay.js";
import type { TrajectoryAction, TrajectorySpan } from "../capture/trajectory.js";

/**
 * Movement-model backend for the local-movement learning subsystem.
 *
 * This is the in-process, pluggable seam that turns reviewed movement
 * trajectories into a small model that can (a) predict the next movement given
 * prior context and (b) generate whole movement sequences — the "repeat the
 * recorded movements" and "generalize to related movements" objectives.
 *
 * The bundled {@link NgramMovementBackend} is a deterministic Markov/n-gram
 * model. It is intentionally CPU-only and dependency-free so it trains and runs
 * in the cloud/CI (no real OS input, no GPU). A real on-device small model
 * (e.g. an MLX-trained policy) can implement {@link MovementModelBackend} and be
 * registered via {@link registerMovementModelBackend} without touching callers.
 */

/** A single normalized movement, derived from a recorded trajectory action. */
export type MovementToken = {
  /** The tool/surface the movement acted through (e.g. "device", "browser"). */
  tool: string;
  /** Human-readable summary, preserved for replay/inspection. */
  summary: string;
  /** Gesture kind when known (tap/swipe/scroll/type/shortcut/click/…). */
  gesture?: string;
  /** UI target the movement addressed, when known. */
  target?: string;
  /** Movement direction when known (up/down/left/right). */
  direction?: string;
};

/** An ordered movement sequence — one training example (one trajectory/replay). */
export type MovementSequence = {
  id: string;
  tokens: MovementToken[];
};

/** A dataset of movement sequences ready for training. */
export type MovementDataset = {
  version: 1;
  sequences: MovementSequence[];
};

/** A predicted next movement with the backend's confidence in it. */
export type MovementPrediction = {
  token: MovementToken;
  /** 0..1 — share of observed continuations that matched this token. */
  confidence: number;
  /** How many prior tokens of context the prediction actually used (backoff). */
  contextUsed: number;
};

/** Serializable form of a trained model so it can be persisted as an artifact. */
export type SerializedMovementModel = {
  backend: string;
  version: 1;
  order: number;
  trainedAt: string;
  sequenceCount: number;
  vocabulary: Array<{ key: string; token: MovementToken }>;
  transitions: Array<{ context: string; next: Array<{ key: string; count: number }> }>;
};

/** A trained model: pure inference, deterministic, no I/O. */
export type TrainedMovementModel = {
  readonly backend: string;
  readonly order: number;
  readonly trainedAt: string;
  readonly sequenceCount: number;
  readonly vocabularySize: number;
  /**
   * Predict the movement most likely to follow `context`, backing off to
   * shorter contexts (and finally global frequency) when the exact context was
   * never observed — this is what lets the model generalize to new-but-related
   * movement prefixes. Returns undefined only for an empty vocabulary.
   */
  predictNext(context: MovementToken[]): MovementPrediction | undefined;
  /**
   * Greedily generate up to `maxSteps` movements continuing from `seed`,
   * appending each prediction to the context. Stops early if the model can no
   * longer predict (empty vocabulary).
   */
  generate(seed: MovementToken[], maxSteps: number): MovementToken[];
  /** Persist the model to a plain, JSON-safe object. */
  serialize(): SerializedMovementModel;
};

/** The pluggable backend seam. */
export interface MovementModelBackend {
  readonly name: string;
  train(dataset: MovementDataset): Promise<TrainedMovementModel>;
}

/** Stable identity key for a token — the unit the model reasons over. */
export function movementTokenKey(token: MovementToken): string {
  const gesture = token.gesture?.trim() ?? "";
  const target = token.target?.trim() ?? "";
  const direction = token.direction?.trim() ?? "";
  const base = `${token.tool}|${gesture}|${target}|${direction}`;
  // When there is no structured detail, fall back to the summary so distinct
  // free-text movements remain distinct tokens instead of collapsing together.
  if (!gesture && !target && !direction) {
    return `${base}|${token.summary.trim()}`;
  }
  return base;
}

/** Derive a normalized {@link MovementToken} from a recorded trajectory action. */
export function tokenizeAction(action: TrajectoryAction): MovementToken {
  const metadata = action.metadata ?? {};
  const gesture = readString(metadata.gesture);
  const target = readString(metadata.target);
  const direction = readString(metadata.direction);
  return {
    tool: action.tool,
    summary: action.summary,
    ...(gesture ? { gesture } : {}),
    ...(target ? { target } : {}),
    ...(direction ? { direction } : {}),
  };
}

/** Build a training dataset from trajectory spans (one sequence per span). */
export function buildMovementDataset(trajectories: TrajectorySpan[]): MovementDataset {
  const sequences: MovementSequence[] = trajectories
    .map((trajectory) => ({
      id: trajectory.id,
      tokens: [...trajectory.actions]
        .sort((a, b) => a.ts - b.ts)
        .map((action) => tokenizeAction(action)),
    }))
    .filter((sequence) => sequence.tokens.length > 0);
  return { version: 1, sequences };
}

/** Build a training dataset from replay manifests (one sequence per replay). */
export function buildMovementDatasetFromReplays(replays: ReplayManifest[]): MovementDataset {
  const sequences: MovementSequence[] = replays
    .map((replay, index) => ({
      id: replay.trajectoryIds[0] ?? `${replay.sessionId}-${index}`,
      tokens: replay.events
        .filter((event): event is Extract<ReplayManifest["events"][number], { kind: "action" }> => event.kind === "action")
        .map((event) => ({ tool: event.tool, summary: event.summary })),
    }))
    .filter((sequence) => sequence.tokens.length > 0);
  return { version: 1, sequences };
}

type TransitionTable = Map<string, Map<string, number>>;

class NgramMovementModel implements TrainedMovementModel {
  readonly backend: string;
  readonly order: number;
  readonly trainedAt: string;
  readonly sequenceCount: number;

  constructor(
    backend: string,
    order: number,
    trainedAt: string,
    sequenceCount: number,
    private readonly vocabulary: Map<string, MovementToken>,
    private readonly transitions: TransitionTable,
  ) {
    this.backend = backend;
    this.order = order;
    this.trainedAt = trainedAt;
    this.sequenceCount = sequenceCount;
  }

  get vocabularySize(): number {
    return this.vocabulary.size;
  }

  predictNext(context: MovementToken[]): MovementPrediction | undefined {
    if (this.vocabulary.size === 0) {
      return undefined;
    }
    const keys = context.map((token) => movementTokenKey(token));
    for (let used = Math.min(this.order, keys.length); used >= 0; used -= 1) {
      const contextKey = keys.slice(keys.length - used).join(">");
      const table = this.transitions.get(contextKey);
      if (!table || table.size === 0) {
        continue;
      }
      const best = argmax(table);
      if (best) {
        const token = this.vocabulary.get(best.key);
        if (token) {
          return { token, confidence: best.count / best.total, contextUsed: used };
        }
      }
    }
    return undefined;
  }

  generate(seed: MovementToken[], maxSteps: number): MovementToken[] {
    const produced: MovementToken[] = [];
    const context = [...seed];
    for (let step = 0; step < maxSteps; step += 1) {
      const prediction = this.predictNext(context);
      if (!prediction) {
        break;
      }
      produced.push(prediction.token);
      context.push(prediction.token);
    }
    return produced;
  }

  serialize(): SerializedMovementModel {
    return {
      backend: this.backend,
      version: 1,
      order: this.order,
      trainedAt: this.trainedAt,
      sequenceCount: this.sequenceCount,
      vocabulary: [...this.vocabulary.entries()]
        .sort((a, b) => (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0))
        .map(([key, token]) => ({ key, token })),
      transitions: [...this.transitions.entries()]
        .sort((a, b) => (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0))
        .map(([context, table]) => ({
          context,
          next: [...table.entries()]
            .sort((a, b) => (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0))
            .map(([key, count]) => ({ key, count })),
        })),
    };
  }
}

/**
 * Deterministic n-gram/Markov backend. `order` is the maximum context length;
 * predictions back off to shorter contexts so unseen prefixes still resolve.
 */
export class NgramMovementBackend implements MovementModelBackend {
  readonly name = "ngram-markov";

  constructor(private readonly order = 2, private readonly clock: () => Date = () => new Date()) {
    if (!Number.isInteger(order) || order < 1) {
      throw new Error(`NgramMovementBackend order must be a positive integer, got ${order}`);
    }
  }

  async train(dataset: MovementDataset): Promise<TrainedMovementModel> {
    const vocabulary = new Map<string, MovementToken>();
    const transitions: TransitionTable = new Map();

    for (const sequence of dataset.sequences) {
      const keys = sequence.tokens.map((token) => {
        const key = movementTokenKey(token);
        if (!vocabulary.has(key)) {
          vocabulary.set(key, token);
        }
        return key;
      });

      for (let index = 0; index < keys.length; index += 1) {
        const nextKey = keys[index]!;
        // Record every backoff order (0..order) so lower-order tables exist for
        // generalization, not just the full-order context.
        const maxContext = Math.min(this.order, index);
        for (let used = 0; used <= maxContext; used += 1) {
          const contextKey = keys.slice(index - used, index).join(">");
          increment(transitions, contextKey, nextKey);
        }
      }
    }

    return new NgramMovementModel(
      this.name,
      this.order,
      this.clock().toISOString(),
      dataset.sequences.length,
      vocabulary,
      transitions,
    );
  }

  /** Reconstruct a trained model from its serialized form. */
  load(serialized: SerializedMovementModel): TrainedMovementModel {
    const vocabulary = new Map<string, MovementToken>();
    for (const entry of serialized.vocabulary) {
      vocabulary.set(entry.key, entry.token);
    }
    const transitions: TransitionTable = new Map();
    for (const entry of serialized.transitions) {
      const table = new Map<string, number>();
      for (const next of entry.next) {
        table.set(next.key, next.count);
      }
      transitions.set(entry.context, table);
    }
    return new NgramMovementModel(
      serialized.backend,
      serialized.order,
      serialized.trainedAt,
      serialized.sequenceCount,
      vocabulary,
      transitions,
    );
  }
}

type MovementModelBackendFactory = () => MovementModelBackend;

const BACKEND_REGISTRY = new Map<string, MovementModelBackendFactory>([
  ["ngram-markov", () => new NgramMovementBackend()],
]);

/**
 * Register a movement-model backend under a name. This is the seam a real
 * on-device small model plugs into: implement {@link MovementModelBackend} and
 * register it, then select it by name without changing any caller.
 */
export function registerMovementModelBackend(name: string, factory: MovementModelBackendFactory): void {
  BACKEND_REGISTRY.set(name, factory);
}

/** Names of all registered backends. */
export function listMovementModelBackends(): string[] {
  return [...BACKEND_REGISTRY.keys()].sort();
}

/** Resolve a backend by name (defaults to the deterministic n-gram mock). */
export function createMovementModelBackend(name = "ngram-markov"): MovementModelBackend {
  const factory = BACKEND_REGISTRY.get(name);
  if (!factory) {
    throw new Error(`unknown movement-model backend "${name}"; known: ${listMovementModelBackends().join(", ")}`);
  }
  return factory();
}

function increment(transitions: TransitionTable, contextKey: string, nextKey: string): void {
  let table = transitions.get(contextKey);
  if (!table) {
    table = new Map<string, number>();
    transitions.set(contextKey, table);
  }
  table.set(nextKey, (table.get(nextKey) ?? 0) + 1);
}

function argmax(table: Map<string, number>): { key: string; count: number; total: number } | undefined {
  let bestKey: string | undefined;
  let bestCount = -1;
  let total = 0;
  // Iterate in sorted key order so ties resolve deterministically.
  for (const key of [...table.keys()].sort()) {
    const count = table.get(key)!;
    total += count;
    if (count > bestCount) {
      bestCount = count;
      bestKey = key;
    }
  }
  if (bestKey === undefined) {
    return undefined;
  }
  return { key: bestKey, count: bestCount, total };
}

function readString(value: unknown): string | undefined {
  if (typeof value === "string" && value.trim().length > 0) {
    return value;
  }
  return undefined;
}
