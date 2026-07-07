import type { ReplayTimelineEvent } from "../capture/replay.js";
import type { TrajectoryAction, TrajectorySpan } from "../capture/trajectory.js";

/**
 * Local-movement learning core (standing objective #2, pieces (c) train + (d)
 * generalize).
 *
 * The training {@link ../training/runner.js LocalAppleSiliconTrainingRunner}
 * emits *external* mlx/axolotl launch plans that only run on a real on-device
 * machine. This module is the complementary *in-process* seam: a pluggable
 * {@link MovementModelBackend} that can actually learn a movement policy from a
 * dataset and predict the next movement, so the train -> infer -> generalize
 * loop is exercisable (and testable) in the cloud with synthetic data.
 *
 * The reference backend is a deterministic back-off n-gram
 * ({@link NgramMovementBackend}). A real on-device small model implements the
 * same interface and is swapped in behind it.
 */

/** A normalized, replayable movement action drawn from a captured trajectory. */
export type MovementToken = string;

export type MovementTokenOptions = {
  /**
   * Include the concrete UI target (e.g. a specific field id) in the token.
   * Defaults to `false` so tokens describe the *structure* of a movement
   * (gesture kind + direction) — the part that generalizes across trajectories
   * that touch new-but-related targets.
   */
  includeTarget?: boolean;
};

export type MovementSequence = {
  trajectoryId: string;
  sessionId?: string;
  tokens: MovementToken[];
};

export type MovementDataset = {
  sequences: MovementSequence[];
};

export type MovementTrainOptions = {
  /** Maximum n-gram order (context length + 1). Defaults to 3. */
  order?: number;
};

export type MovementPrediction = {
  /** Predicted next token, or `null` when the model has learned nothing. */
  token: MovementToken | null;
  /** Conditional probability of the token at the context order actually used. */
  confidence: number;
  /** Context length used after back-off (0 = unconditional prior). */
  order: number;
  /** True when the exact requested context was unseen and back-off kicked in. */
  viaBackoff: boolean;
};

/**
 * A trained, queryable movement policy. Deterministic: identical datasets and
 * contexts always yield identical predictions (ties break by highest count then
 * lexicographically), so it is safe to assert on in tests.
 */
export interface TrainedMovementModel {
  readonly backendId: string;
  readonly order: number;
  /** Best next movement given the recent context (most-recent token last). */
  predict(context: MovementToken[]): MovementPrediction;
  /** All candidate next movements at the used order, best first. */
  rank(context: MovementToken[]): MovementPrediction[];
  /** Vocabulary of movements the model has observed. */
  vocabulary(): MovementToken[];
  /** Plain-object form for persistence / inspection. */
  serialize(): SerializedMovementModel;
}

/**
 * A pluggable movement-model backend. The reference implementation
 * ({@link NgramMovementBackend}) is deterministic and dependency-free; a real
 * on-device small-model backend implements the same shape.
 */
export interface MovementModelBackend {
  readonly id: string;
  train(dataset: MovementDataset, options?: MovementTrainOptions): TrainedMovementModel;
  load(serialized: SerializedMovementModel): TrainedMovementModel;
}

export type SerializedMovementModel = {
  version: 1;
  backendId: string;
  order: number;
  /** For each context order 0..order-1: contextKey -> { token: count }. */
  transitions: Record<string, Record<string, number>>[];
};

const CONTEXT_SEPARATOR = "␟"; // ␟ symbol for unit separator — never appears in tokens.

function slug(value: string): string {
  return value
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 48) || "unknown";
}

function readStringField(metadata: Record<string, unknown> | undefined, key: string): string | undefined {
  const value = metadata?.[key];
  return typeof value === "string" && value.length > 0 ? value : undefined;
}

/**
 * Derive a normalized movement token from a captured action. By default the
 * token captures the movement *structure* (`tool:gesture[:direction]`) rather
 * than the concrete target, so a model trained on one set of targets can repeat
 * the movement on new-but-related targets.
 */
export function movementToken(action: TrajectoryAction, options: MovementTokenOptions = {}): MovementToken {
  const metadata = action.metadata;
  const gesture = readStringField(metadata, "gesture");
  if (gesture) {
    const direction = readStringField(metadata, "direction");
    const parts = [action.tool, gesture];
    if (direction) {
      parts.push(direction);
    }
    if (options.includeTarget) {
      const target = readStringField(metadata, "target");
      if (target) {
        parts.push(slug(target));
      }
    }
    return parts.join(":");
  }
  return `${action.tool}:${slug(action.summary)}`;
}

/** Ordered movement tokens for a single captured trajectory. */
export function extractMovementSequence(
  trajectory: TrajectorySpan,
  options: MovementTokenOptions = {},
): MovementSequence {
  const actions = [...trajectory.actions].sort((a, b) => a.ts - b.ts);
  return {
    trajectoryId: trajectory.id,
    sessionId: trajectory.sessionId,
    tokens: actions.map((action) => movementToken(action, options)),
  };
}

/** Ordered movement tokens from a replay manifest's action events. */
export function extractMovementSequenceFromReplay(
  replay: { trajectoryIds: string[]; sessionId?: string; events: ReplayTimelineEvent[] },
  options: MovementTokenOptions = {},
): MovementSequence {
  const actions = replay.events
    .filter((event): event is Extract<ReplayTimelineEvent, { kind: "action" }> => event.kind === "action")
    .sort((a, b) => a.ts - b.ts)
    .map((event) =>
      movementToken(
        { kind: "action", tool: event.tool, summary: event.summary, ts: event.ts },
        options,
      ),
    );
  return {
    trajectoryId: replay.trajectoryIds[0] ?? "replay",
    sessionId: replay.sessionId,
    tokens: actions,
  };
}

/** Build a movement dataset from captured trajectories (empty sequences dropped). */
export function buildMovementDataset(
  trajectories: TrajectorySpan[],
  options: MovementTokenOptions = {},
): MovementDataset {
  return {
    sequences: trajectories
      .map((trajectory) => extractMovementSequence(trajectory, options))
      .filter((sequence) => sequence.tokens.length > 0),
  };
}

type TransitionTable = Map<string, Map<MovementToken, number>>;

class NgramMovementModel implements TrainedMovementModel {
  readonly backendId: string;
  readonly order: number;
  private readonly tables: TransitionTable[];
  private readonly vocab: MovementToken[];

  constructor(backendId: string, order: number, tables: TransitionTable[]) {
    this.backendId = backendId;
    this.order = order;
    this.tables = tables;
    const seen = new Set<MovementToken>();
    for (const table of tables) {
      for (const counts of table.values()) {
        for (const token of counts.keys()) {
          seen.add(token);
        }
      }
    }
    this.vocab = [...seen].sort();
  }

  private contextKey(tokens: MovementToken[], length: number): string {
    if (length === 0) {
      return "";
    }
    return tokens.slice(tokens.length - length).join(CONTEXT_SEPARATOR);
  }

  private rankAtOrder(counts: Map<MovementToken, number>): { token: MovementToken; confidence: number }[] {
    let total = 0;
    for (const count of counts.values()) {
      total += count;
    }
    return [...counts.entries()]
      .sort((a, b) => (b[1] !== a[1] ? b[1] - a[1] : a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0))
      .map(([token, count]) => ({ token, confidence: total > 0 ? count / total : 0 }));
  }

  /** Longest usable context length <= ideal that has observed continuations. */
  private resolveOrder(context: MovementToken[]): { used: number; ideal: number } | undefined {
    const ideal = Math.min(this.order - 1, context.length);
    for (let length = ideal; length >= 0; length -= 1) {
      const table = this.tables[length];
      if (table && table.has(this.contextKey(context, length))) {
        return { used: length, ideal };
      }
    }
    return undefined;
  }

  predict(context: MovementToken[]): MovementPrediction {
    const ranked = this.rank(context);
    return ranked[0] ?? { token: null, confidence: 0, order: 0, viaBackoff: false };
  }

  rank(context: MovementToken[]): MovementPrediction[] {
    const resolved = this.resolveOrder(context);
    if (!resolved) {
      return [];
    }
    const counts = this.tables[resolved.used].get(this.contextKey(context, resolved.used));
    if (!counts) {
      return [];
    }
    const viaBackoff = resolved.used < resolved.ideal;
    return this.rankAtOrder(counts).map((entry) => ({
      token: entry.token,
      confidence: entry.confidence,
      order: resolved.used,
      viaBackoff,
    }));
  }

  vocabulary(): MovementToken[] {
    return [...this.vocab];
  }

  serialize(): SerializedMovementModel {
    return {
      version: 1,
      backendId: this.backendId,
      order: this.order,
      transitions: this.tables.map((table) => {
        const record: Record<string, Record<string, number>> = {};
        for (const [contextKey, counts] of table.entries()) {
          record[contextKey] = Object.fromEntries(counts.entries());
        }
        return record;
      }),
    };
  }
}

/**
 * Deterministic back-off n-gram movement backend. Learns, for every context
 * length 0..order-1, the empirical distribution of the following movement.
 * Prediction uses the longest context that was observed during training and
 * otherwise backs off to shorter contexts (finally the unconditional prior) —
 * which is exactly what lets it generalize to novel-but-related movement
 * sequences it never saw verbatim.
 */
export class NgramMovementBackend implements MovementModelBackend {
  readonly id = "ngram";

  train(dataset: MovementDataset, options: MovementTrainOptions = {}): TrainedMovementModel {
    const order = Math.max(1, Math.floor(options.order ?? 3));
    const tables: TransitionTable[] = Array.from({ length: order }, () => new Map<string, Map<MovementToken, number>>());

    for (const sequence of dataset.sequences) {
      const tokens = sequence.tokens;
      for (let index = 0; index < tokens.length; index += 1) {
        const next = tokens[index];
        const maxContext = Math.min(order - 1, index);
        for (let length = 0; length <= maxContext; length += 1) {
          const contextTokens = tokens.slice(index - length, index);
          const key = length === 0 ? "" : contextTokens.join(CONTEXT_SEPARATOR);
          const table = tables[length];
          let counts = table.get(key);
          if (!counts) {
            counts = new Map<MovementToken, number>();
            table.set(key, counts);
          }
          counts.set(next, (counts.get(next) ?? 0) + 1);
        }
      }
    }

    return new NgramMovementModel(this.id, order, tables);
  }

  load(serialized: SerializedMovementModel): TrainedMovementModel {
    const tables: TransitionTable[] = serialized.transitions.map((record) => {
      const table: TransitionTable = new Map();
      for (const [contextKey, counts] of Object.entries(record)) {
        table.set(contextKey, new Map(Object.entries(counts)));
      }
      return table;
    });
    return new NgramMovementModel(serialized.backendId, serialized.order, tables);
  }
}
