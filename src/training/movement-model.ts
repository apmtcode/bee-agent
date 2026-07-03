/**
 * Local-movement learning subsystem — in-process model backend.
 *
 * The capture → export pipeline produces reviewed movement trajectories, and
 * {@link LocalAppleSiliconTrainingRunner} plans an *external* on-device training
 * run (mlx/axolotl). That plan cannot execute in the cloud, so bee-agent also
 * needs a backend that can actually learn from a dataset and predict / generalize
 * movements *in process* — for validation, generalization evals, and as the
 * deterministic default when no heavyweight local model is available.
 *
 * The backend is pluggable: {@link MovementModelBackend} is the seam a real
 * on-device small model implements, and {@link MarkovMovementBackend} is a
 * dependency-free, fully deterministic reference implementation that trains an
 * order-k Markov policy with backoff. "Generalization to new but related
 * movements" is realized by a *shape* backoff: when an exact target context was
 * never observed, the model falls back to a context keyed by gesture+direction
 * (dropping the most variable field, the target), so it can still predict a
 * sensible next movement for a target it has never seen.
 */

export type MovementFeature = {
  /** Tool / channel that produced the movement, e.g. "device", "browser". */
  tool: string;
  /** Normalized gesture verb, e.g. "tap", "swipe", "scroll", "type", "shortcut". */
  gesture: string;
  /** Optional spatial direction for swipe/scroll movements. */
  direction?: "up" | "down" | "left" | "right";
  /** Optional target label (button, field, app). The most variable field. */
  target?: string;
};

export type MovementSequence = {
  id: string;
  features: MovementFeature[];
};

export type MovementPredictionStrategy = "exact" | "shape" | "unigram" | "empty";

export type MovementPrediction = {
  feature: MovementFeature;
  /** Estimated probability of this movement given the context (0..1). */
  confidence: number;
  /** Context order actually used to make the prediction (0 = unigram/empty). */
  order: number;
  /** Which backoff level produced the prediction. */
  strategy: MovementPredictionStrategy;
};

export type MovementModelTrainOptions = {
  /** Maximum context length the Markov model conditions on. Default 3. */
  maxOrder?: number;
};

export interface TrainedMovementModel {
  readonly backendId: string;
  /** Predict the single most likely next movement given a context window. */
  predictNext(context: MovementFeature[]): MovementPrediction;
  /** Autoregressively continue a movement sequence for `steps` movements. */
  predictSequence(context: MovementFeature[], steps: number): MovementPrediction[];
  serialize(): SerializedMovementModel;
}

export interface MovementModelBackend {
  readonly id: string;
  train(dataset: MovementSequence[], options?: MovementModelTrainOptions): TrainedMovementModel;
}

// ---------------------------------------------------------------------------
// Feature keying
// ---------------------------------------------------------------------------

const FIELD_SEP = "";
const CONTEXT_SEP = " > ";

/** Canonical key for an exact movement feature (all fields significant). */
export function movementFeatureKey(feature: MovementFeature): string {
  return [feature.tool, feature.gesture, feature.direction ?? "", feature.target ?? ""].join(FIELD_SEP);
}

/**
 * "Shape" key — drops the most variable field (`target`) so movements that
 * differ only in which element they touch collapse together. This is what lets
 * the model generalize to *related* movements it has not seen verbatim.
 */
export function movementShapeKey(feature: MovementFeature): string {
  return [feature.tool, feature.gesture, feature.direction ?? ""].join(FIELD_SEP);
}

function contextKey(context: MovementFeature[], keyOf: (feature: MovementFeature) => string): string {
  return context.map(keyOf).join(CONTEXT_SEP);
}

// ---------------------------------------------------------------------------
// Serialized form (portable, replayable across processes)
// ---------------------------------------------------------------------------

type CountEntry = { feature: MovementFeature; count: number };

/** context-key -> (target-key -> {feature, count}) */
type TransitionTable = Map<string, Map<string, CountEntry>>;

export type SerializedMovementModel = {
  version: 1;
  backendId: string;
  maxOrder: number;
  /** order (1..maxOrder) -> serialized exact transition table */
  ngrams: Array<{ order: number; table: SerializedTable }>;
  /** order (1..maxOrder) -> serialized shape (target-dropped) transition table */
  shapeNgrams: Array<{ order: number; table: SerializedTable }>;
  unigram: SerializedTable;
};

type SerializedTable = Array<{
  context: string;
  targets: Array<{ key: string; feature: MovementFeature; count: number }>;
}>;

function serializeTable(table: TransitionTable): SerializedTable {
  return [...table.entries()]
    .sort((a, b) => (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0))
    .map(([context, targets]) => ({
      context,
      targets: [...targets.entries()]
        .sort((a, b) => (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0))
        .map(([key, entry]) => ({ key, feature: entry.feature, count: entry.count })),
    }));
}

function deserializeTable(serialized: SerializedTable): TransitionTable {
  const table: TransitionTable = new Map();
  for (const row of serialized) {
    const targets = new Map<string, CountEntry>();
    for (const target of row.targets) {
      targets.set(target.key, { feature: target.feature, count: target.count });
    }
    table.set(row.context, targets);
  }
  return table;
}

// ---------------------------------------------------------------------------
// Markov backend (deterministic reference implementation)
// ---------------------------------------------------------------------------

export const MARKOV_BACKEND_ID = "markov-backoff-v1";

export class MarkovMovementBackend implements MovementModelBackend {
  readonly id = MARKOV_BACKEND_ID;

  train(dataset: MovementSequence[], options: MovementModelTrainOptions = {}): TrainedMovementModel {
    const maxOrder = Math.max(1, Math.floor(options.maxOrder ?? 3));
    const ngrams: TransitionTable[] = Array.from({ length: maxOrder + 1 }, () => new Map());
    const shapeNgrams: TransitionTable[] = Array.from({ length: maxOrder + 1 }, () => new Map());
    const unigram: Map<string, CountEntry> = new Map();

    for (const sequence of dataset) {
      const { features } = sequence;
      for (let i = 0; i < features.length; i += 1) {
        const target = features[i];
        if (target === undefined) {
          continue;
        }
        bump(unigram, movementFeatureKey(target), target);
        for (let order = 1; order <= maxOrder; order += 1) {
          if (i - order < 0) {
            break;
          }
          const context = features.slice(i - order, i);
          record(ngrams[order]!, contextKey(context, movementFeatureKey), target);
          record(shapeNgrams[order]!, contextKey(context, movementShapeKey), target);
        }
      }
    }

    return new MarkovMovementModel(this.id, maxOrder, ngrams, shapeNgrams, unigram);
  }
}

class MarkovMovementModel implements TrainedMovementModel {
  constructor(
    readonly backendId: string,
    private readonly maxOrder: number,
    private readonly ngrams: TransitionTable[],
    private readonly shapeNgrams: TransitionTable[],
    private readonly unigram: Map<string, CountEntry>,
  ) {}

  predictNext(context: MovementFeature[]): MovementPrediction {
    const usableOrder = Math.min(this.maxOrder, context.length);

    // 1. Exact context backoff: longest matching full-feature context wins.
    for (let order = usableOrder; order >= 1; order -= 1) {
      const window = context.slice(context.length - order);
      const table = this.ngrams[order];
      const targets = table?.get(contextKey(window, movementFeatureKey));
      const best = argmax(targets);
      if (best) {
        return { feature: best.entry.feature, confidence: best.probability, order, strategy: "exact" };
      }
    }

    // 2. Shape backoff: same context but target-agnostic — generalizes to
    //    related movements whose exact targets were never observed.
    for (let order = usableOrder; order >= 1; order -= 1) {
      const window = context.slice(context.length - order);
      const table = this.shapeNgrams[order];
      const targets = table?.get(contextKey(window, movementShapeKey));
      const best = argmax(targets);
      if (best) {
        return { feature: best.entry.feature, confidence: best.probability, order, strategy: "shape" };
      }
    }

    // 3. Unigram backoff: most frequent movement overall.
    const best = argmax(this.unigram);
    if (best) {
      return { feature: best.entry.feature, confidence: best.probability, order: 0, strategy: "unigram" };
    }

    // 4. Empty model.
    return { feature: { tool: "noop", gesture: "noop" }, confidence: 0, order: 0, strategy: "empty" };
  }

  predictSequence(context: MovementFeature[], steps: number): MovementPrediction[] {
    const window = [...context];
    const predictions: MovementPrediction[] = [];
    for (let step = 0; step < steps; step += 1) {
      const prediction = this.predictNext(window);
      predictions.push(prediction);
      if (prediction.strategy === "empty") {
        break;
      }
      window.push(prediction.feature);
      if (window.length > this.maxOrder) {
        window.shift();
      }
    }
    return predictions;
  }

  serialize(): SerializedMovementModel {
    const toEntries = (tables: TransitionTable[]) =>
      tables
        .map((table, order) => ({ order, table }))
        .filter((entry) => entry.order >= 1 && entry.table.size > 0)
        .map((entry) => ({ order: entry.order, table: serializeTable(entry.table) }));

    return {
      version: 1,
      backendId: this.backendId,
      maxOrder: this.maxOrder,
      ngrams: toEntries(this.ngrams),
      shapeNgrams: toEntries(this.shapeNgrams),
      unigram: serializeTable(new Map([["", this.unigram]])),
    };
  }
}

/**
 * Rehydrate a trained model from its serialized form, so a policy trained in one
 * process (or persisted to the dataset dir) can be replayed in another.
 */
export function deserializeMovementModel(serialized: SerializedMovementModel): TrainedMovementModel {
  const ngrams: TransitionTable[] = Array.from({ length: serialized.maxOrder + 1 }, () => new Map());
  const shapeNgrams: TransitionTable[] = Array.from({ length: serialized.maxOrder + 1 }, () => new Map());
  for (const { order, table } of serialized.ngrams) {
    if (order >= 1 && order <= serialized.maxOrder) {
      ngrams[order] = deserializeTable(table);
    }
  }
  for (const { order, table } of serialized.shapeNgrams) {
    if (order >= 1 && order <= serialized.maxOrder) {
      shapeNgrams[order] = deserializeTable(table);
    }
  }
  const unigramTable = deserializeTable(serialized.unigram);
  const unigram = unigramTable.get("") ?? new Map<string, CountEntry>();
  return new MarkovMovementModel(serialized.backendId, serialized.maxOrder, ngrams, shapeNgrams, unigram);
}

// ---------------------------------------------------------------------------
// Counting helpers
// ---------------------------------------------------------------------------

function record(table: TransitionTable, context: string, target: MovementFeature): void {
  let targets = table.get(context);
  if (!targets) {
    targets = new Map();
    table.set(context, targets);
  }
  bump(targets, movementFeatureKey(target), target);
}

function bump(targets: Map<string, CountEntry>, key: string, feature: MovementFeature): void {
  const existing = targets.get(key);
  if (existing) {
    existing.count += 1;
  } else {
    targets.set(key, { feature, count: 1 });
  }
}

function argmax(targets: Map<string, CountEntry> | undefined): { entry: CountEntry; probability: number } | undefined {
  if (!targets || targets.size === 0) {
    return undefined;
  }
  let total = 0;
  let bestKey: string | undefined;
  let bestEntry: CountEntry | undefined;
  for (const [key, entry] of targets) {
    total += entry.count;
    if (
      bestEntry === undefined ||
      entry.count > bestEntry.count ||
      // Deterministic tie-break: lexicographically smallest key wins.
      (entry.count === bestEntry.count && bestKey !== undefined && key < bestKey)
    ) {
      bestEntry = entry;
      bestKey = key;
    }
  }
  if (!bestEntry) {
    return undefined;
  }
  return { entry: bestEntry, probability: bestEntry.count / total };
}

// ---------------------------------------------------------------------------
// Adapters: derive movement sequences from the capture/export pipeline
// ---------------------------------------------------------------------------

type ReplayActionEvent = { kind: "action"; ts: number; trajectoryId: string; tool: string; summary: string };

type ReplayLike = {
  sessionId: string;
  trajectoryIds: string[];
  events: Array<{ kind: string; ts: number }>;
};

const DIRECTIONS = new Set(["up", "down", "left", "right"]);

const GESTURE_BY_VERB: Record<string, string> = {
  tapped: "tap",
  swiped: "swipe",
  scrolled: "scroll",
  typed: "type",
  triggered: "shortcut",
  clicked: "tap",
  pressed: "shortcut",
};

/**
 * Best-effort parse of a movement feature from a replay action event. Replay
 * manifests only retain `tool` + human `summary` (gesture metadata is dropped
 * at manifest time), so the gesture verb and direction are recovered from the
 * summary text produced by the capture adapters.
 */
export function parseMovementFeatureFromSummary(tool: string, summary: string): MovementFeature {
  const words = summary.trim().toLowerCase().split(/\s+/);
  const verb = words[0] ?? "";
  const gesture = GESTURE_BY_VERB[verb] ?? verb ?? "action";
  const direction = words.find((word): word is MovementFeature["direction"] & string => DIRECTIONS.has(word));
  const targetWords = words.filter((word) => word !== verb && !DIRECTIONS.has(word) && word !== "into" && word !== "on");
  const target = targetWords.length > 0 ? targetWords.join(" ") : undefined;
  return {
    tool,
    gesture: gesture || "action",
    ...(direction ? { direction } : {}),
    ...(target ? { target } : {}),
  };
}

/**
 * Build one {@link MovementSequence} per replay manifest, ordering action
 * movements by timestamp. Works on both the in-memory `ReplayManifest` and the
 * exported `ExportedReplayManifest` (structurally identical action events).
 */
export function buildMovementSequencesFromReplays(replays: ReplayLike[]): MovementSequence[] {
  return replays
    .map((replay) => {
      const actions = replay.events
        .filter((event): event is ReplayActionEvent => event.kind === "action")
        .sort((a, b) => a.ts - b.ts);
      const features = actions.map((action) => parseMovementFeatureFromSummary(action.tool, action.summary));
      return {
        id: replay.trajectoryIds[0] ?? replay.sessionId,
        features,
      };
    })
    .filter((sequence) => sequence.features.length > 0);
}
