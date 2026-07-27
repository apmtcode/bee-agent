/**
 * Pluggable local-movement model backend.
 *
 * The training runner (`runner.ts`) only emits a *plan* + launch script that
 * shells out to a real Apple-Silicon runtime (mlx / axolotl). That can never
 * execute in the cloud/CI, so nothing validated the actual "post-train a local
 * model on the movement dataset, then repeat + generalize the recorded
 * movements" objective.
 *
 * This module closes that gap with a fully in-process, deterministic backend:
 *   - a compact, model-facing movement dataset derived from replay events,
 *   - a `MovementModelBackend` interface (the pluggable seam — a real on-device
 *     small model can implement the same contract), and
 *   - `NgramMovementBackend`, an order-N Markov/back-off reference backend that
 *     (a) reproduces recorded movements and (b) generalizes to novel prefixes.
 *
 * Because it is pure computation (no OS input, no Python, no network) it runs
 * anywhere the tests run, which is exactly what the cloud self-evolution engine
 * needs to validate the pipeline.
 */

/** A replay event shape shared by `ReplayManifest` and `ExportedReplayManifest`. */
export type MovementReplayEvent =
  | { kind: "transcript"; ts: number }
  | { kind: "observation"; ts: number; trajectoryId: string; source: string; summary: string }
  | { kind: "action"; ts: number; trajectoryId: string; tool: string; summary: string };

export type MovementTokenType = "observation" | "action";

/** A single model-facing movement step. `symbol` is the discrete training token. */
export type MovementToken = {
  type: MovementTokenType;
  symbol: string;
  summary: string;
  ts: number;
};

/** One recorded movement (per trajectory), ordered in time. */
export type MovementSequence = {
  id: string;
  tokens: MovementToken[];
};

export type MovementDataset = {
  version: 1;
  sequences: MovementSequence[];
};

/** Sentinels bracket every training sequence so start/stop are learnable. */
export const MOVEMENT_START = "<movement:start>";
export const MOVEMENT_END = "<movement:end>";

const CONTEXT_SEPARATOR = "␟";
const DEFAULT_ORDER = 3;
const DEFAULT_MAX_STEPS = 128;

function slug(value: string): string {
  const normalized = value
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
  return normalized.length > 0 ? normalized : "unknown";
}

/**
 * Convert a replay event into a movement token. Transcript events are not
 * movements and are dropped (returns undefined).
 */
export function movementTokenForEvent(event: MovementReplayEvent): MovementToken | undefined {
  if (event.kind === "action") {
    return {
      type: "action",
      symbol: `action:${slug(event.tool)}:${slug(event.summary)}`,
      summary: event.summary,
      ts: event.ts,
    };
  }
  if (event.kind === "observation") {
    return {
      type: "observation",
      symbol: `observation:${slug(event.source)}:${slug(event.summary)}`,
      summary: event.summary,
      ts: event.ts,
    };
  }
  return undefined;
}

export type BuildMovementDatasetParams = {
  replays: Array<{ trajectoryIds?: string[]; events: MovementReplayEvent[] }>;
};

/**
 * Build a movement dataset from replay manifests. Events are grouped by
 * trajectory (one sequence per trajectory), ordered by timestamp with a stable
 * tie-break so a given input always yields the same dataset.
 */
export function buildMovementDataset(params: BuildMovementDatasetParams): MovementDataset {
  const sequences: MovementSequence[] = [];

  for (const replay of params.replays) {
    const byTrajectory = new Map<string, MovementToken[]>();
    const order: string[] = [];

    const sorted = [...replay.events].sort((a, b) => a.ts - b.ts);
    for (const event of sorted) {
      const token = movementTokenForEvent(event);
      if (!token) {
        continue;
      }
      const trajectoryId = "trajectoryId" in event ? event.trajectoryId : "movement";
      if (!byTrajectory.has(trajectoryId)) {
        byTrajectory.set(trajectoryId, []);
        order.push(trajectoryId);
      }
      byTrajectory.get(trajectoryId)!.push(token);
    }

    for (const trajectoryId of order) {
      const tokens = byTrajectory.get(trajectoryId)!;
      if (tokens.length > 0) {
        sequences.push({ id: trajectoryId, tokens });
      }
    }
  }

  return { version: 1, sequences };
}

export type MovementModelTrainOptions = {
  /** Maximum Markov context length. Higher = more faithful repeat, less backoff. */
  order?: number;
};

export type MovementPredictionCandidate = {
  symbol: string;
  probability: number;
};

export type MovementPrediction = {
  symbol: string;
  probability: number;
  /** Context length that produced the prediction (N..0); lower = more generalization. */
  matchedOrder: number;
  candidates: MovementPredictionCandidate[];
};

export type MovementGenerateOptions = {
  maxSteps?: number;
};

export type SerializedMovementModel = {
  version: 1;
  backendId: string;
  order: number;
  vocabulary: string[];
  /** counts[k] = context-of-length-k → next-symbol → count. */
  levels: Array<Record<string, Record<string, number>>>;
};

export type TrainedMovementModel = {
  readonly backendId: string;
  readonly order: number;
  readonly vocabulary: string[];
  /** Most-likely next symbol given a (possibly novel) context, with backoff. */
  predict(context: string[]): MovementPrediction | undefined;
  /** Roll out a movement from a seed until the end sentinel or maxSteps. */
  generate(seed?: string[], options?: MovementGenerateOptions): string[];
  serialize(): SerializedMovementModel;
};

/** The pluggable seam: a real on-device model can implement this same contract. */
export interface MovementModelBackend {
  readonly id: string;
  train(dataset: MovementDataset, options?: MovementModelTrainOptions): TrainedMovementModel;
}

type CountLevels = Array<Map<string, Map<string, number>>>;

function argmax(distribution: Map<string, number>, exclude: ReadonlySet<string>): MovementPrediction | undefined {
  let total = 0;
  const candidates: MovementPredictionCandidate[] = [];
  for (const [symbol, count] of distribution) {
    if (exclude.has(symbol)) {
      continue;
    }
    total += count;
    candidates.push({ symbol, probability: count });
  }
  if (total === 0) {
    return undefined;
  }
  // Deterministic ordering: probability desc, then symbol asc.
  candidates.sort((a, b) => (b.probability - a.probability) || (a.symbol < b.symbol ? -1 : a.symbol > b.symbol ? 1 : 0));
  const normalized = candidates.map((candidate) => ({ symbol: candidate.symbol, probability: candidate.probability / total }));
  const best = normalized[0]!;
  return { symbol: best.symbol, probability: best.probability, matchedOrder: -1, candidates: normalized };
}

class NgramMovementModel implements TrainedMovementModel {
  constructor(
    readonly backendId: string,
    readonly order: number,
    readonly vocabulary: string[],
    private readonly levels: CountLevels,
  ) {}

  predict(context: string[]): MovementPrediction | undefined {
    const maxOrder = Math.min(this.order, context.length);
    for (let k = maxOrder; k >= 0; k -= 1) {
      const level = this.levels[k];
      if (!level) {
        continue;
      }
      const key = context.slice(context.length - k).join(CONTEXT_SEPARATOR);
      const distribution = level.get(key);
      if (!distribution) {
        continue;
      }
      // START never appears as a prediction target; END is a valid terminal.
      const prediction = argmax(distribution, new Set([MOVEMENT_START]));
      if (prediction) {
        return { ...prediction, matchedOrder: k };
      }
    }
    return undefined;
  }

  generate(seed: string[] = [], options: MovementGenerateOptions = {}): string[] {
    const maxSteps = options.maxSteps ?? DEFAULT_MAX_STEPS;
    const context = [MOVEMENT_START, ...seed];
    const generated: string[] = [];
    for (let step = 0; step < maxSteps; step += 1) {
      const prediction = this.predict(context);
      if (!prediction || prediction.symbol === MOVEMENT_END) {
        break;
      }
      generated.push(prediction.symbol);
      context.push(prediction.symbol);
    }
    return generated;
  }

  serialize(): SerializedMovementModel {
    return {
      version: 1,
      backendId: this.backendId,
      order: this.order,
      vocabulary: [...this.vocabulary],
      levels: this.levels.map((level) => {
        const record: Record<string, Record<string, number>> = {};
        for (const [key, distribution] of level) {
          const inner: Record<string, number> = {};
          for (const [symbol, count] of distribution) {
            inner[symbol] = count;
          }
          record[key] = inner;
        }
        return record;
      }),
    };
  }
}

/**
 * Deterministic order-N Markov backend — the reference local-movement model.
 * It is the "mock" backend that lets cloud/CI validate the full
 * capture → dataset → train → repeat → generalize loop without real hardware.
 */
export class NgramMovementBackend implements MovementModelBackend {
  readonly id = "ngram-markov";

  constructor(private readonly defaultOrder: number = DEFAULT_ORDER) {}

  train(dataset: MovementDataset, options: MovementModelTrainOptions = {}): TrainedMovementModel {
    const order = Math.max(0, options.order ?? this.defaultOrder);
    const levels: CountLevels = Array.from({ length: order + 1 }, () => new Map<string, Map<string, number>>());
    const vocabulary = new Set<string>();

    for (const sequence of dataset.sequences) {
      const symbols = [MOVEMENT_START, ...sequence.tokens.map((token) => token.symbol), MOVEMENT_END];
      for (const token of sequence.tokens) {
        vocabulary.add(token.symbol);
      }
      for (let i = 1; i < symbols.length; i += 1) {
        const next = symbols[i]!;
        const maxOrder = Math.min(order, i);
        for (let k = 0; k <= maxOrder; k += 1) {
          const contextSymbols = symbols.slice(i - k, i);
          const key = contextSymbols.join(CONTEXT_SEPARATOR);
          const level = levels[k]!;
          let distribution = level.get(key);
          if (!distribution) {
            distribution = new Map<string, number>();
            level.set(key, distribution);
          }
          distribution.set(next, (distribution.get(next) ?? 0) + 1);
        }
      }
    }

    return new NgramMovementModel(this.id, order, [...vocabulary].sort(), levels);
  }

  /** Reconstruct a trained model from its serialized form (train once, infer later). */
  static load(serialized: SerializedMovementModel): TrainedMovementModel {
    const levels: CountLevels = serialized.levels.map((record) => {
      const level = new Map<string, Map<string, number>>();
      for (const [key, inner] of Object.entries(record)) {
        const distribution = new Map<string, number>();
        for (const [symbol, count] of Object.entries(inner)) {
          distribution.set(symbol, count);
        }
        level.set(key, distribution);
      }
      return level;
    });
    return new NgramMovementModel(serialized.backendId, serialized.order, [...serialized.vocabulary], levels);
  }
}

export type MovementEvaluation = {
  predictions: number;
  correct: number;
  accuracy: number;
  /** Accuracy restricted to action tokens — the movements we care about repeating. */
  actionPredictions: number;
  actionCorrect: number;
  actionAccuracy: number;
};

/**
 * Generalization / fidelity eval: walk each held-out sequence and measure how
 * often the model's next-symbol prediction matches the recorded movement.
 */
export function evaluateMovementModel(
  model: TrainedMovementModel,
  heldOut: MovementSequence[],
): MovementEvaluation {
  let predictions = 0;
  let correct = 0;
  let actionPredictions = 0;
  let actionCorrect = 0;

  for (const sequence of heldOut) {
    const symbols = [MOVEMENT_START, ...sequence.tokens.map((token) => token.symbol), MOVEMENT_END];
    for (let i = 1; i < symbols.length; i += 1) {
      const context = symbols.slice(0, i);
      const expected = symbols[i]!;
      const prediction = model.predict(context);
      const hit = prediction?.symbol === expected;
      predictions += 1;
      if (hit) {
        correct += 1;
      }
      if (expected.startsWith("action:")) {
        actionPredictions += 1;
        if (hit) {
          actionCorrect += 1;
        }
      }
    }
  }

  return {
    predictions,
    correct,
    accuracy: predictions === 0 ? 0 : correct / predictions,
    actionPredictions,
    actionCorrect,
    actionAccuracy: actionPredictions === 0 ? 0 : actionCorrect / actionPredictions,
  };
}
