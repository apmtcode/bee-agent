import type {
  MovementCandidate,
  MovementDataset,
  MovementModelBackend,
  MovementPrediction,
  MovementToken,
  MovementTrainOptions,
  SerializedMovementModel,
  TrainedMovementModel,
} from "./model-backend.js";

/**
 * Deterministic variable-order Markov movement backend.
 *
 * Trains a set of n-gram transition tables (orders 1..maxOrder plus an
 * unconditional order-0 prior) over movement-token sequences. Prediction uses
 * "stupid backoff": it conditions on the longest context suffix that was
 * observed in training and falls back to shorter contexts otherwise, which is
 * what lets the model *repeat* recorded movements (high-order exact match) and
 * *generalize* to new-but-related movements (lower-order backoff match).
 *
 * It is fully in-process and deterministic — no OS, network, or GPU — so the
 * capture → dataset → train → infer pipeline can be validated in the cloud
 * with synthetic event streams. Ties are broken by descending count then
 * ascending token, so identical datasets always yield identical models.
 */

const CONTEXT_DELIMITER = "";
export const DEFAULT_MARKOV_ORDER = 3;

type TransitionCounts = Record<MovementToken, number>;
/** order -> contextKey -> { nextToken -> count }. Order 0 uses key "". */
type OrderTables = Map<number, Map<string, TransitionCounts>>;

type MarkovSerialized = SerializedMovementModel & {
  backendId: string;
  version: 1;
  maxOrder: number;
  tables: Array<{ order: number; entries: Array<{ context: string; counts: TransitionCounts }> }>;
};

export const MARKOV_BACKEND_ID = "markov-ngram";

export class MarkovMovementBackend implements MovementModelBackend {
  readonly id = MARKOV_BACKEND_ID;

  constructor(private readonly defaultMaxOrder: number = DEFAULT_MARKOV_ORDER) {}

  async train(dataset: MovementDataset, options?: MovementTrainOptions): Promise<TrainedMovementModel> {
    const maxOrder = Math.max(1, options?.maxOrder ?? this.defaultMaxOrder);
    const tables: OrderTables = new Map();
    for (let order = 0; order <= maxOrder; order += 1) {
      tables.set(order, new Map());
    }

    for (const sample of dataset.samples) {
      const { tokens } = sample;
      for (let i = 0; i < tokens.length; i += 1) {
        const next = tokens[i]!;
        for (let order = 0; order <= maxOrder; order += 1) {
          if (i - order < 0) {
            break;
          }
          const context = tokens.slice(i - order, i);
          addTransition(tables.get(order)!, contextKey(context), next);
        }
      }
    }

    return new MarkovMovementModel(maxOrder, tables);
  }

  restore(serialized: SerializedMovementModel): TrainedMovementModel {
    const model = serialized as MarkovSerialized;
    if (model.backendId !== MARKOV_BACKEND_ID) {
      throw new Error(`markov backend cannot restore model from backend "${model.backendId}"`);
    }
    const tables: OrderTables = new Map();
    for (const table of model.tables) {
      const contexts = new Map<string, TransitionCounts>();
      for (const entry of table.entries) {
        contexts.set(entry.context, { ...entry.counts });
      }
      tables.set(table.order, contexts);
    }
    return new MarkovMovementModel(model.maxOrder, tables);
  }
}

class MarkovMovementModel implements TrainedMovementModel {
  readonly backendId = MARKOV_BACKEND_ID;

  constructor(private readonly maxOrder: number, private readonly tables: OrderTables) {}

  predict(context: MovementToken[]): MovementPrediction {
    const startOrder = Math.min(this.maxOrder, context.length);
    for (let order = startOrder; order >= 0; order -= 1) {
      const table = this.tables.get(order);
      if (!table) {
        continue;
      }
      const suffix = order === 0 ? [] : context.slice(context.length - order);
      const counts = table.get(contextKey(suffix));
      if (!counts) {
        continue;
      }
      const candidates = rankCandidates(counts);
      if (candidates.length === 0) {
        continue;
      }
      return {
        token: candidates[0]!.token,
        confidence: candidates[0]!.probability,
        matchedOrder: order,
        candidates,
      };
    }
    return { token: null, confidence: 0, matchedOrder: 0, candidates: [] };
  }

  generate(seed: MovementToken[], steps: number): MovementToken[] {
    const generated: MovementToken[] = [];
    const context = [...seed];
    for (let i = 0; i < steps; i += 1) {
      const prediction = this.predict(context);
      if (prediction.token === null) {
        break;
      }
      generated.push(prediction.token);
      context.push(prediction.token);
    }
    return generated;
  }

  serialize(): SerializedMovementModel {
    const tables = [...this.tables.entries()]
      .sort((a, b) => a[0] - b[0])
      .map(([order, contexts]) => ({
        order,
        entries: [...contexts.entries()]
          .sort((a, b) => (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0))
          .map(([context, counts]) => ({ context, counts: { ...counts } })),
      }));
    const serialized: MarkovSerialized = {
      backendId: MARKOV_BACKEND_ID,
      version: 1,
      maxOrder: this.maxOrder,
      tables,
    };
    return serialized;
  }
}

function contextKey(context: MovementToken[]): string {
  return context.join(CONTEXT_DELIMITER);
}

function addTransition(table: Map<string, TransitionCounts>, key: string, next: MovementToken): void {
  const counts = table.get(key) ?? {};
  counts[next] = (counts[next] ?? 0) + 1;
  table.set(key, counts);
}

/** Rank next-token candidates by descending count, ties broken by token. */
function rankCandidates(counts: TransitionCounts): MovementCandidate[] {
  const total = Object.values(counts).reduce((sum, count) => sum + count, 0);
  if (total === 0) {
    return [];
  }
  return Object.entries(counts)
    .sort((a, b) => (b[1] - a[1]) || (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0))
    .map(([token, count]) => ({ token, probability: count / total }));
}
