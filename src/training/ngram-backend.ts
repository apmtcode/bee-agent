import { MOVEMENT_END, MOVEMENT_START, type MovementDataset } from "./movement-dataset.js";
import type {
  LocalMovementModelBackend,
  MovementPrediction,
  MovementTrainingConfig,
  SerializedMovementModel,
  TrainedMovementModel,
} from "./model-backend.js";

export const NGRAM_BACKEND_ID = "ngram-markov";

type CountMap = Map<string, Map<string, number>>;

/** Control char that cannot appear in movement symbols, so keys are reversible. */
const CONTEXT_SEP = String.fromCharCode(1);

function contextKey(context: string[]): string {
  return context.join(CONTEXT_SEP);
}

function contextFromKey(key: string): string[] {
  return key === "" ? [] : key.split(CONTEXT_SEP);
}

/**
 * Deterministic n-gram Markov model with stupid-backoff.
 *
 * Training counts, for every order 0..maxOrder, how often each symbol follows
 * each context window. Inference conditions on the longest matching suffix of
 * the given context and backs off to shorter windows when the full context is
 * unseen — which is exactly how it *generalizes* to new-but-related movements
 * while still *reproducing* recorded ones when the full context is known.
 */
class NgramMovementModel implements TrainedMovementModel {
  readonly backendId = NGRAM_BACKEND_ID;

  constructor(
    readonly order: number,
    /** grams[o] maps an order-`o` context key to symbol counts. */
    private readonly grams: CountMap[],
    readonly vocabulary: string[],
  ) {}

  private lookup(context: string[]): { counts: Map<string, number>; order: number } | undefined {
    const maxOrder = Math.min(context.length, this.order);
    for (let usedOrder = maxOrder; usedOrder >= 0; usedOrder -= 1) {
      const suffix = usedOrder === 0 ? [] : context.slice(context.length - usedOrder);
      const counts = this.grams[usedOrder]?.get(contextKey(suffix));
      if (counts && counts.size > 0) {
        return { counts, order: usedOrder };
      }
    }
    return undefined;
  }

  distribution(context: string[]): MovementPrediction[] {
    const hit = this.lookup(context);
    if (!hit) {
      return [];
    }
    let total = 0;
    for (const count of hit.counts.values()) {
      total += count;
    }
    return [...hit.counts.entries()]
      .map(([symbol, count]) => ({ symbol, probability: count / total, order: hit.order }))
      .sort((a, b) => (b.probability !== a.probability ? b.probability - a.probability : a.symbol.localeCompare(b.symbol)));
  }

  predict(context: string[]): MovementPrediction | undefined {
    return this.distribution(context)[0];
  }

  generate(seed: string[] = [MOVEMENT_START], options?: { maxLength?: number }): string[] {
    const maxLength = options?.maxLength ?? 256;
    const history = seed.length > 0 ? [...seed] : [MOVEMENT_START];
    const emitted: string[] = [];
    for (let step = 0; step < maxLength; step += 1) {
      const next = this.predict(history);
      if (!next || next.symbol === MOVEMENT_END) {
        break;
      }
      emitted.push(next.symbol);
      history.push(next.symbol);
    }
    return emitted;
  }

  serialize(): SerializedMovementModel {
    const grams: SerializedMovementModel["grams"] = [];
    for (let usedOrder = 0; usedOrder <= this.order; usedOrder += 1) {
      const table = this.grams[usedOrder];
      if (!table) {
        continue;
      }
      for (const [key, counts] of table.entries()) {
        grams.push({
          context: contextFromKey(key),
          next: [...counts.entries()]
            .map(([symbol, count]) => ({ symbol, count }))
            .sort((a, b) => a.symbol.localeCompare(b.symbol)),
        });
      }
    }
    return { backendId: this.backendId, version: 1, order: this.order, vocabulary: [...this.vocabulary], grams };
  }
}

function emptyGrams(order: number): CountMap[] {
  return Array.from({ length: order + 1 }, () => new Map<string, Map<string, number>>());
}

function increment(table: CountMap, context: string[], symbol: string): void {
  const key = contextKey(context);
  let counts = table.get(key);
  if (!counts) {
    counts = new Map<string, number>();
    table.set(key, counts);
  }
  counts.set(symbol, (counts.get(symbol) ?? 0) + 1);
}

function pruneBelow(grams: CountMap[], minCount: number): void {
  if (minCount <= 1) {
    return;
  }
  for (const table of grams) {
    for (const [key, counts] of [...table.entries()]) {
      for (const [symbol, count] of [...counts.entries()]) {
        if (count < minCount) {
          counts.delete(symbol);
        }
      }
      if (counts.size === 0) {
        table.delete(key);
      }
    }
  }
}

export class NgramMovementBackend implements LocalMovementModelBackend {
  readonly id = NGRAM_BACKEND_ID;

  async train(dataset: MovementDataset, config: MovementTrainingConfig): Promise<TrainedMovementModel> {
    const order = Math.max(0, Math.floor(config.order));
    const grams = emptyGrams(order);

    for (const sequence of dataset.sequences) {
      const symbols = [MOVEMENT_START, ...sequence.tokens.map((token) => token.symbol), MOVEMENT_END];
      for (let position = 1; position < symbols.length; position += 1) {
        const target = symbols[position];
        for (let usedOrder = 0; usedOrder <= order; usedOrder += 1) {
          if (usedOrder > position) {
            break;
          }
          const context = usedOrder === 0 ? [] : symbols.slice(position - usedOrder, position);
          increment(grams[usedOrder], context, target);
        }
      }
    }

    pruneBelow(grams, config.minCount ?? 1);
    return new NgramMovementModel(order, grams, [...dataset.vocabulary]);
  }

  load(serialized: SerializedMovementModel): TrainedMovementModel {
    const grams = emptyGrams(serialized.order);
    for (const gram of serialized.grams) {
      const usedOrder = gram.context.length;
      if (usedOrder > serialized.order) {
        continue;
      }
      const counts = new Map<string, number>();
      for (const entry of gram.next) {
        counts.set(entry.symbol, entry.count);
      }
      grams[usedOrder].set(contextKey(gram.context), counts);
    }
    return new NgramMovementModel(serialized.order, grams, [...serialized.vocabulary]);
  }
}

/** Construct the built-in deterministic movement backend. */
export function createDefaultMovementBackend(): NgramMovementBackend {
  return new NgramMovementBackend();
}
