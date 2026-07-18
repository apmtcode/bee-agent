// Deterministic reference backend for the local-movement model.
//
// An n-gram sequence model with stupid-backoff: it learns transition
// statistics over movement tokens at every context length up to `order`, then
// predicts the next movement by using the longest context it has data for and
// backing off to shorter contexts (and finally the unigram distribution) when a
// context is unseen. This is a real learnable model — it "repeats" recorded
// movements (the high-probability path reproduces training sequences) and
// "generalises" (backoff lets it act sensibly on novel prefixes) — while being
// fully deterministic and dependency-free, so it validates the whole pipeline in
// the cloud/CI. A real on-device small model can replace it behind the same
// `MovementModelBackend` interface.

import {
  MOVEMENT_END_TOKEN,
  MOVEMENT_START_TOKEN,
  movementFromToken,
  movementToken,
  type MovementDataset,
  type MovementEvent,
  type MovementGenerateOptions,
  type MovementModel,
  type MovementModelArtifact,
  type MovementModelBackend,
  type MovementPrediction,
  type MovementPredictionAlternative,
  type MovementTrainOptions,
} from "./movement-model.js";

export const NGRAM_BACKEND_NAME = "ngram-backoff";
const DEFAULT_ORDER = 3;
const CONTEXT_SEPARATOR = "";

/** grams[k] maps a k-token context key → (nextToken → count). k ranges 0..order. */
type GramTables = Map<string, Map<string, number>>[];

type NgramPayload = {
  order: number;
  grams: [string, [string, number][]][][];
};

function contextKey(tokens: string[]): string {
  return tokens.join(CONTEXT_SEPARATOR);
}

function increment(table: Map<string, number>, token: string): void {
  table.set(token, (table.get(token) ?? 0) + 1);
}

function emptyGrams(order: number): GramTables {
  return Array.from({ length: order + 1 }, () => new Map<string, Map<string, number>>());
}

export class NgramMovementModel implements MovementModel {
  readonly backend = NGRAM_BACKEND_NAME;

  constructor(
    private readonly order: number,
    private readonly grams: GramTables,
    private readonly vocabulary: string[],
    private readonly sequenceCount: number,
    private readonly eventCount: number,
    private readonly trainedAt?: string,
  ) {}

  predictNext(context: MovementEvent[]): MovementPrediction | undefined {
    const tokens = [MOVEMENT_START_TOKEN, ...context.map((event) => movementToken(event))];
    return this.predictFromTokens(tokens);
  }

  private predictFromTokens(tokens: string[]): MovementPrediction | undefined {
    const maxK = Math.min(this.order, tokens.length);
    for (let k = maxK; k >= 0; k -= 1) {
      const table = this.grams[k]?.get(contextKey(tokens.slice(tokens.length - k)));
      if (table && table.size > 0) {
        return this.buildPrediction(table, k);
      }
    }
    return undefined;
  }

  private buildPrediction(table: Map<string, number>, contextOrderUsed: number): MovementPrediction {
    let total = 0;
    for (const count of table.values()) {
      total += count;
    }
    // Deterministic ranking: probability desc, then token asc for stable ties.
    const ranked = [...table.entries()]
      .map(([token, count]): MovementPredictionAlternative => ({ token, probability: count / total }))
      .sort((a, b) => (b.probability - a.probability) || (a.token < b.token ? -1 : a.token > b.token ? 1 : 0));

    const best = ranked[0]!;
    const alternatives = ranked.slice(0, 5);
    return {
      token: best.token,
      event: movementFromToken(best.token, 0),
      probability: best.probability,
      contextOrderUsed,
      alternatives,
    };
  }

  generate(seed: MovementEvent[], options: MovementGenerateOptions = {}): MovementEvent[] {
    const maxSteps = options.maxSteps ?? 64;
    const tokens = [MOVEMENT_START_TOKEN, ...seed.map((event) => movementToken(event))];
    const generated: MovementEvent[] = [];
    let ts = seed.at(-1)?.ts ?? 0;

    for (let step = 0; step < maxSteps; step += 1) {
      const prediction = this.predictFromTokens(tokens);
      if (!prediction || prediction.token === MOVEMENT_END_TOKEN) {
        break;
      }
      tokens.push(prediction.token);
      ts += 1;
      const event = movementFromToken(prediction.token, ts);
      if (event) {
        generated.push(event);
      }
    }
    return generated;
  }

  toArtifact(): MovementModelArtifact {
    const grams: NgramPayload["grams"] = this.grams.map((table) =>
      [...table.entries()].map(([ctx, next]): [string, [string, number][]] => [ctx, [...next.entries()]]),
    );
    const payload: NgramPayload = { order: this.order, grams };
    return {
      version: 1,
      backend: this.backend,
      vocabulary: [...this.vocabulary],
      sequenceCount: this.sequenceCount,
      eventCount: this.eventCount,
      ...(this.trainedAt ? { trainedAt: this.trainedAt } : {}),
      payload,
    };
  }
}

export type NgramMovementBackendOptions = {
  /** Injectable clock so callers/tests control the `trainedAt` stamp. */
  now?: () => Date;
};

export class NgramMovementBackend implements MovementModelBackend {
  readonly name = NGRAM_BACKEND_NAME;

  constructor(private readonly options: NgramMovementBackendOptions = {}) {}

  train(dataset: MovementDataset, options: MovementTrainOptions = {}): NgramMovementModel {
    const order = Math.max(1, options.order ?? DEFAULT_ORDER);
    const grams = emptyGrams(order);
    const vocabulary = new Set<string>();
    let eventCount = 0;

    for (const sequence of dataset.sequences) {
      const tokens = [
        MOVEMENT_START_TOKEN,
        ...sequence.events.map((event) => movementToken(event)),
        MOVEMENT_END_TOKEN,
      ];
      eventCount += sequence.events.length;
      for (const token of tokens) {
        if (token !== MOVEMENT_START_TOKEN && token !== MOVEMENT_END_TOKEN) {
          vocabulary.add(token);
        }
      }
      // Record every (context of length k) → next-token transition, k = 0..order.
      for (let i = 1; i < tokens.length; i += 1) {
        const next = tokens[i]!;
        for (let k = 0; k <= order; k += 1) {
          if (i - k < 0) {
            break;
          }
          const key = contextKey(tokens.slice(i - k, i));
          const level = grams[k]!;
          let table = level.get(key);
          if (!table) {
            table = new Map<string, number>();
            level.set(key, table);
          }
          increment(table, next);
        }
      }
    }

    const trainedAt = this.options.now?.().toISOString();
    return new NgramMovementModel(
      order,
      grams,
      [...vocabulary].sort(),
      dataset.sequences.length,
      eventCount,
      trainedAt,
    );
  }

  load(artifact: MovementModelArtifact): NgramMovementModel {
    if (artifact.backend !== this.name) {
      throw new Error(`Cannot load artifact for backend "${artifact.backend}" with ${this.name}`);
    }
    const payload = artifact.payload as NgramPayload;
    const grams: GramTables = payload.grams.map((level) => {
      const table = new Map<string, Map<string, number>>();
      for (const [ctx, next] of level) {
        table.set(ctx, new Map(next));
      }
      return table;
    });
    return new NgramMovementModel(
      payload.order,
      grams,
      [...artifact.vocabulary],
      artifact.sequenceCount,
      artifact.eventCount,
      artifact.trainedAt,
    );
  }
}
