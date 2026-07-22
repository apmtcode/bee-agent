import {
  MOVEMENT_BOS,
  MOVEMENT_EOS,
  type MovementDataset,
  type MovementGenerateOptions,
  type MovementModel,
  type MovementModelBackend,
  type MovementPrediction,
  type MovementToken,
  type MovementTrainOptions,
  type SerializedMovementModel,
} from "../movement-model.js";

/**
 * Deterministic reference backend for the local-movement model: a variable-order
 * Markov (n-gram) model with Katz-style back-off.
 *
 * Why this backend exists: it is fully in-process, dependency-free and
 * deterministic, so the whole capture→dataset→train→replay→generalize loop can
 * be validated in the cloud with synthetic event streams (objective #2). It is
 * also a genuinely learning model, not a stub:
 *
 *  - **Repeat recorded movements** — greedy decoding from `<bos>` reproduces the
 *    most frequently recorded movement sequence.
 *  - **Generalize to related movements** — when the full k-token context was
 *    never observed, the model backs off to shorter contexts (k-1, …, 0) until
 *    it finds evidence, so it can predict plausible continuations for prefixes
 *    it never saw verbatim.
 *
 * A production on-device backend (e.g. MLX-trained transformer) implements the
 * same {@link MovementModelBackend} interface and is a drop-in replacement.
 */
export const MARKOV_BACKEND_NAME = "markov-ngram";
const DEFAULT_ORDER = 3;
const MAX_ORDER = 8;

/** Serialized counts keyed by context string; `""` is the unigram context. */
type CountTable = Record<string, Record<MovementToken, number>>;

type MarkovParameters = {
  counts: CountTable;
  vocabulary: MovementToken[];
};

export class MarkovMovementBackend implements MovementModelBackend {
  readonly name = MARKOV_BACKEND_NAME;

  train(dataset: MovementDataset, options?: MovementTrainOptions): MovementModel {
    const order = clampOrder(options?.order ?? DEFAULT_ORDER);
    const counts: CountTable = {};
    const vocabulary = new Set<MovementToken>();

    for (const sequence of dataset.sequences) {
      const padded = [MOVEMENT_BOS, ...sequence.tokens, MOVEMENT_EOS];
      for (const token of padded) {
        vocabulary.add(token);
      }
      for (let index = 1; index < padded.length; index += 1) {
        const target = padded[index]!;
        // Record this target under every context length 0..order, so back-off
        // has statistics at each granularity.
        for (let contextLength = 0; contextLength <= order; contextLength += 1) {
          if (contextLength > index) {
            break;
          }
          const context = padded.slice(index - contextLength, index);
          addCount(counts, contextKey(context), target);
        }
      }
    }

    return new MarkovMovementModel(order, { counts, vocabulary: [...vocabulary].sort() });
  }

  load(data: SerializedMovementModel): MovementModel {
    if (data.backend !== MARKOV_BACKEND_NAME) {
      throw new Error(`markov backend cannot load model produced by "${data.backend}"`);
    }
    const parameters = data.parameters as Partial<MarkovParameters>;
    return new MarkovMovementModel(clampOrder(data.order), {
      counts: parameters.counts ?? {},
      vocabulary: parameters.vocabulary ?? [],
    });
  }
}

class MarkovMovementModel implements MovementModel {
  readonly backend = MARKOV_BACKEND_NAME;

  constructor(
    readonly order: number,
    private readonly parameters: MarkovParameters,
  ) {}

  predictNext(context: MovementToken[]): MovementPrediction {
    // Katz back-off: try the longest available context, shrink until a table hit.
    for (let contextLength = Math.min(this.order, context.length); contextLength >= 0; contextLength -= 1) {
      const window = context.slice(context.length - contextLength);
      const table = this.parameters.counts[contextKey(window)];
      if (!table) {
        continue;
      }
      const best = argmax(table);
      if (best) {
        const total = Object.values(table).reduce((sum, value) => sum + value, 0);
        return {
          token: best.token,
          probability: total === 0 ? 0 : best.count / total,
          conditionedOrder: contextLength,
        };
      }
    }
    // Empty model: fall back to the end sentinel so generation terminates.
    return { token: MOVEMENT_EOS, probability: 0, conditionedOrder: 0 };
  }

  generate(prefix: MovementToken[], options?: MovementGenerateOptions): MovementToken[] {
    const maxLength = options?.maxLength ?? 256;
    const context: MovementToken[] = [MOVEMENT_BOS, ...stripSentinels(prefix)];
    const output: MovementToken[] = [...stripSentinels(prefix)];

    while (output.length < maxLength) {
      const prediction = this.predictNext(context);
      if (prediction.token === MOVEMENT_EOS || prediction.probability === 0) {
        break;
      }
      output.push(prediction.token);
      context.push(prediction.token);
    }
    return output;
  }

  serialize(): SerializedMovementModel {
    return {
      version: 1,
      backend: this.backend,
      order: this.order,
      parameters: {
        counts: this.parameters.counts,
        vocabulary: this.parameters.vocabulary,
      },
    };
  }
}

function clampOrder(order: number): number {
  if (!Number.isFinite(order)) {
    return DEFAULT_ORDER;
  }
  return Math.max(1, Math.min(MAX_ORDER, Math.floor(order)));
}

function contextKey(context: MovementToken[]): string {
  return context.join("");
}

function addCount(counts: CountTable, key: string, token: MovementToken): void {
  const table = (counts[key] ??= {});
  table[token] = (table[token] ?? 0) + 1;
}

function argmax(table: Record<MovementToken, number>): { token: MovementToken; count: number } | undefined {
  let best: { token: MovementToken; count: number } | undefined;
  // Deterministic: highest count wins; ties broken by lexicographically smaller token.
  for (const [token, count] of Object.entries(table)) {
    if (!best || count > best.count || (count === best.count && token < best.token)) {
      best = { token, count };
    }
  }
  return best;
}

function stripSentinels(tokens: MovementToken[]): MovementToken[] {
  return tokens.filter((token) => token !== MOVEMENT_BOS && token !== MOVEMENT_EOS);
}
