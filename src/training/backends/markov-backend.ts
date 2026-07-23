import {
  CONTEXT_DELIMITER,
  LocalModelBackendRegistry,
  SEQUENCE_END_TOKEN,
  SEQUENCE_START_TOKEN,
  type LocalModelArtifact,
  type LocalModelBackend,
  type LocalModelInferenceRequest,
  type LocalModelInferenceResult,
  type LocalModelTrainingRequest,
  type MovementToken,
} from "../backend.js";

/**
 * Deterministic variable-order Markov backend for movement sequences.
 *
 * This is the in-process reference/mock backend for the movement-learning
 * subsystem. It "post-trains" on reviewed movement sequences by counting
 * n-gram transitions (orders 1..N) and predicts continuations with Katz-style
 * backoff: the longest matching context wins, falling back to shorter contexts
 * and finally to the global unigram distribution.
 *
 * Behaviour:
 * - **Repeat**: prompts that match a recorded prefix reproduce the recorded
 *   continuation exactly (highest-order context hits).
 * - **Generalize**: novel-but-related prompts back off to shorter shared
 *   contexts (or the unigram prior), producing plausible continuations rather
 *   than failing.
 *
 * Fully deterministic: argmax ties break lexicographically, so identical inputs
 * always yield identical models and predictions (cloud/CI reproducible).
 */

const DEFAULT_ORDER = 2;
const DEFAULT_MAX_TOKENS = 64;

type TransitionCounts = Record<MovementToken, number>;

type MarkovWeights = {
  order: number;
  /** contexts[String(k)][joinedContext][nextToken] = count, for k in 1..order. */
  contexts: Record<string, Record<string, TransitionCounts>>;
  unigram: TransitionCounts;
};

export class MarkovMovementBackend implements LocalModelBackend {
  readonly name = "markov";

  async train(request: LocalModelTrainingRequest): Promise<LocalModelArtifact> {
    const order = Math.max(1, Math.floor(request.order ?? DEFAULT_ORDER));
    const contexts: MarkovWeights["contexts"] = {};
    const unigram: TransitionCounts = {};
    const vocabulary = new Set<MovementToken>();
    let tokenCount = 0;

    for (const sequence of request.sequences) {
      // Pad the front with START sentinels so short prompts have real contexts,
      // and terminate with END so the model learns where sequences stop.
      const padded = [
        ...Array<MovementToken>(order).fill(SEQUENCE_START_TOKEN),
        ...sequence.tokens,
        SEQUENCE_END_TOKEN,
      ];

      for (let index = order; index < padded.length; index += 1) {
        const target = padded[index]!;
        vocabulary.add(target);
        tokenCount += 1;
        increment(unigram, target);

        for (let k = 1; k <= order; k += 1) {
          const contextTokens = padded.slice(index - k, index);
          const contextKey = contextTokens.join(CONTEXT_DELIMITER);
          const table = (contexts[String(k)] ??= {});
          increment((table[contextKey] ??= {}), target);
        }
      }
    }

    vocabulary.delete(SEQUENCE_END_TOKEN);

    const weights: MarkovWeights = { order, contexts, unigram };
    return {
      backend: this.name,
      jobId: request.jobId,
      version: 1,
      order,
      vocabulary: [...vocabulary].sort(),
      sequenceCount: request.sequences.length,
      tokenCount,
      weights,
    };
  }

  async infer(
    model: LocalModelArtifact,
    request: LocalModelInferenceRequest,
  ): Promise<LocalModelInferenceResult> {
    const weights = model.weights as MarkovWeights;
    const order = weights.order;
    const maxTokens = Math.max(0, request.maxTokens ?? DEFAULT_MAX_TOKENS);

    const generated: MovementToken[] = [];
    const history = [...request.prompt];
    let usedBackoff = false;
    let terminated = false;

    for (let step = 0; step < maxTokens; step += 1) {
      const padded = [
        ...Array<MovementToken>(order).fill(SEQUENCE_START_TOKEN),
        ...history,
      ];
      const prediction = predictNext(weights, padded);
      if (prediction.backoff) {
        usedBackoff = true;
      }
      if (prediction.token === undefined || prediction.token === SEQUENCE_END_TOKEN) {
        terminated = true;
        break;
      }
      generated.push(prediction.token);
      history.push(prediction.token);
    }

    return { tokens: generated, usedBackoff, terminated };
  }
}

function increment(table: TransitionCounts, token: MovementToken): void {
  table[token] = (table[token] ?? 0) + 1;
}

function predictNext(
  weights: MarkovWeights,
  paddedContext: MovementToken[],
): { token: MovementToken | undefined; backoff: boolean } {
  for (let k = weights.order; k >= 1; k -= 1) {
    const contextKey = paddedContext.slice(paddedContext.length - k).join(CONTEXT_DELIMITER);
    const table = weights.contexts[String(k)]?.[contextKey];
    const best = argmax(table);
    if (best !== undefined) {
      return { token: best, backoff: k < weights.order };
    }
  }
  return { token: argmax(weights.unigram), backoff: true };
}

/** Highest-count token; ties broken lexicographically for determinism. */
function argmax(table: TransitionCounts | undefined): MovementToken | undefined {
  if (!table) {
    return undefined;
  }
  let bestToken: MovementToken | undefined;
  let bestCount = -1;
  for (const token of Object.keys(table).sort()) {
    const count = table[token]!;
    if (count > bestCount) {
      bestCount = count;
      bestToken = token;
    }
  }
  return bestToken;
}

/** Convenience factory used when wiring the default backend registry. */
export function createMarkovMovementBackend(): MarkovMovementBackend {
  return new MarkovMovementBackend();
}

/**
 * A registry pre-seeded with the built-in deterministic backend. Real on-device
 * backends can `.register()` themselves onto this without changing call sites.
 */
export function createDefaultLocalModelBackendRegistry(): LocalModelBackendRegistry {
  return new LocalModelBackendRegistry().register(createMarkovMovementBackend());
}
