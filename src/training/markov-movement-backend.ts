import {
  MOVEMENT_BOS,
  MOVEMENT_EOS,
  type MovementDataset,
  type MovementModel,
  type MovementModelBackend,
  type MovementPrediction,
  type MovementTrainOptions,
  type SerializedMovementModel,
} from "./movement-model.js";

/**
 * Deterministic, in-process movement backend used to validate the
 * capture -> dataset -> train -> infer loop without any GPU or real OS input.
 *
 * It learns a variable-order Markov model over action tokens with **stupid
 * backoff**: to predict the next action it conditions on the longest recent
 * context it has seen, falling back to shorter contexts (down to the unigram
 * marginal) when the full context is novel. That backoff is exactly what gives
 * the model its generalization to new-but-related movement sequences (objective
 * #2 part d): an unseen prefix still yields a sensible continuation because a
 * shorter suffix of it was observed during training.
 *
 * Everything is deterministic — argmax with a lexicographic tie-break — so tests
 * are stable and the same reviewed dataset always yields the same policy. A real
 * on-device small model can drop in behind the same `MovementModelBackend`
 * interface without changing any call site.
 */

const CONTEXT_SEPARATOR = "";
const DEFAULT_ORDER = 2;
const MAX_ORDER = 8;

function contextKey(tokens: readonly string[]): string {
  return tokens.join(CONTEXT_SEPARATOR);
}

type TransitionTable = Map<string, Map<string, number>>;

function selectBest(
  counts: Map<string, number>,
): { token: string; probability: number; alternatives: Array<{ token: string; probability: number }> } {
  let total = 0;
  for (const value of counts.values()) {
    total += value;
  }
  const ranked = [...counts.entries()].sort((a, b) => {
    if (b[1] !== a[1]) {
      return b[1] - a[1];
    }
    return a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0;
  });
  const alternatives = ranked.map(([token, count]) => ({
    token,
    probability: total > 0 ? count / total : 0,
  }));
  const best = alternatives[0] ?? { token: MOVEMENT_EOS, probability: 0 };
  return { token: best.token, probability: best.probability, alternatives };
}

class MarkovMovementModel implements MovementModel {
  readonly backendId: string;
  readonly vocabulary: readonly string[];
  private readonly order: number;
  private readonly transitions: TransitionTable;

  constructor(params: {
    backendId: string;
    order: number;
    vocabulary: readonly string[];
    transitions: TransitionTable;
  }) {
    this.backendId = params.backendId;
    this.order = params.order;
    this.vocabulary = params.vocabulary;
    this.transitions = params.transitions;
  }

  predictNext(context: readonly string[]): MovementPrediction {
    const maxK = Math.min(this.order, context.length);
    for (let k = maxK; k >= 1; k -= 1) {
      const key = contextKey(context.slice(context.length - k));
      const counts = this.transitions.get(key);
      if (counts && counts.size > 0) {
        const best = selectBest(counts);
        return {
          token: best.token,
          probability: best.probability,
          backoffOrder: k,
          alternatives: best.alternatives,
        };
      }
    }
    const unigram = this.transitions.get("");
    if (unigram && unigram.size > 0) {
      const best = selectBest(unigram);
      return {
        token: best.token,
        probability: best.probability,
        backoffOrder: 0,
        alternatives: best.alternatives,
      };
    }
    return { token: MOVEMENT_EOS, probability: 0, backoffOrder: 0, alternatives: [] };
  }

  generate(seed: readonly string[], steps: number): string[] {
    // Left-pad with BOS so a cold start (or short seed) is conditioned on the
    // learned start-of-sequence distribution, matching how the model was
    // trained. As generation proceeds the sentinels fall out of the window.
    const padding = Math.max(0, this.order - seed.length);
    const context = [...Array.from({ length: padding }, () => MOVEMENT_BOS), ...seed];
    const generated: string[] = [];
    for (let step = 0; step < steps; step += 1) {
      const prediction = this.predictNext(context);
      if (prediction.token === MOVEMENT_EOS || prediction.probability === 0) {
        break;
      }
      generated.push(prediction.token);
      context.push(prediction.token);
    }
    return generated;
  }

  serialize(): SerializedMovementModel {
    const transitions: Record<string, Record<string, number>> = {};
    for (const [key, counts] of this.transitions) {
      const entry: Record<string, number> = {};
      for (const [token, count] of counts) {
        entry[token] = count;
      }
      transitions[key] = entry;
    }
    return {
      version: 1,
      backendId: this.backendId,
      order: this.order,
      vocabulary: [...this.vocabulary],
      transitions,
    };
  }
}

export class MarkovMovementBackend implements MovementModelBackend {
  readonly id = "markov-backoff";

  async train(dataset: MovementDataset, options?: MovementTrainOptions): Promise<MovementModel> {
    const order = Math.min(MAX_ORDER, Math.max(1, options?.order ?? DEFAULT_ORDER));
    const transitions: TransitionTable = new Map();

    const record = (key: string, token: string, weight: number): void => {
      let counts = transitions.get(key);
      if (!counts) {
        counts = new Map();
        transitions.set(key, counts);
      }
      counts.set(token, (counts.get(token) ?? 0) + weight);
    };

    for (const sample of dataset.samples) {
      const weight = sample.weight > 0 ? sample.weight : 1;
      // Left-pad with BOS so the first real action is predictable from a start
      // context, and terminate with EOS so generation knows when to stop.
      const sequence = [
        ...Array.from({ length: order }, () => MOVEMENT_BOS),
        ...sample.tokens,
        MOVEMENT_EOS,
      ];
      for (let i = order; i < sequence.length; i += 1) {
        const target = sequence[i]!;
        // Record every backoff order from `order` down to the unigram base.
        for (let k = order; k >= 1; k -= 1) {
          const context = sequence.slice(i - k, i);
          record(contextKey(context), target, weight);
        }
        record("", target, weight);
      }
    }

    return new MarkovMovementModel({
      backendId: this.id,
      order,
      vocabulary: [...dataset.vocabulary],
      transitions,
    });
  }

  load(serialized: SerializedMovementModel): MovementModel {
    const transitions: TransitionTable = new Map();
    for (const [key, entry] of Object.entries(serialized.transitions)) {
      const counts = new Map<string, number>();
      for (const [token, count] of Object.entries(entry)) {
        counts.set(token, count);
      }
      transitions.set(key, counts);
    }
    return new MarkovMovementModel({
      backendId: serialized.backendId,
      order: serialized.order,
      vocabulary: [...serialized.vocabulary],
      transitions,
    });
  }
}

/** Convenience: a registry pre-populated with the deterministic backend. */
export function createDefaultMovementBackend(): MarkovMovementBackend {
  return new MarkovMovementBackend();
}
