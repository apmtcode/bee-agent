import {
  MOVEMENT_CONTEXT_SEPARATOR,
  movementContextKey,
  type MovementPrediction,
  type MovementSequence,
  type MovementStep,
  type MovementTrainConfig,
  type MovementTrainingBackend,
  type TrainedMovementModel,
} from "./movement-backend.js";

/**
 * Deterministic, dependency-free movement backend used for cloud/CI validation.
 *
 * Models recorded movements as a variable-order Markov chain over movement
 * tokens. Training counts, for every context length `0..order`, how often each
 * token follows that context. Prediction uses **stupid-backoff**: it tries the
 * longest available context and falls back to shorter contexts until a match is
 * found, ending at the order-0 unigram distribution. This gives two properties
 * the objective requires:
 *
 *  - **Replay** — seeding a rollout with a recorded prefix reproduces the
 *    recorded movement sequence (the trained path is the most probable one).
 *  - **Generalization** — a novel context that shares a suffix with training
 *    still yields a sensible next movement via backoff, and the reported
 *    `contextOrderUsed` reveals that generalization happened.
 *
 * Everything is deterministic: ties are broken lexicographically by token, and
 * no clock or RNG is used. The real on-device backend (small local model) plugs
 * in behind the same {@link MovementTrainingBackend} interface.
 */
export class MarkovMovementBackend implements MovementTrainingBackend {
  readonly id = "markov-v1";

  constructor(private readonly defaultOrder = 2) {}

  async train(sequences: MovementSequence[], config: MovementTrainConfig = {}): Promise<TrainedMovementModel> {
    const order = Math.max(1, Math.floor(config.order ?? this.defaultOrder));
    const transitions: Record<string, Record<string, number>> = {};
    const representatives: Record<string, MovementStep> = {};
    const vocabulary = new Set<string>();
    let trainedStepCount = 0;

    for (const sequence of sequences) {
      for (let index = 0; index < sequence.length; index += 1) {
        const step = sequence[index];
        if (!step) {
          continue;
        }
        vocabulary.add(step.token);
        // Last-writer-wins keeps the representative deterministic across a run.
        representatives[step.token] = normalizeStep(step);
        trainedStepCount += 1;

        // Record this token against every context length from 0..order.
        for (let contextLength = 0; contextLength <= order; contextLength += 1) {
          if (index - contextLength < 0) {
            break;
          }
          const context = sequence.slice(index - contextLength, index);
          const key = movementContextKey(context);
          const bucket = (transitions[key] ??= {});
          bucket[step.token] = (bucket[step.token] ?? 0) + 1;
        }
      }
    }

    return {
      version: 1,
      backendId: this.id,
      order,
      vocabulary: [...vocabulary].sort(),
      transitions,
      representatives,
      trainedSequenceCount: sequences.length,
      trainedStepCount,
    };
  }

  predictNext(model: TrainedMovementModel, context: MovementStep[]): MovementPrediction | undefined {
    const maxContext = Math.min(model.order, context.length);
    for (let contextLength = maxContext; contextLength >= 0; contextLength -= 1) {
      const slice = context.slice(context.length - contextLength);
      const key = movementContextKey(slice);
      const bucket = model.transitions[key];
      if (!bucket) {
        continue;
      }
      const candidates = rankCandidates(bucket);
      const best = candidates[0];
      if (!best) {
        continue;
      }
      const step = model.representatives[best.token];
      if (!step) {
        continue;
      }
      return {
        token: best.token,
        step,
        probability: best.probability,
        contextOrderUsed: contextLength,
        candidates,
      };
    }
    return undefined;
  }
}

function normalizeStep(step: MovementStep): MovementStep {
  return {
    token: step.token,
    tool: step.tool,
    summary: step.summary,
    ...(step.metadata ? { metadata: step.metadata } : {}),
  };
}

/**
 * Rank next-token candidates by count, breaking ties lexicographically by token
 * for full determinism, and attach normalized probabilities.
 */
function rankCandidates(bucket: Record<string, number>): Array<{ token: string; probability: number }> {
  const total = Object.values(bucket).reduce((sum, count) => sum + count, 0);
  if (total <= 0) {
    return [];
  }
  return Object.entries(bucket)
    .map(([token, count]) => ({ token, probability: count / total }))
    .sort((a, b) => {
      if (b.probability !== a.probability) {
        return b.probability - a.probability;
      }
      return a.token < b.token ? -1 : a.token > b.token ? 1 : 0;
    });
}

/** Re-exported for callers that key custom transition maps consistently. */
export { MOVEMENT_CONTEXT_SEPARATOR };
