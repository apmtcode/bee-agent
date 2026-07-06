/**
 * In-process, backend-pluggable movement-policy learning.
 *
 * This is the cloud-runnable core of standing objective #2 (local-movement
 * learning subsystem) pieces (c) "post-train a local model to repeat the
 * recorded movements" and (d) "generalize to perform new but related
 * movements". The heavyweight, on-device MLX/axolotl path lives in
 * ./runner.ts; this module provides a small, deterministic, *trainable* policy
 * that can learn from a movement dataset and do inference entirely in-process,
 * so the whole capture -> dataset -> train -> replay -> generalize loop can be
 * exercised and tested without a real machine or GPU.
 *
 * The backend is pluggable: any {@link MovementPolicyBackend} can be registered
 * and swapped in (see {@link MovementPolicyBackendRegistry}). The bundled
 * {@link NGramMovementBackend} is the deterministic reference/mock backend — an
 * n-gram model with stupid-backoff smoothing, which is what lets it generalize
 * to unseen-but-related contexts. A real on-device small model implements the
 * same interface (documented seam below).
 */

/** A single captured movement, channel-typed (mouse/keyboard/window/etc). */
export type MovementEvent = {
  ts: number;
  channel: "pointer" | "keyboard" | "window" | "tool" | "observation";
  action: string;
  target?: string;
  value?: string;
};

/** An ordered sequence of movement events for one recorded episode. */
export type MovementTrajectory = {
  id: string;
  events: MovementEvent[];
};

/** A replayable dataset — the training/eval unit. */
export type MovementDataset = {
  version: 1;
  trajectories: MovementTrajectory[];
};

/** A tokenized movement — the atomic prediction target. */
export type MovementToken = string;

export type MovementTrainingConfig = {
  /** Maximum context length (n-1). Higher = more specific, less general. */
  maxOrder?: number;
  /** Include the event's `value` field in the token vocabulary. */
  includeValue?: boolean;
};

export type MovementPrediction = {
  token: MovementToken;
  /** Conditional probability of `token` under the matched context. */
  probability: number;
  /** How many context tokens actually matched (after backoff). */
  matchedOrder: number;
};

/** Serialized model — round-trippable so a "trained model" can be persisted. */
export type MovementModelSnapshot = {
  version: 1;
  backendId: string;
  maxOrder: number;
  includeValue: boolean;
  vocabulary: MovementToken[];
  grams: Array<{
    order: number;
    context: string;
    successors: Array<{ token: MovementToken; count: number }>;
  }>;
};

/** A trained policy: predict the next movement, roll out a sequence, persist. */
export interface MovementPolicyModel {
  readonly backendId: string;
  readonly vocabularySize: number;
  /** Predict the next token given a token context (may be empty). */
  predict(context: MovementToken[]): MovementPrediction | undefined;
  /** Autoregressively roll out `steps` tokens from `seed` (replay/generalize). */
  rollout(seed: MovementToken[], steps: number): MovementToken[];
  serialize(): MovementModelSnapshot;
}

/**
 * Pluggable training backend. Implement this to add a new movement-learning
 * backend (e.g. a real on-device small model). `train` must be deterministic
 * for the reference backend so cloud/CI tests are stable.
 */
export interface MovementPolicyBackend {
  readonly id: string;
  train(dataset: MovementDataset, config?: MovementTrainingConfig): MovementPolicyModel;
  /** Reconstruct a model from a snapshot this backend produced. */
  load(snapshot: MovementModelSnapshot): MovementPolicyModel;
}

const TOKEN_FIELD_SEP = ":";
const CONTEXT_SEP = "|";

/** Deterministic tokenizer: `channel:action:target[:value]`. */
export function tokenizeMovementEvent(
  event: MovementEvent,
  options: { includeValue?: boolean } = {},
): MovementToken {
  const parts = [event.channel, event.action, event.target ?? ""];
  if (options.includeValue) {
    parts.push(event.value ?? "");
  }
  return parts.join(TOKEN_FIELD_SEP);
}

export function tokenizeMovementTrajectory(
  trajectory: MovementTrajectory,
  options: { includeValue?: boolean } = {},
): MovementToken[] {
  return [...trajectory.events]
    .sort((a, b) => a.ts - b.ts)
    .map((event) => tokenizeMovementEvent(event, options));
}

class NGramMovementModel implements MovementPolicyModel {
  readonly backendId: string;
  private readonly maxOrder: number;
  private readonly includeValue: boolean;
  private readonly vocabulary: MovementToken[];
  /** grams[k] maps a context of length k to successor counts. k in [0, maxOrder]. */
  private readonly grams: Array<Map<string, Map<MovementToken, number>>>;

  constructor(params: {
    backendId: string;
    maxOrder: number;
    includeValue: boolean;
    vocabulary: MovementToken[];
    grams: Array<Map<string, Map<MovementToken, number>>>;
  }) {
    this.backendId = params.backendId;
    this.maxOrder = params.maxOrder;
    this.includeValue = params.includeValue;
    this.vocabulary = params.vocabulary;
    this.grams = params.grams;
  }

  get vocabularySize(): number {
    return this.vocabulary.length;
  }

  predict(context: MovementToken[]): MovementPrediction | undefined {
    for (let order = Math.min(this.maxOrder, context.length); order >= 0; order -= 1) {
      const table = this.grams[order];
      if (!table) {
        continue;
      }
      const key = order === 0 ? "" : context.slice(context.length - order).join(CONTEXT_SEP);
      const successors = table.get(key);
      if (!successors || successors.size === 0) {
        continue;
      }
      const best = argmax(successors);
      return { token: best.token, probability: best.count / best.total, matchedOrder: order };
    }
    return undefined;
  }

  rollout(seed: MovementToken[], steps: number): MovementToken[] {
    const out: MovementToken[] = [];
    const context = [...seed];
    for (let i = 0; i < steps; i += 1) {
      const prediction = this.predict(context);
      if (!prediction) {
        break;
      }
      out.push(prediction.token);
      context.push(prediction.token);
    }
    return out;
  }

  serialize(): MovementModelSnapshot {
    const grams: MovementModelSnapshot["grams"] = [];
    for (let order = 0; order <= this.maxOrder; order += 1) {
      const table = this.grams[order];
      if (!table) {
        continue;
      }
      for (const [context, successors] of sortedEntries(table)) {
        grams.push({
          order,
          context,
          successors: [...successors.entries()]
            .sort((a, b) => (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0))
            .map(([token, count]) => ({ token, count })),
        });
      }
    }
    return {
      version: 1,
      backendId: this.backendId,
      maxOrder: this.maxOrder,
      includeValue: this.includeValue,
      vocabulary: [...this.vocabulary].sort(),
      grams,
    };
  }
}

/**
 * Deterministic reference backend: an n-gram policy with stupid-backoff.
 * Learning = counting successor frequencies at every context length; inference
 * = pick the most-frequent successor of the longest matching context, backing
 * off to shorter contexts (and finally the global unigram) when the exact
 * context was never seen. The backoff is what generalizes to new-but-related
 * movement sequences.
 */
export class NGramMovementBackend implements MovementPolicyBackend {
  readonly id = "ngram";

  train(dataset: MovementDataset, config: MovementTrainingConfig = {}): MovementPolicyModel {
    const maxOrder = Math.max(0, config.maxOrder ?? 3);
    const includeValue = config.includeValue ?? false;
    const grams: Array<Map<string, Map<MovementToken, number>>> = Array.from(
      { length: maxOrder + 1 },
      () => new Map<string, Map<MovementToken, number>>(),
    );
    const vocabulary = new Set<MovementToken>();

    for (const trajectory of dataset.trajectories) {
      const tokens = tokenizeMovementTrajectory(trajectory, { includeValue });
      for (let i = 0; i < tokens.length; i += 1) {
        const token = tokens[i]!;
        vocabulary.add(token);
        for (let order = 0; order <= maxOrder; order += 1) {
          if (i - order < 0) {
            break;
          }
          const key = order === 0 ? "" : tokens.slice(i - order, i).join(CONTEXT_SEP);
          increment(grams[order]!, key, token);
        }
      }
    }

    return new NGramMovementModel({
      backendId: this.id,
      maxOrder,
      includeValue,
      vocabulary: [...vocabulary],
      grams,
    });
  }

  load(snapshot: MovementModelSnapshot): MovementPolicyModel {
    const grams: Array<Map<string, Map<MovementToken, number>>> = Array.from(
      { length: snapshot.maxOrder + 1 },
      () => new Map<string, Map<MovementToken, number>>(),
    );
    for (const gram of snapshot.grams) {
      if (gram.order < 0 || gram.order > snapshot.maxOrder) {
        continue;
      }
      const successors = new Map<MovementToken, number>();
      for (const successor of gram.successors) {
        successors.set(successor.token, successor.count);
      }
      grams[gram.order]!.set(gram.context, successors);
    }
    return new NGramMovementModel({
      backendId: snapshot.backendId,
      maxOrder: snapshot.maxOrder,
      includeValue: snapshot.includeValue,
      vocabulary: [...snapshot.vocabulary],
      grams,
    });
  }
}

/** Registry of pluggable backends. Register a real on-device model here. */
export class MovementPolicyBackendRegistry {
  private readonly backends = new Map<string, MovementPolicyBackend>();

  register(backend: MovementPolicyBackend): this {
    this.backends.set(backend.id, backend);
    return this;
  }

  get(id: string): MovementPolicyBackend | undefined {
    return this.backends.get(id);
  }

  require(id: string): MovementPolicyBackend {
    const backend = this.backends.get(id);
    if (!backend) {
      throw new Error(`Unknown movement-policy backend: ${id}`);
    }
    return backend;
  }

  list(): string[] {
    return [...this.backends.keys()].sort();
  }
}

/** A registry pre-populated with the bundled deterministic backend. */
export function createDefaultMovementPolicyRegistry(): MovementPolicyBackendRegistry {
  return new MovementPolicyBackendRegistry().register(new NGramMovementBackend());
}

export type MovementEvalReport = {
  backendId: string;
  trajectoriesEvaluated: number;
  predictions: number;
  correct: number;
  /** Top-1 next-token accuracy over all predictable positions. */
  accuracy: number;
  /** Fraction of positions where the model produced any prediction. */
  coverage: number;
  perTrajectory: Array<{ id: string; predictions: number; correct: number; accuracy: number }>;
};

/**
 * Generalization eval harness: teacher-forced top-1 next-token accuracy on a
 * held-out (but related) dataset. Measures how well a trained policy predicts
 * the *next* movement given the true prefix — the operational definition of
 * "repeat and generalize the recorded movements".
 */
export function evaluateMovementPolicy(
  model: MovementPolicyModel,
  heldOut: MovementDataset,
  options: { includeValue?: boolean } = {},
): MovementEvalReport {
  let predictions = 0;
  let correct = 0;
  let positions = 0;
  const perTrajectory: MovementEvalReport["perTrajectory"] = [];

  for (const trajectory of heldOut.trajectories) {
    const tokens = tokenizeMovementTrajectory(trajectory, options);
    let localPredictions = 0;
    let localCorrect = 0;
    for (let i = 0; i < tokens.length; i += 1) {
      positions += 1;
      const prediction = model.predict(tokens.slice(0, i));
      if (!prediction) {
        continue;
      }
      localPredictions += 1;
      if (prediction.token === tokens[i]) {
        localCorrect += 1;
      }
    }
    predictions += localPredictions;
    correct += localCorrect;
    perTrajectory.push({
      id: trajectory.id,
      predictions: localPredictions,
      correct: localCorrect,
      accuracy: localPredictions === 0 ? 0 : localCorrect / localPredictions,
    });
  }

  return {
    backendId: model.backendId,
    trajectoriesEvaluated: heldOut.trajectories.length,
    predictions,
    correct,
    accuracy: predictions === 0 ? 0 : correct / predictions,
    coverage: positions === 0 ? 0 : predictions / positions,
    perTrajectory,
  };
}

function increment(
  table: Map<string, Map<MovementToken, number>>,
  key: string,
  token: MovementToken,
): void {
  let successors = table.get(key);
  if (!successors) {
    successors = new Map<MovementToken, number>();
    table.set(key, successors);
  }
  successors.set(token, (successors.get(token) ?? 0) + 1);
}

function argmax(
  successors: Map<MovementToken, number>,
): { token: MovementToken; count: number; total: number } {
  let best: MovementToken | undefined;
  let bestCount = -1;
  let total = 0;
  for (const [token, count] of successors) {
    total += count;
    // Deterministic tie-break: higher count wins, then lexically smallest token.
    if (count > bestCount || (count === bestCount && best !== undefined && token < best)) {
      best = token;
      bestCount = count;
    }
  }
  return { token: best!, count: bestCount, total };
}

function sortedEntries(
  table: Map<string, Map<MovementToken, number>>,
): Array<[string, Map<MovementToken, number>]> {
  return [...table.entries()].sort((a, b) => (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0));
}
