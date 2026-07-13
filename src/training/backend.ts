import type { ReplayManifest, ReplayTimelineEvent } from "../capture/replay.js";

/**
 * A single quantized "movement" unit the local model reasons over. Tokens are
 * intentionally coarse (the tool / source / role class rather than the free-text
 * summary) so that a novel-but-related event — same tool, different arguments —
 * maps to the same token. That coarseness is what lets a model trained to
 * *repeat* recorded movements also *generalize* to new ones (standing objective
 * #2 (c)/(d)).
 */
export type MovementToken = string;

/**
 * A trained, fully serializable local movement model. This is the artifact a
 * backend produces from a dataset and consumes at inference time. It carries no
 * runtime handles, so it round-trips cleanly through JSON and can be persisted
 * next to the reviewed export / training job artifacts.
 */
export type TrainedMovementModel = {
  version: 1;
  backend: string;
  maxOrder: number;
  vocabulary: MovementToken[];
  sequenceCount: number;
  tokenCount: number;
  /**
   * Backoff n-gram counts keyed by context. The key is the context tokens
   * joined by {@link CONTEXT_SEPARATOR}; the empty key holds the order-0
   * (unconditional) distribution. Order is implied by the number of context
   * tokens, so a single flat record covers every backoff level.
   */
  transitions: Record<string, Record<MovementToken, number>>;
};

export type MovementPredictionAlternative = {
  token: MovementToken;
  probability: number;
};

export type MovementPrediction = {
  /** Best next-movement token, or null when the model has learned nothing. */
  token: MovementToken | null;
  /** Conditional probability of {@link token} at the backoff order used. */
  confidence: number;
  /**
   * The context order the prediction was drawn from. A value below the model's
   * `maxOrder` means the exact context was unseen and the model *generalized*
   * by backing off — the signal we measure in {@link evaluateMovementModel}.
   */
  order: number;
  /** Ranked runner-up tokens at the same order (deterministic ordering). */
  alternatives: MovementPredictionAlternative[];
};

export type TrainMovementModelRequest = {
  sequences: MovementToken[][];
  maxOrder?: number;
};

/**
 * Pluggable local-model backend. A backend turns a dataset of movement
 * sequences into a {@link TrainedMovementModel} and answers next-movement
 * queries from it. Training may be asynchronous (a real on-device backend will
 * shell out to a trainer); inference is a cheap synchronous lookup.
 *
 * The default {@link NgramMovementBackend} is deterministic and dependency-free
 * so it runs in CI / the cloud. A real on-device small-model backend implements
 * the same surface — see the runner's launch plan for the training seam.
 */
export interface LocalModelBackend {
  readonly name: string;
  train(request: TrainMovementModelRequest): Promise<TrainedMovementModel>;
  predict(model: TrainedMovementModel, context: MovementToken[]): MovementPrediction;
}

export const DEFAULT_MOVEMENT_MODEL_ORDER = 2;

const CONTEXT_SEPARATOR = "";

function contextKey(context: MovementToken[]): string {
  return context.join(CONTEXT_SEPARATOR);
}

/**
 * Reduce a replay timeline event to its movement token. Coarse by design (see
 * {@link MovementToken}). Kept total over the event union so new event kinds are
 * a compile error here rather than a silent drop.
 */
export function tokenizeReplayEvent(event: ReplayTimelineEvent): MovementToken {
  switch (event.kind) {
    case "transcript":
      return `transcript:${event.role}`;
    case "observation":
      return `observation:${event.source}`;
    case "action":
      return `action:${event.tool}`;
  }
}

/** Build one token sequence per replay manifest, preserving timeline order. */
export function buildMovementSequences(replays: Pick<ReplayManifest, "events">[]): MovementToken[][] {
  return replays.map((replay) => replay.events.map(tokenizeReplayEvent));
}

function argmaxDistribution(
  distribution: Record<MovementToken, number> | undefined,
): { token: MovementToken; confidence: number; alternatives: MovementPredictionAlternative[] } | undefined {
  if (!distribution) {
    return undefined;
  }
  const entries = Object.entries(distribution);
  if (entries.length === 0) {
    return undefined;
  }
  const total = entries.reduce((sum, [, count]) => sum + count, 0);
  if (total <= 0) {
    return undefined;
  }
  // Deterministic: highest count first, ties broken by token ascending.
  const ranked = entries
    .map(([token, count]) => ({ token, probability: count / total }))
    .sort((a, b) => (b.probability !== a.probability ? b.probability - a.probability : a.token < b.token ? -1 : 1));
  const [best, ...rest] = ranked;
  return { token: best.token, confidence: best.probability, alternatives: rest };
}

/**
 * Deterministic backoff n-gram movement backend. Training counts, for every
 * context length 0..maxOrder, how often each token follows that context.
 * Inference tries the longest available context and backs off toward the
 * unconditional distribution — so it reproduces recorded sequences exactly and
 * still answers for contexts it has never seen.
 */
export class NgramMovementBackend implements LocalModelBackend {
  readonly name: string;

  constructor(name = "mock") {
    this.name = name;
  }

  async train(request: TrainMovementModelRequest): Promise<TrainedMovementModel> {
    const maxOrder = Math.max(0, request.maxOrder ?? DEFAULT_MOVEMENT_MODEL_ORDER);
    const transitions: Record<string, Record<MovementToken, number>> = {};
    const vocabulary = new Set<MovementToken>();
    let tokenCount = 0;

    for (const sequence of request.sequences) {
      for (let index = 0; index < sequence.length; index += 1) {
        const next = sequence[index];
        vocabulary.add(next);
        tokenCount += 1;
        const maxContext = Math.min(maxOrder, index);
        for (let order = 0; order <= maxContext; order += 1) {
          const context = sequence.slice(index - order, index);
          const key = contextKey(context);
          const bucket = (transitions[key] ??= {});
          bucket[next] = (bucket[next] ?? 0) + 1;
        }
      }
    }

    return {
      version: 1,
      backend: this.name,
      maxOrder,
      vocabulary: [...vocabulary].sort(),
      sequenceCount: request.sequences.length,
      tokenCount,
      transitions,
    };
  }

  predict(model: TrainedMovementModel, context: MovementToken[]): MovementPrediction {
    const startOrder = Math.min(model.maxOrder, context.length);
    for (let order = startOrder; order >= 0; order -= 1) {
      const suffix = order === 0 ? [] : context.slice(context.length - order);
      const best = argmaxDistribution(model.transitions[contextKey(suffix)]);
      if (best) {
        return { token: best.token, confidence: best.confidence, order, alternatives: best.alternatives };
      }
    }
    return { token: null, confidence: 0, order: -1, alternatives: [] };
  }
}

/**
 * Greedily generate a movement sequence from a seed context — the inference
 * counterpart to the capture recorder. Stops early if the model can predict no
 * further movement. Pure over the backend/model, so it is deterministic.
 */
export function rolloutMovements(
  backend: LocalModelBackend,
  model: TrainedMovementModel,
  seed: MovementToken[],
  steps: number,
): MovementToken[] {
  const generated: MovementToken[] = [];
  const context = [...seed];
  for (let step = 0; step < steps; step += 1) {
    const prediction = backend.predict(model, context);
    if (prediction.token === null) {
      break;
    }
    generated.push(prediction.token);
    context.push(prediction.token);
  }
  return generated;
}

export type MovementModelEvaluation = {
  predictions: number;
  correct: number;
  accuracy: number;
  /** Fraction of correct predictions that required backoff (generalization). */
  generalizedFraction: number;
  perSequence: Array<{ predictions: number; correct: number; accuracy: number }>;
};

/**
 * Next-token accuracy over held-out sequences — the seed of the generalization
 * eval harness. For each position it feeds the true prefix and checks whether
 * the model's greedy prediction matches the recorded next movement, tracking how
 * often a correct answer came from a backed-off (unseen) context.
 */
export function evaluateMovementModel(
  backend: LocalModelBackend,
  model: TrainedMovementModel,
  sequences: MovementToken[][],
): MovementModelEvaluation {
  let predictions = 0;
  let correct = 0;
  let generalizedCorrect = 0;
  const perSequence: MovementModelEvaluation["perSequence"] = [];

  for (const sequence of sequences) {
    let seqPredictions = 0;
    let seqCorrect = 0;
    for (let index = 1; index < sequence.length; index += 1) {
      const context = sequence.slice(0, index);
      const prediction = backend.predict(model, context);
      predictions += 1;
      seqPredictions += 1;
      if (prediction.token === sequence[index]) {
        correct += 1;
        seqCorrect += 1;
        if (prediction.order < Math.min(model.maxOrder, context.length)) {
          generalizedCorrect += 1;
        }
      }
    }
    perSequence.push({
      predictions: seqPredictions,
      correct: seqCorrect,
      accuracy: seqPredictions === 0 ? 0 : seqCorrect / seqPredictions,
    });
  }

  return {
    predictions,
    correct,
    accuracy: predictions === 0 ? 0 : correct / predictions,
    generalizedFraction: correct === 0 ? 0 : generalizedCorrect / correct,
    perSequence,
  };
}

/**
 * Registry of local-model backends, keyed by name. Makes the training backend
 * pluggable: the runner / execution service resolves a backend by
 * `job.runtime`, tests register a deterministic mock, and a real on-device
 * backend registers under its own name without touching call sites.
 */
export class MovementBackendRegistry {
  private readonly backends = new Map<string, LocalModelBackend>();

  register(backend: LocalModelBackend): this {
    this.backends.set(backend.name, backend);
    return this;
  }

  has(name: string): boolean {
    return this.backends.has(name);
  }

  get(name: string): LocalModelBackend {
    const backend = this.backends.get(name);
    if (!backend) {
      throw new Error(
        `Unknown movement backend "${name}". Registered: ${[...this.backends.keys()].sort().join(", ") || "(none)"}`,
      );
    }
    return backend;
  }

  list(): string[] {
    return [...this.backends.keys()].sort();
  }
}

/** Registry preloaded with the deterministic mock backend under "mock". */
export function createDefaultMovementBackendRegistry(): MovementBackendRegistry {
  return new MovementBackendRegistry().register(new NgramMovementBackend("mock"));
}
