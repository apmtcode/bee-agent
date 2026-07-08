import type { ReplayManifest, ReplayTimelineEvent } from "../capture/replay.js";

/**
 * Local-movement learning: pluggable model backend.
 *
 * bee-agent captures movements/actions as {@link ReplayTimelineEvent}s and exports
 * them into replayable datasets. Objective 2(c)/(d) of the self-evolution charter
 * requires a *local* model that can be trained on that dataset to (c) repeat the
 * recorded movements and (d) generalize to new-but-related movements.
 *
 * The real on-device training runtimes (mlx / axolotl, see {@link
 * ../training/runner.js}) only run on the user's Apple-silicon machine and cannot
 * execute in the cloud. So training and inference are expressed against the
 * pluggable {@link MovementModelBackend} seam below, and this file ships a
 * deterministic in-process backend ({@link MarkovMovementBackend}) that trains and
 * infers with no native dependencies — making the whole capture -> dataset ->
 * train -> repeat -> generalize loop testable in CI. A real backend (e.g. one that
 * wraps an mlx fine-tune) implements the same interface and is swapped in via the
 * {@link MovementBackendRegistry}.
 */

/** A single movement/action token derived from a replay timeline event. */
export type MovementToken = string;

/** An ordered sequence of movement tokens (one recorded trajectory/session). */
export type MovementSequence = MovementToken[];

/** A training dataset: a collection of independent movement sequences. */
export type MovementDataset = {
  sequences: MovementSequence[];
};

export type MovementTrainingOptions = {
  /** Maximum context length (n-gram order) the model conditions on. Default 2. */
  order?: number;
};

export type MovementPrediction = {
  /** The predicted next movement token. */
  token: MovementToken;
  /** Conditional probability of the token given the context actually used. */
  probability: number;
  /** The context suffix the prediction was conditioned on (post-backoff). */
  contextUsed: MovementToken[];
  /** True when the model had to shorten the requested context to find a match. */
  backedOff: boolean;
};

export type ReplayFidelityReport = {
  /** Number of next-token predictions scored. */
  steps: number;
  /** How many predictions matched the ground-truth next token. */
  correct: number;
  /** correct / steps (0 when there were no steps). */
  accuracy: number;
  /** Fraction of predictions that required backoff to a shorter context. */
  backoffRate: number;
};

export type SerializedMovementModel = {
  version: 1;
  backendId: string;
  order: number;
  vocabulary: MovementToken[];
  /** contextKey -> (nextToken -> count). The empty-string key is the unigram model. */
  transitions: Record<string, Record<MovementToken, number>>;
};

/** A trained model that can predict, generate, and self-evaluate movements. */
export interface MovementModel {
  readonly backendId: string;
  readonly order: number;
  readonly vocabulary: MovementToken[];
  /** Predict the single most likely next token given a context (undefined if untrained). */
  predictNext(context: MovementToken[]): MovementPrediction | undefined;
  /**
   * Autoregressively generate up to `steps` new tokens continuing `seed`.
   * Returns only the generated continuation (not the seed). Stops early if the
   * model has no prediction for the current context.
   */
  generate(seed: MovementSequence, steps: number): MovementSequence;
  /**
   * Teacher-forced replay fidelity: for each position in `sequence`, predict the
   * next token from the true prefix and score it against ground truth. This is the
   * core metric for both "repeat recorded movements" and "generalize" evals.
   */
  evaluate(sequence: MovementSequence): ReplayFidelityReport;
  toJSON(): SerializedMovementModel;
}

/** The pluggable training backend seam. */
export interface MovementModelBackend {
  readonly id: string;
  train(dataset: MovementDataset, options?: MovementTrainingOptions): MovementModel;
}

const CONTEXT_SEPARATOR = "";
const DEFAULT_ORDER = 2;

/** Derive a stable movement token from a replay timeline event. */
export function tokenizeReplayEvent(event: ReplayTimelineEvent): MovementToken {
  switch (event.kind) {
    case "action":
      return `action:${event.tool}`;
    case "observation":
      return `observation:${event.source}`;
    case "transcript":
      return `transcript:${event.role}`;
  }
}

/** Build a training dataset from replay manifests (one sequence per replay). */
export function buildMovementDataset(
  replays: Array<Pick<ReplayManifest, "events">>,
): MovementDataset {
  return {
    sequences: replays
      .map((replay) => replay.events.map(tokenizeReplayEvent))
      .filter((sequence) => sequence.length > 0),
  };
}

function contextKey(context: MovementToken[]): string {
  return context.join(CONTEXT_SEPARATOR);
}

/** Deterministic argmax over a count map, tie-broken lexicographically. */
function argmax(counts: Map<MovementToken, number>): { token: MovementToken; count: number; total: number } | undefined {
  let best: MovementToken | undefined;
  let bestCount = -1;
  let total = 0;
  for (const [token, count] of counts) {
    total += count;
    if (count > bestCount || (count === bestCount && best !== undefined && token < best)) {
      best = token;
      bestCount = count;
    }
  }
  if (best === undefined) {
    return undefined;
  }
  return { token: best, count: bestCount, total };
}

class MarkovMovementModel implements MovementModel {
  readonly vocabulary: MovementToken[];

  constructor(
    readonly backendId: string,
    readonly order: number,
    private readonly transitions: Map<string, Map<MovementToken, number>>,
    vocabulary: Set<MovementToken>,
  ) {
    this.vocabulary = [...vocabulary].sort();
  }

  predictNext(context: MovementToken[]): MovementPrediction | undefined {
    const requestedLength = Math.min(this.order, context.length);
    for (let length = requestedLength; length >= 0; length -= 1) {
      const suffix = length === 0 ? [] : context.slice(-length);
      const counts = this.transitions.get(contextKey(suffix));
      if (!counts) {
        continue;
      }
      const best = argmax(counts);
      if (!best) {
        continue;
      }
      return {
        token: best.token,
        probability: best.count / best.total,
        contextUsed: suffix,
        backedOff: length < requestedLength,
      };
    }
    return undefined;
  }

  generate(seed: MovementSequence, steps: number): MovementSequence {
    const generated: MovementSequence = [];
    const context = [...seed];
    for (let step = 0; step < steps; step += 1) {
      const prediction = this.predictNext(context);
      if (!prediction) {
        break;
      }
      generated.push(prediction.token);
      context.push(prediction.token);
    }
    return generated;
  }

  evaluate(sequence: MovementSequence): ReplayFidelityReport {
    let steps = 0;
    let correct = 0;
    let backoffs = 0;
    for (let index = 1; index < sequence.length; index += 1) {
      const prediction = this.predictNext(sequence.slice(0, index));
      steps += 1;
      if (prediction?.token === sequence[index]) {
        correct += 1;
      }
      if (prediction?.backedOff) {
        backoffs += 1;
      }
    }
    return {
      steps,
      correct,
      accuracy: steps === 0 ? 0 : correct / steps,
      backoffRate: steps === 0 ? 0 : backoffs / steps,
    };
  }

  toJSON(): SerializedMovementModel {
    const transitions: Record<string, Record<MovementToken, number>> = {};
    for (const [key, counts] of this.transitions) {
      const serializedCounts: Record<MovementToken, number> = {};
      for (const [token, count] of counts) {
        serializedCounts[token] = count;
      }
      transitions[key] = serializedCounts;
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

/**
 * Deterministic, dependency-free movement backend: a back-off n-gram (Markov)
 * model. It reproduces recorded sequences exactly when contexts are unambiguous
 * (objective 2c) and generalizes to unseen contexts by backing off to shorter,
 * previously-observed suffixes (objective 2d). No randomness — identical dataset
 * in, identical model out — so it is safe for CI and reproducible replays.
 */
export class MarkovMovementBackend implements MovementModelBackend {
  readonly id = "markov-mock";

  train(dataset: MovementDataset, options: MovementTrainingOptions = {}): MovementModel {
    const order = Math.max(0, Math.floor(options.order ?? DEFAULT_ORDER));
    const transitions = new Map<string, Map<MovementToken, number>>();
    const vocabulary = new Set<MovementToken>();

    const bump = (context: MovementToken[], next: MovementToken): void => {
      const key = contextKey(context);
      let counts = transitions.get(key);
      if (!counts) {
        counts = new Map<MovementToken, number>();
        transitions.set(key, counts);
      }
      counts.set(next, (counts.get(next) ?? 0) + 1);
    };

    for (const sequence of dataset.sequences) {
      for (const token of sequence) {
        vocabulary.add(token);
      }
      for (let index = 1; index < sequence.length; index += 1) {
        const next = sequence[index]!;
        // Record the transition at every context length 0..order for backoff.
        for (let length = 0; length <= order; length += 1) {
          const start = Math.max(0, index - length);
          bump(sequence.slice(start, index), next);
        }
      }
    }

    return new MarkovMovementModel(this.id, order, transitions, vocabulary);
  }
}

/** Reconstruct a model from its serialized form (persistence / seam boundary). */
export function restoreMovementModel(serialized: SerializedMovementModel): MovementModel {
  const transitions = new Map<string, Map<MovementToken, number>>();
  for (const [key, counts] of Object.entries(serialized.transitions)) {
    const map = new Map<MovementToken, number>();
    for (const [token, count] of Object.entries(counts)) {
      map.set(token, count);
    }
    transitions.set(key, map);
  }
  return new MarkovMovementModel(
    serialized.backendId,
    serialized.order,
    transitions,
    new Set(serialized.vocabulary),
  );
}

/**
 * A registry of movement-model backends. Register real on-device backends here at
 * startup; the deterministic mock is registered by default so cloud runs work.
 */
export class MovementBackendRegistry {
  private readonly backends = new Map<string, MovementModelBackend>();

  register(backend: MovementModelBackend): this {
    this.backends.set(backend.id, backend);
    return this;
  }

  has(id: string): boolean {
    return this.backends.has(id);
  }

  resolve(id: string): MovementModelBackend {
    const backend = this.backends.get(id);
    if (!backend) {
      throw new Error(
        `Unknown movement-model backend "${id}". Registered: ${this.list().join(", ") || "(none)"}`,
      );
    }
    return backend;
  }

  list(): string[] {
    return [...this.backends.keys()].sort();
  }
}

/** Registry pre-seeded with the deterministic mock backend. */
export function createDefaultMovementBackendRegistry(): MovementBackendRegistry {
  return new MovementBackendRegistry().register(new MarkovMovementBackend());
}
