import type { ReplayTimelineEvent } from "../capture/replay.js";
import type { ExportedReplayManifest } from "./export-manifest.js";

/**
 * Pluggable local-model backend for the movement-learning subsystem.
 *
 * The training pipeline (exporter → job manifest → runner) prepares artifacts
 * for on-device fine-tuning (mlx/axolotl) that only runs on the user's real
 * machine. This module provides the *in-process, backend-agnostic* seam that
 * lets bee-agent actually learn from a reviewed movement dataset and reproduce
 * or generalize the recorded movements — without any real OS or GPU, so it can
 * be validated in the cloud/CI with synthetic event streams.
 *
 * A real on-device small model can implement {@link LocalModelBackend} later;
 * {@link MarkovMovementBackend} is a deterministic reference/mock backend that
 * genuinely learns token transitions from the dataset (repeat) and rolls new
 * sequences forward from novel-but-related seeds (generalize).
 */

/** A compact, discrete representation of a single replay/movement event. */
export type MovementToken = string;

/** An ordered movement token sequence (one recorded trajectory/session). */
export type MovementSequence = MovementToken[];

/** Sentinel tokens used internally for sequence boundaries; never emitted. */
export const MOVEMENT_START_TOKEN = "start";
export const MOVEMENT_END_TOKEN = "end";

/** A replay-shaped event: either the internal or the exported manifest form. */
export type MovementEvent = ReplayTimelineEvent | ExportedReplayManifest["events"][number];

/** One training example: the tokenized movement stream of a single session. */
export type MovementExample = {
  sessionId: string;
  tokens: MovementSequence;
};

/** A dataset the local model trains on. */
export type MovementDataset = {
  examples: MovementExample[];
  /** Sorted, de-duplicated set of every token seen across all examples. */
  vocabulary: MovementToken[];
};

export type MovementTrainingConfig = {
  /**
   * n-gram context order (>= 1). The model conditions the next token on up to
   * `order` preceding tokens, backing off to shorter contexts when a longer one
   * was never observed. Default 2.
   */
  order?: number;
};

export type MovementGenerationOptions = {
  /** Real tokens to seed the roll-out with (echoed back at the front). */
  seed?: MovementSequence;
  /** Hard cap on the produced sequence length (incl. seed). */
  maxLength?: number;
};

/** A trained, queryable local movement model. */
export interface TrainedMovementModel {
  readonly backend: string;
  readonly order: number;
  readonly vocabulary: MovementToken[];
  /** Most likely next token given a context, or undefined if unknown. */
  predictNext(context: MovementSequence): MovementToken | undefined;
  /** Probability (0..1) the model assigns to `next` following `context`. */
  scoreNext(context: MovementSequence, next: MovementToken): number;
  /** Deterministically roll a full sequence forward from an optional seed. */
  generate(options?: MovementGenerationOptions): MovementSequence;
  /** Portable, JSON-safe snapshot; round-trips via {@link loadMovementModel}. */
  serialize(): SerializedMovementModel;
}

/** A backend capable of training a {@link TrainedMovementModel} from a dataset. */
export interface LocalModelBackend {
  readonly name: string;
  train(dataset: MovementDataset, config?: MovementTrainingConfig): Promise<TrainedMovementModel>;
}

export type SerializedMovementModel = {
  version: 1;
  backend: string;
  order: number;
  vocabulary: MovementToken[];
  maxLength: number;
  /** context-key → list of [nextToken, count]. Keys encode the context order. */
  transitions: Record<string, Array<[MovementToken, number]>>;
};

const CONTEXT_SEPARATOR = "";

function normalizeLabel(value: string): string {
  const normalized = value.trim().toLowerCase().replace(/\s+/g, "-");
  return normalized.length > 0 ? normalized : "unknown";
}

/** Turn a single replay/movement event into its discrete movement token. */
export function tokenizeMovementEvent(event: MovementEvent): MovementToken {
  switch (event.kind) {
    case "transcript":
      return `transcript:${event.role}`;
    case "observation":
      return `observation:${normalizeLabel(event.source)}`;
    case "action":
      return `action:${normalizeLabel(event.tool)}`;
  }
}

/** Tokenize a replay's events into a movement sequence (chronological order). */
export function tokenizeMovementEvents(events: MovementEvent[]): MovementSequence {
  return [...events]
    .sort((a, b) => a.ts - b.ts)
    .map((event) => tokenizeMovementEvent(event));
}

/** Build a training dataset from replay manifests (reviewed-export or internal). */
export function buildMovementDataset(
  replays: Array<{ sessionId: string; events: MovementEvent[] }>,
): MovementDataset {
  const vocabulary = new Set<MovementToken>();
  const examples: MovementExample[] = [];
  for (const replay of replays) {
    const tokens = tokenizeMovementEvents(replay.events);
    for (const token of tokens) {
      vocabulary.add(token);
    }
    if (tokens.length > 0) {
      examples.push({ sessionId: replay.sessionId, tokens });
    }
  }
  return {
    examples,
    vocabulary: [...vocabulary].sort(),
  };
}

function contextKey(order: number, context: MovementSequence): string {
  return `${order}${CONTEXT_SEPARATOR}${context.join(CONTEXT_SEPARATOR)}`;
}

/** Deterministic argmax: highest count, ties broken by lexicographic token. */
function argmax(distribution: Map<MovementToken, number>): MovementToken | undefined {
  let best: MovementToken | undefined;
  let bestCount = -1;
  for (const [token, count] of distribution) {
    if (count > bestCount || (count === bestCount && best !== undefined && token < best)) {
      best = token;
      bestCount = count;
    }
  }
  return best;
}

class MarkovMovementModel implements TrainedMovementModel {
  readonly backend: string;
  readonly order: number;
  readonly vocabulary: MovementToken[];
  private readonly counts: Map<string, Map<MovementToken, number>>;
  private readonly maxLength: number;

  constructor(params: {
    backend: string;
    order: number;
    vocabulary: MovementToken[];
    counts: Map<string, Map<MovementToken, number>>;
    maxLength: number;
  }) {
    this.backend = params.backend;
    this.order = params.order;
    this.vocabulary = params.vocabulary;
    this.counts = params.counts;
    this.maxLength = params.maxLength;
  }

  /** Highest-order observed distribution for a context (with back-off). */
  private resolveDistribution(context: MovementSequence): Map<MovementToken, number> | undefined {
    // Left-pad with START so short contexts (incl. the empty start-of-sequence
    // context) resolve against the learned sequence-start distribution, exactly
    // as training padded each example.
    const padded =
      context.length >= this.order
        ? context
        : [...Array<MovementToken>(this.order - context.length).fill(MOVEMENT_START_TOKEN), ...context];
    for (let k = this.order; k >= 0; k -= 1) {
      const slice = padded.slice(padded.length - k);
      const distribution = this.counts.get(contextKey(k, slice));
      if (distribution && distribution.size > 0) {
        return distribution;
      }
    }
    return undefined;
  }

  predictNext(context: MovementSequence): MovementToken | undefined {
    const distribution = this.resolveDistribution(context);
    if (!distribution) {
      return undefined;
    }
    const next = argmax(distribution);
    return next === MOVEMENT_END_TOKEN ? undefined : next;
  }

  scoreNext(context: MovementSequence, next: MovementToken): number {
    const distribution = this.resolveDistribution(context);
    if (!distribution) {
      return 0;
    }
    let total = 0;
    for (const count of distribution.values()) {
      total += count;
    }
    if (total === 0) {
      return 0;
    }
    return (distribution.get(next) ?? 0) / total;
  }

  generate(options: MovementGenerationOptions = {}): MovementSequence {
    const seed = options.seed ?? [];
    const maxLength = Math.max(seed.length, options.maxLength ?? this.maxLength);
    const output: MovementSequence = [...seed];
    while (output.length < maxLength) {
      const distribution = this.resolveDistribution(output);
      if (!distribution) {
        break;
      }
      const next = argmax(distribution);
      if (next === undefined || next === MOVEMENT_END_TOKEN) {
        break;
      }
      output.push(next);
    }
    return output;
  }

  serialize(): SerializedMovementModel {
    const transitions: Record<string, Array<[MovementToken, number]>> = {};
    for (const [key, distribution] of this.counts) {
      transitions[key] = [...distribution.entries()];
    }
    return {
      version: 1,
      backend: this.backend,
      order: this.order,
      vocabulary: [...this.vocabulary],
      maxLength: this.maxLength,
      transitions,
    };
  }
}

/**
 * Deterministic reference backend: an order-N Markov model with stupid-backoff.
 * No randomness, no I/O — identical dataset + config always yields an identical
 * model, so cloud/CI tests over synthetic streams are fully reproducible.
 */
export class MarkovMovementBackend implements LocalModelBackend {
  readonly name = "markov-movement";

  async train(dataset: MovementDataset, config: MovementTrainingConfig = {}): Promise<TrainedMovementModel> {
    const order = Math.max(1, Math.floor(config.order ?? 2));
    const counts = new Map<string, Map<MovementToken, number>>();
    let longestExample = 0;

    for (const example of dataset.examples) {
      longestExample = Math.max(longestExample, example.tokens.length);
      const padded: MovementSequence = [
        ...Array<MovementToken>(order).fill(MOVEMENT_START_TOKEN),
        ...example.tokens,
        MOVEMENT_END_TOKEN,
      ];
      for (let i = order; i < padded.length; i += 1) {
        const target = padded[i];
        for (let k = 0; k <= order; k += 1) {
          const slice = padded.slice(i - k, i);
          const key = contextKey(k, slice);
          let distribution = counts.get(key);
          if (!distribution) {
            distribution = new Map<MovementToken, number>();
            counts.set(key, distribution);
          }
          distribution.set(target, (distribution.get(target) ?? 0) + 1);
        }
      }
    }

    return new MarkovMovementModel({
      backend: this.name,
      order,
      vocabulary: [...dataset.vocabulary],
      counts,
      maxLength: longestExample * 2 + order,
    });
  }
}

/** Reconstruct a model from a {@link SerializedMovementModel} snapshot. */
export function loadMovementModel(serialized: SerializedMovementModel): TrainedMovementModel {
  const counts = new Map<string, Map<MovementToken, number>>();
  for (const [key, entries] of Object.entries(serialized.transitions)) {
    counts.set(key, new Map(entries));
  }
  return new MarkovMovementModel({
    backend: serialized.backend,
    order: serialized.order,
    vocabulary: [...serialized.vocabulary],
    counts,
    maxLength: serialized.maxLength,
  });
}

export type MovementAccuracyReport = {
  total: number;
  correct: number;
  accuracy: number;
};

/**
 * Generalization eval: for every position in each held-out sequence, does the
 * model's top-1 prediction from the true prefix match the actual next token?
 * A model that only memorized training data scores near-zero on held-out but
 * related sequences, so this measures generalization, not just recall.
 */
export function evaluateNextTokenAccuracy(
  model: TrainedMovementModel,
  sequences: MovementSequence[],
): MovementAccuracyReport {
  let total = 0;
  let correct = 0;
  for (const sequence of sequences) {
    for (let i = 0; i < sequence.length; i += 1) {
      total += 1;
      if (model.predictNext(sequence.slice(0, i)) === sequence[i]) {
        correct += 1;
      }
    }
  }
  return {
    total,
    correct,
    accuracy: total === 0 ? 0 : correct / total,
  };
}
