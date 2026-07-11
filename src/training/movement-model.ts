/**
 * Local-movement learning model backend.
 *
 * This module closes the (c) "post-train a local model to repeat recorded
 * movements" and (d) "generalize to related movements" pieces of the
 * movement-learning subsystem. The heavy on-device runtimes (mlx / axolotl,
 * see `runner.ts`) cannot run in the cloud, so the actual trainable model is
 * kept behind a small `LocalMovementModelBackend` seam:
 *
 *   dataset (from reviewed replays) --> backend.train() --> TrainedMovementModel
 *
 * The default `MarkovMovementBackend` is a fully deterministic n-gram model
 * with stupid-backoff. It is a legitimate small on-device model (no external
 * deps, CPU-only, instant) that:
 *   - repeats recorded movements with high fidelity (exact replay of learned
 *     sequences), and
 *   - generalizes to unseen-but-related prefixes by backing off to shorter
 *     contexts.
 *
 * A real neural/local-LLM backend can implement the same interface and slot in
 * without touching callers — that is the pluggability seam objective #2 asks
 * for. Determinism keeps cloud/CI tests reproducible.
 */

/** A single quantized movement/action step in a sequence. */
export type MovementToken = string;

/** Sentinel token appended to every training sequence to mark its end. */
export const MOVEMENT_END_TOKEN = "\u0000END";

/** An ordered sequence of movement tokens for one trajectory/session. */
export type MovementSequence = {
  id: string;
  tokens: MovementToken[];
};

/** A replayable movement dataset derived from reviewed action events. */
export type MovementDataset = {
  version: 1;
  sequences: MovementSequence[];
  /** Sorted, de-duplicated token vocabulary (excludes the END sentinel). */
  vocabulary: MovementToken[];
};

/** Minimal shape shared by ReplayManifest and ExportedReplayManifest. */
export type ReplayEventLike =
  | { kind: "transcript"; ts: number }
  | { kind: "observation"; ts: number }
  | { kind: "action"; ts: number; trajectoryId: string; tool: string; summary: string };

export type ReplaySource = {
  sessionId?: string;
  trajectoryIds?: string[];
  events: Array<{ kind: string; ts: number; [key: string]: unknown }>;
};

export type MovementDatasetOptions = {
  /**
   * Maps an action event to a token. Default preserves tool + summary so
   * learned movements replay exactly; override to bucket/quantize (e.g. round
   * mouse coordinates) for coarser generalization.
   */
  tokenize?: (action: { tool: string; summary: string }) => MovementToken;
  /** Group action events by this key. Default: trajectoryId, then sessionId. */
  groupBy?: "trajectory" | "session";
};

/** Non-printing field separator between tool and summary in the default token. */
export const TOKEN_FIELD_SEPARATOR = "\u0001";

export function defaultTokenize(action: { tool: string; summary: string }): MovementToken {
  return `${action.tool}${TOKEN_FIELD_SEPARATOR}${action.summary}`;
}

/**
 * Build a movement dataset from replay manifests. Only `action` events are
 * used; they are grouped into ordered sequences (by trajectory or session)
 * and sorted by timestamp. Transcript/observation events are ignored here —
 * they belong to the context model, not the movement-repetition model.
 */
export function buildMovementDataset(
  replays: ReplaySource[],
  options: MovementDatasetOptions = {},
): MovementDataset {
  const tokenize = options.tokenize ?? defaultTokenize;
  const groupBy = options.groupBy ?? "trajectory";
  const groups = new Map<string, Array<{ ts: number; token: MovementToken; order: number }>>();

  let ingestOrder = 0;
  for (const replay of replays) {
    const fallbackId = replay.sessionId ?? replay.trajectoryIds?.[0] ?? "unknown";
    for (const event of replay.events) {
      if (event.kind !== "action") {
        continue;
      }
      const tool = String(event.tool ?? "");
      const summary = String(event.summary ?? "");
      const groupId =
        groupBy === "session" ? fallbackId : String(event.trajectoryId ?? fallbackId);
      const bucket = groups.get(groupId) ?? [];
      bucket.push({ ts: Number(event.ts ?? 0), token: tokenize({ tool, summary }), order: ingestOrder++ });
      groups.set(groupId, bucket);
    }
  }

  const sequences: MovementSequence[] = [];
  const vocabulary = new Set<MovementToken>();
  for (const [id, entries] of [...groups.entries()].sort((a, b) => (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0))) {
    // Stable ordering: by timestamp, then by ingest order for ties.
    entries.sort((a, b) => (a.ts !== b.ts ? a.ts - b.ts : a.order - b.order));
    const tokens = entries.map((entry) => entry.token);
    for (const token of tokens) {
      vocabulary.add(token);
    }
    if (tokens.length > 0) {
      sequences.push({ id, tokens });
    }
  }

  return {
    version: 1,
    sequences,
    vocabulary: [...vocabulary].sort(),
  };
}

export type MovementTrainingConfig = {
  /** Maximum n-gram context length. Default 2 (trigram-ish with backoff). */
  order?: number;
  /** Drop context→token transitions observed fewer than this many times. */
  minCount?: number;
};

export type MovementPrediction = {
  token: MovementToken;
  /** Conditional probability of the token given the context used. */
  probability: number;
  /** Length of the context that actually produced the prediction (backoff level). */
  order: number;
};

export type SerializedMovementModel = {
  version: 1;
  backend: string;
  order: number;
  vocabulary: MovementToken[];
  /** order -> contextKey -> token -> count */
  transitions: Record<string, Record<string, Record<string, number>>>;
};

export interface TrainedMovementModel {
  readonly backend: string;
  readonly order: number;
  readonly vocabulary: MovementToken[];
  /** Predict the most likely next token given a context (uses backoff). */
  predictNext(context: MovementToken[]): MovementPrediction | undefined;
  /** Greedily roll out a continuation from a seed until END or maxSteps. */
  generate(seed: MovementToken[], maxSteps: number): MovementToken[];
  serialize(): SerializedMovementModel;
}

export interface LocalMovementModelBackend {
  readonly name: string;
  train(dataset: MovementDataset, config?: MovementTrainingConfig): TrainedMovementModel;
}

const CONTEXT_SEPARATOR = "\u0002";

/**
 * Deterministic n-gram backend with stupid-backoff. Fully reproducible: ties
 * are broken by count (desc) then token (asc), so the same dataset always
 * yields the same model and the same rollouts.
 */
export class MarkovMovementBackend implements LocalMovementModelBackend {
  readonly name = "markov-ngram";

  train(dataset: MovementDataset, config: MovementTrainingConfig = {}): TrainedMovementModel {
    const order = Math.max(1, Math.floor(config.order ?? 2));
    const minCount = Math.max(1, Math.floor(config.minCount ?? 1));

    // transitions[k] maps a length-k context key -> token -> count.
    // k ranges 0..order; k=0 is the unigram (empty-context) distribution.
    const transitions: Map<number, Map<string, Map<MovementToken, number>>> = new Map();
    for (let k = 0; k <= order; k += 1) {
      transitions.set(k, new Map());
    }

    for (const sequence of dataset.sequences) {
      const tokens = [...sequence.tokens, MOVEMENT_END_TOKEN];
      for (let i = 0; i < tokens.length; i += 1) {
        const next = tokens[i]!;
        for (let k = 0; k <= order; k += 1) {
          if (i - k < 0) {
            continue;
          }
          const contextKey = tokens.slice(i - k, i).join(CONTEXT_SEPARATOR);
          const level = transitions.get(k)!;
          const counts = level.get(contextKey) ?? new Map<MovementToken, number>();
          counts.set(next, (counts.get(next) ?? 0) + 1);
          level.set(contextKey, counts);
        }
      }
    }

    if (minCount > 1) {
      for (const level of transitions.values()) {
        for (const [contextKey, counts] of level) {
          for (const [token, count] of counts) {
            if (count < minCount) {
              counts.delete(token);
            }
          }
          if (counts.size === 0) {
            level.delete(contextKey);
          }
        }
      }
    }

    return new MarkovMovementModel(this.name, order, [...dataset.vocabulary], transitions);
  }

  static fromSerialized(serialized: SerializedMovementModel): TrainedMovementModel {
    const transitions = new Map<number, Map<string, Map<MovementToken, number>>>();
    for (const [levelKey, contexts] of Object.entries(serialized.transitions)) {
      const level = new Map<string, Map<MovementToken, number>>();
      for (const [contextKey, counts] of Object.entries(contexts)) {
        level.set(contextKey, new Map(Object.entries(counts)));
      }
      transitions.set(Number(levelKey), level);
    }
    return new MarkovMovementModel(
      serialized.backend,
      serialized.order,
      [...serialized.vocabulary],
      transitions,
    );
  }
}

class MarkovMovementModel implements TrainedMovementModel {
  constructor(
    readonly backend: string,
    readonly order: number,
    readonly vocabulary: MovementToken[],
    private readonly transitions: Map<number, Map<string, Map<MovementToken, number>>>,
  ) {}

  predictNext(context: MovementToken[]): MovementPrediction | undefined {
    for (let k = Math.min(this.order, context.length); k >= 0; k -= 1) {
      const level = this.transitions.get(k);
      if (!level) {
        continue;
      }
      const contextKey = context.slice(context.length - k).join(CONTEXT_SEPARATOR);
      const counts = level.get(contextKey);
      if (!counts || counts.size === 0) {
        continue;
      }
      let total = 0;
      let best: MovementToken | undefined;
      let bestCount = -1;
      for (const [token, count] of counts) {
        total += count;
        if (count > bestCount || (count === bestCount && (best === undefined || token < best))) {
          best = token;
          bestCount = count;
        }
      }
      if (best !== undefined) {
        return { token: best, probability: bestCount / total, order: k };
      }
    }
    return undefined;
  }

  generate(seed: MovementToken[], maxSteps: number): MovementToken[] {
    const context = [...seed];
    const produced: MovementToken[] = [];
    for (let step = 0; step < maxSteps; step += 1) {
      const prediction = this.predictNext(context);
      if (!prediction || prediction.token === MOVEMENT_END_TOKEN) {
        break;
      }
      produced.push(prediction.token);
      context.push(prediction.token);
    }
    return produced;
  }

  serialize(): SerializedMovementModel {
    const transitions: Record<string, Record<string, Record<string, number>>> = {};
    for (const [level, contexts] of this.transitions) {
      const levelRecord: Record<string, Record<string, number>> = {};
      for (const [contextKey, counts] of contexts) {
        levelRecord[contextKey] = Object.fromEntries(counts);
      }
      transitions[String(level)] = levelRecord;
    }
    return {
      version: 1,
      backend: this.backend,
      order: this.order,
      vocabulary: [...this.vocabulary],
      transitions,
    };
  }
}

export type NextTokenAccuracy = {
  total: number;
  correct: number;
  accuracy: number;
};

/**
 * Teacher-forcing next-token accuracy: for every position the true prefix is
 * fed and the model's argmax prediction is compared to the actual next token.
 * Measures how well the model has learned the recorded movements.
 */
export function evaluateNextTokenAccuracy(
  model: TrainedMovementModel,
  sequences: MovementSequence[],
): NextTokenAccuracy {
  let total = 0;
  let correct = 0;
  for (const sequence of sequences) {
    for (let i = 0; i < sequence.tokens.length; i += 1) {
      const context = sequence.tokens.slice(0, i);
      const prediction = model.predictNext(context);
      total += 1;
      if (prediction && prediction.token === sequence.tokens[i]) {
        correct += 1;
      }
    }
  }
  return { total, correct, accuracy: total === 0 ? 0 : correct / total };
}

export type ReplayFidelityReport = {
  sequences: number;
  exactMatches: number;
  /** Fraction of sequences reproduced exactly from their first token. */
  fidelity: number;
  /** Mean per-sequence token overlap (positional). */
  meanTokenOverlap: number;
};

/**
 * Free-running replay fidelity: seed the model with each sequence's first
 * token, let it roll out, and compare to the ground truth. High fidelity means
 * the model reliably *repeats* recorded movements without teacher forcing.
 */
export function evaluateReplayFidelity(
  model: TrainedMovementModel,
  sequences: MovementSequence[],
): ReplayFidelityReport {
  let exactMatches = 0;
  let overlapSum = 0;
  let counted = 0;
  for (const sequence of sequences) {
    if (sequence.tokens.length === 0) {
      continue;
    }
    counted += 1;
    const seed = sequence.tokens.slice(0, 1);
    const expected = sequence.tokens.slice(1);
    const generated = model.generate(seed, expected.length + 4);
    if (generated.length === expected.length && generated.every((token, index) => token === expected[index])) {
      exactMatches += 1;
    }
    if (expected.length === 0) {
      overlapSum += generated.length === 0 ? 1 : 0;
      continue;
    }
    let matched = 0;
    for (let i = 0; i < expected.length; i += 1) {
      if (generated[i] === expected[i]) {
        matched += 1;
      }
    }
    overlapSum += matched / expected.length;
  }
  return {
    sequences: counted,
    exactMatches,
    fidelity: counted === 0 ? 0 : exactMatches / counted,
    meanTokenOverlap: counted === 0 ? 0 : overlapSum / counted,
  };
}
