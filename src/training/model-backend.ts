import type { ReplayTimelineEvent } from "../capture/replay.js";

/**
 * In-process, pluggable local-model backend for the movement-learning
 * subsystem (standing objective #2, parts (c) train-to-repeat and
 * (d) generalize-to-related).
 *
 * The production path (`LocalAppleSiliconTrainingRunner`) emits mlx/axolotl
 * launch scripts for real on-device training. That cannot run in the cloud/CI,
 * so this module provides a fully deterministic backend that trains and infers
 * *in-process* on tokenized movement sequences. It doubles as:
 *   - the default mock backend so cloud/CI tests exercise the train→infer loop,
 *   - a documented seam (`LocalModelBackend`) a real small on-device model can
 *     be swapped behind without touching call sites.
 *
 * The default `NGramMovementBackend` is a variable-order Markov model with
 * Katz-style suffix backoff:
 *   - it *repeats* recorded movements exactly (seed a recorded prefix, generate
 *     until the end-of-sequence marker), and
 *   - it *generalizes* to new-but-related movement contexts by backing off to
 *     the longest previously-seen suffix.
 */

/** Marker appended to every training sequence so generation can stop naturally. */
export const MOVEMENT_EOS = "<eos>";

/**
 * Marker prepended (internally) to every context so the model learns which
 * token *starts* a workflow. Never emitted as a prediction; lets `generate([])`
 * reproduce a whole recording from nothing.
 */
export const MOVEMENT_BOS = "<bos>";

/** A single tokenized movement sequence (ordered tokens, no EOS). */
export type MovementSequence = string[];

/** Training corpus of tokenized movement sequences. */
export type MovementDataset = {
  sequences: MovementSequence[];
};

export type TrainingBackendOptions = {
  /** Maximum Markov context length. Higher = more literal replay, less generalization. */
  order?: number;
};

export type MovementPrediction = {
  /** Most likely next token (may be {@link MOVEMENT_EOS}). */
  token: string;
  /** Conditional probability of `token` given the matched context. */
  probability: number;
  /** Length of the context suffix that was actually matched (0 = unigram backoff). */
  backoffOrder: number;
  /** Whether an exact full-order context match was used (no backoff). */
  exact: boolean;
  /** Other candidate tokens for the matched context, most likely first. */
  alternatives: Array<{ token: string; probability: number }>;
};

export type MovementGeneration = {
  /** Seed prefix followed by the generated continuation (EOS stripped). */
  sequence: MovementSequence;
  /** Only the newly generated tokens (EOS stripped). */
  continuation: MovementSequence;
  /** True if generation stopped because the model emitted {@link MOVEMENT_EOS}. */
  stopped: boolean;
};

/** Serialized model, safe to persist as JSON and restore later. */
export type MovementModelSnapshot = {
  version: 1;
  backend: string;
  order: number;
  vocabulary: string[];
  /** Flattened counts: for each order, context tokens → next token → count. */
  grams: Array<{ order: number; context: string[]; next: string; count: number }>;
};

/**
 * Pluggable backend seam. A real on-device backend (e.g. an mlx-backed small
 * model) implements the same interface so call sites stay identical.
 */
export interface LocalModelBackend {
  readonly id: string;
  train(dataset: MovementDataset, options?: TrainingBackendOptions): Promise<TrainedMovementModel>;
  restore(snapshot: MovementModelSnapshot): TrainedMovementModel;
}

/** A trained model produced by a {@link LocalModelBackend}. */
export interface TrainedMovementModel {
  readonly backend: string;
  readonly order: number;
  /** Distinct tokens observed during training (excludes EOS). */
  readonly vocabulary: string[];
  predictNext(context: MovementSequence): MovementPrediction | undefined;
  generate(seed: MovementSequence, maxSteps: number): MovementGeneration;
  snapshot(): MovementModelSnapshot;
}

const DEFAULT_ORDER = 3;

function contextKey(context: readonly string[]): string {
  // Tokens are opaque strings; use a separator that cannot appear in a single
  // token boundary by joining with a unit-separator control char.
  return context.join("");
}

type CountMap = Map<string, Map<string, number>>;

class NGramMovementModel implements TrainedMovementModel {
  readonly backend = "ngram";

  /** grams[o] maps a context-key of length o to (next-token → count). */
  private readonly grams: CountMap[];

  constructor(
    readonly order: number,
    grams: CountMap[],
    readonly vocabulary: string[],
  ) {
    this.grams = grams;
  }

  predictNext(context: MovementSequence): MovementPrediction | undefined {
    // Prepend BOS so a short/empty context still anchors to the start of a
    // workflow via the same backoff machinery.
    const effective = [MOVEMENT_BOS, ...context];
    const maxOrder = Math.min(this.order, effective.length);
    for (let o = maxOrder; o >= 0; o -= 1) {
      const suffix = o === 0 ? [] : effective.slice(effective.length - o);
      const table = this.grams[o]?.get(contextKey(suffix));
      if (!table || table.size === 0) {
        continue;
      }
      const ranked = rankCounts(table);
      const total = ranked.reduce((sum, entry) => sum + entry.count, 0);
      const best = ranked[0];
      if (!best || total === 0) {
        continue;
      }
      return {
        token: best.token,
        probability: best.count / total,
        backoffOrder: o,
        exact: o === maxOrder,
        alternatives: ranked.slice(1).map((entry) => ({
          token: entry.token,
          probability: entry.count / total,
        })),
      };
    }
    return undefined;
  }

  generate(seed: MovementSequence, maxSteps: number): MovementGeneration {
    const sequence = [...seed];
    const continuation: MovementSequence = [];
    let stopped = false;
    for (let step = 0; step < maxSteps; step += 1) {
      const prediction = this.predictNext(sequence);
      if (!prediction) {
        break;
      }
      if (prediction.token === MOVEMENT_EOS) {
        stopped = true;
        break;
      }
      sequence.push(prediction.token);
      continuation.push(prediction.token);
    }
    return { sequence, continuation, stopped };
  }

  snapshot(): MovementModelSnapshot {
    const grams: MovementModelSnapshot["grams"] = [];
    this.grams.forEach((table, order) => {
      for (const [key, nextCounts] of table) {
        const context = key === "" ? [] : key.split("");
        for (const [next, count] of nextCounts) {
          grams.push({ order, context, next, count });
        }
      }
    });
    return {
      version: 1,
      backend: this.backend,
      order: this.order,
      vocabulary: [...this.vocabulary],
      grams,
    };
  }
}

function rankCounts(table: Map<string, number>): Array<{ token: string; count: number }> {
  return [...table.entries()]
    .map(([token, count]) => ({ token, count }))
    // Deterministic ordering: higher count first, ties broken lexicographically.
    .sort((a, b) => (b.count !== a.count ? b.count - a.count : a.token < b.token ? -1 : a.token > b.token ? 1 : 0));
}

function emptyGrams(order: number): CountMap[] {
  return Array.from({ length: order + 1 }, () => new Map<string, Map<string, number>>());
}

function increment(table: CountMap, context: readonly string[], next: string): void {
  const key = contextKey(context);
  let nextCounts = table.get(key);
  if (!nextCounts) {
    nextCounts = new Map<string, number>();
    table.set(key, nextCounts);
  }
  nextCounts.set(next, (nextCounts.get(next) ?? 0) + 1);
}

/**
 * Deterministic variable-order Markov backend. No randomness, no external
 * process, no Date/Math.random usage — identical input yields identical model.
 */
export class NGramMovementBackend implements LocalModelBackend {
  readonly id = "ngram";

  async train(dataset: MovementDataset, options?: TrainingBackendOptions): Promise<TrainedMovementModel> {
    const order = Math.max(0, Math.floor(options?.order ?? DEFAULT_ORDER));
    const grams = emptyGrams(order);
    const vocabulary = new Set<string>();

    for (const sequence of dataset.sequences) {
      const tokens = [MOVEMENT_BOS, ...sequence, MOVEMENT_EOS];
      for (const token of sequence) {
        vocabulary.add(token);
      }
      // Start at i=1: the BOS at index 0 is only ever context, never a
      // prediction target, so it can never be emitted by `generate`.
      for (let i = 1; i < tokens.length; i += 1) {
        const next = tokens[i]!;
        const maxO = Math.min(order, i);
        for (let o = 0; o <= maxO; o += 1) {
          const context = tokens.slice(i - o, i);
          increment(grams[o]!, context, next);
        }
      }
    }

    return new NGramMovementModel(order, grams, [...vocabulary].sort());
  }

  restore(snapshot: MovementModelSnapshot): TrainedMovementModel {
    const grams = emptyGrams(snapshot.order);
    for (const entry of snapshot.grams) {
      if (entry.order < 0 || entry.order >= grams.length) {
        continue;
      }
      increment(grams[entry.order]!, entry.context, entry.next);
      // increment adds 1; overwrite with the stored count to preserve exact weights.
      grams[entry.order]!.get(contextKey(entry.context))!.set(entry.next, entry.count);
    }
    return new NGramMovementModel(snapshot.order, grams, [...snapshot.vocabulary]);
  }
}

/**
 * Registry of pluggable backends. Seeded with the deterministic n-gram backend
 * so cloud/CI has a working default; register a real on-device backend by id.
 */
export class LocalModelBackendRegistry {
  private readonly backends = new Map<string, LocalModelBackend>();

  constructor(backends: LocalModelBackend[] = [new NGramMovementBackend()]) {
    for (const backend of backends) {
      this.register(backend);
    }
  }

  register(backend: LocalModelBackend): void {
    this.backends.set(backend.id, backend);
  }

  has(id: string): boolean {
    return this.backends.has(id);
  }

  get(id: string): LocalModelBackend {
    const backend = this.backends.get(id);
    if (!backend) {
      throw new Error(`unknown local model backend: ${id}`);
    }
    return backend;
  }

  list(): string[] {
    return [...this.backends.keys()].sort();
  }
}

export function createDefaultModelBackendRegistry(): LocalModelBackendRegistry {
  return new LocalModelBackendRegistry();
}

// --- Tokenization: bridge the capture/replay event schema to movement tokens ---

export type TokenizeReplayOptions = {
  /** Include transcript (chat) events. Off by default — movements are what we model. */
  includeTranscript?: boolean;
};

/**
 * Convert a replay timeline into an ordered movement-token sequence. Tokens
 * encode the *kind* of movement (tool/source/role) so the model learns the
 * shape of a workflow rather than memorizing free-text summaries.
 */
export function tokenizeReplayEvents(
  events: readonly ReplayTimelineEvent[],
  options?: TokenizeReplayOptions,
): MovementSequence {
  const tokens: MovementSequence = [];
  for (const event of events) {
    switch (event.kind) {
      case "action":
        tokens.push(`action:${event.tool}`);
        break;
      case "observation":
        tokens.push(`observation:${event.source}`);
        break;
      case "transcript":
        if (options?.includeTranscript) {
          tokens.push(`transcript:${event.role}`);
        }
        break;
    }
  }
  return tokens;
}

export function datasetFromReplayEvents(
  replays: ReadonlyArray<readonly ReplayTimelineEvent[]>,
  options?: TokenizeReplayOptions,
): MovementDataset {
  return {
    sequences: replays
      .map((events) => tokenizeReplayEvents(events, options))
      .filter((sequence) => sequence.length > 0),
  };
}

// --- Evaluation harness: replay fidelity + generalization ---

export type MovementEvalResult = {
  /** Total next-token predictions scored. */
  predictions: number;
  /** Predictions whose top-1 token matched the held-out ground truth. */
  correct: number;
  /** correct / predictions (0 when there were no predictions). */
  accuracy: number;
  /** Fraction of correct predictions that required backing off below full order. */
  generalizedShare: number;
};

/**
 * Score a trained model against held-out sequences by predicting each token
 * from its preceding context and comparing to ground truth. A prediction that
 * matches after backing off to a shorter suffix counts as *generalization*.
 */
export function evaluateMovementModel(
  model: TrainedMovementModel,
  heldOut: MovementDataset,
): MovementEvalResult {
  let predictions = 0;
  let correct = 0;
  let generalized = 0;

  for (const sequence of heldOut.sequences) {
    const tokens = [...sequence, MOVEMENT_EOS];
    for (let i = 0; i < tokens.length; i += 1) {
      const context = tokens.slice(0, i);
      const truth = tokens[i]!;
      const prediction = model.predictNext(context);
      predictions += 1;
      if (prediction && prediction.token === truth) {
        correct += 1;
        if (!prediction.exact) {
          generalized += 1;
        }
      }
    }
  }

  return {
    predictions,
    correct,
    accuracy: predictions === 0 ? 0 : correct / predictions,
    generalizedShare: correct === 0 ? 0 : generalized / correct,
  };
}
