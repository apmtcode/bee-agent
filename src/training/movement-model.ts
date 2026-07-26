import type { ReplayTimelineEvent } from "../capture/replay.js";
import type { TrajectorySpan } from "../capture/trajectory.js";

/**
 * In-process, deterministic movement-learning model.
 *
 * The rest of the training subsystem (exporter → job-manifest → runner) prepares
 * a reviewed dataset and emits launch scripts for real on-device trainers
 * (MLX / axolotl). Those trainers only run on the user's machine. This module
 * provides the *cloud-runnable, testable* counterpart: a pluggable model backend
 * that actually learns from the recorded movement sequences so bee-agent can
 *   (c) repeat recorded movements, and
 *   (d) generalize to new-but-related movements
 * without any external process — which lets us validate the whole
 * capture → dataset → train → infer → eval loop with synthetic event streams.
 *
 * The default {@link MarkovMovementBackend} is a variable-order Markov policy
 * with Katz-style backoff. High-order context reproduces recorded movements
 * exactly; when an unseen high-order context appears it backs off to shorter
 * contexts, which is where generalization to related movements comes from. The
 * backend is deterministic (argmax with a stable tie-break), so tests are
 * reproducible and no randomness leaks into replay.
 */

/** A single learnable movement symbol (e.g. a normalized `tool` + target). */
export type MovementToken = string;

/** A tokenized movement trajectory ready for training/eval. */
export type MovementSequence = {
  trajectoryId: string;
  tokens: MovementToken[];
};

/** A tokenized dataset plus its observed vocabulary. */
export type MovementDataset = {
  sequences: MovementSequence[];
  vocabulary: MovementToken[];
};

/** Maps a recorded action event to a movement token. Pluggable per caller. */
export type MovementTokenizer = (event: {
  tool: string;
  summary: string;
}) => MovementToken;

const TOKEN_FIELD_SEPARATOR = "";

/** Default tokenizer: `tool` + a whitespace-normalized, lowercased summary. */
export const defaultMovementTokenizer: MovementTokenizer = (event) =>
  `${event.tool.trim()}${TOKEN_FIELD_SEPARATOR}${normalizeSummary(event.summary)}`;

function normalizeSummary(summary: string): string {
  return summary.trim().replace(/\s+/g, " ").toLowerCase();
}

/** Prediction produced by a trained model for a given context. */
export type MovementPrediction = {
  /** Most-likely next token, or `undefined` for an empty/unknown model. */
  token: MovementToken | undefined;
  /** Probability of `token` under the context order that produced it. */
  probability: number;
  /** Context length (Markov order) the prediction backed off to. */
  order: number;
  /** All candidate continuations at the winning order, most-likely first. */
  candidates: { token: MovementToken; probability: number }[];
};

/** Serialized model — a plain JSON value safe to persist and reload. */
export type SerializedMovementModel = {
  version: 1;
  backendId: string;
  order: number;
  /** counts[order][contextKey][token] = observation count. */
  counts: Record<string, Record<string, Record<MovementToken, number>>>;
};

/** A trained model that can predict, generate, and be persisted. */
export interface TrainedMovementModel {
  readonly backendId: string;
  /** Predict the single most-likely next movement for a context. */
  predictNext(context: MovementToken[]): MovementPrediction;
  /** Roll out `steps` movements from `prefix` (deterministic greedy decode). */
  generate(prefix: MovementToken[], steps: number): MovementToken[];
  serialize(): SerializedMovementModel;
}

export type MovementTrainingOptions = {
  /** Maximum Markov order (context length). Higher = more literal replay. */
  order?: number;
};

/**
 * A pluggable local-model backend. Swap in a real on-device implementation by
 * providing another object with this shape; the training/eval code is agnostic.
 */
export interface MovementModelBackend {
  readonly id: string;
  train(dataset: MovementDataset, options?: MovementTrainingOptions): TrainedMovementModel;
  load(serialized: SerializedMovementModel): TrainedMovementModel;
}

const DEFAULT_ORDER = 2;
const CONTEXT_SEPARATOR = "";

function contextKey(tokens: MovementToken[]): string {
  return tokens.join(CONTEXT_SEPARATOR);
}

/**
 * Deterministic argmax over a `token -> count` map. Ties break by ascending
 * token string so training is fully reproducible.
 */
function rankCounts(counts: Record<MovementToken, number>): { token: MovementToken; probability: number }[] {
  const total = Object.values(counts).reduce((sum, count) => sum + count, 0);
  if (total === 0) {
    return [];
  }
  return Object.entries(counts)
    .map(([token, count]) => ({ token, probability: count / total }))
    .sort((a, b) => (b.probability !== a.probability ? b.probability - a.probability : a.token < b.token ? -1 : 1));
}

class MarkovMovementModel implements TrainedMovementModel {
  constructor(
    readonly backendId: string,
    private readonly order: number,
    private readonly counts: Map<number, Map<string, Record<MovementToken, number>>>,
  ) {}

  predictNext(context: MovementToken[]): MovementPrediction {
    for (let candidateOrder = Math.min(this.order, context.length); candidateOrder >= 0; candidateOrder -= 1) {
      const table = this.counts.get(candidateOrder);
      if (!table) {
        continue;
      }
      const key = contextKey(context.slice(context.length - candidateOrder));
      const observed = table.get(key);
      if (!observed) {
        continue;
      }
      const ranked = rankCounts(observed);
      if (ranked.length > 0) {
        return {
          token: ranked[0].token,
          probability: ranked[0].probability,
          order: candidateOrder,
          candidates: ranked,
        };
      }
    }
    return { token: undefined, probability: 0, order: -1, candidates: [] };
  }

  generate(prefix: MovementToken[], steps: number): MovementToken[] {
    const produced: MovementToken[] = [];
    const running = [...prefix];
    for (let step = 0; step < steps; step += 1) {
      const prediction = this.predictNext(running);
      if (prediction.token === undefined) {
        break;
      }
      produced.push(prediction.token);
      running.push(prediction.token);
    }
    return produced;
  }

  serialize(): SerializedMovementModel {
    const counts: SerializedMovementModel["counts"] = {};
    for (const [order, table] of [...this.counts.entries()].sort((a, b) => a[0] - b[0])) {
      const orderRecord: Record<string, Record<MovementToken, number>> = {};
      for (const key of [...table.keys()].sort()) {
        orderRecord[key] = { ...table.get(key)! };
      }
      counts[String(order)] = orderRecord;
    }
    return { version: 1, backendId: this.backendId, order: this.order, counts };
  }
}

/**
 * Variable-order Markov backend with Katz-style backoff. Deterministic: same
 * dataset + options always yields the same model and the same predictions.
 */
export class MarkovMovementBackend implements MovementModelBackend {
  readonly id = "markov";

  train(dataset: MovementDataset, options?: MovementTrainingOptions): TrainedMovementModel {
    const order = Math.max(0, options?.order ?? DEFAULT_ORDER);
    const counts = new Map<number, Map<string, Record<MovementToken, number>>>();
    for (let currentOrder = 0; currentOrder <= order; currentOrder += 1) {
      counts.set(currentOrder, new Map());
    }
    for (const sequence of dataset.sequences) {
      for (let index = 0; index < sequence.tokens.length; index += 1) {
        const next = sequence.tokens[index];
        for (let currentOrder = 0; currentOrder <= order; currentOrder += 1) {
          if (index - currentOrder < 0) {
            break;
          }
          const context = sequence.tokens.slice(index - currentOrder, index);
          const table = counts.get(currentOrder)!;
          const key = contextKey(context);
          const observed = table.get(key) ?? {};
          observed[next] = (observed[next] ?? 0) + 1;
          table.set(key, observed);
        }
      }
    }
    return new MarkovMovementModel(this.id, order, counts);
  }

  load(serialized: SerializedMovementModel): TrainedMovementModel {
    const counts = new Map<number, Map<string, Record<MovementToken, number>>>();
    for (const [orderKey, orderRecord] of Object.entries(serialized.counts)) {
      const table = new Map<string, Record<MovementToken, number>>();
      for (const [key, tokenCounts] of Object.entries(orderRecord)) {
        table.set(key, { ...tokenCounts });
      }
      counts.set(Number(orderKey), table);
    }
    return new MarkovMovementModel(serialized.backendId, serialized.order, counts);
  }
}

/** The default backend registry. Register real on-device backends here. */
const BACKENDS: Record<string, () => MovementModelBackend> = {
  markov: () => new MarkovMovementBackend(),
};

/** Look up a registered backend by id (defaults to the deterministic markov). */
export function resolveMovementBackend(id = "markov"): MovementModelBackend {
  const factory = BACKENDS[id];
  if (!factory) {
    throw new Error(`unknown movement backend: ${id}`);
  }
  return factory();
}

/** Tokenize recorded trajectory spans into training sequences. */
export function movementSequencesFromTrajectories(
  trajectories: TrajectorySpan[],
  tokenizer: MovementTokenizer = defaultMovementTokenizer,
): MovementSequence[] {
  return trajectories.map((trajectory) => ({
    trajectoryId: trajectory.id,
    tokens: [...trajectory.actions]
      .sort((a, b) => a.ts - b.ts)
      .map((action) => tokenizer({ tool: action.tool, summary: action.summary })),
  }));
}

/** Tokenize reviewed replay manifests (the exporter's output) into sequences. */
export function movementSequencesFromReplays(
  replays: { trajectoryIds: string[]; events: ReplayTimelineEvent[] }[],
  tokenizer: MovementTokenizer = defaultMovementTokenizer,
): MovementSequence[] {
  return replays.map((replay, index) => ({
    trajectoryId: replay.trajectoryIds[0] ?? `replay-${index}`,
    tokens: replay.events
      .filter((event): event is Extract<ReplayTimelineEvent, { kind: "action" }> => event.kind === "action")
      .sort((a, b) => a.ts - b.ts)
      .map((event) => tokenizer({ tool: event.tool, summary: event.summary })),
  }));
}

/** Assemble a dataset (with a sorted, de-duplicated vocabulary) from sequences. */
export function buildMovementDataset(sequences: MovementSequence[]): MovementDataset {
  const vocabulary = new Set<MovementToken>();
  for (const sequence of sequences) {
    for (const token of sequence.tokens) {
      vocabulary.add(token);
    }
  }
  return { sequences, vocabulary: [...vocabulary].sort() };
}

/** Convenience: build a dataset and train a model in one call. */
export function trainMovementModel(
  dataset: MovementDataset,
  options?: MovementTrainingOptions & { backend?: MovementModelBackend | string },
): TrainedMovementModel {
  const backend =
    typeof options?.backend === "string"
      ? resolveMovementBackend(options.backend)
      : options?.backend ?? new MarkovMovementBackend();
  return backend.train(dataset, options);
}

export type MovementEvalResult = {
  /** Total teacher-forced next-token predictions attempted. */
  totalPredictions: number;
  /** Predictions whose top-1 token matched the held-out truth. */
  correct: number;
  /** correct / totalPredictions (0 when no predictions were attempted). */
  accuracy: number;
  /** Per-backoff-order breakdown, so you can see how much came from backoff. */
  byOrder: Record<number, { total: number; correct: number }>;
  /** Held-out sequences whose entire tail was reproduced under teacher forcing. */
  exactMatchSequences: number;
  sequenceCount: number;
};

/**
 * Generalization eval harness. Runs teacher-forced next-movement prediction
 * over held-out sequences and reports top-1 accuracy overall and per backoff
 * order. Held-out sequences that share sub-patterns with training but are not
 * identical measure generalization; the `byOrder` split shows how much accuracy
 * came from exact high-order recall vs. lower-order backoff.
 */
export function evaluateMovementModel(
  model: TrainedMovementModel,
  heldOut: MovementSequence[],
  options?: { minContext?: number },
): MovementEvalResult {
  const minContext = Math.max(0, options?.minContext ?? 0);
  const byOrder: Record<number, { total: number; correct: number }> = {};
  let totalPredictions = 0;
  let correct = 0;
  let exactMatchSequences = 0;

  for (const sequence of heldOut) {
    let sequenceExact = sequence.tokens.length > minContext + 1;
    for (let index = Math.max(1, minContext); index < sequence.tokens.length; index += 1) {
      const context = sequence.tokens.slice(0, index);
      const truth = sequence.tokens[index];
      const prediction = model.predictNext(context);
      const bucket = (byOrder[prediction.order] ??= { total: 0, correct: 0 });
      bucket.total += 1;
      totalPredictions += 1;
      if (prediction.token === truth) {
        bucket.correct += 1;
        correct += 1;
      } else {
        sequenceExact = false;
      }
    }
    if (sequenceExact && sequence.tokens.length > minContext + 1) {
      exactMatchSequences += 1;
    }
  }

  return {
    totalPredictions,
    correct,
    accuracy: totalPredictions === 0 ? 0 : correct / totalPredictions,
    byOrder,
    exactMatchSequences,
    sequenceCount: heldOut.length,
  };
}

/**
 * Deterministic synthetic movement-stream generator for validating the pipeline
 * without real OS capture. Given labeled "task templates" (each a list of
 * movement tokens), it produces `count` trajectories per template, optionally
 * perturbing them (swap/insert/drop one step) using a seeded PRNG so held-out
 * sequences are *related but not identical* to the training ones — exactly the
 * generalization case the model must handle.
 */
export function synthesizeMovementSequences(params: {
  templates: { id: string; tokens: MovementToken[] }[];
  countPerTemplate: number;
  seed?: number;
  perturb?: boolean;
}): MovementSequence[] {
  const rng = createSeededRng(params.seed ?? 1);
  const sequences: MovementSequence[] = [];
  for (const template of params.templates) {
    for (let copy = 0; copy < params.countPerTemplate; copy += 1) {
      let tokens = [...template.tokens];
      if (params.perturb && tokens.length > 2 && rng() > 0.5) {
        const pivot = 1 + Math.floor(rng() * (tokens.length - 2));
        // Swap two adjacent middle steps — a related-but-different movement.
        [tokens[pivot], tokens[pivot - 1]] = [tokens[pivot - 1], tokens[pivot]];
      }
      sequences.push({ trajectoryId: `${template.id}-${copy}`, tokens });
    }
  }
  return sequences;
}

/** Small deterministic LCG (numerical-recipes constants) — no Math.random. */
function createSeededRng(seed: number): () => number {
  let state = (seed >>> 0) || 1;
  return () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 0xffffffff;
  };
}
