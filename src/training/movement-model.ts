import type { TrajectorySpan } from "../capture/trajectory.js";
import type { ReplayManifest } from "../capture/replay.js";

/**
 * Movement-model subsystem (objective #2, parts (c) train and (d) generalize).
 *
 * The capture/replay pipeline produces ordered movement *actions*. This module
 * turns those into a compact, replayable token dataset and trains a *pluggable*
 * local model on it that can (c) repeat a recorded movement sequence exactly and
 * (d) generalize to new-but-related sequences by recombining learned transitions.
 *
 * The real on-device backend (a small local model, trained via the
 * MLX/axolotl launch scripts in `runner.ts`) runs only when the user executes
 * bee-agent locally. To keep the train -> infer loop testable in the cloud, the
 * backend is an interface and this file ships a fully deterministic reference
 * implementation: a variable-order back-off Markov policy. No Date.now / random,
 * so cloud CI is reproducible.
 */

/** A single movement step, encoded as an opaque deterministic token. */
export type MovementToken = string;

/** One ordered movement example (e.g. derived from a trajectory or replay). */
export type MovementSequence = {
  id: string;
  /** Optional grouping/context label (outcome summary, app id, task...). */
  context?: string;
  tokens: MovementToken[];
};

/** A replayable training dataset of movement sequences. */
export type MovementDataset = {
  version: 1;
  sequences: MovementSequence[];
};

export type MovementPrediction = {
  token: MovementToken;
  /** Probability over real continuation tokens at this context (sums to ~1). */
  probability: number;
  /** Raw observed count backing the prediction. */
  count: number;
};

export type MovementTrainOptions = {
  /** Maximum Markov context length. Higher = more faithful replay, less general. */
  order?: number;
};

export type MovementGenerateOptions = {
  /** Hard cap on total sequence length (seed included). */
  maxLength?: number;
};

/** A trained, queryable movement policy. */
export interface TrainedMovementModel {
  readonly backend: string;
  readonly order: number;
  /** Distinct real tokens observed during training, sorted. */
  readonly vocabulary: readonly MovementToken[];
  /** Ranked next-token predictions for the given history (most recent last). */
  predictNext(context: readonly MovementToken[], options?: { topK?: number }): MovementPrediction[];
  /**
   * Deterministically roll out from `seed` by always taking the most likely
   * next token, stopping at a learned end-of-sequence or `maxLength`. Returns
   * the full sequence (seed + continuation). With a single training sequence
   * this reproduces it exactly (replay); with several it can recombine shared
   * sub-paths into new related sequences (generalization).
   */
  generate(seed?: readonly MovementToken[], options?: MovementGenerateOptions): MovementToken[];
  serialize(): SerializedMovementModel;
}

/** A pluggable training backend. Swap the Markov mock for a real local model. */
export interface MovementModelBackend {
  readonly name: string;
  train(dataset: MovementDataset, options?: MovementTrainOptions): Promise<TrainedMovementModel>;
}

export type SerializedGram = {
  /** Context length of this n-gram bucket. */
  k: number;
  /** JSON-encoded context key (array of `k` tokens / boundary markers). */
  context: string;
  distribution: Array<{ token: MovementToken; count: number }>;
};

export type SerializedMovementModel = {
  version: 1;
  backend: string;
  order: number;
  grams: SerializedGram[];
};

// Internal boundary markers. Chosen from the C0 control range so they cannot
// collide with human-readable movement tokens. Never surfaced in `vocabulary`.
const BOS = "bos";
const EOS = "eos";

function gramKey(context: readonly MovementToken[]): string {
  return JSON.stringify(context);
}

function sortEntries(entries: Array<{ token: MovementToken; count: number }>): Array<{ token: MovementToken; count: number }> {
  return [...entries].sort((a, b) => {
    if (b.count !== a.count) {
      return b.count - a.count;
    }
    return a.token < b.token ? -1 : a.token > b.token ? 1 : 0;
  });
}

class MarkovMovementModel implements TrainedMovementModel {
  readonly backend: string;
  readonly order: number;
  readonly vocabulary: readonly MovementToken[];
  /** grams[k] maps a context key (k tokens) -> token -> count. */
  private readonly grams: Array<Map<string, Map<MovementToken, number>>>;

  constructor(backend: string, order: number, grams: Array<Map<string, Map<MovementToken, number>>>) {
    this.backend = backend;
    this.order = order;
    this.grams = grams;
    const vocab = new Set<MovementToken>();
    for (const bucket of grams) {
      for (const distribution of bucket.values()) {
        for (const token of distribution.keys()) {
          if (token !== BOS && token !== EOS) {
            vocab.add(token);
          }
        }
      }
    }
    this.vocabulary = [...vocab].sort();
  }

  /** Back-off lookup including boundary markers; longest matching context wins. */
  private rawPredict(context: readonly MovementToken[]): Array<{ token: MovementToken; count: number }> {
    const padded = [...Array<MovementToken>(this.order).fill(BOS), ...context];
    for (let k = this.order; k >= 0; k -= 1) {
      const slice = k === 0 ? [] : padded.slice(padded.length - k);
      const bucket = this.grams[k];
      const distribution = bucket?.get(gramKey(slice));
      if (distribution && distribution.size > 0) {
        return sortEntries([...distribution.entries()].map(([token, count]) => ({ token, count })));
      }
    }
    return [];
  }

  predictNext(context: readonly MovementToken[], options: { topK?: number } = {}): MovementPrediction[] {
    const entries = this.rawPredict(context).filter((entry) => entry.token !== EOS && entry.token !== BOS);
    const total = entries.reduce((sum, entry) => sum + entry.count, 0);
    if (total === 0) {
      return [];
    }
    const predictions = entries.map((entry) => ({
      token: entry.token,
      count: entry.count,
      probability: entry.count / total,
    }));
    return options.topK !== undefined ? predictions.slice(0, Math.max(0, options.topK)) : predictions;
  }

  generate(seed: readonly MovementToken[] = [], options: MovementGenerateOptions = {}): MovementToken[] {
    const maxLength = options.maxLength ?? 256;
    const out = [...seed];
    while (out.length < maxLength) {
      const entries = this.rawPredict(out);
      const top = entries[0];
      if (!top || top.token === EOS) {
        break;
      }
      out.push(top.token);
    }
    return out;
  }

  serialize(): SerializedMovementModel {
    const grams: SerializedGram[] = [];
    this.grams.forEach((bucket, k) => {
      for (const [context, distribution] of bucket.entries()) {
        grams.push({
          k,
          context,
          distribution: sortEntries([...distribution.entries()].map(([token, count]) => ({ token, count }))),
        });
      }
    });
    return { version: 1, backend: this.backend, order: this.order, grams };
  }

  static fromSerialized(model: SerializedMovementModel): MarkovMovementModel {
    const grams: Array<Map<string, Map<MovementToken, number>>> = Array.from(
      { length: model.order + 1 },
      () => new Map<string, Map<MovementToken, number>>(),
    );
    for (const gram of model.grams) {
      if (gram.k < 0 || gram.k > model.order) {
        continue;
      }
      const distribution = new Map<MovementToken, number>();
      for (const entry of gram.distribution) {
        distribution.set(entry.token, entry.count);
      }
      grams[gram.k]?.set(gram.context, distribution);
    }
    return new MarkovMovementModel(model.backend, model.order, grams);
  }
}

/** Deterministic reference backend: a variable-order back-off Markov policy. */
export class MarkovMovementBackend implements MovementModelBackend {
  readonly name = "markov-mock";
  private readonly defaultOrder: number;

  constructor(options: { order?: number } = {}) {
    this.defaultOrder = clampOrder(options.order ?? 2);
  }

  async train(dataset: MovementDataset, options: MovementTrainOptions = {}): Promise<TrainedMovementModel> {
    const order = clampOrder(options.order ?? this.defaultOrder);
    const grams: Array<Map<string, Map<MovementToken, number>>> = Array.from(
      { length: order + 1 },
      () => new Map<string, Map<MovementToken, number>>(),
    );

    const add = (k: number, context: readonly MovementToken[], target: MovementToken): void => {
      const bucket = grams[k];
      if (!bucket) {
        return;
      }
      const key = gramKey(context);
      let distribution = bucket.get(key);
      if (!distribution) {
        distribution = new Map<MovementToken, number>();
        bucket.set(key, distribution);
      }
      distribution.set(target, (distribution.get(target) ?? 0) + 1);
    };

    for (const sequence of dataset.sequences) {
      if (sequence.tokens.length === 0) {
        continue;
      }
      const padded = [...Array<MovementToken>(order).fill(BOS), ...sequence.tokens, EOS];
      for (let i = order; i < padded.length; i += 1) {
        const target = padded[i]!;
        for (let k = 0; k <= order; k += 1) {
          add(k, padded.slice(i - k, i), target);
        }
      }
    }

    return new MarkovMovementModel(this.name, order, grams);
  }
}

/** Rehydrate a previously serialized model (any backend that emitted it). */
export function loadMovementModel(model: SerializedMovementModel): TrainedMovementModel {
  return MarkovMovementModel.fromSerialized(model);
}

function clampOrder(order: number): number {
  if (!Number.isFinite(order)) {
    return 1;
  }
  return Math.min(8, Math.max(1, Math.floor(order)));
}

// ---------------------------------------------------------------------------
// Dataset extraction: capture artifacts -> movement tokens
// ---------------------------------------------------------------------------

/** Deterministic token for a single recorded action. */
export function movementActionToken(action: { tool: string; summary: string }): MovementToken {
  return `${action.tool}${action.summary}`;
}

/** Decode a token back into its `{ tool, summary }` parts for replay. */
export function decodeMovementToken(token: MovementToken): { tool: string; summary: string } {
  const index = token.indexOf("");
  if (index < 0) {
    return { tool: "unknown", summary: token };
  }
  return { tool: token.slice(0, index), summary: token.slice(index + 1) };
}

/** Extract one movement sequence (actions in time order) from a trajectory. */
export function extractMovementSequence(trajectory: TrajectorySpan): MovementSequence {
  const tokens = [...trajectory.actions]
    .sort((a, b) => a.ts - b.ts)
    .map((action) => movementActionToken(action));
  return {
    id: trajectory.id,
    ...(trajectory.outcome?.summary ? { context: trajectory.outcome.summary } : {}),
    tokens,
  };
}

/** Build a dataset from many trajectories (empty action lists are dropped). */
export function extractMovementDataset(trajectories: readonly TrajectorySpan[]): MovementDataset {
  return {
    version: 1,
    sequences: trajectories
      .map((trajectory) => extractMovementSequence(trajectory))
      .filter((sequence) => sequence.tokens.length > 0),
  };
}

/** Extract the action timeline of a replay manifest as a movement sequence. */
export function movementSequenceFromReplay(manifest: ReplayManifest): MovementSequence {
  const tokens = manifest.events
    .filter((event): event is Extract<ReplayManifest["events"][number], { kind: "action" }> => event.kind === "action")
    .slice()
    .sort((a, b) => a.ts - b.ts)
    .map((event) => movementActionToken({ tool: event.tool, summary: event.summary }));
  return { id: manifest.sessionId, tokens };
}

// ---------------------------------------------------------------------------
// Evaluation harness: replay fidelity (c) and generalization (d)
// ---------------------------------------------------------------------------

export type NextTokenEvaluation = {
  total: number;
  correct: number;
  accuracy: number;
  misses: Array<{ sequenceId: string; position: number; expected: MovementToken; predicted?: MovementToken }>;
};

/**
 * Teacher-forced top-1 next-token accuracy over held-out sequences. Measures
 * how well the policy generalizes to movements it was not trained on verbatim.
 */
export function evaluateNextTokenAccuracy(
  model: TrainedMovementModel,
  sequences: readonly MovementSequence[],
): NextTokenEvaluation {
  let total = 0;
  let correct = 0;
  const misses: NextTokenEvaluation["misses"] = [];
  for (const sequence of sequences) {
    for (let i = 0; i < sequence.tokens.length; i += 1) {
      const expected = sequence.tokens[i]!;
      const predicted = model.predictNext(sequence.tokens.slice(0, i), { topK: 1 })[0]?.token;
      total += 1;
      if (predicted === expected) {
        correct += 1;
      } else {
        misses.push({ sequenceId: sequence.id, position: i, expected, ...(predicted ? { predicted } : {}) });
      }
    }
  }
  return { total, correct, accuracy: total === 0 ? 1 : correct / total, misses };
}

export type ReplayFidelityEvaluation = {
  recordedLength: number;
  generatedLength: number;
  matched: number;
  /** Fraction of recorded steps reproduced in order (1 = perfect replay). */
  fidelity: number;
};

/**
 * Roll the model out from the sequence's first token and compare to the
 * recorded movement. A model trained on this sequence should reproduce it
 * exactly (fidelity 1) — the objective's "repeat the recorded movements".
 */
export function evaluateReplayFidelity(
  model: TrainedMovementModel,
  sequence: MovementSequence,
): ReplayFidelityEvaluation {
  const recordedLength = sequence.tokens.length;
  const seed = sequence.tokens.slice(0, 1);
  const generated = model.generate(seed, { maxLength: recordedLength + 1 });
  let matched = 0;
  for (let i = 0; i < recordedLength; i += 1) {
    if (generated[i] === sequence.tokens[i]) {
      matched += 1;
    } else {
      break;
    }
  }
  return {
    recordedLength,
    generatedLength: generated.length,
    matched,
    fidelity: recordedLength === 0 ? 1 : matched / recordedLength,
  };
}
