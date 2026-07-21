import type { ReplayManifest, ReplayTimelineEvent } from "../capture/replay.js";
import type { TrajectorySpan } from "../capture/trajectory.js";

/**
 * Local-movement learning backend (standing objective #2c/#2d).
 *
 * This module implements the *in-process* seam that the external MLX/axolotl
 * training plans (see runner.ts) cannot: a pluggable model backend that learns
 * recorded movement sequences and can (a) reproduce them exactly ("repeat the
 * recorded movements") and (b) generalize to new-but-related sequences via
 * back-off over shorter contexts.
 *
 * The reference backend is a deterministic n-gram with stupid back-off. It runs
 * with zero native dependencies so it validates the capture -> dataset -> train
 * -> infer round-trip in the cloud/CI. A real on-device small model can be
 * dropped in behind {@link MovementModelBackend} without touching callers.
 */

/** A single canonical movement, e.g. an action token `"click:: submit"`. */
export type MovementToken = string;

/** One recorded movement sequence (e.g. the actions of a single trajectory). */
export type MovementSequence = {
  id: string;
  tokens: MovementToken[];
};

export type MovementDataset = {
  sequences: MovementSequence[];
};

export type MovementTrainingConfig = {
  /** Max context length the model conditions on (n-gram order = order). */
  order?: number;
};

/** A single next-movement prediction. */
export type MovementPrediction = {
  token: MovementToken;
  /** Back-off level that produced the prediction (context tokens used). */
  contextLength: number;
  /** Conditional probability of `token` given the matched context. */
  probability: number;
  /** Raw observed count of `token` after the matched context. */
  count: number;
};

/** Serialized model — plain JSON, safe to persist and rehydrate. */
export type MovementModelSnapshot = {
  version: 1;
  backend: string;
  order: number;
  /** grams[k] maps a context of exactly `k` tokens -> { token: count }. */
  grams: Array<Record<string, Record<MovementToken, number>>>;
  vocabulary: MovementToken[];
  sequenceCount: number;
};

/** A trained model. Deterministic: identical input always yields identical output. */
export interface MovementModel {
  readonly backend: string;
  readonly order: number;
  readonly vocabulary: readonly MovementToken[];
  /** Predict the single most likely next movement given a context window. */
  predictNext(context: MovementToken[]): MovementPrediction | undefined;
  /**
   * Generate a full movement sequence from an optional seed. With no seed the
   * model reproduces its most-likely recorded trajectory (objective 2c);
   * seeding with an unseen-but-related prefix exercises generalization (2d).
   */
  generate(seed?: MovementToken[], maxLength?: number): MovementToken[];
  /** Mean per-token log-probability of `tokens` under the model (higher = better fit). */
  scoreSequence(tokens: MovementToken[]): number;
  serialize(): MovementModelSnapshot;
}

/** A pluggable model backend. Register real on-device trainers behind this. */
export interface MovementModelBackend {
  readonly name: string;
  train(dataset: MovementDataset, config?: MovementTrainingConfig): MovementModel;
  restore(snapshot: MovementModelSnapshot): MovementModel;
}

// Sentinels kept out of the real-token space so they never collide with input.
const BOS = "<bos>";
const EOS = "<eos>";
const SEP = "";
const DEFAULT_ORDER = 3;
const DEFAULT_MAX_LENGTH = 256;

/**
 * Deterministic n-gram movement model with stupid back-off.
 *
 * "Repeat the recorded movements": exact recorded contexts resolve to the exact
 * recorded next token, so generation reproduces training trajectories.
 * "Generalize": an unseen context backs off to shorter suffixes (down to the
 * unigram), so related-but-novel prefixes still produce a plausible next move.
 */
class NgramMovementModel implements MovementModel {
  readonly backend = "ngram";
  constructor(
    readonly order: number,
    /** levels[k]: context of length k -> Map<nextToken, count>. */
    private readonly levels: Array<Map<string, Map<MovementToken, number>>>,
    private readonly vocab: MovementToken[],
    private readonly sequenceCount: number,
  ) {}

  get vocabulary(): readonly MovementToken[] {
    return this.vocab;
  }

  predictNext(context: MovementToken[]): MovementPrediction | undefined {
    const prediction = this.backoff([BOS, ...context]);
    if (!prediction || prediction.token === EOS) {
      return undefined;
    }
    return prediction;
  }

  generate(seed: MovementToken[] = [], maxLength: number = DEFAULT_MAX_LENGTH): MovementToken[] {
    const working: MovementToken[] = [BOS, ...seed];
    const output: MovementToken[] = [...seed];
    while (output.length < maxLength) {
      const next = this.backoff(working);
      if (!next || next.token === EOS) {
        break;
      }
      output.push(next.token);
      working.push(next.token);
    }
    return output;
  }

  scoreSequence(tokens: MovementToken[]): number {
    const augmented = [BOS, ...tokens, EOS];
    let total = 0;
    let counted = 0;
    for (let i = 1; i < augmented.length; i += 1) {
      const target = augmented[i]!;
      const context = augmented.slice(0, i);
      const probability = this.probabilityOf(context, target);
      // Floor guards against -Infinity when a target is genuinely unseen.
      total += Math.log(Math.max(probability, 1e-9));
      counted += 1;
    }
    return counted === 0 ? 0 : total / counted;
  }

  serialize(): MovementModelSnapshot {
    return {
      version: 1,
      backend: this.backend,
      order: this.order,
      grams: this.levels.map((level) => {
        const record: Record<string, Record<MovementToken, number>> = {};
        for (const [context, nexts] of level) {
          record[context] = Object.fromEntries(nexts);
        }
        return record;
      }),
      vocabulary: [...this.vocab],
      sequenceCount: this.sequenceCount,
    };
  }

  /** Argmax next token, backing off from the longest usable context to the unigram. */
  private backoff(fullContext: MovementToken[]): MovementPrediction | undefined {
    const maxLevel = Math.min(this.order - 1, fullContext.length);
    for (let k = maxLevel; k >= 0; k -= 1) {
      const contextTokens = fullContext.slice(fullContext.length - k);
      const nexts = this.levels[k]?.get(contextTokens.join(SEP));
      if (!nexts || nexts.size === 0) {
        continue;
      }
      const best = argmax(nexts);
      if (!best) {
        continue;
      }
      const totalForContext = sumCounts(nexts);
      return {
        token: best.token,
        contextLength: k,
        probability: totalForContext === 0 ? 0 : best.count / totalForContext,
        count: best.count,
      };
    }
    return undefined;
  }

  private probabilityOf(fullContext: MovementToken[], target: MovementToken): number {
    const maxLevel = Math.min(this.order - 1, fullContext.length);
    for (let k = maxLevel; k >= 0; k -= 1) {
      const contextTokens = fullContext.slice(fullContext.length - k);
      const nexts = this.levels[k]?.get(contextTokens.join(SEP));
      if (!nexts || nexts.size === 0) {
        continue;
      }
      const total = sumCounts(nexts);
      const count = nexts.get(target) ?? 0;
      if (count > 0) {
        return count / total;
      }
      if (k === 0) {
        // Unigram is the shortest context: nothing left to back off to.
        return 0;
      }
      // Stupid back-off: discounted mass from the next shorter context.
      return 0.4 * this.probabilityOf(contextTokens.slice(1), target);
    }
    return 0;
  }
}

/** Deterministic tie-break: highest count wins, then lexicographically smallest token. */
function argmax(counts: Map<MovementToken, number>): { token: MovementToken; count: number } | undefined {
  let best: { token: MovementToken; count: number } | undefined;
  for (const [token, count] of counts) {
    if (
      !best ||
      count > best.count ||
      (count === best.count && token < best.token)
    ) {
      best = { token, count };
    }
  }
  return best;
}

function sumCounts(counts: Map<MovementToken, number>): number {
  let total = 0;
  for (const value of counts.values()) {
    total += value;
  }
  return total;
}

export class NgramMovementBackend implements MovementModelBackend {
  readonly name = "ngram";

  train(dataset: MovementDataset, config: MovementTrainingConfig = {}): MovementModel {
    const order = Math.max(1, Math.floor(config.order ?? DEFAULT_ORDER));
    const levels: Array<Map<string, Map<MovementToken, number>>> = Array.from(
      { length: order },
      () => new Map<string, Map<MovementToken, number>>(),
    );
    const vocab = new Set<MovementToken>();

    for (const sequence of dataset.sequences) {
      const augmented = [BOS, ...sequence.tokens, EOS];
      for (const token of sequence.tokens) {
        vocab.add(token);
      }
      for (let i = 1; i < augmented.length; i += 1) {
        const target = augmented[i]!;
        for (let k = 0; k < order && i - k >= 0; k += 1) {
          const contextTokens = augmented.slice(i - k, i);
          if (contextTokens.length !== k) {
            continue;
          }
          const level = levels[k]!;
          const key = contextTokens.join(SEP);
          let nexts = level.get(key);
          if (!nexts) {
            nexts = new Map<MovementToken, number>();
            level.set(key, nexts);
          }
          nexts.set(target, (nexts.get(target) ?? 0) + 1);
        }
      }
    }

    return new NgramMovementModel(order, levels, [...vocab].sort(), dataset.sequences.length);
  }

  restore(snapshot: MovementModelSnapshot): MovementModel {
    const levels = snapshot.grams.map((record) => {
      const level = new Map<string, Map<MovementToken, number>>();
      for (const [context, nexts] of Object.entries(record)) {
        level.set(context, new Map(Object.entries(nexts)));
      }
      return level;
    });
    return new NgramMovementModel(snapshot.order, levels, [...snapshot.vocabulary], snapshot.sequenceCount);
  }
}

/** Registry so backends are pluggable (default: the deterministic n-gram). */
export class MovementBackendRegistry {
  private readonly backends = new Map<string, MovementModelBackend>();

  constructor(backends: MovementModelBackend[] = [new NgramMovementBackend()]) {
    for (const backend of backends) {
      this.register(backend);
    }
  }

  register(backend: MovementModelBackend): void {
    this.backends.set(backend.name, backend);
  }

  get(name: string): MovementModelBackend {
    const backend = this.backends.get(name);
    if (!backend) {
      throw new Error(`unknown movement backend: ${name} (registered: ${[...this.backends.keys()].join(", ")})`);
    }
    return backend;
  }

  list(): string[] {
    return [...this.backends.keys()].sort();
  }

  train(name: string, dataset: MovementDataset, config?: MovementTrainingConfig): MovementModel {
    return this.get(name).train(dataset, config);
  }

  restore(snapshot: MovementModelSnapshot): MovementModel {
    return this.get(snapshot.backend).restore(snapshot);
  }
}

export function defaultMovementRegistry(): MovementBackendRegistry {
  return new MovementBackendRegistry();
}

// --- Dataset builders: connect the recording pipeline to the model ----------

/** Canonicalize a movement action into a single stable token. */
export function actionToken(tool: string, summary: string): MovementToken {
  return `${tool.trim()}${SEP}${summary.trim()}`;
}

/** Extract ordered action tokens from a single replay manifest. */
export function movementTokensFromReplay(replay: ReplayManifest): MovementToken[] {
  return replay.events
    .filter((event): event is Extract<ReplayTimelineEvent, { kind: "action" }> => event.kind === "action")
    .sort((a, b) => a.ts - b.ts)
    .map((event) => actionToken(event.tool, event.summary));
}

/** Build a training dataset (one sequence per replay) from replay manifests. */
export function movementDatasetFromReplays(replays: ReplayManifest[]): MovementDataset {
  return {
    sequences: replays
      .map((replay) => ({
        id: replay.trajectoryIds.join(",") || replay.sessionId,
        tokens: movementTokensFromReplay(replay),
      }))
      .filter((sequence) => sequence.tokens.length > 0),
  };
}

/** Build a training dataset directly from trajectory spans (one sequence each). */
export function movementDatasetFromTrajectories(trajectories: TrajectorySpan[]): MovementDataset {
  return {
    sequences: trajectories
      .map((trajectory) => ({
        id: trajectory.id,
        tokens: [...trajectory.actions]
          .sort((a, b) => a.ts - b.ts)
          .map((action) => actionToken(action.tool, action.summary)),
      }))
      .filter((sequence) => sequence.tokens.length > 0),
  };
}

// --- Generalization eval harness (partial: objective 2d measurement) --------

export type MovementEvalResult = {
  /** Fraction of held-out next-token predictions that matched the recording. */
  nextTokenAccuracy: number;
  /** Number of prediction points evaluated. */
  predictions: number;
  /** Mean per-token log-probability across held-out sequences. */
  meanLogProbability: number;
};

/**
 * Measure replay fidelity on held-out (but related) sequences. For each
 * sequence, predict every next token from its prefix and compare to ground
 * truth; also report the model's mean log-probability of the held-out set.
 */
export function evaluateMovementModel(model: MovementModel, heldOut: MovementSequence[]): MovementEvalResult {
  let correct = 0;
  let predictions = 0;
  let logProbTotal = 0;
  let scored = 0;
  for (const sequence of heldOut) {
    for (let i = 0; i < sequence.tokens.length; i += 1) {
      const prediction = model.predictNext(sequence.tokens.slice(0, i));
      predictions += 1;
      if (prediction?.token === sequence.tokens[i]) {
        correct += 1;
      }
    }
    if (sequence.tokens.length > 0) {
      logProbTotal += model.scoreSequence(sequence.tokens);
      scored += 1;
    }
  }
  return {
    nextTokenAccuracy: predictions === 0 ? 0 : correct / predictions,
    predictions,
    meanLogProbability: scored === 0 ? 0 : logProbTotal / scored,
  };
}
