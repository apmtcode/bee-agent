import type { ReplayManifest, ReplayTimelineEvent } from "../capture/replay.js";
import type { TrajectorySpan } from "../capture/trajectory.js";

/**
 * In-process, cloud-runnable movement-model subsystem.
 *
 * Standing objective #2 (c) + (d): post-train a *local* model on recorded
 * movement data so it can (c) repeat the recorded movements and (d) generalize
 * to new-but-related movements. The real on-device training pipeline
 * (`runner.ts`) shells out to mlx/axolotl on Apple Silicon and cannot run in
 * the cloud. This module provides a deterministic, dependency-free backend that
 * DOES run in the cloud/CI, so the capture -> dataset -> train -> infer ->
 * generalize loop is exercisable end-to-end without a real machine.
 *
 * The {@link MovementModelBackend} interface is the pluggable seam: the shipped
 * {@link MarkovMovementBackend} is a small statistical model (an order-k Markov
 * chain with Katz-style backoff); a real neural/on-device backend implements
 * the same interface and is registered in the {@link MovementBackendRegistry}.
 */

/** A single normalized movement token, e.g. `"mouse.click"` or `"keyboard.type"`. */
export type MovementToken = string;

/** Boundary token marking the start of a sequence (lets a model predict from empty context). */
export const MOVEMENT_START_TOKEN: MovementToken = "<s>";

/** An ordered run of movement tokens derived from one trajectory or replay. */
export type MovementSequence = {
  trajectoryId: string;
  tokens: MovementToken[];
};

/** A training dataset: a vocabulary plus the token sequences to learn from. */
export type MovementDataset = {
  version: 1;
  vocab: MovementToken[];
  sequences: MovementSequence[];
};

/** One candidate next-token with its (learned) probability. */
export type MovementCandidate = {
  token: MovementToken;
  probability: number;
};

/** A next-token prediction: the argmax token plus the ranked candidate list and the backoff order used. */
export type MovementPrediction = {
  token: MovementToken;
  probability: number;
  /** Length of the context actually matched after backoff (0 = unigram fallback). */
  order: number;
  candidates: MovementCandidate[];
};

/** The scored likelihood of a whole sequence under a model. */
export type MovementSequenceScore = {
  tokenCount: number;
  logProb: number;
  meanLogProb: number;
  perplexity: number;
};

/** Serialized form of a trained model, for persistence alongside training artifacts. */
export type SerializedMovementModel = {
  version: 1;
  backendId: string;
  order: number;
  vocab: MovementToken[];
  /** contextKey -> (nextToken -> count). Context key is tokens joined by the unit-separator. */
  counts: Record<string, Record<MovementToken, number>>;
};

export type MovementTrainOptions = {
  /** Markov order (max context length). Defaults to 2. */
  order?: number;
};

/** A trained model. Deterministic: {@link generate} uses greedy argmax with lexicographic tie-breaking. */
export interface MovementModel {
  readonly backendId: string;
  readonly order: number;
  readonly vocab: readonly MovementToken[];
  /** Predict the next token given prior context (uses longest matching context, backing off). */
  predictNext(context: readonly MovementToken[]): MovementPrediction | undefined;
  /** Greedily roll out `steps` tokens from `seed`. Stops early if no prediction is available. */
  generate(seed: readonly MovementToken[], steps: number): MovementToken[];
  /** Add-one-smoothed log-likelihood of a token sequence (for fidelity/perplexity eval). */
  scoreSequence(tokens: readonly MovementToken[]): MovementSequenceScore;
  toJSON(): SerializedMovementModel;
}

/** A pluggable training backend. Implement this to swap in a real on-device model. */
export interface MovementModelBackend {
  readonly id: string;
  train(dataset: MovementDataset, options?: MovementTrainOptions): MovementModel;
}

// Unit-separator control char keeps multi-token context keys unambiguous
// (["a","b"] must not collide with ["ab"]).
const CONTEXT_SEPARATOR = String.fromCharCode(31);

// --------------------------------------------------------------------------
// Tokenization: capture artifacts -> movement sequences
// --------------------------------------------------------------------------

/** Normalize an action tool string into a stable movement token. */
export function normalizeMovementToken(tool: string): MovementToken {
  const normalized = tool.trim().toLowerCase().replace(/\s+/g, "-");
  return normalized.length > 0 ? normalized : "unknown";
}

/** Build a movement sequence from a trajectory's actions (ordered by timestamp). */
export function buildMovementSequenceFromTrajectory(trajectory: TrajectorySpan): MovementSequence {
  const tokens = [...trajectory.actions]
    .sort((a, b) => a.ts - b.ts)
    .map((action) => normalizeMovementToken(action.tool));
  return { trajectoryId: trajectory.id, tokens };
}

/** Build movement sequences from a replay manifest, one per trajectory it references. */
export function buildMovementSequencesFromReplay(manifest: ReplayManifest): MovementSequence[] {
  const byTrajectory = new Map<string, { ts: number; token: MovementToken }[]>();
  for (const event of manifest.events) {
    if (event.kind !== "action") {
      continue;
    }
    const actionEvent = event as Extract<ReplayTimelineEvent, { kind: "action" }>;
    const list = byTrajectory.get(actionEvent.trajectoryId) ?? [];
    list.push({ ts: actionEvent.ts, token: normalizeMovementToken(actionEvent.tool) });
    byTrajectory.set(actionEvent.trajectoryId, list);
  }
  return manifest.trajectoryIds
    .filter((trajectoryId) => byTrajectory.has(trajectoryId))
    .map((trajectoryId) => ({
      trajectoryId,
      tokens: (byTrajectory.get(trajectoryId) ?? [])
        .sort((a, b) => a.ts - b.ts)
        .map((entry) => entry.token),
    }));
}

/** Assemble a dataset (with a sorted, de-duplicated vocabulary) from sequences. */
export function buildMovementDataset(sequences: MovementSequence[]): MovementDataset {
  const vocab = new Set<MovementToken>();
  const kept: MovementSequence[] = [];
  for (const sequence of sequences) {
    if (sequence.tokens.length === 0) {
      continue;
    }
    for (const token of sequence.tokens) {
      vocab.add(token);
    }
    kept.push({ trajectoryId: sequence.trajectoryId, tokens: [...sequence.tokens] });
  }
  return {
    version: 1,
    vocab: [...vocab].sort((a, b) => (a < b ? -1 : a > b ? 1 : 0)),
    sequences: kept,
  };
}

// --------------------------------------------------------------------------
// Markov backend (deterministic statistical model)
// --------------------------------------------------------------------------

class MarkovMovementModel implements MovementModel {
  readonly backendId: string;
  readonly order: number;
  readonly vocab: readonly MovementToken[];
  private readonly counts: Map<string, Map<MovementToken, number>>;

  constructor(params: {
    backendId: string;
    order: number;
    vocab: MovementToken[];
    counts: Map<string, Map<MovementToken, number>>;
  }) {
    this.backendId = params.backendId;
    this.order = params.order;
    this.vocab = params.vocab;
    this.counts = params.counts;
  }

  predictNext(context: readonly MovementToken[]): MovementPrediction | undefined {
    for (let k = Math.min(this.order, context.length); k >= 0; k -= 1) {
      const key = contextKey(context.slice(context.length - k));
      const distribution = this.counts.get(key);
      if (!distribution || distribution.size === 0) {
        continue;
      }
      const total = sumCounts(distribution);
      const candidates = [...distribution.entries()]
        .map(([token, count]) => ({ token, probability: count / total }))
        .sort((a, b) =>
          b.probability !== a.probability
            ? b.probability - a.probability
            : a.token < b.token
              ? -1
              : 1,
        );
      const best = candidates[0];
      if (!best) {
        continue;
      }
      return { token: best.token, probability: best.probability, order: k, candidates };
    }
    return undefined;
  }

  generate(seed: readonly MovementToken[], steps: number): MovementToken[] {
    const produced: MovementToken[] = [];
    const context = [...seed];
    for (let step = 0; step < steps; step += 1) {
      const prediction = this.predictNext(context);
      if (!prediction) {
        break;
      }
      produced.push(prediction.token);
      context.push(prediction.token);
    }
    return produced;
  }

  scoreSequence(tokens: readonly MovementToken[]): MovementSequenceScore {
    const vocabSize = Math.max(1, this.vocab.length);
    let logProb = 0;
    for (let i = 0; i < tokens.length; i += 1) {
      const context = tokens.slice(0, i);
      const token = tokens[i]!;
      logProb += Math.log(this.smoothedProbability(context, token, vocabSize));
    }
    const tokenCount = tokens.length;
    const meanLogProb = tokenCount > 0 ? logProb / tokenCount : 0;
    return {
      tokenCount,
      logProb,
      meanLogProb,
      perplexity: tokenCount > 0 ? Math.exp(-meanLogProb) : 1,
    };
  }

  private smoothedProbability(
    context: readonly MovementToken[],
    token: MovementToken,
    vocabSize: number,
  ): number {
    for (let k = Math.min(this.order, context.length); k >= 0; k -= 1) {
      const key = contextKey(context.slice(context.length - k));
      const distribution = this.counts.get(key);
      if (!distribution || distribution.size === 0) {
        continue;
      }
      // Add-one smoothing over the vocabulary so unseen-but-in-vocab tokens stay non-zero.
      const total = sumCounts(distribution) + vocabSize;
      return ((distribution.get(token) ?? 0) + 1) / total;
    }
    return 1 / vocabSize;
  }

  toJSON(): SerializedMovementModel {
    const counts: Record<string, Record<MovementToken, number>> = {};
    for (const [key, distribution] of this.counts) {
      counts[key] = Object.fromEntries(distribution);
    }
    return {
      version: 1,
      backendId: this.backendId,
      order: this.order,
      vocab: [...this.vocab],
      counts,
    };
  }
}

/**
 * Order-k Markov backend with Katz-style backoff.
 *
 * Learning: counts every `context -> nextToken` transition for context lengths
 * `0..order`. Repetition (objective 2c): greedy argmax over the longest matched
 * context reproduces deterministic recorded runs exactly. Generalization
 * (objective 2d): when a held-out sequence presents a context never seen at the
 * full order, prediction backs off to a shorter (seen) context, so related
 * movements still yield a plausible, in-vocabulary continuation.
 */
export class MarkovMovementBackend implements MovementModelBackend {
  readonly id = "markov";

  train(dataset: MovementDataset, options: MovementTrainOptions = {}): MovementModel {
    const order = Math.max(1, Math.floor(options.order ?? 2));
    const counts = new Map<string, Map<MovementToken, number>>();

    for (const sequence of dataset.sequences) {
      // Prefix with a start boundary so the model can predict the first move.
      const tokens = [MOVEMENT_START_TOKEN, ...sequence.tokens];
      for (let i = 1; i < tokens.length; i += 1) {
        const next = tokens[i]!;
        for (let k = 0; k <= order; k += 1) {
          if (k > i) {
            break;
          }
          const context = tokens.slice(i - k, i);
          const key = contextKey(context);
          const distribution = counts.get(key) ?? new Map<MovementToken, number>();
          distribution.set(next, (distribution.get(next) ?? 0) + 1);
          counts.set(key, distribution);
        }
      }
    }

    return new MarkovMovementModel({ backendId: this.id, order, vocab: [...dataset.vocab], counts });
  }
}

/** Rehydrate a model from its serialized form (any backend id round-trips through the Markov shape). */
export function deserializeMovementModel(serialized: SerializedMovementModel): MovementModel {
  const counts = new Map<string, Map<MovementToken, number>>();
  for (const [key, distribution] of Object.entries(serialized.counts)) {
    counts.set(key, new Map(Object.entries(distribution)));
  }
  return new MarkovMovementModel({
    backendId: serialized.backendId,
    order: serialized.order,
    vocab: [...serialized.vocab],
    counts,
  });
}

// --------------------------------------------------------------------------
// Backend registry (the pluggable seam)
// --------------------------------------------------------------------------

export class MovementBackendRegistry {
  private readonly backends = new Map<string, MovementModelBackend>();

  register(backend: MovementModelBackend): this {
    this.backends.set(backend.id, backend);
    return this;
  }

  get(id: string): MovementModelBackend | undefined {
    return this.backends.get(id);
  }

  list(): MovementModelBackend[] {
    return [...this.backends.values()];
  }
}

/** A registry preloaded with the built-in deterministic backend. */
export function createDefaultMovementBackendRegistry(): MovementBackendRegistry {
  return new MovementBackendRegistry().register(new MarkovMovementBackend());
}

// --------------------------------------------------------------------------
// Evaluation harness
// --------------------------------------------------------------------------

export type ReplayFidelityReport = {
  sequenceCount: number;
  exactReplays: number;
  /** Fraction of sequences the model reproduces exactly from their first token. */
  replayFidelity: number;
  meanLogProb: number;
  meanPerplexity: number;
};

/**
 * Objective 2(c): measure how faithfully the model repeats recorded movements.
 * For each sequence, seed the model with the first token and greedily roll out
 * the rest, comparing to the original.
 */
export function evaluateReplayFidelity(
  model: MovementModel,
  sequences: MovementSequence[],
): ReplayFidelityReport {
  let exactReplays = 0;
  let logProbSum = 0;
  let perplexitySum = 0;
  let scored = 0;

  for (const sequence of sequences) {
    if (sequence.tokens.length === 0) {
      continue;
    }
    const [first, ...rest] = sequence.tokens;
    const generated = model.generate([MOVEMENT_START_TOKEN, first!], rest.length);
    if (arraysEqual(generated, rest)) {
      exactReplays += 1;
    }
    const score = model.scoreSequence(sequence.tokens);
    logProbSum += score.meanLogProb;
    perplexitySum += score.perplexity;
    scored += 1;
  }

  return {
    sequenceCount: scored,
    exactReplays,
    replayFidelity: scored > 0 ? exactReplays / scored : 0,
    meanLogProb: scored > 0 ? logProbSum / scored : 0,
    meanPerplexity: scored > 0 ? perplexitySum / scored : 0,
  };
}

export type GeneralizationReport = {
  contextCount: number;
  predictedCount: number;
  /** Fraction of held-out contexts for which the model returns an in-vocabulary prediction. */
  coverage: number;
  /** Fraction of held-out contexts whose predicted token matched the actual next token. */
  nextTokenAccuracy: number;
  meanPerplexity: number;
};

/**
 * Objective 2(d): measure generalization to new-but-related sequences the model
 * was NOT trained on. Walks each held-out sequence position by position, asking
 * the model to predict the next token from the prefix, and reports coverage
 * (did backoff yield any valid prediction), next-token accuracy, and perplexity.
 */
export function evaluateGeneralization(
  model: MovementModel,
  heldOut: MovementSequence[],
): GeneralizationReport {
  const vocab = new Set(model.vocab);
  let contextCount = 0;
  let predictedCount = 0;
  let correct = 0;
  let perplexitySum = 0;
  let scored = 0;

  for (const sequence of heldOut) {
    if (sequence.tokens.length === 0) {
      continue;
    }
    const tokens = [MOVEMENT_START_TOKEN, ...sequence.tokens];
    for (let i = 1; i < tokens.length; i += 1) {
      contextCount += 1;
      const prediction = model.predictNext(tokens.slice(0, i));
      if (prediction && vocab.has(prediction.token)) {
        predictedCount += 1;
        if (prediction.token === tokens[i]) {
          correct += 1;
        }
      }
    }
    perplexitySum += model.scoreSequence(sequence.tokens).perplexity;
    scored += 1;
  }

  return {
    contextCount,
    predictedCount,
    coverage: contextCount > 0 ? predictedCount / contextCount : 0,
    nextTokenAccuracy: contextCount > 0 ? correct / contextCount : 0,
    meanPerplexity: scored > 0 ? perplexitySum / scored : 0,
  };
}

// --------------------------------------------------------------------------
// helpers
// --------------------------------------------------------------------------

function contextKey(context: readonly MovementToken[]): string {
  return context.join(CONTEXT_SEPARATOR);
}

function sumCounts(distribution: Map<MovementToken, number>): number {
  let total = 0;
  for (const count of distribution.values()) {
    total += count;
  }
  return total;
}

function arraysEqual(a: readonly MovementToken[], b: readonly MovementToken[]): boolean {
  if (a.length !== b.length) {
    return false;
  }
  for (let i = 0; i < a.length; i += 1) {
    if (a[i] !== b[i]) {
      return false;
    }
  }
  return true;
}
