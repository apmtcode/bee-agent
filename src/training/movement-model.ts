// Pluggable local movement-model backend for the local-movement learning
// subsystem (standing objective #2, parts (c) post-train a local model to
// repeat recorded movements and (d) generalize to new but related movements).
//
// The real on-device pipeline (`runner.ts`) emits an MLX/axolotl launch script
// that runs outside this process, so it cannot be exercised in the cloud. This
// module provides an *in-process*, fully deterministic model that trains on a
// reviewed movement dataset and predicts/generates the next movements — letting
// the train -> infer loop be validated with synthetic data in CI, behind a
// `MovementModelBackend` seam that a real small local model can later implement.

import type { TrajectorySpan, TrajectoryAction } from "../capture/trajectory.js";
import type { ReplayManifest } from "../capture/replay.js";

/** Sentinel tokens. START pads sequence fronts; END marks completion. */
export const MOVEMENT_START_TOKEN = "<start>";
export const MOVEMENT_END_TOKEN = "<end>";

/** Separator used to key multi-token n-gram contexts (ASCII unit separator). */
const CONTEXT_SEPARATOR = "␟";

/** A single discrete movement, canonicalized to a string token. */
export type MovementToken = string;

/** A recorded sequence of movements derived from one trajectory/replay. */
export type MovementSequence = {
  id: string;
  tokens: MovementToken[];
};

export type MovementModelConfig = {
  /** Maximum n-gram context length used for prediction (>= 1). */
  order: number;
  /** Additive (Laplace) smoothing mass; defaults to 1. Set 0 for pure counts. */
  smoothing?: number;
};

export type MovementPredictionAlternative = {
  token: MovementToken;
  probability: number;
};

export type MovementPrediction = {
  token: MovementToken;
  probability: number;
  /** Length of the context that survived back-off and produced this guess. */
  contextOrder: number;
  /** Full smoothed distribution at the back-off level, prob-desc / token-asc. */
  alternatives: MovementPredictionAlternative[];
};

export type SerializedMovementModel = {
  backend: string;
  version: 1;
  order: number;
  smoothing: number;
  vocabulary: MovementToken[];
  /** context-key -> { emitted token -> observed count }. "" is the unigram. */
  counts: Record<string, Record<MovementToken, number>>;
};

export interface MovementModel {
  readonly backend: string;
  readonly order: number;
  readonly smoothing: number;
  readonly vocabulary: readonly MovementToken[];
  /** Probability the model assigns to `token` following `context`. */
  scoreToken(context: MovementToken[], token: MovementToken): number;
  /** Most likely next movement given `context`, or undefined if untrained. */
  predictNext(context: MovementToken[]): MovementPrediction | undefined;
  /** Greedily roll out movements from `seed` until END or `maxSteps`. */
  generate(seed: MovementToken[], maxSteps: number): MovementToken[];
  toJSON(): SerializedMovementModel;
}

export interface MovementModelBackend {
  readonly name: string;
  train(dataset: MovementSequence[], config: MovementModelConfig): MovementModel;
  load(serialized: SerializedMovementModel): MovementModel;
}

// ---------------------------------------------------------------------------
// Tokenization: trajectories / replays -> movement sequences
// ---------------------------------------------------------------------------

/** Collapse a movement into a stable token so repeats map to one symbol. */
export function canonicalMovementToken(action: { tool: string; summary: string }): MovementToken {
  const tool = action.tool.trim().toLowerCase();
  const summary = action.summary.trim().toLowerCase().replace(/\s+/g, " ");
  return summary ? `${tool}::${summary}` : tool;
}

/** Convert one trajectory's actions (time-ordered) into a movement sequence. */
export function trajectoryToMovementSequence(trajectory: TrajectorySpan): MovementSequence {
  const actions = [...trajectory.actions].sort((a, b) => a.ts - b.ts);
  return { id: trajectory.id, tokens: actions.map(canonicalMovementToken) };
}

/** Group a replay manifest's action events by trajectory into sequences. */
export function replayManifestToMovementSequences(manifest: ReplayManifest): MovementSequence[] {
  const byTrajectory = new Map<string, { ts: number; token: MovementToken }[]>();
  for (const event of manifest.events) {
    if (event.kind !== "action") {
      continue;
    }
    const bucket = byTrajectory.get(event.trajectoryId) ?? [];
    bucket.push({ ts: event.ts, token: canonicalMovementToken(event) });
    byTrajectory.set(event.trajectoryId, bucket);
  }
  return manifest.trajectoryIds
    .filter((id) => byTrajectory.has(id))
    .map((id) => ({
      id,
      tokens: [...byTrajectory.get(id)!].sort((a, b) => a.ts - b.ts).map((entry) => entry.token),
    }));
}

/** Build a training dataset from reviewed trajectories, dropping empty ones. */
export function buildMovementDataset(trajectories: TrajectorySpan[]): MovementSequence[] {
  return trajectories
    .map(trajectoryToMovementSequence)
    .filter((sequence) => sequence.tokens.length > 0);
}

// ---------------------------------------------------------------------------
// Deterministic back-off n-gram model + backend (the cloud-testable mock)
// ---------------------------------------------------------------------------

class BackoffNgramModel implements MovementModel {
  readonly backend = "deterministic-ngram";
  readonly order: number;
  readonly smoothing: number;
  readonly vocabulary: readonly MovementToken[];
  private readonly counts: Map<string, Map<MovementToken, number>>;

  constructor(
    order: number,
    smoothing: number,
    vocabulary: MovementToken[],
    counts: Map<string, Map<MovementToken, number>>,
  ) {
    this.order = order;
    this.smoothing = smoothing;
    this.vocabulary = vocabulary;
    this.counts = counts;
  }

  /** Longest suffix of `context` (length 0..order) that was observed. */
  private resolveContext(context: MovementToken[]): { key: string; length: number } {
    const maxLength = Math.min(context.length, this.order);
    for (let length = maxLength; length >= 1; length -= 1) {
      const key = context.slice(context.length - length).join(CONTEXT_SEPARATOR);
      if (this.counts.has(key)) {
        return { key, length };
      }
    }
    return { key: "", length: 0 };
  }

  private distributionAt(key: string): Map<MovementToken, number> {
    return this.counts.get(key) ?? new Map();
  }

  scoreToken(context: MovementToken[], token: MovementToken): number {
    if (this.vocabulary.length === 0) {
      return 0;
    }
    const { key } = this.resolveContext(context);
    const distribution = this.distributionAt(key);
    let total = 0;
    for (const value of distribution.values()) {
      total += value;
    }
    const numerator = (distribution.get(token) ?? 0) + this.smoothing;
    const denominator = total + this.smoothing * this.vocabulary.length;
    return denominator === 0 ? 0 : numerator / denominator;
  }

  predictNext(context: MovementToken[]): MovementPrediction | undefined {
    if (this.vocabulary.length === 0) {
      return undefined;
    }
    const { key, length } = this.resolveContext(context);
    const distribution = this.distributionAt(key);
    let total = 0;
    for (const value of distribution.values()) {
      total += value;
    }
    const denominator = total + this.smoothing * this.vocabulary.length;
    if (denominator === 0) {
      return undefined;
    }
    const alternatives: MovementPredictionAlternative[] = this.vocabulary
      .map((candidate) => ({
        token: candidate,
        probability: ((distribution.get(candidate) ?? 0) + this.smoothing) / denominator,
      }))
      // Deterministic ordering: probability desc, then token asc as a tie-break.
      .sort((a, b) => (b.probability - a.probability) || (a.token < b.token ? -1 : a.token > b.token ? 1 : 0));

    const best = alternatives[0];
    if (!best) {
      return undefined;
    }
    return { token: best.token, probability: best.probability, contextOrder: length, alternatives };
  }

  generate(seed: MovementToken[], maxSteps: number): MovementToken[] {
    const padding: MovementToken[] = Array.from({ length: this.order }, () => MOVEMENT_START_TOKEN);
    const working = [...padding, ...seed];
    const output: MovementToken[] = [];
    for (let step = 0; step < maxSteps; step += 1) {
      const prediction = this.predictNext(working);
      if (!prediction || prediction.token === MOVEMENT_END_TOKEN) {
        break;
      }
      output.push(prediction.token);
      working.push(prediction.token);
    }
    return output;
  }

  toJSON(): SerializedMovementModel {
    const counts: Record<string, Record<MovementToken, number>> = {};
    for (const [key, distribution] of this.counts) {
      const entry: Record<MovementToken, number> = {};
      for (const [token, value] of distribution) {
        entry[token] = value;
      }
      counts[key] = entry;
    }
    return {
      backend: this.backend,
      version: 1,
      order: this.order,
      smoothing: this.smoothing,
      vocabulary: [...this.vocabulary],
      counts,
    };
  }
}

/**
 * Deterministic n-gram backend. Trains a back-off Markov model over movement
 * tokens; predictions are pure argmax with a lexicographic tie-break, so the
 * same dataset always yields the same model and the same rollouts. Acts as the
 * always-available mock so cloud/CI can validate the train -> infer loop; a real
 * on-device small model can implement `MovementModelBackend` to replace it.
 */
export class DeterministicMovementBackend implements MovementModelBackend {
  readonly name = "deterministic-ngram";

  train(dataset: MovementSequence[], config: MovementModelConfig): MovementModel {
    const order = Math.max(1, Math.floor(config.order));
    const smoothing = config.smoothing ?? 1;
    if (smoothing < 0) {
      throw new Error("smoothing must be >= 0");
    }
    const counts = new Map<string, Map<MovementToken, number>>();
    const vocabulary = new Set<MovementToken>();

    for (const sequence of dataset) {
      // Pad the front with START so the first real movements have context, and
      // terminate with END so the model can learn when a workflow finishes.
      const padded = [
        ...Array.from({ length: order }, () => MOVEMENT_START_TOKEN),
        ...sequence.tokens,
        MOVEMENT_END_TOKEN,
      ];
      for (let i = order; i < padded.length; i += 1) {
        const token = padded[i]!;
        vocabulary.add(token);
        for (let length = 0; length <= order; length += 1) {
          const key = padded.slice(i - length, i).join(CONTEXT_SEPARATOR);
          const distribution = counts.get(key) ?? new Map<MovementToken, number>();
          distribution.set(token, (distribution.get(token) ?? 0) + 1);
          counts.set(key, distribution);
        }
      }
    }

    return new BackoffNgramModel(order, smoothing, [...vocabulary].sort(), counts);
  }

  load(serialized: SerializedMovementModel): MovementModel {
    const counts = new Map<string, Map<MovementToken, number>>();
    for (const [key, distribution] of Object.entries(serialized.counts)) {
      counts.set(key, new Map(Object.entries(distribution)));
    }
    const vocabulary = [...serialized.vocabulary].sort();
    return new BackoffNgramModel(serialized.order, serialized.smoothing, vocabulary, counts);
  }
}

// ---------------------------------------------------------------------------
// Generalization eval harness
// ---------------------------------------------------------------------------

export type MovementEvalResult = {
  /** Held-out sequences scored. */
  sequences: number;
  /** Next-token predictions evaluated across all sequences. */
  predictions: number;
  /** Top-1 predictions that matched the held-out continuation. */
  correct: number;
  /** correct / predictions (0 when there is nothing to predict). */
  accuracy: number;
  /** Mean negative log-probability assigned to the true next token. */
  averageLogLoss: number;
  /** exp(averageLogLoss): lower is better, 1 is perfect. */
  perplexity: number;
};

/**
 * Measure how well a trained model reproduces and generalizes to held-out
 * movement sequences: walk each sequence position-by-position, predicting the
 * next token from the true prefix and scoring top-1 accuracy + perplexity. END
 * is appended so "knowing when to stop" is part of the score.
 */
export function evaluateMovementModel(
  model: MovementModel,
  heldOut: MovementSequence[],
): MovementEvalResult {
  const epsilon = 1e-12;
  let predictions = 0;
  let correct = 0;
  let logLossSum = 0;

  for (const sequence of heldOut) {
    const tokens = [...sequence.tokens, MOVEMENT_END_TOKEN];
    const context: MovementToken[] = [
      ...Array.from({ length: model.order }, () => MOVEMENT_START_TOKEN),
    ];
    for (const trueToken of tokens) {
      const prediction = model.predictNext(context);
      predictions += 1;
      if (prediction?.token === trueToken) {
        correct += 1;
      }
      const probability = model.scoreToken(context, trueToken);
      logLossSum += -Math.log(Math.max(probability, epsilon));
      context.push(trueToken);
    }
  }

  const accuracy = predictions === 0 ? 0 : correct / predictions;
  const averageLogLoss = predictions === 0 ? 0 : logLossSum / predictions;
  return {
    sequences: heldOut.length,
    predictions,
    correct,
    accuracy,
    averageLogLoss,
    perplexity: Math.exp(averageLogLoss),
  };
}

// ---------------------------------------------------------------------------
// Synthetic movement-stream generator (validate pipelines without real OS input)
// ---------------------------------------------------------------------------

/** Deterministic LCG so synthetic data is reproducible without Math.random. */
function createRng(seed: number): () => number {
  let state = (Math.floor(seed) % 2147483647 + 2147483647) % 2147483647 || 1;
  return () => {
    state = (state * 48271) % 2147483647;
    return (state - 1) / 2147483646;
  };
}

export type SyntheticMovementOptions = {
  seed: number;
  count: number;
  /**
   * Ordered movement steps to emit. Each step may declare `variants`; the
   * generator picks one deterministically per sequence, producing related but
   * non-identical trajectories so generalization can be measured.
   */
  workflow?: { variants: MovementToken[] }[];
};

const DEFAULT_WORKFLOW: { variants: MovementToken[] }[] = [
  { variants: ["focus::editor window"] },
  { variants: ["click::file menu", "key::cmd+shift+p"] },
  { variants: ["key::save document", "click::save button"] },
  { variants: ["wait::disk flush"] },
  { variants: ["observe::saved indicator"] },
];

/**
 * Produce deterministic synthetic movement sequences that imitate a recorded
 * workflow with branching variants. Lets capture -> dataset -> replay/train
 * round-trips be validated end-to-end without any real OS input.
 */
export function generateSyntheticMovementSequences(
  options: SyntheticMovementOptions,
): MovementSequence[] {
  const workflow = options.workflow ?? DEFAULT_WORKFLOW;
  const rng = createRng(options.seed);
  const sequences: MovementSequence[] = [];
  for (let index = 0; index < Math.max(0, Math.floor(options.count)); index += 1) {
    const tokens = workflow.map((step) => {
      const choice = Math.min(step.variants.length - 1, Math.floor(rng() * step.variants.length));
      return step.variants[choice]!;
    });
    sequences.push({ id: `synthetic-${options.seed}-${index}`, tokens });
  }
  return sequences;
}
