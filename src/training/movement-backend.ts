import type { ReplayManifest } from "../capture/replay.js";
import type { TrajectoryAction, TrajectorySpan } from "../capture/trajectory.js";

/**
 * Pluggable local-model backend for the movement-learning subsystem.
 *
 * The runner in `runner.ts` prepares an *external* on-device training launch
 * (mlx / axolotl). That path only executes when the user runs bee-agent on a
 * real machine. This module provides the complementary in-process seam: a
 * deterministic backend interface plus a reference implementation that actually
 * trains on a reviewed movement dataset and performs inference — so the
 * capture → dataset → train → replay/generalize loop can be exercised and
 * regression-tested entirely in the cloud without OS access or a subprocess.
 *
 * A movement is modelled as an ordered sequence of discrete tokens (one per
 * recorded action). "Training" learns the sequence statistics; "inference"
 * predicts the next token given a context, which supports two behaviours:
 *   - exact replay of a recorded movement (greedy generation reproduces it), and
 *   - generalization to a novel-but-related movement (an unseen context backs
 *     off to the longest seen suffix and continues plausibly).
 */

/** A single discretized action in a movement sequence. */
export type MovementToken = string;

/**
 * Sentinel appended to every training sequence so the model learns where a
 * movement terminates. Generation stops when this token is predicted, and it is
 * never emitted in generated output.
 */
export const MOVEMENT_END_TOKEN = "\u0001__movement_end__";

/** Separator used to encode a multi-token context into a single map key. */
const CONTEXT_SEPARATOR = "\u0001";

/** One reviewed movement, ready for training. */
export type MovementSample = {
  id: string;
  tokens: MovementToken[];
  /** Optional trajectory reward; used to weight higher-value movements. */
  reward?: number;
  sessionId?: string;
};

export type MovementDataset = {
  version: 1;
  createdAt?: string;
  samples: MovementSample[];
};

export type MovementModelMetadata = {
  backend: string;
  order: number;
  sampleCount: number;
  tokenCount: number;
  vocabularySize: number;
  rewardWeighted: boolean;
};

/**
 * A trained, JSON-serializable movement model. `transitions` holds every
 * back-off level 0..order: the key is the context (the preceding `k` tokens
 * joined by {@link CONTEXT_SEPARATOR}, empty string for the unigram level) and
 * the value maps each observed next token to its accumulated weight.
 */
export type MovementModel = {
  version: 1;
  backend: string;
  order: number;
  vocabulary: MovementToken[];
  startDistribution: Record<MovementToken, number>;
  transitions: Record<string, Record<MovementToken, number>>;
  metadata: MovementModelMetadata;
};

export type MovementPrediction = {
  token: MovementToken;
  probability: number;
};

export type MovementTrainConfig = {
  /** Markov order (context length). Defaults to 2. */
  order?: number;
  /** Weight transitions by `1 + max(reward, 0)`. Defaults to true. */
  rewardWeighting?: boolean;
};

export type MovementPredictOptions = {
  /** Cap on the number of ranked candidates returned. Defaults to all. */
  topK?: number;
};

export type MovementGenerateOptions = {
  /** Tokens to prime generation with (e.g. the first observed action). */
  seedContext?: MovementToken[];
  /** Hard cap on generated length, excluding the seed. Defaults to 256. */
  maxSteps?: number;
};

/**
 * The pluggable backend contract. A real on-device small-model backend can
 * implement this same interface and be swapped in wherever the mock is used.
 */
export interface MovementTrainingBackend {
  readonly name: string;
  train(dataset: MovementDataset, config?: MovementTrainConfig): Promise<MovementModel>;
  /** Ranked next-token candidates for a context, most likely first. */
  predict(model: MovementModel, context: MovementToken[], options?: MovementPredictOptions): MovementPrediction[];
  /** Greedy, deterministic full-sequence generation (replay / generalization). */
  generate(model: MovementModel, options?: MovementGenerateOptions): MovementToken[];
}

function contextKey(tokens: MovementToken[]): string {
  return tokens.join(CONTEXT_SEPARATOR);
}

function addWeight(bucket: Record<MovementToken, number>, token: MovementToken, weight: number): void {
  bucket[token] = (bucket[token] ?? 0) + weight;
}

/**
 * Deterministic n-gram (Markov) backend. Suitable as the cloud/CI mock and as a
 * genuinely useful lightweight local policy: it reproduces recorded movements
 * exactly and generalizes via stupid-backoff to the longest seen context.
 */
export class MarkovMovementBackend implements MovementTrainingBackend {
  readonly name = "markov";

  async train(dataset: MovementDataset, config: MovementTrainConfig = {}): Promise<MovementModel> {
    const order = Math.max(1, Math.floor(config.order ?? 2));
    const rewardWeighting = config.rewardWeighting ?? true;

    const transitions: Record<string, Record<MovementToken, number>> = {};
    const startDistribution: Record<MovementToken, number> = {};
    const vocabulary = new Set<MovementToken>();
    let tokenCount = 0;

    for (const sample of dataset.samples) {
      if (sample.tokens.length === 0) {
        continue;
      }
      const weight = rewardWeighting ? 1 + Math.max(sample.reward ?? 0, 0) : 1;
      // Terminate every sequence so generation learns to stop.
      const sequence = [...sample.tokens, MOVEMENT_END_TOKEN];

      addWeight(startDistribution, sequence[0]!, weight);

      for (let i = 0; i < sequence.length; i += 1) {
        const next = sequence[i]!;
        if (next !== MOVEMENT_END_TOKEN) {
          vocabulary.add(next);
          tokenCount += 1;
        }
        const maxK = Math.min(order, i);
        for (let k = 0; k <= maxK; k += 1) {
          const context = sequence.slice(i - k, i);
          const key = contextKey(context);
          const bucket = (transitions[key] ??= {});
          addWeight(bucket, next, weight);
        }
      }
    }

    return {
      version: 1,
      backend: this.name,
      order,
      vocabulary: [...vocabulary].sort(),
      startDistribution,
      transitions,
      metadata: {
        backend: this.name,
        order,
        sampleCount: dataset.samples.length,
        tokenCount,
        vocabularySize: vocabulary.size,
        rewardWeighted: rewardWeighting,
      },
    };
  }

  predict(model: MovementModel, context: MovementToken[], options: MovementPredictOptions = {}): MovementPrediction[] {
    const distribution = this.backoffDistribution(model, context);
    if (!distribution) {
      return [];
    }
    const total = Object.values(distribution).reduce((sum, weight) => sum + weight, 0);
    if (total <= 0) {
      return [];
    }
    const ranked = Object.entries(distribution)
      .map(([token, weight]) => ({ token, probability: weight / total }))
      // Deterministic ordering: probability desc, then token asc for ties.
      .sort((a, b) => (b.probability - a.probability) || (a.token < b.token ? -1 : a.token > b.token ? 1 : 0));
    return options.topK !== undefined ? ranked.slice(0, Math.max(0, options.topK)) : ranked;
  }

  generate(model: MovementModel, options: MovementGenerateOptions = {}): MovementToken[] {
    const maxSteps = options.maxSteps ?? 256;
    const generated: MovementToken[] = [];
    const context: MovementToken[] = [...(options.seedContext ?? [])];

    for (const seedToken of context) {
      if (seedToken === MOVEMENT_END_TOKEN) {
        break;
      }
      generated.push(seedToken);
    }

    for (let step = 0; step < maxSteps; step += 1) {
      const predictions =
        context.length === 0 ? this.rankDistribution(model.startDistribution) : this.predict(model, context);
      const next = predictions[0]?.token;
      if (next === undefined || next === MOVEMENT_END_TOKEN) {
        break;
      }
      generated.push(next);
      context.push(next);
    }

    return generated;
  }

  /** Stupid-backoff: longest matching context suffix wins; unigram is the floor. */
  private backoffDistribution(
    model: MovementModel,
    context: MovementToken[],
  ): Record<MovementToken, number> | undefined {
    const maxK = Math.min(model.order, context.length);
    for (let k = maxK; k >= 0; k -= 1) {
      const suffix = context.slice(context.length - k);
      const distribution = model.transitions[contextKey(suffix)];
      if (distribution && Object.keys(distribution).length > 0) {
        return distribution;
      }
    }
    return undefined;
  }

  private rankDistribution(distribution: Record<MovementToken, number>): MovementPrediction[] {
    const total = Object.values(distribution).reduce((sum, weight) => sum + weight, 0);
    if (total <= 0) {
      return [];
    }
    return Object.entries(distribution)
      .map(([token, weight]) => ({ token, probability: weight / total }))
      .sort((a, b) => (b.probability - a.probability) || (a.token < b.token ? -1 : a.token > b.token ? 1 : 0));
  }
}

export type MovementActionTokenizer = (action: TrajectoryAction) => MovementToken;

/**
 * Default tokenizer: `tool::summary`. This preserves enough detail for exact
 * replay. Pass a coarser tokenizer (e.g. `(action) => action.tool`) to trade
 * fidelity for stronger generalization across related movements.
 */
export const defaultMovementTokenizer: MovementActionTokenizer = (action) =>
  `${action.tool}::${action.summary}`;

/** Build a training dataset from recorded trajectories. */
export function buildMovementDatasetFromTrajectories(
  trajectories: TrajectorySpan[],
  options: { tokenizer?: MovementActionTokenizer; approvedOnly?: boolean } = {},
): MovementDataset {
  const tokenizer = options.tokenizer ?? defaultMovementTokenizer;
  const approvedOnly = options.approvedOnly ?? false;
  const samples: MovementSample[] = [];

  for (const trajectory of trajectories) {
    if (approvedOnly && trajectory.review?.status !== "approved") {
      continue;
    }
    const actions = [...trajectory.actions].sort((a, b) => a.ts - b.ts);
    if (actions.length === 0) {
      continue;
    }
    samples.push({
      id: trajectory.id,
      sessionId: trajectory.sessionId,
      tokens: actions.map(tokenizer),
      ...(trajectory.outcome?.reward !== undefined ? { reward: trajectory.outcome.reward } : {}),
    });
  }

  return { version: 1, samples };
}

/** Build a training dataset from replay manifests (action events only). */
export function buildMovementDatasetFromReplays(manifests: ReplayManifest[]): MovementDataset {
  const samples: MovementSample[] = [];
  for (const manifest of manifests) {
    const tokens = manifest.events
      .filter((event): event is Extract<typeof event, { kind: "action" }> => event.kind === "action")
      .map((event) => `${event.tool}::${event.summary}`);
    if (tokens.length === 0) {
      continue;
    }
    samples.push({ id: manifest.sessionId, sessionId: manifest.sessionId, tokens });
  }
  return { version: 1, samples };
}

export type MovementEvaluation = {
  sampleCount: number;
  /** Fraction of sequences the greedy policy reproduces exactly (replay). */
  exactReplayRate: number;
  /** Top-1 next-token accuracy across all prediction points (generalization). */
  nextTokenAccuracy: number;
  predictionCount: number;
};

/**
 * Measure how well a trained model reproduces / continues a held-out dataset.
 * Seeds the roadmap's "generalization eval harness": evaluate on held-out but
 * related synthetic trajectories to score replay fidelity.
 */
export function evaluateMovementModel(
  backend: MovementTrainingBackend,
  model: MovementModel,
  dataset: MovementDataset,
): MovementEvaluation {
  let exactReplays = 0;
  let correctPredictions = 0;
  let predictionCount = 0;
  let evaluableSamples = 0;

  for (const sample of dataset.samples) {
    if (sample.tokens.length === 0) {
      continue;
    }
    evaluableSamples += 1;

    const replay = backend.generate(model, { seedContext: sample.tokens.slice(0, 1) });
    if (replay.length === sample.tokens.length && replay.every((token, index) => token === sample.tokens[index])) {
      exactReplays += 1;
    }

    for (let i = 1; i < sample.tokens.length; i += 1) {
      const predictions = backend.predict(model, sample.tokens.slice(0, i), { topK: 1 });
      predictionCount += 1;
      if (predictions[0]?.token === sample.tokens[i]) {
        correctPredictions += 1;
      }
    }
  }

  return {
    sampleCount: evaluableSamples,
    exactReplayRate: evaluableSamples === 0 ? 0 : exactReplays / evaluableSamples,
    nextTokenAccuracy: predictionCount === 0 ? 0 : correctPredictions / predictionCount,
    predictionCount,
  };
}
