import type { MovementDataset } from "./movement-event.js";

/**
 * Pluggable local-model backend seam for the movement learning subsystem.
 *
 * The objective is on-device post-training of a small model that can repeat
 * recorded movements and generalize to new-but-related ones. Because the engine
 * runs in the cloud with no access to a real machine or GPU, training must be
 * expressible behind an interface that a deterministic, dependency-free backend
 * can satisfy for tests/CI, while leaving a documented seam for a real backend
 * (e.g. an MLX/llama.cpp adapter) to drop in unchanged.
 *
 * A backend takes a {@link MovementDataset} and returns a {@link TrainedMovementModel}
 * that can score the next token given a context and autoregressively generate
 * new token sequences. Determinism is a hard requirement here: no `Date.now()`
 * or `Math.random()` — callers pass a seeded RNG so runs are reproducible.
 */
export type MovementTokenProbability = {
  token: string;
  probability: number;
};

export type MovementTokenDistribution = MovementTokenProbability[];

export type MovementModelMetadata = {
  backend: string;
  /** N-gram order / context window the model was trained with. */
  order: number;
  vocabularySize: number;
  sequenceCount: number;
  tokenCount: number;
};

export type MovementGenerateParams = {
  /** Prior context tokens to condition generation on (may be empty). */
  seed?: string[];
  /** Maximum number of tokens to emit (excludes the terminal sentinel). */
  maxLength: number;
  /** Deterministic RNG in [0, 1). Omit for greedy (argmax) decoding. */
  rng?: () => number;
};

export interface TrainedMovementModel {
  readonly metadata: MovementModelMetadata;
  readonly vocabulary: string[];
  /** Probability distribution over the next token given a context. */
  predictNext(context: string[]): MovementTokenDistribution;
  /** Autoregressively generate a token sequence (sentinels stripped). */
  generate(params: MovementGenerateParams): string[];
}

export type MovementTrainOptions = {
  /** Context length (n-gram order). Defaults are backend-specific. */
  order?: number;
  /** Additive (Laplace) smoothing mass. */
  smoothing?: number;
};

export interface MovementModelBackend {
  readonly name: string;
  train(dataset: MovementDataset, options?: MovementTrainOptions): Promise<TrainedMovementModel>;
}

/**
 * Deterministic, seedable PRNG (mulberry32). Used everywhere the subsystem needs
 * randomness so cloud/CI runs are reproducible — the codebase forbids
 * `Math.random()`/`Date.now()` in this subsystem for exactly this reason.
 */
export function createSeededRng(seed: number): () => number {
  let state = seed >>> 0;
  return function next(): number {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Sample one token from a distribution using a seeded RNG. The distribution need
 * not be normalized; probabilities are treated as weights.
 */
export function sampleFromDistribution(
  distribution: MovementTokenDistribution,
  rng: () => number,
): string | undefined {
  const total = distribution.reduce((sum, entry) => sum + Math.max(0, entry.probability), 0);
  if (total <= 0) {
    return distribution[0]?.token;
  }
  let threshold = rng() * total;
  for (const entry of distribution) {
    threshold -= Math.max(0, entry.probability);
    if (threshold < 0) {
      return entry.token;
    }
  }
  return distribution[distribution.length - 1]?.token;
}

/** Deterministic argmax over a distribution, tie-broken by token order. */
export function argmaxToken(distribution: MovementTokenDistribution): string | undefined {
  let best: MovementTokenProbability | undefined;
  for (const entry of distribution) {
    if (!best || entry.probability > best.probability) {
      best = entry;
    }
  }
  return best?.token;
}
