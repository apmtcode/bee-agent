import type { TrajectoryAction, TrajectorySpan } from "../capture/trajectory.js";

/**
 * In-process, pluggable local-movement model pipeline.
 *
 * This is the cloud-safe counterpart to {@link LocalAppleSiliconTrainingRunner},
 * which only emits shell launch plans for a real on-device runtime (MLX /
 * Axolotl). Those plans cannot be executed in Anthropic's cloud, so the
 * self-training + inference loop (objective #2 parts c and d — post-train a
 * model to *repeat* recorded movements and *generalize* to new-but-related
 * ones) could not previously be validated here.
 *
 * This module provides:
 *  - a canonical *movement token* schema derived from recorded trajectory
 *    actions (tool + gesture + direction/target bucket),
 *  - a {@link MovementDataset} format (sequences of tokens, one per trajectory),
 *  - a {@link MovementModelBackend} interface so the model backend is pluggable
 *    (swap the deterministic mock for a real on-device small model later),
 *  - {@link NgramMovementBackend}: a deterministic, dependency-free n-gram
 *    policy with Katz-style backoff. Exact recorded prefixes replay verbatim;
 *    unseen-but-related prefixes still yield a sensible next move by backing off
 *    to lower-order statistics — the generalization the objective asks for,
 *  - a {@link MovementModelRegistry} for registering/looking up backends.
 *
 * Everything is deterministic (no randomness, no clock) so it is fully testable
 * in CI with synthetic event streams.
 */

/** Canonical, replayable representation of a single recorded movement. */
export type MovementToken = string;

/** Sentinel marking the start of a sequence (used as backoff context). */
export const MOVEMENT_START_TOKEN: MovementToken = "<start>";
/** Sentinel marking the end of a sequence (a rollout stops when it emits this). */
export const MOVEMENT_END_TOKEN: MovementToken = "<end>";

/** One recorded movement sequence, tokenized from a trajectory's actions. */
export type MovementSequence = {
  trajectoryId: string;
  tokens: MovementToken[];
};

/** A tokenized, replayable training corpus. */
export type MovementDataset = {
  version: 1;
  sequences: MovementSequence[];
  /** Sorted, de-duplicated vocabulary (excludes start/end sentinels). */
  vocabulary: MovementToken[];
};

export type MovementModelKind = "ngram";

/** A trained model artifact. Serializable so it can be persisted/inspected. */
export type MovementModel = {
  kind: MovementModelKind;
  /** Highest context order the model was trained at (n-1 previous tokens). */
  order: number;
  vocabulary: MovementToken[];
  /**
   * Transition counts keyed by context. The context key is the last up-to-order
   * tokens joined with the unit separator. Each context maps token -> count.
   */
  transitions: Record<string, Record<MovementToken, number>>;
  sequenceCount: number;
  tokenCount: number;
};

/** A single next-move prediction. */
export type MovementPrediction = {
  token: MovementToken;
  probability: number;
  /** Context order actually used after backoff (0 = unigram prior). */
  backoffOrder: number;
  /** Ranked alternatives (includes the chosen token first). */
  candidates: Array<{ token: MovementToken; probability: number }>;
};

export type MovementTrainOptions = {
  /** Max context order (number of previous tokens conditioned on). Default 2. */
  order?: number;
};

export type MovementRolloutOptions = {
  /** Hard cap on generated tokens (excluding the end sentinel). Default 64. */
  maxTokens?: number;
  /** Seed context; defaults to `[MOVEMENT_START_TOKEN]`. */
  context?: MovementToken[];
};

/**
 * Pluggable local-model backend. The deterministic mock ({@link NgramMovementBackend})
 * satisfies this so cloud/CI tests pass; a real on-device small-model backend
 * can implement the same seam later.
 */
export interface MovementModelBackend {
  readonly name: string;
  train(dataset: MovementDataset, options?: MovementTrainOptions): MovementModel;
  /** Predict the next movement given a context window. */
  predict(model: MovementModel, context: MovementToken[]): MovementPrediction;
}

const UNIT_SEPARATOR = "␟";

/**
 * Derive a canonical movement token from a recorded trajectory action. Prefers
 * the structured gesture metadata written by {@link DeviceCaptureAdapter}
 * (gesture kind + direction/target), falling back to the human summary so
 * non-device actions still tokenize deterministically.
 */
export function tokenizeAction(action: TrajectoryAction): MovementToken {
  const metadata = action.metadata ?? {};
  const gesture = typeof metadata.gesture === "string" ? metadata.gesture : undefined;
  const direction = typeof metadata.direction === "string" ? metadata.direction : undefined;
  const target = typeof metadata.target === "string" ? metadata.target : undefined;
  const parts = [action.tool];
  if (gesture) {
    parts.push(gesture);
    if (direction) {
      parts.push(direction);
    } else if (target) {
      parts.push(bucketTarget(target));
    }
  } else {
    parts.push(slug(action.summary));
  }
  return parts.join(":");
}

/**
 * Collapse a free-form UI target into a stable low-cardinality bucket so
 * related-but-distinct targets (e.g. "row-3" vs "row-7") map to the same token,
 * which is what lets the model generalize across similar movements.
 */
function bucketTarget(target: string): string {
  const normalized = slug(target);
  const withoutIndex = normalized.replace(/-?\d+$/u, "");
  return withoutIndex.length > 0 ? withoutIndex : normalized;
}

function slug(value: string): string {
  return value
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/gu, "-")
    .replace(/^-+|-+$/gu, "")
    .slice(0, 48) || "action";
}

/** Build a tokenized {@link MovementDataset} from recorded trajectory spans. */
export function buildMovementDataset(trajectories: TrajectorySpan[]): MovementDataset {
  const sequences: MovementSequence[] = [];
  const vocabulary = new Set<MovementToken>();
  for (const trajectory of trajectories) {
    const actions = [...trajectory.actions].sort((a, b) => a.ts - b.ts);
    if (actions.length === 0) {
      continue;
    }
    const tokens = actions.map((action) => {
      const token = tokenizeAction(action);
      vocabulary.add(token);
      return token;
    });
    sequences.push({ trajectoryId: trajectory.id, tokens });
  }
  return {
    version: 1,
    sequences,
    vocabulary: [...vocabulary].sort(),
  };
}

/**
 * Deterministic n-gram movement backend with Katz-style backoff.
 *
 * Training counts, for every context of length 0..order, how often each token
 * follows it. Prediction tries the longest available context and backs off to
 * shorter contexts until it finds one that was observed, ending at the unigram
 * prior. Ties break by token order so predictions are fully deterministic.
 */
export class NgramMovementBackend implements MovementModelBackend {
  readonly name = "ngram-mock";

  train(dataset: MovementDataset, options?: MovementTrainOptions): MovementModel {
    const order = Math.max(1, Math.floor(options?.order ?? 2));
    const transitions: Record<string, Record<MovementToken, number>> = {};
    let tokenCount = 0;
    for (const sequence of dataset.sequences) {
      const padded = [MOVEMENT_START_TOKEN, ...sequence.tokens, MOVEMENT_END_TOKEN];
      for (let index = 1; index < padded.length; index += 1) {
        const next = padded[index]!;
        tokenCount += 1;
        // Record this transition at every context order 0..order.
        for (let contextLength = 0; contextLength <= order; contextLength += 1) {
          const start = index - contextLength;
          if (start < 0) {
            continue;
          }
          const contextKey = padded.slice(start, index).join(UNIT_SEPARATOR);
          const bucket = (transitions[contextKey] ??= {});
          bucket[next] = (bucket[next] ?? 0) + 1;
        }
      }
    }
    return {
      kind: "ngram",
      order,
      vocabulary: [...dataset.vocabulary],
      transitions,
      sequenceCount: dataset.sequences.length,
      tokenCount,
    };
  }

  predict(model: MovementModel, context: MovementToken[]): MovementPrediction {
    const window = context.length > 0 ? context : [MOVEMENT_START_TOKEN];
    for (let contextLength = Math.min(model.order, window.length); contextLength >= 0; contextLength -= 1) {
      const slice = window.slice(window.length - contextLength);
      const contextKey = slice.join(UNIT_SEPARATOR);
      const bucket = model.transitions[contextKey];
      if (bucket && Object.keys(bucket).length > 0) {
        return rankBucket(bucket, contextLength);
      }
      if (contextLength === 0) {
        break;
      }
    }
    // Empty model: nothing to predict; deterministically emit the end sentinel.
    return {
      token: MOVEMENT_END_TOKEN,
      probability: 1,
      backoffOrder: 0,
      candidates: [{ token: MOVEMENT_END_TOKEN, probability: 1 }],
    };
  }
}

function rankBucket(bucket: Record<MovementToken, number>, backoffOrder: number): MovementPrediction {
  const total = Object.values(bucket).reduce((sum, count) => sum + count, 0);
  const candidates = Object.entries(bucket)
    .map(([token, count]) => ({ token, probability: count / total }))
    .sort((a, b) => {
      if (b.probability !== a.probability) {
        return b.probability - a.probability;
      }
      return a.token < b.token ? -1 : a.token > b.token ? 1 : 0;
    });
  const chosen = candidates[0]!;
  return {
    token: chosen.token,
    probability: chosen.probability,
    backoffOrder,
    candidates,
  };
}

/**
 * Deterministically roll out a full movement sequence from the trained model —
 * this is "repeat the recorded movements" when seeded from `<start>`, and
 * "perform a new-but-related movement" when seeded from a novel prefix (backoff
 * fills the gaps). Stops at the end sentinel or `maxTokens`.
 */
export function rolloutMovements(
  backend: MovementModelBackend,
  model: MovementModel,
  options?: MovementRolloutOptions,
): MovementToken[] {
  const maxTokens = Math.max(1, Math.floor(options?.maxTokens ?? 64));
  const context = options?.context ? [...options.context] : [MOVEMENT_START_TOKEN];
  const generated: MovementToken[] = [];
  for (let step = 0; step < maxTokens; step += 1) {
    const prediction = backend.predict(model, context);
    if (prediction.token === MOVEMENT_END_TOKEN) {
      break;
    }
    generated.push(prediction.token);
    context.push(prediction.token);
  }
  return generated;
}

/**
 * Fidelity of a rolled-out sequence against a reference recording — the
 * generalization / replay eval metric. Returns the fraction of reference tokens
 * reproduced in order (longest-common-subsequence ratio), in [0, 1].
 */
export function replayFidelity(reference: MovementToken[], produced: MovementToken[]): number {
  if (reference.length === 0) {
    return produced.length === 0 ? 1 : 0;
  }
  const lcs = longestCommonSubsequence(reference, produced);
  return lcs / reference.length;
}

function longestCommonSubsequence(a: MovementToken[], b: MovementToken[]): number {
  const rows = a.length + 1;
  const cols = b.length + 1;
  const table = new Array<number>(rows * cols).fill(0);
  for (let i = 1; i < rows; i += 1) {
    for (let j = 1; j < cols; j += 1) {
      if (a[i - 1] === b[j - 1]) {
        table[i * cols + j] = table[(i - 1) * cols + (j - 1)]! + 1;
      } else {
        table[i * cols + j] = Math.max(table[(i - 1) * cols + j]!, table[i * cols + (j - 1)]!);
      }
    }
  }
  return table[rows * cols - 1]!;
}

/** Registry of pluggable movement-model backends (keyed by backend name). */
export class MovementModelRegistry {
  private readonly backends = new Map<string, MovementModelBackend>();

  register(backend: MovementModelBackend): this {
    this.backends.set(backend.name, backend);
    return this;
  }

  get(name: string): MovementModelBackend | undefined {
    return this.backends.get(name);
  }

  require(name: string): MovementModelBackend {
    const backend = this.backends.get(name);
    if (!backend) {
      throw new Error(`unknown movement model backend: ${name}`);
    }
    return backend;
  }

  list(): string[] {
    return [...this.backends.keys()].sort();
  }
}

/** A registry pre-seeded with the built-in deterministic mock backend. */
export function createDefaultMovementModelRegistry(): MovementModelRegistry {
  return new MovementModelRegistry().register(new NgramMovementBackend());
}
