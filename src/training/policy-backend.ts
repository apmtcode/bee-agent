import type { ReplayManifest, ReplayTimelineEvent } from "../capture/replay.js";

/**
 * Pluggable local-movement policy backend.
 *
 * This is the on-device "brain" of the local-movement learning subsystem
 * (standing objective #2c/#2d): it post-trains a small local model on a
 * dataset of recorded movement sequences so bee-agent can (a) *repeat* the
 * recorded movements deterministically and (b) *generalize* to new-but-related
 * movements by sampling from what it learned.
 *
 * The backend is intentionally an interface so the real on-device runtime
 * (MLX / a small local transformer, wired by `runner.ts`) can be swapped in
 * without touching call sites. The default {@link MarkovMovementBackend} is a
 * dependency-free, fully deterministic n-gram model that runs anywhere —
 * including the cloud/CI, where there is no GPU and no real OS input — so the
 * whole pipeline (dataset → train → infer → eval) can be validated with
 * synthetic event streams.
 */

/** A single normalized movement primitive, e.g. `"mouse.move"`, `"key.press"`. */
export type MovementToken = string;

/** Marks the beginning of a recorded movement sequence. */
export const MOVEMENT_BOS: MovementToken = "bos";
/** Marks the end of a recorded movement sequence. */
export const MOVEMENT_EOS: MovementToken = "eos";

/** One recorded, replayable movement sequence extracted from a trajectory. */
export type MovementSequence = {
  trajectoryId: string;
  tokens: MovementToken[];
};

/** A training dataset: the movement sequences a policy learns from. */
export type MovementDataset = {
  version: 1;
  sequences: MovementSequence[];
};

export type MovementGenerateParams = {
  /** Prefix of tokens to continue from (a partial recording, or `[]` to start fresh). */
  prompt?: MovementToken[];
  /** Hard cap on generated tokens (excludes the prompt). */
  maxSteps: number;
  /**
   * 0 → greedy/argmax decoding (deterministic; reproduces the dominant
   * recorded movement). >0 → seeded sampling that generalizes to related
   * sequences. Never uses `Math.random()` — always seeded for reproducibility.
   */
  temperature?: number;
  /** Seed for sampling. Same seed + same model + same params → same output. */
  seed?: number;
};

/** A trained, serializable movement policy ready for inference and eval. */
export interface TrainedMovementPolicy {
  readonly backend: string;
  readonly order: number;
  /** Continue a movement sequence. Stops at {@link MOVEMENT_EOS} or `maxSteps`. */
  generate(params: MovementGenerateParams): MovementToken[];
  /** Mean per-token log-probability the policy assigns to a full sequence. */
  scoreSequence(tokens: MovementToken[]): number;
  /** Serialize the learned parameters (for persisting a trained model). */
  toJSON(): SerializedMovementPolicy;
}

/** The backend that turns a dataset into a {@link TrainedMovementPolicy}. */
export interface MovementPolicyBackend {
  readonly name: string;
  train(dataset: MovementDataset): TrainedMovementPolicy;
  /** Rehydrate a previously-trained policy from its serialized form. */
  load(serialized: SerializedMovementPolicy): TrainedMovementPolicy;
}

export type SerializedMovementPolicy = {
  backend: string;
  version: 1;
  order: number;
  /** context-key → { nextToken → count }. */
  transitions: Record<string, Record<MovementToken, number>>;
  vocabulary: MovementToken[];
};

const CONTEXT_SEPARATOR = "";

/**
 * Extract the ordered movement vocabulary from a replay manifest. Only
 * `action` events are movements (observations/transcript are context, not
 * things the body does). The token is the tool primitive, which is exactly
 * the unit a movement policy should learn to sequence.
 */
export function extractMovementTokens(events: ReplayTimelineEvent[]): MovementToken[] {
  return events
    .filter((event): event is Extract<ReplayTimelineEvent, { kind: "action" }> => event.kind === "action")
    .map((event) => event.tool);
}

/** Build a training dataset from replay manifests (one sequence per manifest). */
export function buildMovementDataset(replays: ReplayManifest[]): MovementDataset {
  const sequences: MovementSequence[] = [];
  for (const replay of replays) {
    const tokens = extractMovementTokens(replay.events);
    if (tokens.length === 0) {
      continue;
    }
    sequences.push({
      trajectoryId: replay.trajectoryIds[0] ?? replay.sessionId,
      tokens,
    });
  }
  return { version: 1, sequences };
}

/**
 * Deterministic n-gram (Markov) movement backend. The default, zero-dependency
 * policy: learns P(next | last `order` tokens) from recorded sequences with
 * add-k smoothing. Greedy decoding reproduces recordings; seeded sampling
 * generalizes. Serves as the CI-safe stand-in for a real on-device model.
 */
export class MarkovMovementBackend implements MovementPolicyBackend {
  readonly name = "markov";

  constructor(
    private readonly order = 2,
    /** Add-k Laplace smoothing constant for unseen transitions. */
    private readonly smoothing = 0.01,
  ) {
    if (order < 1) {
      throw new Error("Markov order must be >= 1");
    }
  }

  train(dataset: MovementDataset): TrainedMovementPolicy {
    const transitions: Record<string, Record<MovementToken, number>> = {};
    const vocabulary = new Set<MovementToken>([MOVEMENT_EOS]);

    for (const sequence of dataset.sequences) {
      const padded = [
        ...Array.from({ length: this.order }, () => MOVEMENT_BOS),
        ...sequence.tokens,
        MOVEMENT_EOS,
      ];
      for (const token of sequence.tokens) {
        vocabulary.add(token);
      }
      for (let i = this.order; i < padded.length; i += 1) {
        const context = padded.slice(i - this.order, i);
        const next = padded[i]!;
        const key = contextKey(context);
        const row = (transitions[key] ??= {});
        row[next] = (row[next] ?? 0) + 1;
      }
    }

    return new MarkovMovementPolicy(this.order, this.smoothing, transitions, [...vocabulary]);
  }

  load(serialized: SerializedMovementPolicy): TrainedMovementPolicy {
    return new MarkovMovementPolicy(
      serialized.order,
      this.smoothing,
      serialized.transitions,
      serialized.vocabulary,
    );
  }
}

class MarkovMovementPolicy implements TrainedMovementPolicy {
  readonly backend = "markov";

  constructor(
    readonly order: number,
    private readonly smoothing: number,
    private readonly transitions: Record<string, Record<MovementToken, number>>,
    private readonly vocabulary: MovementToken[],
  ) {}

  generate(params: MovementGenerateParams): MovementToken[] {
    const temperature = params.temperature ?? 0;
    const rng = createSeededRng(params.seed ?? 0);
    const history = [
      ...Array.from({ length: this.order }, () => MOVEMENT_BOS),
      ...(params.prompt ?? []),
    ];
    const generated: MovementToken[] = [];

    for (let step = 0; step < params.maxSteps; step += 1) {
      const context = history.slice(history.length - this.order);
      const distribution = this.distributionFor(context);
      if (distribution.length === 0) {
        break;
      }
      const next = temperature <= 0
        ? pickArgmax(distribution)
        : sampleFromDistribution(distribution, temperature, rng);
      if (next === MOVEMENT_EOS) {
        break;
      }
      generated.push(next);
      history.push(next);
    }

    return generated;
  }

  scoreSequence(tokens: MovementToken[]): number {
    const padded = [
      ...Array.from({ length: this.order }, () => MOVEMENT_BOS),
      ...tokens,
      MOVEMENT_EOS,
    ];
    let logProbTotal = 0;
    let count = 0;
    for (let i = this.order; i < padded.length; i += 1) {
      const context = padded.slice(i - this.order, i);
      const next = padded[i]!;
      logProbTotal += Math.log(this.probabilityOf(context, next));
      count += 1;
    }
    return count === 0 ? 0 : logProbTotal / count;
  }

  toJSON(): SerializedMovementPolicy {
    return {
      backend: this.backend,
      version: 1,
      order: this.order,
      transitions: this.transitions,
      vocabulary: [...this.vocabulary],
    };
  }

  /** Smoothed probability of `next` given `context`, over the full vocabulary. */
  private probabilityOf(context: MovementToken[], next: MovementToken): number {
    const row = this.transitions[contextKey(context)] ?? {};
    const observed = row[next] ?? 0;
    const total = Object.values(row).reduce((sum, value) => sum + value, 0);
    const denominator = total + this.smoothing * this.vocabulary.length;
    if (denominator === 0) {
      return 1 / Math.max(this.vocabulary.length, 1);
    }
    return (observed + this.smoothing) / denominator;
  }

  /** Sorted (token, probability) list for a context, restricted to seen next tokens. */
  private distributionFor(context: MovementToken[]): Array<{ token: MovementToken; probability: number }> {
    const row = this.transitions[contextKey(context)];
    if (!row) {
      return [];
    }
    const total = Object.values(row).reduce((sum, value) => sum + value, 0);
    return Object.entries(row)
      .map(([token, count]) => ({ token, probability: total === 0 ? 0 : count / total }))
      .sort((a, b) => (b.probability - a.probability) || compareToken(a.token, b.token));
  }
}

function contextKey(context: MovementToken[]): string {
  return context.join(CONTEXT_SEPARATOR);
}

function compareToken(a: MovementToken, b: MovementToken): number {
  return a < b ? -1 : a > b ? 1 : 0;
}

function pickArgmax(distribution: Array<{ token: MovementToken; probability: number }>): MovementToken {
  // distribution is pre-sorted by probability desc then token asc, so [0] is a
  // stable argmax (deterministic tie-break).
  return distribution[0]!.token;
}

function sampleFromDistribution(
  distribution: Array<{ token: MovementToken; probability: number }>,
  temperature: number,
  rng: () => number,
): MovementToken {
  const weights = distribution.map((entry) => Math.pow(Math.max(entry.probability, 1e-9), 1 / temperature));
  const total = weights.reduce((sum, weight) => sum + weight, 0);
  let target = rng() * total;
  for (let i = 0; i < distribution.length; i += 1) {
    target -= weights[i]!;
    if (target <= 0) {
      return distribution[i]!.token;
    }
  }
  return distribution[distribution.length - 1]!.token;
}

/** Small, deterministic LCG so sampling never depends on `Math.random()`. */
export function createSeededRng(seed: number): () => number {
  let state = (Math.floor(seed) % 2147483647 + 2147483647) % 2147483647;
  if (state === 0) {
    state = 1;
  }
  return () => {
    state = (state * 48271) % 2147483647;
    return (state - 1) / 2147483646;
  };
}

export type MovementPolicyEvaluation = {
  sequenceCount: number;
  tokenCount: number;
  /** Fraction of next-token predictions where greedy argmax matched the held-out truth. */
  nextTokenAccuracy: number;
  /** Mean per-token log-probability the policy assigns to the held-out sequences (higher = better fit). */
  meanLogProbability: number;
  /** Fraction of held-out sequences the policy reproduces exactly via greedy decode from BOS. */
  exactReplayRate: number;
};

/**
 * Generalization eval harness (roadmap: "measure replay fidelity on held-out
 * but related synthetic trajectories"). Runs greedy next-token prediction and
 * whole-sequence replay against sequences the model did not necessarily train
 * on, and reports fidelity metrics.
 */
export function evaluateMovementPolicy(
  policy: TrainedMovementPolicy,
  heldOut: MovementDataset,
): MovementPolicyEvaluation {
  let correct = 0;
  let tokenCount = 0;
  let logProbTotal = 0;
  let exactReplays = 0;

  for (const sequence of heldOut.sequences) {
    for (let i = 0; i < sequence.tokens.length; i += 1) {
      const prefix = sequence.tokens.slice(0, i);
      const predicted = policy.generate({ prompt: prefix, maxSteps: 1, temperature: 0 });
      if (predicted[0] === sequence.tokens[i]) {
        correct += 1;
      }
      tokenCount += 1;
    }
    logProbTotal += policy.scoreSequence(sequence.tokens);

    const replayed = policy.generate({ prompt: [], maxSteps: sequence.tokens.length + 4, temperature: 0 });
    if (arraysEqual(replayed, sequence.tokens)) {
      exactReplays += 1;
    }
  }

  const sequenceCount = heldOut.sequences.length;
  return {
    sequenceCount,
    tokenCount,
    nextTokenAccuracy: tokenCount === 0 ? 0 : correct / tokenCount,
    meanLogProbability: sequenceCount === 0 ? 0 : logProbTotal / sequenceCount,
    exactReplayRate: sequenceCount === 0 ? 0 : exactReplays / sequenceCount,
  };
}

function arraysEqual(a: MovementToken[], b: MovementToken[]): boolean {
  if (a.length !== b.length) {
    return false;
  }
  return a.every((value, index) => value === b[index]);
}
