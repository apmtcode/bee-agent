import type { TrajectoryAction, TrajectorySpan } from "../capture/trajectory.js";

/**
 * In-cloud trainable movement policy — the "post-train a local model on the
 * recorded dataset to repeat the movements and generalize to new but related
 * ones" half of the local-movement learning subsystem (objective 2c / 2d).
 *
 * The runner (`runner.ts`) emits shell plans for real on-device trainers
 * (mlx / axolotl); that is the heavyweight seam. This module provides the
 * *lightweight, deterministic, dependency-free* seam: a pluggable backend that
 * learns from recorded trajectories and can both reproduce a recorded movement
 * sequence and generalize to related-but-unseen ones — entirely in-process, so
 * it trains and runs in the cloud and in CI without any OS input or GPU.
 *
 * A real on-device small-model backend can be registered later behind the same
 * {@link MovementPolicyBackend} / {@link MovementPolicy} interfaces.
 */

/** Beginning-of-sequence / end-of-sequence sentinels used in token streams. */
export const MOVEMENT_BOS = "<bos>";
export const MOVEMENT_EOS = "<eos>";

const CONTEXT_SEP = "";

/** Turns a recorded action into an abstract, generalization-friendly token. */
export type MovementTokenizer = (action: TrajectoryAction) => string;

/**
 * Default tokenizer. Abstracts away concrete coordinates / ids / names so that
 * "click at (120, 340)" and "click at (88, 12)" collapse to the same movement
 * token — this is what lets the policy generalize across related movements
 * instead of only memorizing exact pixel positions.
 */
export function defaultMovementTokenizer(action: TrajectoryAction): string {
  const summary = action.summary
    .toLowerCase()
    .replace(/\d+/g, "#")
    .replace(/\s+/g, " ")
    .trim();
  return `${action.tool}|${summary}`;
}

/** A single (context -> next token) training example. */
export type MovementExample = {
  trajectoryId: string;
  /** Preceding tokens, most-recent last (length 0..order). */
  context: string[];
  /** Token to predict (an action token or {@link MOVEMENT_EOS}). */
  next: string;
};

/** The full token stream for one trajectory, including BOS/goal and EOS. */
export type MovementSequence = {
  trajectoryId: string;
  tokens: string[];
};

/** A prepared movement dataset: per-trajectory sequences plus sliding examples. */
export type MovementDataset = {
  order: number;
  sequences: MovementSequence[];
  examples: MovementExample[];
};

export type BuildMovementDatasetOptions = {
  /** Max context length the model conditions on. Default 2. */
  order?: number;
  tokenizer?: MovementTokenizer;
  /** Emit an EOS example after the last action so the model learns to stop. */
  includeEos?: boolean;
  /**
   * Optional per-trajectory "goal" token inserted right after BOS. Lets the
   * policy condition its opening move on the task family (e.g. the first
   * observation source or the intended outcome).
   */
  goalToken?: (trajectory: TrajectorySpan) => string | undefined;
};

/**
 * Build a movement dataset from recorded trajectories. Actions are read in
 * timestamp order; observations are ignored for the action-sequence model.
 */
export function buildMovementDataset(
  trajectories: TrajectorySpan[],
  options: BuildMovementDatasetOptions = {},
): MovementDataset {
  const order = Math.max(1, options.order ?? 2);
  const tokenizer = options.tokenizer ?? defaultMovementTokenizer;
  const includeEos = options.includeEos ?? true;

  const sequences: MovementSequence[] = trajectories.map((trajectory) => {
    const tokens: string[] = [MOVEMENT_BOS];
    const goal = options.goalToken?.(trajectory);
    if (goal) {
      tokens.push(goal);
    }
    const orderedActions = [...trajectory.actions].sort((a, b) => a.ts - b.ts);
    for (const action of orderedActions) {
      tokens.push(tokenizer(action));
    }
    if (includeEos) {
      tokens.push(MOVEMENT_EOS);
    }
    return { trajectoryId: trajectory.id, tokens };
  });

  const examples: MovementExample[] = [];
  for (const sequence of sequences) {
    for (let i = 1; i < sequence.tokens.length; i += 1) {
      const context = sequence.tokens.slice(Math.max(0, i - order), i);
      examples.push({ trajectoryId: sequence.trajectoryId, context, next: sequence.tokens[i]! });
    }
  }

  return { order, sequences, examples };
}

/** A single prediction with its backed-off context and full distribution. */
export type MovementPrediction = {
  token: string;
  /** count(token) / total within the matched context. */
  probability: number;
  /** How many trailing context tokens actually matched (backoff order). */
  matchedOrder: number;
  distribution: Array<{ token: string; probability: number }>;
};

export type MovementGenerateOptions = {
  /** Starting context; defaults to `[MOVEMENT_BOS]`. */
  seed?: string[];
  /** Max tokens to emit before stopping. Default 64. */
  maxSteps?: number;
  /** Token that ends generation (excluded from output). Default EOS. */
  stopToken?: string;
};

export type SerializedMovementPolicy = {
  version: 1;
  backendId: string;
  order: number;
  transitions: Array<{ context: string[]; counts: Array<[string, number]> }>;
};

/** A trained, runnable movement policy. */
export interface MovementPolicy {
  readonly backendId: string;
  readonly order: number;
  /** Predict the next movement token given a context window. */
  predict(context: string[]): MovementPrediction | undefined;
  /** Roll out a full movement sequence deterministically (argmax). */
  generate(options?: MovementGenerateOptions): string[];
  serialize(): SerializedMovementPolicy;
}

/** Pluggable trainer. Implement this to add a real on-device model backend. */
export interface MovementPolicyBackend {
  readonly id: string;
  train(dataset: MovementDataset): MovementPolicy;
}

/**
 * Deterministic n-gram movement backend with Katz-style backoff.
 *
 * Training counts next-token frequencies for every context length 0..order.
 * Prediction uses the longest context suffix that was ever observed and backs
 * off to shorter contexts when the full window is unseen — that backoff is what
 * yields generalization: a never-seen full sequence still predicts sensibly
 * from its learned sub-patterns. Argmax with a deterministic tie-break
 * (lexicographic token order) keeps it reproducible for tests.
 */
export class MarkovMovementBackend implements MovementPolicyBackend {
  readonly id = "markov";

  train(dataset: MovementDataset): MovementPolicy {
    const transitions = new Map<string, Map<string, number>>();
    for (const example of dataset.examples) {
      const maxK = Math.min(dataset.order, example.context.length);
      for (let k = 0; k <= maxK; k += 1) {
        const key = example.context.slice(example.context.length - k).join(CONTEXT_SEP);
        let counts = transitions.get(key);
        if (!counts) {
          counts = new Map();
          transitions.set(key, counts);
        }
        counts.set(example.next, (counts.get(example.next) ?? 0) + 1);
      }
    }
    return new MarkovMovementPolicy(this.id, dataset.order, transitions);
  }

  /** Reconstruct a policy from {@link MovementPolicy.serialize}. */
  static load(serialized: SerializedMovementPolicy): MovementPolicy {
    const transitions = new Map<string, Map<string, number>>();
    for (const entry of serialized.transitions) {
      transitions.set(entry.context.join(CONTEXT_SEP), new Map(entry.counts));
    }
    return new MarkovMovementPolicy(serialized.backendId, serialized.order, transitions);
  }
}

class MarkovMovementPolicy implements MovementPolicy {
  constructor(
    readonly backendId: string,
    readonly order: number,
    private readonly transitions: Map<string, Map<string, number>>,
  ) {}

  predict(context: string[]): MovementPrediction | undefined {
    const maxK = Math.min(this.order, context.length);
    for (let k = maxK; k >= 0; k -= 1) {
      const key = context.slice(context.length - k).join(CONTEXT_SEP);
      const counts = this.transitions.get(key);
      if (!counts || counts.size === 0) {
        continue;
      }
      const total = [...counts.values()].reduce((sum, value) => sum + value, 0);
      const distribution = [...counts.entries()]
        .map(([token, count]) => ({ token, probability: count / total }))
        .sort((a, b) => {
          if (b.probability !== a.probability) {
            return b.probability - a.probability;
          }
          return a.token < b.token ? -1 : a.token > b.token ? 1 : 0;
        });
      const best = distribution[0]!;
      return { token: best.token, probability: best.probability, matchedOrder: k, distribution };
    }
    return undefined;
  }

  generate(options: MovementGenerateOptions = {}): string[] {
    const stopToken = options.stopToken ?? MOVEMENT_EOS;
    const maxSteps = options.maxSteps ?? 64;
    const context = [...(options.seed ?? [MOVEMENT_BOS])];
    const output: string[] = [];
    for (let step = 0; step < maxSteps; step += 1) {
      const prediction = this.predict(context);
      if (!prediction || prediction.token === stopToken) {
        break;
      }
      output.push(prediction.token);
      context.push(prediction.token);
    }
    return output;
  }

  serialize(): SerializedMovementPolicy {
    const transitions = [...this.transitions.entries()]
      .map(([key, counts]) => ({
        context: key === "" ? [] : key.split(CONTEXT_SEP),
        counts: [...counts.entries()] as Array<[string, number]>,
      }))
      .sort((a, b) => (a.context.length - b.context.length) || (a.context.join(CONTEXT_SEP) < b.context.join(CONTEXT_SEP) ? -1 : 1));
    return { version: 1, backendId: this.backendId, order: this.order, transitions };
  }
}

const BACKEND_FACTORIES: Record<string, () => MovementPolicyBackend> = {
  markov: () => new MarkovMovementBackend(),
};

/** Registry seam — resolve a pluggable backend by id. */
export function createMovementPolicyBackend(id: string): MovementPolicyBackend {
  const factory = BACKEND_FACTORIES[id];
  if (!factory) {
    throw new Error(`Unknown movement policy backend: ${id}`);
  }
  return factory();
}

export function listMovementPolicyBackends(): string[] {
  return Object.keys(BACKEND_FACTORIES);
}

export type MovementEvalOptions = {
  /**
   * How many leading tokens (BOS/goal/opening moves) to teacher-force before
   * scoring. Default 1 (just the BOS/goal seed). Sequences shorter than this
   * are skipped.
   */
  seedLength?: number;
};

export type MovementEvalResult = {
  trajectoryCount: number;
  evaluatedSteps: number;
  correctSteps: number;
  /** Teacher-forced next-token accuracy across all scored positions. */
  nextTokenAccuracy: number;
  /** Mean free-rollout fidelity: matched-prefix length / continuation length. */
  rolloutFidelity: number;
  perTrajectory: Array<{
    trajectoryId: string;
    steps: number;
    correct: number;
    rolloutMatched: number;
    rolloutTotal: number;
  }>;
};

/**
 * Generalization eval harness. Measures how faithfully a trained policy
 * reproduces held-out (ideally related-but-unseen) movement sequences, via both
 * teacher-forced next-token accuracy and free-running rollout fidelity.
 */
export function evaluateMovementPolicy(
  policy: MovementPolicy,
  sequences: MovementSequence[],
  options: MovementEvalOptions = {},
): MovementEvalResult {
  const seedLength = Math.max(1, options.seedLength ?? 1);
  const perTrajectory: MovementEvalResult["perTrajectory"] = [];
  let evaluatedSteps = 0;
  let correctSteps = 0;
  let fidelitySum = 0;
  let scoredSequences = 0;

  for (const sequence of sequences) {
    const tokens = sequence.tokens;
    if (tokens.length <= seedLength) {
      continue;
    }
    scoredSequences += 1;

    let correct = 0;
    let steps = 0;
    for (let i = seedLength; i < tokens.length; i += 1) {
      const prediction = policy.predict(tokens.slice(0, i));
      steps += 1;
      evaluatedSteps += 1;
      if (prediction && prediction.token === tokens[i]) {
        correct += 1;
        correctSteps += 1;
      }
    }

    const truth = tokens.slice(seedLength).filter((token) => token !== MOVEMENT_EOS);
    const rollout = policy.generate({ seed: tokens.slice(0, seedLength), maxSteps: truth.length });
    let matched = 0;
    for (let i = 0; i < truth.length && i < rollout.length; i += 1) {
      if (rollout[i] === truth[i]) {
        matched += 1;
      } else {
        break;
      }
    }
    const rolloutTotal = truth.length;
    fidelitySum += rolloutTotal === 0 ? 1 : matched / rolloutTotal;

    perTrajectory.push({
      trajectoryId: sequence.trajectoryId,
      steps,
      correct,
      rolloutMatched: matched,
      rolloutTotal,
    });
  }

  return {
    trajectoryCount: scoredSequences,
    evaluatedSteps,
    correctSteps,
    nextTokenAccuracy: evaluatedSteps === 0 ? 0 : correctSteps / evaluatedSteps,
    rolloutFidelity: scoredSequences === 0 ? 0 : fidelitySum / scoredSequences,
    perTrajectory,
  };
}
