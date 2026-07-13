/**
 * In-process movement-model backend.
 *
 * The {@link LocalAppleSiliconTrainingRunner} emits launch scripts that fine-tune
 * a real model on-device (mlx / axolotl). Those runtimes only exist on the
 * operator's Apple-silicon machine, so nothing in the cloud/CI can exercise the
 * "post-train a local model to repeat the recorded movements" objective.
 *
 * This module fills that gap with a *pluggable* backend seam plus a deterministic,
 * dependency-free reference backend (a variable-order Markov policy with suffix
 * backoff). It learns from recorded movement sequences, predicts the next
 * movement, generalizes to new-but-related sequences via backoff, and serializes
 * to a plain JSON artifact — so the whole capture -> dataset -> train -> infer
 * loop can be validated without any real OS input or GPU.
 *
 * Real on-device backends implement the same {@link MovementModelBackend}
 * interface and register with {@link MovementModelBackendRegistry}.
 */

import type { ExportedReplayManifest } from "./export-manifest.js";
import type { TrajectorySpan } from "../capture/trajectory.js";

/** A single movement token — typically an action's `tool`, but any symbol works. */
export type MovementToken = string;

/** Sentinel prepended to every training sequence (models the "first move"). */
export const MOVEMENT_START = "start";
/** Sentinel appended to every training sequence (models "stop here"). */
export const MOVEMENT_END = "end";

/** One recorded movement sequence: an ordered run of movement tokens. */
export type MovementSequence = {
  id: string;
  tokens: MovementToken[];
};

/** A replayable movement dataset — the training input for a backend. */
export type MovementDataset = {
  version: 1;
  sequences: MovementSequence[];
};

/** A ranked next-movement prediction. */
export type MovementPrediction = {
  token: MovementToken;
  probability: number;
};

/** Serialized model — plain JSON, safe to persist and reload. */
export type MovementModelArtifact = {
  version: 1;
  backend: string;
  order: number;
  vocabulary: MovementToken[];
  /** contextKey -> (token -> count). "" is the unigram (order-0) context. */
  transitions: Record<string, Record<MovementToken, number>>;
  sequenceCount: number;
  tokenCount: number;
};

/** A trained movement model — deterministic inference over prior movements. */
export interface MovementModel {
  readonly backend: string;
  readonly order: number;
  /**
   * Rank likely next movements given the prior movement context. Uses the
   * longest matching suffix of `context` and backs off to shorter contexts,
   * which is what lets the model generalize to unseen full sequences that share
   * local structure with the training data. Ranked high→low; ties broken by
   * token order so results are deterministic.
   */
  predictNext(context: MovementToken[]): MovementPrediction[];
  serialize(): MovementModelArtifact;
}

export type MovementTrainingOptions = {
  /** Maximum Markov order (context length). Defaults to 3. */
  order?: number;
};

/** A pluggable movement-model backend (mock, mlx-bridge, onnx, …). */
export interface MovementModelBackend {
  readonly name: string;
  train(dataset: MovementDataset, options?: MovementTrainingOptions): MovementModel;
  load(artifact: MovementModelArtifact): MovementModel;
}

const CONTEXT_DELIMITER = "";

function contextKey(suffix: MovementToken[]): string {
  return suffix.join(CONTEXT_DELIMITER);
}

function rankDistribution(distribution: Record<MovementToken, number>): MovementPrediction[] {
  const total = Object.values(distribution).reduce((sum, count) => sum + count, 0);
  if (total === 0) {
    return [];
  }
  return Object.entries(distribution)
    .map(([token, count]) => ({ token, probability: count / total }))
    .sort((a, b) => {
      if (b.probability !== a.probability) {
        return b.probability - a.probability;
      }
      return a.token < b.token ? -1 : a.token > b.token ? 1 : 0;
    });
}

class MarkovMovementModel implements MovementModel {
  readonly backend = "markov";

  constructor(
    readonly order: number,
    private readonly transitions: Record<string, Record<MovementToken, number>>,
    private readonly vocabulary: MovementToken[],
    private readonly sequenceCount: number,
    private readonly tokenCount: number,
  ) {}

  predictNext(context: MovementToken[]): MovementPrediction[] {
    // Prepend the start sentinel so an empty context predicts from the learned
    // first-move distribution P(token | start) rather than the raw unigram.
    const padded = [MOVEMENT_START, ...context];
    const maxSuffix = Math.min(this.order, padded.length);
    for (let length = maxSuffix; length >= 0; length -= 1) {
      const suffix = padded.slice(padded.length - length);
      const distribution = this.transitions[contextKey(suffix)];
      if (distribution && Object.keys(distribution).length > 0) {
        return rankDistribution(distribution);
      }
    }
    return [];
  }

  serialize(): MovementModelArtifact {
    return {
      version: 1,
      backend: this.backend,
      order: this.order,
      vocabulary: [...this.vocabulary],
      transitions: cloneTransitions(this.transitions),
      sequenceCount: this.sequenceCount,
      tokenCount: this.tokenCount,
    };
  }
}

function cloneTransitions(
  transitions: Record<string, Record<MovementToken, number>>,
): Record<string, Record<MovementToken, number>> {
  const clone: Record<string, Record<MovementToken, number>> = {};
  for (const [key, distribution] of Object.entries(transitions)) {
    clone[key] = { ...distribution };
  }
  return clone;
}

/**
 * Deterministic reference backend: a variable-order Markov chain with suffix
 * backoff. No native deps, no randomness — same dataset always yields the same
 * model, so it is a stable stand-in for a real on-device model in tests.
 */
export class MarkovMovementBackend implements MovementModelBackend {
  readonly name = "markov";

  train(dataset: MovementDataset, options: MovementTrainingOptions = {}): MovementModel {
    const order = Math.max(0, Math.floor(options.order ?? 3));
    const transitions: Record<string, Record<MovementToken, number>> = {};
    const vocabulary = new Set<MovementToken>();
    let tokenCount = 0;

    for (const sequence of dataset.sequences) {
      const stream = [MOVEMENT_START, ...sequence.tokens, MOVEMENT_END];
      for (const token of sequence.tokens) {
        vocabulary.add(token);
      }
      tokenCount += sequence.tokens.length;
      for (let index = 1; index < stream.length; index += 1) {
        const next = stream[index];
        const maxSuffix = Math.min(order, index);
        for (let length = 0; length <= maxSuffix; length += 1) {
          const suffix = stream.slice(index - length, index);
          const key = contextKey(suffix);
          const distribution = (transitions[key] ??= {});
          distribution[next] = (distribution[next] ?? 0) + 1;
        }
      }
    }

    return new MarkovMovementModel(
      order,
      transitions,
      [...vocabulary].sort(),
      dataset.sequences.length,
      tokenCount,
    );
  }

  load(artifact: MovementModelArtifact): MovementModel {
    return new MarkovMovementModel(
      artifact.order,
      cloneTransitions(artifact.transitions),
      [...artifact.vocabulary],
      artifact.sequenceCount,
      artifact.tokenCount,
    );
  }
}

/**
 * Registry of pluggable backends. Ships with the deterministic {@link
 * MarkovMovementBackend} registered as the default; on-device backends register
 * themselves under their own name and can be selected by the training runner.
 */
export class MovementModelBackendRegistry {
  private readonly backends = new Map<string, MovementModelBackend>();

  constructor(seed: MovementModelBackend[] = [new MarkovMovementBackend()]) {
    for (const backend of seed) {
      this.register(backend);
    }
  }

  register(backend: MovementModelBackend): void {
    this.backends.set(backend.name, backend);
  }

  has(name: string): boolean {
    return this.backends.has(name);
  }

  get(name: string): MovementModelBackend {
    const backend = this.backends.get(name);
    if (!backend) {
      throw new Error(`unknown movement-model backend: ${name}`);
    }
    return backend;
  }

  list(): string[] {
    return [...this.backends.keys()].sort();
  }
}

/** Convenience: a registry preloaded with the built-in deterministic backend. */
export function createDefaultMovementBackendRegistry(): MovementModelBackendRegistry {
  return new MovementModelBackendRegistry();
}

export type MovementDatasetBuildOptions = {
  /** Which action field to use as the movement token. Defaults to `tool`. */
  tokenField?: "tool" | "summary";
  /** Drop sequences with fewer than this many tokens. Defaults to 1. */
  minLength?: number;
};

/** Build a movement dataset from recorded trajectory spans (actions in ts order). */
export function buildMovementDatasetFromTrajectories(
  trajectories: TrajectorySpan[],
  options: MovementDatasetBuildOptions = {},
): MovementDataset {
  const tokenField = options.tokenField ?? "tool";
  const minLength = options.minLength ?? 1;
  const sequences: MovementSequence[] = [];
  for (const trajectory of trajectories) {
    const tokens = [...trajectory.actions]
      .sort((a, b) => a.ts - b.ts)
      .map((action) => (tokenField === "summary" ? action.summary : action.tool));
    if (tokens.length >= minLength) {
      sequences.push({ id: trajectory.id, tokens });
    }
  }
  return { version: 1, sequences };
}

/** Build a movement dataset from exported replay manifests (action events in ts order). */
export function buildMovementDatasetFromReplays(
  replays: ExportedReplayManifest[],
  options: MovementDatasetBuildOptions = {},
): MovementDataset {
  const tokenField = options.tokenField ?? "tool";
  const minLength = options.minLength ?? 1;
  const sequences: MovementSequence[] = [];
  for (const replay of replays) {
    const tokens = replay.events
      .filter((event): event is Extract<typeof event, { kind: "action" }> => event.kind === "action")
      .slice()
      .sort((a, b) => a.ts - b.ts)
      .map((event) => (tokenField === "summary" ? event.summary : event.tool));
    if (tokens.length >= minLength) {
      sequences.push({ id: replay.trajectoryIds.join("+") || replay.sessionId, tokens });
    }
  }
  return { version: 1, sequences };
}

export type MovementEvaluation = {
  /** Held-out sequences scored. */
  sequences: number;
  /** Total next-token predictions made across all sequences. */
  predictions: number;
  /** How many top-1 predictions matched the recorded next movement. */
  top1Correct: number;
  top1Accuracy: number;
  /** How many times the recorded next movement fell within the top-k ranked. */
  topKCorrect: number;
  topKAccuracy: number;
  /** Mean negative log2 probability assigned to the true next movement. */
  logLoss: number;
  /** 2^logLoss — lower is a tighter fit to held-out movement structure. */
  perplexity: number;
};

export type MovementEvaluationOptions = {
  /** k for top-k accuracy. Defaults to 3. */
  topK?: number;
  /**
   * Whether to score the terminal MOVEMENT_END transition (did the model know
   * to stop?). Defaults to true.
   */
  scoreTermination?: boolean;
};

/**
 * Generalization / replay-fidelity harness. Walks each held-out sequence and
 * asks the model to predict each next movement from the growing prefix, scoring
 * top-1 accuracy, top-k accuracy, and perplexity. Held-out sequences the model
 * never trained on measure how well it generalizes to new-but-related movements.
 */
export function evaluateMovementModel(
  model: MovementModel,
  heldOut: MovementSequence[],
  options: MovementEvaluationOptions = {},
): MovementEvaluation {
  const topK = Math.max(1, Math.floor(options.topK ?? 3));
  const scoreTermination = options.scoreTermination ?? true;
  // Floor keeps perplexity finite when the true movement is unseen. Computed
  // once — the vocabulary is fixed for a trained model.
  const floor = 1 / (model.serialize().vocabulary.length + 2);
  let predictions = 0;
  let top1Correct = 0;
  let topKCorrect = 0;
  let logLossSum = 0;

  for (const sequence of heldOut) {
    const targets = scoreTermination ? [...sequence.tokens, MOVEMENT_END] : [...sequence.tokens];
    const context: MovementToken[] = [];
    for (const target of targets) {
      const ranked = model.predictNext(context);
      predictions += 1;
      if (ranked[0]?.token === target) {
        top1Correct += 1;
      }
      if (ranked.slice(0, topK).some((prediction) => prediction.token === target)) {
        topKCorrect += 1;
      }
      const match = ranked.find((prediction) => prediction.token === target);
      const probability = match ? Math.max(match.probability, floor) : floor;
      logLossSum += -Math.log2(probability);
      if (target !== MOVEMENT_END) {
        context.push(target);
      }
    }
  }

  const logLoss = predictions === 0 ? 0 : logLossSum / predictions;
  return {
    sequences: heldOut.length,
    predictions,
    top1Correct,
    top1Accuracy: predictions === 0 ? 0 : top1Correct / predictions,
    topKCorrect,
    topKAccuracy: predictions === 0 ? 0 : topKCorrect / predictions,
    logLoss,
    perplexity: 2 ** logLoss,
  };
}
