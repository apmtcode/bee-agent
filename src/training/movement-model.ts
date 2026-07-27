import { readJsonFile, writeJsonAtomic } from "../shared/fs.js";
import type { TrajectoryAction, TrajectorySpan } from "../capture/trajectory.js";

/**
 * Local-movement learning: an in-process, pluggable model backend that trains on
 * captured movement sequences and can (a) *repeat* recorded movements and (b)
 * *generalize* to new-but-related ones.
 *
 * The real on-device training path (mlx/axolotl) is emitted by
 * `LocalAppleSiliconTrainingRunner` as an external launch plan and cannot run in
 * the cloud. This module provides the missing seam: a fully deterministic,
 * dependency-free backend (an order-N Markov model over movement tokens) so the
 * end-to-end loop — tokenize → train → predict → generate → evaluate — is
 * exercised and tested here, and a documented `MovementModelBackend` interface
 * so a real small on-device model can be swapped in behind the same contract.
 */

/** A single discrete movement token, e.g. `device:tap:submit-button`. */
export type MovementToken = string;

/** Sentinel tokens delimiting a trajectory so the model can learn starts/ends. */
export const MOVEMENT_START_TOKEN: MovementToken = "<start>";
export const MOVEMENT_END_TOKEN: MovementToken = "<end>";

/** One tokenized movement sequence, tagged with its source trajectory. */
export type MovementSequence = {
  trajectoryId: string;
  tokens: MovementToken[];
};

export type MovementDataset = {
  version: 1;
  sequences: MovementSequence[];
  /** Sorted set of every token observed across the dataset (excludes sentinels). */
  vocabulary: MovementToken[];
};

/** A trained, serializable model. `kind` selects the backend that produced it. */
export type MovementModelArtifact = {
  version: 1;
  kind: string;
  order: number;
  vocabulary: MovementToken[];
  sequenceCount: number;
  tokenCount: number;
  /**
   * Transition table keyed by the joined context (`ctx.join("␟")`), each
   * mapping a next-token to the number of times it followed that context.
   */
  transitions: Record<string, Record<MovementToken, number>>;
};

export type MovementPrediction = {
  token: MovementToken;
  probability: number;
  /** How many context tokens actually matched (after back-off). */
  contextLength: number;
};

/** Inference surface produced from a trained artifact. */
export interface MovementPredictor {
  /** Most-likely next token for a context, or `undefined` if unknown. */
  predictNext(context: MovementToken[]): MovementPrediction | undefined;
  /** Full next-token distribution (after back-off), highest probability first. */
  distribution(context: MovementToken[]): MovementPrediction[];
  /**
   * Deterministically roll the model forward from a seed until it emits
   * `<end>` or hits `maxSteps`. Sentinels are stripped from the result.
   */
  generate(seed: MovementToken[], maxSteps?: number): MovementToken[];
}

export type MovementTrainOptions = {
  /** Markov order (context window). Defaults to 2. */
  order?: number;
};

/** Pluggable backend contract. A real on-device model implements this too. */
export interface MovementModelBackend {
  readonly id: string;
  train(dataset: MovementDataset, options?: MovementTrainOptions): MovementModelArtifact;
  createPredictor(artifact: MovementModelArtifact): MovementPredictor;
}

const CONTEXT_SEPARATOR = "␟";

/**
 * Turn a captured action into a stable, low-cardinality token. Prefers the
 * structured gesture/target/direction metadata that the device/browser/os
 * adapters attach; falls back to the tool name so every action tokenizes.
 */
export function tokenizeMovementAction(action: TrajectoryAction): MovementToken {
  const metadata = action.metadata ?? {};
  const parts: string[] = [normalizeTokenPart(action.tool) || "action"];
  const gesture = pickString(metadata.gesture);
  if (gesture) {
    parts.push(normalizeTokenPart(gesture));
  }
  const direction = pickString(metadata.direction);
  const target = pickString(metadata.target);
  if (target) {
    parts.push(normalizeTokenPart(target));
  } else if (direction) {
    parts.push(normalizeTokenPart(direction));
  }
  return parts.filter((part) => part.length > 0).join(":");
}

/** Tokenize one trajectory's actions (chronologically) into a movement sequence. */
export function tokenizeTrajectory(trajectory: TrajectorySpan): MovementSequence {
  const actions = trajectory.review?.redactedActions
    ? trajectory.review.redactedActions.map<TrajectoryAction>((action) => ({
        kind: "action",
        tool: action.tool,
        summary: action.summary,
        ts: action.ts,
      }))
    : trajectory.actions;
  const tokens = [...actions]
    .sort((a, b) => a.ts - b.ts)
    .map((action) => tokenizeMovementAction(action));
  return { trajectoryId: trajectory.id, tokens };
}

/** Build a training dataset from trajectory spans. Empty sequences are dropped. */
export function buildMovementDataset(trajectories: TrajectorySpan[]): MovementDataset {
  const sequences = trajectories
    .map((trajectory) => tokenizeTrajectory(trajectory))
    .filter((sequence) => sequence.tokens.length > 0);
  const vocabulary = [...new Set(sequences.flatMap((sequence) => sequence.tokens))].sort();
  return { version: 1, sequences, vocabulary };
}

/**
 * Deterministic Markov backend. Learns order-N transition counts and predicts by
 * greedy argmax with a stable tie-break (probability desc, then token asc), so
 * results are reproducible across runs and machines — a safe cloud/CI default.
 *
 * - *Repeat*: seeding with a recorded prefix reproduces that trajectory's tail.
 * - *Generalize*: because transitions are shared across trajectories, an unseen
 *   seed still yields a plausible continuation composed from observed moves,
 *   backing off to shorter contexts when the full window was never seen.
 */
export class MarkovMovementBackend implements MovementModelBackend {
  readonly id = "markov";

  train(dataset: MovementDataset, options: MovementTrainOptions = {}): MovementModelArtifact {
    const order = Math.max(1, Math.floor(options.order ?? 2));
    const transitions: Record<string, Record<MovementToken, number>> = {};
    let tokenCount = 0;

    for (const sequence of dataset.sequences) {
      const padded = [MOVEMENT_START_TOKEN, ...sequence.tokens, MOVEMENT_END_TOKEN];
      tokenCount += sequence.tokens.length;
      for (let index = 1; index < padded.length; index += 1) {
        const next = padded[index];
        // Record every back-off context length from 1..order for this position.
        for (let span = 1; span <= order; span += 1) {
          const start = index - span;
          if (start < 0) {
            break;
          }
          const context = padded.slice(start, index);
          const key = contextKey(context);
          const row = (transitions[key] ??= {});
          row[next] = (row[next] ?? 0) + 1;
        }
      }
    }

    return {
      version: 1,
      kind: this.id,
      order,
      vocabulary: [...dataset.vocabulary],
      sequenceCount: dataset.sequences.length,
      tokenCount,
      transitions,
    };
  }

  createPredictor(artifact: MovementModelArtifact): MovementPredictor {
    return new MarkovMovementPredictor(artifact);
  }
}

class MarkovMovementPredictor implements MovementPredictor {
  constructor(private readonly artifact: MovementModelArtifact) {}

  distribution(context: MovementToken[]): MovementPrediction[] {
    const window = context.slice(-this.artifact.order);
    // Back off from the longest available context to the shortest.
    for (let span = window.length; span >= 1; span -= 1) {
      const key = contextKey(window.slice(window.length - span));
      const row = this.artifact.transitions[key];
      if (!row) {
        continue;
      }
      const total = Object.values(row).reduce((sum, count) => sum + count, 0);
      if (total <= 0) {
        continue;
      }
      return Object.entries(row)
        .map(([token, count]) => ({ token, probability: count / total, contextLength: span }))
        .sort(comparePredictions);
    }
    return [];
  }

  predictNext(context: MovementToken[]): MovementPrediction | undefined {
    return this.distribution(context)[0];
  }

  generate(seed: MovementToken[], maxSteps = 64): MovementToken[] {
    const emitted: MovementToken[] = [];
    let context = [MOVEMENT_START_TOKEN, ...seed];
    for (let step = 0; step < maxSteps; step += 1) {
      const prediction = this.predictNext(context);
      if (!prediction || prediction.token === MOVEMENT_END_TOKEN) {
        break;
      }
      emitted.push(prediction.token);
      context = [...context, prediction.token];
    }
    return [...stripSentinels(seed), ...emitted];
  }
}

/** Registry so alternative (e.g. real on-device) backends can be looked up by id. */
export class MovementModelRegistry {
  private readonly backends = new Map<string, MovementModelBackend>();

  constructor(backends: MovementModelBackend[] = [new MarkovMovementBackend()]) {
    for (const backend of backends) {
      this.register(backend);
    }
  }

  register(backend: MovementModelBackend): void {
    this.backends.set(backend.id, backend);
  }

  get(id: string): MovementModelBackend | undefined {
    return this.backends.get(id);
  }

  /** Resolve the backend that produced an artifact, for inference. */
  predictorFor(artifact: MovementModelArtifact): MovementPredictor {
    const backend = this.backends.get(artifact.kind);
    if (!backend) {
      throw new Error(`no registered movement backend for kind "${artifact.kind}"`);
    }
    return backend.createPredictor(artifact);
  }

  list(): string[] {
    return [...this.backends.keys()].sort();
  }
}

export type MovementEvaluation = {
  /** Held-out sequences that contributed at least one prediction. */
  evaluatedSequences: number;
  predictedTokens: number;
  /** Top-1 next-token accuracy over every held-out position. */
  nextTokenAccuracy: number;
  /**
   * Fraction of held-out sequences the model reproduces *exactly* when seeded
   * with their first token — the core "repeat the recorded movement" metric.
   */
  exactReplayRate: number;
  /** Sequences whose held-out tokens were entirely covered by the vocabulary. */
  inVocabularySequences: number;
};

/**
 * Generalization eval harness: measures how well a trained predictor reproduces
 * and continues held-out trajectories it was not trained on.
 */
export function evaluateMovementModel(
  predictor: MovementPredictor,
  heldOut: MovementSequence[],
): MovementEvaluation {
  const vocabulary = new Set<MovementToken>();
  let predictedTokens = 0;
  let correctTokens = 0;
  let evaluatedSequences = 0;
  let exactReplays = 0;
  let inVocabularySequences = 0;

  for (const sequence of heldOut) {
    if (sequence.tokens.length === 0) {
      continue;
    }
    evaluatedSequences += 1;

    // Teacher-forced next-token accuracy across the sequence.
    let sequenceContributed = false;
    let context = [MOVEMENT_START_TOKEN];
    for (const actual of sequence.tokens) {
      const prediction = predictor.predictNext(context);
      if (prediction) {
        sequenceContributed = true;
        predictedTokens += 1;
        if (prediction.token === actual) {
          correctTokens += 1;
        }
      }
      context = [...context, actual];
      vocabulary.add(actual);
    }
    if (!sequenceContributed) {
      evaluatedSequences -= 1;
    }

    // Free-running replay from the first token.
    const seed = [sequence.tokens[0]];
    const generated = predictor.generate(seed, sequence.tokens.length + 4);
    if (tokensEqual(generated, sequence.tokens)) {
      exactReplays += 1;
    }
    inVocabularySequences += 1;
  }

  return {
    evaluatedSequences,
    predictedTokens,
    nextTokenAccuracy: predictedTokens === 0 ? 0 : correctTokens / predictedTokens,
    exactReplayRate: heldOut.length === 0 ? 0 : exactReplays / heldOut.length,
    inVocabularySequences,
  };
}

export async function saveMovementModel(filePath: string, artifact: MovementModelArtifact): Promise<void> {
  await writeJsonAtomic(filePath, artifact);
}

export async function loadMovementModel(filePath: string): Promise<MovementModelArtifact | undefined> {
  return await readJsonFile<MovementModelArtifact | undefined>(filePath, undefined);
}

function contextKey(context: MovementToken[]): string {
  return context.join(CONTEXT_SEPARATOR);
}

function comparePredictions(a: MovementPrediction, b: MovementPrediction): number {
  if (b.probability !== a.probability) {
    return b.probability - a.probability;
  }
  return a.token < b.token ? -1 : a.token > b.token ? 1 : 0;
}

function stripSentinels(tokens: MovementToken[]): MovementToken[] {
  return tokens.filter((token) => token !== MOVEMENT_START_TOKEN && token !== MOVEMENT_END_TOKEN);
}

function tokensEqual(a: MovementToken[], b: MovementToken[]): boolean {
  if (a.length !== b.length) {
    return false;
  }
  return a.every((token, index) => token === b[index]);
}

function normalizeTokenPart(value: string): string {
  return value
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
}

function pickString(value: unknown): string | undefined {
  return typeof value === "string" && value.trim().length > 0 ? value : undefined;
}
