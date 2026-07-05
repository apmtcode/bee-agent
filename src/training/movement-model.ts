import type { TrajectoryAction, TrajectorySpan } from "../capture/trajectory.js";
import type { ReplayManifest } from "../capture/replay.js";

/**
 * Local-movement learning: a small, deterministic, on-device-friendly model
 * that learns to repeat recorded movements and generalize to new-but-related
 * ones. The heavy on-device backends (mlx/axolotl) are described by
 * {@link ./runner.js}; this module provides the *pluggable model contract* plus
 * a deterministic backend that runs anywhere (cloud/CI) so the whole
 * capture -> dataset -> train -> infer loop can be exercised without a GPU.
 *
 * Design objective (self-evolution standing objective #2c/#2d): post-train a
 * local model on the recorded dataset to (a) repeat the recorded movements and
 * (b) generalize to new but related movements. The default backend is an
 * n-gram model with stupid-backoff over movement tokens plus a coarser
 * "movement class" backoff, which is exactly the mechanism that yields
 * generalization: an unseen full context falls back to shorter contexts, and an
 * unseen concrete target falls back to its movement class (tool + gesture +
 * direction), so a novel target reachable by a known gesture is still
 * predicted.
 */

/** A single discretized movement step. Concrete target is generalizable-away. */
export type MovementToken = {
  tool: string;
  gesture?: string;
  target?: string;
  direction?: string;
  value?: string;
};

/** A recorded movement episode: an optional context plus an ordered token run. */
export type MovementSequence = {
  sequenceId: string;
  context: MovementSequenceContext;
  tokens: MovementToken[];
};

export type MovementSequenceContext = {
  appId?: string;
  goal?: string;
  outcome?: "success" | "failure" | "aborted";
};

/** Prefix handed to the model to ask "what movement comes next?". */
export type MovementContext = {
  context?: MovementSequenceContext;
  history: MovementToken[];
};

export type MovementPrediction = {
  token: MovementToken;
  /** Stupid-backoff score in (0, 1]; higher is more likely. Not normalized. */
  score: number;
  /** Context length actually used after backoff (n-gram order - 1). */
  backoffOrder: number;
  /** True when the token was reconstructed via class backoff (generalization). */
  generalized: boolean;
};

/** Opaque, serializable model state so backends can persist/round-trip. */
export type MovementModelSnapshot = {
  backend: string;
  version: 1;
  order: number;
  trainedSequences: number;
  trainedTokens: number;
  /** Backend-specific learned parameters. */
  parameters: Record<string, unknown>;
};

export type MovementTrainOptions = {
  /** Max n-gram order (context window). Clamped to [1, 8]. Default 3. */
  order?: number;
  /** Stupid-backoff discount applied per dropped context token. Default 0.4. */
  backoff?: number;
};

/**
 * Pluggable backend contract. Real on-device backends (a small local model,
 * mlx-lora, etc.) implement the same surface; {@link NGramMovementBackend} is
 * the deterministic reference used in tests and as a graceful fallback.
 */
export interface MovementModelBackend {
  readonly name: string;
  train(dataset: MovementSequence[], options?: MovementTrainOptions): MovementModelSnapshot;
  predictNext(model: MovementModelSnapshot, context: MovementContext): MovementPrediction[];
}

/** Stable string key for exact-match n-gram counting. */
export function movementTokenKey(token: MovementToken): string {
  return [
    token.tool,
    token.gesture ?? "",
    token.direction ?? "",
    token.target ?? "",
    token.value ?? "",
  ].join("␟");
}

/** Coarser key that drops the concrete target/value — the generalizable class. */
export function movementClassKey(token: MovementToken): string {
  return [token.tool, token.gesture ?? "", token.direction ?? ""].join("␟");
}

/**
 * Reconstruct a {@link MovementToken} from a recorded trajectory action. The
 * device/os/browser adapters stash gesture/target/direction in
 * `action.metadata`, so we recover structure where available and fall back to
 * the action summary otherwise.
 */
export function movementTokenFromAction(action: TrajectoryAction): MovementToken {
  const metadata = action.metadata ?? {};
  const token: MovementToken = { tool: action.tool };
  const gesture = readString(metadata.gesture);
  const target = readString(metadata.target);
  const direction = readString(metadata.direction);
  const value = readString(metadata.valueSummary) ?? readString(metadata.value);
  if (gesture) token.gesture = gesture;
  if (target) token.target = target;
  if (direction) token.direction = direction;
  if (value) token.value = value;
  if (!gesture && !target && !direction && action.summary) {
    token.target = action.summary;
  }
  return token;
}

/** Build training sequences from recorded trajectory spans. */
export function buildMovementSequences(trajectories: TrajectorySpan[]): MovementSequence[] {
  return trajectories
    .map((trajectory) => {
      const tokens = [...trajectory.actions]
        .sort((a, b) => a.ts - b.ts)
        .map((action) => movementTokenFromAction(action));
      const appId = readString(trajectory.observations.find((o) => o.source === "device")?.metadata?.appId);
      const context: MovementSequenceContext = {};
      if (appId) context.appId = appId;
      if (trajectory.outcome?.status) context.outcome = trajectory.outcome.status;
      if (trajectory.outcome?.summary) context.goal = trajectory.outcome.summary;
      return { sequenceId: trajectory.id, context, tokens };
    })
    .filter((sequence) => sequence.tokens.length > 0);
}

/** Build training sequences from a replay manifest's action timeline. */
export function buildMovementSequencesFromReplay(replay: ReplayManifest): MovementSequence[] {
  const byTrajectory = new Map<string, MovementToken[]>();
  for (const event of replay.events) {
    if (event.kind !== "action") continue;
    const tokens = byTrajectory.get(event.trajectoryId) ?? [];
    tokens.push({ tool: event.tool, target: event.summary });
    byTrajectory.set(event.trajectoryId, tokens);
  }
  return [...byTrajectory.entries()]
    .map(([trajectoryId, tokens]) => ({
      sequenceId: `${replay.sessionId}:${trajectoryId}`,
      context: {},
      tokens,
    }))
    .filter((sequence) => sequence.tokens.length > 0);
}

type NGramParameters = {
  backoff: number;
  /** context-key -> next-token-key -> count */
  transitions: Record<string, Record<string, number>>;
  /** class-context-key -> next-class-key -> count (generalization layer). */
  classTransitions: Record<string, Record<string, number>>;
  /** class-key -> canonical token (for reconstructing generalized predictions). */
  classExemplars: Record<string, MovementToken>;
  /** token-key -> count (unigram floor). */
  unigrams: Record<string, number>;
  /** token-key -> canonical token. */
  vocabulary: Record<string, MovementToken>;
};

const BOUNDARY = "␂"; // sequence-start marker so first-step prediction works.

/**
 * Deterministic n-gram model with stupid-backoff. No randomness, no I/O — safe
 * to run in the cloud and to unit-test. Generalizes two ways:
 *  1. Context backoff: unseen full history falls back to shorter suffixes.
 *  2. Class backoff: an unseen concrete target falls back to its movement class
 *     (tool+gesture+direction), reconstructed from a learned exemplar — so a
 *     new-but-related target reachable by a known gesture is still predicted.
 */
export class NGramMovementBackend implements MovementModelBackend {
  readonly name = "ngram-backoff";

  train(dataset: MovementSequence[], options: MovementTrainOptions = {}): MovementModelSnapshot {
    const order = clamp(Math.trunc(options.order ?? 3), 1, 8);
    const backoff = clampFloat(options.backoff ?? 0.4, 0.05, 0.95);
    const parameters: NGramParameters = {
      backoff,
      transitions: {},
      classTransitions: {},
      classExemplars: {},
      unigrams: {},
      vocabulary: {},
    };

    let trainedTokens = 0;
    for (const sequence of dataset) {
      const keys = sequence.tokens.map((token) => movementTokenKey(token));
      const classKeys = sequence.tokens.map((token) => movementClassKey(token));
      sequence.tokens.forEach((token, index) => {
        parameters.vocabulary[keys[index]] = token;
        parameters.unigrams[keys[index]] = (parameters.unigrams[keys[index]] ?? 0) + 1;
        if (!parameters.classExemplars[classKeys[index]]) {
          parameters.classExemplars[classKeys[index]] = { ...token };
          delete parameters.classExemplars[classKeys[index]].target;
          delete parameters.classExemplars[classKeys[index]].value;
        }
        trainedTokens += 1;
        // Record every backoff order for this position: contexts of length 0..order-1.
        for (let ctxLen = 0; ctxLen < order; ctxLen += 1) {
          const atStart = index - ctxLen <= 0;
          // Skip the position-agnostic empty context at non-start positions: it
          // would lump all mid-sequence tokens together and short-circuit the
          // class-backoff generalization path. The `unigrams` table is the floor.
          if (ctxLen === 0 && !atStart) {
            continue;
          }
          const historyKeys = keys.slice(Math.max(0, index - ctxLen), index);
          const contextKey = joinContext(historyKeys, atStart);
          increment(parameters.transitions, contextKey, keys[index]);
          const classHistoryKeys = classKeys.slice(Math.max(0, index - ctxLen), index);
          const classContextKey = joinContext(classHistoryKeys, atStart);
          increment(parameters.classTransitions, classContextKey, classKeys[index]);
        }
      });
    }

    return {
      backend: this.name,
      version: 1,
      order,
      trainedSequences: dataset.length,
      trainedTokens,
      parameters: parameters as unknown as Record<string, unknown>,
    };
  }

  predictNext(model: MovementModelSnapshot, context: MovementContext): MovementPrediction[] {
    const parameters = model.parameters as unknown as NGramParameters;
    const historyKeys = context.history.map((token) => movementTokenKey(token));

    // Try the longest context first; back off toward the unigram floor. A
    // context is a "sequence start" context (BOUNDARY-marked, matching training)
    // exactly when it spans the whole observed history.
    for (let ctxLen = Math.min(model.order - 1, historyKeys.length); ctxLen >= 0; ctxLen -= 1) {
      const suffix = historyKeys.slice(historyKeys.length - ctxLen);
      const contextKey = joinContext(suffix, ctxLen === historyKeys.length);
      const counts = parameters.transitions[contextKey];
      if (!counts) continue;
      const predictions = scoreCounts(counts, parameters, ctxLen, model.order);
      if (predictions.length > 0) {
        return predictions;
      }
    }

    // Exact context exhausted — generalize through the movement-class layer.
    const generalized = this.predictViaClass(parameters, context, model.order);
    if (generalized.length > 0) {
      return generalized;
    }

    // Final floor: unigram frequencies.
    return scoreCounts(parameters.unigrams, parameters, 0, model.order);
  }

  private predictViaClass(
    parameters: NGramParameters,
    context: MovementContext,
    order: number,
  ): MovementPrediction[] {
    const classHistory = context.history.map((token) => movementClassKey(token));
    for (let ctxLen = Math.min(order - 1, classHistory.length); ctxLen >= 0; ctxLen -= 1) {
      const suffix = classHistory.slice(classHistory.length - ctxLen);
      const contextKey = joinContext(suffix, ctxLen === classHistory.length);
      const counts = parameters.classTransitions[contextKey];
      if (!counts) continue;
      const total = sumValues(counts);
      if (total === 0) continue;
      const discount = Math.pow(parameters.backoff, order - 1 - ctxLen + 1);
      return Object.entries(counts)
        .map(([classKey, count]) => {
          const exemplar = parameters.classExemplars[classKey] ?? { tool: classKey.split("␟")[0] };
          return {
            token: { ...exemplar },
            score: (count / total) * discount,
            backoffOrder: ctxLen,
            generalized: true,
          } satisfies MovementPrediction;
        })
        .sort(comparePredictions);
    }
    return [];
  }
}

/**
 * High-level learner: dataset in, predictions/rollouts out. Pluggable backend
 * defaults to the deterministic n-gram model.
 */
export class MovementLearner {
  private snapshot: MovementModelSnapshot | undefined;

  constructor(private readonly backend: MovementModelBackend = new NGramMovementBackend()) {}

  train(dataset: MovementSequence[], options?: MovementTrainOptions): MovementModelSnapshot {
    this.snapshot = this.backend.train(dataset, options);
    return this.snapshot;
  }

  loadSnapshot(snapshot: MovementModelSnapshot): void {
    this.snapshot = snapshot;
  }

  predictNext(context: MovementContext): MovementPrediction[] {
    return this.backend.predictNext(this.requireSnapshot(), context);
  }

  /**
   * Autoregressive replay: from a seed history, greedily emit the most likely
   * next movement `steps` times. This is how the model "repeats the recorded
   * movements" and, via backoff, generalizes to related ones.
   */
  rollout(seed: MovementContext, steps: number): MovementToken[] {
    const history = [...seed.history];
    const produced: MovementToken[] = [];
    for (let step = 0; step < steps; step += 1) {
      const predictions = this.backend.predictNext(this.requireSnapshot(), {
        context: seed.context,
        history,
      });
      const next = predictions[0];
      if (!next) break;
      produced.push(next.token);
      history.push(next.token);
    }
    return produced;
  }

  private requireSnapshot(): MovementModelSnapshot {
    if (!this.snapshot) {
      throw new Error("MovementLearner: train() or loadSnapshot() must be called before prediction");
    }
    return this.snapshot;
  }
}

export type MovementGeneralizationReport = {
  evaluatedSequences: number;
  evaluatedSteps: number;
  /** Exact next-token top-1 accuracy. */
  exactAccuracy: number;
  /** Movement-class top-1 accuracy (tolerates a new concrete target). */
  classAccuracy: number;
  /** Share of correct class predictions produced via generalization backoff. */
  generalizationRate: number;
};

/**
 * Generalization eval harness: for each held-out sequence, walk its prefixes and
 * ask the model to predict the next movement. Measures how well the model
 * repeats known movements (exactAccuracy) and generalizes to related ones
 * (classAccuracy), plus how often the correct call came from the class-backoff
 * generalization path.
 */
export function evaluateMovementGeneralization(
  backend: MovementModelBackend,
  snapshot: MovementModelSnapshot,
  heldOut: MovementSequence[],
): MovementGeneralizationReport {
  let steps = 0;
  let exactHits = 0;
  let classHits = 0;
  let generalizedClassHits = 0;
  let evaluatedSequences = 0;

  for (const sequence of heldOut) {
    if (sequence.tokens.length < 1) continue;
    evaluatedSequences += 1;
    for (let index = 0; index < sequence.tokens.length; index += 1) {
      const expected = sequence.tokens[index];
      const predictions = backend.predictNext(snapshot, {
        context: sequence.context,
        history: sequence.tokens.slice(0, index),
      });
      const top = predictions[0];
      steps += 1;
      if (!top) continue;
      if (movementTokenKey(top.token) === movementTokenKey(expected)) {
        exactHits += 1;
      }
      if (movementClassKey(top.token) === movementClassKey(expected)) {
        classHits += 1;
        if (top.generalized) generalizedClassHits += 1;
      }
    }
  }

  return {
    evaluatedSequences,
    evaluatedSteps: steps,
    exactAccuracy: steps === 0 ? 0 : exactHits / steps,
    classAccuracy: steps === 0 ? 0 : classHits / steps,
    generalizationRate: classHits === 0 ? 0 : generalizedClassHits / classHits,
  };
}

function scoreCounts(
  counts: Record<string, number>,
  parameters: NGramParameters,
  ctxLen: number,
  order: number,
): MovementPrediction[] {
  const total = sumValues(counts);
  if (total === 0) return [];
  const discount = Math.pow(parameters.backoff, order - 1 - ctxLen);
  return Object.entries(counts)
    .map(([tokenKey, count]) => ({
      token: parameters.vocabulary[tokenKey] ?? { tool: tokenKey.split("␟")[0] },
      score: (count / total) * discount,
      backoffOrder: ctxLen,
      generalized: false,
    }))
    .sort(comparePredictions);
}

function comparePredictions(a: MovementPrediction, b: MovementPrediction): number {
  if (b.score !== a.score) return b.score - a.score;
  // Deterministic tiebreak on the token key.
  return movementTokenKey(a.token).localeCompare(movementTokenKey(b.token));
}

function joinContext(historyKeys: string[], atStart: boolean): string {
  const parts = atStart ? [BOUNDARY, ...historyKeys] : historyKeys;
  return parts.join("␅");
}

function increment(table: Record<string, Record<string, number>>, contextKey: string, tokenKey: string): void {
  const row = table[contextKey] ?? (table[contextKey] = {});
  row[tokenKey] = (row[tokenKey] ?? 0) + 1;
}

function sumValues(counts: Record<string, number>): number {
  let total = 0;
  for (const value of Object.values(counts)) total += value;
  return total;
}

function readString(value: unknown): string | undefined {
  return typeof value === "string" && value.length > 0 ? value : undefined;
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function clampFloat(value: number, min: number, max: number): number {
  if (Number.isNaN(value)) return min;
  return Math.min(max, Math.max(min, value));
}
