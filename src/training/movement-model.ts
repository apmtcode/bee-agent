/**
 * Pluggable movement-model backend for the local-movement learning subsystem.
 *
 * The capture pipeline (`src/capture`) records movements as {@link TrajectorySpan}
 * actions; the training pipeline (`src/training`) exports a reviewed dataset and
 * generates on-device (MLX / Axolotl) launch scripts. What was missing is the
 * piece that actually *learns* from that dataset so bee-agent can (c) repeat a
 * recorded movement and (d) generalize to new-but-related movements.
 *
 * This module provides:
 *   - a {@link MovementDataset} representation derived from trajectories/exports,
 *   - a {@link MovementModelBackend} interface (the pluggable seam — a real
 *     on-device small model can implement it later),
 *   - {@link MarkovMovementBackend}, a deterministic, dependency-free mock backend
 *     that learns a variable-order back-off n-gram over movement tokens. It is
 *     fully reproducible (no randomness, no clock) so it validates the whole
 *     capture → dataset → train → replay/generalize loop in the cloud/CI without
 *     touching a real machine, and
 *   - {@link evaluateReplayFidelity}, a held-out generalization eval harness.
 */

import type { ReviewedExportManifest } from "./export-manifest.js";
import type { ReplayTimelineEvent } from "../capture/replay.js";
import type { TrajectorySpan } from "../capture/trajectory.js";

// ---------------------------------------------------------------------------
// Dataset representation
// ---------------------------------------------------------------------------

/** A single observable movement (one recorded action) in an abstract form. */
export type MovementToken = {
  tool: string;
  gesture?: string;
  target?: string;
  direction?: string;
  summary: string;
};

/** One recorded movement trajectory: an ordered sequence of tokens toward a goal. */
export type MovementSequence = {
  id: string;
  goal: string;
  tokens: MovementToken[];
  reward?: number;
};

export type MovementDataset = {
  version: 1;
  sequences: MovementSequence[];
};

export type MovementTrainOptions = {
  /** Maximum n-gram context order. Higher = more faithful replay, less generalization. */
  maxOrder?: number;
};

export type MovementModelStats = {
  backendId: string;
  order: number;
  sequenceCount: number;
  goalCount: number;
  tokenCount: number;
  vocabularySize: number;
};

export type MovementGenerateMode = "auto" | "replay" | "generalize";

export type MovementGenerateRequest = {
  goal: string;
  mode?: MovementGenerateMode;
  maxLength?: number;
};

export type MovementGenerateResult = {
  goal: string;
  mode: Exclude<MovementGenerateMode, "auto">;
  /** The learned goal the model matched against (undefined for a cold generalization). */
  matchedGoal?: string;
  matchScore: number;
  tokens: MovementToken[];
};

export type MovementPrediction = {
  token: MovementToken;
  probability: number;
};

export type MovementModelSnapshot = {
  version: 1;
  backendId: string;
  order: number;
  vocabulary: Record<string, MovementToken>;
  starts: Record<string, Record<string, number>>;
  transitions: Record<string, Record<string, Record<string, number>>>;
  goals: Array<{ goal: string; startKey: string; tokens: string[]; reward: number }>;
};

// ---------------------------------------------------------------------------
// Dataset derivation
// ---------------------------------------------------------------------------

/** Turn recorded trajectory spans into a training-ready movement dataset. */
export function deriveMovementDatasetFromTrajectories(trajectories: TrajectorySpan[]): MovementDataset {
  const sequences = trajectories
    .map((trajectory) => {
      const tokens = trajectory.actions.map((action) => actionToToken(action.tool, action.summary, action.metadata));
      if (tokens.length === 0) {
        return undefined;
      }
      return {
        id: trajectory.id,
        goal: deriveGoal(trajectory),
        tokens,
        ...(trajectory.outcome?.reward !== undefined ? { reward: trajectory.outcome.reward } : {}),
      } satisfies MovementSequence;
    })
    .filter((sequence): sequence is MovementSequence => sequence !== undefined);
  return { version: 1, sequences };
}

/** Derive a movement dataset from a reviewed export manifest's replay timelines. */
export function deriveMovementDatasetFromExport(manifest: ReviewedExportManifest): MovementDataset {
  const rewardByTrajectory = new Map<string, number>();
  for (const trajectory of manifest.trajectories) {
    if (trajectory.reward !== undefined) {
      rewardByTrajectory.set(trajectory.id, trajectory.reward);
    }
  }

  const sequences: MovementSequence[] = [];
  for (const replay of manifest.replays) {
    const byTrajectory = new Map<string, ReplayTimelineEvent[]>();
    for (const event of replay.events) {
      if (event.kind !== "action") {
        continue;
      }
      const bucket = byTrajectory.get(event.trajectoryId) ?? [];
      bucket.push(event);
      byTrajectory.set(event.trajectoryId, bucket);
    }
    for (const [trajectoryId, events] of byTrajectory) {
      const ordered = [...events].sort((a, b) => a.ts - b.ts);
      const tokens = ordered.map((event) =>
        event.kind === "action" ? actionToToken(event.tool, event.summary) : actionToToken("unknown", ""),
      );
      if (tokens.length === 0) {
        continue;
      }
      const reward = rewardByTrajectory.get(trajectoryId);
      sequences.push({
        id: trajectoryId,
        goal: deriveGoalFromTokens(tokens),
        tokens,
        ...(reward !== undefined ? { reward } : {}),
      });
    }
  }
  return { version: 1, sequences };
}

function deriveGoal(trajectory: TrajectorySpan): string {
  if (trajectory.outcome?.summary) {
    return trajectory.outcome.summary;
  }
  const firstObservation = trajectory.observations[0];
  if (firstObservation) {
    return firstObservation.summary;
  }
  return trajectory.sessionId;
}

function deriveGoalFromTokens(tokens: MovementToken[]): string {
  return tokens.map((token) => token.summary).join(" then ");
}

function actionToToken(tool: string, summary: string, metadata?: Record<string, unknown>): MovementToken {
  const gesture = readString(metadata, "gesture");
  const target = readString(metadata, "target");
  const direction = readString(metadata, "direction");
  return {
    tool,
    ...(gesture ? { gesture } : {}),
    ...(target ? { target } : {}),
    ...(direction ? { direction } : {}),
    summary,
  };
}

function readString(metadata: Record<string, unknown> | undefined, key: string): string | undefined {
  const value = metadata?.[key];
  return typeof value === "string" && value.length > 0 ? value : undefined;
}

// ---------------------------------------------------------------------------
// Pluggable backend interface
// ---------------------------------------------------------------------------

/**
 * A pluggable movement-model backend. The deterministic {@link MarkovMovementBackend}
 * ships in-tree for cloud/CI; a real on-device backend (e.g. MLX-trained small
 * model) implements this same shape so the training pipeline is backend-agnostic.
 */
export interface MovementModelBackend {
  readonly id: string;
  train(dataset: MovementDataset, options?: MovementTrainOptions): Promise<TrainedMovementModel>;
}

export interface TrainedMovementModel {
  readonly backendId: string;
  readonly stats: MovementModelStats;
  /** Produce a full movement toward a goal (exact replay or generalized). */
  generate(request: MovementGenerateRequest): MovementGenerateResult;
  /** Predict the single most likely next token following a prefix. */
  predictNext(prefix: MovementToken[]): MovementPrediction | undefined;
  /** Serialize to a persistable, restorable snapshot. */
  snapshot(): MovementModelSnapshot;
}

// ---------------------------------------------------------------------------
// Deterministic back-off n-gram backend
// ---------------------------------------------------------------------------

const END = "__END__";
const CTX_SEP = "|";
const DEFAULT_ORDER = 2;
const DEFAULT_MAX_LENGTH = 32;

export class MarkovMovementBackend implements MovementModelBackend {
  readonly id = "markov-mock";

  async train(dataset: MovementDataset, options: MovementTrainOptions = {}): Promise<TrainedMovementModel> {
    const order = Math.max(1, Math.floor(options.maxOrder ?? DEFAULT_ORDER));
    return MarkovMovementModel.fromDataset(this.id, dataset, order);
  }
}

/** Restore a previously trained model from a snapshot (persistence seam). */
export function restoreMovementModel(snapshot: MovementModelSnapshot): TrainedMovementModel {
  return MarkovMovementModel.fromSnapshot(snapshot);
}

type CountTable = Map<string, number>;

type GoalEntry = { goal: string; goalKey: string; startKey: string; tokens: string[]; reward: number };

class MarkovMovementModel implements TrainedMovementModel {
  private constructor(
    readonly backendId: string,
    private readonly order: number,
    private readonly vocabulary: Map<string, MovementToken>,
    // starts[goalKey] -> startKey -> weight (goal-similarity weighting happens at query time)
    private readonly starts: Map<string, CountTable>,
    // transitions[order] -> contextKey -> nextKey -> count
    private readonly transitions: Map<number, Map<string, CountTable>>,
    private readonly goals: GoalEntry[],
  ) {}

  get stats(): MovementModelStats {
    return {
      backendId: this.backendId,
      order: this.order,
      sequenceCount: this.goals.length,
      goalCount: new Set(this.goals.map((entry) => entry.goalKey)).size,
      tokenCount: this.goals.reduce((total, entry) => total + entry.tokens.length, 0),
      vocabularySize: this.vocabulary.size,
    };
  }

  static fromDataset(backendId: string, dataset: MovementDataset, order: number): MarkovMovementModel {
    const vocabulary = new Map<string, MovementToken>();
    const starts = new Map<string, CountTable>();
    const transitions = new Map<number, Map<string, CountTable>>();
    const goals: GoalEntry[] = [];

    for (const sequence of dataset.sequences) {
      const keys = sequence.tokens.map((token) => {
        const key = tokenKey(token);
        if (!vocabulary.has(key)) {
          vocabulary.set(key, token);
        }
        return key;
      });
      if (keys.length === 0) {
        continue;
      }
      const goalKey = normalizeGoal(sequence.goal);
      const startTable = starts.get(goalKey) ?? new Map<string, number>();
      increment(startTable, keys[0]!);
      starts.set(goalKey, startTable);

      const augmented = [...keys, END];
      for (let i = 1; i < augmented.length; i += 1) {
        for (let ord = 1; ord <= order; ord += 1) {
          if (i - ord < 0) {
            continue;
          }
          const context = augmented.slice(i - ord, i).join(CTX_SEP);
          recordTransition(transitions, ord, context, augmented[i]!);
        }
        // unigram (order 0) frequency of following tokens
        recordTransition(transitions, 0, "", augmented[i]!);
      }

      goals.push({
        goal: sequence.goal,
        goalKey,
        startKey: keys[0]!,
        tokens: keys,
        reward: sequence.reward ?? 0,
      });
    }

    return new MarkovMovementModel(backendId, order, vocabulary, starts, transitions, goals);
  }

  generate(request: MovementGenerateRequest): MovementGenerateResult {
    const maxLength = Math.max(1, Math.floor(request.maxLength ?? DEFAULT_MAX_LENGTH));
    const requestKey = normalizeGoal(request.goal);
    const best = this.bestGoalMatch(requestKey);
    const requestedMode = request.mode ?? "auto";
    const useReplay = requestedMode === "replay" || (requestedMode === "auto" && best?.score === 1);

    if (useReplay && best) {
      const canonical = this.canonicalSequenceFor(best.goalKey);
      if (canonical) {
        return {
          goal: request.goal,
          mode: "replay",
          matchedGoal: canonical.goal,
          matchScore: best.score,
          tokens: canonical.tokens.map((key) => this.materialize(key)),
        };
      }
    }

    const seed = this.chooseStart(requestKey);
    const keys: string[] = [];
    if (seed) {
      keys.push(seed);
    }
    while (keys.length < maxLength) {
      const next = this.greedyNext(keys);
      if (next === undefined || next === END) {
        break;
      }
      keys.push(next);
    }

    return {
      goal: request.goal,
      mode: "generalize",
      ...(best ? { matchedGoal: best.goal } : {}),
      matchScore: best?.score ?? 0,
      tokens: keys.map((key) => this.materialize(key)),
    };
  }

  predictNext(prefix: MovementToken[]): MovementPrediction | undefined {
    const keys = prefix.map((token) => tokenKey(token));
    const next = this.greedyNext(keys);
    if (next === undefined || next === END) {
      return undefined;
    }
    const probability = this.transitionProbability(keys, next);
    return { token: this.materialize(next), probability };
  }

  snapshot(): MovementModelSnapshot {
    return {
      version: 1,
      backendId: this.backendId,
      order: this.order,
      vocabulary: Object.fromEntries(this.vocabulary),
      starts: Object.fromEntries([...this.starts].map(([key, table]) => [key, Object.fromEntries(table)])),
      transitions: Object.fromEntries(
        [...this.transitions].map(([ord, table]) => [
          String(ord),
          Object.fromEntries([...table].map(([ctx, counts]) => [ctx, Object.fromEntries(counts)])),
        ]),
      ),
      goals: this.goals.map((entry) => ({
        goal: entry.goal,
        startKey: entry.startKey,
        tokens: entry.tokens,
        reward: entry.reward,
      })),
    };
  }

  static fromSnapshot(snapshot: MovementModelSnapshot): MarkovMovementModel {
    const vocabulary = new Map(Object.entries(snapshot.vocabulary));
    const starts = new Map(
      Object.entries(snapshot.starts).map(([key, table]) => [key, new Map(Object.entries(table))] as const),
    );
    const transitions = new Map(
      Object.entries(snapshot.transitions).map(
        ([ord, table]) =>
          [
            Number(ord),
            new Map(Object.entries(table).map(([ctx, counts]) => [ctx, new Map(Object.entries(counts))] as const)),
          ] as const,
      ),
    );
    const goals = snapshot.goals.map((entry) => ({
      goal: entry.goal,
      goalKey: normalizeGoal(entry.goal),
      startKey: entry.startKey,
      tokens: entry.tokens,
      reward: entry.reward,
    }));
    return new MarkovMovementModel(snapshot.backendId, snapshot.order, vocabulary, starts, transitions, goals);
  }

  // --- internals ----------------------------------------------------------

  private materialize(key: string): MovementToken {
    return this.vocabulary.get(key) ?? { tool: "unknown", summary: key };
  }

  private bestGoalMatch(requestKey: string): { goal: string; goalKey: string; score: number } | undefined {
    let best: { goal: string; goalKey: string; score: number } | undefined;
    for (const entry of this.goals) {
      const score = goalSimilarity(requestKey, entry.goalKey);
      if (score <= 0) {
        continue;
      }
      if (
        best === undefined ||
        score > best.score ||
        (score === best.score && entry.goalKey < best.goalKey)
      ) {
        best = { goal: entry.goal, goalKey: entry.goalKey, score };
      }
    }
    return best;
  }

  private canonicalSequenceFor(goalKey: string): { goal: string; tokens: string[] } | undefined {
    let best: { goal: string; tokens: string[]; reward: number } | undefined;
    for (const entry of this.goals) {
      if (entry.goalKey !== goalKey) {
        continue;
      }
      if (best === undefined || entry.reward > best.reward) {
        best = { goal: entry.goal, tokens: entry.tokens, reward: entry.reward };
      }
    }
    return best ? { goal: best.goal, tokens: best.tokens } : undefined;
  }

  private chooseStart(requestKey: string): string | undefined {
    // Weight every learned start token by the similarity of its goal to the request.
    const weighted = new Map<string, number>();
    for (const [goalKey, table] of this.starts) {
      const similarity = goalSimilarity(requestKey, goalKey);
      if (similarity <= 0) {
        continue;
      }
      for (const [startKey, count] of table) {
        weighted.set(startKey, (weighted.get(startKey) ?? 0) + similarity * count);
      }
    }
    if (weighted.size > 0) {
      return argmax(weighted);
    }
    // Cold start: fall back to the globally most frequent start token.
    const global = new Map<string, number>();
    for (const table of this.starts.values()) {
      for (const [startKey, count] of table) {
        global.set(startKey, (global.get(startKey) ?? 0) + count);
      }
    }
    return global.size > 0 ? argmax(global) : undefined;
  }

  private greedyNext(prefixKeys: string[]): string | undefined {
    for (let ord = Math.min(this.order, prefixKeys.length); ord >= 1; ord -= 1) {
      const context = prefixKeys.slice(prefixKeys.length - ord).join(CTX_SEP);
      const table = this.transitions.get(ord)?.get(context);
      if (table && table.size > 0) {
        return argmax(table);
      }
    }
    const unigram = this.transitions.get(0)?.get("");
    return unigram && unigram.size > 0 ? argmax(unigram) : undefined;
  }

  private transitionProbability(prefixKeys: string[], nextKey: string): number {
    for (let ord = Math.min(this.order, prefixKeys.length); ord >= 1; ord -= 1) {
      const context = prefixKeys.slice(prefixKeys.length - ord).join(CTX_SEP);
      const table = this.transitions.get(ord)?.get(context);
      if (table && table.size > 0) {
        return ratio(table, nextKey);
      }
    }
    const unigram = this.transitions.get(0)?.get("");
    return unigram && unigram.size > 0 ? ratio(unigram, nextKey) : 0;
  }
}

// ---------------------------------------------------------------------------
// Generalization eval harness
// ---------------------------------------------------------------------------

export type ReplayFidelityReport = {
  evaluated: number;
  exactMatches: number;
  exactMatchRate: number;
  meanTokenAccuracy: number;
  perSequence: Array<{ id: string; goal: string; exact: boolean; tokenAccuracy: number }>;
};

/**
 * Measure how faithfully a trained model reproduces held-out movements — the
 * generalization signal for the subsystem. Sequences whose goals were unseen in
 * training test (d) generalization; seen goals test (c) replay.
 */
export function evaluateReplayFidelity(
  model: TrainedMovementModel,
  heldOut: MovementSequence[],
  options: { mode?: MovementGenerateMode } = {},
): ReplayFidelityReport {
  const perSequence: ReplayFidelityReport["perSequence"] = [];
  let exactMatches = 0;
  let tokenAccuracySum = 0;

  for (const sequence of heldOut) {
    const expected = sequence.tokens.map((token) => tokenKey(token));
    const generated = model
      .generate({ goal: sequence.goal, mode: options.mode ?? "auto", maxLength: Math.max(1, expected.length) })
      .tokens.map((token) => tokenKey(token));
    const exact = keysEqual(expected, generated);
    const tokenAccuracy = positionalAccuracy(expected, generated);
    if (exact) {
      exactMatches += 1;
    }
    tokenAccuracySum += tokenAccuracy;
    perSequence.push({ id: sequence.id, goal: sequence.goal, exact, tokenAccuracy });
  }

  const evaluated = heldOut.length;
  return {
    evaluated,
    exactMatches,
    exactMatchRate: evaluated === 0 ? 0 : exactMatches / evaluated,
    meanTokenAccuracy: evaluated === 0 ? 0 : tokenAccuracySum / evaluated,
    perSequence,
  };
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

export function tokenKey(token: MovementToken): string {
  if (token.gesture) {
    const focus = token.target ?? token.direction ?? "";
    return `${token.tool}:${token.gesture}:${focus}`.toLowerCase();
  }
  return `${token.tool}:${token.summary}`.toLowerCase();
}

const GOAL_STOPWORDS = new Set(["the", "a", "an", "to", "in", "on", "of", "then", "and", "with", "into", "for"]);

function normalizeGoal(goal: string): string {
  return goalWords(goal).join(" ");
}

function goalWords(goal: string): string[] {
  return goal
    .toLowerCase()
    .split(/[^a-z0-9]+/)
    .filter((word) => word.length > 0 && !GOAL_STOPWORDS.has(word));
}

/** Jaccard similarity over goal keywords, in [0, 1]. Identical goals => 1. */
function goalSimilarity(a: string, b: string): number {
  if (a === b) {
    return 1;
  }
  const left = new Set(a.split(" ").filter((word) => word.length > 0));
  const right = new Set(b.split(" ").filter((word) => word.length > 0));
  if (left.size === 0 || right.size === 0) {
    return 0;
  }
  let intersection = 0;
  for (const word of left) {
    if (right.has(word)) {
      intersection += 1;
    }
  }
  const union = left.size + right.size - intersection;
  return union === 0 ? 0 : intersection / union;
}

function increment(table: CountTable, key: string): void {
  table.set(key, (table.get(key) ?? 0) + 1);
}

function recordTransition(
  transitions: Map<number, Map<string, CountTable>>,
  order: number,
  context: string,
  nextKey: string,
): void {
  const byContext = transitions.get(order) ?? new Map<string, CountTable>();
  const counts = byContext.get(context) ?? new Map<string, number>();
  increment(counts, nextKey);
  byContext.set(context, counts);
  transitions.set(order, byContext);
}

/** Deterministic argmax: highest count, ties broken by lexicographically smaller key. */
function argmax(table: CountTable): string | undefined {
  let bestKey: string | undefined;
  let bestCount = -Infinity;
  for (const [key, count] of table) {
    if (count > bestCount || (count === bestCount && (bestKey === undefined || key < bestKey))) {
      bestKey = key;
      bestCount = count;
    }
  }
  return bestKey;
}

function ratio(table: CountTable, key: string): number {
  let total = 0;
  for (const count of table.values()) {
    total += count;
  }
  return total === 0 ? 0 : (table.get(key) ?? 0) / total;
}

function keysEqual(a: string[], b: string[]): boolean {
  if (a.length !== b.length) {
    return false;
  }
  return a.every((key, index) => key === b[index]);
}

function positionalAccuracy(expected: string[], generated: string[]): number {
  if (expected.length === 0) {
    return generated.length === 0 ? 1 : 0;
  }
  const span = Math.max(expected.length, generated.length);
  let matches = 0;
  for (let i = 0; i < span; i += 1) {
    if (expected[i] !== undefined && expected[i] === generated[i]) {
      matches += 1;
    }
  }
  return matches / span;
}
