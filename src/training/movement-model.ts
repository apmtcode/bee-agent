// Local-movement learning subsystem — objective #2 (c) train + (d) generalize.
//
// This module provides an *in-process, pluggable* movement-model backend so
// bee-agent can post-train a small local model on recorded movement/action
// trajectories and then (a) repeat the recorded movements and (b) generalize to
// new-but-related movements — without any GPU, OS input access, or external
// process. The real on-device backends (mlx / axolotl, wired by `runner.ts`)
// live behind the same `MovementModelBackend` interface; the deterministic
// n-gram backend here is the reference/mock implementation used by tests and by
// cloud runs where no accelerator is available.
//
// Everything is deterministic: identical datasets + config yield identical
// models and identical predictions, so training is fully reproducible in CI.

import type { ReplayManifest } from "../capture/replay.js";
import type { TrajectorySpan } from "../capture/trajectory.js";

/** A single normalized movement token (e.g. `"mouse.move"`, `"key.press"`). */
export type MovementToken = string;

/** Sentinel tokens framing every training sequence. */
export const MOVEMENT_START_TOKEN: MovementToken = "start";
export const MOVEMENT_END_TOKEN: MovementToken = "end";

/** One ordered sequence of movement tokens derived from a trajectory. */
export type MovementSequence = {
  id: string;
  tokens: MovementToken[];
};

/** A dataset is just a bag of ordered sequences. */
export type MovementDataset = {
  sequences: MovementSequence[];
};

export type MovementTrainingConfig = {
  /** Context length (order) of the model. Higher = more literal replay. */
  order: number;
};

export const DEFAULT_MOVEMENT_TRAINING_CONFIG: MovementTrainingConfig = {
  order: 2,
};

export type MovementPrediction = {
  token: MovementToken;
  /** Estimated probability of `token` given the supplied context. */
  probability: number;
  /** How many context tokens were actually used after back-off (0 = unigram). */
  backoffOrder: number;
};

/** Portable, JSON-serializable form of a trained model. */
export type SerializedMovementModel = {
  version: 1;
  backend: string;
  order: number;
  /** context-key -> (token -> observed count). */
  transitions: Record<string, Record<MovementToken, number>>;
  /** token -> observed count (unigram fallback). */
  unigram: Record<MovementToken, number>;
  vocabulary: MovementToken[];
  sequenceCount: number;
  tokenCount: number;
};

/** A model produced by a backend. Stateless after training. */
export interface TrainedMovementModel {
  readonly backend: string;
  readonly order: number;
  /** Most-likely next token given a context, or undefined if the model is empty. */
  predictNext(context: MovementToken[]): MovementPrediction | undefined;
  /** Full ranked next-token distribution for a context (deterministically ordered). */
  distribution(context: MovementToken[]): MovementPrediction[];
  /** Greedily roll out up to `maxSteps` tokens from `prefix`, stopping at END. */
  generate(prefix: MovementToken[], maxSteps: number): MovementToken[];
  serialize(): SerializedMovementModel;
}

/** A pluggable training backend. Real on-device backends implement this too. */
export interface MovementModelBackend {
  readonly name: string;
  train(dataset: MovementDataset, config?: Partial<MovementTrainingConfig>): TrainedMovementModel;
  restore(model: SerializedMovementModel): TrainedMovementModel;
}

// ---------------------------------------------------------------------------
// Tokenization: trajectories / replay manifests -> movement sequences
// ---------------------------------------------------------------------------

/**
 * Normalize a recorded action into a stable, low-cardinality movement token.
 * We key on the tool plus a coarse verb extracted from the summary so related
 * movements share tokens (enabling generalization) without leaking raw content.
 */
export function normalizeMovementToken(action: { tool: string; summary?: string }): MovementToken {
  const tool = action.tool.trim().toLowerCase();
  const verb = coarseVerb(action.summary);
  return verb ? `${tool}:${verb}` : tool;
}

function coarseVerb(summary: string | undefined): string | undefined {
  if (!summary) {
    return undefined;
  }
  const first = summary.trim().toLowerCase().split(/[\s_:/-]+/, 1)[0];
  if (!first || first.length > 24) {
    return undefined;
  }
  // Keep only simple alphabetic verbs; anything else collapses to the bare tool.
  return /^[a-z]+$/.test(first) ? first : undefined;
}

/** Build a training sequence from a single trajectory's ordered actions. */
export function buildMovementSequenceFromTrajectory(trajectory: TrajectorySpan): MovementSequence {
  const tokens = [...trajectory.actions]
    .sort((a, b) => a.ts - b.ts)
    .map((action) => normalizeMovementToken(action));
  return { id: trajectory.id, tokens };
}

export function buildMovementDatasetFromTrajectories(trajectories: TrajectorySpan[]): MovementDataset {
  return {
    sequences: trajectories
      .map((trajectory) => buildMovementSequenceFromTrajectory(trajectory))
      .filter((sequence) => sequence.tokens.length > 0),
  };
}

/** Build a training sequence from a replay manifest's action events. */
export function buildMovementSequenceFromReplay(manifest: ReplayManifest): MovementSequence {
  const tokens = manifest.events
    .filter((event): event is Extract<typeof event, { kind: "action" }> => event.kind === "action")
    .sort((a, b) => a.ts - b.ts)
    .map((event) => normalizeMovementToken({ tool: event.tool, summary: event.summary }));
  return { id: manifest.sessionId, tokens };
}

export function buildMovementDatasetFromReplays(manifests: ReplayManifest[]): MovementDataset {
  return {
    sequences: manifests
      .map((manifest) => buildMovementSequenceFromReplay(manifest))
      .filter((sequence) => sequence.tokens.length > 0),
  };
}

// ---------------------------------------------------------------------------
// Deterministic n-gram backend (the reference/mock local model)
// ---------------------------------------------------------------------------

const CONTEXT_SEP = "";

class NGramMovementModel implements TrainedMovementModel {
  constructor(
    readonly backend: string,
    readonly order: number,
    private readonly transitions: Map<string, Map<MovementToken, number>>,
    private readonly unigram: Map<MovementToken, number>,
    private readonly sequenceCount: number,
    private readonly tokenCount: number,
  ) {}

  predictNext(context: MovementToken[]): MovementPrediction | undefined {
    const ranked = this.distribution(context);
    return ranked[0];
  }

  distribution(context: MovementToken[]): MovementPrediction[] {
    // Back off from the full order down to unigram until a context matches.
    for (let used = Math.min(this.order, context.length); used >= 1; used--) {
      const key = contextKey(context.slice(context.length - used));
      const counts = this.transitions.get(key);
      if (counts && counts.size > 0) {
        return rank(counts, used);
      }
    }
    if (this.unigram.size > 0) {
      return rank(this.unigram, 0);
    }
    return [];
  }

  generate(prefix: MovementToken[], maxSteps: number): MovementToken[] {
    const out: MovementToken[] = [];
    let context = [MOVEMENT_START_TOKEN, ...prefix];
    for (let step = 0; step < maxSteps; step++) {
      const next = this.predictNext(context);
      if (!next || next.token === MOVEMENT_END_TOKEN) {
        break;
      }
      out.push(next.token);
      context = [...context, next.token];
    }
    return out;
  }

  serialize(): SerializedMovementModel {
    const transitions: Record<string, Record<MovementToken, number>> = {};
    for (const [key, counts] of this.transitions) {
      transitions[key] = Object.fromEntries([...counts.entries()].sort(byTokenAsc));
    }
    const vocabulary = [...this.unigram.keys()].sort();
    return {
      version: 1,
      backend: this.backend,
      order: this.order,
      transitions,
      unigram: Object.fromEntries([...this.unigram.entries()].sort(byTokenAsc)),
      vocabulary,
      sequenceCount: this.sequenceCount,
      tokenCount: this.tokenCount,
    };
  }
}

function rank(counts: Map<MovementToken, number>, backoffOrder: number): MovementPrediction[] {
  const total = [...counts.values()].reduce((sum, count) => sum + count, 0);
  return [...counts.entries()]
    // Deterministic order: higher count first, then lexical token for ties.
    .sort((a, b) => (b[1] - a[1]) || (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0))
    .map(([token, count]) => ({ token, probability: total === 0 ? 0 : count / total, backoffOrder }));
}

function byTokenAsc(a: [string, number], b: [string, number]): number {
  return a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0;
}

function contextKey(context: MovementToken[]): string {
  return context.join(CONTEXT_SEP);
}

/**
 * Deterministic n-gram transition backend. Learns, for each context of up to
 * `order` tokens, the distribution of the following token. Back-off makes it
 * generalize: a novel prefix that ends in a familiar suffix still predicts.
 */
export class NGramMovementBackend implements MovementModelBackend {
  readonly name = "ngram";

  train(dataset: MovementDataset, config?: Partial<MovementTrainingConfig>): TrainedMovementModel {
    const order = Math.max(1, config?.order ?? DEFAULT_MOVEMENT_TRAINING_CONFIG.order);
    const transitions = new Map<string, Map<MovementToken, number>>();
    const unigram = new Map<MovementToken, number>();
    let tokenCount = 0;

    for (const sequence of dataset.sequences) {
      const framed = [MOVEMENT_START_TOKEN, ...sequence.tokens, MOVEMENT_END_TOKEN];
      for (let i = 1; i < framed.length; i++) {
        const next = framed[i];
        bump(unigram, next);
        tokenCount++;
        // Record this transition at every context length 1..order (for back-off).
        for (let used = 1; used <= order && i - used >= 0; used++) {
          const key = contextKey(framed.slice(i - used, i));
          let counts = transitions.get(key);
          if (!counts) {
            counts = new Map<MovementToken, number>();
            transitions.set(key, counts);
          }
          bump(counts, next);
        }
      }
    }

    return new NGramMovementModel(
      this.name,
      order,
      transitions,
      unigram,
      dataset.sequences.length,
      tokenCount,
    );
  }

  restore(model: SerializedMovementModel): TrainedMovementModel {
    const transitions = new Map<string, Map<MovementToken, number>>();
    for (const [key, counts] of Object.entries(model.transitions)) {
      transitions.set(key, new Map(Object.entries(counts)));
    }
    const unigram = new Map<MovementToken, number>(Object.entries(model.unigram));
    return new NGramMovementModel(
      model.backend,
      model.order,
      transitions,
      unigram,
      model.sequenceCount,
      model.tokenCount,
    );
  }
}

function bump(map: Map<MovementToken, number>, token: MovementToken): void {
  map.set(token, (map.get(token) ?? 0) + 1);
}

// ---------------------------------------------------------------------------
// Pluggable backend registry
// ---------------------------------------------------------------------------

const backendRegistry = new Map<string, MovementModelBackend>();

export function registerMovementBackend(backend: MovementModelBackend): void {
  backendRegistry.set(backend.name, backend);
}

export function getMovementBackend(name: string): MovementModelBackend | undefined {
  return backendRegistry.get(name);
}

export function listMovementBackends(): string[] {
  return [...backendRegistry.keys()].sort();
}

// Register the built-in reference backend.
registerMovementBackend(new NGramMovementBackend());

// ---------------------------------------------------------------------------
// Generalization eval harness
// ---------------------------------------------------------------------------

export type MovementEvalResult = {
  /** Next-token prediction accuracy over every position of every sequence. */
  nextTokenAccuracy: number;
  /** Sequences whose greedy rollout from START exactly reproduces the tokens. */
  exactReplayRate: number;
  /** Token-level fidelity of greedy rollouts vs. the reference sequences. */
  replayFidelity: number;
  predictedTokens: number;
  correctTokens: number;
  evaluatedSequences: number;
};

/**
 * Measure how well a trained model reproduces / generalizes to a set of
 * reference sequences (typically held out from training). Deterministic.
 */
export function evaluateMovementModel(
  model: TrainedMovementModel,
  reference: MovementDataset,
): MovementEvalResult {
  let predictedTokens = 0;
  let correctTokens = 0;
  let exactReplays = 0;
  let fidelitySum = 0;
  let evaluatedSequences = 0;

  for (const sequence of reference.sequences) {
    if (sequence.tokens.length === 0) {
      continue;
    }
    evaluatedSequences++;
    const framed = [MOVEMENT_START_TOKEN, ...sequence.tokens, MOVEMENT_END_TOKEN];

    // Teacher-forced next-token accuracy.
    for (let i = 1; i < framed.length; i++) {
      const prediction = model.predictNext(framed.slice(0, i));
      predictedTokens++;
      if (prediction && prediction.token === framed[i]) {
        correctTokens++;
      }
    }

    // Free-running greedy rollout fidelity.
    const rollout = model.generate([], sequence.tokens.length);
    const matched = countPrefixMatches(rollout, sequence.tokens);
    fidelitySum += sequence.tokens.length === 0 ? 1 : matched / sequence.tokens.length;
    if (rollout.length === sequence.tokens.length && matched === sequence.tokens.length) {
      exactReplays++;
    }
  }

  return {
    nextTokenAccuracy: predictedTokens === 0 ? 0 : correctTokens / predictedTokens,
    exactReplayRate: evaluatedSequences === 0 ? 0 : exactReplays / evaluatedSequences,
    replayFidelity: evaluatedSequences === 0 ? 0 : fidelitySum / evaluatedSequences,
    predictedTokens,
    correctTokens,
    evaluatedSequences,
  };
}

function countPrefixMatches(a: MovementToken[], b: MovementToken[]): number {
  const limit = Math.min(a.length, b.length);
  let matched = 0;
  for (let i = 0; i < limit; i++) {
    if (a[i] === b[i]) {
      matched++;
    } else {
      break;
    }
  }
  return matched;
}
