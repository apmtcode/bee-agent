import type { ReplayTimelineEvent } from "../capture/replay.js";
import type { TrajectorySpan } from "../capture/trajectory.js";

/**
 * Movement-model backend: the (c) train + (d) generalize half of the
 * local-movement learning subsystem.
 *
 * The capture/replay pipeline turns real user input into ordered
 * {@link ReplayTimelineEvent} timelines. This module discretizes those
 * timelines into {@link MovementToken} sequences and learns a *sequence model*
 * over them so bee-agent can (a) replay a recorded movement and (b) generalize
 * to a novel-but-related movement by following learned transitions.
 *
 * The backend is intentionally pluggable: on-device the user may swap in a real
 * small local model (MLX/axolotl, wired via the {@link LocalAppleSiliconTrainingRunner}).
 * In the cloud/CI we run the deterministic {@link MarkovMovementBackend} so the
 * full train -> infer -> evaluate loop is exercised with zero external deps and
 * zero randomness.
 */

/** A discretized movement — e.g. `action:device:swipe:down`. */
export type MovementToken = string;

/** Sentinel appended to every training sequence so generation can terminate. */
export const MOVEMENT_END_TOKEN: MovementToken = "end";

export type MovementSequence = {
  id: string;
  tokens: MovementToken[];
};

export type MovementTrainingConfig = {
  /** Maximum context length (n-gram order). Defaults to 2. */
  order?: number;
};

export type MovementPrediction = {
  token: MovementToken;
  /** How many tokens of context actually matched (after backoff). */
  contextLength: number;
  /** Estimated probability of this token within the backed-off context. */
  probability: number;
  /** True when the model predicts the end of the movement. */
  terminal: boolean;
};

export type MovementGenerateOptions = {
  /** Hard cap on generated tokens (excluding the terminal). Defaults to 64. */
  maxSteps?: number;
};

export type MovementGramSnapshot = {
  context: MovementToken[];
  next: MovementToken;
  count: number;
};

export type MovementModelSnapshot = {
  version: 1;
  backend: string;
  order: number;
  vocabulary: MovementToken[];
  grams: MovementGramSnapshot[];
};

export interface MovementModel {
  readonly backend: string;
  readonly order: number;
  /** Predict the single most likely next token given a prior context. */
  predictNext(context: MovementToken[]): MovementPrediction | undefined;
  /** Greedily roll out a continuation from a seed context. */
  generate(seed: MovementToken[], options?: MovementGenerateOptions): MovementToken[];
  serialize(): MovementModelSnapshot;
}

export interface MovementModelBackend {
  readonly name: string;
  train(dataset: MovementSequence[], config?: MovementTrainingConfig): MovementModel;
}

// ---------------------------------------------------------------------------
// Tokenizers: capture/replay artifacts -> MovementSequence
// ---------------------------------------------------------------------------

function normalizeSegment(value: string | undefined): string {
  if (!value) {
    return "";
  }
  return value
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
}

/**
 * Canonicalize a single replay action/observation event into a movement token.
 * Transcript events are conversational, not movements, so they are skipped by
 * {@link tokenizeReplayEvents}; this helper only handles the movement kinds.
 */
export function tokenizeReplayEvent(event: ReplayTimelineEvent): MovementToken | undefined {
  if (event.kind === "action") {
    const tool = normalizeSegment(event.tool) || "tool";
    const summary = normalizeSegment(event.summary) || "act";
    return `action:${tool}:${summary}`;
  }
  if (event.kind === "observation") {
    const source = normalizeSegment(event.source) || "source";
    const summary = normalizeSegment(event.summary) || "obs";
    return `observation:${source}:${summary}`;
  }
  return undefined;
}

/**
 * Turn a replay timeline into a movement token sequence. Events are assumed to
 * already be time-ordered (as {@link buildReplayManifest} guarantees); we keep
 * only the movement-bearing kinds.
 */
export function tokenizeReplayEvents(events: readonly ReplayTimelineEvent[]): MovementToken[] {
  const tokens: MovementToken[] = [];
  for (const event of events) {
    const token = tokenizeReplayEvent(event);
    if (token !== undefined) {
      tokens.push(token);
    }
  }
  return tokens;
}

/** Turn a trajectory span's observations + actions into a movement sequence. */
export function tokenizeTrajectory(trajectory: TrajectorySpan): MovementSequence {
  const merged = [
    ...trajectory.observations.map((observation) => ({ ts: observation.ts, event: observation })),
    ...trajectory.actions.map((action) => ({ ts: action.ts, event: action })),
  ].sort((a, b) => a.ts - b.ts);

  const tokens: MovementToken[] = [];
  for (const { event } of merged) {
    if (event.kind === "action") {
      const tool = normalizeSegment(event.tool) || "tool";
      const summary = normalizeSegment(event.summary) || "act";
      tokens.push(`action:${tool}:${summary}`);
    } else {
      const source = normalizeSegment(event.source) || "source";
      const summary = normalizeSegment(event.summary) || "obs";
      tokens.push(`observation:${source}:${summary}`);
    }
  }

  return { id: trajectory.id, tokens };
}

// ---------------------------------------------------------------------------
// MarkovMovementBackend: deterministic, cloud-runnable reference backend
// ---------------------------------------------------------------------------

type GramCounts = Map<string, Map<MovementToken, number>>;

function contextKey(context: readonly MovementToken[]): string {
  return context.join(" ");
}

/**
 * Order-N n-gram sequence model with stupid-backoff decoding. Fully
 * deterministic: ties are broken by (higher count, then lexicographically
 * smaller token), so the same dataset always yields the same model and the same
 * rollouts — a hard requirement for reproducible CI.
 */
export class MarkovMovementBackend implements MovementModelBackend {
  readonly name = "markov";

  constructor(private readonly defaultOrder: number = 2) {}

  train(dataset: MovementSequence[], config: MovementTrainingConfig = {}): MovementModel {
    const order = Math.max(1, Math.floor(config.order ?? this.defaultOrder));
    const counts: GramCounts = new Map();
    const vocabulary = new Set<MovementToken>();

    const record = (context: MovementToken[], next: MovementToken): void => {
      const key = contextKey(context);
      let bucket = counts.get(key);
      if (!bucket) {
        bucket = new Map();
        counts.set(key, bucket);
      }
      bucket.set(next, (bucket.get(next) ?? 0) + 1);
    };

    for (const sequence of dataset) {
      const stream = [...sequence.tokens, MOVEMENT_END_TOKEN];
      for (const token of sequence.tokens) {
        vocabulary.add(token);
      }
      for (let index = 0; index < stream.length; index += 1) {
        const next = stream[index]!;
        // Record every context length from 0..order ending just before `next`.
        for (let ctxLen = 0; ctxLen <= order; ctxLen += 1) {
          const start = index - ctxLen;
          if (start < 0) {
            continue;
          }
          record(stream.slice(start, index), next);
        }
      }
    }

    return new MarkovMovementModel(this.name, order, counts, [...vocabulary].sort());
  }
}

class MarkovMovementModel implements MovementModel {
  constructor(
    readonly backend: string,
    readonly order: number,
    private readonly counts: GramCounts,
    private readonly vocabulary: MovementToken[],
  ) {}

  predictNext(context: MovementToken[]): MovementPrediction | undefined {
    const maxContext = Math.min(this.order, context.length);
    for (let ctxLen = maxContext; ctxLen >= 0; ctxLen -= 1) {
      const suffix = context.slice(context.length - ctxLen);
      const bucket = this.counts.get(contextKey(suffix));
      if (!bucket || bucket.size === 0) {
        continue;
      }
      let bestToken: MovementToken | undefined;
      let bestCount = -1;
      let total = 0;
      for (const [token, count] of bucket) {
        total += count;
        if (count > bestCount || (count === bestCount && bestToken !== undefined && token < bestToken)) {
          bestCount = count;
          bestToken = token;
        }
      }
      if (bestToken === undefined) {
        continue;
      }
      return {
        token: bestToken,
        contextLength: ctxLen,
        probability: total > 0 ? bestCount / total : 0,
        terminal: bestToken === MOVEMENT_END_TOKEN,
      };
    }
    return undefined;
  }

  generate(seed: MovementToken[], options: MovementGenerateOptions = {}): MovementToken[] {
    const maxSteps = Math.max(0, options.maxSteps ?? 64);
    const produced: MovementToken[] = [];
    const context = [...seed];
    for (let step = 0; step < maxSteps; step += 1) {
      const prediction = this.predictNext(context);
      if (!prediction || prediction.terminal) {
        break;
      }
      produced.push(prediction.token);
      context.push(prediction.token);
    }
    return produced;
  }

  serialize(): MovementModelSnapshot {
    const grams: MovementGramSnapshot[] = [];
    for (const [key, bucket] of this.counts) {
      const context = key === "" ? [] : key.split(" ");
      for (const [next, count] of bucket) {
        grams.push({ context, next, count });
      }
    }
    // Deterministic ordering so snapshots are stable across runs.
    grams.sort((a, b) => {
      const ca = contextKey(a.context);
      const cb = contextKey(b.context);
      if (ca !== cb) {
        return ca < cb ? -1 : 1;
      }
      return a.next < b.next ? -1 : a.next > b.next ? 1 : 0;
    });
    return {
      version: 1,
      backend: this.backend,
      order: this.order,
      vocabulary: [...this.vocabulary],
      grams,
    };
  }
}

/** Rehydrate a movement model from a serialized snapshot. */
export function loadMovementModel(snapshot: MovementModelSnapshot): MovementModel {
  const counts: GramCounts = new Map();
  for (const gram of snapshot.grams) {
    const key = contextKey(gram.context);
    let bucket = counts.get(key);
    if (!bucket) {
      bucket = new Map();
      counts.set(key, bucket);
    }
    bucket.set(gram.next, gram.count);
  }
  return new MarkovMovementModel(snapshot.backend, snapshot.order, counts, [...snapshot.vocabulary]);
}

// ---------------------------------------------------------------------------
// Generalization eval harness
// ---------------------------------------------------------------------------

export type MovementEvalCase = {
  id: string;
  /** Context the model is seeded with. */
  seed: MovementToken[];
  /** The continuation the model is expected to reproduce. */
  expected: MovementToken[];
};

export type MovementEvalResult = {
  id: string;
  predicted: MovementToken[];
  /** Fraction of aligned positions that match (0..1). */
  tokenAccuracy: number;
  /** True when the model reproduced the continuation exactly. */
  exactMatch: boolean;
  /** True when the very first predicted token matched. */
  firstStepMatch: boolean;
};

export type MovementEvalReport = {
  caseCount: number;
  exactMatchRate: number;
  meanTokenAccuracy: number;
  firstStepAccuracy: number;
  cases: MovementEvalResult[];
};

/**
 * Build eval cases by holding out the tail of each sequence: the first
 * `seedLength` tokens seed the model and the remainder is the expected
 * continuation. Sequences too short to split are skipped.
 */
export function buildHeldOutEvalCases(
  sequences: MovementSequence[],
  seedLength = 1,
): MovementEvalCase[] {
  const boundedSeed = Math.max(0, Math.floor(seedLength));
  const cases: MovementEvalCase[] = [];
  for (const sequence of sequences) {
    if (sequence.tokens.length <= boundedSeed) {
      continue;
    }
    cases.push({
      id: sequence.id,
      seed: sequence.tokens.slice(0, boundedSeed),
      expected: sequence.tokens.slice(boundedSeed),
    });
  }
  return cases;
}

/** Measure how faithfully a model reproduces / generalizes held-out movements. */
export function evaluateMovementModel(
  model: MovementModel,
  cases: MovementEvalCase[],
): MovementEvalReport {
  const results: MovementEvalResult[] = cases.map((evalCase) => {
    const predicted = model.generate(evalCase.seed, {
      maxSteps: Math.max(evalCase.expected.length, 1),
    });
    const alignLength = Math.max(evalCase.expected.length, predicted.length);
    let matches = 0;
    for (let index = 0; index < alignLength; index += 1) {
      if (predicted[index] !== undefined && predicted[index] === evalCase.expected[index]) {
        matches += 1;
      }
    }
    const tokenAccuracy = alignLength === 0 ? 1 : matches / alignLength;
    const exactMatch =
      predicted.length === evalCase.expected.length &&
      predicted.every((token, index) => token === evalCase.expected[index]);
    const firstStepMatch =
      evalCase.expected.length > 0 && predicted[0] === evalCase.expected[0];
    return {
      id: evalCase.id,
      predicted,
      tokenAccuracy,
      exactMatch,
      firstStepMatch,
    };
  });

  const caseCount = results.length;
  const sum = (selector: (result: MovementEvalResult) => number): number =>
    results.reduce((accumulator, result) => accumulator + selector(result), 0);

  return {
    caseCount,
    exactMatchRate: caseCount === 0 ? 0 : sum((result) => (result.exactMatch ? 1 : 0)) / caseCount,
    meanTokenAccuracy: caseCount === 0 ? 0 : sum((result) => result.tokenAccuracy) / caseCount,
    firstStepAccuracy: caseCount === 0 ? 0 : sum((result) => (result.firstStepMatch ? 1 : 0)) / caseCount,
    cases: results,
  };
}
