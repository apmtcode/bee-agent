import type { ReplayManifest, ReplayTimelineEvent } from "../capture/replay.js";
import type { TrajectorySpan } from "../capture/trajectory.js";

/**
 * Local-movement learning subsystem — model backend + trainer.
 *
 * This module implements pieces (c) and (d) of the movement objective: given a
 * dataset of recorded movement/action sequences, *post-train a local model* that
 * (c) repeats the recorded movements and (d) generalizes to new-but-related
 * movements. It is intentionally backend-pluggable: {@link LocalMovementModelBackend}
 * is the seam a real on-device small model implements later, while the built-in
 * {@link NgramMovementModelBackend} is deterministic so it can be exercised in the
 * cloud / CI with synthetic event streams (no real OS input, no GPU).
 *
 * The learned unit is a *movement token* — a compact, replayable label for one
 * recorded action (e.g. `action:mouse.move`). Tokens are extracted from the same
 * {@link ReplayTimelineEvent} timeline the rest of the pipeline already produces,
 * so capture → dataset → train → infer/replay is a single coherent flow.
 */

export type MovementToken = string;

/** Sentinel token marking the start of a movement sequence (padding context). */
export const MOVEMENT_START_TOKEN: MovementToken = "<start>";
/** Sentinel token marking a predicted end-of-sequence. */
export const MOVEMENT_END_TOKEN: MovementToken = "<end>";

/** One recorded movement sequence the model learns from. */
export type MovementSample = {
  /** Optional provenance (trajectory / session id) for auditing datasets. */
  sourceId?: string;
  /** Ordered movement tokens, oldest first. */
  tokens: MovementToken[];
};

export type MovementPredictionContext = {
  /** Recent tokens, oldest first. Only the last `order` are consulted. */
  history: MovementToken[];
};

export type MovementPredictionSource = "exact" | "backoff" | "prior" | "unknown";

export type MovementPrediction = {
  token: MovementToken;
  /** Empirical P(token | matched-context) in [0, 1]. */
  confidence: number;
  /** How many context tokens were actually matched (0 = fell back to prior). */
  matchedOrder: number;
  source: MovementPredictionSource;
};

export type TrainedMovementModel = {
  version: 1;
  backend: string;
  /** Max n-gram context length used during training. */
  order: number;
  vocabulary: MovementToken[];
  sampleCount: number;
  tokenCount: number;
  /** Backend-specific serialized parameters (JSON-safe, so models persist). */
  parameters: Record<string, unknown>;
};

export type MovementTrainingConfig = {
  /** Max context length (n-gram order). Default 3. Clamped to >= 1. */
  order?: number;
};

/**
 * The pluggable backend seam. A real on-device model (MLX/llama.cpp/etc.) can
 * implement this same shape; the deterministic n-gram backend below is the
 * cloud/CI-safe default.
 */
export interface LocalMovementModelBackend {
  readonly name: string;
  train(samples: MovementSample[], config?: MovementTrainingConfig): TrainedMovementModel;
  predict(model: TrainedMovementModel, context: MovementPredictionContext): MovementPrediction;
}

// --- token extraction -------------------------------------------------------

/**
 * Coarse-grain a free-text action summary into a stable movement sub-label so
 * the model generalizes across near-identical recordings (e.g. slightly
 * different pixel coordinates) instead of memorizing every unique string.
 */
function bucketSummary(summary: string): string | undefined {
  const normalized = summary.trim().toLowerCase();
  if (!normalized) {
    return undefined;
  }
  const first = normalized.split(/[^a-z0-9]+/).find((part) => part.length > 0);
  return first;
}

/** Turn a single replay action event into a movement token. */
export function actionEventToToken(event: Extract<ReplayTimelineEvent, { kind: "action" }>): MovementToken {
  const bucket = bucketSummary(event.summary);
  return bucket ? `action:${event.tool}:${bucket}` : `action:${event.tool}`;
}

/** Extract one movement sample (ordered action tokens) from a replay manifest. */
export function replayToMovementSample(replay: ReplayManifest): MovementSample {
  const tokens = replay.events
    .filter((event): event is Extract<ReplayTimelineEvent, { kind: "action" }> => event.kind === "action")
    .map(actionEventToToken);
  return { sourceId: replay.sessionId, tokens };
}

/** Extract one movement sample from a trajectory span's ordered actions. */
export function trajectoryToMovementSample(trajectory: TrajectorySpan): MovementSample {
  const source = trajectory.review?.redactedActions
    ? trajectory.review.redactedActions.map((action) => ({ tool: action.tool, summary: action.summary, ts: action.ts }))
    : trajectory.actions.map((action) => ({ tool: action.tool, summary: action.summary, ts: action.ts }));
  const tokens = [...source]
    .sort((a, b) => a.ts - b.ts)
    .map((action) => {
      const bucket = bucketSummary(action.summary);
      return bucket ? `action:${action.tool}:${bucket}` : `action:${action.tool}`;
    });
  return { sourceId: trajectory.id, tokens };
}

/** Extract movement samples from many replays, dropping empty sequences. */
export function extractMovementSamples(replays: ReplayManifest[]): MovementSample[] {
  return replays.map(replayToMovementSample).filter((sample) => sample.tokens.length > 0);
}

// --- n-gram backend ---------------------------------------------------------

type NgramCounts = Record<string, Record<MovementToken, number>>;

type NgramParameters = {
  /** context-key -> { nextToken -> count }, for every context length 0..order. */
  counts: NgramCounts;
};

const CONTEXT_DELIMITER = "";

function contextKey(context: MovementToken[]): string {
  return context.length === 0 ? "" : context.join(CONTEXT_DELIMITER);
}

/**
 * Deterministic n-gram movement model with stupid-backoff generalization.
 *
 * Training records next-token frequencies for every context length `0..order`.
 * Prediction consults the longest context suffix that was ever observed and, on
 * a miss, backs off to progressively shorter contexts and finally the unigram
 * prior — this is what lets it emit sensible movements for *new* contexts that
 * share a tail with recorded ones (objective piece (d)) while still reproducing
 * recorded sequences exactly (objective piece (c)). Ties break lexicographically
 * so the same dataset always yields the same model and the same rollouts.
 */
export class NgramMovementModelBackend implements LocalMovementModelBackend {
  readonly name = "ngram-backoff";

  train(samples: MovementSample[], config: MovementTrainingConfig = {}): TrainedMovementModel {
    const order = Math.max(1, Math.floor(config.order ?? 3));
    const counts: NgramCounts = {};
    const vocabulary = new Set<MovementToken>();
    let tokenCount = 0;

    for (const sample of samples) {
      if (sample.tokens.length === 0) {
        continue;
      }
      const padded = [MOVEMENT_START_TOKEN, ...sample.tokens, MOVEMENT_END_TOKEN];
      for (const token of sample.tokens) {
        vocabulary.add(token);
      }
      for (let i = 1; i < padded.length; i += 1) {
        const next = padded[i]!;
        tokenCount += 1;
        for (let ctxLen = 0; ctxLen <= order; ctxLen += 1) {
          if (ctxLen > i) {
            break;
          }
          const context = padded.slice(i - ctxLen, i);
          const key = contextKey(context);
          const bucket = (counts[key] ??= {});
          bucket[next] = (bucket[next] ?? 0) + 1;
        }
      }
    }

    return {
      version: 1,
      backend: this.name,
      order,
      vocabulary: [...vocabulary].sort(),
      sampleCount: samples.filter((sample) => sample.tokens.length > 0).length,
      tokenCount,
      parameters: { counts } satisfies NgramParameters,
    };
  }

  predict(model: TrainedMovementModel, context: MovementPredictionContext): MovementPrediction {
    const counts = (model.parameters as NgramParameters).counts ?? {};
    const history = [MOVEMENT_START_TOKEN, ...context.history];
    const maxOrder = Math.min(model.order, history.length);

    for (let ctxLen = maxOrder; ctxLen >= 0; ctxLen -= 1) {
      const suffix = ctxLen === 0 ? [] : history.slice(history.length - ctxLen);
      const bucket = counts[contextKey(suffix)];
      if (!bucket) {
        continue;
      }
      const best = pickBest(bucket);
      if (!best) {
        continue;
      }
      const total = Object.values(bucket).reduce((sum, value) => sum + value, 0);
      return {
        token: best.token,
        confidence: total > 0 ? best.count / total : 0,
        matchedOrder: ctxLen,
        source: ctxLen === Math.min(model.order, context.history.length) && ctxLen > 0 ? "exact" : ctxLen > 0 ? "backoff" : "prior",
      };
    }

    return { token: MOVEMENT_END_TOKEN, confidence: 0, matchedOrder: 0, source: "unknown" };
  }
}

function pickBest(bucket: Record<MovementToken, number>): { token: MovementToken; count: number } | undefined {
  let best: { token: MovementToken; count: number } | undefined;
  for (const [token, count] of Object.entries(bucket)) {
    if (!best || count > best.count || (count === best.count && token < best.token)) {
      best = { token, count };
    }
  }
  return best;
}

// --- rollout (inference / replay generation) --------------------------------

export type MovementRolloutOptions = {
  /** Seed context the rollout continues from. Default: empty (fresh start). */
  seed?: MovementToken[];
  /** Hard cap on generated tokens (excludes the seed). Default 64. */
  maxSteps?: number;
};

export type MovementRolloutStep = MovementPrediction & { token: MovementToken };

export type MovementRollout = {
  seed: MovementToken[];
  steps: MovementRolloutStep[];
  /** Full emitted token sequence (excludes sentinels). */
  tokens: MovementToken[];
  stoppedReason: "end-token" | "max-steps" | "unknown";
};

/**
 * Generate a movement sequence from a trained model — the inference half of the
 * subsystem. Repeatedly predicts the next token given the growing history until
 * the model emits {@link MOVEMENT_END_TOKEN}, hits `maxSteps`, or has nothing to
 * emit. Deterministic given the same model + options.
 */
export function rolloutMovements(
  backend: LocalMovementModelBackend,
  model: TrainedMovementModel,
  options: MovementRolloutOptions = {},
): MovementRollout {
  const seed = [...(options.seed ?? [])];
  const maxSteps = Math.max(0, Math.floor(options.maxSteps ?? 64));
  const history = [...seed];
  const steps: MovementRolloutStep[] = [];
  let stoppedReason: MovementRollout["stoppedReason"] = "max-steps";

  for (let i = 0; i < maxSteps; i += 1) {
    const prediction = backend.predict(model, { history });
    if (prediction.source === "unknown") {
      stoppedReason = "unknown";
      break;
    }
    if (prediction.token === MOVEMENT_END_TOKEN) {
      stoppedReason = "end-token";
      break;
    }
    steps.push(prediction);
    history.push(prediction.token);
  }

  return {
    seed,
    steps,
    tokens: steps.map((step) => step.token),
    stoppedReason,
  };
}

// --- generalization eval harness -------------------------------------------

export type MovementEvalResult = {
  /** Total next-token decisions evaluated. */
  total: number;
  correct: number;
  accuracy: number;
  /** Correct decisions that required backing off to a shorter context. */
  generalizedCorrect: number;
  /** Correct decisions from an exact longest-context match. */
  exactCorrect: number;
  /** Mean confidence assigned to the *true* next token's prediction. */
  meanConfidence: number;
  byMatchedOrder: Record<number, { total: number; correct: number }>;
};

/**
 * Measure next-token prediction fidelity on held-out movement samples — the
 * generalization metric. For each position it feeds the true prefix as context
 * and checks whether the model's top prediction equals the true next token,
 * tracking how often correct predictions came from a backed-off (generalizing)
 * context versus an exact match.
 */
export function evaluateMovementModel(
  backend: LocalMovementModelBackend,
  model: TrainedMovementModel,
  heldOut: MovementSample[],
): MovementEvalResult {
  let total = 0;
  let correct = 0;
  let generalizedCorrect = 0;
  let exactCorrect = 0;
  let confidenceSum = 0;
  const byMatchedOrder: Record<number, { total: number; correct: number }> = {};

  for (const sample of heldOut) {
    const padded = [...sample.tokens, MOVEMENT_END_TOKEN];
    for (let i = 0; i < padded.length; i += 1) {
      const history = sample.tokens.slice(0, i);
      const expected = padded[i]!;
      const prediction = backend.predict(model, { history });
      total += 1;
      confidenceSum += prediction.confidence;
      const orderBucket = (byMatchedOrder[prediction.matchedOrder] ??= { total: 0, correct: 0 });
      orderBucket.total += 1;
      if (prediction.token === expected) {
        correct += 1;
        orderBucket.correct += 1;
        if (prediction.source === "exact") {
          exactCorrect += 1;
        } else {
          generalizedCorrect += 1;
        }
      }
    }
  }

  return {
    total,
    correct,
    accuracy: total > 0 ? correct / total : 0,
    generalizedCorrect,
    exactCorrect,
    meanConfidence: total > 0 ? confidenceSum / total : 0,
    byMatchedOrder,
  };
}

// --- synthetic event-stream generator ---------------------------------------

/**
 * Tiny deterministic PRNG (mulberry32) so synthetic datasets are reproducible
 * without depending on `Math.random`, and identical across machines/runs.
 */
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export type SyntheticMovementSpec = {
  seed: number;
  /** Number of samples to generate. */
  sampleCount: number;
  /**
   * Named movement "grammar": each key is a workflow whose value is the ordered
   * list of movement tokens it emits. The generator picks a workflow per sample
   * and optionally injects small variations (skips/repeats) to exercise
   * generalization. Defaults to a small built-in grammar.
   */
  workflows?: Record<string, MovementToken[]>;
  /** Probability [0,1] of perturbing a sample (skip/duplicate a step). Default 0.25. */
  variationRate?: number;
};

const DEFAULT_WORKFLOWS: Record<string, MovementToken[]> = {
  "open-edit-save": [
    "action:window.focus:editor",
    "action:mouse.click:menu",
    "action:key.press:open",
    "action:key.type:edit",
    "action:key.press:save",
  ],
  "browse-search-copy": [
    "action:window.focus:browser",
    "action:mouse.click:address",
    "action:key.type:search",
    "action:key.press:enter",
    "action:mouse.click:result",
    "action:key.press:copy",
  ],
};

/**
 * Generate reproducible synthetic movement samples from a small grammar of
 * workflows — validates the capture→dataset→train→infer pipeline without any
 * real OS input, and produces the near-duplicate variations the eval harness
 * uses to measure generalization.
 */
export function generateSyntheticMovementSamples(spec: SyntheticMovementSpec): MovementSample[] {
  const workflows = spec.workflows ?? DEFAULT_WORKFLOWS;
  const workflowNames = Object.keys(workflows);
  if (workflowNames.length === 0) {
    return [];
  }
  const variationRate = Math.min(1, Math.max(0, spec.variationRate ?? 0.25));
  const rng = mulberry32(spec.seed);
  const samples: MovementSample[] = [];

  for (let i = 0; i < Math.max(0, Math.floor(spec.sampleCount)); i += 1) {
    const name = workflowNames[Math.floor(rng() * workflowNames.length)]!;
    const base = workflows[name]!;
    let tokens = [...base];
    if (rng() < variationRate && tokens.length > 2) {
      if (rng() < 0.5) {
        // skip one interior step (still related, but a novel sequence)
        const dropIndex = 1 + Math.floor(rng() * (tokens.length - 2));
        tokens = tokens.filter((_, index) => index !== dropIndex);
      } else {
        // duplicate one step (a plausible human repeat)
        const dupIndex = Math.floor(rng() * tokens.length);
        tokens = [...tokens.slice(0, dupIndex + 1), tokens[dupIndex]!, ...tokens.slice(dupIndex + 1)];
      }
    }
    samples.push({ sourceId: `${name}#${i}`, tokens });
  }

  return samples;
}
