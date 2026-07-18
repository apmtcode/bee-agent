import type { TrajectoryAction, TrajectorySpan } from "../capture/trajectory.js";

// ---------------------------------------------------------------------------
// Movement model
//
// Closes the local-movement learning loop in-process: recorded trajectory
// actions are tokenized into movement sequences, a pluggable backend learns
// from them, and the model predicts / rolls out the next movement. The default
// backend is a deterministic variable-order back-off n-gram model, so training
// and inference run entirely in the cloud (no OS, no external trainer) and are
// fully reproducible in tests. Unseen full prefixes back off to the longest
// observed suffix, which is how the model generalizes to new-but-related
// movements. The real on-device training path (mlx/axolotl via the runner)
// remains a separate, pluggable backend behind the same interface.
// ---------------------------------------------------------------------------

export type MovementToken = string;

export type MovementStep = {
  token: MovementToken;
  tool: string;
  summary: string;
};

export type MovementSequence = {
  trajectoryId: string;
  sessionId: string;
  outcome?: "success" | "failure" | "aborted";
  steps: MovementStep[];
};

export type MovementTokenizerOptions = {
  // Fold the first meaningful word of the action summary into the token so the
  // model distinguishes "type:email" from "type:search". On by default.
  includeSummaryIntent?: boolean;
};

function normalizeSegment(value: string): string {
  return value.trim().toLowerCase().replace(/\s+/g, "-");
}

function summaryIntent(summary: string): string {
  const match = summary.trim().toLowerCase().match(/[a-z0-9]+/);
  return match ? match[0] : "";
}

export function tokenizeMovementAction(
  action: Pick<TrajectoryAction, "tool" | "summary">,
  options: MovementTokenizerOptions = {},
): MovementStep {
  const tool = normalizeSegment(action.tool) || "action";
  const intent = options.includeSummaryIntent === false ? "" : summaryIntent(action.summary);
  const token = intent ? `${tool}:${intent}` : tool;
  return { token, tool: action.tool, summary: action.summary };
}

export type BuildMovementDatasetOptions = {
  // Only include trajectories a human reviewer approved (training-safe).
  requireApproved?: boolean;
  // Drop sequences shorter than this after tokenizing (default 1).
  minSteps?: number;
  tokenizer?: MovementTokenizerOptions;
};

export function buildMovementDataset(
  trajectories: TrajectorySpan[],
  options: BuildMovementDatasetOptions = {},
): MovementSequence[] {
  const minSteps = options.minSteps ?? 1;
  const sequences: MovementSequence[] = [];
  for (const trajectory of trajectories) {
    if (options.requireApproved && trajectory.review?.status !== "approved") {
      continue;
    }
    const steps = [...trajectory.actions]
      .sort((a, b) => a.ts - b.ts)
      .map((action) => tokenizeMovementAction(action, options.tokenizer));
    if (steps.length < minSteps) {
      continue;
    }
    sequences.push({
      trajectoryId: trajectory.id,
      sessionId: trajectory.sessionId,
      ...(trajectory.outcome ? { outcome: trajectory.outcome.status } : {}),
      steps,
    });
  }
  return sequences;
}

export function movementSequenceTokens(sequence: MovementSequence): MovementToken[] {
  return sequence.steps.map((step) => step.token);
}

// --- Backend contract --------------------------------------------------------

export type MovementCandidate = {
  token: MovementToken;
  probability: number;
  count: number;
};

export type MovementPrediction = {
  // null only when the model has learned nothing at all.
  token: MovementToken | null;
  probability: number;
  // Length of the context suffix actually used (back-off depth). 0 = unigram.
  order: number;
  candidates: MovementCandidate[];
};

export type MovementModelState = {
  backend: string;
  order: number;
  vocabulary: MovementToken[];
  sequenceCount: number;
  stepCount: number;
  payload: unknown;
};

export interface MovementModelBackend {
  readonly name: string;
  readonly order: number;
  train(dataset: MovementSequence[]): MovementModelState;
  predict(state: MovementModelState, context: MovementToken[]): MovementPrediction;
}

const CONTEXT_SEPARATOR = "";

type NgramPayload = {
  // context-key -> next-token -> count
  contexts: Record<string, Record<MovementToken, number>>;
};

function contextKey(tokens: MovementToken[]): string {
  return tokens.join(CONTEXT_SEPARATOR);
}

// Deterministic variable-order back-off n-gram model. Trains and predicts
// purely from observed transition counts, so results are reproducible without
// any OS or GPU. Ties break by descending count then lexical token order.
export class BackoffNgramMovementBackend implements MovementModelBackend {
  readonly name = "backoff-ngram";
  readonly order: number;

  constructor(order = 3) {
    this.order = Math.max(1, Math.floor(order));
  }

  train(dataset: MovementSequence[]): MovementModelState {
    const contexts: NgramPayload["contexts"] = {};
    const vocabulary = new Set<MovementToken>();
    let stepCount = 0;

    const record = (context: MovementToken[], next: MovementToken): void => {
      const key = contextKey(context);
      const bucket = (contexts[key] ??= {});
      bucket[next] = (bucket[next] ?? 0) + 1;
    };

    for (const sequence of dataset) {
      const tokens = movementSequenceTokens(sequence);
      for (let i = 0; i < tokens.length; i += 1) {
        const next = tokens[i]!;
        vocabulary.add(next);
        stepCount += 1;
        // Record every back-off context length from 0..order-1.
        const maxContext = Math.min(i, this.order - 1);
        for (let len = 0; len <= maxContext; len += 1) {
          record(tokens.slice(i - len, i), next);
        }
      }
    }

    return {
      backend: this.name,
      order: this.order,
      vocabulary: [...vocabulary].sort(),
      sequenceCount: dataset.length,
      stepCount,
      payload: { contexts } satisfies NgramPayload,
    };
  }

  predict(state: MovementModelState, context: MovementToken[]): MovementPrediction {
    const payload = state.payload as NgramPayload;
    const maxLen = Math.min(context.length, this.order - 1);
    for (let len = maxLen; len >= 0; len -= 1) {
      const key = contextKey(context.slice(context.length - len));
      const bucket = payload.contexts[key];
      if (!bucket) {
        continue;
      }
      const entries = Object.entries(bucket);
      const total = entries.reduce((sum, [, count]) => sum + count, 0);
      if (total === 0) {
        continue;
      }
      const candidates: MovementCandidate[] = entries
        .map(([token, count]) => ({ token, count, probability: count / total }))
        .sort((a, b) => (b.count - a.count) || (a.token < b.token ? -1 : a.token > b.token ? 1 : 0));
      const best = candidates[0]!;
      return { token: best.token, probability: best.probability, order: len, candidates };
    }
    return { token: null, probability: 0, order: 0, candidates: [] };
  }
}

// --- Model façade ------------------------------------------------------------

export type MovementGenerateOptions = {
  // Stop the rollout early if it repeats the same token this many times.
  maxRepeat?: number;
};

export class MovementModel {
  private constructor(
    private readonly backend: MovementModelBackend,
    private readonly state: MovementModelState,
  ) {}

  static train(
    dataset: MovementSequence[],
    backend: MovementModelBackend = new BackoffNgramMovementBackend(),
  ): MovementModel {
    return new MovementModel(backend, backend.train(dataset));
  }

  static fromState(state: MovementModelState, backend: MovementModelBackend): MovementModel {
    return new MovementModel(backend, state);
  }

  get modelState(): MovementModelState {
    return this.state;
  }

  get vocabulary(): MovementToken[] {
    return this.state.vocabulary;
  }

  predictNext(context: MovementToken[]): MovementPrediction {
    return this.backend.predict(this.state, context);
  }

  // Autoregressive rollout: repeat/generalize a recorded movement pattern by
  // predicting forward from a prefix. Uses a sliding window of the model order.
  generate(prefix: MovementToken[], steps: number, options: MovementGenerateOptions = {}): MovementToken[] {
    const maxRepeat = options.maxRepeat ?? Number.POSITIVE_INFINITY;
    const context = [...prefix];
    const generated: MovementToken[] = [];
    let repeatToken: MovementToken | null = null;
    let repeatCount = 0;
    for (let i = 0; i < steps; i += 1) {
      const window = context.slice(Math.max(0, context.length - (this.state.order - 1)));
      const prediction = this.predictNext(window);
      if (prediction.token === null) {
        break;
      }
      if (prediction.token === repeatToken) {
        repeatCount += 1;
        if (repeatCount >= maxRepeat) {
          break;
        }
      } else {
        repeatToken = prediction.token;
        repeatCount = 1;
      }
      generated.push(prediction.token);
      context.push(prediction.token);
    }
    return generated;
  }
}

// --- Generalization eval -----------------------------------------------------

export type MovementEvalOptions = {
  // Minimum context length before a position is scored (default 1: skip the
  // cold-start first token, measure genuine next-movement prediction).
  minContext?: number;
};

export type MovementEvalResult = {
  sequences: number;
  predictions: number;
  correct: number;
  accuracy: number;
  averageBackoffOrder: number;
};

// Measures top-1 next-movement accuracy on held-out (but related) sequences —
// the generalization signal for objective 2(d).
export function evaluateNextActionAccuracy(
  model: MovementModel,
  heldOut: MovementSequence[],
  options: MovementEvalOptions = {},
): MovementEvalResult {
  const minContext = Math.max(0, options.minContext ?? 1);
  let predictions = 0;
  let correct = 0;
  let orderSum = 0;
  for (const sequence of heldOut) {
    const tokens = movementSequenceTokens(sequence);
    for (let i = minContext; i < tokens.length; i += 1) {
      const prediction = model.predictNext(tokens.slice(0, i));
      if (prediction.token === null) {
        continue;
      }
      predictions += 1;
      orderSum += prediction.order;
      if (prediction.token === tokens[i]) {
        correct += 1;
      }
    }
  }
  return {
    sequences: heldOut.length,
    predictions,
    correct,
    accuracy: predictions === 0 ? 0 : correct / predictions,
    averageBackoffOrder: predictions === 0 ? 0 : orderSum / predictions,
  };
}
