import type { ReplayManifest } from "../capture/replay.js";
import type { TrajectoryAction, TrajectorySpan } from "../capture/trajectory.js";

/**
 * Local-movement learning model (standing objective #2, pieces c + d).
 *
 * This module provides an in-process, deterministic, cloud-testable pipeline
 * that turns recorded movement trajectories into a trained sequence model able
 * to (c) replay the recorded movements and (d) generalize to new-but-related
 * movements. The heavy on-device training path (`runner.ts` /
 * `execution-service.ts`) shells out to real tooling (mlx/axolotl) that cannot
 * run in the cloud; this module is the pluggable seam that DOES run anywhere so
 * the capture -> dataset -> train -> infer -> evaluate loop can be validated
 * with synthetic event streams and, when bee-agent runs locally, swapped for a
 * real small-model backend implementing {@link MovementModelBackend}.
 */

/**
 * A single movement token. Canonical form is `channel:verb:target`, e.g.
 * `device:tap:sendButton` or `keyboard:shortcut:cmd+c`. Tokens are the atomic
 * unit the model learns transitions over.
 */
export type MovementToken = string;

/** Sentinel emitted before a sequence's first real token. */
export const MOVEMENT_START = "<start>" as const;
/** Sentinel emitted after a sequence's last real token; terminates generation. */
export const MOVEMENT_END = "<end>" as const;

/** A named, tokenized sequence of movements — the model's dataset row. */
export type MovementSequence = {
  id: string;
  tokens: MovementToken[];
};

/** A next-token prediction with backoff transparency. */
export type MovementPrediction = {
  token: MovementToken;
  /** Empirical probability of `token` given the matched context. */
  probability: number;
  /** Length of the context actually used (after backoff). */
  order: number;
};

export type MovementTrainOptions = {
  /** Maximum context length. Defaults to the backend's configured order. */
  order?: number;
};

/**
 * Serializable trained-model artifact. Backends are free to add fields; the
 * `backend` discriminator lets a loader route to the right implementation.
 */
export type MovementModelState = {
  backend: string;
  version: 1;
  order: number;
  vocabulary: MovementToken[];
  sequenceCount: number;
  tokenCount: number;
  [extra: string]: unknown;
};

/**
 * Pluggable backend contract. A real on-device model (GGUF weights, an ONNX
 * runtime, etc.) implements the same three members; the rest of the pipeline
 * (generation, evaluation) is backend-agnostic and lives in free functions.
 */
export interface MovementModelBackend {
  readonly name: string;
  train(dataset: MovementSequence[], options?: MovementTrainOptions): MovementModelState;
  /**
   * Predict the most likely next token given a (possibly long) context.
   * Returns `undefined` only for an untrained/empty model.
   */
  predictNext(state: MovementModelState, context: MovementToken[]): MovementPrediction | undefined;
}

// ---------------------------------------------------------------------------
// Tokenization: TrajectoryAction / ReplayManifest -> MovementSequence
// ---------------------------------------------------------------------------

function sanitizeTokenPart(value: string): string {
  const cleaned = value.trim().toLowerCase().replace(/[\s:]+/g, "-").replace(/[^a-z0-9+_./-]/g, "");
  return cleaned.length > 0 ? cleaned : "-";
}

/**
 * Derive a canonical movement token from a recorded trajectory action. Prefers
 * structured `metadata` (gesture/target/direction) and falls back to the
 * human-readable summary so device, keyboard, and generic tool actions all map
 * to a stable `channel:verb:target` form.
 */
export function tokenizeMovementAction(action: TrajectoryAction): MovementToken {
  const metadata = (action.metadata ?? {}) as Record<string, unknown>;
  const channel = sanitizeTokenPart(action.tool || "action");

  const gesture = typeof metadata.gesture === "string" ? metadata.gesture : undefined;
  const verb = sanitizeTokenPart(gesture ?? firstWord(action.summary) ?? "act");

  const target =
    pickString(metadata.target) ??
    pickString(metadata.direction) ??
    pickString(metadata.valueSummary) ??
    lastWord(action.summary) ??
    "-";

  return `${channel}:${verb}:${sanitizeTokenPart(target)}`;
}

function pickString(value: unknown): string | undefined {
  return typeof value === "string" && value.trim().length > 0 ? value : undefined;
}

function firstWord(summary: string): string | undefined {
  const word = summary.trim().split(/\s+/)[0];
  return word && word.length > 0 ? word : undefined;
}

function lastWord(summary: string): string | undefined {
  const words = summary.trim().split(/\s+/);
  const word = words.at(-1);
  return word && word.length > 0 ? word : undefined;
}

/** Build one movement sequence from a trajectory span's actions (time-ordered). */
export function movementSequenceFromTrajectory(span: TrajectorySpan): MovementSequence {
  const tokens = [...span.actions]
    .sort((a, b) => a.ts - b.ts)
    .map((action) => tokenizeMovementAction(action));
  return { id: span.id, tokens };
}

/**
 * Extract per-trajectory movement sequences from a replay manifest, grouping
 * `action` timeline events by trajectory and preserving timeline order.
 */
export function movementSequencesFromReplayManifest(manifest: ReplayManifest): MovementSequence[] {
  const byTrajectory = new Map<string, MovementToken[]>();
  for (const trajectoryId of manifest.trajectoryIds) {
    byTrajectory.set(trajectoryId, []);
  }
  for (const event of manifest.events) {
    if (event.kind !== "action") {
      continue;
    }
    const token = tokenizeMovementAction({
      kind: "action",
      tool: event.tool,
      summary: event.summary,
      ts: event.ts,
    });
    const bucket = byTrajectory.get(event.trajectoryId) ?? [];
    bucket.push(token);
    byTrajectory.set(event.trajectoryId, bucket);
  }
  return [...byTrajectory.entries()]
    .filter(([, tokens]) => tokens.length > 0)
    .map(([id, tokens]) => ({ id, tokens }));
}

// ---------------------------------------------------------------------------
// Markov backend (deterministic, in-process reference implementation)
// ---------------------------------------------------------------------------

type TransitionCounts = Record<string, Record<MovementToken, number>>;

interface MarkovModelState extends MovementModelState {
  backend: "markov";
  /** contextKey -> { nextToken: count } for every order 0..order. */
  transitions: TransitionCounts;
}

const CONTEXT_SEPARATOR = "";

function contextKey(context: MovementToken[]): string {
  return context.join(CONTEXT_SEPARATOR);
}

/**
 * Order-N Markov backend with stupid-backoff. Learning a sequence once makes
 * its transitions deterministic (exact replay); an unseen high-order context
 * backs off to shorter contexts, yielding plausible *related* movements —
 * generalization without any randomness, so results are reproducible in CI.
 */
export class MarkovMovementBackend implements MovementModelBackend {
  readonly name = "markov";

  constructor(private readonly defaultOrder = 2) {
    if (!Number.isInteger(defaultOrder) || defaultOrder < 1) {
      throw new Error(`Markov order must be a positive integer, got ${defaultOrder}`);
    }
  }

  train(dataset: MovementSequence[], options: MovementTrainOptions = {}): MovementModelState {
    const order = options.order ?? this.defaultOrder;
    if (!Number.isInteger(order) || order < 1) {
      throw new Error(`Markov order must be a positive integer, got ${order}`);
    }

    const transitions: TransitionCounts = {};
    const vocabulary = new Set<MovementToken>();
    let tokenCount = 0;

    for (const sequence of dataset) {
      const padded = [
        ...Array.from({ length: order }, () => MOVEMENT_START),
        ...sequence.tokens,
        MOVEMENT_END,
      ];
      for (const token of sequence.tokens) {
        vocabulary.add(token);
        tokenCount += 1;
      }
      for (let i = order; i < padded.length; i += 1) {
        const next = padded[i]!;
        // Record this transition at every context length 0..order (backoff).
        for (let k = 0; k <= order; k += 1) {
          const context = padded.slice(i - k, i);
          const key = contextKey(context);
          const bucket = (transitions[key] ??= {});
          bucket[next] = (bucket[next] ?? 0) + 1;
        }
      }
    }

    const state: MarkovModelState = {
      backend: "markov",
      version: 1,
      order,
      vocabulary: [...vocabulary].sort(),
      sequenceCount: dataset.length,
      tokenCount,
      transitions,
    };
    return state;
  }

  predictNext(state: MovementModelState, context: MovementToken[]): MovementPrediction | undefined {
    const model = state as MarkovModelState;
    const order = model.order;
    for (let k = Math.min(order, context.length); k >= 0; k -= 1) {
      const truncated = context.slice(context.length - k);
      const bucket = model.transitions[contextKey(truncated)];
      if (!bucket) {
        continue;
      }
      const best = argmaxToken(bucket);
      if (best) {
        return { token: best.token, probability: best.probability, order: k };
      }
    }
    return undefined;
  }
}

function argmaxToken(bucket: Record<MovementToken, number>):
  | { token: MovementToken; probability: number }
  | undefined {
  let total = 0;
  let bestToken: MovementToken | undefined;
  let bestCount = -1;
  // Deterministic: highest count wins, ties broken by lexicographically
  // smallest token so a state serialized in any key order predicts identically.
  for (const [token, count] of Object.entries(bucket)) {
    total += count;
    if (count > bestCount || (count === bestCount && (bestToken === undefined || token < bestToken))) {
      bestCount = count;
      bestToken = token;
    }
  }
  if (bestToken === undefined || total === 0) {
    return undefined;
  }
  return { token: bestToken, probability: bestCount / total };
}

// ---------------------------------------------------------------------------
// Generation + evaluation (backend-agnostic)
// ---------------------------------------------------------------------------

export type GenerateMovementsOptions = {
  /** Hard cap on generated tokens (excludes the terminating sentinel). */
  maxSteps?: number;
  /** Seed context; defaults to the START padding the model was trained with. */
  prompt?: MovementToken[];
};

/**
 * Greedily roll out a movement sequence from a trained model. Stops at
 * {@link MOVEMENT_END}, when the model has no prediction, or at `maxSteps`.
 * With a fully-recorded prompt this reproduces the recording (replay); with a
 * novel prompt it produces a generalized continuation.
 */
export function generateMovements(
  backend: MovementModelBackend,
  state: MovementModelState,
  options: GenerateMovementsOptions = {},
): MovementToken[] {
  const maxSteps = options.maxSteps ?? 64;
  const context: MovementToken[] = [
    ...Array.from({ length: state.order }, () => MOVEMENT_START),
    ...(options.prompt ?? []),
  ];
  const generated: MovementToken[] = [];
  for (let step = 0; step < maxSteps; step += 1) {
    const prediction = backend.predictNext(state, context);
    if (!prediction || prediction.token === MOVEMENT_END) {
      break;
    }
    generated.push(prediction.token);
    context.push(prediction.token);
  }
  return generated;
}

export type ReplayFidelityReport = {
  sequences: number;
  steps: number;
  correct: number;
  /** Top-1 next-token accuracy across all held-out steps (0..1). */
  accuracy: number;
  perSequence: Array<{ id: string; steps: number; correct: number; accuracy: number }>;
};

/**
 * Teacher-forced generalization eval: for each held-out sequence, walk its real
 * tokens and check whether the model's top-1 prediction matches the actual next
 * token at each step. Measures replay fidelity on recorded sequences and
 * generalization on unseen-but-related ones.
 */
export function evaluateReplayFidelity(
  backend: MovementModelBackend,
  state: MovementModelState,
  heldOut: MovementSequence[],
): ReplayFidelityReport {
  let totalSteps = 0;
  let totalCorrect = 0;
  const perSequence: ReplayFidelityReport["perSequence"] = [];

  for (const sequence of heldOut) {
    const context: MovementToken[] = Array.from({ length: state.order }, () => MOVEMENT_START);
    const targets = [...sequence.tokens, MOVEMENT_END];
    let steps = 0;
    let correct = 0;
    for (const expected of targets) {
      const prediction = backend.predictNext(state, context);
      steps += 1;
      if (prediction?.token === expected) {
        correct += 1;
      }
      context.push(expected);
    }
    totalSteps += steps;
    totalCorrect += correct;
    perSequence.push({
      id: sequence.id,
      steps,
      correct,
      accuracy: steps === 0 ? 0 : correct / steps,
    });
  }

  return {
    sequences: heldOut.length,
    steps: totalSteps,
    correct: totalCorrect,
    accuracy: totalSteps === 0 ? 0 : totalCorrect / totalSteps,
    perSequence,
  };
}

// ---------------------------------------------------------------------------
// Trainer: dataset assembly + deterministic holdout split
// ---------------------------------------------------------------------------

export type TrainFromTrajectoriesResult = {
  state: MovementModelState;
  train: MovementSequence[];
  holdout: MovementSequence[];
  evaluation: ReplayFidelityReport;
};

/**
 * Convenience orchestrator: assemble sequences from recorded trajectories,
 * deterministically hold out every Nth sequence, train the backend on the rest,
 * and report replay fidelity on the holdout. `holdoutEvery = 0` trains on
 * everything and evaluates on the training set (pure replay check).
 */
export class MovementModelTrainer {
  constructor(private readonly backend: MovementModelBackend) {}

  trainFromTrajectories(
    spans: TrajectorySpan[],
    options: { holdoutEvery?: number; order?: number } = {},
  ): TrainFromTrajectoriesResult {
    const sequences = spans
      .map((span) => movementSequenceFromTrajectory(span))
      .filter((sequence) => sequence.tokens.length > 0);
    return this.trainFromSequences(sequences, options);
  }

  trainFromSequences(
    sequences: MovementSequence[],
    options: { holdoutEvery?: number; order?: number } = {},
  ): TrainFromTrajectoriesResult {
    const holdoutEvery = options.holdoutEvery ?? 0;
    const train: MovementSequence[] = [];
    const holdout: MovementSequence[] = [];
    sequences.forEach((sequence, index) => {
      if (holdoutEvery > 0 && (index + 1) % holdoutEvery === 0) {
        holdout.push(sequence);
      } else {
        train.push(sequence);
      }
    });

    const state = this.backend.train(train, { order: options.order });
    const evalSet = holdout.length > 0 ? holdout : train;
    const evaluation = evaluateReplayFidelity(this.backend, state, evalSet);
    return { state, train, holdout, evaluation };
  }
}

// ---------------------------------------------------------------------------
// Synthetic movement generator (deterministic, no RNG) for cloud validation
// ---------------------------------------------------------------------------

export type SyntheticMovementTemplate = {
  /** Ordered choices per slot; the generator walks combinations deterministically. */
  channel: string;
  steps: Array<{ verb: string; targets: string[] }>;
};

/**
 * Produce related-but-distinct movement sequences from a template by rotating
 * each step's target choice via a mixing index. Fully deterministic (seeded by
 * `count`/index arithmetic, never `Math.random`), so it can seed reproducible
 * generalization tests: models trained on some variants should predict the
 * held-out variants' shared structure.
 */
export function synthesizeMovementSequences(
  template: SyntheticMovementTemplate,
  count: number,
): MovementSequence[] {
  if (count < 0) {
    throw new Error(`count must be non-negative, got ${count}`);
  }
  const sequences: MovementSequence[] = [];
  for (let variant = 0; variant < count; variant += 1) {
    const tokens = template.steps.map((step, stepIndex) => {
      const choices = step.targets.length > 0 ? step.targets : ["-"];
      // Deterministic rotation: different variants pick different targets while
      // preserving channel/verb structure, so the shared skeleton is learnable.
      const target = choices[(variant + stepIndex) % choices.length]!;
      return `${sanitizeTokenPart(template.channel)}:${sanitizeTokenPart(step.verb)}:${sanitizeTokenPart(target)}`;
    });
    sequences.push({ id: `synthetic-${variant}`, tokens });
  }
  return sequences;
}
