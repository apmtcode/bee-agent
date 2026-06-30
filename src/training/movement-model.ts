/**
 * In-process movement model for the local-movement learning subsystem.
 *
 * The existing training runner ({@link LocalAppleSiliconTrainingRunner})
 * produces shell plans that shell out to external Python (mlx/axolotl) on the
 * user's machine. That path cannot run — let alone be tested — in the cloud,
 * so it leaves objective 2(d) "post-train a local model that repeats recorded
 * movements" and 2(e) "generalize to new but related movements" unimplemented
 * and unverifiable here.
 *
 * This module fills that gap with a small, fully in-process, deterministic
 * model that is trainable and runnable anywhere (cloud, CI, or a laptop with
 * no GPU). It is pluggable: {@link MovementModelBackend} is the seam, and
 * {@link NgramMovementBackend} is the reference/default backend (an order-k
 * Markov model with Katz-style backoff). A heavier on-device backend (a real
 * small neural policy) can implement the same interface later without touching
 * call sites or the eval harness.
 *
 * Nothing here touches the real OS — it operates purely on movement sequences
 * (recorded {@link TrajectorySpan} actions, or synthetic ones), so it is safe
 * to train and evaluate in the cloud against simulated event streams.
 */

import type { TrajectoryAction, TrajectorySpan, ReviewedTrajectoryAction } from "../capture/trajectory.js";

/** Sentinel prepended to every training sequence so beginnings are learnable. */
export const MOVEMENT_START_TOKEN = "<start>";

/** Sentinel appended to every training sequence so generation can terminate. */
export const MOVEMENT_END_TOKEN = "<end>";

/** A discrete, replayable movement step distilled from a trajectory action. */
export type MovementStep = {
  tool: string;
  gesture?: string;
  direction?: string;
  /** A coarse target label (button/field/region) — already redaction-safe. */
  target?: string;
};

/** A movement step encoded as a single stable vocabulary token. */
export type MovementToken = string;

/** One ordered run of movements (e.g. all actions in one trajectory span). */
export type MovementSequence = MovementToken[];

const FIELD_PLACEHOLDER = "·";
const FIELD_SEPARATOR = "|";

/** Encode a structured step into a single deterministic vocabulary token. */
export function tokenizeStep(step: MovementStep): MovementToken {
  return [step.tool, step.gesture, step.direction, step.target]
    .map((field) => normalizeField(field))
    .join(FIELD_SEPARATOR);
}

/** Decode a token back into its structured step (inverse of tokenizeStep). */
export function parseToken(token: MovementToken): MovementStep {
  const [tool, gesture, direction, target] = token.split(FIELD_SEPARATOR);
  const step: MovementStep = { tool: tool ?? "" };
  if (gesture && gesture !== FIELD_PLACEHOLDER) step.gesture = gesture;
  if (direction && direction !== FIELD_PLACEHOLDER) step.direction = direction;
  if (target && target !== FIELD_PLACEHOLDER) step.target = target;
  return step;
}

function normalizeField(value: string | undefined): string {
  const trimmed = value?.trim();
  if (!trimmed) return FIELD_PLACEHOLDER;
  // Keep the vocabulary stable and separator-safe.
  return trimmed.toLowerCase().replaceAll(FIELD_SEPARATOR, "/");
}

/** Distill a recorded action into a structured movement step. */
export function stepFromAction(action: TrajectoryAction | ReviewedTrajectoryAction): MovementStep {
  const metadata = "metadata" in action ? (action.metadata ?? {}) : {};
  const gesture = readString(metadata.gesture);
  const direction = readString(metadata.direction);
  const target = readString(metadata.target) ?? readString(metadata.valueSummary);
  const step: MovementStep = { tool: action.tool };
  if (gesture) step.gesture = gesture;
  if (direction) step.direction = direction;
  if (target) step.target = target;
  return step;
}

/** Extract an ordered movement sequence (tokens) from a trajectory span. */
export function sequenceFromSpan(span: TrajectorySpan): MovementSequence {
  return [...span.actions]
    .sort((a, b) => a.ts - b.ts)
    .map((action) => tokenizeStep(stepFromAction(action)));
}

function readString(value: unknown): string | undefined {
  return typeof value === "string" && value.trim().length > 0 ? value.trim() : undefined;
}

// ---------------------------------------------------------------------------
// Pluggable backend seam
// ---------------------------------------------------------------------------

export type MovementTrainingOptions = {
  /** Markov order (context window). Higher = more literal replay, less generalization. */
  order?: number;
};

/** A single next-step prediction with its full candidate distribution. */
export type MovementPrediction = {
  /** Most likely next token (deterministic argmax; ties broken lexically). */
  token: MovementToken;
  probability: number;
  /** Candidate tokens sorted by descending probability then token. */
  distribution: Array<{ token: MovementToken; probability: number }>;
  /** Context length actually used after backoff (0 = unigram prior). */
  contextOrderUsed: number;
};

export type MovementModelSnapshot = {
  backend: string;
  order: number;
  /** context-key → { nextToken → count }. Empty key "" is the unigram table. */
  transitions: Record<string, Record<MovementToken, number>>;
};

/** A trained, serializable movement policy. */
export interface MovementModel {
  readonly backend: string;
  readonly order: number;
  readonly vocabularySize: number;
  /** Predict the next movement given the trailing context. */
  predict(context: MovementSequence): MovementPrediction;
  /** Roll out up to `steps` movements from a seed, stopping at the end sentinel. */
  generate(seed: MovementSequence, steps: number): MovementSequence;
  serialize(): MovementModelSnapshot;
}

/** The pluggable training backend. Implement this to swap in a real model. */
export interface MovementModelBackend {
  readonly name: string;
  train(dataset: MovementSequence[], options?: MovementTrainingOptions): MovementModel;
  restore(snapshot: MovementModelSnapshot): MovementModel;
}

// ---------------------------------------------------------------------------
// Reference backend: order-k Markov with Katz-style backoff
// ---------------------------------------------------------------------------

const DEFAULT_ORDER = 2;

/**
 * Deterministic n-gram backend. Trainable on movement sequences, runs entirely
 * in-process, and generalizes to unseen contexts by backing off to shorter
 * contexts (and ultimately the unigram prior) — so a novel-but-related
 * movement still yields a plausible next step instead of nothing.
 */
export class NgramMovementBackend implements MovementModelBackend {
  readonly name = "ngram-markov";

  train(dataset: MovementSequence[], options: MovementTrainingOptions = {}): MovementModel {
    const order = Math.max(1, Math.floor(options.order ?? DEFAULT_ORDER));
    const transitions = new Map<string, Map<MovementToken, number>>();

    for (const rawSequence of dataset) {
      // Anchor the start (so beginnings are learnable) and terminate the end
      // (so the model learns when movement stops).
      const sequence = [MOVEMENT_START_TOKEN, ...rawSequence, MOVEMENT_END_TOKEN];
      // i starts at 1: START is only ever a context, never a predicted next.
      for (let i = 1; i < sequence.length; i += 1) {
        const next = sequence[i]!;
        // Record this transition for every context length 0..order.
        for (let k = 0; k <= order; k += 1) {
          if (i - k < 0) break;
          const contextKey = sequence.slice(i - k, i).join(" ");
          bump(transitions, contextKey, next);
        }
      }
    }

    return new NgramMovementModel(this.name, order, transitions);
  }

  restore(snapshot: MovementModelSnapshot): MovementModel {
    const transitions = new Map<string, Map<MovementToken, number>>();
    for (const [contextKey, counts] of Object.entries(snapshot.transitions)) {
      transitions.set(contextKey, new Map(Object.entries(counts)));
    }
    return new NgramMovementModel(snapshot.backend, snapshot.order, transitions);
  }
}

function bump(
  transitions: Map<string, Map<MovementToken, number>>,
  contextKey: string,
  next: MovementToken,
): void {
  let counts = transitions.get(contextKey);
  if (!counts) {
    counts = new Map();
    transitions.set(contextKey, counts);
  }
  counts.set(next, (counts.get(next) ?? 0) + 1);
}

class NgramMovementModel implements MovementModel {
  readonly vocabularySize: number;

  constructor(
    readonly backend: string,
    readonly order: number,
    private readonly transitions: Map<string, Map<MovementToken, number>>,
  ) {
    const vocab = new Set<MovementToken>();
    for (const counts of transitions.values()) {
      for (const token of counts.keys()) vocab.add(token);
    }
    vocab.delete(MOVEMENT_START_TOKEN);
    vocab.delete(MOVEMENT_END_TOKEN);
    this.vocabularySize = vocab.size;
  }

  predict(context: MovementSequence): MovementPrediction {
    // Katz-style backoff: longest matching context wins; shrink until a hit.
    const maxContext = Math.min(this.order, context.length);
    for (let k = maxContext; k >= 0; k -= 1) {
      const contextKey = context.slice(context.length - k).join(" ");
      const counts = this.transitions.get(contextKey);
      if (counts && counts.size > 0) {
        return this.toPrediction(counts, k);
      }
    }
    // Untrained model: nothing to predict.
    return { token: MOVEMENT_END_TOKEN, probability: 1, distribution: [], contextOrderUsed: 0 };
  }

  generate(seed: MovementSequence, steps: number): MovementSequence {
    const out: MovementSequence = [];
    // Anchor with START so an empty seed still rolls out a plausible beginning.
    const context = [MOVEMENT_START_TOKEN, ...seed];
    for (let i = 0; i < steps; i += 1) {
      const prediction = this.predict(context);
      if (prediction.token === MOVEMENT_END_TOKEN || prediction.distribution.length === 0) {
        break;
      }
      out.push(prediction.token);
      context.push(prediction.token);
    }
    return out;
  }

  serialize(): MovementModelSnapshot {
    const transitions: Record<string, Record<MovementToken, number>> = {};
    for (const [contextKey, counts] of this.transitions) {
      transitions[contextKey] = Object.fromEntries(counts);
    }
    return { backend: this.backend, order: this.order, transitions };
  }

  private toPrediction(counts: Map<MovementToken, number>, contextOrderUsed: number): MovementPrediction {
    let total = 0;
    for (const count of counts.values()) total += count;
    const distribution = [...counts.entries()]
      .map(([token, count]) => ({ token, probability: count / total }))
      // Deterministic ordering: probability desc, then token asc for stable ties.
      .sort((a, b) => b.probability - a.probability || (a.token < b.token ? -1 : 1));
    const best = distribution[0]!;
    return { token: best.token, probability: best.probability, distribution, contextOrderUsed };
  }
}

/** Convenience: train the default reference backend in one call. */
export function trainMovementModel(
  dataset: MovementSequence[],
  options?: MovementTrainingOptions,
): MovementModel {
  return new NgramMovementBackend().train(dataset, options);
}

// ---------------------------------------------------------------------------
// Generalization / replay-fidelity eval harness
// ---------------------------------------------------------------------------

export type MovementFidelityReport = {
  sequences: number;
  predictions: number;
  /** Next-step top-1 accuracy across all held-out positions. */
  nextStepAccuracy: number;
  /** Fraction of held-out sequences reproduced exactly by greedy rollout. */
  fullSequenceAccuracy: number;
  /** Average backoff context length used — a proxy for memorization vs. generalization. */
  averageContextOrderUsed: number;
};

/**
 * Measure how well a trained model repeats and generalizes movements on a
 * held-out set. `nextStepAccuracy` checks the next-step prediction at each
 * position; `fullSequenceAccuracy` greedily rolls out from each sequence's
 * first token and checks exact reproduction.
 */
export function evaluateMovementFidelity(
  model: MovementModel,
  heldOut: MovementSequence[],
): MovementFidelityReport {
  let predictions = 0;
  let correctNext = 0;
  let contextOrderSum = 0;
  let exactSequences = 0;
  let scoredSequences = 0;

  for (const sequence of heldOut) {
    if (sequence.length === 0) continue;
    scoredSequences += 1;

    for (let i = 0; i < sequence.length; i += 1) {
      // Anchor with START so the first movement is predicted from a real
      // context rather than the ambiguous unigram prior.
      const prediction = model.predict([MOVEMENT_START_TOKEN, ...sequence.slice(0, i)]);
      predictions += 1;
      contextOrderSum += prediction.contextOrderUsed;
      if (prediction.token === sequence[i]) correctNext += 1;
    }

    // Reproduce the whole movement from scratch (empty seed -> START anchor).
    const reproduced = model.generate([], sequence.length);
    if (reproduced.length === sequence.length && reproduced.every((token, idx) => token === sequence[idx])) {
      exactSequences += 1;
    }
  }

  return {
    sequences: scoredSequences,
    predictions,
    nextStepAccuracy: predictions === 0 ? 0 : correctNext / predictions,
    fullSequenceAccuracy: scoredSequences === 0 ? 0 : exactSequences / scoredSequences,
    averageContextOrderUsed: predictions === 0 ? 0 : contextOrderSum / predictions,
  };
}

// ---------------------------------------------------------------------------
// Synthetic event-stream generator (no real OS input required)
// ---------------------------------------------------------------------------

export type SyntheticMovementSpec = {
  /** Named workflows; each is a deterministic ordered list of steps. */
  workflows: Array<{ name: string; steps: MovementStep[] }>;
  /** How many sequences to emit (cycled across workflows). */
  count: number;
  /**
   * Inject a deterministic, related variation every `variantEvery` sequences
   * (a swapped target on the final step) so eval sets contain novel-but-related
   * movements that exercise backoff generalization rather than pure memorization.
   */
  variantEvery?: number;
};

/**
 * Produce deterministic synthetic movement sequences (no randomness, so tests
 * are stable) that simulate recorded local-computer workflows. Used to validate
 * the capture→dataset→train→replay round-trip without any real OS input.
 */
export function generateSyntheticMovementSequences(spec: SyntheticMovementSpec): MovementSequence[] {
  const sequences: MovementSequence[] = [];
  const variantEvery = spec.variantEvery && spec.variantEvery > 0 ? spec.variantEvery : 0;

  for (let i = 0; i < spec.count; i += 1) {
    const workflow = spec.workflows[i % spec.workflows.length];
    if (!workflow || workflow.steps.length === 0) continue;
    const isVariant = variantEvery > 0 && (i + 1) % variantEvery === 0;
    const steps = workflow.steps.map((step, idx) => {
      if (isVariant && idx === workflow.steps.length - 1) {
        return { ...step, target: `${step.target ?? "target"}-alt` };
      }
      return step;
    });
    sequences.push(steps.map((step) => tokenizeStep(step)));
  }

  return sequences;
}
