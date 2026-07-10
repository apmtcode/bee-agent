import type { ReplayTimelineEvent } from "../capture/replay.js";
import type { TrajectoryAction, TrajectorySpan } from "../capture/trajectory.js";

/**
 * In-process, cloud-runnable "train + infer" pipeline for the local-movement
 * learning subsystem (standing objective #2).
 *
 * The capture/replay/export code produces *sequences of movement events*. This
 * module turns those into a trainable dataset, defines a pluggable model
 * backend, and ships a deterministic backend (a variable-order Markov model
 * with structured backoff) so the whole pipeline — dataset → train → predict →
 * generalize — can be exercised and validated in the cloud with no real OS
 * input and no external ML runtime.
 *
 * A real on-device small model plugs in behind {@link MovementModelBackend}:
 * the mock is deterministic (argmax, string tie-breaks — no RNG) so it doubles
 * as the CI reference implementation.
 */

/** Placeholder used when a movement field is absent or abstracted away. */
export const MOVEMENT_WILDCARD = "*";

/** A single normalized movement — the atomic unit the model learns to emit. */
export type MovementStep = {
  /** Tool/surface the movement happened on (e.g. "device", "os", "browser"). */
  tool: string;
  /** Gesture/interaction kind (e.g. "tap", "type", "scroll", "click"). */
  gesture: string;
  /** Concrete target the movement acted on (button label, field, url, …). */
  target: string;
  /** Optional direction for directional gestures (swipe/scroll). */
  direction?: string;
};

/** An ordered movement sequence, optionally labelled by its source trajectory. */
export type MovementSequence = {
  /** Identifier of the trajectory/session this sequence was derived from. */
  sourceId: string;
  steps: MovementStep[];
};

export type MovementDataset = {
  version: 1;
  sequences: MovementSequence[];
};

/**
 * A prediction for the movement that should follow a given context.
 * `level` records how the prediction was reached, which is the signal the
 * generalization eval uses to distinguish memorized replay from generalization.
 */
export type MovementPrediction = {
  step: MovementStep;
  /** P(step | context) at the matched order, in [0, 1]. */
  confidence: number;
  /** Length of the context actually matched (0 = unconditional prior). */
  matchedOrder: number;
  /**
   * How the match was found:
   * - "specific": exact prior-step tokens matched a learned transition.
   * - "shape": exact tokens were unseen; matched on abstracted (target-less)
   *   context — this is the generalization path to novel-but-related movements.
   * - "prior": no context matched; fell back to the global most-frequent step.
   */
  level: "specific" | "shape" | "prior";
};

export interface TrainedMovementModel {
  readonly backendId: string;
  /** Highest context order the model was trained with. */
  readonly order: number;
  /** Predict the next movement given the movements observed so far. */
  predictNext(context: MovementStep[]): MovementPrediction | undefined;
  /** Autoregressively generate a continuation from a seed. */
  generate(seed: MovementStep[], maxLength: number): MovementStep[];
}

export interface MovementModelBackend {
  readonly id: string;
  train(dataset: MovementDataset, options?: MovementTrainOptions): TrainedMovementModel;
}

export type MovementTrainOptions = {
  /** Maximum Markov context order (default 2). Higher = more memorization. */
  maxOrder?: number;
};

// ---------------------------------------------------------------------------
// Tokenization
// ---------------------------------------------------------------------------

function normalizeField(value: string | undefined): string {
  const trimmed = value?.trim();
  return trimmed && trimmed.length > 0 ? trimmed.toLowerCase() : MOVEMENT_WILDCARD;
}

/** Canonical, fully-specific token for a step (used for exact-match contexts). */
export function movementStepToken(step: MovementStep): string {
  return [
    normalizeField(step.tool),
    normalizeField(step.gesture),
    normalizeField(step.target),
    normalizeField(step.direction),
  ].join("|");
}

/**
 * Abstracted token: the concrete target is dropped. Two movements with the same
 * tool+gesture+direction but different targets share a shape — this is what lets
 * the model generalize to a new-but-related movement it never saw verbatim.
 */
export function movementStepShape(step: MovementStep): string {
  return [
    normalizeField(step.tool),
    normalizeField(step.gesture),
    MOVEMENT_WILDCARD,
    normalizeField(step.direction),
  ].join("|");
}

/** Derive a movement step from a captured trajectory action. */
export function movementStepFromAction(action: TrajectoryAction): MovementStep {
  const metadata = action.metadata ?? {};
  const gesture = typeof metadata.gesture === "string" ? metadata.gesture : action.tool;
  const target =
    typeof metadata.target === "string"
      ? metadata.target
      : typeof metadata.valueSummary === "string"
        ? metadata.valueSummary
        : action.summary;
  const direction = typeof metadata.direction === "string" ? metadata.direction : undefined;
  return {
    tool: action.tool,
    gesture,
    target,
    ...(direction ? { direction } : {}),
  };
}

/** Derive an ordered movement sequence from a single trajectory span. */
export function movementSequenceFromTrajectory(trajectory: TrajectorySpan): MovementSequence {
  const steps = [...trajectory.actions]
    .sort((a, b) => a.ts - b.ts)
    .map((action) => movementStepFromAction(action));
  return { sourceId: trajectory.id, steps };
}

/** Derive a movement sequence from a replay manifest's timeline events. */
export function movementSequenceFromReplayEvents(
  sourceId: string,
  events: ReplayTimelineEvent[],
): MovementSequence {
  const steps = events
    .filter((event): event is Extract<ReplayTimelineEvent, { kind: "action" }> => event.kind === "action")
    .sort((a, b) => a.ts - b.ts)
    .map((event) => ({ tool: event.tool, gesture: event.tool, target: event.summary }));
  return { sourceId, steps };
}

/** Assemble a dataset from captured trajectories. */
export function buildMovementDataset(trajectories: TrajectorySpan[]): MovementDataset {
  return {
    version: 1,
    sequences: trajectories
      .map((trajectory) => movementSequenceFromTrajectory(trajectory))
      .filter((sequence) => sequence.steps.length > 0),
  };
}

// ---------------------------------------------------------------------------
// Deterministic Markov backend (the mock/reference local model)
// ---------------------------------------------------------------------------

type TransitionCounts = Map<string, Map<string, number>>;

function incrementTransition(counts: TransitionCounts, contextKey: string, token: string): void {
  let inner = counts.get(contextKey);
  if (!inner) {
    inner = new Map();
    counts.set(contextKey, inner);
  }
  inner.set(token, (inner.get(token) ?? 0) + 1);
}

/**
 * Deterministic argmax over a count map. Ties break on the lexicographically
 * smallest token so the same dataset always trains the same model (required for
 * reproducible eval and cached workflow runs).
 */
function argmax(counts: Map<string, number>): { token: string; count: number; total: number } | undefined {
  let bestToken: string | undefined;
  let bestCount = -1;
  let total = 0;
  for (const [token, count] of counts) {
    total += count;
    if (count > bestCount || (count === bestCount && bestToken !== undefined && token < bestToken)) {
      bestToken = token;
      bestCount = count;
    }
  }
  if (bestToken === undefined) {
    return undefined;
  }
  return { token: bestToken, count: bestCount, total };
}

class MarkovMovementModel implements TrainedMovementModel {
  readonly backendId = "markov";

  constructor(
    readonly order: number,
    private readonly specific: TransitionCounts,
    private readonly shape: TransitionCounts,
    private readonly prior: Map<string, number>,
    private readonly tokenToStep: Map<string, MovementStep>,
  ) {}

  predictNext(context: MovementStep[]): MovementPrediction | undefined {
    // 1. Longest exact (target-aware) context wins — this replays verbatim.
    for (let k = Math.min(this.order, context.length); k >= 1; k -= 1) {
      const key = contextKey(context, k, movementStepToken);
      const best = this.specific.get(key) ? argmax(this.specific.get(key)!) : undefined;
      if (best) {
        return this.materialize(best, k, "specific");
      }
    }
    // 2. Back off to abstracted (target-less) context — generalization path.
    for (let k = Math.min(this.order, context.length); k >= 1; k -= 1) {
      const key = contextKey(context, k, movementStepShape);
      const best = this.shape.get(key) ? argmax(this.shape.get(key)!) : undefined;
      if (best) {
        return this.materialize(best, k, "shape");
      }
    }
    // 3. Global prior — the most frequent movement overall.
    const best = argmax(this.prior);
    if (best) {
      return this.materialize(best, 0, "prior");
    }
    return undefined;
  }

  generate(seed: MovementStep[], maxLength: number): MovementStep[] {
    const generated: MovementStep[] = [];
    const context = [...seed];
    for (let i = 0; i < maxLength; i += 1) {
      const prediction = this.predictNext(context);
      if (!prediction) {
        break;
      }
      generated.push(prediction.step);
      context.push(prediction.step);
    }
    return generated;
  }

  private materialize(
    best: { token: string; count: number; total: number },
    matchedOrder: number,
    level: MovementPrediction["level"],
  ): MovementPrediction {
    const step = this.tokenToStep.get(best.token) ?? parseToken(best.token);
    return {
      step,
      confidence: best.total > 0 ? best.count / best.total : 0,
      matchedOrder,
      level,
    };
  }
}

function contextKey(context: MovementStep[], k: number, tokenizer: (step: MovementStep) => string): string {
  return context
    .slice(context.length - k)
    .map((step) => tokenizer(step))
    .join(">");
}

function parseToken(token: string): MovementStep {
  const [tool = MOVEMENT_WILDCARD, gesture = MOVEMENT_WILDCARD, target = MOVEMENT_WILDCARD, direction = MOVEMENT_WILDCARD] =
    token.split("|");
  return {
    tool,
    gesture,
    target,
    ...(direction && direction !== MOVEMENT_WILDCARD ? { direction } : {}),
  };
}

export class MarkovMovementBackend implements MovementModelBackend {
  readonly id = "markov";

  train(dataset: MovementDataset, options: MovementTrainOptions = {}): TrainedMovementModel {
    const order = Math.max(1, options.maxOrder ?? 2);
    const specific: TransitionCounts = new Map();
    const shape: TransitionCounts = new Map();
    const prior = new Map<string, number>();
    const tokenToStep = new Map<string, MovementStep>();

    for (const sequence of dataset.sequences) {
      const steps = sequence.steps;
      for (let i = 0; i < steps.length; i += 1) {
        const step = steps[i];
        const token = movementStepToken(step);
        if (!tokenToStep.has(token)) {
          tokenToStep.set(token, step);
        }
        prior.set(token, (prior.get(token) ?? 0) + 1);
        for (let k = 1; k <= order && i - k >= 0; k += 1) {
          const priorSteps = steps.slice(i - k, i);
          incrementTransition(specific, priorSteps.map(movementStepToken).join(">"), token);
          incrementTransition(shape, priorSteps.map(movementStepShape).join(">"), token);
        }
      }
    }

    return new MarkovMovementModel(order, specific, shape, prior, tokenToStep);
  }
}

// ---------------------------------------------------------------------------
// Generalization eval harness
// ---------------------------------------------------------------------------

export type MovementEvalMetrics = {
  /** Positions evaluated (every step with at least one preceding step). */
  predictions: number;
  /** Fraction where a prediction was produced at all. */
  coverage: number;
  /** Exact next-step token accuracy (memorized-replay fidelity). */
  exactAccuracy: number;
  /** Accuracy up to the abstracted shape (tool+gesture+direction). */
  shapeAccuracy: number;
  /**
   * Generalization rate: among correctly-shaped predictions reached *without*
   * an exact-context match (shape/prior levels), how often the shape was right.
   * This measures repeating a related movement, not the memorized one.
   */
  generalizationRate: number;
  byLevel: Record<MovementPrediction["level"], number>;
};

/**
 * Score a trained model against held-out sequences via teacher-forced next-step
 * prediction. Held-out sequences should be *related but unseen* (e.g. same
 * templates with substituted targets) to measure generalization rather than
 * recall.
 */
export function evaluateMovementModel(
  model: TrainedMovementModel,
  sequences: MovementSequence[],
): MovementEvalMetrics {
  let predictions = 0;
  let covered = 0;
  let exact = 0;
  let shapeCorrect = 0;
  let generalizedOpportunities = 0;
  let generalizedCorrect = 0;
  const byLevel: Record<MovementPrediction["level"], number> = { specific: 0, shape: 0, prior: 0 };

  for (const sequence of sequences) {
    const steps = sequence.steps;
    for (let i = 1; i < steps.length; i += 1) {
      predictions += 1;
      const prediction = model.predictNext(steps.slice(0, i));
      if (!prediction) {
        continue;
      }
      covered += 1;
      byLevel[prediction.level] += 1;
      const actual = steps[i];
      const exactHit = movementStepToken(prediction.step) === movementStepToken(actual);
      const shapeHit = movementStepShape(prediction.step) === movementStepShape(actual);
      if (exactHit) {
        exact += 1;
      }
      if (shapeHit) {
        shapeCorrect += 1;
      }
      if (prediction.level !== "specific") {
        generalizedOpportunities += 1;
        if (shapeHit) {
          generalizedCorrect += 1;
        }
      }
    }
  }

  return {
    predictions,
    coverage: predictions > 0 ? covered / predictions : 0,
    exactAccuracy: predictions > 0 ? exact / predictions : 0,
    shapeAccuracy: predictions > 0 ? shapeCorrect / predictions : 0,
    generalizationRate: generalizedOpportunities > 0 ? generalizedCorrect / generalizedOpportunities : 0,
    byLevel,
  };
}

// ---------------------------------------------------------------------------
// Synthetic movement-stream generator (validates the pipeline without real OS input)
// ---------------------------------------------------------------------------

export type MovementTemplate = {
  id: string;
  /** Ordered steps with a `{target}` placeholder substituted per variation. */
  steps: Array<Omit<MovementStep, "target"> & { target: string }>;
};

/**
 * Deterministically expand templates into related-but-distinct sequences by
 * substituting a target pool into any `{slot}` placeholders. No RNG: variation
 * `n` uses `targets[n % targets.length]`, so the same inputs always yield the
 * same corpus (safe for cached workflow runs and reproducible eval splits).
 */
export function synthesizeMovementSequences(params: {
  templates: MovementTemplate[];
  targets: string[];
  variationsPerTemplate: number;
}): MovementSequence[] {
  const { templates, targets, variationsPerTemplate } = params;
  const sequences: MovementSequence[] = [];
  for (const template of templates) {
    for (let v = 0; v < variationsPerTemplate; v += 1) {
      const target = targets.length > 0 ? targets[v % targets.length] : "";
      const steps = template.steps.map((step) => ({
        ...step,
        target: step.target.replaceAll("{slot}", target),
      }));
      sequences.push({ sourceId: `${template.id}#${v}`, steps });
    }
  }
  return sequences;
}
