// Pluggable local movement-model backend.
//
// The self-evolution objective requires bee-agent to post-train a *local* model
// on recorded movement datasets so it can (c) repeat recorded movements and
// (d) generalize to new-but-related movements. The real on-device training runs
// only when the user runs bee-agent locally (see `LocalAppleSiliconTrainingRunner`,
// which emits an mlx/axolotl launch plan). To make that pipeline testable and
// runnable in the cloud, the model layer is expressed behind a small backend
// interface with a deterministic in-process backend (`MarkovMovementBackend`).
//
// The Markov backend is intentionally simple but genuinely *learned*: it fits a
// variable-order transition table over movement-action tokens and predicts the
// next action by longest-suffix backoff. Backoff is what gives it generalization
// — an unseen full-order context still yields a sensible prediction from a
// shorter matching suffix (or the global action prior). A heavier on-device model
// can be dropped in later by implementing `MovementModelBackend`.

/** A single recorded or predicted movement/action step (mouse/keyboard/window/UI). */
export type MovementStep = {
  /** Discrete action verb, e.g. "mouse.move", "mouse.click", "key.press", "window.focus". */
  action: string;
  /** Optional named parameters; numeric params (dx/dy/x/y) are movement deltas. */
  params?: Record<string, number | string>;
};

/** An ordered sequence of movement steps captured from one task/trajectory. */
export type MovementSequence = {
  id: string;
  steps: MovementStep[];
};

export type MovementPrediction = {
  /** Predicted next action verb. */
  action: string;
  /** Estimated probability in [0,1] within the matched context. */
  probability: number;
  /**
   * Number of context steps that actually matched after backoff. Equals the
   * model order for a full-context hit, and 0 for a unigram (global prior) hit.
   */
  matchedContext: number;
};

export type SerializedMovementModel = {
  version: 1;
  backendId: string;
  order: number;
  /** contextKey -> (action -> count). The empty-string key is the unigram prior. */
  transitions: Record<string, Record<string, number>>;
  /** action -> representative step emitted during generation. */
  prototypes: Record<string, MovementStep>;
};

export type TrainMovementOptions = {
  /** Maximum context length (n-gram order). Defaults to 2. */
  order?: number;
};

export interface TrainedMovementModel {
  readonly backendId: string;
  readonly order: number;
  /** Distinct action verbs the model knows. */
  vocabulary(): string[];
  /** Predict the most likely next action given prior steps; undefined if untrained/empty. */
  predictNext(context: MovementStep[]): MovementPrediction | undefined;
  /** Roll out up to `maxSteps` new steps continuing from `seed`. */
  generate(seed: MovementStep[], maxSteps: number): MovementStep[];
  serialize(): SerializedMovementModel;
}

export interface MovementModelBackend {
  readonly id: string;
  train(dataset: MovementSequence[], options?: TrainMovementOptions): TrainedMovementModel;
  load(serialized: SerializedMovementModel): TrainedMovementModel;
}

const CONTEXT_DELIMITER = "\x1f";
const DEFAULT_ORDER = 2;

/** Deterministic, dependency-free variable-order Markov movement backend. */
export class MarkovMovementBackend implements MovementModelBackend {
  readonly id = "markov";

  train(dataset: MovementSequence[], options: TrainMovementOptions = {}): TrainedMovementModel {
    const order = normalizeOrder(options.order);
    const transitions: Record<string, Record<string, number>> = {};
    const prototypeAccumulators = new Map<string, PrototypeAccumulator>();

    for (const sequence of dataset) {
      const steps = sequence.steps;
      for (let index = 0; index < steps.length; index += 1) {
        const step = steps[index];
        accumulatePrototype(prototypeAccumulators, step);
        // Register this step as the continuation of every context length 0..order.
        for (let contextLength = 0; contextLength <= order; contextLength += 1) {
          if (index - contextLength < 0) {
            break;
          }
          const contextSteps = steps.slice(index - contextLength, index);
          const key = contextKey(contextSteps);
          const bucket = (transitions[key] ??= {});
          bucket[step.action] = (bucket[step.action] ?? 0) + 1;
        }
      }
    }

    const prototypes: Record<string, MovementStep> = {};
    for (const [action, accumulator] of prototypeAccumulators) {
      prototypes[action] = finalizePrototype(action, accumulator);
    }

    return new MarkovTrainedModel(this.id, order, transitions, prototypes);
  }

  load(serialized: SerializedMovementModel): TrainedMovementModel {
    return new MarkovTrainedModel(
      serialized.backendId,
      normalizeOrder(serialized.order),
      cloneTransitions(serialized.transitions),
      { ...serialized.prototypes },
    );
  }
}

class MarkovTrainedModel implements TrainedMovementModel {
  constructor(
    readonly backendId: string,
    readonly order: number,
    private readonly transitions: Record<string, Record<string, number>>,
    private readonly prototypes: Record<string, MovementStep>,
  ) {}

  vocabulary(): string[] {
    return Object.keys(this.prototypes).sort();
  }

  predictNext(context: MovementStep[]): MovementPrediction | undefined {
    const maxContext = Math.min(this.order, context.length);
    // Longest-suffix backoff: try the fullest context first, then shrink.
    for (let contextLength = maxContext; contextLength >= 0; contextLength -= 1) {
      const suffix = context.slice(context.length - contextLength, context.length);
      const bucket = this.transitions[contextKey(suffix)];
      if (!bucket) {
        continue;
      }
      const best = argmaxAction(bucket);
      if (!best) {
        continue;
      }
      const total = Object.values(bucket).reduce((sum, count) => sum + count, 0);
      return {
        action: best.action,
        probability: total > 0 ? best.count / total : 0,
        matchedContext: contextLength,
      };
    }
    return undefined;
  }

  generate(seed: MovementStep[], maxSteps: number): MovementStep[] {
    const generated: MovementStep[] = [];
    const running = [...seed];
    for (let step = 0; step < maxSteps; step += 1) {
      const prediction = this.predictNext(running);
      if (!prediction) {
        break;
      }
      const next = this.prototypes[prediction.action] ?? { action: prediction.action };
      const emitted: MovementStep = cloneStep(next);
      generated.push(emitted);
      running.push(emitted);
    }
    return generated;
  }

  serialize(): SerializedMovementModel {
    return {
      version: 1,
      backendId: this.backendId,
      order: this.order,
      transitions: cloneTransitions(this.transitions),
      prototypes: Object.fromEntries(
        Object.entries(this.prototypes).map(([action, step]) => [action, cloneStep(step)]),
      ),
    };
  }
}

export type MovementEvalResult = {
  /** Number of (context, actual-next) prediction points evaluated. */
  sampleCount: number;
  /** Predictions whose top-1 action matched the actual next action. */
  correct: number;
  /** Top-1 next-action accuracy in [0,1]. */
  accuracy: number;
  /** Fraction of predictions that fell back below the model's full order. */
  backoffRate: number;
  /** Prediction points where the model returned nothing (unknown context). */
  unpredicted: number;
};

/**
 * Generalization eval harness: measure top-1 next-action accuracy on held-out
 * sequences the model was not trained on. High accuracy on *related* held-out
 * data (same grammar, different parameters) demonstrates generalization rather
 * than memorization.
 */
export function evaluateNextStepAccuracy(
  model: TrainedMovementModel,
  heldOut: MovementSequence[],
): MovementEvalResult {
  let sampleCount = 0;
  let correct = 0;
  let backoff = 0;
  let unpredicted = 0;

  for (const sequence of heldOut) {
    const steps = sequence.steps;
    for (let index = 1; index < steps.length; index += 1) {
      sampleCount += 1;
      const context = steps.slice(0, index);
      const prediction = model.predictNext(context);
      if (!prediction) {
        unpredicted += 1;
        continue;
      }
      if (prediction.matchedContext < model.order) {
        backoff += 1;
      }
      if (prediction.action === steps[index].action) {
        correct += 1;
      }
    }
  }

  return {
    sampleCount,
    correct,
    accuracy: sampleCount > 0 ? correct / sampleCount : 0,
    backoffRate: sampleCount > 0 ? backoff / sampleCount : 0,
    unpredicted,
  };
}

/** Registry making the movement-model backend pluggable (mock now, on-device later). */
export class MovementBackendRegistry {
  private readonly backends = new Map<string, MovementModelBackend>();

  register(backend: MovementModelBackend): this {
    this.backends.set(backend.id, backend);
    return this;
  }

  resolve(id: string): MovementModelBackend {
    const backend = this.backends.get(id);
    if (!backend) {
      throw new Error(`unknown movement backend: ${id} (known: ${this.list().join(", ") || "none"})`);
    }
    return backend;
  }

  has(id: string): boolean {
    return this.backends.has(id);
  }

  list(): string[] {
    return [...this.backends.keys()].sort();
  }
}

/** A registry pre-populated with the deterministic in-process Markov backend. */
export function createDefaultMovementBackendRegistry(): MovementBackendRegistry {
  return new MovementBackendRegistry().register(new MarkovMovementBackend());
}

type PrototypeAccumulator = {
  numeric: Map<string, { sum: number; count: number }>;
  strings: Map<string, Map<string, number>>;
};

function accumulatePrototype(accumulators: Map<string, PrototypeAccumulator>, step: MovementStep): void {
  const accumulator = accumulators.get(step.action) ?? { numeric: new Map(), strings: new Map() };
  accumulators.set(step.action, accumulator);
  if (!step.params) {
    return;
  }
  for (const [name, value] of Object.entries(step.params)) {
    if (typeof value === "number") {
      const entry = accumulator.numeric.get(name) ?? { sum: 0, count: 0 };
      entry.sum += value;
      entry.count += 1;
      accumulator.numeric.set(name, entry);
    } else {
      const counts = accumulator.strings.get(name) ?? new Map<string, number>();
      counts.set(value, (counts.get(value) ?? 0) + 1);
      accumulator.strings.set(name, counts);
    }
  }
}

function finalizePrototype(action: string, accumulator: PrototypeAccumulator): MovementStep {
  const params: Record<string, number | string> = {};
  for (const [name, entry] of accumulator.numeric) {
    // Round the mean to avoid noisy floats in a replayable prototype.
    params[name] = entry.count > 0 ? roundTo(entry.sum / entry.count, 3) : 0;
  }
  for (const [name, counts] of accumulator.strings) {
    const mode = argmaxAction(Object.fromEntries(counts));
    if (mode) {
      params[name] = mode.action;
    }
  }
  return Object.keys(params).length > 0 ? { action, params } : { action };
}

function argmaxAction(bucket: Record<string, number>): { action: string; count: number } | undefined {
  let best: { action: string; count: number } | undefined;
  // Deterministic: highest count wins; ties broken by lexicographic action order.
  for (const action of Object.keys(bucket).sort()) {
    const count = bucket[action];
    if (!best || count > best.count) {
      best = { action, count };
    }
  }
  return best;
}

function contextKey(steps: MovementStep[]): string {
  return steps.map((step) => step.action).join(CONTEXT_DELIMITER);
}

function normalizeOrder(order: number | undefined): number {
  if (order === undefined || !Number.isFinite(order)) {
    return DEFAULT_ORDER;
  }
  return Math.max(1, Math.floor(order));
}

function cloneStep(step: MovementStep): MovementStep {
  return step.params ? { action: step.action, params: { ...step.params } } : { action: step.action };
}

function cloneTransitions(
  transitions: Record<string, Record<string, number>>,
): Record<string, Record<string, number>> {
  const clone: Record<string, Record<string, number>> = {};
  for (const [key, bucket] of Object.entries(transitions)) {
    clone[key] = { ...bucket };
  }
  return clone;
}

function roundTo(value: number, decimals: number): number {
  const factor = 10 ** decimals;
  return Math.round(value * factor) / factor;
}
