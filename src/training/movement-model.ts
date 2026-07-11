import type { ReplayTimelineEvent } from "../capture/replay.js";
import type { TrajectorySpan } from "../capture/trajectory.js";

/**
 * A single normalized movement in a replayable sequence. `tool`/`gesture`
 * describe the *kind* of movement (tap, type, click, scroll…); `target` and
 * `summary` carry the concrete instance. Learning happens over a structural
 * token derived from `tool`/`gesture` (see {@link movementStepToken}), so a
 * model trained on one set of targets generalizes to new but related movements
 * that share the same structure.
 */
export type MovementStep = {
  tool: string;
  gesture?: string;
  target?: string;
  summary: string;
};

export type MovementSequence = {
  id: string;
  label?: string;
  steps: MovementStep[];
};

export type MovementDataset = {
  sequences: MovementSequence[];
};

/** Sentinel token emitted when a sequence terminates. */
export const MOVEMENT_END_TOKEN = "<end>";

/**
 * Structural token for a step: `tool:gesture`. Deliberately excludes `target`
 * and `summary` so the learned policy captures movement *structure* (the shape
 * of a task) rather than memorizing concrete targets — this is what lets the
 * model perform new-but-related movements.
 */
export function movementStepToken(step: MovementStep): string {
  const tool = step.tool.trim().toLowerCase() || "unknown";
  const gesture = step.gesture?.trim().toLowerCase();
  return gesture ? `${tool}:${gesture}` : `${tool}:-`;
}

export type MovementPrediction = {
  /** The predicted next token; equals {@link MOVEMENT_END_TOKEN} at a natural stop. */
  token: string;
  /** A canonical step for the predicted token, or `undefined` when `end` is true. */
  step?: MovementStep;
  /** Empirical probability of this token given the matched context. */
  probability: number;
  /** Context length (Markov order) that produced the prediction; 0 = unconditional. */
  order: number;
  /** True when the model predicts the sequence should end here. */
  end: boolean;
};

export type TrainMovementModelOptions = {
  /** Maximum Markov order (context window). Defaults to 3. */
  order?: number;
};

export type GenerateMovementOptions = {
  /** Hard cap on generated steps (excludes the end sentinel). Defaults to 32. */
  maxSteps?: number;
};

/**
 * A trained, in-process movement policy. Backends produce one of these from a
 * {@link MovementDataset}; it can predict the next movement and roll out whole
 * sequences from a prefix (replay / generalization).
 */
export interface TrainedMovementModel {
  readonly backend: string;
  /** Predict the next movement given a context of prior steps. */
  predictNext(context: MovementStep[]): MovementPrediction | undefined;
  /** Greedily roll out a full sequence starting from `prefix`. */
  generate(prefix: MovementStep[], options?: GenerateMovementOptions): MovementStep[];
  /** JSON-serializable model state (for artifact persistence). */
  serialize(): SerializedMarkovModel;
}

/**
 * Pluggable movement-model backend. The default {@link MarkovMovementBackend}
 * is a deterministic, dependency-free model suitable for cloud/CI. Real
 * on-device small-model backends (e.g. an MLX-trained policy) implement the
 * same interface behind this seam.
 */
export interface MovementModelBackend {
  readonly name: string;
  train(dataset: MovementDataset, options?: TrainMovementModelOptions): Promise<TrainedMovementModel>;
}

export type SerializedMarkovModel = {
  version: 1;
  backend: string;
  order: number;
  /** context-key -> { token -> count }. Context key = tokens joined by "|". */
  transitions: Record<string, Record<string, number>>;
  /** token -> canonical step for reconstruction. */
  representatives: Record<string, MovementStep>;
};

const START_TOKEN = "<start>";

/**
 * Order-k Markov backend with Katz-style backoff. It counts, for every context
 * of up to `order` prior tokens, the distribution over the next token
 * (including an end sentinel). At prediction time it uses the longest context
 * with observed data and backs off to shorter contexts when the exact prefix is
 * unseen — so novel sequences that share local structure still get sensible
 * predictions. Fully deterministic: ties break on token ascending.
 */
export class MarkovMovementBackend implements MovementModelBackend {
  readonly name = "markov";

  async train(dataset: MovementDataset, options?: TrainMovementModelOptions): Promise<TrainedMovementModel> {
    const order = Math.max(1, Math.floor(options?.order ?? 3));
    const transitions = new Map<string, Map<string, number>>();
    const representatives = new Map<string, MovementStep>();

    const bump = (contextTokens: string[], next: string): void => {
      const key = contextTokens.join("|");
      let dist = transitions.get(key);
      if (!dist) {
        dist = new Map<string, number>();
        transitions.set(key, dist);
      }
      dist.set(next, (dist.get(next) ?? 0) + 1);
    };

    for (const sequence of dataset.sequences) {
      const tokens: string[] = [];
      for (const step of sequence.steps) {
        const token = movementStepToken(step);
        if (!representatives.has(token)) {
          representatives.set(token, { ...step });
        }
        // Record this token against every context length 0..order.
        const history = [START_TOKEN, ...tokens];
        for (let ctxLen = 0; ctxLen <= order; ctxLen += 1) {
          bump(history.slice(history.length - ctxLen), token);
        }
        tokens.push(token);
      }
      // Terminal transition so generation knows where a sequence ends.
      const history = [START_TOKEN, ...tokens];
      for (let ctxLen = 0; ctxLen <= order; ctxLen += 1) {
        bump(history.slice(history.length - ctxLen), MOVEMENT_END_TOKEN);
      }
    }

    return new MarkovMovementModel(this.name, order, transitions, representatives);
  }

  static fromSerialized(model: SerializedMarkovModel): TrainedMovementModel {
    const transitions = new Map<string, Map<string, number>>();
    for (const [key, dist] of Object.entries(model.transitions)) {
      transitions.set(key, new Map(Object.entries(dist)));
    }
    const representatives = new Map<string, MovementStep>(Object.entries(model.representatives));
    return new MarkovMovementModel(model.backend, model.order, transitions, representatives);
  }
}

class MarkovMovementModel implements TrainedMovementModel {
  constructor(
    readonly backend: string,
    private readonly order: number,
    private readonly transitions: Map<string, Map<string, number>>,
    private readonly representatives: Map<string, MovementStep>,
  ) {}

  predictNext(context: MovementStep[]): MovementPrediction | undefined {
    const history = [START_TOKEN, ...context.map(movementStepToken)];
    const maxOrder = Math.min(this.order, history.length);
    for (let ctxLen = maxOrder; ctxLen >= 0; ctxLen -= 1) {
      const key = history.slice(history.length - ctxLen).join("|");
      const dist = this.transitions.get(key);
      if (!dist || dist.size === 0) {
        continue;
      }
      const { token, count, total } = argmax(dist);
      const end = token === MOVEMENT_END_TOKEN;
      return {
        token,
        step: end ? undefined : this.representatives.get(token),
        probability: total > 0 ? count / total : 0,
        order: ctxLen,
        end,
      };
    }
    return undefined;
  }

  generate(prefix: MovementStep[], options?: GenerateMovementOptions): MovementStep[] {
    const maxSteps = Math.max(0, Math.floor(options?.maxSteps ?? 32));
    const steps: MovementStep[] = [...prefix];
    while (steps.length < maxSteps) {
      const prediction = this.predictNext(steps);
      if (!prediction || prediction.end || !prediction.step) {
        break;
      }
      steps.push(prediction.step);
    }
    return steps.slice(prefix.length);
  }

  serialize(): SerializedMarkovModel {
    const transitions: Record<string, Record<string, number>> = {};
    for (const [key, dist] of this.transitions) {
      transitions[key] = Object.fromEntries(dist);
    }
    return {
      version: 1,
      backend: this.backend,
      order: this.order,
      transitions,
      representatives: Object.fromEntries(this.representatives),
    };
  }
}

function argmax(dist: Map<string, number>): { token: string; count: number; total: number } {
  let bestToken = "";
  let bestCount = -1;
  let total = 0;
  for (const [token, count] of dist) {
    total += count;
    if (count > bestCount || (count === bestCount && token < bestToken)) {
      bestCount = count;
      bestToken = token;
    }
  }
  return { token: bestToken, count: bestCount, total };
}

export type MovementEvalResult = {
  sequences: number;
  steps: number;
  correct: number;
  /** Next-step top-1 accuracy over all evaluated positions. */
  accuracy: number;
  /** Fraction of positions where the model produced any prediction. */
  coverage: number;
};

/**
 * Generalization eval harness: for each held-out sequence, walk the prefix and
 * measure whether the model's next-step prediction matches the actual next
 * movement. Run on synthetic sequences that are *related but unseen* to measure
 * whether the policy generalizes beyond its training set.
 */
export function evaluateMovementModel(model: TrainedMovementModel, heldOut: MovementDataset): MovementEvalResult {
  let steps = 0;
  let correct = 0;
  let predicted = 0;
  for (const sequence of heldOut.sequences) {
    for (let i = 0; i < sequence.steps.length; i += 1) {
      steps += 1;
      const prediction = model.predictNext(sequence.steps.slice(0, i));
      if (!prediction) {
        continue;
      }
      predicted += 1;
      if (!prediction.end && prediction.token === movementStepToken(sequence.steps[i]!)) {
        correct += 1;
      }
    }
  }
  return {
    sequences: heldOut.sequences.length,
    steps,
    correct,
    accuracy: steps > 0 ? correct / steps : 0,
    coverage: steps > 0 ? predicted / steps : 0,
  };
}

/** Bridge: derive a movement sequence from a replay manifest's timeline events. */
export function movementSequenceFromReplayEvents(
  id: string,
  events: ReplayTimelineEvent[],
  label?: string,
): MovementSequence {
  const steps: MovementStep[] = events
    .filter((event): event is Extract<ReplayTimelineEvent, { kind: "action" }> => event.kind === "action")
    .map((event) => ({ tool: event.tool, summary: event.summary }));
  return { id, ...(label ? { label } : {}), steps };
}

/** Bridge: derive a movement sequence from a captured trajectory span's actions. */
export function movementSequenceFromTrajectory(trajectory: TrajectorySpan, label?: string): MovementSequence {
  const steps: MovementStep[] = trajectory.actions.map((action) => {
    const gesture = action.metadata?.["gesture"];
    const target = action.metadata?.["target"];
    return {
      tool: action.tool,
      ...(typeof gesture === "string" ? { gesture } : {}),
      ...(typeof target === "string" ? { target } : {}),
      summary: action.summary,
    };
  });
  return { id: trajectory.id, ...(label ? { label } : {}), steps };
}
