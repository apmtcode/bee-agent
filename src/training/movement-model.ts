import type { ReplayTimelineEvent } from "../capture/replay.js";
import type { TrajectorySpan } from "../capture/trajectory.js";

/**
 * Local-movement learning: pluggable model backend.
 *
 * This module closes objective #2 parts (c) + (d): post-train a *local* model on
 * a recorded movement dataset so it can (c) repeat the recorded movements and
 * (d) generalize to new-but-related movements. The rest of `src/training`
 * already exports reviewed datasets and emits on-device (mlx/axolotl) launch
 * plans; what was missing is an in-process, deterministic model backend that
 * actually learns from the dataset and runs inference -- so the train->infer
 * loop can be validated in the cloud/CI with no real OS input or GPU.
 *
 * The {@link MovementModelBackend} interface is the pluggable seam: a real
 * on-device small model implements the same `train`/`infer` contract, while the
 * bundled {@link MarkovMovementBackend} is a deterministic, dependency-free
 * reference backend used by tests and as a sane default.
 */

/** A single normalized movement/action step (mouse, key, gesture, ...). */
export type MovementStep = {
  /** Structural gesture verb, e.g. "tap", "swipe", "type", "click", "scroll". */
  gesture: string;
  /** UI target / element the gesture acted on, if known. */
  target?: string;
  /** Spatial direction for swipes/scrolls, if any. */
  direction?: string;
  /** Summary of any typed/entered value (already redacted upstream). */
  value?: string;
};

/** An ordered sequence of steps captured under one context (app/screen). */
export type MovementSequence = {
  id: string;
  /** Context label (e.g. `app:screen`) used to condition start/transitions. */
  context?: string;
  steps: MovementStep[];
};

export type MovementTrainingDataset = {
  sequences: MovementSequence[];
};

export type TrainMovementModelOptions = {
  /** Max Markov order (history length) to learn. Higher = more literal recall. */
  order?: number;
};

/** Serializable learned parameters. A real backend can define its own shape. */
export type MovementModel = {
  version: 1;
  backend: string;
  order: number;
  /** Sequence count the model was trained on. */
  trainedSequences: number;
  /** Per-context start-token counts. Key: context (or "") -> token -> count. */
  starts: Record<string, Record<string, number>>;
  /**
   * Transition counts per order. `transitions[k]` maps a joined history of the
   * last `k` tokens (scoped by context) -> next-token -> count.
   */
  transitions: Array<Record<string, Record<string, number>>>;
  /**
   * History-conditioned concrete targets: keyed by the token-path that produced
   * the step (context + last-`order` tokens + token) -> target -> count. This
   * lets a gesture that repeats with different targets (e.g. tap username, tap
   * password) be reproduced positionally instead of collapsing to one target.
   */
  targets: Record<string, Record<string, number>>;
  /** Token-only target fallback used when the precise path is unseen. */
  targetsByToken: Record<string, Record<string, number>>;
  /** History-conditioned value fills (for "type" steps). */
  values: Record<string, Record<string, number>>;
  /** Token-only value fallback. */
  valuesByToken: Record<string, Record<string, number>>;
};

export type MovementInferencePrompt = {
  /** Context to condition on. Falls back to global stats when unseen. */
  context?: string;
  /** Optional leading steps to continue from (repeat/steer). */
  seed?: MovementStep[];
  /** Hard cap on generated steps (including the seed). Default 32. */
  maxSteps?: number;
  /**
   * Per-target substitution used for generalization: when the model would emit
   * target `A` but `A -> B` is provided here, it emits `B` while preserving the
   * gesture structure. Enables "do the same movement on a related element".
   */
  targetOverrides?: Record<string, string>;
};

export type MovementInferenceResult = {
  steps: MovementStep[];
  /** True if any step required backing off below the model's max order. */
  backedOff: boolean;
  /** Fraction of transitions that were seen at the full model order [0,1]. */
  confidence: number;
};

export interface MovementModelBackend {
  readonly name: string;
  train(dataset: MovementTrainingDataset, options?: TrainMovementModelOptions): Promise<MovementModel>;
  infer(model: MovementModel, prompt: MovementInferencePrompt): Promise<MovementInferenceResult>;
}

// Control characters that cannot appear in gesture/target/direction labels, so
// tokens and history keys are unambiguously delimited.
const END_TOKEN = "end";
const FIELD_SEP = ""; // separates gesture from direction inside a token
const HISTORY_SEP = ""; // separates tokens inside a history key

/** Structural token for a step: gesture + direction, target-independent. */
export function movementToken(step: MovementStep): string {
  return step.direction ? `${step.gesture}${FIELD_SEP}${step.direction}` : step.gesture;
}

const DEFAULT_ORDER = 4;

function contextKey(context: string | undefined, suffix: string): string {
  return `${context ?? ""}${HISTORY_SEP}${suffix}`;
}

/** Token-path key: context + the last-`order` tokens + the emitted token. */
function pathKey(context: string | undefined, history: string[], token: string, order: number): string {
  return contextKey(context, [...history.slice(-order), token].join(HISTORY_SEP));
}

/** The context scopes a sequence is recorded under: its own, plus global (""). */
function recordScopes(context: string | undefined): Array<string | undefined> {
  return context === undefined || context === "" ? [undefined] : [context, undefined];
}

function increment(bucket: Record<string, Record<string, number>>, key: string, token: string): void {
  const inner = (bucket[key] ??= {});
  inner[token] = (inner[token] ?? 0) + 1;
}

/**
 * Deterministic n-gram (variable-order Markov) movement backend.
 *
 * Greedy decoding with count-based backoff makes it exactly reproduce a
 * sequence it has seen (repeat), and fall back to lower-order / global
 * transitions for novel contexts (generalize). It is intentionally free of any
 * native/GPU dependency so the full train->infer->eval loop runs in CI.
 */
export class MarkovMovementBackend implements MovementModelBackend {
  readonly name = "markov";

  async train(
    dataset: MovementTrainingDataset,
    options: TrainMovementModelOptions = {},
  ): Promise<MovementModel> {
    const order = Math.max(1, Math.min(options.order ?? DEFAULT_ORDER, 8));
    const starts: MovementModel["starts"] = {};
    const transitions: MovementModel["transitions"] = Array.from({ length: order }, () => ({}));
    const targets: MovementModel["targets"] = {};
    const targetsByToken: MovementModel["targetsByToken"] = {};
    const values: MovementModel["values"] = {};
    const valuesByToken: MovementModel["valuesByToken"] = {};

    for (const sequence of dataset.sequences) {
      const tokens = sequence.steps.map(movementToken);
      if (tokens.length === 0) {
        continue;
      }
      // Record under the sequence's own context AND a global "" scope, so an
      // unseen-but-related context can still back off to what was learned.
      const scopes = recordScopes(sequence.context);

      for (const scope of scopes) {
        const startBucket = (starts[scope ?? ""] ??= {});
        startBucket[tokens[0]!] = (startBucket[tokens[0]!] ?? 0) + 1;
      }

      // Slot memory: history-conditioned + token-only target/value fills.
      sequence.steps.forEach((step, index) => {
        const token = tokens[index]!;
        const history = tokens.slice(0, index);
        for (const scope of scopes) {
          if (step.target) {
            increment(targets, pathKey(scope, history, token, order), step.target);
            increment(targetsByToken, contextKey(scope, token), step.target);
          }
          if (step.value) {
            increment(values, pathKey(scope, history, token, order), step.value);
            increment(valuesByToken, contextKey(scope, token), step.value);
          }
        }
      });

      // Terminate every training sequence with an explicit end token so the
      // model learns *when to stop*, not just what comes next.
      const terminated = [...tokens, END_TOKEN];
      for (let index = 1; index < terminated.length; index += 1) {
        const next = terminated[index]!;
        for (let k = 1; k <= order; k += 1) {
          if (index - k < 0) {
            break;
          }
          const history = terminated.slice(index - k, index).join(HISTORY_SEP);
          for (const scope of scopes) {
            increment(transitions[k - 1]!, contextKey(scope, history), next);
          }
        }
      }
    }

    return {
      version: 1,
      backend: this.name,
      order,
      trainedSequences: dataset.sequences.length,
      starts,
      transitions,
      targets,
      targetsByToken,
      values,
      valuesByToken,
    };
  }

  async infer(model: MovementModel, prompt: MovementInferencePrompt): Promise<MovementInferenceResult> {
    const maxSteps = Math.max(1, prompt.maxSteps ?? 32);
    const context = prompt.context;
    const overrides = prompt.targetOverrides ?? {};

    const seed = prompt.seed ?? [];
    const emitted: MovementStep[] = seed.map((step) => applyOverride(step, overrides));
    const history = seed.map(movementToken);

    let backedOffTransitions = 0;
    let totalTransitions = 0;

    // If no seed, sample the (deterministic) most-likely start token.
    if (history.length === 0) {
      const specificTop = context === undefined ? undefined : pickTop(model.starts[context]);
      const start = specificTop ?? pickTop(model.starts[""]);
      if (start === undefined) {
        return { steps: [], backedOff: false, confidence: 1 };
      }
      const global = specificTop === undefined && context !== undefined;
      totalTransitions += 1;
      if (global) {
        backedOffTransitions += 1;
      }
      emitted.push(fillStep(model, context, [], start, overrides, { order: 1, global }));
      history.push(start);
    }

    while (emitted.length < maxSteps) {
      const prediction = predictNext(model, context, history);
      totalTransitions += 1;
      const expectedOrder = Math.min(model.order, history.length);
      if (prediction.token === undefined || prediction.token === END_TOKEN) {
        break;
      }
      if (prediction.order < expectedOrder || prediction.global) {
        backedOffTransitions += 1;
      }
      emitted.push(
        fillStep(model, context, history, prediction.token, overrides, { order: prediction.order, global: prediction.global }),
      );
      history.push(prediction.token);
    }

    const confidence = totalTransitions === 0 ? 1 : 1 - backedOffTransitions / totalTransitions;
    return { steps: emitted, backedOff: backedOffTransitions > 0, confidence };
  }
}

type ResolveScope = { order: number; global: boolean };

function predictNext(
  model: MovementModel,
  context: string | undefined,
  history: string[],
): { token: string | undefined; order: number; global: boolean } {
  // Back off from the model's max order down to 1. At each order, prefer the
  // context-specific stats; fall back to global (context-less) stats.
  for (let k = Math.min(model.order, history.length); k >= 1; k -= 1) {
    const suffix = history.slice(history.length - k).join(HISTORY_SEP);
    if (context !== undefined) {
      const token = pickTop(model.transitions[k - 1]?.[contextKey(context, suffix)]);
      if (token !== undefined) {
        return { token, order: k, global: false };
      }
    }
    const globalToken = pickTop(model.transitions[k - 1]?.[contextKey(undefined, suffix)]);
    if (globalToken !== undefined) {
      return { token: globalToken, order: k, global: context !== undefined };
    }
  }
  return { token: undefined, order: 0, global: false };
}

function fillStep(
  model: MovementModel,
  context: string | undefined,
  history: string[],
  token: string,
  overrides: Record<string, string>,
  scope: ResolveScope,
): MovementStep {
  const [gesture, direction] = token.split(FIELD_SEP);
  const step: MovementStep = { gesture: gesture! };
  if (direction) {
    step.direction = direction;
  }
  const scopedContext = scope.global ? undefined : context;
  const path = pathKey(scopedContext, history, token, model.order);

  const target =
    pickTop(model.targets[path]) ??
    pickTop(model.targetsByToken[contextKey(scopedContext, token)]) ??
    pickTop(model.targetsByToken[contextKey(undefined, token)]);
  if (target !== undefined) {
    step.target = overrides[target] ?? target;
  }

  const value =
    pickTop(model.values[path]) ??
    pickTop(model.valuesByToken[contextKey(scopedContext, token)]) ??
    pickTop(model.valuesByToken[contextKey(undefined, token)]);
  if (value !== undefined) {
    step.value = value;
  }
  return step;
}

function applyOverride(step: MovementStep, overrides: Record<string, string>): MovementStep {
  if (step.target && overrides[step.target]) {
    return { ...step, target: overrides[step.target]! };
  }
  return { ...step };
}

/**
 * Deterministic arg-max: highest count wins. Ties are broken by token order,
 * except the END sentinel always loses a tie to a real token so a legitimate
 * continuation is preferred over stopping early (it still wins if strictly most
 * frequent).
 */
function pickTop(bucket: Record<string, number> | undefined): string | undefined {
  if (!bucket) {
    return undefined;
  }
  let best: string | undefined;
  let bestCount = -Infinity;
  for (const [token, count] of Object.entries(bucket)) {
    if (count > bestCount || (count === bestCount && best !== undefined && preferOver(token, best))) {
      best = token;
      bestCount = count;
    }
  }
  return best;
}

/** True if `candidate` should beat `incumbent` on an equal-count tie. */
function preferOver(candidate: string, incumbent: string): boolean {
  const candidateEnd = candidate === END_TOKEN;
  const incumbentEnd = incumbent === END_TOKEN;
  if (candidateEnd !== incumbentEnd) {
    return incumbentEnd; // prefer the non-END token
  }
  return candidate < incumbent;
}

/** Extract a {@link MovementSequence} from a captured trajectory span. */
export function movementSequenceFromTrajectory(trajectory: TrajectorySpan): MovementSequence {
  const contextObservation = trajectory.observations[0];
  const context = contextObservation
    ? `${contextObservation.source}:${contextObservation.metadata?.["appName"] ?? contextObservation.summary}`
    : undefined;
  return {
    id: trajectory.id,
    ...(context ? { context } : {}),
    steps: trajectory.actions.map((action) => movementStepFromMetadata(action.tool, action.summary, action.metadata)),
  };
}

/** Extract a {@link MovementSequence} from a replay timeline (action events). */
export function movementSequenceFromReplay(
  sequenceId: string,
  events: ReplayTimelineEvent[],
  context?: string,
): MovementSequence {
  const steps = events
    .filter((event): event is Extract<ReplayTimelineEvent, { kind: "action" }> => event.kind === "action")
    .map((event) => movementStepFromMetadata(event.tool, event.summary, undefined));
  return { id: sequenceId, ...(context ? { context } : {}), steps };
}

function movementStepFromMetadata(
  tool: string,
  summary: string,
  metadata: Record<string, unknown> | undefined,
): MovementStep {
  const gesture = typeof metadata?.["gesture"] === "string" ? (metadata["gesture"] as string) : tool;
  const step: MovementStep = { gesture };
  if (typeof metadata?.["target"] === "string") {
    step.target = metadata["target"] as string;
  } else if (!metadata) {
    // Best-effort target recovery from a replay summary like "tapped Submit".
    const recovered = summary.split(" ").slice(1).join(" ").trim();
    if (recovered) {
      step.target = recovered;
    }
  }
  if (typeof metadata?.["direction"] === "string") {
    step.direction = metadata["direction"] as string;
  }
  if (typeof metadata?.["valueSummary"] === "string") {
    step.value = metadata["valueSummary"] as string;
  }
  return step;
}

export type MovementEvalResult = {
  /** Sequences evaluated. */
  sequences: number;
  /** Mean token-level fidelity across sequences [0,1]. */
  tokenFidelity: number;
  /** Fraction of sequences reproduced exactly (structural tokens). */
  exactMatch: number;
  /** Mean inference confidence reported by the backend [0,1]. */
  meanConfidence: number;
};

/**
 * Generalization eval harness: for each held-out sequence, seed the model with
 * its first step and measure how closely the model's continuation matches the
 * recorded structural tokens (longest-common-prefix / length). This is the
 * measurable signal for objective #2(d).
 */
export async function evaluateMovementModel(
  backend: MovementModelBackend,
  model: MovementModel,
  heldOut: MovementSequence[],
): Promise<MovementEvalResult> {
  if (heldOut.length === 0) {
    return { sequences: 0, tokenFidelity: 1, exactMatch: 1, meanConfidence: 1 };
  }

  let fidelitySum = 0;
  let exactCount = 0;
  let confidenceSum = 0;

  for (const sequence of heldOut) {
    const expected = sequence.steps.map(movementToken);
    const result = await backend.infer(model, {
      ...(sequence.context ? { context: sequence.context } : {}),
      seed: sequence.steps.slice(0, 1),
      maxSteps: Math.max(expected.length, 1),
    });
    const predicted = result.steps.map(movementToken);
    confidenceSum += result.confidence;

    const denominator = Math.max(expected.length, 1);
    let matched = 0;
    for (let index = 0; index < expected.length; index += 1) {
      if (predicted[index] === expected[index]) {
        matched += 1;
      } else {
        break;
      }
    }
    fidelitySum += matched / denominator;
    if (matched === expected.length && predicted.length === expected.length) {
      exactCount += 1;
    }
  }

  return {
    sequences: heldOut.length,
    tokenFidelity: fidelitySum / heldOut.length,
    exactMatch: exactCount / heldOut.length,
    meanConfidence: confidenceSum / heldOut.length,
  };
}
