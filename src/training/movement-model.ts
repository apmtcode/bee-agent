/**
 * Local-movement learning model (standing objective #2, parts c & d).
 *
 * This module closes the loop of the movement-learning subsystem: after
 * capture (`src/capture`) records reviewed trajectories and the exporter turns
 * them into a reviewed dataset, this module post-trains a *local, in-process*
 * model that can (c) repeat the recorded movements and (d) generalize to new
 * but related movements.
 *
 * The model backend is pluggable: `MovementModelBackend` is the seam a real
 * on-device backend (e.g. a small local net trained via mlx/axolotl — see
 * `runner.ts`) implements. Shipped here is `MarkovMovementBackend`, a fully
 * deterministic, dependency-free backoff n-gram backend that runs anywhere —
 * including cloud/CI — so the whole pipeline is testable without real OS input
 * or GPU training. Generalization comes from Katz-style backoff: contexts never
 * seen at the highest order fall back to shorter contexts, letting the model
 * stitch together transitions it saw only in parts into novel-but-related
 * sequences.
 *
 * Everything here is deterministic (no RNG, no wall-clock) so training and
 * evaluation are reproducible and diffable.
 */

/** A single normalized movement — the unit the model predicts. */
export interface MovementStep {
  /** Originating tool/surface, e.g. "device", "os", "browser". */
  tool: string;
  /** Canonical action verb, e.g. "tap", "swipe", "focus-changed", "type". */
  action: string;
  /** Optional target the action applied to, e.g. "submit-button". */
  target?: string;
  /** Optional direction for directional gestures, e.g. "down". */
  direction?: string;
}

/**
 * A movement encoded as a single opaque string token. Four fields joined by
 * `|` (tool|action|target|direction) so tokens are reversible and stable.
 */
export type MovementToken = string;

/** One recorded trajectory as an ordered list of movement tokens. */
export interface MovementSequence {
  id: string;
  tokens: MovementToken[];
}

/** A replayable, trainable dataset of movement sequences. */
export interface MovementDataset {
  version: 1;
  sequences: MovementSequence[];
}

/** Minimal shape of a recorded action this module can tokenize. */
export interface MovementActionLike {
  tool: string;
  summary: string;
  ts: number;
  metadata?: Record<string, unknown>;
}

const TOKEN_SEP = "|";

function coerceString(value: unknown): string | undefined {
  if (typeof value !== "string") {
    return undefined;
  }
  const trimmed = value.trim();
  return trimmed.length > 0 ? trimmed : undefined;
}

/** Derive a normalized {@link MovementStep} from a recorded action. */
export function movementStepFromAction(action: MovementActionLike): MovementStep {
  const metadata = action.metadata ?? {};
  const actionName =
    coerceString(metadata.gesture) ??
    coerceString(metadata.event) ??
    coerceString(action.summary)?.split(/\s+/)[0] ??
    action.tool;
  const target =
    coerceString(metadata.target) ??
    coerceString(metadata.windowTitle) ??
    coerceString(metadata.filePath) ??
    coerceString(metadata.commandSummary);
  const direction = coerceString(metadata.direction);
  const step: MovementStep = { tool: action.tool, action: actionName };
  if (target !== undefined) {
    step.target = target;
  }
  if (direction !== undefined) {
    step.direction = direction;
  }
  return step;
}

/** Encode a {@link MovementStep} into a stable {@link MovementToken}. */
export function encodeMovementToken(step: MovementStep): MovementToken {
  return [step.tool, step.action, step.target ?? "", step.direction ?? ""]
    .map((part) => part.replaceAll(TOKEN_SEP, "/"))
    .join(TOKEN_SEP);
}

/** Decode a {@link MovementToken} back into a {@link MovementStep}. */
export function decodeMovementToken(token: MovementToken): MovementStep {
  const [tool = "", action = "", target = "", direction = ""] = token.split(TOKEN_SEP);
  const step: MovementStep = { tool, action };
  if (target.length > 0) {
    step.target = target;
  }
  if (direction.length > 0) {
    step.direction = direction;
  }
  return step;
}

/**
 * Build a {@link MovementDataset} from recorded trajectories. Actions within a
 * trajectory are sorted by timestamp so the token order reflects the movement
 * order regardless of how they were stored.
 */
export function buildMovementDataset(
  trajectories: Array<{ id: string; actions: MovementActionLike[] }>,
): MovementDataset {
  const sequences = trajectories
    .map((trajectory) => ({
      id: trajectory.id,
      tokens: [...trajectory.actions]
        .sort((a, b) => a.ts - b.ts)
        .map((action) => encodeMovementToken(movementStepFromAction(action))),
    }))
    .filter((sequence) => sequence.tokens.length > 0);
  return { version: 1, sequences };
}

/** A single ranked candidate for the next movement. */
export interface MovementCandidate {
  token: MovementToken;
  probability: number;
}

/** The model's prediction for the next movement given a context. */
export interface MovementPrediction {
  /** Best next token, or undefined when the model has no basis to predict. */
  token: MovementToken | undefined;
  /** Probability mass of the chosen token within the backoff context. */
  confidence: number;
  /** Context order actually used after backoff (0 = unigram prior). */
  order: number;
  /** All candidates ranked by probability (deterministic tie-break). */
  ranked: MovementCandidate[];
}

/** A serialized, persistable trained model (round-trips via a backend). */
export interface MovementModelSnapshot {
  version: 1;
  backendId: string;
  maxOrder: number;
  vocabulary: MovementToken[];
  /** order -> contextKey -> nextToken -> count. Order 0 uses key "". */
  counts: Record<string, Record<string, Record<string, number>>>;
}

/** A trained, ready-to-use movement model. */
export interface TrainedMovementModel {
  readonly backendId: string;
  readonly maxOrder: number;
  readonly vocabulary: MovementToken[];
  /** Predict the movement most likely to follow `context`. */
  predictNext(context: MovementToken[]): MovementPrediction;
  /**
   * Greedily generate a continuation from `seed`, up to `maxLength` tokens.
   * Stops early when the model can no longer predict or would loop.
   */
  generate(seed: MovementToken[], maxLength: number): MovementToken[];
  /** Serialize for persistence / later replay. */
  snapshot(): MovementModelSnapshot;
}

/** Options accepted by a backend's `train`. */
export interface MovementTrainOptions {
  /** Highest n-gram context order to model (default 3). */
  maxOrder?: number;
}

/**
 * The pluggable seam. A real on-device backend implements this to train a
 * local model from the reviewed dataset; the mock backend below implements it
 * deterministically for cloud/CI.
 */
export interface MovementModelBackend {
  readonly id: string;
  train(dataset: MovementDataset, options?: MovementTrainOptions): Promise<TrainedMovementModel>;
  /** Rehydrate a previously trained model from a snapshot. */
  restore(snapshot: MovementModelSnapshot): TrainedMovementModel;
}

type CountsByOrder = Map<number, Map<string, Map<MovementToken, number>>>;

/**
 * End-of-sequence sentinel. Appended to every sequence during training so the
 * model learns *when to stop*; `generate` halts when it predicts this. Uses a
 * NUL prefix so it can never collide with a real `tool|action|target|direction`
 * token, and it is excluded from the vocabulary and the order-0 prior.
 */
const END_TOKEN = "\u0000END";

function contextKey(context: MovementToken[]): string {
  return context.join("\n");
}

class MarkovMovementModel implements TrainedMovementModel {
  readonly backendId: string;
  readonly maxOrder: number;
  readonly vocabulary: MovementToken[];
  private readonly counts: CountsByOrder;

  constructor(params: {
    backendId: string;
    maxOrder: number;
    vocabulary: MovementToken[];
    counts: CountsByOrder;
  }) {
    this.backendId = params.backendId;
    this.maxOrder = params.maxOrder;
    this.vocabulary = params.vocabulary;
    this.counts = params.counts;
  }

  predictNext(context: MovementToken[]): MovementPrediction {
    for (let order = Math.min(this.maxOrder, context.length); order >= 1; order -= 1) {
      const key = contextKey(context.slice(context.length - order));
      const prediction = this.rank(order, key);
      if (prediction) {
        return prediction;
      }
    }
    // Order-0 backoff: the unconditional prior over observed movements.
    return this.rank(0, "") ?? { token: undefined, confidence: 0, order: 0, ranked: [] };
  }

  private rank(order: number, key: string): MovementPrediction | undefined {
    const nextCounts = this.counts.get(order)?.get(key);
    if (!nextCounts || nextCounts.size === 0) {
      return undefined;
    }
    let total = 0;
    for (const count of nextCounts.values()) {
      total += count;
    }
    const ranked: MovementCandidate[] = [...nextCounts.entries()]
      .map(([token, count]) => ({ token, probability: count / total }))
      // Deterministic ordering: probability desc, then real tokens before the
      // END sentinel (so a real continuation is preferred on ties), then asc.
      .sort((a, b) => {
        if (b.probability !== a.probability) {
          return b.probability - a.probability;
        }
        const aEnd = a.token === END_TOKEN;
        const bEnd = b.token === END_TOKEN;
        if (aEnd !== bEnd) {
          return aEnd ? 1 : -1;
        }
        return a.token < b.token ? -1 : a.token > b.token ? 1 : 0;
      });
    const best = ranked[0];
    return { token: best.token, confidence: best.probability, order, ranked };
  }

  generate(seed: MovementToken[], maxLength: number): MovementToken[] {
    const output: MovementToken[] = [];
    const context = [...seed];
    for (let i = 0; i < maxLength; i += 1) {
      const prediction = this.predictNext(context);
      if (prediction.token === undefined || prediction.token === END_TOKEN) {
        break;
      }
      // Loop guard: bail on a period-2 (A,B,A,B) oscillation that unigram
      // backoff can otherwise sustain forever when END is not reachable.
      const n = output.length;
      if (n >= 3 && prediction.token === output[n - 2] && output[n - 1] === output[n - 3]) {
        break;
      }
      output.push(prediction.token);
      context.push(prediction.token);
    }
    return output;
  }

  snapshot(): MovementModelSnapshot {
    const counts: MovementModelSnapshot["counts"] = {};
    for (const [order, byContext] of this.counts) {
      const orderObj: Record<string, Record<string, number>> = {};
      for (const [key, nextCounts] of byContext) {
        orderObj[key] = Object.fromEntries(nextCounts);
      }
      counts[String(order)] = orderObj;
    }
    return {
      version: 1,
      backendId: this.backendId,
      maxOrder: this.maxOrder,
      vocabulary: [...this.vocabulary],
      counts,
    };
  }
}

/**
 * Deterministic, dependency-free backoff n-gram backend. Serves as both the
 * default local backend and the reference/mock implementation for tests.
 */
export class MarkovMovementBackend implements MovementModelBackend {
  readonly id = "markov-backoff";
  private readonly defaultMaxOrder: number;

  constructor(options: { maxOrder?: number } = {}) {
    this.defaultMaxOrder = Math.max(1, options.maxOrder ?? 3);
  }

  async train(dataset: MovementDataset, options: MovementTrainOptions = {}): Promise<TrainedMovementModel> {
    const maxOrder = Math.max(1, options.maxOrder ?? this.defaultMaxOrder);
    const counts: CountsByOrder = new Map();
    const vocabulary = new Set<MovementToken>();

    const bump = (order: number, key: string, next: MovementToken): void => {
      let byContext = counts.get(order);
      if (!byContext) {
        byContext = new Map();
        counts.set(order, byContext);
      }
      let nextCounts = byContext.get(key);
      if (!nextCounts) {
        nextCounts = new Map();
        byContext.set(key, nextCounts);
      }
      nextCounts.set(next, (nextCounts.get(next) ?? 0) + 1);
    };

    for (const sequence of dataset.sequences) {
      // Append the END sentinel so the model learns the terminal transition.
      const tokens = [...sequence.tokens, END_TOKEN];
      for (let i = 0; i < tokens.length; i += 1) {
        const next = tokens[i];
        if (next !== END_TOKEN) {
          vocabulary.add(next);
          // Order-0 (unigram prior) counts every observed real movement.
          bump(0, "", next);
        }
        for (let order = 1; order <= maxOrder; order += 1) {
          if (i - order < 0) {
            break;
          }
          const key = contextKey(tokens.slice(i - order, i));
          bump(order, key, next);
        }
      }
    }

    return new MarkovMovementModel({
      backendId: this.id,
      maxOrder,
      vocabulary: [...vocabulary].sort(),
      counts,
    });
  }

  restore(snapshot: MovementModelSnapshot): TrainedMovementModel {
    const counts: CountsByOrder = new Map();
    for (const [orderStr, byContext] of Object.entries(snapshot.counts)) {
      const order = Number(orderStr);
      const orderMap = new Map<string, Map<MovementToken, number>>();
      for (const [key, nextCounts] of Object.entries(byContext)) {
        orderMap.set(key, new Map(Object.entries(nextCounts)));
      }
      counts.set(order, orderMap);
    }
    return new MarkovMovementModel({
      backendId: snapshot.backendId,
      maxOrder: snapshot.maxOrder,
      vocabulary: [...snapshot.vocabulary],
      counts,
    });
  }
}

/** Aggregate metrics from evaluating a trained model on held-out sequences. */
export interface MovementEvalResult {
  sequenceCount: number;
  /** Total teacher-forced prediction points (one per non-first token). */
  predictedSteps: number;
  correctSteps: number;
  /** Teacher-forced top-1 next-token accuracy (0..1). */
  nextTokenAccuracy: number;
  /** Sequences whose full continuation the model reproduced exactly. */
  exactSequenceReplays: number;
  /** Fraction of sequences reproduced exactly (0..1). */
  exactReplayRate: number;
  /** Mean confidence across prediction points. */
  averageConfidence: number;
  /** Fraction of predictions that had to back off below `maxOrder` (0..1). */
  backoffRate: number;
}

/**
 * Evaluate a trained model. Combines two lenses:
 *  - teacher-forced next-token accuracy (how well it *repeats*, obj 2c), and
 *  - exact-replay rate from a one-token seed (end-to-end fidelity).
 * Run it on held-out-but-related sequences to measure generalization (obj 2d).
 */
export function evaluateMovementModel(
  model: TrainedMovementModel,
  sequences: MovementSequence[],
): MovementEvalResult {
  let predictedSteps = 0;
  let correctSteps = 0;
  let confidenceSum = 0;
  let backoffCount = 0;
  let exactReplays = 0;
  let evaluableSequences = 0;

  for (const sequence of sequences) {
    const tokens = sequence.tokens;
    if (tokens.length < 2) {
      continue;
    }
    evaluableSequences += 1;
    for (let i = 1; i < tokens.length; i += 1) {
      const prediction = model.predictNext(tokens.slice(0, i));
      predictedSteps += 1;
      confidenceSum += prediction.confidence;
      if (prediction.order < model.maxOrder) {
        backoffCount += 1;
      }
      if (prediction.token === tokens[i]) {
        correctSteps += 1;
      }
    }
    const generated = model.generate([tokens[0]], tokens.length - 1);
    if (generated.length === tokens.length - 1 && generated.every((token, idx) => token === tokens[idx + 1])) {
      exactReplays += 1;
    }
  }

  return {
    sequenceCount: evaluableSequences,
    predictedSteps,
    correctSteps,
    nextTokenAccuracy: predictedSteps === 0 ? 0 : correctSteps / predictedSteps,
    exactSequenceReplays: exactReplays,
    exactReplayRate: evaluableSequences === 0 ? 0 : exactReplays / evaluableSequences,
    averageConfidence: predictedSteps === 0 ? 0 : confidenceSum / predictedSteps,
    backoffRate: predictedSteps === 0 ? 0 : backoffCount / predictedSteps,
  };
}

/** A named movement workflow template for the synthetic generator. */
export interface SyntheticMovementWorkflow {
  id: string;
  steps: MovementStep[];
}

/** Spec for {@link generateSyntheticMovementDataset}. */
export interface SyntheticMovementSpec {
  workflows: SyntheticMovementWorkflow[];
  /** How many sequences to emit per workflow (default 1). */
  repetitionsPerWorkflow?: number;
}

/**
 * Deterministically synthesize a movement dataset from workflow templates —
 * used to validate the capture→dataset→train→replay round-trip without real OS
 * input. No RNG: repetitions are labelled by index so runs are reproducible.
 */
export function generateSyntheticMovementDataset(spec: SyntheticMovementSpec): MovementDataset {
  const repetitions = Math.max(1, spec.repetitionsPerWorkflow ?? 1);
  const sequences: MovementSequence[] = [];
  for (const workflow of spec.workflows) {
    for (let rep = 0; rep < repetitions; rep += 1) {
      sequences.push({
        id: repetitions === 1 ? workflow.id : `${workflow.id}#${rep}`,
        tokens: workflow.steps.map((step) => encodeMovementToken(step)),
      });
    }
  }
  return { version: 1, sequences };
}
