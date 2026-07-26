import type { TrajectoryAction, TrajectorySpan } from "../capture/trajectory.js";
import type { ReplayTimelineEvent } from "../capture/replay.js";

/**
 * Movement-policy learning: an in-cloud-trainable, pluggable model that learns
 * recorded movement sequences, **repeats** them faithfully, and **generalizes**
 * to new-but-related movements.
 *
 * This is the train/infer half of the local-movement learning subsystem
 * (standing objective #2, pieces (c)/(d)). Unlike {@link
 * ../training/runner.ts LocalAppleSiliconTrainingRunner} — which emits launch
 * scripts for real on-device MLX/Axolotl training — this module runs fully in
 * the cloud with no OS/GPU access, so the whole capture -> dataset -> train ->
 * infer -> replay loop can be validated with synthetic event streams and unit
 * tests. The backend is pluggable: the deterministic n-gram backend shipped
 * here is the default/mock, and {@link registerMovementPolicyBackend} is the
 * documented seam for a real on-device neural backend.
 */

const FIELD_SEP = "␟";
/** Sentinel target that marks the end of a recorded movement sequence. */
export const MOVEMENT_END = "__end__";

/**
 * A single discrete movement, normalized from a captured trajectory action.
 * `tool` is the effector (e.g. "device", "keyboard"), `gesture` the verb
 * (e.g. "tap", "type", "click"), `target` the concrete object acted on, and
 * `direction` an optional direction for directional gestures.
 */
export type MovementToken = {
  tool: string;
  gesture: string;
  target?: string;
  direction?: string;
};

export type MovementPrediction = {
  /** The predicted next movement, or the end sentinel when the sequence should stop. */
  token: MovementToken | "__end__";
  /** P(token | context) within the matched context, in [0, 1]. */
  confidence: number;
  /** Raw occurrence count backing the prediction. */
  support: number;
  /** Length of the context that produced the match (higher = more specific). */
  order: number;
  /**
   * True when the prediction came from the coarse feature model (target
   * dropped) rather than an exact token-sequence match — i.e. the model
   * generalized to a new-but-related movement it never saw in this exact
   * context.
   */
  generalized: boolean;
};

export type MovementTrainOptions = {
  /** Maximum n-gram order (context length + 1). Defaults to 3. */
  maxOrder?: number;
};

/** JSON-serializable trained model. Plain data — persist with `JSON.stringify`. */
export type MovementPolicyModel = {
  version: 1;
  backend: string;
  maxOrder: number;
  trainedSequences: number;
  trainedTokens: number;
  /** contextKey -> tokenKey -> count (exact token sequences). */
  transitions: Record<string, Record<string, number>>;
  /** coarse contextKey -> featureKey -> count (target-agnostic generalization). */
  featureTransitions: Record<string, Record<string, number>>;
  /** tokenKey -> the concrete movement it decodes to. */
  tokens: Record<string, MovementToken>;
  /** featureKey -> tokenKey -> count, to resolve a predicted feature to a concrete token. */
  featureTokens: Record<string, Record<string, number>>;
};

/**
 * Pluggable movement-policy backend. The default is a deterministic n-gram
 * model; a real on-device small model can implement the same interface and be
 * registered via {@link registerMovementPolicyBackend}.
 */
export interface MovementPolicyBackend {
  readonly id: string;
  train(sequences: MovementToken[][], options?: MovementTrainOptions): MovementPolicyModel;
  predictNext(model: MovementPolicyModel, context: MovementToken[]): MovementPrediction | undefined;
}

// --- tokenization -----------------------------------------------------------

function asString(value: unknown): string | undefined {
  return typeof value === "string" && value.length > 0 ? value : undefined;
}

function deriveGesture(summary: string): string {
  const first = summary.trim().toLowerCase().split(/\s+/)[0];
  return first && first.length > 0 ? first : "act";
}

/** Normalize a captured trajectory action into a movement token. */
export function movementTokenFromAction(action: TrajectoryAction): MovementToken {
  const metadata = action.metadata ?? {};
  const gesture = asString(metadata.gesture) ?? deriveGesture(action.summary);
  const target = asString(metadata.target);
  const direction = asString(metadata.direction);
  return {
    tool: action.tool,
    gesture,
    ...(target ? { target } : {}),
    ...(direction ? { direction } : {}),
  };
}

/** Ordered movement sequence for one trajectory (actions sorted by timestamp). */
export function extractMovementSequence(trajectory: TrajectorySpan): MovementToken[] {
  return [...trajectory.actions]
    .sort((a, b) => a.ts - b.ts)
    .map((action) => movementTokenFromAction(action));
}

/** Ordered movement sequence from a replay manifest's timeline events. */
export function movementSequenceFromReplayEvents(events: ReplayTimelineEvent[]): MovementToken[] {
  return events
    .filter((event): event is Extract<ReplayTimelineEvent, { kind: "action" }> => event.kind === "action")
    .sort((a, b) => a.ts - b.ts)
    .map((event) => ({ tool: event.tool, gesture: deriveGesture(event.summary) }));
}

/** Stable full-fidelity key for a movement token (used for exact matches). */
export function movementKey(token: MovementToken): string {
  return [token.tool, token.gesture, token.target ?? "", token.direction ?? ""].join(FIELD_SEP);
}

/** Coarse, target-agnostic key so the model can generalize across targets. */
export function movementFeatureKey(token: MovementToken): string {
  return [token.tool, token.gesture, token.direction ?? ""].join(FIELD_SEP);
}

function contextKey(tokens: MovementToken[], keyer: (token: MovementToken) => string): string {
  return tokens.map(keyer).join("␄");
}

function argmax(counts: Record<string, number>): { key: string; count: number; total: number } | undefined {
  let bestKey: string | undefined;
  let bestCount = -1;
  let total = 0;
  for (const [key, count] of Object.entries(counts)) {
    total += count;
    // Deterministic tie-break: higher count wins, then lexically smaller key.
    if (count > bestCount || (count === bestCount && bestKey !== undefined && key < bestKey)) {
      bestKey = key;
      bestCount = count;
    }
  }
  return bestKey === undefined ? undefined : { key: bestKey, count: bestCount, total };
}

// --- n-gram backend ---------------------------------------------------------

export class NGramMovementBackend implements MovementPolicyBackend {
  readonly id = "ngram";

  train(sequences: MovementToken[][], options?: MovementTrainOptions): MovementPolicyModel {
    const maxOrder = Math.max(1, options?.maxOrder ?? 3);
    const model: MovementPolicyModel = {
      version: 1,
      backend: this.id,
      maxOrder,
      trainedSequences: 0,
      trainedTokens: 0,
      transitions: {},
      featureTransitions: {},
      tokens: {},
      featureTokens: {},
    };

    for (const rawSequence of sequences) {
      if (rawSequence.length === 0) {
        continue;
      }
      model.trainedSequences += 1;
      // Append the end sentinel so the model learns where sequences stop.
      const sequence: (MovementToken | "__end__")[] = [...rawSequence, MOVEMENT_END];

      for (let index = 0; index < sequence.length; index += 1) {
        const target = sequence[index]!;
        const targetKey = target === MOVEMENT_END ? MOVEMENT_END : movementKey(target);
        const targetFeatureKey = target === MOVEMENT_END ? MOVEMENT_END : movementFeatureKey(target);
        if (target !== MOVEMENT_END) {
          model.trainedTokens += 1;
          model.tokens[targetKey] = target;
          const bucket = (model.featureTokens[targetFeatureKey] ??= {});
          bucket[targetKey] = (bucket[targetKey] ?? 0) + 1;
        }

        // Register every context length from 1..maxOrder-1 that fits before index.
        const priorMovements = sequence.slice(0, index).filter((token): token is MovementToken => token !== MOVEMENT_END);
        for (let order = 1; order <= maxOrder - 1; order += 1) {
          if (priorMovements.length < order) {
            break;
          }
          const window = priorMovements.slice(priorMovements.length - order);
          const exactCtx = contextKey(window, movementKey);
          const featureCtx = contextKey(window, movementFeatureKey);
          const exactBucket = (model.transitions[exactCtx] ??= {});
          exactBucket[targetKey] = (exactBucket[targetKey] ?? 0) + 1;
          const featureBucket = (model.featureTransitions[featureCtx] ??= {});
          featureBucket[targetFeatureKey] = (featureBucket[targetFeatureKey] ?? 0) + 1;
        }
      }
    }

    return model;
  }

  predictNext(model: MovementPolicyModel, context: MovementToken[]): MovementPrediction | undefined {
    if (model.trainedTokens === 0) {
      return undefined;
    }
    const maxContext = Math.min(model.maxOrder - 1, context.length);

    // 1. Exact token-sequence backoff (most specific first). This reproduces
    //    recorded movements faithfully.
    for (let order = maxContext; order >= 1; order -= 1) {
      const window = context.slice(context.length - order);
      const best = argmax(model.transitions[contextKey(window, movementKey)] ?? {});
      if (best) {
        return {
          token: best.key === MOVEMENT_END ? MOVEMENT_END : model.tokens[best.key]!,
          confidence: best.total > 0 ? best.count / best.total : 0,
          support: best.count,
          order,
          generalized: false,
        };
      }
    }

    // 2. Coarse feature backoff: the exact context was never seen, but a
    //    structurally-similar (target-agnostic) context was — generalize to a
    //    new-but-related movement.
    for (let order = maxContext; order >= 1; order -= 1) {
      const window = context.slice(context.length - order);
      const best = argmax(model.featureTransitions[contextKey(window, movementFeatureKey)] ?? {});
      if (best) {
        if (best.key === MOVEMENT_END) {
          return { token: MOVEMENT_END, confidence: best.total > 0 ? best.count / best.total : 0, support: best.count, order, generalized: true };
        }
        const concrete = argmax(model.featureTokens[best.key] ?? {});
        if (concrete && model.tokens[concrete.key]) {
          return {
            token: model.tokens[concrete.key]!,
            confidence: best.total > 0 ? best.count / best.total : 0,
            support: best.count,
            order,
            generalized: true,
          };
        }
      }
    }

    // 3. Global fallback: most frequent movement overall.
    const global = argmax(
      Object.fromEntries(
        Object.entries(model.featureTokens).flatMap(([, byToken]) => Object.entries(byToken)),
      ),
    );
    if (global && model.tokens[global.key]) {
      return {
        token: model.tokens[global.key]!,
        confidence: global.total > 0 ? global.count / global.total : 0,
        support: global.count,
        order: 0,
        generalized: true,
      };
    }
    return undefined;
  }
}

// --- rollout + evaluation ---------------------------------------------------

export type MovementRolloutStep = MovementPrediction & { token: MovementToken };

/**
 * Autoregressively roll out a movement sequence from a seed prefix: repeatedly
 * predict the next movement and append it, stopping at the end sentinel or
 * `maxSteps`. With a deterministic model this reproduces recorded sequences and
 * extends related ones.
 */
export function rolloutMovements(
  backend: MovementPolicyBackend,
  model: MovementPolicyModel,
  seed: MovementToken[] = [],
  options?: { maxSteps?: number },
): MovementRolloutStep[] {
  const maxSteps = options?.maxSteps ?? 64;
  const context = [...seed];
  const steps: MovementRolloutStep[] = [];
  for (let i = 0; i < maxSteps; i += 1) {
    const prediction = backend.predictNext(model, context);
    if (!prediction || prediction.token === MOVEMENT_END) {
      break;
    }
    steps.push({ ...prediction, token: prediction.token });
    context.push(prediction.token);
  }
  return steps;
}

/**
 * Next-movement prediction accuracy over held-out sequences — the generalization
 * eval harness. For each position the model predicts the next token from the
 * true prefix; accuracy is the fraction of exact-key matches.
 */
export function evaluateNextMovementAccuracy(
  backend: MovementPolicyBackend,
  model: MovementPolicyModel,
  heldOut: MovementToken[][],
): { predictions: number; correct: number; accuracy: number } {
  let predictions = 0;
  let correct = 0;
  for (const sequence of heldOut) {
    for (let index = 0; index < sequence.length; index += 1) {
      predictions += 1;
      const prediction = backend.predictNext(model, sequence.slice(0, index));
      const expected = movementKey(sequence[index]!);
      if (prediction && prediction.token !== MOVEMENT_END && movementKey(prediction.token) === expected) {
        correct += 1;
      }
    }
  }
  return { predictions, correct, accuracy: predictions > 0 ? correct / predictions : 0 };
}

// --- pluggable backend registry ---------------------------------------------

const backendRegistry = new Map<string, () => MovementPolicyBackend>([
  ["ngram", () => new NGramMovementBackend()],
]);

/**
 * Register a movement-policy backend factory (e.g. a real on-device neural
 * model). Additive seam — the deterministic n-gram backend remains the default.
 */
export function registerMovementPolicyBackend(id: string, factory: () => MovementPolicyBackend): void {
  backendRegistry.set(id, factory);
}

export function createMovementPolicyBackend(id: string = "ngram"): MovementPolicyBackend {
  const factory = backendRegistry.get(id);
  if (!factory) {
    throw new Error(`unknown movement-policy backend: ${id}`);
  }
  return factory();
}

export function listMovementPolicyBackends(): string[] {
  return [...backendRegistry.keys()];
}

/** Round-trip persistence guard for a serialized model. */
export function loadMovementPolicyModel(json: string): MovementPolicyModel {
  const parsed = JSON.parse(json) as MovementPolicyModel;
  if (parsed?.version !== 1 || typeof parsed.transitions !== "object") {
    throw new Error("invalid movement-policy model");
  }
  return parsed;
}
