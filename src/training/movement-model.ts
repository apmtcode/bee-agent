import type { ReplayManifest } from "../capture/replay.js";
import type { TrajectoryAction, TrajectorySpan } from "../capture/trajectory.js";

/**
 * In-process movement-learning model.
 *
 * This is the deterministic, cloud-runnable backend for standing objective #2:
 * post-train a local model on recorded movement trajectories so it can (c)
 * repeat the recorded movements and (d) generalize to new-but-related
 * movements. The heavy on-device runtimes (MLX / Axolotl) are launched by
 * {@link LocalAppleSiliconTrainingRunner}; this module provides an in-process
 * model that trains and infers with no external process, so the whole
 * capture -> dataset -> train -> replay pipeline can be validated with
 * synthetic event streams in tests.
 *
 * The backend is pluggable ({@link MovementModelBackend}): the shipped
 * implementation is an n-gram model with Katz-style backoff, which
 * generalizes to unseen contexts by falling back to shorter contexts and,
 * ultimately, the global movement distribution. A real small on-device model
 * can register itself under {@link MOVEMENT_BACKENDS} without changing callers.
 */

/** A single recorded movement (mouse/keyboard/gesture/window action). */
export type MovementToken = {
  /** Origin channel, e.g. "device", "browser", "os", "mouse", "keyboard". */
  channel: string;
  /** Coarse action verb, e.g. "tap", "swipe", "scroll", "type", "click". */
  action: string;
  /** Optional coarse target label (button/field name, element role). */
  target?: string;
  /** Optional cardinal direction for swipes/scrolls. */
  direction?: string;
};

/** An ordered sequence of movements that belong to one trajectory. */
export type MovementSequence = {
  id: string;
  tokens: MovementToken[];
};

/** A replayable, trainable collection of movement sequences. */
export type MovementDataset = {
  sequences: MovementSequence[];
};

export type TrainMovementModelOptions = {
  /**
   * Model order: the number of tokens (self + prior context) the model
   * conditions on. Order 3 conditions each prediction on up to 2 prior
   * movements. Clamped to >= 1.
   */
  order?: number;
};

export type MovementPredictionCandidate = {
  key: string;
  token: MovementToken;
  count: number;
  probability: number;
};

export type MovementPrediction = {
  token: MovementToken;
  key: string;
  probability: number;
  /**
   * How many prior tokens the winning context matched (backoff depth). A
   * value below `order - 1` means the model generalized past an unseen
   * context by backing off to a shorter one. 0 = global unigram fallback.
   */
  contextOrderUsed: number;
  candidates: MovementPredictionCandidate[];
};

export type MovementRolloutOptions = {
  /** Maximum number of movements to generate. Defaults to 32. */
  maxSteps?: number;
};

export type MovementPolicySnapshot = {
  version: 1;
  backendId: string;
  order: number;
  tokens: Array<{ key: string; token: MovementToken }>;
  /** Backoff levels, index = context length (0 = unigram). */
  levels: Array<Array<{ context: string; next: Array<{ key: string; count: number }> }>>;
};

/** A trained, queryable movement policy. */
export interface MovementPolicy {
  readonly backendId: string;
  readonly order: number;
  readonly vocabulary: readonly string[];
  /** Predict the next movement given prior context (may be empty). */
  predictNext(context: MovementToken[]): MovementPrediction | undefined;
  /** Autoregressively generate a continuation from a seed context. */
  rollout(seed: MovementToken[], options?: MovementRolloutOptions): MovementToken[];
  /** Serialize to a plain JSON snapshot for persistence. */
  snapshot(): MovementPolicySnapshot;
}

/** A pluggable training backend. */
export interface MovementModelBackend {
  readonly id: string;
  train(dataset: MovementDataset, options?: TrainMovementModelOptions): Promise<MovementPolicy>;
}

const CONTEXT_SEPARATOR = "\u001f";
const FIELD_SEPARATOR = "\u001e";

const START_TOKEN: MovementToken = { channel: "\u0000meta", action: "start" };
const END_TOKEN: MovementToken = { channel: "\u0000meta", action: "end" };
const START_KEY = movementTokenKey(START_TOKEN);
const END_KEY = movementTokenKey(END_TOKEN);

/** Canonical, reversible-by-index key for a movement token. */
export function movementTokenKey(token: MovementToken): string {
  return [token.channel, token.action, token.direction ?? "", token.target ?? ""].join(FIELD_SEPARATOR);
}

export function isMovementEndToken(token: MovementToken): boolean {
  return movementTokenKey(token) === END_KEY;
}

function isSentinelKey(key: string): boolean {
  return key === START_KEY || key === END_KEY;
}

/** N-gram movement backend with Katz-style backoff generalization. */
export class NGramMovementBackend implements MovementModelBackend {
  readonly id = "ngram";

  constructor(private readonly defaultOrder = 3) {}

  async train(dataset: MovementDataset, options?: TrainMovementModelOptions): Promise<MovementPolicy> {
    const order = Math.max(1, Math.trunc(options?.order ?? this.defaultOrder));
    const tokenIndex = new Map<string, MovementToken>();
    const levels: Array<Map<string, Map<string, number>>> = Array.from({ length: order }, () => new Map());

    for (const sequence of dataset.sequences) {
      const padded: MovementToken[] = [
        ...Array.from({ length: order - 1 }, () => START_TOKEN),
        ...sequence.tokens,
        END_TOKEN,
      ];
      const keys = padded.map(movementTokenKey);

      for (let i = order - 1; i < padded.length; i += 1) {
        const nextKey = keys[i];
        // Every token that can be produced (real move or END) is indexed;
        // START is only ever context, never a prediction target.
        if (!tokenIndex.has(nextKey)) {
          tokenIndex.set(nextKey, padded[i]);
        }
        for (let level = 0; level < order; level += 1) {
          const context = keys.slice(i - level, i).join(CONTEXT_SEPARATOR);
          const transitions = levels[level];
          let counts = transitions.get(context);
          if (!counts) {
            counts = new Map();
            transitions.set(context, counts);
          }
          counts.set(nextKey, (counts.get(nextKey) ?? 0) + 1);
        }
      }
    }

    return new NGramMovementPolicy(this.id, order, levels, tokenIndex);
  }
}

class NGramMovementPolicy implements MovementPolicy {
  readonly vocabulary: readonly string[];

  constructor(
    readonly backendId: string,
    readonly order: number,
    private readonly levels: Array<Map<string, Map<string, number>>>,
    private readonly tokenIndex: Map<string, MovementToken>,
  ) {
    this.vocabulary = [...tokenIndex.keys()].filter((key) => !isSentinelKey(key)).sort();
  }

  predictNext(context: MovementToken[]): MovementPrediction | undefined {
    const window = Math.max(0, this.order - 1);
    const padded = padContext(context, window);
    const paddedKeys = padded.map(movementTokenKey);

    for (let level = window; level >= 0; level -= 1) {
      const transitions = this.levels[level];
      if (!transitions) {
        continue;
      }
      const contextKey = paddedKeys.slice(paddedKeys.length - level).join(CONTEXT_SEPARATOR);
      const counts = transitions.get(contextKey);
      if (!counts || counts.size === 0) {
        continue;
      }
      const total = [...counts.values()].reduce((sum, value) => sum + value, 0);
      const ranked = [...counts.entries()].sort(compareCandidate);
      const candidates: MovementPredictionCandidate[] = ranked.map(([key, count]) => ({
        key,
        token: this.tokenIndex.get(key) ?? END_TOKEN,
        count,
        probability: count / total,
      }));
      const best = candidates[0];
      return {
        token: best.token,
        key: best.key,
        probability: best.probability,
        contextOrderUsed: level,
        candidates,
      };
    }
    return undefined;
  }

  rollout(seed: MovementToken[], options?: MovementRolloutOptions): MovementToken[] {
    const maxSteps = Math.max(0, options?.maxSteps ?? 32);
    const generated: MovementToken[] = [];
    const current = [...seed];
    for (let step = 0; step < maxSteps; step += 1) {
      const prediction = this.predictNext(current);
      if (!prediction || prediction.key === END_KEY) {
        break;
      }
      generated.push(prediction.token);
      current.push(prediction.token);
    }
    return generated;
  }

  snapshot(): MovementPolicySnapshot {
    return {
      version: 1,
      backendId: this.backendId,
      order: this.order,
      tokens: [...this.tokenIndex.entries()].map(([key, token]) => ({ key, token })),
      levels: this.levels.map((transitions) =>
        [...transitions.entries()].map(([context, counts]) => ({
          context,
          next: [...counts.entries()].map(([key, count]) => ({ key, count })),
        })),
      ),
    };
  }

  static fromSnapshot(snapshot: MovementPolicySnapshot): MovementPolicy {
    const tokenIndex = new Map<string, MovementToken>(snapshot.tokens.map((entry) => [entry.key, entry.token]));
    const levels = snapshot.levels.map((level) => {
      const transitions = new Map<string, Map<string, number>>();
      for (const entry of level) {
        transitions.set(entry.context, new Map(entry.next.map((next) => [next.key, next.count])));
      }
      return transitions;
    });
    return new NGramMovementPolicy(snapshot.backendId, snapshot.order, levels, tokenIndex);
  }
}

export function movementPolicyFromSnapshot(snapshot: MovementPolicySnapshot): MovementPolicy {
  return NGramMovementPolicy.fromSnapshot(snapshot);
}

function padContext(context: MovementToken[], size: number): MovementToken[] {
  if (size <= 0) {
    return [];
  }
  const trimmed = context.slice(-size);
  const padCount = size - trimmed.length;
  if (padCount <= 0) {
    return trimmed;
  }
  return [...Array.from({ length: padCount }, () => START_TOKEN), ...trimmed];
}

function compareCandidate(a: [string, number], b: [string, number]): number {
  if (a[1] !== b[1]) {
    return b[1] - a[1];
  }
  return a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0;
}

// ---------------------------------------------------------------------------
// Pluggable backend registry
// ---------------------------------------------------------------------------

export type MovementBackendFactory = () => MovementModelBackend;

/**
 * Registry of movement-model backends. The deterministic in-process `ngram`
 * backend ships by default; a real on-device small-model backend can register
 * itself here so {@link createMovementBackend} callers pick it up without
 * code changes.
 */
export const MOVEMENT_BACKENDS: Record<string, MovementBackendFactory> = {
  ngram: () => new NGramMovementBackend(),
};

export function registerMovementBackend(id: string, factory: MovementBackendFactory): void {
  MOVEMENT_BACKENDS[id] = factory;
}

export function createMovementBackend(id = "ngram"): MovementModelBackend {
  const factory = MOVEMENT_BACKENDS[id];
  if (!factory) {
    throw new Error(`unknown movement backend: ${id}`);
  }
  return factory();
}

// ---------------------------------------------------------------------------
// Capture-pipeline bridges (trajectory / replay -> movement dataset)
// ---------------------------------------------------------------------------

export function movementTokenFromAction(action: TrajectoryAction): MovementToken {
  const metadata = action.metadata ?? {};
  const gesture = typeof metadata["gesture"] === "string" ? (metadata["gesture"] as string) : undefined;
  const direction = typeof metadata["direction"] === "string" ? (metadata["direction"] as string) : undefined;
  const target = typeof metadata["target"] === "string" ? (metadata["target"] as string) : undefined;
  return {
    channel: action.tool,
    action: gesture ?? firstWord(action.summary),
    ...(direction ? { direction } : {}),
    ...(target ? { target } : {}),
  };
}

export function movementSequenceFromTrajectory(trajectory: TrajectorySpan): MovementSequence {
  const tokens = [...trajectory.actions]
    .sort((a, b) => a.ts - b.ts)
    .map(movementTokenFromAction);
  return { id: trajectory.id, tokens };
}

export function movementDatasetFromTrajectories(trajectories: TrajectorySpan[]): MovementDataset {
  return {
    sequences: trajectories
      .map(movementSequenceFromTrajectory)
      .filter((sequence) => sequence.tokens.length > 0),
  };
}

export function movementSequenceFromReplayManifest(manifest: ReplayManifest): MovementSequence {
  const tokens = manifest.events
    .filter((event): event is Extract<typeof event, { kind: "action" }> => event.kind === "action")
    .map((event) => ({
      channel: event.tool,
      action: firstWord(event.summary),
    }));
  return { id: manifest.sessionId, tokens };
}

export function movementDatasetFromReplayManifests(manifests: ReplayManifest[]): MovementDataset {
  return {
    sequences: manifests
      .map(movementSequenceFromReplayManifest)
      .filter((sequence) => sequence.tokens.length > 0),
  };
}

function firstWord(summary: string): string {
  const trimmed = summary.trim();
  if (!trimmed) {
    return "unknown";
  }
  return trimmed.split(/\s+/)[0].toLowerCase();
}

// ---------------------------------------------------------------------------
// Generalization / replay-fidelity evaluation harness
// ---------------------------------------------------------------------------

export type MovementFidelityResult = {
  sequenceId: string;
  promptLength: number;
  expected: MovementToken[];
  predicted: MovementToken[];
  matched: number;
  fidelity: number;
};

export type MovementEvaluation = {
  perSequence: MovementFidelityResult[];
  averageFidelity: number;
  exactSequences: number;
};

export type EvaluateMovementPolicyOptions = {
  /** Fraction of each sequence used as the prompt. Defaults to 0.5. */
  promptRatio?: number;
};

/**
 * Measure how well a trained policy reproduces the continuation of held-out
 * sequences: prompt it with a prefix and compare its rollout, position by
 * position, against the true continuation. Held-out sequences exercise
 * generalization — the model has not seen them during training.
 */
export function evaluateMovementPolicy(
  policy: MovementPolicy,
  heldOut: MovementSequence[],
  options?: EvaluateMovementPolicyOptions,
): MovementEvaluation {
  const promptRatio = clamp(options?.promptRatio ?? 0.5, 0, 1);
  const perSequence: MovementFidelityResult[] = [];

  for (const sequence of heldOut) {
    if (sequence.tokens.length === 0) {
      continue;
    }
    const promptLength = Math.min(
      sequence.tokens.length - 1,
      Math.max(0, Math.floor(sequence.tokens.length * promptRatio)),
    );
    const prompt = sequence.tokens.slice(0, promptLength);
    const expected = sequence.tokens.slice(promptLength);
    const predicted = policy.rollout(prompt, { maxSteps: expected.length });
    let matched = 0;
    for (let i = 0; i < expected.length; i += 1) {
      const predictedToken = predicted[i];
      if (predictedToken && movementTokenKey(predictedToken) === movementTokenKey(expected[i])) {
        matched += 1;
      }
    }
    perSequence.push({
      sequenceId: sequence.id,
      promptLength,
      expected,
      predicted,
      matched,
      fidelity: expected.length === 0 ? 1 : matched / expected.length,
    });
  }

  const averageFidelity =
    perSequence.length === 0
      ? 0
      : perSequence.reduce((sum, result) => sum + result.fidelity, 0) / perSequence.length;
  const exactSequences = perSequence.filter((result) => result.fidelity === 1).length;

  return { perSequence, averageFidelity, exactSequences };
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}
