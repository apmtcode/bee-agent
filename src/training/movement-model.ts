import type { TrajectoryAction, TrajectorySpan } from "../capture/trajectory.js";

/**
 * Movement-learning model layer (standing objective #2d).
 *
 * The capture subsystem records movement/action trajectories; the runner
 * builds shell plans for real on-device training (mlx/axolotl). This module
 * provides the *pluggable model backend* that sits between them: it turns
 * recorded trajectories into a training dataset, trains a model that can
 * (a) repeat recorded movement sequences and (b) generalize to new-but-related
 * movements, and exposes a stable {@link MovementModelBackend} seam so a real
 * on-device small model can be dropped in later.
 *
 * The default {@link MarkovMovementBackend} is fully deterministic and runs
 * in-cloud with synthetic event streams — no OS access, no randomness — so the
 * whole pipeline can be validated in CI. Generalization is achieved by backing
 * off from an exact (tool+gesture+target) context to a coarse (tool+gesture)
 * context, letting the model propose the right *kind* of movement for a target
 * it has never seen.
 */

/** A single movement, normalized from a recorded {@link TrajectoryAction}. */
export type MovementToken = {
  /** Recording tool that produced the movement (e.g. `"device"`, `"browser"`). */
  tool: string;
  /** Coarse movement kind (e.g. `"tap"`, `"swipe"`, `"type"`, `"scroll"`). */
  gesture: string;
  /** Specific UI target the movement acted on, when known. */
  target?: string;
  /** Movement direction, when applicable (e.g. `"up"`, `"left"`). */
  direction?: string;
};

/** An ordered movement sequence extracted from one trajectory span. */
export type MovementSequence = {
  trajectoryId: string;
  sessionId: string;
  tokens: MovementToken[];
};

/** Training dataset: movement sequences plus a flattened token vocabulary. */
export type MovementDataset = {
  version: 1;
  sequences: MovementSequence[];
  /** Distinct full-key tokens observed across all sequences. */
  vocabulary: MovementToken[];
};

export type MovementTrainingConfig = {
  /** Maximum context length (n-gram order). Defaults to 2. */
  order?: number;
};

/** A prediction for the movement that most likely follows a context. */
export type MovementPrediction = {
  token: MovementToken;
  /** Share of observations for the matched context (0..1). */
  confidence: number;
  /** Context length that produced the match; 0 = unconditional prior. */
  matchedOrder: number;
  /**
   * True when the prediction came from the coarse (tool+gesture) backoff —
   * i.e. the model generalized to a target/context it had not seen exactly.
   */
  generalized: boolean;
};

export type MovementModelStats = {
  backendId: string;
  order: number;
  sequenceCount: number;
  tokenCount: number;
  vocabularySize: number;
};

/** A trained model that can predict and generate movements. */
export interface TrainedMovementModel {
  readonly backendId: string;
  /** Predict the movement most likely to follow `context`, or undefined. */
  predict(context: MovementToken[]): MovementPrediction | undefined;
  /**
   * Autoregressively generate up to `maxSteps` movements starting from `seed`,
   * stopping early when the model has no confident continuation.
   */
  generate(seed: MovementToken[], maxSteps: number): MovementToken[];
  stats(): MovementModelStats;
}

/** Pluggable training backend seam. Swap the impl for a real local model. */
export interface MovementModelBackend {
  readonly id: string;
  train(dataset: MovementDataset, config?: MovementTrainingConfig): TrainedMovementModel;
}

const DEFAULT_ORDER = 2;

/** Full identity key for a token: distinguishes targets and directions. */
export function movementTokenKey(token: MovementToken): string {
  return [token.tool, token.gesture, token.target ?? "", token.direction ?? ""].join("");
}

/** Coarse key: only tool+gesture, used for generalization backoff. */
export function movementGestureKey(token: MovementToken): string {
  return [token.tool, token.gesture].join("");
}

/** Normalize a recorded action into a {@link MovementToken}. */
export function movementTokenFromAction(action: TrajectoryAction): MovementToken {
  const metadata = action.metadata ?? {};
  const gesture = typeof metadata.gesture === "string" ? metadata.gesture : action.tool;
  const target = typeof metadata.target === "string" ? metadata.target : undefined;
  const direction = typeof metadata.direction === "string" ? metadata.direction : undefined;
  return {
    tool: action.tool,
    gesture,
    ...(target !== undefined ? { target } : {}),
    ...(direction !== undefined ? { direction } : {}),
  };
}

/** Build a movement dataset from recorded trajectory spans. */
export function buildMovementDataset(trajectories: TrajectorySpan[]): MovementDataset {
  const sequences: MovementSequence[] = [];
  const vocabulary = new Map<string, MovementToken>();

  for (const trajectory of trajectories) {
    const tokens = [...trajectory.actions]
      .sort((a, b) => a.ts - b.ts)
      .map(movementTokenFromAction);
    if (tokens.length === 0) {
      continue;
    }
    for (const token of tokens) {
      const key = movementTokenKey(token);
      if (!vocabulary.has(key)) {
        vocabulary.set(key, token);
      }
    }
    sequences.push({ trajectoryId: trajectory.id, sessionId: trajectory.sessionId, tokens });
  }

  return {
    version: 1,
    sequences,
    vocabulary: [...vocabulary.values()],
  };
}

type Candidate = { token: MovementToken; count: number };

/**
 * Deterministic n-gram (Markov) movement model with generalization backoff.
 *
 * Serves as the reference/mock backend: no randomness, no OS access. It learns
 * transition frequencies at every order from `n` down to `1` keyed on the full
 * token identity, plus a parallel coarse table keyed on tool+gesture only.
 * Prediction tries exact contexts longest-first, then the coarse contexts
 * (generalizing to unseen targets), then an unconditional prior. Ties break
 * lexicographically on the token key, so results are fully reproducible.
 */
export class MarkovMovementModel implements TrainedMovementModel {
  readonly backendId: string;
  private readonly order: number;
  private readonly full = new Map<string, Map<string, Candidate>>();
  private readonly coarse = new Map<string, Map<string, Candidate>>();
  private readonly prior = new Map<string, Candidate>();
  private sequenceCount = 0;
  private tokenCount = 0;
  private readonly vocabularySize: number;

  constructor(backendId: string, dataset: MovementDataset, order: number) {
    this.backendId = backendId;
    this.order = Math.max(1, Math.floor(order));
    this.vocabularySize = dataset.vocabulary.length;
    for (const sequence of dataset.sequences) {
      this.ingest(sequence.tokens);
    }
  }

  private ingest(tokens: MovementToken[]): void {
    this.sequenceCount += 1;
    this.tokenCount += tokens.length;
    for (let i = 0; i < tokens.length; i += 1) {
      const next = tokens[i]!;
      bump(this.prior, movementTokenKey(next), next);
      for (let k = 1; k <= this.order; k += 1) {
        if (i - k < 0) {
          break;
        }
        const context = tokens.slice(i - k, i);
        bumpNested(this.full, fullContextKey(context), movementTokenKey(next), next);
        bumpNested(this.coarse, coarseContextKey(context), movementGestureKey(next), coarseToken(next));
      }
    }
  }

  predict(context: MovementToken[]): MovementPrediction | undefined {
    // Exact-identity contexts, longest first.
    for (let k = Math.min(this.order, context.length); k >= 1; k -= 1) {
      const suffix = context.slice(context.length - k);
      const table = this.full.get(fullContextKey(suffix));
      const best = argmax(table);
      if (best) {
        return { token: best.token, confidence: best.confidence, matchedOrder: k, generalized: false };
      }
    }
    // Coarse tool+gesture contexts — generalize to unseen targets.
    for (let k = Math.min(this.order, context.length); k >= 1; k -= 1) {
      const suffix = context.slice(context.length - k);
      const table = this.coarse.get(coarseContextKey(suffix));
      const best = argmax(table);
      if (best) {
        return { token: best.token, confidence: best.confidence, matchedOrder: k, generalized: true };
      }
    }
    // Unconditional prior.
    const best = argmax(this.prior);
    if (best) {
      return { token: best.token, confidence: best.confidence, matchedOrder: 0, generalized: context.length > 0 };
    }
    return undefined;
  }

  generate(seed: MovementToken[], maxSteps: number): MovementToken[] {
    const produced: MovementToken[] = [];
    const history = [...seed];
    for (let step = 0; step < Math.max(0, maxSteps); step += 1) {
      const prediction = this.predict(history);
      if (!prediction) {
        break;
      }
      produced.push(prediction.token);
      history.push(prediction.token);
    }
    return produced;
  }

  stats(): MovementModelStats {
    return {
      backendId: this.backendId,
      order: this.order,
      sequenceCount: this.sequenceCount,
      tokenCount: this.tokenCount,
      vocabularySize: this.vocabularySize,
    };
  }
}

/** Default deterministic backend. Suitable for CI and as a real-model stub. */
export class MarkovMovementBackend implements MovementModelBackend {
  readonly id = "markov";

  train(dataset: MovementDataset, config?: MovementTrainingConfig): TrainedMovementModel {
    return new MarkovMovementModel(this.id, dataset, config?.order ?? DEFAULT_ORDER);
  }
}

const BACKENDS = new Map<string, MovementModelBackend>([["markov", new MarkovMovementBackend()]]);

/** Register a pluggable backend (e.g. a real on-device model) by id. */
export function registerMovementBackend(backend: MovementModelBackend): void {
  BACKENDS.set(backend.id, backend);
}

/** Resolve a registered backend, defaulting to the deterministic Markov model. */
export function createMovementBackend(id = "markov"): MovementModelBackend {
  const backend = BACKENDS.get(id);
  if (!backend) {
    throw new Error(`unknown movement backend: ${id}`);
  }
  return backend;
}

/** Ids of all registered backends, sorted for stable listing. */
export function listMovementBackends(): string[] {
  return [...BACKENDS.keys()].sort();
}

function coarseToken(token: MovementToken): MovementToken {
  return { tool: token.tool, gesture: token.gesture };
}

function fullContextKey(context: MovementToken[]): string {
  return context.map(movementTokenKey).join("");
}

function coarseContextKey(context: MovementToken[]): string {
  return context.map(movementGestureKey).join("");
}

function bump(table: Map<string, Candidate>, key: string, token: MovementToken): void {
  const existing = table.get(key);
  if (existing) {
    existing.count += 1;
  } else {
    table.set(key, { token, count: 1 });
  }
}

function bumpNested(
  table: Map<string, Map<string, Candidate>>,
  contextKey: string,
  tokenKey: string,
  token: MovementToken,
): void {
  let inner = table.get(contextKey);
  if (!inner) {
    inner = new Map<string, Candidate>();
    table.set(contextKey, inner);
  }
  bump(inner, tokenKey, token);
}

function argmax(
  table: Map<string, Candidate> | undefined,
): { token: MovementToken; confidence: number } | undefined {
  if (!table || table.size === 0) {
    return undefined;
  }
  let total = 0;
  let best: { key: string; candidate: Candidate } | undefined;
  for (const [key, candidate] of table) {
    total += candidate.count;
    if (
      !best ||
      candidate.count > best.candidate.count ||
      (candidate.count === best.candidate.count && key < best.key)
    ) {
      best = { key, candidate };
    }
  }
  if (!best) {
    return undefined;
  }
  return { token: best.candidate.token, confidence: best.candidate.count / total };
}
