import type { ReplayTimelineEvent } from "../capture/replay.js";
import type { TrajectorySpan } from "../capture/trajectory.js";

/**
 * In-process, pluggable movement-model backend.
 *
 * Standing objective #2 (local-movement learning) needs bee-agent to (c) train a
 * local model on recorded movement trajectories and (d) generalize to new but
 * related movements. The existing {@link LocalAppleSiliconTrainingRunner} only
 * *emits shell plans* for real on-device MLX/axolotl training — nothing that can
 * learn or infer inside the process, so those pieces cannot be validated in the
 * cloud/CI.
 *
 * This module closes that gap with a small, deterministic, dependency-free
 * backend that actually learns from movement sequences and predicts the next
 * movement, behind a pluggable {@link MovementModelBackend} interface so a real
 * on-device small model can be dropped in later without touching call sites.
 */

/** A normalized movement primitive the model learns over (e.g. `device::tapped submit`). */
export type MovementToken = string;

/** Sentinel framing the start of a recorded sequence, so the first movement is conditioned. */
export const MOVEMENT_START_TOKEN: MovementToken = "__movement_start__";

/** Sentinel marking the end of a recorded sequence, so the model can learn to stop. */
export const MOVEMENT_END_TOKEN: MovementToken = "__movement_end__";

/** A single normalized movement extracted from a trajectory action or replay event. */
export type MovementEvent = {
  tool: string;
  summary: string;
  token: MovementToken;
};

/** One recorded movement sequence (typically one trajectory's ordered actions). */
export type MovementSequence = {
  trajectoryId?: string;
  tokens: MovementToken[];
};

/** The dataset a backend trains on. */
export type MovementDataset = {
  sequences: MovementSequence[];
};

export type MovementTrainOptions = {
  /** Max n-gram order (context length + 1). Higher = more precise, less general. Default 3. */
  order?: number;
};

export type MovementPredictOptions = {
  /** If set, predictions whose confidence is below this are suppressed (returns undefined). */
  minConfidence?: number;
};

export type MovementGenerateOptions = {
  maxSteps?: number;
  minConfidence?: number;
};

/** How a prediction was produced — used for observability and eval. */
export type MovementPredictionSource = "ngram" | "family" | "prior";

export type MovementPrediction = {
  token: MovementToken;
  /** Empirical probability of this token given the matched context. */
  confidence: number;
  /** The n-gram order actually used after backoff (0 = unigram prior). */
  order: number;
  source: MovementPredictionSource;
};

/** Serializable, replayable snapshot of a trained policy. */
export type MovementPolicySnapshot = {
  version: 1;
  backend: string;
  order: number;
  /** context-key -> (token -> count). The empty string key holds the unigram prior. */
  transitions: Array<[string, Array<[MovementToken, number]>]>;
  /** family-signature -> (next-token -> count), the generalization fallback for unseen tokens. */
  families: Array<[string, Array<[MovementToken, number]>]>;
};

/** A trained policy: predict, generate, and serialize. Backend-agnostic. */
export interface MovementPolicy {
  readonly backend: string;
  predict(context: MovementToken[], options?: MovementPredictOptions): MovementPrediction | undefined;
  generate(seed: MovementToken[], options?: MovementGenerateOptions): MovementToken[];
  serialize(): MovementPolicySnapshot;
}

/** A pluggable training backend. Register real on-device backends alongside the mock. */
export interface MovementModelBackend {
  readonly name: string;
  train(dataset: MovementDataset, options?: MovementTrainOptions): MovementPolicy;
  restore(snapshot: MovementPolicySnapshot): MovementPolicy;
}

// --- Normalization / extraction ---------------------------------------------

/** Collapse a free-form action summary into a stable, comparable form. */
export function normalizeMovementSummary(summary: string): string {
  return summary
    .trim()
    .toLowerCase()
    .replace(/\s+/g, " ")
    .replace(/[.,;:!?]+$/g, "");
}

/** Build the canonical token for a (tool, summary) movement. */
export function movementToken(tool: string, summary: string): MovementToken {
  return `${tool.trim().toLowerCase()}::${normalizeMovementSummary(summary)}`;
}

/** Recover the tool portion of a token. */
export function movementTokenTool(token: MovementToken): string {
  const index = token.indexOf("::");
  return index === -1 ? token : token.slice(0, index);
}

/**
 * The generalization signature of a token: its tool plus the leading verb of the
 * summary (e.g. `device::type` for `device::type a new body`). Movements that
 * share a signature are treated as "related", so a policy can transfer a learned
 * continuation to an unseen-but-related movement.
 */
export function movementFamily(token: MovementToken): string {
  const separator = token.indexOf("::");
  if (separator === -1) {
    return token;
  }
  const tool = token.slice(0, separator);
  const summary = token.slice(separator + 2);
  const verb = summary.split(" ", 1)[0] ?? "";
  return verb ? `${tool}::${verb}` : tool;
}

/** Extract an ordered movement sequence from a trajectory's actions (chronological). */
export function extractMovementSequence(trajectory: TrajectorySpan): MovementSequence {
  const tokens = [...trajectory.actions]
    .sort((a, b) => a.ts - b.ts)
    .map((action) => movementToken(action.tool, action.summary));
  return { trajectoryId: trajectory.id, tokens };
}

/** Extract a movement sequence from a replay manifest's action events. */
export function movementSequenceFromReplayEvents(events: ReplayTimelineEvent[]): MovementSequence {
  const actions = events.filter(
    (event): event is Extract<ReplayTimelineEvent, { kind: "action" }> => event.kind === "action",
  );
  const sorted = [...actions].sort((a, b) => a.ts - b.ts);
  return {
    trajectoryId: sorted[0]?.trajectoryId,
    tokens: sorted.map((action) => movementToken(action.tool, action.summary)),
  };
}

/** Build a training dataset from trajectory spans. Empty sequences are dropped. */
export function buildMovementDataset(trajectories: TrajectorySpan[]): MovementDataset {
  return {
    sequences: trajectories
      .map(extractMovementSequence)
      .filter((sequence) => sequence.tokens.length > 0),
  };
}

// --- Deterministic n-gram backend -------------------------------------------

const DEFAULT_ORDER = 3;

type CountTable = Map<string, Map<MovementToken, number>>;

function bump(table: CountTable, key: string, token: MovementToken): void {
  let row = table.get(key);
  if (!row) {
    row = new Map();
    table.set(key, row);
  }
  row.set(token, (row.get(token) ?? 0) + 1);
}

// A control char that never appears in a `tool::summary` token, so joined
// context keys of different token tuples can never collide.
const CONTEXT_DELIMITER = "\\u001f";

function contextKey(context: MovementToken[]): string {
  return context.join(CONTEXT_DELIMITER);
}

/** Deterministic argmax over a count row: highest count wins, ties broken by token order. */
function argmax(row: Map<MovementToken, number>): { token: MovementToken; count: number; total: number } | undefined {
  let best: MovementToken | undefined;
  let bestCount = -1;
  let total = 0;
  for (const [token, count] of row) {
    total += count;
    if (count > bestCount || (count === bestCount && best !== undefined && token < best)) {
      best = token;
      bestCount = count;
    }
  }
  if (best === undefined) {
    return undefined;
  }
  return { token: best, count: bestCount, total };
}

class NgramMovementPolicy implements MovementPolicy {
  readonly backend = "ngram";

  constructor(
    private readonly order: number,
    private readonly transitions: CountTable,
    private readonly families: CountTable,
  ) {}

  predict(context: MovementToken[], options?: MovementPredictOptions): MovementPrediction | undefined {
    const prediction = this.predictInternal(context);
    if (!prediction) {
      return undefined;
    }
    if (options?.minConfidence !== undefined && prediction.confidence < options.minConfidence) {
      return undefined;
    }
    return prediction;
  }

  private predictInternal(context: MovementToken[]): MovementPrediction | undefined {
    // Frame with the start sentinel so the first movement is conditioned too.
    const framed = [MOVEMENT_START_TOKEN, ...context];

    // Backoff over n-gram orders: longest matched context first.
    const maxContext = Math.min(this.order - 1, framed.length);
    for (let length = maxContext; length >= 1; length--) {
      const key = contextKey(framed.slice(framed.length - length));
      const row = this.transitions.get(key);
      const top = row ? argmax(row) : undefined;
      if (top) {
        return { token: top.token, confidence: top.count / top.total, order: length + 1, source: "ngram" };
      }
    }

    // Generalization: the exact context is unseen, but the most recent movement's
    // family (tool + gesture verb) may have a learned continuation — this lets the
    // model handle a new-but-related movement.
    const last = context[context.length - 1];
    if (last !== undefined) {
      const familyRow = this.families.get(movementFamily(last));
      const familyTop = familyRow ? argmax(familyRow) : undefined;
      if (familyTop) {
        return { token: familyTop.token, confidence: familyTop.count / familyTop.total, order: 1, source: "family" };
      }
    }

    // Final fallback: the global unigram prior (most common movement overall).
    const priorRow = this.transitions.get("");
    const priorTop = priorRow ? argmax(priorRow) : undefined;
    if (priorTop) {
      return { token: priorTop.token, confidence: priorTop.count / priorTop.total, order: 0, source: "prior" };
    }
    return undefined;
  }

  generate(seed: MovementToken[], options?: MovementGenerateOptions): MovementToken[] {
    const maxSteps = options?.maxSteps ?? 64;
    const generated: MovementToken[] = [];
    const context = [...seed];
    for (let step = 0; step < maxSteps; step++) {
      const prediction = this.predict(context, { minConfidence: options?.minConfidence ?? 0 });
      if (!prediction || prediction.token === MOVEMENT_END_TOKEN) {
        break;
      }
      generated.push(prediction.token);
      context.push(prediction.token);
    }
    return generated;
  }

  serialize(): MovementPolicySnapshot {
    return {
      version: 1,
      backend: this.backend,
      order: this.order,
      transitions: serializeTable(this.transitions),
      families: serializeTable(this.families),
    };
  }
}

function serializeTable(table: CountTable): Array<[string, Array<[MovementToken, number]>]> {
  return [...table.entries()].map(([key, row]) => [key, [...row.entries()]]);
}

function deserializeTable(entries: Array<[string, Array<[MovementToken, number]>]>): CountTable {
  const table: CountTable = new Map();
  for (const [key, row] of entries) {
    table.set(key, new Map(row));
  }
  return table;
}

/**
 * The default in-process backend: a variable-order Markov (n-gram) policy with
 * family backoff for generalization. Fully deterministic — identical datasets
 * yield identical policies and predictions — so it is a reliable mock for the
 * pluggable training seam and passes in the cloud with no OS/GPU access.
 */
export class NgramMovementModelBackend implements MovementModelBackend {
  readonly name = "ngram";

  train(dataset: MovementDataset, options?: MovementTrainOptions): MovementPolicy {
    const order = Math.max(2, Math.floor(options?.order ?? DEFAULT_ORDER));
    const transitions: CountTable = new Map();
    const families: CountTable = new Map();

    for (const sequence of dataset.sequences) {
      // Frame each recorded sequence: START <movements...> END.
      const framed = [MOVEMENT_START_TOKEN, ...sequence.tokens, MOVEMENT_END_TOKEN];

      // Targets are every framed token except the leading START (never predicted).
      for (let index = 1; index < framed.length; index++) {
        const token = framed[index]!;

        // Unigram prior (empty context key) over real targets + END.
        bump(transitions, "", token);

        // Every context length from 1..order-1 ending just before `index`.
        for (let length = 1; length < order; length++) {
          if (index - length < 0) {
            break;
          }
          const context = framed.slice(index - length, index);
          bump(transitions, contextKey(context), token);
        }

        // Family generalization: previous movement's signature -> this token.
        const previous = framed[index - 1]!;
        if (previous !== MOVEMENT_START_TOKEN) {
          bump(families, movementFamily(previous), token);
        }
      }
    }

    return new NgramMovementPolicy(order, transitions, families);
  }

  restore(snapshot: MovementPolicySnapshot): MovementPolicy {
    return new NgramMovementPolicy(
      snapshot.order,
      deserializeTable(snapshot.transitions),
      deserializeTable(snapshot.families),
    );
  }
}

// --- Pluggable registry ------------------------------------------------------

/**
 * Registry of movement-model backends, keyed by name. Lets bee-agent select a
 * backend at runtime (e.g. `ngram` in the cloud, a real on-device small model
 * locally) without call sites depending on a concrete class.
 */
export class MovementModelRegistry {
  private readonly backends = new Map<string, MovementModelBackend>();

  register(backend: MovementModelBackend): this {
    this.backends.set(backend.name, backend);
    return this;
  }

  has(name: string): boolean {
    return this.backends.has(name);
  }

  list(): string[] {
    return [...this.backends.keys()].sort();
  }

  get(name: string): MovementModelBackend {
    const backend = this.backends.get(name);
    if (!backend) {
      throw new Error(`unknown movement-model backend: ${name} (registered: ${this.list().join(", ") || "none"})`);
    }
    return backend;
  }

  train(name: string, dataset: MovementDataset, options?: MovementTrainOptions): MovementPolicy {
    return this.get(name).train(dataset, options);
  }

  restore(snapshot: MovementPolicySnapshot): MovementPolicy {
    return this.get(snapshot.backend).restore(snapshot);
  }
}

/** A registry pre-loaded with the deterministic in-process backend. */
export function defaultMovementModelRegistry(): MovementModelRegistry {
  return new MovementModelRegistry().register(new NgramMovementModelBackend());
}

/** Convenience: train a policy with the default `ngram` backend. */
export function trainMovementModel(dataset: MovementDataset, options?: MovementTrainOptions): MovementPolicy {
  return new NgramMovementModelBackend().train(dataset, options);
}

// --- Generalization eval harness --------------------------------------------

export type MovementEvalResult = {
  /** Held-out next-movement predictions attempted. */
  predictions: number;
  /** Predictions whose top-1 token matched the recorded next movement. */
  correct: number;
  /** correct / predictions (0 when no predictions were made). */
  topOneAccuracy: number;
  /** Mean confidence the policy assigned to its predictions. */
  meanConfidence: number;
};

/**
 * Measure a policy's next-movement top-1 accuracy on held-out sequences — the
 * generalization signal for the movement subsystem. For each held-out sequence,
 * every prefix is fed to the policy and its prediction compared to the true next
 * token (the end sentinel is included so "stop" is scored too).
 */
export function evaluateMovementPolicy(policy: MovementPolicy, heldOut: MovementSequence[]): MovementEvalResult {
  let predictions = 0;
  let correct = 0;
  let confidenceSum = 0;

  for (const sequence of heldOut) {
    const targets = [...sequence.tokens, MOVEMENT_END_TOKEN];
    for (let index = 0; index < targets.length; index++) {
      const context = sequence.tokens.slice(0, index);
      const prediction = policy.predict(context);
      predictions += 1;
      if (prediction) {
        confidenceSum += prediction.confidence;
        if (prediction.token === targets[index]) {
          correct += 1;
        }
      }
    }
  }

  return {
    predictions,
    correct,
    topOneAccuracy: predictions === 0 ? 0 : correct / predictions,
    meanConfidence: predictions === 0 ? 0 : confidenceSum / predictions,
  };
}
