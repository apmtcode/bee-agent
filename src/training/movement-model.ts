import type { ReplayTimelineEvent } from "../capture/replay.js";
import type { TrajectorySpan } from "../capture/trajectory.js";

/**
 * Local-movement learning: pluggable model backend.
 *
 * Standing objective #2(c/d): post-train a *local* model on the recorded
 * movement dataset so it can (i) repeat recorded movements exactly and
 * (ii) generalize to related-but-unseen movements.
 *
 * The real on-device training (MLX / axolotl) runs only on the user's machine
 * and cannot execute in the cloud. This module therefore defines the *backend
 * seam* — a `MovementModelBackend` interface — plus a fully in-process,
 * deterministic reference backend (`MarkovMovementBackend`) that trains and
 * infers with zero native deps, so the entire capture -> dataset -> train ->
 * infer loop is exercisable in CI against synthetic event streams. A real
 * backend (a small local model) implements the same interface and swaps in.
 */

/** A single movement action, tokenized into a stable, comparable string. */
export type MovementToken = string;

/** One recorded movement: an ordered sequence of action tokens. */
export type MovementSequence = MovementToken[];

export type MovementDataset = {
  sequences: MovementSequence[];
};

export type MovementTrainOptions = {
  /** Maximum Markov context length. Higher = more literal replay, less generalization. */
  order?: number;
};

export type MovementPredictionAlternative = {
  token: MovementToken;
  probability: number;
};

export type MovementPrediction = {
  token: MovementToken;
  /** Probability mass of the chosen token within the matched context. */
  confidence: number;
  /**
   * The context length that actually matched. When it is lower than the
   * requested order the model *generalized* via backoff — the exact prefix was
   * never recorded, so a shorter, more general context supplied the answer.
   */
  backoffOrder: number;
  /** Whether termination (end of movement) is the most likely next token. */
  isTerminal: boolean;
  alternatives: MovementPredictionAlternative[];
};

/** Serializable model parameters — the on-device persistence/transfer seam. */
export type MovementModelSnapshot = {
  version: 1;
  backendId: string;
  order: number;
  sequenceCount: number;
  vocabulary: MovementToken[];
  /** transitions[k] maps a k-token context key -> { nextToken: count }. */
  transitions: Array<Record<string, Record<MovementToken, number>>>;
};

export interface MovementModel {
  readonly backendId: string;
  readonly order: number;
  /** Predict the next action given the movement so far. Undefined only if untrained. */
  predictNext(context: MovementToken[]): MovementPrediction | undefined;
  /** Reconstruct a full movement from a seed prefix (default: from the start). */
  generate(seed?: MovementToken[], options?: { maxSteps?: number }): MovementToken[];
  serialize(): MovementModelSnapshot;
}

export interface MovementModelBackend {
  readonly id: string;
  train(dataset: MovementDataset, options?: MovementTrainOptions): MovementModel;
  load(snapshot: MovementModelSnapshot): MovementModel;
}

/** Field separator inside a token (tool ␟ summary) — a control char absent from tool names/summaries. */
const TOKEN_FIELD_SEP = "\u241f";
/** Join separator between tokens when forming a context key (collision-safe control char). */
const CONTEXT_SEP = "\u0001";
/** Sentinel marking the end of a movement, so `generate` knows when to stop. */
export const MOVEMENT_END_TOKEN: MovementToken = "\u0004END";

export const DEFAULT_MOVEMENT_ORDER = 2;

/** Encode a captured action into a stable token. */
export function tokenizeAction(tool: string, summary: string): MovementToken {
  return `${tool}${TOKEN_FIELD_SEP}${summary}`;
}

/** Decode a token back into its action fields (the terminal sentinel returns undefined). */
export function parseMovementToken(token: MovementToken): { tool: string; summary: string } | undefined {
  if (token === MOVEMENT_END_TOKEN) {
    return undefined;
  }
  const index = token.indexOf(TOKEN_FIELD_SEP);
  if (index < 0) {
    return { tool: token, summary: "" };
  }
  return { tool: token.slice(0, index), summary: token.slice(index + TOKEN_FIELD_SEP.length) };
}

/** Build a training dataset from replay timelines (one sequence per replay). */
export function movementDatasetFromReplays(
  replays: Array<{ events: ReplayTimelineEvent[] }>,
): MovementDataset {
  const sequences = replays
    .map((replay) =>
      replay.events
        .filter((event): event is Extract<ReplayTimelineEvent, { kind: "action" }> => event.kind === "action")
        .slice()
        .sort((a, b) => a.ts - b.ts)
        .map((event) => tokenizeAction(event.tool, event.summary)),
    )
    .filter((sequence) => sequence.length > 0);
  return { sequences };
}

/** Build a training dataset from trajectory spans (one sequence per trajectory). */
export function movementDatasetFromTrajectories(trajectories: TrajectorySpan[]): MovementDataset {
  const sequences = trajectories
    .map((trajectory) =>
      trajectory.actions
        .slice()
        .sort((a, b) => a.ts - b.ts)
        .map((action) => tokenizeAction(action.tool, action.summary)),
    )
    .filter((sequence) => sequence.length > 0);
  return { sequences };
}

class MarkovMovementModel implements MovementModel {
  constructor(private readonly snapshot: MovementModelSnapshot) {}

  get backendId(): string {
    return this.snapshot.backendId;
  }

  get order(): number {
    return this.snapshot.order;
  }

  predictNext(context: MovementToken[]): MovementPrediction | undefined {
    const maxK = Math.min(this.snapshot.order, context.length);
    for (let k = maxK; k >= 0; k -= 1) {
      const table = this.snapshot.transitions[k];
      if (!table) {
        continue;
      }
      const key = k === 0 ? "" : context.slice(context.length - k).join(CONTEXT_SEP);
      const distribution = table[key];
      if (!distribution) {
        continue;
      }
      // Deterministic argmax: highest count, ties broken by lexicographic token.
      const entries = Object.entries(distribution).sort((a, b) => (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0));
      if (entries.length === 0) {
        continue;
      }
      let total = 0;
      let bestToken = entries[0][0];
      let bestCount = -1;
      for (const [token, count] of entries) {
        total += count;
        if (count > bestCount) {
          bestCount = count;
          bestToken = token;
        }
      }
      const alternatives = entries
        .map(([token, count]) => ({ token, probability: count / total }))
        .sort((a, b) => b.probability - a.probability || (a.token < b.token ? -1 : 1));
      return {
        token: bestToken,
        confidence: bestCount / total,
        backoffOrder: k,
        isTerminal: bestToken === MOVEMENT_END_TOKEN,
        alternatives,
      };
    }
    return undefined;
  }

  generate(seed: MovementToken[] = [], options: { maxSteps?: number } = {}): MovementToken[] {
    const maxSteps = options.maxSteps ?? 1000;
    const output = [...seed];
    for (let step = 0; step < maxSteps; step += 1) {
      const prediction = this.predictNext(output);
      if (!prediction || prediction.token === MOVEMENT_END_TOKEN) {
        break;
      }
      output.push(prediction.token);
    }
    return output;
  }

  serialize(): MovementModelSnapshot {
    return {
      version: 1,
      backendId: this.snapshot.backendId,
      order: this.snapshot.order,
      sequenceCount: this.snapshot.sequenceCount,
      vocabulary: [...this.snapshot.vocabulary],
      transitions: this.snapshot.transitions.map((table) => {
        const copy: Record<string, Record<MovementToken, number>> = {};
        for (const [key, distribution] of Object.entries(table)) {
          copy[key] = { ...distribution };
        }
        return copy;
      }),
    };
  }
}

/**
 * Deterministic variable-order Markov backend with stupid-backoff generalization.
 *
 * Reference implementation of `MovementModelBackend`: it "post-trains" on the
 * dataset by counting context -> next-token transitions at every order up to
 * `order`, and predicts by preferring the longest matching context (literal
 * replay) and backing off to shorter contexts when the exact prefix was never
 * seen (generalization to related movements). Fully in-process, no native
 * deps, and byte-for-byte reproducible — ideal for cloud/CI validation.
 */
export class MarkovMovementBackend implements MovementModelBackend {
  readonly id = "markov-backoff";

  train(dataset: MovementDataset, options: MovementTrainOptions = {}): MovementModel {
    const order = Math.max(0, options.order ?? DEFAULT_MOVEMENT_ORDER);
    const transitions: Array<Record<string, Record<MovementToken, number>>> = Array.from(
      { length: order + 1 },
      () => ({}),
    );
    const vocabulary = new Set<MovementToken>();

    for (const rawSequence of dataset.sequences) {
      // Append the terminal sentinel so termination is a learnable event.
      const sequence = [...rawSequence, MOVEMENT_END_TOKEN];
      for (let i = 0; i < sequence.length; i += 1) {
        const target = sequence[i];
        if (target !== MOVEMENT_END_TOKEN) {
          vocabulary.add(target);
        }
        const maxK = Math.min(order, i);
        for (let k = 0; k <= maxK; k += 1) {
          const key = k === 0 ? "" : sequence.slice(i - k, i).join(CONTEXT_SEP);
          const table = transitions[k];
          const distribution = (table[key] ??= {});
          distribution[target] = (distribution[target] ?? 0) + 1;
        }
      }
    }

    return new MarkovMovementModel({
      version: 1,
      backendId: this.id,
      order,
      sequenceCount: dataset.sequences.length,
      vocabulary: [...vocabulary].sort(),
      transitions,
    });
  }

  load(snapshot: MovementModelSnapshot): MovementModel {
    return new MarkovMovementModel(snapshot);
  }
}

/** Reconstruct a model from a persisted snapshot, dispatching on backend id. */
export function loadMovementModel(
  snapshot: MovementModelSnapshot,
  backends: MovementModelBackend[] = [new MarkovMovementBackend()],
): MovementModel {
  const backend = backends.find((candidate) => candidate.id === snapshot.backendId);
  if (!backend) {
    throw new Error(`no movement backend registered for id "${snapshot.backendId}"`);
  }
  return backend.load(snapshot);
}

/** Convenience: train with the default (Markov) backend. */
export function trainMovementModel(
  dataset: MovementDataset,
  options?: MovementTrainOptions,
): MovementModel {
  return new MarkovMovementBackend().train(dataset, options);
}

export type MovementReplayFidelity = {
  total: number;
  reproduced: number;
  /** Fraction of movements the model reconstructs exactly from their first action. */
  accuracy: number;
  failures: Array<{ expected: MovementSequence; actual: MovementSequence }>;
};

/**
 * Generalization / replay eval: for each movement, seed the model with its
 * first action and check whether `generate` reconstructs the full sequence.
 * Use held-out sequences (not in the training set) to measure generalization.
 */
export function evaluateMovementReplay(
  model: MovementModel,
  sequences: MovementSequence[],
): MovementReplayFidelity {
  const failures: MovementReplayFidelity["failures"] = [];
  let reproduced = 0;
  for (const expected of sequences) {
    if (expected.length === 0) {
      continue;
    }
    const actual = model.generate([expected[0]], { maxSteps: expected.length + 8 });
    if (actual.length === expected.length && actual.every((token, index) => token === expected[index])) {
      reproduced += 1;
    } else {
      failures.push({ expected, actual });
    }
  }
  const total = sequences.filter((sequence) => sequence.length > 0).length;
  return {
    total,
    reproduced,
    accuracy: total === 0 ? 1 : reproduced / total,
    failures,
  };
}
