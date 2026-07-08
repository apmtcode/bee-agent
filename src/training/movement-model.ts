import type { ReplayTimelineEvent } from "../capture/replay.js";
import type { TrajectorySpan } from "../capture/trajectory.js";

/**
 * In-process, cloud-safe movement-policy model for the local-movement learning
 * subsystem (standing objective #2c/#2d).
 *
 * The recording pipeline (`src/capture`) produces ordered movement streams; the
 * reviewed exporter (`src/training/exporter.ts`) packages them into replay
 * manifests. This module closes the loop: it learns a *policy* from those
 * streams that can (c) repeat recorded movements and (d) generalise to new but
 * related movements — without touching the real OS, so it runs and is validated
 * in the cloud on synthetic data.
 *
 * The model backend is pluggable (`MovementModelBackend`). The default
 * `MarkovMovementModelBackend` is a deterministic order-k Markov policy with
 * stupid-backoff smoothing: greedy decoding reproduces recorded movements
 * exactly, while backoff lets an unseen high-order context fall back to a seen
 * shorter suffix, which is what produces generalisation. A real on-device small
 * model can be dropped in behind the same interface.
 */

export type MovementTokenKind = "observation" | "action";

/** A single discrete movement step. Tokenised from capture events. */
export type MovementToken = {
  kind: MovementTokenKind;
  /** The channel the step happened on, e.g. an app id, `device`, `os`, `browser`. */
  channel: string;
  /** A normalised verb/intent, e.g. `tapped`, `typed`, `focused`, `opened`. */
  verb: string;
};

export type MovementSequence = {
  id: string;
  tokens: MovementToken[];
};

export type MovementDataset = {
  sequences: MovementSequence[];
};

export type MovementPrediction = {
  token: MovementToken;
  /** Conditional probability of `token` under the context order that fired. */
  probability: number;
  /** How many trailing tokens of context were actually matched (0 = unigram). */
  backoffOrder: number;
};

export type MovementTrainingOptions = {
  /** Maximum Markov order (context length). Defaults to 2. */
  order?: number;
};

export type MovementModelSnapshot = {
  backend: string;
  order: number;
  /** `contexts[k]` maps an order-k context key to a token-key → count map. */
  contexts: Record<string, Record<string, number>>[];
  vocabulary: MovementToken[];
};

export interface TrainedMovementModel {
  readonly backend: string;
  readonly order: number;
  /**
   * Predict the next movement given the recent tokens (most recent last).
   * Returns `undefined` only for an empty, never-trained model.
   */
  predictNext(context: MovementToken[]): MovementPrediction | undefined;
  /** Greedily generate up to `steps` movements from a seed context. */
  rollout(seed: MovementToken[], steps: number): MovementToken[];
  snapshot(): MovementModelSnapshot;
}

export interface MovementModelBackend {
  readonly name: string;
  train(dataset: MovementDataset, options?: MovementTrainingOptions): Promise<TrainedMovementModel>;
}

// Sentinels use a control-char prefix that real token keys can never produce.
const START_KEY = "START";
const END_KEY = "END";

export function encodeMovementToken(token: MovementToken): string {
  return `${token.kind} ${token.channel} ${token.verb}`;
}

export function decodeMovementToken(key: string): MovementToken {
  const [kind, channel, verb] = key.split(" ");
  return {
    kind: kind === "observation" ? "observation" : "action",
    channel: channel ?? "",
    verb: verb ?? "",
  };
}

function normalizeVerb(summary: string, fallback: string): string {
  const first = summary.trim().toLowerCase().split(/\s+/u)[0];
  return first && /[a-z0-9]/u.test(first) ? first.replace(/[^a-z0-9-]/gu, "") : fallback;
}

/** Tokenise a replay manifest's ordered event timeline into movement tokens. */
export function tokenizeReplayEvents(events: ReplayTimelineEvent[]): MovementToken[] {
  const tokens: MovementToken[] = [];
  for (const event of events) {
    if (event.kind === "observation") {
      tokens.push({ kind: "observation", channel: event.source, verb: normalizeVerb(event.summary, "observed") });
    } else if (event.kind === "action") {
      tokens.push({ kind: "action", channel: event.tool, verb: normalizeVerb(event.summary, "acted") });
    }
    // transcript events carry no movement; skipped.
  }
  return tokens;
}

/** Tokenise a raw trajectory span (observations + actions) in timestamp order. */
export function tokenizeTrajectory(trajectory: TrajectorySpan): MovementToken[] {
  const entries: { ts: number; token: MovementToken }[] = [
    ...trajectory.observations.map((observation) => ({
      ts: observation.ts,
      token: { kind: "observation" as const, channel: observation.source, verb: normalizeVerb(observation.summary, "observed") },
    })),
    ...trajectory.actions.map((action) => ({
      ts: action.ts,
      token: { kind: "action" as const, channel: action.tool, verb: normalizeVerb(action.summary, "acted") },
    })),
  ];
  entries.sort((a, b) => (a.ts !== b.ts ? a.ts - b.ts : a.token.kind === "observation" ? -1 : 1));
  return entries.map((entry) => entry.token);
}

export function buildMovementDatasetFromReplays(
  replays: { trajectoryIds: string[]; events: ReplayTimelineEvent[] }[],
): MovementDataset {
  return {
    sequences: replays.map((replay, index) => ({
      id: replay.trajectoryIds.join("+") || `replay-${index}`,
      tokens: tokenizeReplayEvents(replay.events),
    })),
  };
}

export function buildMovementDatasetFromTrajectories(trajectories: TrajectorySpan[]): MovementDataset {
  return {
    sequences: trajectories.map((trajectory) => ({ id: trajectory.id, tokens: tokenizeTrajectory(trajectory) })),
  };
}

/**
 * Deterministic order-k Markov policy with stupid-backoff smoothing. No RNG,
 * no OS access — safe to train and evaluate in the cloud on synthetic streams.
 */
export class MarkovMovementModelBackend implements MovementModelBackend {
  readonly name = "markov-backoff";

  async train(dataset: MovementDataset, options?: MovementTrainingOptions): Promise<TrainedMovementModel> {
    const order = Math.max(0, Math.floor(options?.order ?? 2));
    // contexts[k]: context-key (k trailing token keys joined by ) -> {tokenKey: count}
    const contexts: Map<string, Map<string, number>>[] = Array.from({ length: order + 1 }, () => new Map());
    const vocabulary = new Map<string, MovementToken>();

    for (const sequence of dataset.sequences) {
      const keys = [START_KEY, ...sequence.tokens.map(encodeMovementToken), END_KEY];
      for (const token of sequence.tokens) {
        vocabulary.set(encodeMovementToken(token), token);
      }
      for (let i = 1; i < keys.length; i += 1) {
        const nextKey = keys[i]!;
        for (let k = 0; k <= order; k += 1) {
          if (i - k < 0) {
            break;
          }
          const contextKey = keys.slice(i - k, i).join("");
          const table = contexts[k]!;
          const counts = table.get(contextKey) ?? new Map<string, number>();
          counts.set(nextKey, (counts.get(nextKey) ?? 0) + 1);
          table.set(contextKey, counts);
        }
      }
    }

    return new MarkovMovementModel(this.name, order, contexts, vocabulary);
  }
}

class MarkovMovementModel implements TrainedMovementModel {
  constructor(
    readonly backend: string,
    readonly order: number,
    private readonly contexts: Map<string, Map<string, number>>[],
    private readonly vocabulary: Map<string, MovementToken>,
  ) {}

  private predictKey(paddedKeys: string[]): { key: string; probability: number; backoffOrder: number } | undefined {
    for (let k = this.order; k >= 0; k -= 1) {
      const table = this.contexts[k];
      if (!table) {
        continue;
      }
      const contextKey = paddedKeys.slice(paddedKeys.length - k).join("");
      const counts = table.get(contextKey);
      if (!counts || counts.size === 0) {
        continue;
      }
      let total = 0;
      let bestKey: string | undefined;
      let bestCount = -1;
      // Deterministic argmax: higher count wins; ties break lexicographically.
      for (const [key, count] of counts) {
        total += count;
        if (count > bestCount || (count === bestCount && (bestKey === undefined || key < bestKey))) {
          bestCount = count;
          bestKey = key;
        }
      }
      if (bestKey !== undefined) {
        return { key: bestKey, probability: bestCount / total, backoffOrder: k };
      }
    }
    return undefined;
  }

  predictNext(context: MovementToken[]): MovementPrediction | undefined {
    const prediction = this.predictNextKey(context);
    if (!prediction || prediction.key === END_KEY) {
      return undefined;
    }
    return {
      token: this.vocabulary.get(prediction.key) ?? decodeMovementToken(prediction.key),
      probability: prediction.probability,
      backoffOrder: prediction.backoffOrder,
    };
  }

  private predictNextKey(context: MovementToken[]): { key: string; probability: number; backoffOrder: number } | undefined {
    const keys = context.map(encodeMovementToken);
    const padded = [...Array.from({ length: this.order }, () => START_KEY), ...keys];
    return this.predictKey(padded);
  }

  rollout(seed: MovementToken[], steps: number): MovementToken[] {
    const generated: MovementToken[] = [];
    const context = [...seed];
    for (let i = 0; i < steps; i += 1) {
      const prediction = this.predictNextKey(context);
      if (!prediction || prediction.key === END_KEY) {
        break;
      }
      const token = this.vocabulary.get(prediction.key) ?? decodeMovementToken(prediction.key);
      generated.push(token);
      context.push(token);
    }
    return generated;
  }

  snapshot(): MovementModelSnapshot {
    return {
      backend: this.backend,
      order: this.order,
      contexts: this.contexts.map((table) => {
        const record: Record<string, Record<string, number>> = {};
        for (const [contextKey, counts] of table) {
          record[contextKey] = Object.fromEntries(counts);
        }
        return record;
      }),
      vocabulary: Array.from(this.vocabulary.values()),
    };
  }
}

const DEFAULT_BACKEND = new MarkovMovementModelBackend();

/** Convenience: train the default deterministic backend on a dataset. */
export async function trainMovementModel(
  dataset: MovementDataset,
  options?: MovementTrainingOptions & { backend?: MovementModelBackend },
): Promise<TrainedMovementModel> {
  const backend = options?.backend ?? DEFAULT_BACKEND;
  return await backend.train(dataset, options);
}
