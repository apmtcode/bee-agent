import type { ReplayTimelineEvent } from "../capture/replay.js";
import type { DeviceCaptureInput } from "../capture/device-adapter.js";

/**
 * In-process movement-learning backend for objective #2 (post-train a local
 * model on recorded movements to repeat them, and generalize to new but related
 * movements).
 *
 * The real on-device training path shells out to mlx/axolotl (see runner.ts).
 * This module provides the pluggable *backend interface* plus a deterministic
 * reference backend so the whole capture -> dataset -> train -> infer -> replay
 * loop can be exercised and validated entirely in the cloud with synthetic data,
 * with zero access to a real machine. Swap `NGramMovementBackend` for a real
 * small-model backend by implementing {@link MovementModelBackend}.
 */

/** A single discretized movement, the atomic unit the model learns over. */
export type MovementEvent = {
  /** Contextual scope of the movement — typically the app/window id. */
  context: string;
  /** Which capture channel produced the movement. */
  channel: "device" | "os" | "tool" | "transcript";
  /** The movement itself, e.g. "tap", "scroll:down", "type", or a tool name. */
  action: string;
  /** Optional element/target the movement acted on. */
  target?: string;
};

/** An ordered run of movements captured within one trajectory/session. */
export type MovementSequence = {
  id: string;
  events: MovementEvent[];
};

/** The training corpus: a set of movement sequences. */
export type MovementDataset = {
  sequences: MovementSequence[];
};

export type MovementTrainingConfig = {
  /** Maximum n-gram context length the backend conditions on. Default 3. */
  order?: number;
};

/** A prediction for the next movement given a context window. */
export type MovementPrediction = {
  /** The predicted next movement, or undefined when the model is empty. */
  event: MovementEvent | undefined;
  /** Estimated probability of `event` at the backoff level that fired (0..1). */
  confidence: number;
  /**
   * Which backoff level produced the prediction:
   * - `ngram-<k>`: an exact k-token context match (highest fidelity),
   * - `feature`: generalization from context+channel of the last movement,
   * - `prior`: global majority movement,
   * - `empty`: the model has no data.
   */
  level: string;
};

/**
 * A serializable, backend-agnostic trained model. Kept as plain JSON so it can
 * be persisted next to the reviewed-export artifacts and reloaded without the
 * originating backend instance.
 */
export type MovementModelArtifact = {
  backend: string;
  version: 1;
  order: number;
  /** order -> contextKey -> nextTokenKey -> count */
  orders: Record<string, Record<string, Record<string, number>>>;
  /** featureKey -> nextTokenKey -> count (generalization layer) */
  featureCounts: Record<string, Record<string, number>>;
  /** nextTokenKey -> count (global prior / majority baseline) */
  priorCounts: Record<string, number>;
  /** tokenKey -> the movement it decodes to (for rollout). */
  tokenMap: Record<string, MovementEvent>;
};

/** Pluggable movement-model backend. Implement this to add a real local model. */
export interface MovementModelBackend {
  readonly id: string;
  train(dataset: MovementDataset, config?: MovementTrainingConfig): Promise<MovementModelArtifact>;
  predict(model: MovementModelArtifact, context: MovementEvent[]): MovementPrediction;
}

const DEFAULT_ORDER = 3;

/** Canonical string key for a movement (used as the model's token). */
export function movementTokenKey(event: MovementEvent): string {
  return `${event.channel}${event.action}${event.target ?? ""}${event.context}`;
}

/** Feature signature used for generalization backoff: app + channel only. */
function movementFeatureKey(event: MovementEvent): string {
  return `${event.context}${event.channel}`;
}

function contextKey(events: MovementEvent[]): string {
  return events.map(movementTokenKey).join("");
}

function increment(table: Record<string, number>, key: string): void {
  table[key] = (table[key] ?? 0) + 1;
}

function topEntry(table: Record<string, number> | undefined): { key: string; count: number; total: number } | undefined {
  if (!table) {
    return undefined;
  }
  let bestKey: string | undefined;
  let bestCount = -1;
  let total = 0;
  // Deterministic: ties broken by lexical key order.
  for (const key of Object.keys(table).sort()) {
    const count = table[key] ?? 0;
    total += count;
    if (count > bestCount) {
      bestCount = count;
      bestKey = key;
    }
  }
  if (bestKey === undefined || total === 0) {
    return undefined;
  }
  return { key: bestKey, count: bestCount, total };
}

/**
 * Deterministic reference backend: a variable-order n-gram model with stupid
 * backoff and a feature-conditioned generalization layer. It reproduces
 * recorded movement sequences exactly (exact-context match) and still emits a
 * calibrated prediction for unseen-but-related contexts by backing off to the
 * app+channel feature distribution, then the global prior.
 */
export class NGramMovementBackend implements MovementModelBackend {
  readonly id = "ngram-backoff";

  async train(dataset: MovementDataset, config?: MovementTrainingConfig): Promise<MovementModelArtifact> {
    const order = Math.max(1, config?.order ?? DEFAULT_ORDER);
    const orders: MovementModelArtifact["orders"] = {};
    const featureCounts: MovementModelArtifact["featureCounts"] = {};
    const priorCounts: MovementModelArtifact["priorCounts"] = {};
    const tokenMap: MovementModelArtifact["tokenMap"] = {};

    for (let k = 1; k <= order; k += 1) {
      orders[String(k)] = {};
    }

    for (const sequence of dataset.sequences) {
      const events = sequence.events;
      for (let i = 0; i < events.length; i += 1) {
        const nextToken = movementTokenKey(events[i]);
        tokenMap[nextToken] = events[i];
        increment(priorCounts, nextToken);

        if (i > 0) {
          const feature = movementFeatureKey(events[i - 1]);
          featureCounts[feature] ??= {};
          increment(featureCounts[feature], nextToken);
        }

        for (let k = 1; k <= order; k += 1) {
          if (i - k < 0) {
            break;
          }
          const ctx = contextKey(events.slice(i - k, i));
          const table = (orders[String(k)][ctx] ??= {});
          increment(table, nextToken);
        }
      }
    }

    return { backend: this.id, version: 1, order, orders, featureCounts, priorCounts, tokenMap };
  }

  predict(model: MovementModelArtifact, context: MovementEvent[]): MovementPrediction {
    // Highest fidelity first: longest exact context match, then shorter.
    for (let k = Math.min(model.order, context.length); k >= 1; k -= 1) {
      const ctx = contextKey(context.slice(context.length - k));
      const hit = topEntry(model.orders[String(k)]?.[ctx]);
      if (hit) {
        return decode(model, hit, `ngram-${k}`);
      }
    }

    // Generalization: condition only on the last movement's app+channel.
    const last = context[context.length - 1];
    if (last) {
      const hit = topEntry(model.featureCounts[movementFeatureKey(last)]);
      if (hit) {
        return decode(model, hit, "feature");
      }
    }

    // Floor: global majority movement.
    const prior = topEntry(model.priorCounts);
    if (prior) {
      return decode(model, prior, "prior");
    }

    return { event: undefined, confidence: 0, level: "empty" };
  }
}

function decode(
  model: MovementModelArtifact,
  hit: { key: string; count: number; total: number },
  level: string,
): MovementPrediction {
  return {
    event: model.tokenMap[hit.key],
    confidence: hit.total > 0 ? hit.count / hit.total : 0,
    level,
  };
}

/**
 * Greedily roll out a predicted movement sequence from a seed. This is the
 * "repeat / generalize the recorded movement" inference path: given a starting
 * context, the model proposes the next movements one at a time.
 */
export function rolloutMovements(
  backend: MovementModelBackend,
  model: MovementModelArtifact,
  seed: MovementEvent[],
  steps: number,
): MovementEvent[] {
  const history = [...seed];
  const produced: MovementEvent[] = [];
  for (let i = 0; i < steps; i += 1) {
    const prediction = backend.predict(model, history);
    if (!prediction.event) {
      break;
    }
    produced.push(prediction.event);
    history.push(prediction.event);
  }
  return produced;
}

/** Adapt a reviewed replay manifest's timeline into learnable movements. */
export function movementsFromReplayEvents(events: ReplayTimelineEvent[]): MovementEvent[] {
  return events.map((event): MovementEvent => {
    switch (event.kind) {
      case "action":
        return { context: event.trajectoryId, channel: "tool", action: event.tool, target: event.summary };
      case "observation":
        return { context: event.trajectoryId, channel: "os", action: event.source, target: event.summary };
      case "transcript":
        return { context: event.messageId, channel: "transcript", action: event.role, target: undefined };
    }
  });
}

/** Adapt a low-level device gesture capture into a learnable movement. */
export function movementFromDeviceInput(input: DeviceCaptureInput): MovementEvent | undefined {
  if (!input.gesture) {
    return undefined;
  }
  const direction = input.gesture.direction ? `:${input.gesture.direction}` : "";
  return {
    context: input.appId,
    channel: "device",
    action: `${input.gesture.kind}${direction}`,
    target: input.gesture.target,
  };
}
