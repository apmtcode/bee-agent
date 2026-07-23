import {
  MOVEMENT_START_TOKEN,
  tokenizeEvent,
  tokenizeSequence,
  type MovementDataset,
  type MovementEvent,
  type MovementSequence,
  type TokenizeOptions,
} from "./event.js";

/**
 * Pluggable local-model backend for movement learning (standing objective #2,
 * parts c & d).
 *
 * The subsystem is deliberately backend-agnostic: `MovementModelBackend` is the
 * seam a real on-device small model (MLX / llama.cpp / a tiny transformer)
 * plugs into. The bundled {@link MarkovMovementBackend} is a deterministic,
 * dependency-free implementation that trains and infers entirely in-process, so
 * the whole capture -> train -> repeat/generalize loop can be validated in the
 * cloud without OS access or a heavyweight ML runtime.
 *
 * A backend must:
 *  - train() a serializable {@link MovementModelArtifact} from a dataset, and
 *  - predict() the next movement given a context, plus generate() a rollout so
 *    a recorded workflow can be repeated and related ones extrapolated.
 */
export interface MovementModelBackend {
  readonly id: string;
  train(dataset: MovementDataset, options?: MovementTrainOptions): Promise<MovementModelArtifact>;
  predict(model: MovementModelArtifact, context: MovementContext): Promise<MovementPrediction>;
  generate(model: MovementModelArtifact, context: MovementContext, options?: MovementGenerateOptions): Promise<MovementEvent[]>;
}

export type MovementTrainOptions = {
  /** Maximum n-gram context order. Defaults to 3. */
  order?: number;
  tokenize?: TokenizeOptions;
};

export type MovementContext = {
  history: MovementEvent[];
};

export type MovementGenerateOptions = {
  /** Maximum events to roll forward. Defaults to 32. */
  maxSteps?: number;
  /** Stop when this token is predicted (e.g. a learned end sentinel). */
  stopToken?: string;
};

export type MovementTokenProbability = {
  token: string;
  probability: number;
};

export type MovementPrediction = {
  /** Argmax next-event token, or undefined when the model has no signal. */
  token?: string;
  /** A concrete decoded event exemplar for `token`, ready to replay. */
  event?: MovementEvent;
  confidence: number;
  /** Full next-token distribution, descending by probability. */
  distribution: MovementTokenProbability[];
  /** n-gram order actually used after backoff (0 = unigram/prior). */
  backoffOrder: number;
};

/**
 * Serializable model artifact. Kept as plain JSON so it can be persisted with
 * the repo's atomic-write helpers and inspected/diffed like every other store.
 */
export type MovementModelArtifact = {
  version: 1;
  backend: string;
  order: number;
  tokenize: TokenizeOptions;
  /**
   * Transition counts keyed by context string ("" = unigram, otherwise the
   * joined last-k tokens) -> next token -> count.
   */
  transitions: Record<string, Record<string, number>>;
  /** One representative event per token, so predictions decode to real events. */
  exemplars: Record<string, MovementEvent>;
  stats: {
    sequenceCount: number;
    eventCount: number;
    vocabularySize: number;
  };
};

const CONTEXT_SEPARATOR = "␟";

export class MarkovMovementBackend implements MovementModelBackend {
  readonly id = "markov";

  async train(dataset: MovementDataset, options: MovementTrainOptions = {}): Promise<MovementModelArtifact> {
    const order = Math.max(1, options.order ?? 3);
    const tokenizeOptions = options.tokenize ?? {};
    const transitions: Record<string, Record<string, number>> = {};
    const exemplars: Record<string, MovementEvent> = {};
    const vocabulary = new Set<string>();
    let eventCount = 0;

    for (const sequence of dataset.sequences) {
      const tokens = tokenizeSequence(sequence, tokenizeOptions);
      tokens.forEach((token, index) => {
        vocabulary.add(token);
        // Keep the latest event seen for a token as its replay exemplar.
        const event = sequence.events[index];
        if (event) {
          exemplars[token] = event;
        }
      });
      eventCount += tokens.length;

      // Prepend START sentinels so the first real token is predictable.
      const padded = [...Array<string>(order).fill(MOVEMENT_START_TOKEN), ...tokens];
      for (let i = order; i < padded.length; i += 1) {
        const next = padded[i]!;
        for (let k = 0; k <= order; k += 1) {
          const context = padded.slice(i - k, i).join(CONTEXT_SEPARATOR);
          const bucket = (transitions[context] ??= {});
          bucket[next] = (bucket[next] ?? 0) + 1;
        }
      }
    }

    return {
      version: 1,
      backend: this.id,
      order,
      tokenize: tokenizeOptions,
      transitions,
      exemplars,
      stats: {
        sequenceCount: dataset.sequences.length,
        eventCount,
        vocabularySize: vocabulary.size,
      },
    };
  }

  async predict(model: MovementModelArtifact, context: MovementContext): Promise<MovementPrediction> {
    return this.predictSync(model, context.history);
  }

  async generate(
    model: MovementModelArtifact,
    context: MovementContext,
    options: MovementGenerateOptions = {},
  ): Promise<MovementEvent[]> {
    const maxSteps = options.maxSteps ?? 32;
    const generated: MovementEvent[] = [];
    const history = [...context.history];

    for (let step = 0; step < maxSteps; step += 1) {
      const prediction = this.predictSync(model, history);
      if (!prediction.token || prediction.confidence === 0) {
        break;
      }
      if (options.stopToken && prediction.token === options.stopToken) {
        break;
      }
      const event = prediction.event ?? decodeToken(prediction.token);
      generated.push(event);
      history.push(event);
    }

    return generated;
  }

  private predictSync(model: MovementModelArtifact, history: MovementEvent[]): MovementPrediction {
    const tokens = history.map((event) => tokenizeEvent(event, model.tokenize));
    const padded = [...Array<string>(model.order).fill(MOVEMENT_START_TOKEN), ...tokens];

    // Katz-style backoff: try the longest available context, shrink until a
    // matching context bucket exists. Order 0 (empty context) is the unigram
    // prior and always exists once anything was trained.
    for (let k = model.order; k >= 0; k -= 1) {
      const context = padded.slice(padded.length - k).join(CONTEXT_SEPARATOR);
      const bucket = model.transitions[context];
      if (!bucket) {
        continue;
      }
      const distribution = toDistribution(bucket);
      if (distribution.length === 0) {
        continue;
      }
      const top = distribution[0]!;
      return {
        token: top.token,
        event: model.exemplars[top.token] ?? decodeToken(top.token),
        confidence: top.probability,
        distribution,
        backoffOrder: k,
      };
    }

    return { confidence: 0, distribution: [], backoffOrder: 0 };
  }
}

function toDistribution(bucket: Record<string, number>): MovementTokenProbability[] {
  const total = Object.values(bucket).reduce((sum, count) => sum + count, 0);
  if (total === 0) {
    return [];
  }
  return Object.entries(bucket)
    .map(([token, count]) => ({ token, probability: count / total }))
    // Deterministic ordering: probability desc, then token asc as tie-break.
    .sort((a, b) => (b.probability - a.probability) || (a.token < b.token ? -1 : a.token > b.token ? 1 : 0));
}

/**
 * Reconstruct a best-effort concrete event from a token. Used only when no
 * exemplar was stored (e.g. a token synthesized outside training). Prefer the
 * stored exemplar, which carries real coordinates/targets.
 */
export function decodeToken(token: string): MovementEvent {
  const [head, ...rest] = token.split(":");
  const type = (head ?? "wait") as MovementEvent["type"];
  const event: MovementEvent = { t: 0, type };

  for (const part of rest) {
    if (part.startsWith("@")) {
      event.target = part.slice(1);
    } else if (part.startsWith("#")) {
      const [x, y] = part.slice(1).split(",").map((value) => Number(value));
      if (Number.isFinite(x)) event.x = x;
      if (Number.isFinite(y)) event.y = y;
    } else if (type === "click" || type === "drag") {
      event.button = part as MovementEvent["button"];
    } else if (type === "key" || type === "type") {
      event.key = part;
    } else if (type === "scroll" || type === "move") {
      applyDirection(event, part);
    }
  }

  return event;
}

function applyDirection(event: MovementEvent, direction: string): void {
  switch (direction) {
    case "up":
      event.dy = -1;
      break;
    case "down":
      event.dy = 1;
      break;
    case "left":
      event.dx = -1;
      break;
    case "right":
      event.dx = 1;
      break;
    default:
      break;
  }
}

export function isMovementSequence(value: unknown): value is MovementSequence {
  return (
    typeof value === "object" &&
    value !== null &&
    Array.isArray((value as MovementSequence).events)
  );
}
