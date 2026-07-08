import type { ReplayTimelineEvent } from "../capture/replay.js";

/**
 * Pluggable local-model backend for the movement-learning subsystem.
 *
 * Objective 2(c)/2(d): after movements are captured, exported, and reviewed,
 * bee-agent must be able to post-train a *local* model on that dataset so it can
 * (i) repeat the recorded movements and (ii) generalize to new-but-related
 * movements. The real on-device training runs when the user runs bee-agent
 * locally (see `LocalAppleSiliconTrainingRunner`, which emits MLX/axolotl launch
 * scripts). That path cannot run in the cloud, so this module provides a
 * *backend-agnostic* seam plus a deterministic in-process backend that actually
 * learns from and predicts over the dataset — enough to validate the
 * capture → dataset → train → infer → generalize loop end-to-end in tests.
 *
 * A movement is modelled as a stream of discrete tokens derived from
 * `ReplayTimelineEvent`s. A backend consumes tokenized sequences and returns a
 * `TrainedMovementModel` that can predict the next token given a context window
 * (repetition) and roll forward a whole continuation (replay/generalization).
 */

/** A single discrete unit of movement/observation, e.g. `action:click`. */
export type MovementToken = string;

/** Sentinel appended to every training sequence so models can learn to stop. */
export const MOVEMENT_END_TOKEN: MovementToken = "<end>";

/** One recorded movement sequence, already tokenized. */
export type MovementSequence = {
  /** Stable id (usually the source trajectory/session id) for traceability. */
  id: string;
  tokens: MovementToken[];
};

/** A ranked next-token guess with a normalized probability in [0, 1]. */
export type MovementPrediction = {
  token: MovementToken;
  probability: number;
};

export type MovementModelBackendInfo = {
  /** Backend identifier, surfaced in manifests and telemetry. */
  name: string;
  /** True when the backend runs entirely in-process (no external runtime). */
  inProcess: boolean;
};

export type TrainMovementModelOptions = {
  /**
   * Highest context order the model may condition on. Larger orders memorize
   * longer movements exactly; backoff to smaller orders is what lets the model
   * generalize to unseen contexts. Defaults to 3.
   */
  order?: number;
};

export type MovementGenerateOptions = {
  /** Hard cap on generated tokens (excludes the prompt). Defaults to 64. */
  maxSteps?: number;
  /**
   * Stop as soon as `MOVEMENT_END_TOKEN` is produced. Defaults to true. The end
   * token is not included in the returned continuation.
   */
  stopAtEnd?: boolean;
};

/** A trained model instance produced by a backend. */
export type TrainedMovementModel = {
  readonly backend: string;
  readonly order: number;
  readonly vocabulary: MovementToken[];
  /** Ranked next-token distribution for a context (most probable first). */
  predict(context: MovementToken[]): MovementPrediction[];
  /** Single most probable next token, or `undefined` if the model is empty. */
  predictNext(context: MovementToken[]): MovementToken | undefined;
  /** Roll the model forward deterministically from a prompt. */
  generate(prompt: MovementToken[], options?: MovementGenerateOptions): MovementToken[];
  /** Portable, JSON-serializable form for persistence and cross-process reload. */
  serialize(): SerializedMovementModel;
};

/** Backends turn a tokenized dataset into a `TrainedMovementModel`. */
export type MovementModelBackend = {
  readonly info: MovementModelBackendInfo;
  train(dataset: MovementSequence[], options?: TrainMovementModelOptions): Promise<TrainedMovementModel>;
};

/**
 * Tokenize a replay timeline into movement tokens. Only `action` and
 * `observation` events describe device/UI movement; `transcript` events are
 * conversational context and are dropped. Tokens are intentionally coarse
 * (kind + tool/source) so the model learns *movement structure* rather than
 * memorizing free-text summaries — this is what makes generalization possible.
 */
export function tokenizeReplayEvents(events: ReplayTimelineEvent[]): MovementToken[] {
  const tokens: MovementToken[] = [];
  for (const event of events) {
    if (event.kind === "action") {
      tokens.push(`action:${slug(event.tool)}`);
    } else if (event.kind === "observation") {
      tokens.push(`observe:${slug(event.source)}`);
    }
  }
  return tokens;
}

/** Build a training sequence from a replay, appending the end sentinel. */
export function toMovementSequence(id: string, events: ReplayTimelineEvent[]): MovementSequence {
  return { id, tokens: [...tokenizeReplayEvents(events), MOVEMENT_END_TOKEN] };
}

function slug(value: string): string {
  return value.trim().toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "") || "unknown";
}

// ---------------------------------------------------------------------------
// Markov backoff backend (deterministic, in-process reference implementation)
// ---------------------------------------------------------------------------

export type SerializedMovementModel = {
  version: 1;
  backend: string;
  order: number;
  vocabulary: MovementToken[];
  /** context-string -> (token -> observed count). Context "" is the unigram. */
  counts: Record<string, Record<MovementToken, number>>;
};

const CONTEXT_SEPARATOR = "";

/**
 * A variable-order Markov model with stupid-backoff smoothing.
 *
 * Learning: counts every (context, next-token) pair for context lengths
 * `0..order`. Repetition: with enough data the highest-order context recovers
 * recorded movements exactly. Generalization: when a context was never seen at
 * full order, `predict` backs off to progressively shorter contexts, so a novel
 * combination of familiar sub-movements still yields a sensible next step.
 */
export class MarkovMovementModel implements TrainedMovementModel {
  readonly backend: string;
  readonly order: number;
  readonly vocabulary: MovementToken[];
  private readonly counts: Map<string, Map<MovementToken, number>>;

  constructor(params: {
    backend: string;
    order: number;
    counts: Map<string, Map<MovementToken, number>>;
  }) {
    this.backend = params.backend;
    this.order = params.order;
    this.counts = params.counts;
    const vocab = new Set<MovementToken>();
    for (const successors of params.counts.values()) {
      for (const token of successors.keys()) {
        vocab.add(token);
      }
    }
    this.vocabulary = [...vocab].sort();
  }

  predict(context: MovementToken[]): MovementPrediction[] {
    for (let length = Math.min(this.order, context.length); length >= 0; length -= 1) {
      const key = contextKey(context.slice(context.length - length));
      const successors = this.counts.get(key);
      if (successors && successors.size > 0) {
        return normalize(successors);
      }
    }
    return [];
  }

  predictNext(context: MovementToken[]): MovementToken | undefined {
    return this.predict(context)[0]?.token;
  }

  generate(prompt: MovementToken[], options: MovementGenerateOptions = {}): MovementToken[] {
    const maxSteps = options.maxSteps ?? 64;
    const stopAtEnd = options.stopAtEnd ?? true;
    const context = [...prompt];
    const generated: MovementToken[] = [];
    for (let step = 0; step < maxSteps; step += 1) {
      const next = this.predictNext(context);
      if (next === undefined) {
        break;
      }
      if (next === MOVEMENT_END_TOKEN) {
        if (stopAtEnd) {
          break;
        }
        generated.push(next);
        context.push(next);
        continue;
      }
      generated.push(next);
      context.push(next);
    }
    return generated;
  }

  serialize(): SerializedMovementModel {
    const counts: Record<string, Record<MovementToken, number>> = {};
    for (const [key, successors] of this.counts) {
      const entry: Record<MovementToken, number> = {};
      for (const [token, count] of successors) {
        entry[token] = count;
      }
      counts[key] = entry;
    }
    return {
      version: 1,
      backend: this.backend,
      order: this.order,
      vocabulary: [...this.vocabulary],
      counts,
    };
  }

  static deserialize(serialized: SerializedMovementModel): MarkovMovementModel {
    const counts = new Map<string, Map<MovementToken, number>>();
    for (const [key, successors] of Object.entries(serialized.counts)) {
      const inner = new Map<MovementToken, number>();
      for (const [token, count] of Object.entries(successors)) {
        inner.set(token, count);
      }
      counts.set(key, inner);
    }
    return new MarkovMovementModel({ backend: serialized.backend, order: serialized.order, counts });
  }
}

/** In-process reference backend. Deterministic — safe for cloud/CI tests. */
export class MarkovMovementBackend implements MovementModelBackend {
  readonly info: MovementModelBackendInfo = { name: "markov-backoff", inProcess: true };

  async train(dataset: MovementSequence[], options: TrainMovementModelOptions = {}): Promise<TrainedMovementModel> {
    const order = Math.max(0, Math.floor(options.order ?? 3));
    const counts = new Map<string, Map<MovementToken, number>>();
    for (const sequence of dataset) {
      const tokens = sequence.tokens;
      for (let index = 0; index < tokens.length; index += 1) {
        const next = tokens[index]!;
        for (let length = 0; length <= order; length += 1) {
          if (length > index) {
            break;
          }
          const key = contextKey(tokens.slice(index - length, index));
          let successors = counts.get(key);
          if (!successors) {
            successors = new Map<MovementToken, number>();
            counts.set(key, successors);
          }
          successors.set(next, (successors.get(next) ?? 0) + 1);
        }
      }
    }
    return new MarkovMovementModel({ backend: this.info.name, order, counts });
  }
}

/**
 * Registry so callers can select a backend by name and the seam stays open for
 * a real on-device small-model backend added later. New backends register here.
 */
export class MovementModelBackendRegistry {
  private readonly backends = new Map<string, MovementModelBackend>();

  constructor(backends: MovementModelBackend[] = [new MarkovMovementBackend()]) {
    for (const backend of backends) {
      this.register(backend);
    }
  }

  register(backend: MovementModelBackend): void {
    this.backends.set(backend.info.name, backend);
  }

  get(name: string): MovementModelBackend | undefined {
    return this.backends.get(name);
  }

  list(): MovementModelBackendInfo[] {
    return [...this.backends.values()].map((backend) => backend.info);
  }
}

// ---------------------------------------------------------------------------
// Replay-fidelity / generalization eval
// ---------------------------------------------------------------------------

export type MovementReplayEval = {
  /** Tokens the model produced given the prompt. */
  predicted: MovementToken[];
  /** Ground-truth continuation (excluding the prompt). */
  expected: MovementToken[];
  /** Longest common prefix length between predicted and expected. */
  matchedPrefix: number;
  /** matchedPrefix / expected.length, in [0, 1]. 1.0 == exact replay. */
  fidelity: number;
};

/**
 * Measure how faithfully a model replays a sequence: prompt it with the first
 * `promptLength` tokens and compare its rolled-forward continuation against the
 * true remainder. Run over a held-out but related sequence, this is the
 * generalization signal for objective 2(d).
 */
export function evaluateReplayFidelity(
  model: TrainedMovementModel,
  sequence: MovementSequence,
  promptLength = 1,
): MovementReplayEval {
  const tokens = stripEnd(sequence.tokens);
  const boundedPrompt = Math.max(0, Math.min(promptLength, tokens.length));
  const prompt = tokens.slice(0, boundedPrompt);
  const expected = tokens.slice(boundedPrompt);
  const predicted = model.generate(prompt, { maxSteps: expected.length });
  let matchedPrefix = 0;
  while (
    matchedPrefix < expected.length &&
    matchedPrefix < predicted.length &&
    expected[matchedPrefix] === predicted[matchedPrefix]
  ) {
    matchedPrefix += 1;
  }
  return {
    predicted,
    expected,
    matchedPrefix,
    fidelity: expected.length === 0 ? 1 : matchedPrefix / expected.length,
  };
}

function stripEnd(tokens: MovementToken[]): MovementToken[] {
  return tokens[tokens.length - 1] === MOVEMENT_END_TOKEN ? tokens.slice(0, -1) : [...tokens];
}

function contextKey(context: MovementToken[]): string {
  return context.join(CONTEXT_SEPARATOR);
}

function normalize(successors: Map<MovementToken, number>): MovementPrediction[] {
  const total = [...successors.values()].reduce((sum, count) => sum + count, 0);
  const predictions = [...successors.entries()].map(([token, count]) => ({
    token,
    probability: total > 0 ? count / total : 0,
  }));
  predictions.sort((a, b) => {
    if (b.probability !== a.probability) {
      return b.probability - a.probability;
    }
    return a.token < b.token ? -1 : a.token > b.token ? 1 : 0;
  });
  return predictions;
}
