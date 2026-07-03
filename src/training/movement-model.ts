/**
 * Pluggable local-movement model backend.
 *
 * The capture subsystem records local movements/actions (mouse, keyboard,
 * gestures, window/OS events) into {@link TrajectorySpan}s and replay
 * manifests. This module turns those recorded action sequences into a discrete
 * token dataset and fits a small, *local*, in-process model that can (a) repeat
 * the recorded movements and (b) generalize to new-but-related movements.
 *
 * The real on-device training (MLX/axolotl) is orchestrated by
 * {@link ../training/runner.ts}. That path cannot run in the cloud/CI. This
 * module provides a deterministic, dependency-free backend (an n-gram Markov
 * model with Katz-style backoff) so the train -> infer -> generalize loop is
 * fully exercised by tests without any real OS access — and a
 * {@link MovementModelBackend} seam so a real small local model can be swapped
 * in via {@link registerMovementBackend}.
 */

export type MovementToken = string;

export type MovementEvent = {
  tool: string;
  summary: string;
  metadata?: Record<string, unknown>;
};

export type MovementSequence = {
  id: string;
  tokens: MovementToken[];
};

export type MovementDataset = {
  version: 1;
  sequences: MovementSequence[];
  /** Sorted unique tokens observed across all sequences. */
  vocabulary: MovementToken[];
};

export type MovementActionTokenizer = (event: MovementEvent) => MovementToken;

/** Marks the end of a recorded sequence so generation knows when to stop. */
export const MOVEMENT_END_TOKEN: MovementToken = "<end>";

function slug(value: string): string {
  return value
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
}

function metaString(metadata: Record<string, unknown> | undefined, key: string): string | undefined {
  const value = metadata?.[key];
  return typeof value === "string" && value.trim().length > 0 ? value : undefined;
}

/**
 * Canonical, structure-preserving token for a recorded action. Device gestures
 * (which carry `gesture`/`direction`/`target` metadata) tokenize to a compound
 * `tool:gesture:direction:target` symbol so related movements share a prefix and
 * the model can generalize across them; everything else falls back to
 * `tool:summary-slug`.
 */
export function defaultActionTokenizer(event: MovementEvent): MovementToken {
  const tool = slug(event.tool) || "action";
  const gesture = metaString(event.metadata, "gesture");
  if (gesture) {
    const parts = [tool, slug(gesture)];
    const direction = metaString(event.metadata, "direction");
    const target = metaString(event.metadata, "target");
    if (direction) {
      parts.push(slug(direction));
    }
    if (target) {
      parts.push(slug(target));
    }
    return parts.filter((part) => part.length > 0).join(":");
  }
  const summary = slug(event.summary) || "step";
  return `${tool}:${summary}`;
}

function collectVocabulary(sequences: MovementSequence[]): MovementToken[] {
  const vocabulary = new Set<MovementToken>();
  for (const sequence of sequences) {
    for (const token of sequence.tokens) {
      vocabulary.add(token);
    }
  }
  return [...vocabulary].sort();
}

/**
 * Build a {@link MovementDataset} from any source of ordered movement events.
 * Each source item becomes one sequence; a trailing {@link MOVEMENT_END_TOKEN}
 * is appended so a trained model can learn where recorded movements terminate.
 */
export function buildMovementDataset(params: {
  sources: Array<{ id: string; events: MovementEvent[] }>;
  tokenizer?: MovementActionTokenizer;
  appendEndToken?: boolean;
}): MovementDataset {
  const tokenizer = params.tokenizer ?? defaultActionTokenizer;
  const appendEndToken = params.appendEndToken ?? true;
  const sequences: MovementSequence[] = params.sources
    .map((source) => {
      const tokens = source.events.map((event) => tokenizer(event));
      if (appendEndToken && tokens.length > 0) {
        tokens.push(MOVEMENT_END_TOKEN);
      }
      return { id: source.id, tokens };
    })
    .filter((sequence) => sequence.tokens.length > 0);

  return {
    version: 1,
    sequences,
    vocabulary: collectVocabulary(sequences),
  };
}

/** Adapter: derive movement sources from trajectory spans (their `actions`). */
export function movementSourcesFromTrajectories(
  trajectories: Array<{ id: string; actions: MovementEvent[] }>,
): Array<{ id: string; events: MovementEvent[] }> {
  return trajectories.map((trajectory) => ({ id: trajectory.id, events: trajectory.actions }));
}

/** Adapter: derive movement sources from replay manifests (their action events). */
export function movementSourcesFromReplays(
  replays: Array<{
    sessionId: string;
    events: Array<{ kind: string; tool?: string; summary?: string; metadata?: Record<string, unknown> }>;
  }>,
): Array<{ id: string; events: MovementEvent[] }> {
  return replays.map((replay) => ({
    id: replay.sessionId,
    events: replay.events
      .filter((event) => event.kind === "action")
      .map((event) => ({
        tool: event.tool ?? "action",
        summary: event.summary ?? "",
        ...(event.metadata ? { metadata: event.metadata } : {}),
      })),
  }));
}

export type MovementPrediction = {
  token: MovementToken;
  probability: number;
};

export type MovementTrainingOptions = {
  /** Maximum Markov context length (n-gram order). Default 2. */
  order?: number;
  /** Additive (Laplace) smoothing applied to every vocabulary token. Default 0. */
  smoothing?: number;
};

export type MovementGenerateOptions = {
  maxLength: number;
  /**
   * "argmax" (default) deterministically follows the highest-probability
   * transition — ideal for repeating a recorded movement. "sample" draws from
   * the distribution using the provided seeded RNG for reproducible variety.
   */
  strategy?: "argmax" | "sample";
  rng?: () => number;
  /** Stop generating when this token is produced. Default MOVEMENT_END_TOKEN. */
  stopToken?: MovementToken | null;
};

export type MovementModelEvaluation = {
  sequenceCount: number;
  tokenCount: number;
  /** Fraction of positions where the model's top-1 prediction matched. */
  topOneAccuracy: number;
  /** Mean per-token perplexity across all evaluated positions (lower is better). */
  perplexity: number;
};

export type SerializedMovementModel = {
  version: 1;
  backend: string;
  order: number;
  smoothing: number;
  vocabulary: MovementToken[];
  /** context-key -> (nextToken -> count). "" is the empty (unigram) context. */
  contexts: Record<string, Record<MovementToken, number>>;
};

export interface TrainedMovementModel {
  readonly backend: string;
  readonly order: number;
  readonly vocabulary: MovementToken[];
  /** Ranked next-token distribution for a context (most probable first). */
  predictNext(context: MovementToken[]): MovementPrediction[];
  /** Generate a continuation from a seed — repeat (argmax) or vary (sample). */
  generate(seed: MovementToken[], options: MovementGenerateOptions): MovementToken[];
  /** Natural-log likelihood of a full token sequence under the model. */
  sequenceLogLikelihood(tokens: MovementToken[]): number;
  /** Per-token perplexity of a sequence (exp of mean negative log-likelihood). */
  perplexity(tokens: MovementToken[]): number;
  serialize(): SerializedMovementModel;
}

/** Pluggable seam: real on-device backends implement this and register. */
export interface MovementModelBackend {
  readonly name: string;
  train(dataset: MovementDataset, options?: MovementTrainingOptions): TrainedMovementModel;
}

const CONTEXT_DELIMITER = "";

function contextKey(context: MovementToken[]): string {
  return context.join(CONTEXT_DELIMITER);
}

class MarkovMovementModel implements TrainedMovementModel {
  readonly backend: string;
  readonly order: number;
  readonly vocabulary: MovementToken[];
  private readonly smoothing: number;
  private readonly contexts: Map<string, Map<MovementToken, number>>;

  constructor(params: {
    backend: string;
    order: number;
    smoothing: number;
    vocabulary: MovementToken[];
    contexts: Map<string, Map<MovementToken, number>>;
  }) {
    this.backend = params.backend;
    this.order = params.order;
    this.smoothing = params.smoothing;
    this.vocabulary = params.vocabulary;
    this.contexts = params.contexts;
  }

  /**
   * Katz-style backoff: try the longest suffix of `context` (up to `order`)
   * that was observed during training; if none was, fall back to progressively
   * shorter suffixes and finally the unigram distribution. This is what lets an
   * unseen-but-related context still produce a plausible next movement.
   */
  private resolveCounts(context: MovementToken[]): Map<MovementToken, number> {
    for (let length = Math.min(this.order, context.length); length >= 1; length -= 1) {
      const suffix = context.slice(context.length - length);
      const counts = this.contexts.get(contextKey(suffix));
      if (counts && counts.size > 0) {
        return counts;
      }
    }
    return this.contexts.get("") ?? new Map();
  }

  predictNext(context: MovementToken[]): MovementPrediction[] {
    const counts = this.resolveCounts(context);
    const smoothing = this.smoothing;
    const denominator =
      [...counts.values()].reduce((sum, count) => sum + count, 0) + smoothing * this.vocabulary.length;

    const scored = new Map<MovementToken, number>();
    if (smoothing > 0) {
      for (const token of this.vocabulary) {
        scored.set(token, smoothing);
      }
    }
    for (const [token, count] of counts) {
      scored.set(token, (scored.get(token) ?? 0) + count);
    }

    if (denominator === 0 || scored.size === 0) {
      return [];
    }

    return [...scored.entries()]
      .map(([token, weight]) => ({ token, probability: weight / denominator }))
      .sort((a, b) => (b.probability !== a.probability ? b.probability - a.probability : a.token < b.token ? -1 : 1));
  }

  generate(seed: MovementToken[], options: MovementGenerateOptions): MovementToken[] {
    const strategy = options.strategy ?? "argmax";
    const stopToken = options.stopToken === undefined ? MOVEMENT_END_TOKEN : options.stopToken;
    const rng = options.rng ?? (() => 0);
    const produced: MovementToken[] = [];
    let context = [...seed];

    for (let step = 0; step < options.maxLength; step += 1) {
      const predictions = this.predictNext(context.slice(Math.max(0, context.length - this.order)));
      if (predictions.length === 0) {
        break;
      }
      const next = strategy === "sample" ? samplePrediction(predictions, rng()) : predictions[0].token;
      if (stopToken !== null && next === stopToken) {
        break;
      }
      produced.push(next);
      context = [...context, next];
    }
    return produced;
  }

  sequenceLogLikelihood(tokens: MovementToken[]): number {
    let total = 0;
    for (let index = 0; index < tokens.length; index += 1) {
      const context = tokens.slice(Math.max(0, index - this.order), index);
      const predictions = this.predictNext(context);
      const match = predictions.find((prediction) => prediction.token === tokens[index]);
      // Unseen token under a zero-smoothing model: floor at a tiny epsilon so the
      // likelihood is finite (and perplexity is comparable) rather than -Infinity.
      const probability = match?.probability ?? 1e-9;
      total += Math.log(probability);
    }
    return total;
  }

  perplexity(tokens: MovementToken[]): number {
    if (tokens.length === 0) {
      return 1;
    }
    return Math.exp(-this.sequenceLogLikelihood(tokens) / tokens.length);
  }

  serialize(): SerializedMovementModel {
    const contexts: Record<string, Record<MovementToken, number>> = {};
    for (const [key, counts] of this.contexts) {
      const outputKey = key === "" ? "" : key.split(CONTEXT_DELIMITER).join(" ");
      contexts[outputKey] = Object.fromEntries([...counts.entries()]);
    }
    return {
      version: 1,
      backend: this.backend,
      order: this.order,
      smoothing: this.smoothing,
      vocabulary: [...this.vocabulary],
      contexts,
    };
  }
}

function samplePrediction(predictions: MovementPrediction[], roll: number): MovementToken {
  const total = predictions.reduce((sum, prediction) => sum + prediction.probability, 0);
  let threshold = roll * total;
  for (const prediction of predictions) {
    threshold -= prediction.probability;
    if (threshold <= 0) {
      return prediction.token;
    }
  }
  return predictions[predictions.length - 1].token;
}

/**
 * Deterministic n-gram Markov backend. Fully in-process and dependency-free, so
 * the movement train/infer loop runs in the cloud and in CI. Serves as the
 * mock/reference backend behind {@link MovementModelBackend}; register a real
 * on-device small model with the same interface to swap it in.
 */
export class MarkovMovementBackend implements MovementModelBackend {
  readonly name = "markov";

  train(dataset: MovementDataset, options?: MovementTrainingOptions): TrainedMovementModel {
    const order = Math.max(1, Math.floor(options?.order ?? 2));
    const smoothing = Math.max(0, options?.smoothing ?? 0);
    const contexts = new Map<string, Map<MovementToken, number>>();

    const bump = (context: MovementToken[], next: MovementToken): void => {
      const key = contextKey(context);
      const counts = contexts.get(key) ?? new Map<MovementToken, number>();
      counts.set(next, (counts.get(next) ?? 0) + 1);
      contexts.set(key, counts);
    };

    for (const sequence of dataset.sequences) {
      for (let index = 0; index < sequence.tokens.length; index += 1) {
        const next = sequence.tokens[index];
        // Record the transition at every context length from 0..order so backoff
        // has counts to fall through to when a longer context is unseen.
        for (let length = 0; length <= order; length += 1) {
          if (length > index) {
            break;
          }
          bump(sequence.tokens.slice(index - length, index), next);
        }
      }
    }

    const vocabulary = dataset.vocabulary.length > 0 ? [...dataset.vocabulary] : collectVocabulary(dataset.sequences);
    return new MarkovMovementModel({ backend: this.name, order, smoothing, vocabulary, contexts });
  }
}

const backends = new Map<string, MovementModelBackend>();

/** Register (or override) a movement-model backend by name. */
export function registerMovementBackend(backend: MovementModelBackend): void {
  backends.set(backend.name, backend);
}

/** Resolve a registered backend (defaults to the deterministic "markov" one). */
export function createMovementModelBackend(name = "markov"): MovementModelBackend {
  const backend = backends.get(name);
  if (!backend) {
    throw new Error(`Unknown movement-model backend: ${name}. Registered: ${listMovementBackends().join(", ") || "none"}`);
  }
  return backend;
}

/** List the names of all registered backends. */
export function listMovementBackends(): string[] {
  return [...backends.keys()].sort();
}

// The deterministic Markov backend is always available.
registerMovementBackend(new MarkovMovementBackend());

/** Reconstruct a trained model from its serialized form (round-trippable). */
export function deserializeMovementModel(serialized: SerializedMovementModel): TrainedMovementModel {
  const contexts = new Map<string, Map<MovementToken, number>>();
  for (const [key, counts] of Object.entries(serialized.contexts)) {
    const innerKey = key === "" ? "" : key.split(" ").join(CONTEXT_DELIMITER);
    contexts.set(innerKey, new Map(Object.entries(counts)));
  }
  return new MarkovMovementModel({
    backend: serialized.backend,
    order: serialized.order,
    smoothing: serialized.smoothing,
    vocabulary: [...serialized.vocabulary],
    contexts,
  });
}

/**
 * Generalization eval harness: measure a trained model against held-out
 * (unseen-but-related) sequences via next-token top-1 accuracy and perplexity.
 */
export function evaluateMovementModel(
  model: TrainedMovementModel,
  heldOut: MovementSequence[],
): MovementModelEvaluation {
  let tokenCount = 0;
  let correct = 0;
  let logLikelihood = 0;

  for (const sequence of heldOut) {
    for (let index = 0; index < sequence.tokens.length; index += 1) {
      const context = sequence.tokens.slice(Math.max(0, index - model.order), index);
      const predictions = model.predictNext(context);
      const expected = sequence.tokens[index];
      if (predictions.length > 0 && predictions[0].token === expected) {
        correct += 1;
      }
      const probability = predictions.find((prediction) => prediction.token === expected)?.probability ?? 1e-9;
      logLikelihood += Math.log(probability);
      tokenCount += 1;
    }
  }

  return {
    sequenceCount: heldOut.length,
    tokenCount,
    topOneAccuracy: tokenCount === 0 ? 0 : correct / tokenCount,
    perplexity: tokenCount === 0 ? 1 : Math.exp(-logLikelihood / tokenCount),
  };
}

/**
 * Small seeded PRNG (mulberry32). `Math.random` is fine in production but tests
 * need reproducibility, so synthetic generation and "sample" generation accept
 * an explicit RNG built from a seed.
 */
export function createSeededRng(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export type SyntheticMovementFlow = {
  /** e.g. "open-app", "focus-field", "type", "submit" — a stage template. */
  stages: MovementEvent[][];
};

/**
 * Deterministically synthesize related movement sequences from a set of flow
 * templates. Each generated sequence walks the flow, picking one event variant
 * per stage via the seeded RNG. Used to validate capture -> dataset -> train ->
 * generalize round-trips without any real OS input.
 */
export function generateSyntheticMovementSequences(params: {
  flow: SyntheticMovementFlow;
  count: number;
  seed: number;
  idPrefix?: string;
}): Array<{ id: string; events: MovementEvent[] }> {
  const rng = createSeededRng(params.seed);
  const idPrefix = params.idPrefix ?? "synthetic";
  const sequences: Array<{ id: string; events: MovementEvent[] }> = [];

  for (let index = 0; index < params.count; index += 1) {
    const events: MovementEvent[] = [];
    for (const stage of params.flow.stages) {
      if (stage.length === 0) {
        continue;
      }
      const choice = Math.min(stage.length - 1, Math.floor(rng() * stage.length));
      events.push(stage[choice]);
    }
    sequences.push({ id: `${idPrefix}-${index}`, events });
  }
  return sequences;
}
