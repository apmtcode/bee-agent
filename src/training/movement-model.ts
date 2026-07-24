import type { ReplayTimelineEvent } from "../capture/replay.js";

/**
 * Movement-model backend: the in-process learning + inference layer for the
 * local-movement learning subsystem (standing objective #2, parts c & d).
 *
 * The capture/replay pipeline turns recorded mouse/keyboard/UI activity into an
 * ordered stream of {@link ReplayTimelineEvent}s. This module tokenizes that
 * stream into a discrete movement vocabulary, learns a next-action policy from
 * it, and can (a) *repeat* recorded movements and (b) *generalize* to new but
 * related movements via context backoff.
 *
 * Everything here is deterministic and dependency-free so it runs in the cloud
 * (and CI) against synthetic event streams. The {@link MovementModelBackend}
 * interface is the pluggable seam: a real on-device small model (MLX, llama.cpp,
 * etc.) can register a backend with the same contract and be selected at
 * runtime, while the bundled {@link NgramMovementBackend} is the always-available
 * reference/mock implementation.
 */

// A movement is discretized into a stable string token, e.g. "act:device:tap".
export type MovementToken = string;

export type MovementTokenKind = "observation" | "action" | "transcript";

export type TaggedMovementToken = {
  token: MovementToken;
  kind: MovementTokenKind;
};

// One supervised training example: predict `action` given the preceding
// `context` (most-recent token last).
export type MovementExample = {
  context: MovementToken[];
  action: MovementToken;
};

export type MovementPredictionSource = "exact" | "backoff" | "prior";

export type MovementPrediction = {
  action: MovementToken;
  /** Probability of the chosen action under the matched context (0..1). */
  confidence: number;
  /** Length of the context suffix that produced the prediction (0 = prior). */
  matchedOrder: number;
  source: MovementPredictionSource;
  /** Ranked alternatives (most likely first), including the chosen action. */
  candidates: Array<{ action: MovementToken; probability: number }>;
};

export type SerializedMovementModel = {
  backendId: string;
  order: number;
  /** Keyed by "<n><joined-context>" -> action -> observed count. */
  grams: Record<string, Record<MovementToken, number>>;
  /** Unigram action counts (the n=0 backoff). */
  prior: Record<MovementToken, number>;
};

export type MovementModelHyperparams = {
  /** Maximum context length the model conditions on. Defaults to 3. */
  order?: number;
};

export interface TrainedMovementModel {
  readonly backendId: string;
  readonly order: number;
  readonly vocabulary: MovementToken[];
  /** Predict the next action for a context (most-recent token last). */
  predict(context: MovementToken[]): MovementPrediction | undefined;
  /** Roll out a movement sequence from a seed context. */
  generate(seed: MovementToken[], steps: number): MovementToken[];
  serialize(): SerializedMovementModel;
}

export interface MovementModelBackend {
  readonly id: string;
  train(examples: MovementExample[], hyperparams?: MovementModelHyperparams): TrainedMovementModel;
  load(serialized: SerializedMovementModel): TrainedMovementModel;
}

const CONTEXT_SEPARATOR = "";
const DEFAULT_ORDER = 3;
const MAX_CANDIDATES = 5;

// ---------------------------------------------------------------------------
// Tokenization
// ---------------------------------------------------------------------------

export type TokenizeOptions = {
  /** Include transcript (message) events as context tokens. Default false. */
  includeTranscript?: boolean;
};

/** Slug a free-text summary into a stable, comparable token fragment. */
export function slugifyMovement(value: string): string {
  const slug = value
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
  return slug.length > 0 ? slug.slice(0, 48) : "unknown";
}

/**
 * Turn an ordered replay timeline into tagged movement tokens. Observations
 * become `obs:<source>:<slug>`, actions become `act:<tool>:<slug>`, transcript
 * messages (when included) become `msg:<role>`.
 */
export function tokenizeReplayEvents(
  events: readonly ReplayTimelineEvent[],
  options: TokenizeOptions = {},
): TaggedMovementToken[] {
  const tokens: TaggedMovementToken[] = [];
  for (const event of events) {
    switch (event.kind) {
      case "observation":
        tokens.push({ kind: "observation", token: `obs:${slugifyMovement(event.source)}:${slugifyMovement(event.summary)}` });
        break;
      case "action":
        tokens.push({ kind: "action", token: `act:${slugifyMovement(event.tool)}:${slugifyMovement(event.summary)}` });
        break;
      case "transcript":
        if (options.includeTranscript) {
          tokens.push({ kind: "transcript", token: `msg:${event.role}` });
        }
        break;
    }
  }
  return tokens;
}

/**
 * Build supervised examples from a tagged token stream. Each *action* token is
 * a prediction target; its context is the up-to-`order` tokens that preceded it
 * (observations and prior actions alike).
 */
export function buildMovementExamples(tokens: readonly TaggedMovementToken[], order: number = DEFAULT_ORDER): MovementExample[] {
  const boundedOrder = Math.max(1, Math.floor(order));
  const examples: MovementExample[] = [];
  for (let index = 0; index < tokens.length; index += 1) {
    const current = tokens[index];
    if (current.kind !== "action") {
      continue;
    }
    const start = Math.max(0, index - boundedOrder);
    const context = tokens.slice(start, index).map((entry) => entry.token);
    examples.push({ context, action: current.token });
  }
  return examples;
}

// ---------------------------------------------------------------------------
// Backend: deterministic n-gram with context backoff
// ---------------------------------------------------------------------------

class NgramMovementModel implements TrainedMovementModel {
  readonly backendId: string;
  readonly order: number;
  readonly vocabulary: MovementToken[];
  private readonly grams: Map<string, Map<MovementToken, number>>;
  private readonly prior: Map<MovementToken, number>;

  constructor(backendId: string, order: number, grams: Map<string, Map<MovementToken, number>>, prior: Map<MovementToken, number>) {
    this.backendId = backendId;
    this.order = order;
    this.grams = grams;
    this.prior = prior;
    this.vocabulary = [...prior.keys()].sort();
  }

  predict(context: MovementToken[]): MovementPrediction | undefined {
    const requestedOrder = Math.min(this.order, context.length);
    for (let n = requestedOrder; n >= 1; n -= 1) {
      const key = gramKey(n, context.slice(context.length - n));
      const counts = this.grams.get(key);
      if (counts) {
        const source: MovementPredictionSource = n === requestedOrder ? "exact" : "backoff";
        return buildPrediction(counts, n, source);
      }
    }
    if (this.prior.size > 0) {
      return buildPrediction(this.prior, 0, "prior");
    }
    return undefined;
  }

  generate(seed: MovementToken[], steps: number): MovementToken[] {
    const boundedSteps = Math.max(0, Math.floor(steps));
    const context = [...seed];
    const produced: MovementToken[] = [];
    for (let step = 0; step < boundedSteps; step += 1) {
      const prediction = this.predict(context.slice(Math.max(0, context.length - this.order)));
      if (!prediction) {
        break;
      }
      produced.push(prediction.action);
      context.push(prediction.action);
    }
    return produced;
  }

  serialize(): SerializedMovementModel {
    const grams: Record<string, Record<MovementToken, number>> = {};
    for (const [key, counts] of this.grams) {
      grams[key] = Object.fromEntries([...counts.entries()].sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0)));
    }
    const prior = Object.fromEntries([...this.prior.entries()].sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0)));
    return { backendId: this.backendId, order: this.order, grams, prior };
  }
}

/**
 * Reference/mock backend: an order-k n-gram next-action policy with stupid
 * backoff. Longer matching contexts win (exact repeat); when an unseen context
 * appears, it backs off to progressively shorter contexts and finally the
 * unigram prior — this is what lets it generalize to new but related movements.
 * Fully deterministic: ties break by token order, so training + inference are
 * reproducible in the cloud.
 */
export class NgramMovementBackend implements MovementModelBackend {
  readonly id = "ngram-backoff";

  train(examples: MovementExample[], hyperparams: MovementModelHyperparams = {}): TrainedMovementModel {
    const order = Math.max(1, Math.floor(hyperparams.order ?? DEFAULT_ORDER));
    const grams = new Map<string, Map<MovementToken, number>>();
    const prior = new Map<MovementToken, number>();
    for (const example of examples) {
      increment(prior, example.action);
      const maxN = Math.min(order, example.context.length);
      for (let n = 1; n <= maxN; n += 1) {
        const key = gramKey(n, example.context.slice(example.context.length - n));
        let counts = grams.get(key);
        if (!counts) {
          counts = new Map<MovementToken, number>();
          grams.set(key, counts);
        }
        increment(counts, example.action);
      }
    }
    return new NgramMovementModel(this.id, order, grams, prior);
  }

  load(serialized: SerializedMovementModel): TrainedMovementModel {
    const grams = new Map<string, Map<MovementToken, number>>();
    for (const [key, counts] of Object.entries(serialized.grams)) {
      grams.set(key, new Map(Object.entries(counts)));
    }
    const prior = new Map(Object.entries(serialized.prior));
    return new NgramMovementModel(serialized.backendId, serialized.order, grams, prior);
  }
}

// ---------------------------------------------------------------------------
// Registry (pluggable backend selection)
// ---------------------------------------------------------------------------

export class MovementModelRegistry {
  private readonly backends = new Map<string, MovementModelBackend>();

  constructor(backends: MovementModelBackend[] = []) {
    for (const backend of backends) {
      this.register(backend);
    }
  }

  register(backend: MovementModelBackend): void {
    this.backends.set(backend.id, backend);
  }

  has(id: string): boolean {
    return this.backends.has(id);
  }

  get(id: string): MovementModelBackend {
    const backend = this.backends.get(id);
    if (!backend) {
      throw new Error(`Unknown movement-model backend: ${id}. Registered: ${this.list().join(", ") || "(none)"}`);
    }
    return backend;
  }

  list(): string[] {
    return [...this.backends.keys()].sort();
  }
}

export function createDefaultMovementModelRegistry(): MovementModelRegistry {
  return new MovementModelRegistry([new NgramMovementBackend()]);
}

// ---------------------------------------------------------------------------
// Convenience: train straight from the reviewed export / replay manifests
// ---------------------------------------------------------------------------

export type MovementTrainingSource = {
  replays?: Array<{ events: readonly ReplayTimelineEvent[] }>;
};

export type TrainMovementModelOptions = MovementModelHyperparams &
  TokenizeOptions & {
    backendId?: string;
    registry?: MovementModelRegistry;
  };

/**
 * Tokenize every replay in a reviewed export (or any object exposing
 * `replays[].events`) and train the selected backend on the combined examples.
 */
export function trainMovementModelFromReplays(
  source: MovementTrainingSource,
  options: TrainMovementModelOptions = {},
): TrainedMovementModel {
  const registry = options.registry ?? createDefaultMovementModelRegistry();
  const backend = registry.get(options.backendId ?? new NgramMovementBackend().id);
  const order = options.order ?? DEFAULT_ORDER;
  const examples: MovementExample[] = [];
  for (const replay of source.replays ?? []) {
    const tokens = tokenizeReplayEvents(replay.events, { includeTranscript: options.includeTranscript });
    examples.push(...buildMovementExamples(tokens, order));
  }
  return backend.train(examples, { order });
}

// ---------------------------------------------------------------------------
// Evaluation harness (repeat fidelity + generalization)
// ---------------------------------------------------------------------------

export type MovementEvalResult = {
  totalActions: number;
  correct: number;
  accuracy: number;
  bySource: Record<MovementPredictionSource | "none", { total: number; correct: number }>;
};

/**
 * Walk each held-out token sequence and, at every action position, predict from
 * the preceding context window and compare against the recorded action. Reports
 * overall accuracy plus a breakdown by prediction source, so callers can tell
 * exact-repeat matches apart from backoff-driven generalization.
 */
export function evaluateMovementModel(
  model: TrainedMovementModel,
  sequences: ReadonlyArray<readonly TaggedMovementToken[]>,
): MovementEvalResult {
  const bySource: MovementEvalResult["bySource"] = {
    exact: { total: 0, correct: 0 },
    backoff: { total: 0, correct: 0 },
    prior: { total: 0, correct: 0 },
    none: { total: 0, correct: 0 },
  };
  let totalActions = 0;
  let correct = 0;
  for (const sequence of sequences) {
    for (let index = 0; index < sequence.length; index += 1) {
      if (sequence[index].kind !== "action") {
        continue;
      }
      totalActions += 1;
      const start = Math.max(0, index - model.order);
      const context = sequence.slice(start, index).map((entry) => entry.token);
      const prediction = model.predict(context);
      const bucket = prediction ? prediction.source : "none";
      bySource[bucket].total += 1;
      if (prediction && prediction.action === sequence[index].token) {
        correct += 1;
        bySource[bucket].correct += 1;
      }
    }
  }
  return {
    totalActions,
    correct,
    accuracy: totalActions === 0 ? 0 : correct / totalActions,
    bySource,
  };
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

function gramKey(n: number, suffix: MovementToken[]): string {
  return `${n}${CONTEXT_SEPARATOR}${suffix.join(CONTEXT_SEPARATOR)}`;
}

function increment(counts: Map<MovementToken, number>, token: MovementToken): void {
  counts.set(token, (counts.get(token) ?? 0) + 1);
}

function buildPrediction(counts: Map<MovementToken, number>, matchedOrder: number, source: MovementPredictionSource): MovementPrediction {
  let total = 0;
  for (const count of counts.values()) {
    total += count;
  }
  const ranked = [...counts.entries()]
    .sort((a, b) => (b[1] - a[1]) || (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0))
    .map(([action, count]) => ({ action, probability: total === 0 ? 0 : count / total }));
  const candidates = ranked.slice(0, MAX_CANDIDATES);
  const best = ranked[0];
  return {
    action: best.action,
    confidence: best.probability,
    matchedOrder,
    source,
    candidates,
  };
}
