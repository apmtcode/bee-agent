import type { TrajectoryAction, TrajectorySpan } from "../capture/trajectory.js";
import type { ReplayTimelineEvent } from "../capture/replay.js";

/**
 * Local-movement learning: model layer.
 *
 * This module implements the "post-train a local model to repeat recorded
 * movements, and generalize to new but related movements" objective in a way
 * that is fully testable in the cloud. It provides:
 *
 *  - a compact, replayable movement token schema derived from captured actions;
 *  - a pluggable model-backend interface (`MovementModelBackend`) so a real
 *    on-device small model can be dropped in later without touching callers;
 *  - a deterministic reference backend (`MarkovMovementBackend`) that actually
 *    learns action->action transitions with abstraction backoff, so it both
 *    reproduces recorded movements and generalizes to related ones; and
 *  - snapshot serialization so a trained model can be persisted / shipped and
 *    reloaded on-device.
 *
 * The Markov backend is deliberately dependency-free and deterministic — it is
 * the mock/reference backend that keeps CI green with no GPU, no native model,
 * and no randomness. Real backends (MLX, llama.cpp, ...) implement the same
 * interface and register themselves in a `MovementBackendRegistry`.
 */

const FIELD_SEPARATOR = "␟"; // unit separator glyph; safe inside JSON keys
const END_TOKEN_KEY = "␄"; // end-of-sequence sentinel key
const START_TOKEN_KEY = "␃"; // start-of-sequence anchor (target-agnostic)

/** A single discretized movement the model learns over. */
export type MovementToken = {
  tool: string;
  /** normalized verb, e.g. a device gesture kind ("tap") or the action's lead word. */
  action: string;
  /** optional UI/element target the movement acted on. */
  target?: string;
};

/** An ordered movement demonstration (one captured trajectory's actions). */
export type MovementSequence = {
  id: string;
  /** optional conditioning label (e.g. appId or goal) — reserved for richer backends. */
  context?: string;
  tokens: MovementToken[];
};

export type MovementDataset = {
  version: 1;
  sequences: MovementSequence[];
};

/** `undefined` token means end-of-sequence was predicted. */
export type MovementCandidate = {
  token: MovementToken | undefined;
  probability: number;
};

export type MovementPrediction = {
  token: MovementToken | undefined;
  confidence: number;
  /** which backoff order produced the prediction (0 = unigram fallback). */
  orderUsed: number;
  /** true when the prediction came from an abstraction (target-agnostic) match. */
  abstracted: boolean;
  candidates: MovementCandidate[];
};

export type MovementGenerateOptions = {
  seed?: MovementToken[];
  maxSteps?: number;
};

export type MovementModelSnapshot = {
  version: 1;
  backend: string;
  order: number;
  /** counts[k] maps a context key of k tokens to next-token-key -> count. */
  counts: Array<Record<string, Record<string, number>>>;
  abstractCounts: Array<Record<string, Record<string, number>>>;
  /** token dictionary: token key -> token object (END omitted). */
  tokens: Record<string, MovementToken>;
};

export interface MovementModel {
  readonly backend: string;
  readonly order: number;
  /** predict the next movement given the movements observed so far. */
  predict(context: MovementToken[]): MovementPrediction;
  /** roll out a full movement sequence (for replay / execution). */
  generate(options?: MovementGenerateOptions): MovementToken[];
  snapshot(): MovementModelSnapshot;
}

export type MovementTrainOptions = {
  /** highest context order to model; predictions back off toward the unigram. */
  order?: number;
};

export interface MovementModelBackend {
  readonly name: string;
  train(dataset: MovementDataset, options?: MovementTrainOptions): MovementModel;
}

/** Stable, collision-resistant key for a concrete movement token. */
export function movementTokenKey(token: MovementToken): string {
  return `${token.tool}${FIELD_SEPARATOR}${token.action}${FIELD_SEPARATOR}${token.target ?? ""}`;
}

/** Target-agnostic key — the structural abstraction used for generalization. */
export function movementAbstractKey(token: MovementToken): string {
  return `${token.tool}${FIELD_SEPARATOR}${token.action}`;
}

function firstWord(summary: string): string {
  const match = summary.trim().toLowerCase().match(/[a-z0-9]+/);
  return match ? match[0] : "act";
}

/** Derive a movement token from a captured trajectory action. */
export function tokenizeAction(action: TrajectoryAction): MovementToken {
  const metadata = action.metadata ?? {};
  const gesture = typeof metadata.gesture === "string" ? metadata.gesture : undefined;
  const verb = gesture ?? firstWord(action.summary);
  const target =
    typeof metadata.target === "string"
      ? metadata.target
      : typeof metadata.direction === "string"
        ? metadata.direction
        : undefined;
  return { tool: action.tool, action: verb, ...(target ? { target } : {}) };
}

/** Derive a movement token from a replay-timeline action event. */
export function tokenizeReplayEvent(event: ReplayTimelineEvent): MovementToken | undefined {
  if (event.kind !== "action") {
    return undefined;
  }
  return { tool: event.tool, action: firstWord(event.summary) };
}

/** Build a training dataset from captured trajectories (actions only, time-ordered). */
export function buildMovementDataset(trajectories: TrajectorySpan[]): MovementDataset {
  return {
    version: 1,
    sequences: trajectories.map((trajectory) => ({
      id: trajectory.id,
      tokens: [...trajectory.actions]
        .sort((a, b) => a.ts - b.ts)
        .map((action) => tokenizeAction(action)),
    })),
  };
}

type Counter = Map<string, number>;

function bump(map: Map<string, Counter>, contextKey: string, nextKey: string): void {
  let counter = map.get(contextKey);
  if (!counter) {
    counter = new Map<string, number>();
    map.set(contextKey, counter);
  }
  counter.set(nextKey, (counter.get(nextKey) ?? 0) + 1);
}

function contextKeyFrom(keys: string[]): string {
  return keys.join("␞");
}

/**
 * Deterministic, dependency-free reference backend.
 *
 * Learns next-movement distributions at every context order 1..N with two
 * indexes: a concrete index (keyed on full tokens incl. target) and an
 * abstract index (keyed on target-agnostic tokens). Prediction backs off from
 * the highest concrete order down, then tries the abstract index at each order,
 * and finally the unigram — so an unseen-but-structurally-related context still
 * yields the right action. That backoff chain is what lets a model trained on
 * one set of targets perform the same movements against new targets.
 */
export class MarkovMovementBackend implements MovementModelBackend {
  readonly name = "markov";

  train(dataset: MovementDataset, options?: MovementTrainOptions): MovementModel {
    const order = Math.max(1, Math.min(options?.order ?? 3, 8));
    const concrete: Array<Map<string, Counter>> = [];
    const abstract: Array<Map<string, Counter>> = [];
    for (let k = 0; k <= order; k += 1) {
      concrete.push(new Map());
      abstract.push(new Map());
    }
    const dictionary = new Map<string, MovementToken>();

    for (const sequence of dataset.sequences) {
      const concreteKeys = sequence.tokens.map((token) => movementTokenKey(token));
      const abstractKeys = sequence.tokens.map((token) => movementAbstractKey(token));
      for (let i = 0; i < sequence.tokens.length; i += 1) {
        dictionary.set(concreteKeys[i], sequence.tokens[i]);
      }
      // Pad with a start anchor so the first movement is itself predictable and
      // free rollouts have a real seed context (not the unconditional unigram).
      const paddedConcrete = [START_TOKEN_KEY, ...concreteKeys];
      const paddedAbstract = [START_TOKEN_KEY, ...abstractKeys];
      const total = sequence.tokens.length; // predict positions 0..total (END last)
      for (let i = 0; i <= total; i += 1) {
        const nextKey = i < total ? concreteKeys[i] : END_TOKEN_KEY;
        const historyLen = i + 1; // START plus i real tokens
        for (let k = 1; k <= order; k += 1) {
          if (k > historyLen) {
            continue; // not enough history (incl. START) for this order
          }
          bump(concrete[k], contextKeyFrom(paddedConcrete.slice(historyLen - k, historyLen)), nextKey);
          bump(abstract[k], contextKeyFrom(paddedAbstract.slice(historyLen - k, historyLen)), nextKey);
        }
        // unigram (order 0): unconditional next-token frequency
        bump(concrete[0], "", nextKey);
      }
    }

    return new MarkovMovementModel(order, concrete, abstract, dictionary);
  }

  static fromSnapshot(snapshot: MovementModelSnapshot): MovementModel {
    if (snapshot.backend !== "markov") {
      throw new Error(`Cannot load snapshot from backend "${snapshot.backend}" into markov`);
    }
    const concrete = snapshot.counts.map((level) => deserializeLevel(level));
    const abstract = snapshot.abstractCounts.map((level) => deserializeLevel(level));
    const dictionary = new Map<string, MovementToken>(Object.entries(snapshot.tokens));
    return new MarkovMovementModel(snapshot.order, concrete, abstract, dictionary);
  }
}

function deserializeLevel(level: Record<string, Record<string, number>>): Map<string, Counter> {
  const map = new Map<string, Counter>();
  for (const [contextKey, counts] of Object.entries(level)) {
    map.set(contextKey, new Map(Object.entries(counts)));
  }
  return map;
}

function serializeLevel(level: Map<string, Counter>): Record<string, Record<string, number>> {
  const out: Record<string, Record<string, number>> = {};
  for (const [contextKey, counter] of level.entries()) {
    out[contextKey] = Object.fromEntries(counter.entries());
  }
  return out;
}

class MarkovMovementModel implements MovementModel {
  readonly backend = "markov";

  constructor(
    readonly order: number,
    private readonly concrete: Array<Map<string, Counter>>,
    private readonly abstract: Array<Map<string, Counter>>,
    private readonly dictionary: Map<string, MovementToken>,
  ) {}

  predict(context: MovementToken[]): MovementPrediction {
    const concreteKeys = [START_TOKEN_KEY, ...context.map((token) => movementTokenKey(token))];
    const abstractKeys = [START_TOKEN_KEY, ...context.map((token) => movementAbstractKey(token))];
    const historyLen = concreteKeys.length; // context.length + 1 (the START anchor)

    for (let k = Math.min(this.order, historyLen); k >= 1; k -= 1) {
      const concreteCtx = contextKeyFrom(concreteKeys.slice(historyLen - k));
      const concreteCounter = this.concrete[k]?.get(concreteCtx);
      if (concreteCounter) {
        return this.fromCounter(concreteCounter, k, false);
      }
      const abstractCtx = contextKeyFrom(abstractKeys.slice(historyLen - k));
      const abstractCounter = this.abstract[k]?.get(abstractCtx);
      if (abstractCounter) {
        return this.fromCounter(abstractCounter, k, true);
      }
    }

    const unigram = this.concrete[0]?.get("");
    if (unigram) {
      return this.fromCounter(unigram, 0, false);
    }
    return { token: undefined, confidence: 1, orderUsed: 0, abstracted: false, candidates: [] };
  }

  generate(options: MovementGenerateOptions = {}): MovementToken[] {
    const maxSteps = options.maxSteps ?? 64;
    const produced: MovementToken[] = [...(options.seed ?? [])];
    for (let step = 0; step < maxSteps; step += 1) {
      const prediction = this.predict(produced);
      if (!prediction.token) {
        break;
      }
      produced.push(prediction.token);
    }
    return produced;
  }

  snapshot(): MovementModelSnapshot {
    return {
      version: 1,
      backend: this.backend,
      order: this.order,
      counts: this.concrete.map((level) => serializeLevel(level)),
      abstractCounts: this.abstract.map((level) => serializeLevel(level)),
      tokens: Object.fromEntries(this.dictionary.entries()),
    };
  }

  private fromCounter(counter: Counter, orderUsed: number, abstracted: boolean): MovementPrediction {
    let total = 0;
    for (const count of counter.values()) {
      total += count;
    }
    // Deterministic ranking: by count desc, then token key asc.
    const ranked = [...counter.entries()].sort((a, b) => {
      if (a[1] !== b[1]) {
        return b[1] - a[1];
      }
      return a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0;
    });
    const candidates: MovementCandidate[] = ranked.map(([key, count]) => ({
      token: key === END_TOKEN_KEY ? undefined : this.dictionary.get(key),
      probability: total > 0 ? count / total : 0,
    }));
    const top = candidates[0];
    return {
      token: top?.token,
      confidence: top?.probability ?? 0,
      orderUsed,
      abstracted,
      candidates,
    };
  }
}

/** Pluggable registry so real on-device backends can be registered by name. */
export class MovementBackendRegistry {
  private readonly backends = new Map<string, MovementModelBackend>();

  register(backend: MovementModelBackend): this {
    this.backends.set(backend.name, backend);
    return this;
  }

  get(name: string): MovementModelBackend {
    const backend = this.backends.get(name);
    if (!backend) {
      throw new Error(`Unknown movement backend "${name}" (have: ${[...this.backends.keys()].join(", ") || "none"})`);
    }
    return backend;
  }

  has(name: string): boolean {
    return this.backends.has(name);
  }

  list(): string[] {
    return [...this.backends.keys()];
  }
}

/** Registry seeded with the deterministic reference backend. */
export function createDefaultMovementBackendRegistry(): MovementBackendRegistry {
  return new MovementBackendRegistry().register(new MarkovMovementBackend());
}
