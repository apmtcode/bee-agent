/**
 * Pluggable, in-process movement-model backend for the local-movement learning
 * subsystem (standing objective #2 c/d).
 *
 * The existing training pipeline (`runner.ts`, `execution-service.ts`) only
 * *emits* shell command plans for external on-device runtimes (mlx / axolotl)
 * that cannot run in the cloud/CI. This module adds a real, deterministic,
 * in-process model that:
 *   (c) post-trains on a reviewed movement dataset to *repeat* recorded
 *       movements, and
 *   (d) *generalizes* to new-but-related movement sequences via back-off.
 *
 * The backend is pluggable (`MovementModelBackend`); the default
 * `MarkovMovementBackend` is a variable-order Markov model with stupid-backoff
 * smoothing — fully deterministic (no Date/Math.random), so cloud tests pass.
 * A real on-device small model can implement the same interface later.
 */

/** A canonical token describing a single movement/action. */
export type MovementToken = string;

/** One ordered movement sequence (e.g. all actions from one trajectory). */
export type MovementSample = {
  /** Optional id for traceability (trajectory/session id). */
  id?: string;
  tokens: MovementToken[];
};

export type MovementDataset = {
  samples: MovementSample[];
};

export type MovementCandidate = {
  token: MovementToken;
  probability: number;
};

export type MovementPrediction = {
  token: MovementToken;
  probability: number;
  /** How many trailing context tokens actually matched (back-off depth). */
  contextOrderUsed: number;
  /** All next-token candidates for the matched context, most likely first. */
  candidates: MovementCandidate[];
};

export type MovementModelSnapshot = {
  backend: string;
  order: number;
  /** context (joined tokens) -> next token -> count. */
  transitions: Record<string, Record<MovementToken, number>>;
  vocabulary: MovementToken[];
};

export interface MovementModel {
  readonly backend: string;
  readonly order: number;
  /** Most likely next movement given a context window, or undefined if empty. */
  predictNext(context: MovementToken[]): MovementPrediction | undefined;
  /** Probability the model assigns to a specific next token (with back-off). */
  scoreNext(context: MovementToken[], token: MovementToken): number;
  /** Greedily roll out `length` movements continuing from `seed`. */
  generate(seed: MovementToken[], length: number): MovementToken[];
  toJSON(): MovementModelSnapshot;
}

export interface MovementModelBackend {
  readonly name: string;
  train(dataset: MovementDataset): MovementModel;
}

const CONTEXT_SEPARATOR = "";

// ---------------------------------------------------------------------------
// Tokenization — turn captured actions/replay events into movement tokens.
// ---------------------------------------------------------------------------

export type TokenizableAction = {
  tool: string;
  summary: string;
  metadata?: Record<string, unknown>;
};

function slug(value: string): string {
  return value
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
}

function metaString(metadata: Record<string, unknown> | undefined, key: string): string | undefined {
  const value = metadata?.[key];
  return typeof value === "string" && value.length > 0 ? value : undefined;
}

/**
 * Canonicalize one action into a stable movement token. Structured device
 * gestures (recorded by `DeviceCaptureAdapter` into `metadata`) are preferred
 * because they generalize better than free-text summaries; otherwise the token
 * falls back to `tool:slug(summary)`.
 */
export function canonicalMovementToken(action: TokenizableAction): MovementToken {
  const gesture = metaString(action.metadata, "gesture");
  if (gesture) {
    const direction = metaString(action.metadata, "direction");
    const target = metaString(action.metadata, "target");
    const parts = [slug(gesture)];
    if (direction) {
      parts.push(slug(direction));
    }
    const base = parts.join(":");
    return target ? `${base}>${slug(target)}` : base;
  }
  const tool = slug(action.tool) || "action";
  const summary = slug(action.summary);
  return summary ? `${tool}:${summary}` : tool;
}

type OrderedActionSource = {
  id?: string;
  actions: { ts: number; tool: string; summary: string; metadata?: Record<string, unknown> }[];
};

function sampleFromActions(source: OrderedActionSource): MovementSample {
  const tokens = [...source.actions]
    .sort((a, b) => a.ts - b.ts)
    .map((action) => canonicalMovementToken(action));
  return source.id ? { id: source.id, tokens } : { tokens };
}

/** Build a dataset from trajectory spans (one sample per span). */
export function movementDatasetFromTrajectories(
  spans: { id?: string; actions: { ts: number; tool: string; summary: string; metadata?: Record<string, unknown> }[] }[],
): MovementDataset {
  return {
    samples: spans
      .map((span) => sampleFromActions(span))
      .filter((sample) => sample.tokens.length > 0),
  };
}

/** Build a dataset from replay manifests (one sample per replay, action events only). */
export function movementDatasetFromReplays(
  replays: { trajectoryIds?: string[]; events: { kind: string; ts: number; tool?: string; summary?: string }[] }[],
): MovementDataset {
  return {
    samples: replays
      .map((replay) => {
        const actions = replay.events
          .filter((event): event is { kind: "action"; ts: number; tool: string; summary: string } =>
            event.kind === "action" && typeof event.tool === "string" && typeof event.summary === "string",
          )
          .map((event) => ({ ts: event.ts, tool: event.tool, summary: event.summary }));
        const id = replay.trajectoryIds?.[0];
        return sampleFromActions(id ? { id, actions } : { actions });
      })
      .filter((sample) => sample.tokens.length > 0),
  };
}

// ---------------------------------------------------------------------------
// Default backend — variable-order Markov model with stupid-backoff.
// ---------------------------------------------------------------------------

export type MarkovMovementBackendOptions = {
  /** Maximum context length (n-gram order minus one). Default 3. */
  order?: number;
  /** Back-off discount applied per dropped context token. Default 0.4. */
  backoffDiscount?: number;
};

class MarkovMovementModel implements MovementModel {
  readonly backend = "markov";

  constructor(
    readonly order: number,
    private readonly backoffDiscount: number,
    /** contextKey -> (token -> count). Includes the empty context (order 0). */
    private readonly transitions: Map<string, Map<MovementToken, number>>,
    private readonly totals: Map<string, number>,
    private readonly vocabulary: MovementToken[],
  ) {}

  private contextKey(context: MovementToken[]): string {
    return context.join(CONTEXT_SEPARATOR);
  }

  /** Longest matching suffix of `context` (length 0..order) that has data. */
  private resolveContext(context: MovementToken[]): { key: string; used: number } | undefined {
    const maxLen = Math.min(this.order, context.length);
    for (let len = maxLen; len >= 0; len -= 1) {
      const suffix = len === 0 ? [] : context.slice(context.length - len);
      const key = this.contextKey(suffix);
      if (this.transitions.has(key)) {
        return { key, used: len };
      }
    }
    return undefined;
  }

  predictNext(context: MovementToken[]): MovementPrediction | undefined {
    const resolved = this.resolveContext(context);
    if (!resolved) {
      return undefined;
    }
    const counts = this.transitions.get(resolved.key);
    const total = this.totals.get(resolved.key);
    if (!counts || !total) {
      return undefined;
    }
    const candidates = [...counts.entries()]
      .map(([token, count]) => ({ token, probability: count / total }))
      // Deterministic ordering: probability desc, then token asc.
      .sort((a, b) => (b.probability - a.probability) || (a.token < b.token ? -1 : a.token > b.token ? 1 : 0));
    const best = candidates[0];
    return {
      token: best.token,
      probability: best.probability,
      contextOrderUsed: resolved.used,
      candidates,
    };
  }

  scoreNext(context: MovementToken[], token: MovementToken): number {
    const maxLen = Math.min(this.order, context.length);
    for (let len = maxLen; len >= 0; len -= 1) {
      const suffix = len === 0 ? [] : context.slice(context.length - len);
      const key = this.contextKey(suffix);
      const counts = this.transitions.get(key);
      const total = this.totals.get(key);
      if (counts && total) {
        const count = counts.get(token);
        if (count !== undefined) {
          // Stupid-backoff: discount once per dropped context token.
          const dropped = maxLen - len;
          return (count / total) * Math.pow(this.backoffDiscount, dropped);
        }
      }
    }
    return 0;
  }

  generate(seed: MovementToken[], length: number): MovementToken[] {
    const out: MovementToken[] = [];
    let context = [...seed];
    for (let i = 0; i < length; i += 1) {
      const prediction = this.predictNext(context);
      if (!prediction) {
        break;
      }
      out.push(prediction.token);
      context = [...context, prediction.token].slice(-this.order);
    }
    return out;
  }

  toJSON(): MovementModelSnapshot {
    const transitions: Record<string, Record<MovementToken, number>> = {};
    for (const [key, counts] of this.transitions.entries()) {
      transitions[key] = Object.fromEntries(counts.entries());
    }
    return {
      backend: this.backend,
      order: this.order,
      transitions,
      vocabulary: [...this.vocabulary],
    };
  }
}

export class MarkovMovementBackend implements MovementModelBackend {
  readonly name = "markov";
  private readonly order: number;
  private readonly backoffDiscount: number;

  constructor(options: MarkovMovementBackendOptions = {}) {
    this.order = Math.max(1, options.order ?? 3);
    this.backoffDiscount = options.backoffDiscount ?? 0.4;
  }

  train(dataset: MovementDataset): MovementModel {
    const transitions = new Map<string, Map<MovementToken, number>>();
    const totals = new Map<string, number>();
    const vocabulary = new Set<MovementToken>();

    const record = (context: MovementToken[], next: MovementToken): void => {
      const key = context.join(CONTEXT_SEPARATOR);
      let counts = transitions.get(key);
      if (!counts) {
        counts = new Map();
        transitions.set(key, counts);
      }
      counts.set(next, (counts.get(next) ?? 0) + 1);
      totals.set(key, (totals.get(key) ?? 0) + 1);
    };

    for (const sample of dataset.samples) {
      const { tokens } = sample;
      for (const token of tokens) {
        vocabulary.add(token);
      }
      for (let i = 0; i < tokens.length; i += 1) {
        const next = tokens[i];
        const maxContext = Math.min(this.order, i);
        for (let len = 0; len <= maxContext; len += 1) {
          const context = len === 0 ? [] : tokens.slice(i - len, i);
          record(context, next);
        }
      }
    }

    return new MarkovMovementModel(
      this.order,
      this.backoffDiscount,
      transitions,
      totals,
      [...vocabulary].sort(),
    );
  }
}

/** Convenience: train the default backend on a dataset. */
export function trainMovementModel(
  dataset: MovementDataset,
  options: MarkovMovementBackendOptions = {},
): MovementModel {
  return new MarkovMovementBackend(options).train(dataset);
}

// ---------------------------------------------------------------------------
// Generalization eval harness — measure replay fidelity on held-out samples.
// ---------------------------------------------------------------------------

export type MovementEvalResult = {
  /** Scored prediction positions (positions with >= minContext preceding tokens). */
  total: number;
  /** Correct next-token predictions. */
  correct: number;
  /** correct / total (0 when total is 0). */
  accuracy: number;
  /** Correct predictions that used the full model order of context. */
  exactContextCorrect: number;
  /** Correct predictions that required backing off to a shorter context. */
  backoffCorrect: number;
  /**
   * Mean probability the model assigned to the *true* next token across scored
   * positions — a soft generalization score robust to ties (0..1).
   */
  meanTrueProbability: number;
};

/**
 * Train on `train`, then teacher-force over every position of each `holdout`
 * sample, comparing the model's predicted next movement to the recorded one.
 * Measures both fidelity (when holdout overlaps seen contexts) and
 * generalization (when it does not, via back-off).
 */
export function evaluateMovementGeneralization(params: {
  backend: MovementModelBackend;
  train: MovementDataset;
  holdout: MovementDataset;
  /** Minimum preceding tokens before a position is scored. Default 1. */
  minContext?: number;
}): MovementEvalResult {
  const model = params.backend.train(params.train);
  const minContext = Math.max(1, params.minContext ?? 1);

  let total = 0;
  let correct = 0;
  let exactContextCorrect = 0;
  let backoffCorrect = 0;
  let probabilitySum = 0;

  for (const sample of params.holdout.samples) {
    const { tokens } = sample;
    for (let i = minContext; i < tokens.length; i += 1) {
      const context = tokens.slice(0, i);
      const expected = tokens[i];
      total += 1;
      probabilitySum += model.scoreNext(context, expected);
      const prediction = model.predictNext(context);
      if (prediction && prediction.token === expected) {
        correct += 1;
        if (prediction.contextOrderUsed >= Math.min(model.order, context.length)) {
          exactContextCorrect += 1;
        } else {
          backoffCorrect += 1;
        }
      }
    }
  }

  return {
    total,
    correct,
    accuracy: total === 0 ? 0 : correct / total,
    exactContextCorrect,
    backoffCorrect,
    meanTrueProbability: total === 0 ? 0 : probabilitySum / total,
  };
}

/** Deterministically split a dataset into train/holdout (every Nth → holdout). */
export function splitMovementDataset(
  dataset: MovementDataset,
  holdoutEvery = 3,
): { train: MovementDataset; holdout: MovementDataset } {
  const train: MovementSample[] = [];
  const holdout: MovementSample[] = [];
  dataset.samples.forEach((sample, index) => {
    if (holdoutEvery > 0 && (index + 1) % holdoutEvery === 0) {
      holdout.push(sample);
    } else {
      train.push(sample);
    }
  });
  return { train: { samples: train }, holdout: { samples: holdout } };
}
