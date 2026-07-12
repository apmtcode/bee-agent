/**
 * In-process, pluggable local-model training + inference for recorded movements.
 *
 * The `runner.ts` module generates launch scripts for external on-device runtimes
 * (mlx / axolotl). This module complements it with a *deterministic, dependency-free*
 * movement model that trains and infers entirely in-process, so the capture →
 * dataset → train → infer → generalize loop (standing objective #2) can be
 * validated in the cloud on synthetic/replay data without any real OS or GPU.
 *
 * The backend is pluggable via {@link MovementModelBackendRegistry}: the shipped
 * {@link MarkovMovementBackend} (an order-k Markov chain with stupid-backoff, which
 * both *repeats* exact recorded movements and *generalizes* to novel-but-related
 * prefixes) and {@link MemorizingMovementBackend} (an exact-replay baseline) are
 * the reference backends; a real on-device small model can be registered later
 * behind the same interface.
 */

import type { ReplayTimelineEvent } from "../capture/replay.js";

export type MovementToken = string;

export type MovementStepKind = "action" | "observation";

export type MovementStep = {
  /** Stable, comparable token used by the model. */
  token: MovementToken;
  kind: MovementStepKind;
  tool?: string;
  source?: string;
  summary: string;
  ts: number;
};

export type MovementSequence = {
  /** Origin identifier (session or trajectory id). */
  id: string;
  steps: MovementStep[];
};

export type MovementDataset = {
  version: 1;
  sequences: MovementSequence[];
  /** Sorted vocabulary of every token observed across all sequences. */
  vocab: MovementToken[];
};

export type BuildMovementDatasetOptions = {
  /** Include `observation` events as context steps (default: false — actions only). */
  includeObservations?: boolean;
  /** Max characters of the normalized summary folded into a token (default: 48). */
  summaryTokenLength?: number;
};

const DEFAULT_SUMMARY_TOKEN_LENGTH = 48;

function normalizeSummary(summary: string, maxLength: number): string {
  const collapsed = summary.trim().toLowerCase().replace(/\s+/g, " ");
  return collapsed.slice(0, maxLength);
}

/** Deterministically derive a comparable token from a single replay event. */
export function tokenizeReplayEvent(
  event: ReplayTimelineEvent,
  options: BuildMovementDatasetOptions = {},
): MovementStep | undefined {
  const summaryTokenLength = options.summaryTokenLength ?? DEFAULT_SUMMARY_TOKEN_LENGTH;
  if (event.kind === "action") {
    const summary = normalizeSummary(event.summary, summaryTokenLength);
    return {
      token: `action:${event.tool}:${summary}`,
      kind: "action",
      tool: event.tool,
      summary: event.summary,
      ts: event.ts,
    };
  }
  if (event.kind === "observation" && options.includeObservations) {
    const summary = normalizeSummary(event.summary, summaryTokenLength);
    return {
      token: `observation:${event.source}:${summary}`,
      kind: "observation",
      source: event.source,
      summary: event.summary,
      ts: event.ts,
    };
  }
  return undefined;
}

export type ReplaySource = {
  id: string;
  events: ReplayTimelineEvent[];
};

/** Build a movement dataset from replay manifests (or exported replays). */
export function buildMovementDataset(
  replays: ReplaySource[],
  options: BuildMovementDatasetOptions = {},
): MovementDataset {
  const vocab = new Set<MovementToken>();
  const sequences: MovementSequence[] = [];

  for (const replay of replays) {
    const orderedEvents = [...replay.events].sort((a, b) => a.ts - b.ts);
    const steps: MovementStep[] = [];
    for (const event of orderedEvents) {
      const step = tokenizeReplayEvent(event, options);
      if (step) {
        steps.push(step);
        vocab.add(step.token);
      }
    }
    if (steps.length > 0) {
      sequences.push({ id: replay.id, steps });
    }
  }

  return {
    version: 1,
    sequences,
    vocab: [...vocab].sort(),
  };
}

export type MovementPrediction = {
  token: MovementToken;
  probability: number;
  /** Context order (in tokens) that contributed the winning evidence. */
  order: number;
};

export type PredictNextOptions = {
  topK?: number;
};

export type MovementModelSnapshot = {
  version: 1;
  backendId: string;
  order: number;
  vocab: MovementToken[];
  sequenceCount: number;
  /** Per-order context → { nextToken → count }. Index 0 is the unigram table. */
  grams: Array<Array<[string, Array<[MovementToken, number]>]>>;
};

export interface TrainedMovementModel {
  readonly backendId: string;
  readonly order: number;
  readonly vocabSize: number;
  readonly sequenceCount: number;
  /** Ranked next-movement candidates for a context (most recent token last). */
  predictNext(context: MovementToken[], options?: PredictNextOptions): MovementPrediction[];
  /** Greedily roll out `steps` movements from a seed context. Deterministic. */
  generate(seed: MovementToken[], steps: number): MovementToken[];
  snapshot(): MovementModelSnapshot;
}

export type TrainMovementModelOptions = {
  /** Maximum context length in tokens (default: 2). */
  order?: number;
};

export interface MovementModelBackend {
  readonly id: string;
  train(dataset: MovementDataset, options?: TrainMovementModelOptions): Promise<TrainedMovementModel>;
}

const DEFAULT_ORDER = 2;
const BACKOFF = 0.4;
const CONTEXT_SEPARATOR = "";

type GramTable = Map<string, Map<MovementToken, number>>;

function contextKey(context: MovementToken[]): string {
  return context.join(CONTEXT_SEPARATOR);
}

function buildGrams(dataset: MovementDataset, order: number): GramTable[] {
  const grams: GramTable[] = Array.from({ length: order + 1 }, () => new Map());
  for (const sequence of dataset.sequences) {
    const tokens = sequence.steps.map((step) => step.token);
    for (let i = 0; i < tokens.length; i += 1) {
      const next = tokens[i]!;
      for (let n = 0; n <= order; n += 1) {
        if (i - n < 0) {
          break;
        }
        const context = tokens.slice(i - n, i);
        const key = contextKey(context);
        const table = grams[n]!;
        const counts = table.get(key) ?? new Map<MovementToken, number>();
        counts.set(next, (counts.get(next) ?? 0) + 1);
        table.set(key, counts);
      }
    }
  }
  return grams;
}

/**
 * Order-k Markov movement model with stupid-backoff smoothing.
 *
 * Longer matched contexts dominate (exact repetition of recorded movements),
 * while lower orders fill in unseen prefixes (generalization to related
 * movements). Scoring is fully deterministic; ties break by token order.
 */
class MarkovMovementModel implements TrainedMovementModel {
  constructor(
    readonly backendId: string,
    readonly order: number,
    private readonly grams: GramTable[],
    private readonly vocab: MovementToken[],
    readonly sequenceCount: number,
  ) {}

  get vocabSize(): number {
    return this.vocab.length;
  }

  predictNext(context: MovementToken[], options: PredictNextOptions = {}): MovementPrediction[] {
    const scores = new Map<MovementToken, { score: number; order: number }>();

    for (let n = 0; n <= this.order; n += 1) {
      const table = this.grams[n];
      if (!table) {
        continue;
      }
      const slice = n === 0 ? [] : context.slice(-n);
      if (slice.length < n) {
        continue;
      }
      const counts = table.get(contextKey(slice));
      if (!counts) {
        continue;
      }
      let total = 0;
      for (const value of counts.values()) {
        total += value;
      }
      if (total === 0) {
        continue;
      }
      const weight = BACKOFF ** (this.order - n);
      for (const [token, count] of counts) {
        const contribution = weight * (count / total);
        const existing = scores.get(token);
        if (existing) {
          existing.score += contribution;
          existing.order = Math.max(existing.order, n);
        } else {
          scores.set(token, { score: contribution, order: n });
        }
      }
    }

    if (scores.size === 0) {
      return [];
    }

    let normalizer = 0;
    for (const { score } of scores.values()) {
      normalizer += score;
    }

    const ranked = [...scores.entries()]
      .map(([token, { score, order }]) => ({
        token,
        probability: score / normalizer,
        order,
      }))
      .sort((a, b) => {
        if (b.probability !== a.probability) {
          return b.probability - a.probability;
        }
        if (b.order !== a.order) {
          return b.order - a.order;
        }
        return a.token < b.token ? -1 : a.token > b.token ? 1 : 0;
      });

    const topK = options.topK ?? ranked.length;
    return ranked.slice(0, Math.max(0, topK));
  }

  generate(seed: MovementToken[], steps: number): MovementToken[] {
    const context = [...seed];
    const generated: MovementToken[] = [];
    for (let i = 0; i < steps; i += 1) {
      const [prediction] = this.predictNext(context, { topK: 1 });
      if (!prediction) {
        break;
      }
      generated.push(prediction.token);
      context.push(prediction.token);
    }
    return generated;
  }

  snapshot(): MovementModelSnapshot {
    return {
      version: 1,
      backendId: this.backendId,
      order: this.order,
      vocab: [...this.vocab],
      sequenceCount: this.sequenceCount,
      grams: this.grams.map((table) =>
        [...table.entries()]
          .map(([key, counts]) => [key, [...counts.entries()]] as [string, Array<[MovementToken, number]>])
          .sort((a, b) => (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0)),
      ),
    };
  }
}

export class MarkovMovementBackend implements MovementModelBackend {
  readonly id = "markov";

  async train(dataset: MovementDataset, options: TrainMovementModelOptions = {}): Promise<TrainedMovementModel> {
    const order = Math.max(0, options.order ?? DEFAULT_ORDER);
    const grams = buildGrams(dataset, order);
    return new MarkovMovementModel(this.id, order, grams, [...dataset.vocab], dataset.sequences.length);
  }
}

/**
 * Exact-replay baseline: predicts only the token that followed the *full*
 * observed context, with no backoff. Memorizes recorded movements but cannot
 * generalize to unseen prefixes — a useful control for the eval harness.
 */
export class MemorizingMovementBackend implements MovementModelBackend {
  readonly id = "memorizing";

  async train(dataset: MovementDataset, options: TrainMovementModelOptions = {}): Promise<TrainedMovementModel> {
    const order = Math.max(1, options.order ?? DEFAULT_ORDER);
    const full = buildGrams(dataset, order);
    // Keep only the highest-order table so lower-order backoff cannot fire.
    const grams: GramTable[] = Array.from({ length: order + 1 }, (_, n) => (n === order ? full[order]! : new Map()));
    return new MarkovMovementModel(this.id, order, grams, [...dataset.vocab], dataset.sequences.length);
  }
}

/** Rehydrate a model from a serialized snapshot (no retraining required). */
export function restoreMovementModel(snapshot: MovementModelSnapshot): TrainedMovementModel {
  const grams: GramTable[] = snapshot.grams.map((table) => {
    const map: GramTable = new Map();
    for (const [key, counts] of table) {
      map.set(key, new Map(counts));
    }
    return map;
  });
  return new MarkovMovementModel(
    snapshot.backendId,
    snapshot.order,
    grams,
    [...snapshot.vocab],
    snapshot.sequenceCount,
  );
}

export class MovementModelBackendRegistry {
  private readonly backends = new Map<string, MovementModelBackend>();
  private defaultId?: string;

  register(backend: MovementModelBackend, options: { makeDefault?: boolean } = {}): this {
    this.backends.set(backend.id, backend);
    if (options.makeDefault || this.defaultId === undefined) {
      this.defaultId = backend.id;
    }
    return this;
  }

  has(id: string): boolean {
    return this.backends.has(id);
  }

  get(id: string): MovementModelBackend {
    const backend = this.backends.get(id);
    if (!backend) {
      throw new Error(`Unknown movement model backend: ${id}`);
    }
    return backend;
  }

  getDefault(): MovementModelBackend {
    if (this.defaultId === undefined) {
      throw new Error("No movement model backend registered");
    }
    return this.get(this.defaultId);
  }

  list(): string[] {
    return [...this.backends.keys()].sort();
  }
}

/** Registry pre-seeded with the reference in-process backends. */
export function createDefaultMovementModelRegistry(): MovementModelBackendRegistry {
  return new MovementModelBackendRegistry()
    .register(new MarkovMovementBackend(), { makeDefault: true })
    .register(new MemorizingMovementBackend());
}

export type MovementEvalReport = {
  /** Total (context, next-token) prediction points evaluated. */
  samples: number;
  /** Fraction where the model's top prediction matched the actual next token. */
  top1Accuracy: number;
  /** Fraction where the actual next token appeared in the top-K predictions. */
  topKAccuracy: number;
  topK: number;
  /** Samples the model could make no prediction for (empty candidate set). */
  abstained: number;
};

export type EvaluateMovementModelOptions = {
  topK?: number;
  /** Skip the first `minContext` tokens of each sequence (default: 0). */
  minContext?: number;
};

/**
 * Measure next-movement prediction fidelity on held-out sequences. Slides a
 * growing prefix across each sequence and checks whether the model predicts the
 * real next movement, giving a deterministic generalization score.
 */
export function evaluateMovementModel(
  model: TrainedMovementModel,
  heldOut: MovementSequence[],
  options: EvaluateMovementModelOptions = {},
): MovementEvalReport {
  const topK = Math.max(1, options.topK ?? 3);
  const minContext = Math.max(0, options.minContext ?? 0);
  let samples = 0;
  let top1 = 0;
  let inTopK = 0;
  let abstained = 0;

  for (const sequence of heldOut) {
    const tokens = sequence.steps.map((step) => step.token);
    for (let i = Math.max(1, minContext); i < tokens.length; i += 1) {
      const context = tokens.slice(0, i);
      const actual = tokens[i]!;
      const predictions = model.predictNext(context, { topK });
      samples += 1;
      if (predictions.length === 0) {
        abstained += 1;
        continue;
      }
      if (predictions[0]!.token === actual) {
        top1 += 1;
      }
      if (predictions.some((prediction) => prediction.token === actual)) {
        inTopK += 1;
      }
    }
  }

  return {
    samples,
    top1Accuracy: samples === 0 ? 0 : top1 / samples,
    topKAccuracy: samples === 0 ? 0 : inTopK / samples,
    topK,
    abstained,
  };
}
