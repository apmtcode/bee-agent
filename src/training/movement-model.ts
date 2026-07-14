import type { ExportedReplayManifest, ReviewedExportManifest } from "./export-manifest.js";

/**
 * Local-movement learning backend.
 *
 * bee-agent's training subsystem (see {@link ./runner.ts}) produces *plans* that
 * launch real on-device trainers (MLX/axolotl) on the user's machine. Those
 * cannot run in the cloud/CI. This module provides the complementary piece: a
 * pluggable, in-process backend seam that can actually train a small local model
 * on a recorded movement dataset, predict/generalize continuations, and be
 * evaluated for replay fidelity — deterministically, with no OS access or heavy
 * dependencies.
 *
 * The default {@link MarkovMovementBackend} is an n-gram model with
 * "stupid-backoff" smoothing. It genuinely learns the transition statistics of
 * recorded movements and generalizes to unseen-but-related prefixes, which makes
 * it a faithful stand-in for objective #2(c)/(d) that CI can exercise. A real
 * on-device small model registers against the same {@link LocalMovementBackend}
 * interface via {@link MovementBackendRegistry}.
 */

/** Boundary tokens injected around each recorded movement sequence. */
export const MOVEMENT_START_TOKEN = "START";
export const MOVEMENT_END_TOKEN = "END";

/** A single recorded movement sequence, tokenized into ordered primitives. */
export type MovementSample = {
  id: string;
  /** Ordered movement tokens, e.g. tool names or action signatures. */
  tokens: string[];
};

/** A dataset of recorded movement sequences ready for training. */
export type MovementDataset = {
  samples: MovementSample[];
};

/** Configuration for a movement-model training run. */
export type MovementModelConfig = {
  /** Context length (n-gram order). Must be >= 1. Defaults to 2. */
  order?: number;
  /** Additive (Laplace) smoothing weight applied over the vocabulary. */
  smoothing?: number;
  /** Optional ISO timestamp stamped into the trained model (kept out of core logic for determinism). */
  trainedAt?: string;
};

/** A trained, fully-serializable movement model. */
export type TrainedMovementModel = {
  version: 1;
  backend: string;
  order: number;
  smoothing: number;
  /** Observed emittable tokens (includes END, excludes START). */
  vocab: string[];
  /** context-key -> next-token -> count. */
  transitions: Record<string, Record<string, number>>;
  sampleCount: number;
  tokenCount: number;
  trainedAt?: string;
};

/** A ranked next-movement prediction. */
export type MovementPrediction = {
  token: string;
  probability: number;
  /** How many context tokens were actually used (backoff depth); `order` means no backoff. */
  contextUsed: number;
};

/** Result of evaluating replay fidelity on held-out movement samples. */
export type MovementEvalResult = {
  sampleCount: number;
  predictions: number;
  correct: number;
  /** Top-1 next-token accuracy over every position in every held-out sample. */
  accuracy: number;
  /** Geometric-mean per-token perplexity (lower is better). */
  perplexity: number;
};

/**
 * Pluggable local-movement backend. Training is async (real backends spawn
 * work); inference is synchronous over a loaded model.
 */
export type LocalMovementBackend = {
  readonly name: string;
  train(dataset: MovementDataset, config?: MovementModelConfig): Promise<TrainedMovementModel>;
  /** Ranked candidates for the movement that follows `context`. */
  predictNext(model: TrainedMovementModel, context: string[]): MovementPrediction[];
  /** Roll out a full movement sequence from an optional seed prefix. */
  generate(model: TrainedMovementModel, seed?: string[], maxLength?: number): string[];
  /** Measure how faithfully the model reproduces held-out related movements. */
  evaluate(model: TrainedMovementModel, heldOut: MovementDataset): MovementEvalResult;
};

const DEFAULT_ORDER = 2;
const DEFAULT_SMOOTHING = 0.01;
const DEFAULT_MAX_GENERATION = 256;

function contextKey(tokens: string[]): string {
  return tokens.join("");
}

/**
 * Deterministic n-gram movement backend with stupid-backoff.
 *
 * Learns P(next | last-N-movements) from the recorded dataset. On an unseen
 * context it backs off to progressively shorter histories (down to the unigram
 * distribution), which is what lets it generalize to new-but-related movement
 * prefixes rather than only replaying memorized sequences.
 */
export class MarkovMovementBackend implements LocalMovementBackend {
  readonly name = "markov";

  async train(dataset: MovementDataset, config: MovementModelConfig = {}): Promise<TrainedMovementModel> {
    const order = Math.max(1, Math.floor(config.order ?? DEFAULT_ORDER));
    const smoothing = config.smoothing ?? DEFAULT_SMOOTHING;
    const transitions: Record<string, Record<string, number>> = {};
    const vocab = new Set<string>();
    let tokenCount = 0;

    for (const sample of dataset.samples) {
      const padded = [
        ...Array<string>(order).fill(MOVEMENT_START_TOKEN),
        ...sample.tokens,
        MOVEMENT_END_TOKEN,
      ];
      for (const token of sample.tokens) {
        vocab.add(token);
      }
      vocab.add(MOVEMENT_END_TOKEN);
      tokenCount += sample.tokens.length;

      // Record transitions for every backoff depth (1..order) so inference can
      // consult a shorter history when the full context was never observed.
      for (let position = order; position < padded.length; position += 1) {
        const next = padded[position]!;
        for (let depth = order; depth >= 1; depth -= 1) {
          const history = padded.slice(position - depth, position);
          const key = contextKey(history);
          const bucket = (transitions[key] ??= {});
          bucket[next] = (bucket[next] ?? 0) + 1;
        }
      }
    }

    return {
      version: 1,
      backend: this.name,
      order,
      smoothing,
      vocab: [...vocab].sort(),
      transitions,
      sampleCount: dataset.samples.length,
      tokenCount,
      trainedAt: config.trainedAt,
    };
  }

  predictNext(model: TrainedMovementModel, context: string[]): MovementPrediction[] {
    const padded = [...Array<string>(model.order).fill(MOVEMENT_START_TOKEN), ...context];
    // Stupid-backoff: try the longest available history, shrinking until a
    // context we have counts for is found.
    for (let depth = model.order; depth >= 1; depth -= 1) {
      const history = padded.slice(padded.length - depth);
      const bucket = model.transitions[contextKey(history)];
      if (bucket && Object.keys(bucket).length > 0) {
        return rankBucket(bucket, model, depth);
      }
    }
    // No history at all — fall back to a smoothed uniform over the vocabulary.
    return rankBucket({}, model, 0);
  }

  generate(model: TrainedMovementModel, seed: string[] = [], maxLength = DEFAULT_MAX_GENERATION): string[] {
    const generated: string[] = [];
    const context = [...seed];
    for (let step = 0; step < maxLength; step += 1) {
      const [best] = this.predictNext(model, context);
      if (!best || best.token === MOVEMENT_END_TOKEN) {
        break;
      }
      generated.push(best.token);
      context.push(best.token);
    }
    return generated;
  }

  evaluate(model: TrainedMovementModel, heldOut: MovementDataset): MovementEvalResult {
    let predictions = 0;
    let correct = 0;
    let logProbSum = 0;

    for (const sample of heldOut.samples) {
      const targets = [...sample.tokens, MOVEMENT_END_TOKEN];
      const context: string[] = [];
      for (const expected of targets) {
        const ranked = this.predictNext(model, context);
        const top = ranked[0];
        if (top && top.token === expected) {
          correct += 1;
        }
        const match = ranked.find((prediction) => prediction.token === expected);
        const probability = match?.probability ?? smallestProbability(model);
        logProbSum += Math.log(probability);
        predictions += 1;
        if (expected !== MOVEMENT_END_TOKEN) {
          context.push(expected);
        }
      }
    }

    return {
      sampleCount: heldOut.samples.length,
      predictions,
      correct,
      accuracy: predictions === 0 ? 0 : correct / predictions,
      perplexity: predictions === 0 ? 0 : Math.exp(-logProbSum / predictions),
    };
  }
}

function rankBucket(
  bucket: Record<string, number>,
  model: TrainedMovementModel,
  contextUsed: number,
): MovementPrediction[] {
  const smoothing = model.smoothing;
  const candidates = new Set<string>([...model.vocab, ...Object.keys(bucket)]);
  const total = Object.values(bucket).reduce((sum, count) => sum + count, 0);
  const denominator = total + smoothing * candidates.size;
  const predictions: MovementPrediction[] = [...candidates].map((token) => ({
    token,
    probability: denominator === 0 ? 0 : ((bucket[token] ?? 0) + smoothing) / denominator,
    contextUsed,
  }));
  // Deterministic ordering: probability desc, then token asc to break ties.
  predictions.sort((a, b) => (b.probability - a.probability) || (a.token < b.token ? -1 : a.token > b.token ? 1 : 0));
  return predictions;
}

function smallestProbability(model: TrainedMovementModel): number {
  const size = Math.max(1, model.vocab.length);
  const smoothing = model.smoothing > 0 ? model.smoothing : 1e-9;
  return smoothing / (smoothing * size);
}

/**
 * Pluggable registry of movement backends. The deterministic `markov` backend is
 * registered by default; a real on-device small-model backend registers here
 * under its own name so call sites can select it without code changes.
 */
export class MovementBackendRegistry {
  private readonly factories = new Map<string, () => LocalMovementBackend>();

  constructor(registerDefaults = true) {
    if (registerDefaults) {
      this.register("markov", () => new MarkovMovementBackend());
    }
  }

  register(name: string, factory: () => LocalMovementBackend): void {
    this.factories.set(name, factory);
  }

  has(name: string): boolean {
    return this.factories.has(name);
  }

  names(): string[] {
    return [...this.factories.keys()].sort();
  }

  create(name: string): LocalMovementBackend {
    const factory = this.factories.get(name);
    if (!factory) {
      throw new Error(`Unknown movement backend "${name}". Registered: ${this.names().join(", ") || "(none)"}`);
    }
    return factory();
  }
}

/** Shared default registry with the deterministic `markov` backend preloaded. */
export const defaultMovementBackendRegistry = new MovementBackendRegistry();

/** How a movement action event is turned into a training token. Defaults to the tool name. */
export type MovementTokenizer = (action: { tool: string; summary: string }) => string;

const defaultTokenizer: MovementTokenizer = (action) => action.tool;

/**
 * Build a training dataset from the `action` events of a set of replay
 * manifests (the movement records exported by {@link ./exporter.ts}). Each
 * replay becomes one movement sample; empty samples are dropped.
 */
export function buildMovementDatasetFromReplays(
  replays: readonly ExportedReplayManifest[],
  tokenize: MovementTokenizer = defaultTokenizer,
): MovementDataset {
  const samples: MovementSample[] = [];
  for (const replay of replays) {
    const tokens = replay.events
      .filter((event): event is Extract<ExportedReplayManifest["events"][number], { kind: "action" }> =>
        event.kind === "action",
      )
      .map((event) => tokenize({ tool: event.tool, summary: event.summary }));
    if (tokens.length > 0) {
      samples.push({ id: replay.sessionId, tokens });
    }
  }
  return { samples };
}

/** Convenience: build a movement dataset directly from a reviewed export manifest. */
export function buildMovementDatasetFromExport(
  manifest: ReviewedExportManifest,
  tokenize: MovementTokenizer = defaultTokenizer,
): MovementDataset {
  return buildMovementDatasetFromReplays(manifest.replays, tokenize);
}

/** Validate + narrow a parsed object into a {@link TrainedMovementModel}. */
export function deserializeMovementModel(raw: unknown): TrainedMovementModel {
  if (!raw || typeof raw !== "object") {
    throw new Error("Invalid movement model: not an object");
  }
  const model = raw as Partial<TrainedMovementModel>;
  if (model.version !== 1 || typeof model.order !== "number" || !model.transitions || !Array.isArray(model.vocab)) {
    throw new Error("Invalid movement model: missing required fields");
  }
  return model as TrainedMovementModel;
}
