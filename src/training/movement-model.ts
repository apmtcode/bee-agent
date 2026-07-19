/**
 * Pluggable local movement-model backend.
 *
 * This is the in-process training + inference seam for the local-movement
 * learning subsystem (standing objective #2, parts c & d). It lets bee-agent:
 *   (c) train a model on recorded movement trajectories and *replay* them, and
 *   (d) *generalize* to new-but-related movement sequences composed from learned
 *       transitions.
 *
 * The engine runs in Anthropic's cloud with no access to a real machine, so the
 * default backend here is a deterministic, dependency-free order-k Markov model
 * ({@link MarkovMovementBackend}). It requires no GPU, no network, and produces
 * identical results on every run — which is exactly what CI/cloud tests need.
 *
 * The {@link MovementModelBackend} interface is the documented seam for a real
 * on-device small model (e.g. an MLX/GGUF LoRA adapter): implement `train` to
 * shell out to the on-device runtime and return a {@link MovementModel} whose
 * `predictNext`/`generate` call the trained weights. Register it via
 * {@link registerMovementBackend} and select it by id — nothing else changes.
 */

import type { TrajectoryAction, TrajectorySpan } from "../capture/trajectory.js";

/** A single movement token. Opaque string so backends stay tokenizer-agnostic. */
export type MovementToken = string;

/** Emitted at the start of every training sequence; a seed for generation. */
export const MOVEMENT_START_TOKEN: MovementToken = "<start>";
/** Emitted at the end of every training sequence; the natural stop symbol. */
export const MOVEMENT_END_TOKEN: MovementToken = "<end>";

/** One training example: an ordered sequence of movement tokens. */
export type MovementSample = {
  sessionId: string;
  trajectoryId: string;
  tokens: MovementToken[];
};

/** A tokenized dataset ready for a backend to train on. */
export type MovementDataset = {
  samples: MovementSample[];
  vocabulary: MovementToken[];
};

/** A ranked next-token prediction. `probability` sums to ~1 across a context. */
export type MovementPrediction = {
  token: MovementToken;
  probability: number;
  count: number;
};

export type MovementGenerateOptions = {
  /** Hard cap on emitted tokens (excludes the terminating <end>). Default 64. */
  maxSteps?: number;
  /** Stop as soon as this token would be emitted. Default <end>. */
  stopToken?: MovementToken;
};

/** A trained, serializable movement model used for inference/replay. */
export interface MovementModel {
  readonly backendId: string;
  readonly order: number;
  readonly vocabulary: MovementToken[];
  /** Ranked predictions for the token following `context` (most-likely first). */
  predictNext(context: MovementToken[]): MovementPrediction[];
  /** Greedy continuation from `seed` until <end>/stop/maxSteps (seed excluded). */
  generate(seed: MovementToken[], options?: MovementGenerateOptions): MovementToken[];
  /** Deterministic JSON snapshot; rebuild with {@link loadMovementModel}. */
  serialize(): MovementModelSnapshot;
}

export type MovementTrainingOptions = {
  /** Markov context length (how many prior tokens condition the next). */
  order?: number;
};

/** A backend turns a dataset into a {@link MovementModel}. The pluggable seam. */
export interface MovementModelBackend {
  readonly id: string;
  train(dataset: MovementDataset, options?: MovementTrainingOptions): MovementModel;
}

/** Serialized model — plain JSON, safe to persist and reload deterministically. */
export type MovementModelSnapshot = {
  version: 1;
  backendId: string;
  order: number;
  vocabulary: MovementToken[];
  /** context-key -> (nextToken -> count). Keys join tokens with "". */
  transitions: Record<string, Record<MovementToken, number>>;
};

// ---------------------------------------------------------------------------
// Tokenization: TrajectorySpan -> movement tokens
// ---------------------------------------------------------------------------

/**
 * Derive a stable movement token from a recorded action. Prefers the structured
 * gesture kind (device/browser captures stash it in metadata) and falls back to
 * a slug of the tool + summary, so semantically-equal movements collide to the
 * same token — which is what makes generalization across trajectories possible.
 */
export function defaultMovementTokenizer(action: TrajectoryAction): MovementToken {
  const gesture = typeof action.metadata?.gesture === "string" ? action.metadata.gesture : undefined;
  const target =
    typeof action.metadata?.target === "string"
      ? action.metadata.target
      : typeof action.metadata?.direction === "string"
        ? action.metadata.direction
        : undefined;
  const facet = gesture ?? slug(action.summary);
  const tool = slug(action.tool) || "tool";
  return target ? `${tool}:${facet}:${slug(target)}` : `${tool}:${facet}`;
}

/** Tokenize a single trajectory's actions (ts-ordered) into a movement sample. */
export function tokenizeTrajectory(
  trajectory: TrajectorySpan,
  tokenizer: (action: TrajectoryAction) => MovementToken = defaultMovementTokenizer,
): MovementSample {
  const ordered = [...trajectory.actions].sort((a, b) => a.ts - b.ts);
  return {
    sessionId: trajectory.sessionId,
    trajectoryId: trajectory.id,
    tokens: [MOVEMENT_START_TOKEN, ...ordered.map((action) => tokenizer(action)), MOVEMENT_END_TOKEN],
  };
}

/** Build a full dataset from many trajectories, deriving a sorted vocabulary. */
export function buildMovementDataset(
  trajectories: TrajectorySpan[],
  tokenizer: (action: TrajectoryAction) => MovementToken = defaultMovementTokenizer,
): MovementDataset {
  const samples = trajectories
    .map((trajectory) => tokenizeTrajectory(trajectory, tokenizer))
    // A trajectory with no actions is just [<start>, <end>] — nothing to learn.
    .filter((sample) => sample.tokens.length > 2);
  const vocabulary = [...new Set(samples.flatMap((sample) => sample.tokens))].sort();
  return { samples, vocabulary };
}

// ---------------------------------------------------------------------------
// Deterministic Markov backend (the cloud/CI-safe default)
// ---------------------------------------------------------------------------

const CONTEXT_SEPARATOR = "";

function contextKey(context: MovementToken[]): string {
  return context.join(CONTEXT_SEPARATOR);
}

/** Deterministic ranking: by count desc, then token asc for stable tie-breaks. */
function rankTransitions(counts: Record<MovementToken, number>): MovementPrediction[] {
  const total = Object.values(counts).reduce((sum, value) => sum + value, 0);
  return Object.entries(counts)
    .map(([token, count]) => ({ token, count, probability: total === 0 ? 0 : count / total }))
    .sort((a, b) => (b.count !== a.count ? b.count - a.count : a.token < b.token ? -1 : 1));
}

class MarkovMovementModel implements MovementModel {
  constructor(
    readonly backendId: string,
    readonly order: number,
    readonly vocabulary: MovementToken[],
    private readonly transitions: Map<string, Record<MovementToken, number>>,
  ) {}

  predictNext(context: MovementToken[]): MovementPrediction[] {
    // Back off from the full order down to unigram until we find a known context.
    for (let take = Math.min(this.order, context.length); take >= 0; take -= 1) {
      const suffix = take === 0 ? [] : context.slice(context.length - take);
      const counts = this.transitions.get(contextKey(suffix));
      if (counts && Object.keys(counts).length > 0) {
        return rankTransitions(counts);
      }
    }
    return [];
  }

  generate(seed: MovementToken[], options?: MovementGenerateOptions): MovementToken[] {
    const maxSteps = options?.maxSteps ?? 64;
    const stopToken = options?.stopToken ?? MOVEMENT_END_TOKEN;
    const context = seed.length > 0 ? [...seed] : [MOVEMENT_START_TOKEN];
    const produced: MovementToken[] = [];
    for (let step = 0; step < maxSteps; step += 1) {
      const predictions = this.predictNext(context);
      const next = predictions[0]?.token;
      if (next === undefined || next === stopToken) {
        break;
      }
      produced.push(next);
      context.push(next);
    }
    return produced;
  }

  serialize(): MovementModelSnapshot {
    const transitions: Record<string, Record<MovementToken, number>> = {};
    for (const [key, counts] of this.transitions) {
      transitions[key] = { ...counts };
    }
    return {
      version: 1,
      backendId: this.backendId,
      order: this.order,
      vocabulary: [...this.vocabulary],
      transitions,
    };
  }
}

/**
 * Order-k Markov backend. Counts every context->next transition across all
 * training samples (including the shorter prefixes, so it can back off), giving
 * a fully deterministic model. Training on a single trajectory reproduces it
 * exactly under {@link MovementModel.generate}; training on several that share
 * sub-sequences lets `generate` compose novel-but-valid continuations — the
 * generalization property the objective asks for.
 */
export class MarkovMovementBackend implements MovementModelBackend {
  readonly id = "markov";

  constructor(private readonly defaultOrder = 2) {}

  train(dataset: MovementDataset, options?: MovementTrainingOptions): MovementModel {
    const order = Math.max(1, options?.order ?? this.defaultOrder);
    const transitions = new Map<string, Record<MovementToken, number>>();
    const record = (context: MovementToken[], next: MovementToken): void => {
      const key = contextKey(context);
      const counts = transitions.get(key) ?? {};
      counts[next] = (counts[next] ?? 0) + 1;
      transitions.set(key, counts);
    };

    for (const sample of dataset.samples) {
      const tokens = sample.tokens;
      for (let index = 0; index + 1 < tokens.length; index += 1) {
        const next = tokens[index + 1] as MovementToken;
        // Record every context length 0..order ending at `index` so predictNext
        // can back off from the full order to the unigram distribution.
        for (let take = 0; take <= order; take += 1) {
          const start = index + 1 - take;
          if (start < 0) {
            break;
          }
          record(tokens.slice(start, index + 1), next);
        }
      }
    }

    const vocabulary =
      dataset.vocabulary.length > 0
        ? [...dataset.vocabulary]
        : [...new Set(dataset.samples.flatMap((sample) => sample.tokens))].sort();
    return new MarkovMovementModel(this.id, order, vocabulary, transitions);
  }
}

/** Rebuild a model from a snapshot for inference — no retraining needed. */
export function loadMovementModel(snapshot: MovementModelSnapshot): MovementModel {
  const transitions = new Map<string, Record<MovementToken, number>>(
    Object.entries(snapshot.transitions).map(([key, counts]) => [key, { ...counts }]),
  );
  return new MarkovMovementModel(snapshot.backendId, snapshot.order, [...snapshot.vocabulary], transitions);
}

// ---------------------------------------------------------------------------
// Backend registry (the pluggable selection point)
// ---------------------------------------------------------------------------

const backendFactories = new Map<string, () => MovementModelBackend>([
  ["markov", () => new MarkovMovementBackend()],
]);

/** Register a custom backend (e.g. a real on-device model) under an id. */
export function registerMovementBackend(id: string, factory: () => MovementModelBackend): void {
  backendFactories.set(id, factory);
}

/** Ids of all registered backends. */
export function listMovementBackends(): string[] {
  return [...backendFactories.keys()].sort();
}

/** Instantiate a backend by id. Defaults to the deterministic markov backend. */
export function createMovementBackend(id = "markov"): MovementModelBackend {
  const factory = backendFactories.get(id);
  if (!factory) {
    throw new Error(`Unknown movement backend: ${id}. Registered: ${listMovementBackends().join(", ")}`);
  }
  return factory();
}

// ---------------------------------------------------------------------------
// Generalization / replay-fidelity eval harness
// ---------------------------------------------------------------------------

export type MovementEvalResult = {
  sampleCount: number;
  /** Predictions where the model's top-1 next token matched the held-out one. */
  correct: number;
  /** Held-out prediction points that had zero next-token candidates. */
  unpredicted: number;
  /** correct / (predicted decision points). 1 = perfect next-token replay. */
  topOneAccuracy: number;
  /** Fraction of held-out samples whose full continuation replays exactly. */
  exactSequenceMatch: number;
};

/**
 * Measure how well a trained model reproduces held-out movement sequences —
 * the generalization eval harness. For each held-out sample it (a) checks
 * top-1 next-token accuracy at every step given the true prefix, and
 * (b) checks whether a free generation from <start> replays the whole sequence.
 */
export function evaluateMovementModel(model: MovementModel, heldOut: MovementSample[]): MovementEvalResult {
  let correct = 0;
  let decisions = 0;
  let unpredicted = 0;
  let exactMatches = 0;

  for (const sample of heldOut) {
    const tokens = sample.tokens;
    for (let index = 0; index + 1 < tokens.length; index += 1) {
      const expected = tokens[index + 1];
      const predictions = model.predictNext(tokens.slice(0, index + 1));
      if (predictions.length === 0) {
        unpredicted += 1;
        continue;
      }
      decisions += 1;
      if (predictions[0]?.token === expected) {
        correct += 1;
      }
    }

    const seed = tokens[0] === MOVEMENT_START_TOKEN ? [MOVEMENT_START_TOKEN] : [];
    const generated = model.generate(seed, { maxSteps: tokens.length });
    const truth = tokens.filter((token) => token !== MOVEMENT_START_TOKEN && token !== MOVEMENT_END_TOKEN);
    if (generated.length === truth.length && generated.every((token, index) => token === truth[index])) {
      exactMatches += 1;
    }
  }

  return {
    sampleCount: heldOut.length,
    correct,
    unpredicted,
    topOneAccuracy: decisions === 0 ? 0 : correct / decisions,
    exactSequenceMatch: heldOut.length === 0 ? 0 : exactMatches / heldOut.length,
  };
}

function slug(value: string): string {
  return value
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
}
