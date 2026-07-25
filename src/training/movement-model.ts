/**
 * Pluggable local-movement model backend + a deterministic mock backend.
 *
 * Standing objective #2 (c/d): post-train a *local* model on a reviewed
 * movement dataset so it can (c) repeat the recorded movements and (d)
 * generalize to new-but-related movements. Real on-device training runs
 * mlx/axolotl (see `runner.ts`); that path cannot execute in the cloud, so
 * this module defines the backend *interface* and ships a deterministic
 * in-process backend (an order-k Markov model with stupid-backoff) that
 * trains and infers with zero native deps. Tests exercise the full
 * dataset -> train -> repeat -> generalize loop on synthetic event streams.
 *
 * The backend is a seam: swap `MarkovMovementBackend` for an MLX/GGUF-backed
 * implementation of `MovementModelBackend` and the rest of the pipeline is
 * unchanged.
 */

import type { ExportedReplayManifest } from "./export-manifest.js";
import type { ReplayManifest, ReplayTimelineEvent } from "../capture/replay.js";

/** A single canonical movement token, e.g. `"device:tap:submit-button"`. */
export type MovementToken = string;

/** Sentinel tokens. START pads the front of every sequence; END marks stop. */
export const MOVEMENT_START_TOKEN: MovementToken = "<s>";
export const MOVEMENT_END_TOKEN: MovementToken = "</s>";

const CONTEXT_SEPARATOR = "";
const DEFAULT_ORDER = 2;
const MAX_ALTERNATIVES = 4;
const DEFAULT_MAX_GENERATION_STEPS = 64;

/** One training example: an ordered sequence of movement tokens. */
export type MovementExample = {
  trajectoryId: string;
  tokens: MovementToken[];
};

/** A backend-agnostic, replayable movement dataset. */
export type MovementDataset = {
  version: 1;
  examples: MovementExample[];
  vocabulary: MovementToken[];
};

export type MovementPredictionAlternative = {
  token: MovementToken;
  probability: number;
};

export type MovementPrediction = {
  token: MovementToken;
  probability: number;
  /** Backoff order actually used for this prediction (order..0). */
  order: number;
  alternatives: MovementPredictionAlternative[];
};

export type MovementTrainOptions = {
  /** Markov order (context window). Defaults to 2. */
  order?: number;
};

export type MovementGenerateOptions = {
  maxSteps?: number;
};

/** Serialized form so a trained model can be persisted and reloaded. */
export type SerializedMovementModel = {
  version: 1;
  backendId: string;
  order: number;
  vocabulary: MovementToken[];
  transitions: Array<{
    context: MovementToken[];
    entries: Array<{ token: MovementToken; count: number }>;
  }>;
};

/** A model that predicts and generates movement continuations. */
export interface TrainedMovementModel {
  readonly backendId: string;
  readonly order: number;
  /** Most-likely next token given a context, or undefined if the model is empty. */
  predictNext(context: MovementToken[]): MovementPrediction | undefined;
  /** Greedily continue `prompt` until END or `maxSteps`. Deterministic. */
  generate(prompt: MovementToken[], options?: MovementGenerateOptions): MovementToken[];
  serialize(): SerializedMovementModel;
}

/** Pluggable training backend. Swap the mock for a real on-device runtime. */
export interface MovementModelBackend {
  readonly id: string;
  train(dataset: MovementDataset, options?: MovementTrainOptions): Promise<TrainedMovementModel>;
}

/**
 * Turn a movement action summary into a stable, low-cardinality token slug so
 * that semantically-equal movements collapse to the same token (better
 * generalization) while distinct movements stay distinct.
 */
export function slugifyMovement(value: string): string {
  const slug = value
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
  return slug.length > 0 ? slug : "movement";
}

/** Canonical token for a replay action event. */
export function movementTokenForAction(tool: string, summary: string): MovementToken {
  return `${slugifyMovement(tool)}:${slugifyMovement(summary)}`;
}

type ReplayLike = Pick<ReplayManifest, "trajectoryIds" | "events"> | ExportedReplayManifest;

function isActionEvent(
  event: ReplayTimelineEvent | ExportedReplayManifest["events"][number],
): event is Extract<ReplayTimelineEvent, { kind: "action" }> {
  return event.kind === "action";
}

/**
 * Build a training dataset from reviewed replay manifests. Each manifest's
 * ordered `action` events become one example's token sequence. Manifests with
 * fewer than `minTokens` actions are dropped (nothing to learn from).
 */
export function buildMovementDataset(
  replays: ReplayLike[],
  options: { minTokens?: number } = {},
): MovementDataset {
  const minTokens = Math.max(1, options.minTokens ?? 1);
  const vocabulary = new Set<MovementToken>();
  const examples: MovementExample[] = [];

  for (const replay of replays) {
    const tokens: MovementToken[] = [];
    for (const event of replay.events) {
      if (!isActionEvent(event)) {
        continue;
      }
      const token = movementTokenForAction(event.tool, event.summary);
      tokens.push(token);
      vocabulary.add(token);
    }
    if (tokens.length < minTokens) {
      continue;
    }
    examples.push({
      trajectoryId: replay.trajectoryIds[0] ?? `replay-${examples.length}`,
      tokens,
    });
  }

  return {
    version: 1,
    examples,
    vocabulary: [...vocabulary].sort(),
  };
}

function encodeContext(context: MovementToken[]): string {
  return context.join(CONTEXT_SEPARATOR);
}

function padFront(context: MovementToken[], order: number): MovementToken[] {
  if (context.length >= order) {
    return context.slice(context.length - order);
  }
  const padding = new Array<MovementToken>(order - context.length).fill(MOVEMENT_START_TOKEN);
  return [...padding, ...context];
}

class MarkovMovementModel implements TrainedMovementModel {
  constructor(
    readonly backendId: string,
    readonly order: number,
    private readonly vocabulary: MovementToken[],
    private readonly transitions: Map<string, Map<MovementToken, number>>,
  ) {}

  predictNext(context: MovementToken[]): MovementPrediction | undefined {
    const padded = padFront(context, this.order);
    for (let take = this.order; take >= 0; take -= 1) {
      const suffix = padded.slice(padded.length - take);
      const counts = this.transitions.get(encodeContext(suffix));
      if (!counts || counts.size === 0) {
        continue;
      }
      const ranked = rankCounts(counts);
      const total = ranked.reduce((sum, entry) => sum + entry.count, 0);
      const best = ranked[0];
      if (!best || total === 0) {
        continue;
      }
      return {
        token: best.token,
        probability: best.count / total,
        order: take,
        alternatives: ranked.slice(0, MAX_ALTERNATIVES).map((entry) => ({
          token: entry.token,
          probability: entry.count / total,
        })),
      };
    }
    return undefined;
  }

  generate(prompt: MovementToken[], options: MovementGenerateOptions = {}): MovementToken[] {
    const maxSteps = Math.max(0, options.maxSteps ?? DEFAULT_MAX_GENERATION_STEPS);
    const context = [...prompt];
    const output: MovementToken[] = [];
    for (let step = 0; step < maxSteps; step += 1) {
      const prediction = this.predictNext(context);
      if (!prediction || prediction.token === MOVEMENT_END_TOKEN) {
        break;
      }
      output.push(prediction.token);
      context.push(prediction.token);
    }
    return output;
  }

  serialize(): SerializedMovementModel {
    const transitions = [...this.transitions.entries()]
      .map(([context, counts]) => ({
        context: context.length === 0 ? [] : context.split(CONTEXT_SEPARATOR),
        entries: rankCounts(counts).map((entry) => ({ token: entry.token, count: entry.count })),
      }))
      .sort((a, b) => encodeContext(a.context).localeCompare(encodeContext(b.context)));
    return {
      version: 1,
      backendId: this.backendId,
      order: this.order,
      vocabulary: [...this.vocabulary],
      transitions,
    };
  }
}

/** Deterministic tie-break: higher count first, then lexicographic token. */
function rankCounts(counts: Map<MovementToken, number>): Array<{ token: MovementToken; count: number }> {
  return [...counts.entries()]
    .map(([token, count]) => ({ token, count }))
    .sort((a, b) => (b.count !== a.count ? b.count - a.count : a.token.localeCompare(b.token)));
}

/**
 * Deterministic in-process training backend. An order-k Markov model with
 * stupid-backoff: it memorizes recorded transitions (so it replays recorded
 * movements exactly) and falls back to shorter contexts for unseen prefixes
 * (so it generalizes to new-but-related movements). No native deps — runs in
 * the cloud/CI and stands in for the real on-device runtime.
 */
export class MarkovMovementBackend implements MovementModelBackend {
  readonly id = "markov-mock";

  async train(dataset: MovementDataset, options: MovementTrainOptions = {}): Promise<TrainedMovementModel> {
    const order = Math.max(0, options.order ?? DEFAULT_ORDER);
    const transitions = new Map<string, Map<MovementToken, number>>();

    const record = (context: MovementToken[], next: MovementToken): void => {
      const key = encodeContext(context);
      let counts = transitions.get(key);
      if (!counts) {
        counts = new Map<MovementToken, number>();
        transitions.set(key, counts);
      }
      counts.set(next, (counts.get(next) ?? 0) + 1);
    };

    for (const example of dataset.examples) {
      const padding = new Array<MovementToken>(order).fill(MOVEMENT_START_TOKEN);
      const sequence = [...padding, ...example.tokens, MOVEMENT_END_TOKEN];
      for (let i = order; i < sequence.length; i += 1) {
        const next = sequence[i]!;
        for (let take = 0; take <= order; take += 1) {
          const context = sequence.slice(i - take, i);
          record(context, next);
        }
      }
    }

    return new MarkovMovementModel(this.id, order, [...dataset.vocabulary].sort(), transitions);
  }
}

/** Rehydrate a persisted model without retraining (mirrors loading a GGUF). */
export function deserializeMovementModel(serialized: SerializedMovementModel): TrainedMovementModel {
  const transitions = new Map<string, Map<MovementToken, number>>();
  for (const entry of serialized.transitions) {
    const counts = new Map<MovementToken, number>();
    for (const { token, count } of entry.entries) {
      counts.set(token, count);
    }
    transitions.set(encodeContext(entry.context), counts);
  }
  return new MarkovMovementModel(
    serialized.backendId,
    serialized.order,
    [...serialized.vocabulary],
    transitions,
  );
}

/**
 * Deterministic synthetic movement-event generator for cloud validation. Given
 * a set of named "workflows" (each an ordered token template), it emits
 * repeated, lightly-perturbed examples so tests can validate the
 * capture -> dataset -> train -> replay/generalize round-trip without any real
 * OS input. Seeded LCG -> byte-for-byte reproducible.
 */
export function generateSyntheticMovementDataset(params: {
  workflows: Array<{ id: string; tokens: MovementToken[] }>;
  repetitions: number;
  seed?: number;
  /** Probability [0,1) of dropping an optional trailing token, for variation. */
  dropTailProbability?: number;
}): MovementDataset {
  const repetitions = Math.max(1, Math.floor(params.repetitions));
  const dropTail = Math.min(0.9, Math.max(0, params.dropTailProbability ?? 0));
  let state = (params.seed ?? 1) >>> 0 || 1;
  const nextRandom = (): number => {
    // Numerical Recipes LCG — deterministic, no Math.random dependency.
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 0xffffffff;
  };

  const vocabulary = new Set<MovementToken>();
  const examples: MovementExample[] = [];
  for (let rep = 0; rep < repetitions; rep += 1) {
    for (const workflow of params.workflows) {
      const tokens = [...workflow.tokens];
      if (dropTail > 0 && tokens.length > 1 && nextRandom() < dropTail) {
        tokens.pop();
      }
      for (const token of tokens) {
        vocabulary.add(token);
      }
      examples.push({ trajectoryId: `${workflow.id}-${rep}`, tokens });
    }
  }

  return {
    version: 1,
    examples,
    vocabulary: [...vocabulary].sort(),
  };
}
