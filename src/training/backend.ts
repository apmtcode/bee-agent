/**
 * Pluggable local-model backend for the movement-learning subsystem.
 *
 * The recording pipeline (`src/capture/*`) and the on-device training launcher
 * (`src/training/runner.ts`) produce a *reviewed* dataset of replayable
 * movement events and a launch plan for a real small model on Apple Silicon.
 * That real training runs only when the user runs bee-agent locally — it cannot
 * run in the cloud where this engine evolves the code.
 *
 * This module supplies the *backend seam* that the rest of bee-agent programs
 * against: an in-process interface that can {@link MovementModelBackend.train}
 * on a movement dataset and {@link MovementModelBackend.predictNext} / generate
 * new movement sequences. The shipped {@link MarkovMovementBackend} is a
 * deterministic, dependency-free reference implementation so the train → infer
 * loop is fully exercisable (and testable) in CI/cloud. A real on-device model
 * (mlx/axolotl) can implement the same interface and be swapped in without the
 * caller knowing the difference.
 *
 * Generalization (objective 2d): the Markov backend uses stupid-backoff over
 * progressively shorter contexts, so a movement prefix never seen verbatim
 * still yields a sensible next-movement prediction from its longest known
 * suffix — i.e. it performs *new but related* movements rather than only
 * verbatim replay.
 */
import type { ReplayManifest, ReplayTimelineEvent } from "../capture/replay.js";

/** A single learnable unit of movement, derived from a replay action event. */
export type MovementToken = string;

/** Sentinel tokens framing each sequence so the model can learn starts/ends. */
export const MOVEMENT_START_TOKEN = "start";
export const MOVEMENT_END_TOKEN = "end";

const CONTEXT_SEPARATOR = "";

/** An ordered movement sequence for one trajectory. */
export type MovementSequence = {
  trajectoryId: string;
  tokens: MovementToken[];
};

export type MovementDataset = {
  sequences: MovementSequence[];
};

/**
 * Canonical token for an action event. Tools are kept verbatim (they are the
 * primary signal); summaries are slugged so superficially different phrasings
 * of the same gesture collapse to the same token, which is what lets the model
 * generalize across recordings.
 */
export function actionToToken(tool: string, summary: string): MovementToken {
  const slug = summary
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 48);
  return slug ? `${tool}:${slug}` : tool;
}

/**
 * Build a movement dataset from replay manifests. Only `action` timeline events
 * participate (observations/transcript are context, not movements); they retain
 * the manifest's chronological order and are grouped per trajectory.
 */
export function buildMovementDataset(manifests: ReplayManifest[]): MovementDataset {
  const byTrajectory = new Map<string, MovementToken[]>();
  for (const manifest of manifests) {
    for (const event of manifest.events) {
      if (!isActionEvent(event)) {
        continue;
      }
      const tokens = byTrajectory.get(event.trajectoryId) ?? [];
      tokens.push(actionToToken(event.tool, event.summary));
      byTrajectory.set(event.trajectoryId, tokens);
    }
  }
  return {
    sequences: [...byTrajectory.entries()]
      .map(([trajectoryId, tokens]) => ({ trajectoryId, tokens }))
      .sort((a, b) => (a.trajectoryId < b.trajectoryId ? -1 : a.trajectoryId > b.trajectoryId ? 1 : 0)),
  };
}

function isActionEvent(event: ReplayTimelineEvent): event is Extract<ReplayTimelineEvent, { kind: "action" }> {
  return event.kind === "action";
}

/** A serializable trained movement model (JSON-safe; no closures). */
export type TrainedMovementModel = {
  version: 1;
  backend: string;
  order: number;
  vocabulary: MovementToken[];
  /** context (tokens joined by CONTEXT_SEPARATOR) → next-token → count. */
  transitions: Record<string, Record<MovementToken, number>>;
  /** global next-token frequency, used as the final backoff. */
  unigram: Record<MovementToken, number>;
  trainedSequences: number;
  trainedTokens: number;
};

export type MovementPrediction = {
  token: MovementToken;
  /** P(token | matched context) at the context length actually used. */
  probability: number;
  /** Context length used after backoff: `order` = exact match, 0 = unigram. */
  matchedOrder: number;
};

export type MovementBackendTrainOptions = {
  /** Max context length (n-gram order minus one). Defaults to 2. */
  order?: number;
};

export type MovementGenerateOptions = {
  maxLength?: number;
  /** Stop as soon as the end sentinel is produced (default true). */
  stopAtEnd?: boolean;
};

/**
 * The backend seam. A real on-device model implements the same three methods;
 * callers depend only on this interface, never on a concrete backend.
 */
export interface MovementModelBackend {
  readonly id: string;
  train(dataset: MovementDataset, options?: MovementBackendTrainOptions): Promise<TrainedMovementModel>;
  predictNext(model: TrainedMovementModel, context: MovementToken[]): MovementPrediction | undefined;
  generate(model: TrainedMovementModel, seed: MovementToken[], options?: MovementGenerateOptions): MovementToken[];
}

/**
 * Deterministic n-gram backend with stupid-backoff. Pure and dependency-free so
 * it is identical in cloud and on device, which makes it the reference backend
 * for tests and a safe default when no local model is available.
 */
export class MarkovMovementBackend implements MovementModelBackend {
  readonly id = "markov-backoff";

  async train(dataset: MovementDataset, options: MovementBackendTrainOptions = {}): Promise<TrainedMovementModel> {
    const order = Math.max(1, Math.floor(options.order ?? 2));
    const transitions: Record<string, Record<MovementToken, number>> = {};
    const unigram: Record<MovementToken, number> = {};
    const vocabulary = new Set<MovementToken>();
    let trainedTokens = 0;

    for (const sequence of dataset.sequences) {
      const framed = [MOVEMENT_START_TOKEN, ...sequence.tokens, MOVEMENT_END_TOKEN];
      for (let i = 1; i < framed.length; i += 1) {
        const next = framed[i] as MovementToken;
        vocabulary.add(next);
        unigram[next] = (unigram[next] ?? 0) + 1;
        trainedTokens += 1;
        for (let k = 1; k <= order; k += 1) {
          if (i - k < 0) {
            break;
          }
          const context = framed.slice(i - k, i);
          const key = contextKey(context);
          const bucket = (transitions[key] ??= {});
          bucket[next] = (bucket[next] ?? 0) + 1;
        }
      }
    }

    return {
      version: 1,
      backend: this.id,
      order,
      vocabulary: [...vocabulary].sort(),
      transitions,
      unigram,
      trainedSequences: dataset.sequences.length,
      trainedTokens,
    };
  }

  predictNext(model: TrainedMovementModel, context: MovementToken[]): MovementPrediction | undefined {
    const maxK = Math.min(model.order, context.length);
    for (let k = maxK; k >= 1; k -= 1) {
      const key = contextKey(context.slice(context.length - k));
      const bucket = model.transitions[key];
      const prediction = argmax(bucket);
      if (prediction) {
        return { ...prediction, matchedOrder: k };
      }
    }
    const unigramPrediction = argmax(model.unigram);
    return unigramPrediction ? { ...unigramPrediction, matchedOrder: 0 } : undefined;
  }

  generate(
    model: TrainedMovementModel,
    seed: MovementToken[],
    options: MovementGenerateOptions = {},
  ): MovementToken[] {
    const maxLength = Math.max(0, options.maxLength ?? 64);
    const stopAtEnd = options.stopAtEnd ?? true;
    const context: MovementToken[] = seed.length > 0 ? [...seed] : [MOVEMENT_START_TOKEN];
    const generated: MovementToken[] = [];
    while (generated.length < maxLength) {
      const prediction = this.predictNext(model, context);
      if (!prediction || prediction.token === MOVEMENT_END_TOKEN) {
        if (stopAtEnd) {
          break;
        }
        if (!prediction) {
          break;
        }
      }
      if (prediction.token === MOVEMENT_START_TOKEN) {
        // never emit the start sentinel mid-stream; advance context and retry
        context.push(prediction.token);
        continue;
      }
      generated.push(prediction.token);
      context.push(prediction.token);
    }
    return generated;
  }
}

function contextKey(context: MovementToken[]): string {
  return context.join(CONTEXT_SEPARATOR);
}

/** Deterministic argmax: highest count, ties broken by token name ascending. */
function argmax(
  counts: Record<MovementToken, number> | undefined,
): { token: MovementToken; probability: number } | undefined {
  if (!counts) {
    return undefined;
  }
  let best: MovementToken | undefined;
  let bestCount = -1;
  let total = 0;
  for (const [token, count] of Object.entries(counts)) {
    total += count;
    if (count > bestCount || (count === bestCount && best !== undefined && token < best)) {
      best = token;
      bestCount = count;
    }
  }
  if (best === undefined || total === 0) {
    return undefined;
  }
  return { token: best, probability: bestCount / total };
}
