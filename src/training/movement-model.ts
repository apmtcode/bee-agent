import type { ReplayManifest, ReplayTimelineEvent } from "../capture/replay.js";
import type { TrajectoryAction, TrajectoryObservation, TrajectorySpan } from "../capture/trajectory.js";

/**
 * Local-movement learning: in-process, pluggable model backend.
 *
 * The training runner (`runner.ts`) only *plans* a shell-out to an on-device
 * trainer (mlx / axolotl) that cannot run in the cloud. This module provides a
 * real, in-process training + inference pipeline that operates on movement
 * sequences derived from captured trajectories, so the capture -> dataset ->
 * train -> infer loop can be exercised and tested without any OS access.
 *
 * The backend is pluggable: `LocalMovementModelBackend` is the seam. The
 * default `NgramMovementModelBackend` is a deterministic variable-order Markov
 * model with backoff — it *repeats* recorded movements (exact-context match)
 * and *generalizes* to new-but-related movements (shorter-context backoff). A
 * real on-device small model can implement the same interface later.
 */

/** A canonical, discrete movement token derived from one timeline event. */
export type MovementToken = string;

/** Marks the start of a sequence so the first real token can be predicted. */
export const MOVEMENT_START_TOKEN: MovementToken = "<s>";

export type MovementSequence = {
  id: string;
  tokens: MovementToken[];
};

export type MovementDataset = {
  version: 1;
  sequences: MovementSequence[];
};

export type MovementPredictionSource = "exact" | "backoff" | "prior" | "none";

export type MovementPrediction = {
  token: MovementToken | undefined;
  confidence: number;
  /** How the prediction was made — full-order match, backed-off, unigram prior, or nothing learned. */
  source: MovementPredictionSource;
  /** Length of the context actually used to make the prediction. */
  contextLength: number;
};

export type TrainMovementOptions = {
  /** Maximum context order (Markov order). Defaults to 3. */
  order?: number;
};

export type ReplayFidelityReport = {
  steps: number;
  correct: number;
  accuracy: number;
  perSequence: Array<{ id: string; steps: number; correct: number; accuracy: number }>;
};

export type TrainedMovementModel = {
  readonly backend: string;
  readonly version: 1;
  readonly order: number;
  readonly vocabularySize: number;
  readonly sequenceCount: number;
  readonly tokenCount: number;
  /** Predict the next token given a context (only the trailing `order` tokens are used). */
  predict(context: MovementToken[]): MovementPrediction;
  /** Autoregressively generate up to `steps` tokens from a seed context. */
  generate(seed: MovementToken[], steps: number): MovementToken[];
  /** Opaque JSON string that `LocalMovementModelBackend.load` can restore. */
  serialize(): string;
};

/** Pluggable seam: swap the deterministic mock for a real on-device model. */
export interface LocalMovementModelBackend {
  readonly name: string;
  train(dataset: MovementDataset, options?: TrainMovementOptions): Promise<TrainedMovementModel>;
  load(serialized: string): TrainedMovementModel;
}

// ---------------------------------------------------------------------------
// Tokenization: timeline events -> movement tokens
// ---------------------------------------------------------------------------

function slug(value: string | undefined, fallback: string): string {
  const normalized = (value ?? "")
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
  return normalized.length > 0 ? normalized : fallback;
}

function metaString(metadata: Record<string, unknown> | undefined, key: string): string | undefined {
  const value = metadata?.[key];
  return typeof value === "string" ? value : undefined;
}

export function tokenizeObservation(observation: TrajectoryObservation): MovementToken {
  return ["O", slug(observation.source, "src"), slug(observation.summary, "obs")].join("|");
}

export function tokenizeAction(action: TrajectoryAction): MovementToken {
  const gesture = metaString(action.metadata, "gesture");
  const target = metaString(action.metadata, "target") ?? metaString(action.metadata, "direction");
  const verb = gesture ?? slug(action.summary, "act");
  return ["A", slug(action.tool, "tool"), slug(verb, "verb"), slug(target, "any")].join("|");
}

export function tokenizeReplayEvent(event: ReplayTimelineEvent): MovementToken | undefined {
  switch (event.kind) {
    case "observation":
      return ["O", slug(event.source, "src"), slug(event.summary, "obs")].join("|");
    case "action":
      return ["A", slug(event.tool, "tool"), slug(event.summary, "verb"), "any"].join("|");
    case "transcript":
      // Transcript turns are conversational context, not movements.
      return undefined;
  }
}

/** Convert a captured trajectory into an ordered movement sequence. */
export function tokenizeTrajectory(trajectory: TrajectorySpan): MovementSequence {
  const observations = (trajectory.review?.redactedObservations ?? trajectory.observations).map((observation) => ({
    ts: observation.ts,
    token: tokenizeObservation({
      kind: "observation",
      source: observation.source,
      summary: observation.summary,
      ts: observation.ts,
      metadata: "metadata" in observation ? (observation as TrajectoryObservation).metadata : undefined,
    }),
  }));
  const actions = (trajectory.review?.redactedActions ?? trajectory.actions).map((action) => ({
    ts: action.ts,
    token: tokenizeAction({
      kind: "action",
      tool: action.tool,
      summary: action.summary,
      ts: action.ts,
      metadata: "metadata" in action ? (action as TrajectoryAction).metadata : undefined,
    }),
  }));

  const tokens = [...observations, ...actions]
    .sort((a, b) => a.ts - b.ts)
    .map((entry) => entry.token);

  return { id: trajectory.id, tokens };
}

/** Convert a replay manifest into a movement sequence (transcript turns dropped). */
export function tokenizeReplayManifest(manifest: ReplayManifest): MovementSequence {
  const tokens = manifest.events
    .map((event) => tokenizeReplayEvent(event))
    .filter((token): token is MovementToken => token !== undefined);
  return { id: manifest.sessionId, tokens };
}

/** Build a movement dataset from a set of trajectories (empty sequences dropped). */
export function buildMovementDataset(trajectories: TrajectorySpan[]): MovementDataset {
  const sequences = trajectories
    .map((trajectory) => tokenizeTrajectory(trajectory))
    .filter((sequence) => sequence.tokens.length > 0);
  return { version: 1, sequences };
}

// ---------------------------------------------------------------------------
// Deterministic n-gram backend with backoff
// ---------------------------------------------------------------------------

type ContextCounts = Map<string, Map<MovementToken, number>>;

const CONTEXT_SEPARATOR = "";

function contextKey(context: MovementToken[]): string {
  return context.join(CONTEXT_SEPARATOR);
}

function argmax(counts: Map<MovementToken, number>): { token: MovementToken; count: number; total: number } | undefined {
  let best: { token: MovementToken; count: number } | undefined;
  let total = 0;
  // Deterministic iteration: sort tokens so ties resolve lexicographically.
  const entries = [...counts.entries()].sort((a, b) => (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0));
  for (const [token, count] of entries) {
    total += count;
    if (!best || count > best.count) {
      best = { token, count };
    }
  }
  return best ? { token: best.token, count: best.count, total } : undefined;
}

class NgramMovementModel implements TrainedMovementModel {
  readonly backend = "ngram";
  readonly version = 1 as const;

  constructor(
    readonly order: number,
    private readonly contexts: ContextCounts,
    private readonly vocabulary: Set<MovementToken>,
    readonly sequenceCount: number,
    readonly tokenCount: number,
  ) {}

  get vocabularySize(): number {
    return this.vocabulary.size;
  }

  predict(context: MovementToken[]): MovementPrediction {
    // Left-pad with start tokens so the query is scored exactly as training
    // stored it — this makes the empty context predict the start-of-sequence
    // distribution rather than the corpus-wide unigram.
    const padded = [...Array<MovementToken>(this.order).fill(MOVEMENT_START_TOKEN), ...context];
    const maxLen = this.order;
    for (let len = maxLen; len >= 0; len -= 1) {
      const suffix = padded.slice(padded.length - len);
      const counts = this.contexts.get(contextKey(suffix));
      if (!counts || counts.size === 0) {
        continue;
      }
      const best = argmax(counts);
      if (!best) {
        continue;
      }
      const source: MovementPredictionSource = len === maxLen ? "exact" : len > 0 ? "backoff" : "prior";
      return {
        token: best.token === MOVEMENT_START_TOKEN ? undefined : best.token,
        confidence: best.total > 0 ? best.count / best.total : 0,
        source,
        contextLength: len,
      };
    }
    return { token: undefined, confidence: 0, source: "none", contextLength: 0 };
  }

  generate(seed: MovementToken[], steps: number): MovementToken[] {
    const generated: MovementToken[] = [];
    let context = [...seed];
    for (let i = 0; i < steps; i += 1) {
      const prediction = this.predict(context);
      if (prediction.token === undefined) {
        break;
      }
      generated.push(prediction.token);
      context = [...context, prediction.token].slice(-this.order);
    }
    return generated;
  }

  serialize(): string {
    const contexts: Array<[string, Array<[MovementToken, number]>]> = [];
    for (const [key, counts] of this.contexts.entries()) {
      contexts.push([key, [...counts.entries()]]);
    }
    return JSON.stringify({
      backend: this.backend,
      version: this.version,
      order: this.order,
      separator: CONTEXT_SEPARATOR,
      sequenceCount: this.sequenceCount,
      tokenCount: this.tokenCount,
      vocabulary: [...this.vocabulary],
      contexts,
    });
  }

  static deserialize(serialized: string): NgramMovementModel {
    const parsed = JSON.parse(serialized) as {
      order: number;
      sequenceCount: number;
      tokenCount: number;
      vocabulary: MovementToken[];
      contexts: Array<[string, Array<[MovementToken, number]>]>;
    };
    const contexts: ContextCounts = new Map();
    for (const [key, counts] of parsed.contexts) {
      contexts.set(key, new Map(counts));
    }
    return new NgramMovementModel(
      parsed.order,
      contexts,
      new Set(parsed.vocabulary),
      parsed.sequenceCount,
      parsed.tokenCount,
    );
  }
}

export class NgramMovementModelBackend implements LocalMovementModelBackend {
  readonly name = "ngram";

  async train(dataset: MovementDataset, options?: TrainMovementOptions): Promise<TrainedMovementModel> {
    const order = Math.max(1, Math.floor(options?.order ?? 3));
    const contexts: ContextCounts = new Map();
    const vocabulary = new Set<MovementToken>();
    let tokenCount = 0;

    for (const sequence of dataset.sequences) {
      // Pad the front so short contexts (including the first token) are learnable.
      const padded = [...Array<MovementToken>(order).fill(MOVEMENT_START_TOKEN), ...sequence.tokens];
      for (const token of sequence.tokens) {
        vocabulary.add(token);
      }
      tokenCount += sequence.tokens.length;

      for (let i = order; i < padded.length; i += 1) {
        const target = padded[i];
        if (target === undefined) {
          continue;
        }
        for (let len = 0; len <= order; len += 1) {
          const context = padded.slice(i - len, i);
          const key = contextKey(context);
          let counts = contexts.get(key);
          if (!counts) {
            counts = new Map();
            contexts.set(key, counts);
          }
          counts.set(target, (counts.get(target) ?? 0) + 1);
        }
      }
    }

    return new NgramMovementModel(order, contexts, vocabulary, dataset.sequences.length, tokenCount);
  }

  load(serialized: string): TrainedMovementModel {
    return NgramMovementModel.deserialize(serialized);
  }
}

// ---------------------------------------------------------------------------
// Generalization / replay-fidelity evaluation
// ---------------------------------------------------------------------------

/**
 * Measure how well a trained model reproduces a set of movement sequences.
 * Running this on the *training* sequences measures replay fidelity; running
 * it on held-out but related sequences measures generalization.
 */
export function evaluateReplayFidelity(
  model: TrainedMovementModel,
  sequences: MovementSequence[],
): ReplayFidelityReport {
  let totalSteps = 0;
  let totalCorrect = 0;
  const perSequence: ReplayFidelityReport["perSequence"] = [];

  for (const sequence of sequences) {
    let steps = 0;
    let correct = 0;
    for (let i = 0; i < sequence.tokens.length; i += 1) {
      const context = sequence.tokens.slice(Math.max(0, i - model.order), i);
      const prediction = model.predict(context);
      steps += 1;
      if (prediction.token !== undefined && prediction.token === sequence.tokens[i]) {
        correct += 1;
      }
    }
    totalSteps += steps;
    totalCorrect += correct;
    perSequence.push({ id: sequence.id, steps, correct, accuracy: steps > 0 ? correct / steps : 0 });
  }

  return {
    steps: totalSteps,
    correct: totalCorrect,
    accuracy: totalSteps > 0 ? totalCorrect / totalSteps : 0,
    perSequence,
  };
}
