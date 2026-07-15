import type { ExportedReplayManifest, ReviewedExportManifest } from "./export-manifest.js";

/**
 * Pluggable local-movement model layer.
 *
 * bee-agent's training subsystem (runner.ts / execution-service.ts) prepares
 * launch scripts for *external* on-device trainers (MLX / Axolotl). Those only
 * run on the user's real machine. This module adds a complementary, fully
 * in-process model layer that trains and infers in pure TypeScript, so the
 * capture -> dataset -> train -> infer loop can be validated in the cloud/CI on
 * synthetic event streams, and so a real on-device small model can be dropped in
 * behind the same {@link MovementModelBackend} interface later.
 *
 * A "movement" here is one replay-timeline event (an observation or an action)
 * reduced to a coarse, target-agnostic token. Collapsing concrete targets to the
 * gesture/verb is what lets a learned model *generalize*: transitions observed
 * against one target apply to related targets it never saw paired together.
 */

/** Sentinel prepended to every training sequence; also the default generation seed. */
export const MOVEMENT_START = "<start>";
/** Sentinel appended to every training sequence; generation stops when produced. */
export const MOVEMENT_END = "<end>";

/** A single normalized movement, e.g. `act:device:tapped` or `obs:os:focused`. */
export type MovementToken = string;

/** One recorded movement trajectory, tokenized for training. */
export type MovementSequence = {
  sessionId?: string;
  trajectoryId?: string;
  tokens: MovementToken[];
};

export type MovementDataset = {
  version: 1;
  sequences: MovementSequence[];
};

export type TokenizeReplayOptions = {
  /** Include transcript (chat) events as `msg:<role>` tokens. Off by default — movements only. */
  includeTranscript?: boolean;
};

type AnyReplayEvent = ExportedReplayManifest["events"][number];

function firstWord(summary: string): string {
  const word = summary.trim().split(/\s+/)[0] ?? "";
  return word.toLowerCase() || "event";
}

/**
 * Reduce one replay-timeline event to a coarse movement token. Observations and
 * actions keep their source/tool and the *verb* of their summary (e.g. "tapped
 * Submit" -> `act:device:tapped`); concrete targets are dropped so the model
 * learns transferable gesture-level structure.
 */
export function tokenizeReplayEvent(
  event: AnyReplayEvent,
  options: TokenizeReplayOptions = {},
): MovementToken | undefined {
  switch (event.kind) {
    case "action":
      return `act:${event.tool}:${firstWord(event.summary)}`;
    case "observation":
      return `obs:${event.source}:${firstWord(event.summary)}`;
    case "transcript":
      return options.includeTranscript ? `msg:${event.role}` : undefined;
  }
}

/** Tokenize an ordered list of replay events into a movement sequence's tokens. */
export function tokenizeReplayEvents(
  events: readonly AnyReplayEvent[],
  options: TokenizeReplayOptions = {},
): MovementToken[] {
  const tokens: MovementToken[] = [];
  for (const event of events) {
    const token = tokenizeReplayEvent(event, options);
    if (token !== undefined) {
      tokens.push(token);
    }
  }
  return tokens;
}

/** Build a training dataset from a set of exported replay manifests. */
export function buildMovementDataset(
  replays: readonly ExportedReplayManifest[],
  options: TokenizeReplayOptions = {},
): MovementDataset {
  const sequences = replays
    .map<MovementSequence>((replay) => ({
      sessionId: replay.sessionId,
      trajectoryId: replay.trajectoryIds[0],
      tokens: tokenizeReplayEvents(replay.events, options),
    }))
    .filter((sequence) => sequence.tokens.length > 0);
  return { version: 1, sequences };
}

/** Build a dataset directly from a reviewed training export. */
export function buildMovementDatasetFromExport(
  manifest: ReviewedExportManifest,
  options: TokenizeReplayOptions = {},
): MovementDataset {
  return buildMovementDataset(manifest.replays, options);
}

export type MovementCandidate = {
  token: MovementToken;
  probability: number;
};

export type MovementPrediction = {
  /** Most likely next token, or undefined if the model has never seen any context. */
  token?: MovementToken;
  /** Probability of {@link token} within the predicted distribution (0 when empty). */
  probability: number;
  /** Full next-token distribution, highest probability first. */
  distribution: MovementCandidate[];
  /** How many context tokens were actually matched (0 = unigram fallback). */
  matchedOrder: number;
};

export type MovementGenerateOptions = {
  /** Priming tokens. Defaults to `[MOVEMENT_START]`. */
  seed?: MovementToken[];
  /** Hard cap on generated tokens (excluding the seed). Defaults to 64. */
  maxLength?: number;
  /** Include sentinel tokens in the returned sequence. Defaults to false. */
  includeSentinels?: boolean;
};

/** A trained, serializable movement model produced by a {@link MovementModelBackend}. */
export interface MovementModel {
  readonly backend: string;
  /** All tokens the model has observed (excluding sentinels), sorted. */
  readonly vocabulary: MovementToken[];
  /** Predict the next movement given a context. */
  predictNext(context: MovementToken[]): MovementPrediction;
  /** Greedily roll out a movement sequence, deterministically. */
  generate(options?: MovementGenerateOptions): MovementToken[];
  /** Serialize to a plain JSON value for persistence. */
  toJSON(): SerializedMovementModel;
}

export type SerializedMovementModel = {
  version: 1;
  backend: string;
  [key: string]: unknown;
};

export type MovementTrainOptions = {
  /** Backend-specific hyperparameters (e.g. Markov order). */
  [key: string]: unknown;
};

/**
 * Pluggable training + inference backend. The in-process Markov backend is the
 * default (deterministic, cloud-testable); a real on-device small model can
 * implement the same interface and be swapped in via {@link MovementModelRegistry}.
 */
export interface MovementModelBackend {
  readonly name: string;
  train(dataset: MovementDataset, options?: MovementTrainOptions): MovementModel;
  load(serialized: SerializedMovementModel): MovementModel;
}

/** Registry of movement-model backends, keyed by name. */
export class MovementModelRegistry {
  private readonly backends = new Map<string, MovementModelBackend>();

  register(backend: MovementModelBackend): this {
    this.backends.set(backend.name, backend);
    return this;
  }

  has(name: string): boolean {
    return this.backends.has(name);
  }

  get(name: string): MovementModelBackend {
    const backend = this.backends.get(name);
    if (!backend) {
      throw new Error(`Unknown movement-model backend: ${name}`);
    }
    return backend;
  }

  list(): string[] {
    return [...this.backends.keys()].sort();
  }

  /** Rehydrate a model from its serialized form using the recorded backend. */
  load(serialized: SerializedMovementModel): MovementModel {
    return this.get(serialized.backend).load(serialized);
  }
}
