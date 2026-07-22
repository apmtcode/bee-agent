import type { ReplayManifest, ReplayTimelineEvent } from "../capture/replay.js";
import type { TrajectorySpan } from "../capture/trajectory.js";

/**
 * Local-movement learning subsystem — model layer.
 *
 * The capture pipeline records movements/actions as {@link ReplayTimelineEvent}s
 * and the exporter turns reviewed trajectories into replay manifests. This
 * module defines the *pluggable local-model* seam (objective #2c/#2d): a small,
 * on-device sequence model that (a) learns the grammar of recorded movements so
 * it can *repeat* them, and (b) *generalizes* to novel-but-related movement
 * sequences.
 *
 * Everything here is backend-agnostic and runs fully in-process so it can be
 * validated in the cloud with synthetic event streams. A real on-device model
 * (e.g. an MLX-trained small model) can be dropped in behind
 * {@link MovementModelBackend} without touching callers.
 */

/** A single discrete movement token, e.g. `act:mouse.click`, `<bos>`, `<eos>`. */
export type MovementToken = string;

/** Start-of-sequence sentinel emitted before the first recorded movement. */
export const MOVEMENT_BOS: MovementToken = "<bos>";
/** End-of-sequence sentinel emitted after the last recorded movement. */
export const MOVEMENT_EOS: MovementToken = "<eos>";

/** An ordered movement sequence distilled from one recorded session/trajectory. */
export type MovementSequence = {
  sessionId: string;
  /** Movement tokens in temporal order, WITHOUT bos/eos sentinels. */
  tokens: MovementToken[];
};

/** A training corpus of movement sequences ready to be learned. */
export type MovementDataset = {
  version: 1;
  sequences: MovementSequence[];
};

/** Result of asking a trained model for the next movement given a context. */
export type MovementPrediction = {
  token: MovementToken;
  /** Estimated probability of `token` given the context under the model. */
  probability: number;
  /**
   * How many context tokens the model could actually condition on. A value
   * below the requested order means the model *backed off* to a shorter context
   * — the mechanism by which it generalizes to unseen prefixes.
   */
  conditionedOrder: number;
};

export type MovementGenerateOptions = {
  /** Hard cap on generated tokens (excluding sentinels). Defaults to 256. */
  maxLength?: number;
};

/** A trained, serializable movement model. Implementations must be deterministic. */
export interface MovementModel {
  readonly backend: string;
  /** Highest context order (in tokens) the model conditions on. */
  readonly order: number;
  /** Predict the single most likely next token given a context window. */
  predictNext(context: MovementToken[]): MovementPrediction;
  /**
   * Greedily generate a movement sequence continuing from `prefix`.
   * Generation stops at {@link MOVEMENT_EOS} or `maxLength`. Sentinels are
   * stripped from the returned tokens.
   */
  generate(prefix: MovementToken[], options?: MovementGenerateOptions): MovementToken[];
  /** JSON-serializable snapshot for persistence as a training artifact. */
  serialize(): SerializedMovementModel;
}

/** Persisted form of a {@link MovementModel}; the on-disk training artifact. */
export type SerializedMovementModel = {
  version: 1;
  backend: string;
  order: number;
  /** Backend-specific learned parameters (counts, weights, …). */
  parameters: Record<string, unknown>;
};

export type MovementTrainOptions = {
  /** Context order (in tokens) to learn; backends may clamp to a supported range. */
  order?: number;
};

/** Pluggable local-model backend: trains a {@link MovementModel} from a dataset. */
export interface MovementModelBackend {
  readonly name: string;
  train(dataset: MovementDataset, options?: MovementTrainOptions): MovementModel;
  /** Reconstruct a previously trained model from its serialized artifact. */
  load(data: SerializedMovementModel): MovementModel;
}

/**
 * Turn one recorded movement event into a discrete token. The token captures
 * the *movement verb* (which tool/source/role acted) rather than free-text
 * detail, so the model learns the reusable grammar of movements — `click` then
 * `type` then `submit` — instead of memorizing exact summaries.
 */
export function tokenizeReplayEvent(event: ReplayTimelineEvent): MovementToken {
  switch (event.kind) {
    case "action":
      return `act:${normalizeVerb(event.tool)}`;
    case "observation":
      return `obs:${normalizeVerb(event.source)}`;
    case "transcript":
      return `msg:${normalizeVerb(event.role)}`;
  }
}

/** Tokenize a trajectory span directly (observations + actions, temporal order). */
export function tokenizeTrajectory(trajectory: TrajectorySpan): MovementToken[] {
  const events: Array<{ ts: number; token: MovementToken; order: number }> = [
    ...trajectory.observations.map((observation) => ({
      ts: observation.ts,
      token: `obs:${normalizeVerb(observation.source)}`,
      order: 0,
    })),
    ...trajectory.actions.map((action) => ({
      ts: action.ts,
      token: `act:${normalizeVerb(action.tool)}`,
      order: 1,
    })),
  ];
  return events
    .sort((a, b) => (a.ts !== b.ts ? a.ts - b.ts : a.order - b.order))
    .map((entry) => entry.token);
}

/**
 * Build a training dataset from replay manifests. Only `observation`/`action`
 * events are kept — the *movements* — while `transcript` chatter is dropped so
 * the model learns motor sequences rather than dialogue. Sequences with no
 * movements are omitted.
 */
export function buildMovementDataset(replays: ReplayManifest[]): MovementDataset {
  const sequences: MovementSequence[] = [];
  for (const replay of replays) {
    const tokens = replay.events
      .filter((event) => event.kind === "action" || event.kind === "observation")
      .map((event) => tokenizeReplayEvent(event));
    if (tokens.length > 0) {
      sequences.push({ sessionId: replay.sessionId, tokens });
    }
  }
  return { version: 1, sequences };
}

/** Combine two datasets (e.g. accumulate across export runs). Pure. */
export function mergeMovementDatasets(...datasets: MovementDataset[]): MovementDataset {
  return {
    version: 1,
    sequences: datasets.flatMap((dataset) => dataset.sequences.map((sequence) => ({
      sessionId: sequence.sessionId,
      tokens: [...sequence.tokens],
    }))),
  };
}

export type ReplayFidelityResult = {
  /** True if greedy generation reproduces the recorded sequence exactly. */
  exactMatch: boolean;
  /** Fraction of positions where the generated token matches the recorded one. */
  tokenAccuracy: number;
  expected: MovementToken[];
  generated: MovementToken[];
};

/**
 * Measure how faithfully a model *repeats* a recorded movement sequence
 * (objective #2c). Greedily generates from an empty prefix and compares against
 * the recorded tokens position-by-position.
 */
export function evaluateReplayFidelity(model: MovementModel, sequence: MovementSequence): ReplayFidelityResult {
  const generated = model.generate([], { maxLength: sequence.tokens.length + 4 });
  const length = Math.max(sequence.tokens.length, generated.length);
  let matches = 0;
  for (let index = 0; index < length; index += 1) {
    if (sequence.tokens[index] !== undefined && sequence.tokens[index] === generated[index]) {
      matches += 1;
    }
  }
  const tokenAccuracy = length === 0 ? 1 : matches / length;
  return {
    exactMatch: tokensEqual(sequence.tokens, generated),
    tokenAccuracy,
    expected: [...sequence.tokens],
    generated,
  };
}

export type GeneralizationResult = {
  /** Top-1 next-token accuracy under teacher forcing over held-out sequences. */
  nextTokenAccuracy: number;
  /** How often the model had to back off below full order to make a prediction. */
  backoffRate: number;
  evaluatedPredictions: number;
};

/**
 * Measure how well a model *generalizes* to novel-but-related movement
 * sequences (objective #2d). For each held-out sequence, walk every position
 * under teacher forcing and check whether the model's top prediction matches the
 * real next movement. `backoffRate` reports how often the model generalized via
 * a shorter context rather than an exact match — high accuracy with non-zero
 * backoff is the signal that it is composing, not memorizing.
 */
export function evaluateGeneralization(
  model: MovementModel,
  heldOut: MovementSequence[],
): GeneralizationResult {
  let predictions = 0;
  let correct = 0;
  let backoffs = 0;
  for (const sequence of heldOut) {
    const tokens = [...sequence.tokens, MOVEMENT_EOS];
    for (let index = 0; index < tokens.length; index += 1) {
      const context = tokens.slice(0, index);
      const prediction = model.predictNext(context);
      predictions += 1;
      if (prediction.token === tokens[index]) {
        correct += 1;
      }
      if (prediction.conditionedOrder < Math.min(model.order, context.length)) {
        backoffs += 1;
      }
    }
  }
  return {
    nextTokenAccuracy: predictions === 0 ? 0 : correct / predictions,
    backoffRate: predictions === 0 ? 0 : backoffs / predictions,
    evaluatedPredictions: predictions,
  };
}

function normalizeVerb(value: string): MovementToken {
  const normalized = value.trim().toLowerCase().replace(/\s+/g, "_");
  return normalized.length > 0 ? normalized : "unknown";
}

function tokensEqual(a: MovementToken[], b: MovementToken[]): boolean {
  if (a.length !== b.length) {
    return false;
  }
  return a.every((token, index) => token === b[index]);
}
