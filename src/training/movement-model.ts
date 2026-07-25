import type { ReplayManifest, ReplayTimelineEvent } from "../capture/replay.js";
import type { TrajectorySpan } from "../capture/trajectory.js";

/**
 * Local-movement learning subsystem — model layer.
 *
 * The capture/replay/exporter pipeline records movements and builds a reviewed
 * dataset; the {@link LocalAppleSiliconTrainingRunner} emits the *real*
 * on-device (MLX/axolotl) training command. Neither of those can actually
 * *train and run* a model inside the cloud sandbox, so objective #2 pieces
 * (c) "post-train a local model to repeat the recorded movements" and
 * (d) "generalize to new but related movements" were previously untestable.
 *
 * This module closes that gap with a pluggable {@link MovementModelBackend}
 * seam and a deterministic, dependency-free reference backend
 * ({@link MarkovMovementBackend}) that genuinely learns transition statistics
 * from recorded trajectories/replays. It is exercisable end-to-end with
 * synthetic event streams (no real OS input, no GPU), while a real on-device
 * small-model backend can be dropped in behind the same interface.
 */

/** A single movement token in a sequence (e.g. `act:mouse.move`, `obs:screen`). */
export type MovementToken = string;

/** Emitted at the start of every training/generation sequence. */
export const MOVEMENT_BOS: MovementToken = "<bos>";
/** Emitted at the end of every training/generation sequence. */
export const MOVEMENT_EOS: MovementToken = "<eos>";

/** One ordered movement sequence derived from a trajectory or replay. */
export type MovementSequence = {
  /** Source trajectory id, for eval attribution. */
  trajectoryId: string;
  /** Ordered movement tokens (BOS/EOS are added by the model, not stored here). */
  tokens: MovementToken[];
};

/** A replayable, model-ready dataset of movement sequences. */
export type MovementDataset = {
  version: 1;
  sequences: MovementSequence[];
};

export type MovementModelConfig = {
  /**
   * Markov context order (k >= 1). Higher orders reproduce recorded movements
   * more exactly; lower orders generalize more aggressively across sequences
   * that share sub-movements. Back-off means an unseen k-context degrades to
   * shorter contexts rather than failing.
   */
  order: number;
};

export const DEFAULT_MOVEMENT_MODEL_CONFIG: MovementModelConfig = { order: 2 };

/** A trained model's serialized form — persist to disk, reload for inference. */
export type SerializedMovementModel = {
  version: 1;
  backendId: string;
  order: number;
  /** context-key -> (next-token -> observed count). */
  transitions: Record<string, Record<MovementToken, number>>;
};

export type MovementPrediction = {
  token: MovementToken;
  probability: number;
};

/**
 * A trained movement model. Deterministic: identical input always yields
 * identical predictions (ties broken lexicographically), so replay and eval
 * are reproducible in CI.
 */
export interface MovementModel {
  readonly backendId: string;
  readonly order: number;
  /** Most-likely next token given the trailing context, or undefined if none. */
  predict(context: MovementToken[]): MovementToken | undefined;
  /** Full next-token distribution (descending probability, deterministic order). */
  distribution(context: MovementToken[]): MovementPrediction[];
  /**
   * Roll the model forward from a seed context, appending predicted tokens
   * until EOS or `maxLength` tokens are produced. BOS/EOS are handled
   * internally and never appear in the returned movement list.
   */
  generate(seed?: MovementToken[], options?: { maxLength?: number }): MovementToken[];
  toJSON(): SerializedMovementModel;
}

/** Pluggable training seam. Swap the reference backend for a real on-device one. */
export interface MovementModelBackend {
  readonly id: string;
  train(dataset: MovementDataset, config?: Partial<MovementModelConfig>): MovementModel;
}

// --------------------------------------------------------------------------
// Tokenization: trajectories / replays -> movement sequences
// --------------------------------------------------------------------------

/** Map a single trajectory span to one ordered movement sequence. */
export function tokenizeTrajectory(trajectory: TrajectorySpan): MovementSequence {
  const events: Array<{ ts: number; order: number; token: MovementToken }> = [];
  for (const observation of trajectory.observations) {
    events.push({ ts: observation.ts, order: 0, token: `obs:${observation.source}` });
  }
  for (const action of trajectory.actions) {
    events.push({ ts: action.ts, order: 1, token: `act:${action.tool}` });
  }
  events.sort((a, b) => (a.ts !== b.ts ? a.ts - b.ts : a.order - b.order));
  return { trajectoryId: trajectory.id, tokens: events.map((event) => event.token) };
}

/** Map a replay manifest's timeline into one movement sequence per trajectory. */
export function tokenizeReplayManifest(manifest: ReplayManifest): MovementSequence[] {
  const byTrajectory = new Map<string, MovementToken[]>();
  for (const trajectoryId of manifest.trajectoryIds) {
    byTrajectory.set(trajectoryId, []);
  }
  for (const event of manifest.events) {
    if (event.kind === "transcript") {
      continue; // transcript turns are not movements
    }
    const token = replayEventToken(event);
    if (!token) {
      continue;
    }
    const list = byTrajectory.get(event.trajectoryId) ?? [];
    list.push(token);
    byTrajectory.set(event.trajectoryId, list);
  }
  return [...byTrajectory.entries()].map(([trajectoryId, tokens]) => ({ trajectoryId, tokens }));
}

function replayEventToken(event: ReplayTimelineEvent): MovementToken | undefined {
  switch (event.kind) {
    case "observation":
      return `obs:${event.source}`;
    case "action":
      return `act:${event.tool}`;
    case "transcript":
      return undefined; // transcript turns are not movements
  }
}

/** Build a model-ready dataset from recorded trajectory spans. */
export function buildMovementDataset(trajectories: TrajectorySpan[]): MovementDataset {
  return {
    version: 1,
    sequences: trajectories.map((trajectory) => tokenizeTrajectory(trajectory)).filter((sequence) => sequence.tokens.length > 0),
  };
}

/** Build a model-ready dataset from replay manifests (exporter output). */
export function buildMovementDatasetFromReplays(manifests: ReplayManifest[]): MovementDataset {
  return {
    version: 1,
    sequences: manifests.flatMap((manifest) => tokenizeReplayManifest(manifest)).filter((sequence) => sequence.tokens.length > 0),
  };
}

// --------------------------------------------------------------------------
// Reference backend: deterministic back-off Markov model
// --------------------------------------------------------------------------

const CONTEXT_SEP = "";

class MarkovMovementModel implements MovementModel {
  constructor(
    readonly backendId: string,
    readonly order: number,
    private readonly transitions: Map<string, Map<MovementToken, number>>,
  ) {}

  distribution(context: MovementToken[]): MovementPrediction[] {
    const counts = this.lookup(context);
    if (!counts) {
      return [];
    }
    const total = [...counts.values()].reduce((sum, count) => sum + count, 0);
    if (total === 0) {
      return [];
    }
    return [...counts.entries()]
      .map(([token, count]) => ({ token, probability: count / total }))
      .filter((prediction) => prediction.token !== MOVEMENT_BOS)
      .sort((a, b) => (b.probability !== a.probability ? b.probability - a.probability : compareToken(a.token, b.token)));
  }

  predict(context: MovementToken[]): MovementToken | undefined {
    return this.distribution(context)[0]?.token;
  }

  generate(seed: MovementToken[] = [], options: { maxLength?: number } = {}): MovementToken[] {
    const maxLength = options.maxLength ?? 256;
    const history: MovementToken[] = [MOVEMENT_BOS, ...seed];
    const produced: MovementToken[] = [...seed];
    while (produced.length < maxLength) {
      const next = this.predict(history);
      if (next === undefined || next === MOVEMENT_EOS) {
        break;
      }
      produced.push(next);
      history.push(next);
    }
    return produced;
  }

  toJSON(): SerializedMovementModel {
    const transitions: Record<string, Record<MovementToken, number>> = {};
    for (const [contextKey, counts] of this.transitions.entries()) {
      transitions[contextKey] = Object.fromEntries(counts.entries());
    }
    return { version: 1, backendId: this.backendId, order: this.order, transitions };
  }

  /** Back-off lookup: try the full k-context, then progressively shorter ones. */
  private lookup(context: MovementToken[]): Map<MovementToken, number> | undefined {
    for (let size = Math.min(this.order, context.length); size >= 0; size -= 1) {
      const key = contextKey(context.slice(context.length - size));
      const counts = this.transitions.get(key);
      if (counts && counts.size > 0) {
        return counts;
      }
    }
    return undefined;
  }
}

export class MarkovMovementBackend implements MovementModelBackend {
  readonly id = "markov-backoff";

  train(dataset: MovementDataset, config: Partial<MovementModelConfig> = {}): MovementModel {
    const order = Math.max(1, Math.floor(config.order ?? DEFAULT_MOVEMENT_MODEL_CONFIG.order));
    const transitions = new Map<string, Map<MovementToken, number>>();

    for (const sequence of dataset.sequences) {
      const tokens = [MOVEMENT_BOS, ...sequence.tokens, MOVEMENT_EOS];
      for (let index = 1; index < tokens.length; index += 1) {
        const next = tokens[index]!;
        // Record this transition at every context width 0..order (back-off table).
        for (let width = 0; width <= order; width += 1) {
          const start = Math.max(0, index - width);
          const key = contextKey(tokens.slice(start, index));
          const counts = transitions.get(key) ?? new Map<MovementToken, number>();
          counts.set(next, (counts.get(next) ?? 0) + 1);
          transitions.set(key, counts);
        }
      }
    }

    return new MarkovMovementModel(this.id, order, transitions);
  }
}

/** Reconstruct a model from its serialized form for inference-only use. */
export function loadMovementModel(serialized: SerializedMovementModel): MovementModel {
  const transitions = new Map<string, Map<MovementToken, number>>();
  for (const [contextKey_, counts] of Object.entries(serialized.transitions)) {
    transitions.set(contextKey_, new Map(Object.entries(counts)));
  }
  return new MarkovMovementModel(serialized.backendId, serialized.order, transitions);
}

// --------------------------------------------------------------------------
// Generalization eval harness
// --------------------------------------------------------------------------

export type MovementEvalResult = {
  sequenceCount: number;
  /** Next-token predictions attempted (teacher-forced across held-out sequences). */
  predictionCount: number;
  /** Predictions that matched the recorded next movement. */
  correct: number;
  /** correct / predictionCount (0 when nothing was predicted). */
  accuracy: number;
  /** Sequences the model reproduced exactly from BOS. */
  exactReplays: number;
};

/**
 * Measure how well a trained model reproduces and generalizes to a held-out
 * set of movement sequences. Uses teacher forcing for per-token accuracy and a
 * free-running generation for exact-replay fidelity.
 */
export function evaluateMovementModel(model: MovementModel, heldOut: MovementDataset): MovementEvalResult {
  let predictionCount = 0;
  let correct = 0;
  let exactReplays = 0;

  for (const sequence of heldOut.sequences) {
    const target = [...sequence.tokens, MOVEMENT_EOS];
    const context: MovementToken[] = [MOVEMENT_BOS];
    for (const expected of target) {
      const predicted = model.predict(context);
      predictionCount += 1;
      if (predicted === expected) {
        correct += 1;
      }
      context.push(expected); // teacher forcing
    }

    const generated = model.generate([], { maxLength: sequence.tokens.length + 1 });
    if (tokensEqual(generated, sequence.tokens)) {
      exactReplays += 1;
    }
  }

  return {
    sequenceCount: heldOut.sequences.length,
    predictionCount,
    correct,
    accuracy: predictionCount === 0 ? 0 : correct / predictionCount,
    exactReplays,
  };
}

function tokensEqual(a: MovementToken[], b: MovementToken[]): boolean {
  return a.length === b.length && a.every((token, index) => token === b[index]);
}

function contextKey(context: MovementToken[]): string {
  return context.join(CONTEXT_SEP);
}

function compareToken(a: MovementToken, b: MovementToken): number {
  return a < b ? -1 : a > b ? 1 : 0;
}
