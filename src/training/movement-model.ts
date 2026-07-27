import type { ReplayManifest, ReplayTimelineEvent } from "../capture/replay.js";
import type { TrajectorySpan } from "../capture/trajectory.js";

/**
 * Local-movement learning: a pluggable, on-device model backend that learns to
 * *repeat* recorded movement sequences and *generalize* to novel-but-related
 * ones.
 *
 * This module provides the code, schema, and a deterministic reference backend
 * so the pipeline is fully testable in the cloud (objective #2 of the
 * self-evolution engine). The heavy on-device backends (MLX/axolotl, produced
 * by {@link LocalAppleSiliconTrainingRunner}) plug into the same
 * {@link MovementModelBackend} interface and execute when the user runs
 * bee-agent locally.
 *
 * The reference {@link MarkovMovementBackend} is a variable-order n-gram model
 * with stupid-backoff. It is intentionally simple and fully deterministic:
 * - trained on a recorded sequence, greedy generation *reproduces* it;
 * - given an unseen high-order context, it *backs off* to shorter contexts, so
 *   related-but-new prefixes still yield a plausible next movement.
 */

/** A single discrete movement token (e.g. `act:device:swiped down`). */
export type MovementToken = string;

/** One recorded movement sequence — the ordered tokens of a trajectory/replay. */
export type MovementSequence = {
  id: string;
  tokens: MovementToken[];
};

/** A replayable, tokenized movement dataset (objective #2b/#2c). */
export type MovementDataset = {
  version: 1;
  sequences: MovementSequence[];
};

/** Which timeline event kinds are treated as movement tokens. */
export type MovementTokenizeOptions = {
  /** Include `action` events (device gestures / tool calls). Default: true. */
  includeActions?: boolean;
  /** Include `observation` events (UI / device context). Default: true. */
  includeObservations?: boolean;
  /** Include `transcript` messages as `msg:<role>` tokens. Default: false. */
  includeTranscript?: boolean;
};

const UNIT_SEP = "␟";
/** Beginning-of-sequence marker — lets the model predict the first movement. */
export const MOVEMENT_BOS: MovementToken = "␂";
/** End-of-sequence marker — lets generation know when to stop. */
export const MOVEMENT_EOS: MovementToken = "␃";

function normalizeSummary(summary: string): string {
  return summary.trim().replace(/\s+/g, " ").toLowerCase();
}

/** Canonicalize a single replay timeline event into a movement token. */
export function tokenizeMovementEvent(event: ReplayTimelineEvent): MovementToken | undefined {
  switch (event.kind) {
    case "action":
      return `act:${event.tool}:${normalizeSummary(event.summary)}`;
    case "observation":
      return `obs:${event.source}:${normalizeSummary(event.summary)}`;
    case "transcript":
      return `msg:${event.role}`;
  }
}

/** Tokenize an ordered list of replay events into a movement token stream. */
export function tokenizeMovementEvents(
  events: readonly ReplayTimelineEvent[],
  options: MovementTokenizeOptions = {},
): MovementToken[] {
  const includeActions = options.includeActions ?? true;
  const includeObservations = options.includeObservations ?? true;
  const includeTranscript = options.includeTranscript ?? false;
  const tokens: MovementToken[] = [];
  for (const event of events) {
    if (event.kind === "action" && !includeActions) continue;
    if (event.kind === "observation" && !includeObservations) continue;
    if (event.kind === "transcript" && !includeTranscript) continue;
    const token = tokenizeMovementEvent(event);
    if (token !== undefined) {
      tokens.push(token);
    }
  }
  return tokens;
}

/** Build a movement dataset from replay manifests (one sequence per replay). */
export function buildMovementDatasetFromReplays(
  replays: readonly ReplayManifest[],
  options: MovementTokenizeOptions = {},
): MovementDataset {
  const sequences: MovementSequence[] = [];
  for (const replay of replays) {
    const tokens = tokenizeMovementEvents(replay.events, options);
    if (tokens.length > 0) {
      sequences.push({ id: replay.sessionId, tokens });
    }
  }
  return { version: 1, sequences };
}

/** Build a movement dataset directly from trajectory spans (actions + observations). */
export function buildMovementDatasetFromTrajectories(
  trajectories: readonly TrajectorySpan[],
  options: MovementTokenizeOptions = {},
): MovementDataset {
  const includeActions = options.includeActions ?? true;
  const includeObservations = options.includeObservations ?? true;
  const sequences: MovementSequence[] = [];
  for (const trajectory of trajectories) {
    const events: ReplayTimelineEvent[] = [
      ...(includeObservations
        ? trajectory.observations.map<ReplayTimelineEvent>((observation) => ({
            kind: "observation",
            ts: observation.ts,
            trajectoryId: trajectory.id,
            source: observation.source,
            summary: observation.summary,
          }))
        : []),
      ...(includeActions
        ? trajectory.actions.map<ReplayTimelineEvent>((action) => ({
            kind: "action",
            ts: action.ts,
            trajectoryId: trajectory.id,
            tool: action.tool,
            summary: action.summary,
          }))
        : []),
    ].sort((a, b) => a.ts - b.ts);
    const tokens = tokenizeMovementEvents(events, { ...options, includeTranscript: false });
    if (tokens.length > 0) {
      sequences.push({ id: trajectory.id, tokens });
    }
  }
  return { version: 1, sequences };
}

/** Options controlling greedy movement generation. */
export type MovementGenerateOptions = {
  /** Prime generation with a partial context (defaults to just the BOS marker). */
  seed?: MovementToken[];
  /** Hard cap on generated tokens (excluding EOS). Default: 256. */
  maxLength?: number;
};

/** A trained, queryable movement model produced by a {@link MovementModelBackend}. */
export interface TrainedMovementModel {
  readonly backend: string;
  /**
   * Predict the single most-likely next token given a context, or `undefined`
   * when the model would stop (EOS) or has no information.
   */
  predictNext(context: MovementToken[]): MovementToken | undefined;
  /** Greedily roll out a full movement sequence (repeat / generalize). */
  generate(options?: MovementGenerateOptions): MovementToken[];
  /** Serialize to a portable JSON string (round-trips via the backend). */
  serialize(): string;
}

/** A pluggable local-model backend (mock, MLX, axolotl, …). */
export interface MovementModelBackend {
  readonly name: string;
  train(dataset: MovementDataset): TrainedMovementModel;
  deserialize(serialized: string): TrainedMovementModel;
}

// --- Deterministic reference backend: variable-order n-gram with backoff -----

export type MarkovMovementConfig = {
  /** Maximum context length used for prediction. Default: 3. */
  order?: number;
};

type TransitionTable = Record<string, Record<MovementToken, number>>;

type SerializedMarkovModel = {
  backend: "markov";
  version: 1;
  order: number;
  transitions: TransitionTable;
};

function contextKey(context: readonly MovementToken[]): string {
  return `${context.length}${UNIT_SEP}${context.join(UNIT_SEP)}`;
}

/**
 * Deterministically pick the highest-count token, breaking ties lexicographically
 * so results are stable across runs and machines (cloud-testable).
 */
function argmaxToken(counts: Record<MovementToken, number>): MovementToken | undefined {
  let best: MovementToken | undefined;
  let bestCount = -1;
  for (const token of Object.keys(counts).sort()) {
    const count = counts[token] ?? 0;
    if (count > bestCount) {
      bestCount = count;
      best = token;
    }
  }
  return best;
}

class MarkovTrainedModel implements TrainedMovementModel {
  readonly backend = "markov";

  constructor(
    private readonly order: number,
    private readonly transitions: TransitionTable,
  ) {}

  predictNext(context: MovementToken[]): MovementToken | undefined {
    const token = this.predictWithEos(context);
    return token === MOVEMENT_EOS ? undefined : token;
  }

  /** Internal: prediction that may return the EOS marker (used by generate). */
  private predictWithEos(context: MovementToken[]): MovementToken | undefined {
    const maxK = Math.min(context.length, this.order);
    for (let k = maxK; k >= 0; k -= 1) {
      const slice = context.slice(context.length - k);
      const counts = this.transitions[contextKey(slice)];
      if (counts && Object.keys(counts).length > 0) {
        return argmaxToken(counts);
      }
    }
    return undefined;
  }

  generate(options: MovementGenerateOptions = {}): MovementToken[] {
    const maxLength = options.maxLength ?? 256;
    const seed = options.seed ?? [];
    // Prime with BOS unless the caller already supplied it.
    const context: MovementToken[] = seed[0] === MOVEMENT_BOS ? [...seed] : [MOVEMENT_BOS, ...seed];
    const generated: MovementToken[] = seed.filter((token) => token !== MOVEMENT_BOS);
    while (generated.length < maxLength) {
      const next = this.predictWithEos(context);
      if (next === undefined || next === MOVEMENT_EOS) {
        break;
      }
      generated.push(next);
      context.push(next);
    }
    return generated;
  }

  serialize(): string {
    const payload: SerializedMarkovModel = {
      backend: "markov",
      version: 1,
      order: this.order,
      transitions: this.transitions,
    };
    return JSON.stringify(payload);
  }
}

export class MarkovMovementBackend implements MovementModelBackend {
  readonly name = "markov";
  private readonly order: number;

  constructor(config: MarkovMovementConfig = {}) {
    this.order = Math.max(1, Math.floor(config.order ?? 3));
  }

  train(dataset: MovementDataset): TrainedMovementModel {
    const transitions: TransitionTable = {};
    for (const sequence of dataset.sequences) {
      const padded = [MOVEMENT_BOS, ...sequence.tokens, MOVEMENT_EOS];
      for (let i = 1; i < padded.length; i += 1) {
        const target = padded[i];
        const maxK = Math.min(i, this.order);
        for (let k = 0; k <= maxK; k += 1) {
          const key = contextKey(padded.slice(i - k, i));
          const counts = (transitions[key] ??= {});
          counts[target] = (counts[target] ?? 0) + 1;
        }
      }
    }
    return new MarkovTrainedModel(this.order, transitions);
  }

  deserialize(serialized: string): TrainedMovementModel {
    const payload = JSON.parse(serialized) as SerializedMarkovModel;
    if (payload.backend !== "markov") {
      throw new Error(`expected a markov model, received backend "${payload.backend}"`);
    }
    return new MarkovTrainedModel(payload.order, payload.transitions);
  }
}

/** Registry of pluggable backends, keyed by name. */
export class MovementModelRegistry {
  private readonly backends = new Map<string, MovementModelBackend>();

  constructor(backends: MovementModelBackend[] = [new MarkovMovementBackend()]) {
    for (const backend of backends) {
      this.register(backend);
    }
  }

  register(backend: MovementModelBackend): void {
    this.backends.set(backend.name, backend);
  }

  get(name: string): MovementModelBackend {
    const backend = this.backends.get(name);
    if (!backend) {
      throw new Error(`unknown movement-model backend "${name}"`);
    }
    return backend;
  }

  list(): string[] {
    return [...this.backends.keys()].sort();
  }

  /** Load a serialized model by reading its `backend` discriminator. */
  deserialize(serialized: string): TrainedMovementModel {
    const backendName = (JSON.parse(serialized) as { backend?: string }).backend;
    if (!backendName) {
      throw new Error("serialized movement model is missing a backend discriminator");
    }
    return this.get(backendName).deserialize(serialized);
  }
}

/** Convenience: exact-match fidelity of a generated sequence vs. a reference. */
export function movementSequenceFidelity(
  reference: readonly MovementToken[],
  candidate: readonly MovementToken[],
): number {
  if (reference.length === 0) {
    return candidate.length === 0 ? 1 : 0;
  }
  const length = Math.min(reference.length, candidate.length);
  let matches = 0;
  for (let i = 0; i < length; i += 1) {
    if (reference[i] === candidate[i]) {
      matches += 1;
    }
  }
  return matches / reference.length;
}
