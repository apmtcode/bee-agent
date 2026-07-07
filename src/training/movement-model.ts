import type { ReplayTimelineEvent } from "../capture/replay.js";
import type { TrajectorySpan } from "../capture/trajectory.js";

/**
 * Movement-model backend layer.
 *
 * This module provides the inference/learning half of the local-movement
 * learning subsystem (standing objective #2, parts c + d): given recorded
 * movement sequences it trains a model that can (c) *repeat* the recorded
 * movements and (d) *generalize* to new-but-related movements.
 *
 * The {@link MovementModelBackend} interface is the pluggable seam. bee-agent
 * ships a deterministic, dependency-free {@link InMemoryMovementModelBackend}
 * (an n-gram sequence model with stupid-backoff) so the whole pipeline can be
 * exercised in the cloud / CI with no real OS input and no GPU. A real
 * on-device backend (e.g. the mlx/axolotl model produced by
 * {@link LocalAppleSiliconTrainingRunner}) implements the *same* interface:
 * `train` becomes a no-op wrapper around a launched job and `load` reconstructs
 * a model from a trained-artifact snapshot. Call sites depend only on the
 * interface, never on a concrete backend.
 */

/** A single normalized movement — one recorded action or observation. */
export type MovementEvent =
  | { kind: "action"; tool: string; summary: string }
  | { kind: "observation"; source: string; summary: string };

/** An ordered run of movements captured within one trajectory/session. */
export type MovementSequence = {
  id: string;
  events: MovementEvent[];
};

/** A replayable, backend-agnostic training dataset. */
export type MovementDataset = {
  version: 1;
  sequences: MovementSequence[];
};

export type MovementTrainingConfig = {
  /** Maximum context length (n-gram order). Higher = more literal recall. */
  order?: number;
};

export type MovementPredictOptions = {
  /** Hard cap on generated movements (safety bound against runaway loops). */
  maxSteps?: number;
};

/** Serialized model state — the persistence/hand-off seam for real backends. */
export type MovementModelSnapshot = {
  version: 1;
  backendId: string;
  order: number;
  /** vocab token -> decoded event, so generated tokens can be replayed. */
  vocab: Record<string, MovementEvent>;
  /** order (k) -> context-key -> nextToken -> count. */
  transitions: Record<string, Record<string, Record<string, number>>>;
};

/** A trained model instance. Deterministic given the same dataset + config. */
export interface MovementModel {
  readonly backendId: string;
  /**
   * Produce the movements that most likely follow `context`. An empty context
   * generates the most likely full sequence from the start marker.
   */
  predict(context: MovementEvent[], options?: MovementPredictOptions): MovementEvent[];
  serialize(): MovementModelSnapshot;
}

/** Pluggable backend — the extension point for real on-device models. */
export interface MovementModelBackend {
  readonly id: string;
  train(dataset: MovementDataset, config?: MovementTrainingConfig): Promise<MovementModel>;
  /** Reconstruct a model from a snapshot (must have `backendId === this.id`). */
  load(snapshot: MovementModelSnapshot): MovementModel;
}

const START_TOKEN = "\u0000<start>";
const END_TOKEN = "\u0000<end>";
const FIELD_SEP = "\u0001";
const CONTEXT_SEP = "\u0002";
const DEFAULT_ORDER = 3;
const DEFAULT_MAX_STEPS = 256;

function encodeEvent(event: MovementEvent): string {
  return event.kind === "action"
    ? `a${FIELD_SEP}${event.tool}${FIELD_SEP}${event.summary}`
    : `o${FIELD_SEP}${event.source}${FIELD_SEP}${event.summary}`;
}

function contextKey(tokens: string[]): string {
  return tokens.join(CONTEXT_SEP);
}

/**
 * Deterministic n-gram movement model with stupid-backoff.
 *
 * - Exact recall: when the context matches a recorded prefix, the highest-order
 *   n-gram uniquely determines the continuation, so recorded movements replay
 *   verbatim.
 * - Generalization: for an unseen high-order context it backs off to shorter
 *   contexts (down to the unigram), so a new-but-related prefix still yields a
 *   plausible continuation drawn from related recordings.
 *
 * Ties are broken by (count desc, token asc) so predictions are fully
 * reproducible across runs and machines.
 */
export class InMemoryMovementModel implements MovementModel {
  readonly backendId: string;

  constructor(
    backendId: string,
    private readonly order: number,
    private readonly vocab: Map<string, MovementEvent>,
    /** transitions[k] maps a k-token context key -> (token -> count). */
    private readonly transitions: Map<string, Map<string, number>>[],
  ) {
    this.backendId = backendId;
  }

  predict(context: MovementEvent[], options: MovementPredictOptions = {}): MovementEvent[] {
    const maxSteps = options.maxSteps ?? DEFAULT_MAX_STEPS;
    const tokens: string[] = [START_TOKEN, ...context.map(encodeEvent)];
    const generated: MovementEvent[] = [];

    for (let step = 0; step < maxSteps; step += 1) {
      const next = this.nextToken(tokens);
      if (next === undefined || next === END_TOKEN) {
        break;
      }
      tokens.push(next);
      const event = this.vocab.get(next);
      if (event) {
        generated.push(event);
      }
    }

    return generated;
  }

  private nextToken(tokens: string[]): string | undefined {
    const maxContext = Math.min(this.order, tokens.length);
    for (let k = maxContext; k >= 0; k -= 1) {
      const key = contextKey(tokens.slice(tokens.length - k));
      const counts = this.transitions[k]?.get(key);
      if (counts && counts.size > 0) {
        return argmaxToken(counts);
      }
    }
    return undefined;
  }

  serialize(): MovementModelSnapshot {
    const vocab: Record<string, MovementEvent> = {};
    for (const [token, event] of this.vocab) {
      vocab[token] = event;
    }
    const transitions: Record<string, Record<string, Record<string, number>>> = {};
    this.transitions.forEach((table, k) => {
      const byContext: Record<string, Record<string, number>> = {};
      for (const [key, counts] of table) {
        const byToken: Record<string, number> = {};
        for (const [token, count] of counts) {
          byToken[token] = count;
        }
        byContext[key] = byToken;
      }
      transitions[String(k)] = byContext;
    });
    return { version: 1, backendId: this.backendId, order: this.order, vocab, transitions };
  }

  static fromSnapshot(snapshot: MovementModelSnapshot): InMemoryMovementModel {
    const vocab = new Map<string, MovementEvent>(Object.entries(snapshot.vocab));
    const transitions: Map<string, Map<string, number>>[] = [];
    for (let k = 0; k <= snapshot.order; k += 1) {
      const table = new Map<string, Map<string, number>>();
      const byContext = snapshot.transitions[String(k)] ?? {};
      for (const [key, byToken] of Object.entries(byContext)) {
        table.set(key, new Map(Object.entries(byToken)));
      }
      transitions[k] = table;
    }
    return new InMemoryMovementModel(snapshot.backendId, snapshot.order, vocab, transitions);
  }
}

function argmaxToken(counts: Map<string, number>): string {
  let best: string | undefined;
  let bestCount = -Infinity;
  for (const [token, count] of counts) {
    if (count > bestCount || (count === bestCount && best !== undefined && token < best)) {
      best = token;
      bestCount = count;
    }
  }
  // `best` is defined because callers only invoke this on non-empty maps.
  return best as string;
}

export const IN_MEMORY_MOVEMENT_BACKEND_ID = "in-memory-ngram";

export class InMemoryMovementModelBackend implements MovementModelBackend {
  readonly id = IN_MEMORY_MOVEMENT_BACKEND_ID;

  async train(dataset: MovementDataset, config: MovementTrainingConfig = {}): Promise<MovementModel> {
    const order = Math.max(1, config.order ?? DEFAULT_ORDER);
    const vocab = new Map<string, MovementEvent>();
    const transitions: Map<string, Map<string, number>>[] = [];
    for (let k = 0; k <= order; k += 1) {
      transitions[k] = new Map<string, Map<string, number>>();
    }

    for (const sequence of dataset.sequences) {
      const tokens = [START_TOKEN, ...sequence.events.map((event) => {
        const token = encodeEvent(event);
        vocab.set(token, event);
        return token;
      }), END_TOKEN];

      for (let i = 1; i < tokens.length; i += 1) {
        const nextToken = tokens[i];
        for (let k = 0; k <= order; k += 1) {
          if (i - k < 0) {
            break;
          }
          const key = contextKey(tokens.slice(i - k, i));
          const table = transitions[k];
          const counts = table.get(key) ?? new Map<string, number>();
          counts.set(nextToken, (counts.get(nextToken) ?? 0) + 1);
          table.set(key, counts);
        }
      }
    }

    return new InMemoryMovementModel(this.id, order, vocab, transitions);
  }

  load(snapshot: MovementModelSnapshot): MovementModel {
    if (snapshot.backendId !== this.id) {
      throw new Error(
        `snapshot backend "${snapshot.backendId}" cannot be loaded by "${this.id}"`,
      );
    }
    return InMemoryMovementModel.fromSnapshot(snapshot);
  }
}

/** Registry so backends are selectable by id (real vs. mock) at runtime. */
export class MovementModelBackendRegistry {
  private readonly backends = new Map<string, MovementModelBackend>();

  register(backend: MovementModelBackend): this {
    this.backends.set(backend.id, backend);
    return this;
  }

  get(id: string): MovementModelBackend {
    const backend = this.backends.get(id);
    if (!backend) {
      throw new Error(`unknown movement-model backend "${id}" (have: ${[...this.backends.keys()].join(", ") || "none"})`);
    }
    return backend;
  }

  has(id: string): boolean {
    return this.backends.has(id);
  }

  list(): string[] {
    return [...this.backends.keys()];
  }
}

/** A registry pre-loaded with the deterministic in-memory backend. */
export function createDefaultMovementBackendRegistry(): MovementModelBackendRegistry {
  return new MovementModelBackendRegistry().register(new InMemoryMovementModelBackend());
}

/** Map a replay-timeline event to a normalized movement (transcripts dropped). */
export function movementEventFromTimeline(event: ReplayTimelineEvent): MovementEvent | undefined {
  if (event.kind === "action") {
    return { kind: "action", tool: event.tool, summary: event.summary };
  }
  if (event.kind === "observation") {
    return { kind: "observation", source: event.source, summary: event.summary };
  }
  return undefined;
}

/** Build a dataset from replay manifests (one sequence per manifest). */
export function datasetFromReplayManifests(
  manifests: { trajectoryIds: string[]; events: ReplayTimelineEvent[] }[],
): MovementDataset {
  const sequences = manifests.map((manifest, index) => ({
    id: manifest.trajectoryIds.join("+") || `replay-${index}`,
    events: manifest.events
      .map(movementEventFromTimeline)
      .filter((event): event is MovementEvent => event !== undefined),
  }));
  return { version: 1, sequences };
}

/** Build a dataset from trajectory spans (observations + actions, ts-ordered). */
export function datasetFromTrajectories(trajectories: TrajectorySpan[]): MovementDataset {
  const sequences = trajectories.map((trajectory) => {
    const events: (MovementEvent & { ts: number })[] = [
      ...trajectory.observations.map((observation) => ({
        kind: "observation" as const,
        source: observation.source,
        summary: observation.summary,
        ts: observation.ts,
      })),
      ...trajectory.actions.map((action) => ({
        kind: "action" as const,
        tool: action.tool,
        summary: action.summary,
        ts: action.ts,
      })),
    ].sort((a, b) => a.ts - b.ts);
    return {
      id: trajectory.id,
      events: events.map(({ ts: _ts, ...event }) => event),
    };
  });
  return { version: 1, sequences };
}

export type MovementEvalResult = {
  sequences: number;
  /** Predicted movements that exactly matched the held-out continuation. */
  matched: number;
  /** Total held-out movements evaluated across all sequences. */
  total: number;
  /** matched / total (1 = perfect replay fidelity), or 1 when total is 0. */
  fidelity: number;
};

/**
 * Generalization/replay-fidelity harness: for each held-out sequence, seed the
 * model with the first `contextLength` movements and score how many of the
 * remaining movements it reproduces in order.
 */
export function evaluateMovementModel(
  model: MovementModel,
  heldOut: MovementSequence[],
  contextLength = 1,
): MovementEvalResult {
  let matched = 0;
  let total = 0;
  for (const sequence of heldOut) {
    const seed = sequence.events.slice(0, contextLength);
    const expected = sequence.events.slice(contextLength);
    if (expected.length === 0) {
      continue;
    }
    const predicted = model.predict(seed, { maxSteps: expected.length });
    for (let i = 0; i < expected.length; i += 1) {
      total += 1;
      if (predicted[i] && encodeEvent(predicted[i]) === encodeEvent(expected[i])) {
        matched += 1;
      }
    }
  }
  return {
    sequences: heldOut.length,
    matched,
    total,
    fidelity: total === 0 ? 1 : matched / total,
  };
}
