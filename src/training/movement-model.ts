/**
 * Movement-policy learning: the in-process, pluggable train + inference layer
 * for bee-agent's local-movement learning subsystem (standing objective #2 c/d).
 *
 * The existing training runner ({@link ./runner.ts}) emits shell launch scripts
 * that drive real on-device trainers (mlx / axolotl) on Apple Silicon. That is
 * the *production* backend, and it cannot run in the cloud/CI. This module adds
 * the missing seam: a `MovementPolicyBackend` interface plus a deterministic,
 * dependency-free reference backend that actually *learns* from recorded
 * movement sequences and *predicts* the next movement — so the whole
 * capture → dataset → train → infer → generalize loop can be validated with
 * synthetic event streams, with no OS access and no external model.
 *
 * The reference backend is a variable-order Markov model with stupid-backoff.
 * The backoff is what gives generalization: an unseen full-length context falls
 * back to the longest seen suffix, so a *new but related* movement sequence
 * still yields a sensible next-movement prediction. A real small local model
 * (an on-device transformer, an RL policy, …) can implement the same interface
 * and be swapped in behind it — the backend is pluggable by construction.
 *
 * Everything here is pure and deterministic (no wall-clock, no RNG except an
 * explicitly-seeded PRNG in the synthetic generator), so it is fully testable.
 */

import type { ReplayManifest, ReplayTimelineEvent } from "../capture/replay.js";
import type { TrajectorySpan } from "../capture/trajectory.js";

/** A single discrete movement, tokenized for the model plus a human label. */
export type MovementEvent = {
  /** Whether this movement is a passive observation or an active action. */
  kind: "observation" | "action";
  /** Canonical discrete token the model learns over, e.g. `act:click`. */
  token: string;
  /** Human-readable original (tool/source + summary), kept for replay. */
  label: string;
  /** Event timestamp (ms) — preserved for ordering / replay fidelity. */
  ts: number;
};

/** An ordered movement sequence for one trajectory / session. */
export type MovementSequence = {
  id: string;
  events: MovementEvent[];
};

/** One training example: a bounded context of tokens and the token that follows. */
export type MovementExample = {
  context: string[];
  next: string;
};

/** A dataset ready to be handed to a {@link MovementPolicyBackend}. */
export type MovementDataset = {
  /** Maximum context length the model may condition on. */
  order: number;
  /** Full tokenized sequences (backends may build their own n-gram counts). */
  sequences: string[][];
  /** Pre-windowed (context → next) examples, for backends that want them. */
  examples: MovementExample[];
  /** Sorted list of every distinct token observed. */
  vocabulary: string[];
};

/** A ranked next-movement guess. */
export type MovementCandidate = {
  token: string;
  probability: number;
};

/** The result of asking a policy to predict the next movement. */
export type MovementPrediction = {
  /** Most likely next token, or `undefined` if the model is empty. */
  token: string | undefined;
  /** Probability mass on {@link token} (0..1). */
  confidence: number;
  /** Ranked candidates (highest probability first). */
  candidates: MovementCandidate[];
  /**
   * Context length actually used to produce the prediction. A value shorter
   * than the supplied context means the model *generalized* via backoff — it
   * had never seen this exact context and fell back to a seen suffix.
   */
  backoffOrder: number;
};

/** A serialized, persistable policy — plain JSON, safe to write to disk. */
export type SerializedMovementPolicy = {
  version: 1;
  backendId: string;
  order: number;
  /** counts[k] maps a joined k-length context to its next-token counts. */
  counts: Array<Record<string, Record<string, number>>>;
};

/** A trained, runnable movement policy. */
export interface MovementPolicy {
  readonly backendId: string;
  readonly order: number;
  /** Predict the next movement given the trailing context. */
  predict(context: string[]): MovementPrediction;
  /** Serialize to plain JSON for persistence / transport. */
  serialize(): SerializedMovementPolicy;
}

/**
 * Pluggable training backend. The reference implementation is
 * {@link NgramMovementBackend}; a real on-device model implements the same
 * shape and is swapped in without touching callers.
 */
export interface MovementPolicyBackend {
  readonly id: string;
  train(dataset: MovementDataset): MovementPolicy;
}

const CONTEXT_SEPARATOR = "";

function joinContext(context: string[]): string {
  return context.join(CONTEXT_SEPARATOR);
}

/** Canonical token for a replay/trajectory event. */
function canonicalToken(kind: "observation" | "action", primary: string): string {
  const normalized = primary.trim().toLowerCase().replace(/\s+/g, "-") || "unknown";
  return `${kind === "action" ? "act" : "obs"}:${normalized}`;
}

/** Build a movement sequence from a capture replay manifest. */
export function sequenceFromReplay(
  replay: Pick<ReplayManifest, "sessionId" | "events">,
  options: { includeObservations?: boolean } = {},
): MovementSequence {
  const includeObservations = options.includeObservations ?? true;
  const events: MovementEvent[] = [];
  for (const event of replay.events) {
    const mapped = movementEventFromReplayEvent(event, includeObservations);
    if (mapped) {
      events.push(mapped);
    }
  }
  return { id: replay.sessionId, events };
}

function movementEventFromReplayEvent(
  event: ReplayTimelineEvent,
  includeObservations: boolean,
): MovementEvent | undefined {
  if (event.kind === "action") {
    return {
      kind: "action",
      token: canonicalToken("action", event.tool),
      label: `${event.tool}: ${event.summary}`,
      ts: event.ts,
    };
  }
  if (event.kind === "observation") {
    if (!includeObservations) {
      return undefined;
    }
    return {
      kind: "observation",
      token: canonicalToken("observation", event.source),
      label: `${event.source}: ${event.summary}`,
      ts: event.ts,
    };
  }
  // Transcript events carry no discrete movement.
  return undefined;
}

/** Build a movement sequence directly from a trajectory span. */
export function sequenceFromTrajectory(
  trajectory: Pick<TrajectorySpan, "id" | "observations" | "actions">,
  options: { includeObservations?: boolean } = {},
): MovementSequence {
  const includeObservations = options.includeObservations ?? true;
  const events: MovementEvent[] = [];
  if (includeObservations) {
    for (const observation of trajectory.observations) {
      events.push({
        kind: "observation",
        token: canonicalToken("observation", observation.source),
        label: `${observation.source}: ${observation.summary}`,
        ts: observation.ts,
      });
    }
  }
  for (const action of trajectory.actions) {
    events.push({
      kind: "action",
      token: canonicalToken("action", action.tool),
      label: `${action.tool}: ${action.summary}`,
      ts: action.ts,
    });
  }
  events.sort((a, b) => a.ts - b.ts);
  return { id: trajectory.id, events };
}

/** Extract the raw token stream from a movement sequence. */
export function tokensOf(sequence: MovementSequence): string[] {
  return sequence.events.map((event) => event.token);
}

/**
 * Turn movement sequences into a training dataset. Produces both the full token
 * sequences (for backends that build their own counts) and pre-windowed
 * `(context → next)` examples at every context length up to `order`.
 */
export function buildMovementDataset(
  sequences: MovementSequence[],
  options: { order?: number } = {},
): MovementDataset {
  const order = Math.max(1, Math.floor(options.order ?? 2));
  const tokenSequences = sequences.map(tokensOf).filter((tokens) => tokens.length > 0);
  const examples: MovementExample[] = [];
  const vocabulary = new Set<string>();

  for (const tokens of tokenSequences) {
    for (const token of tokens) {
      vocabulary.add(token);
    }
    for (let i = 1; i < tokens.length; i += 1) {
      const start = Math.max(0, i - order);
      examples.push({ context: tokens.slice(start, i), next: tokens[i]! });
    }
  }

  return {
    order,
    sequences: tokenSequences,
    examples,
    vocabulary: [...vocabulary].sort(),
  };
}

/** Concrete policy produced by {@link NgramMovementBackend}. */
class NgramMovementPolicy implements MovementPolicy {
  constructor(
    readonly backendId: string,
    readonly order: number,
    /** counts[k] = joined k-length context → (nextToken → count). */
    private readonly counts: Array<Map<string, Map<string, number>>>,
  ) {}

  predict(context: string[]): MovementPrediction {
    const maxOrder = Math.min(this.order, context.length);
    for (let k = maxOrder; k >= 0; k -= 1) {
      const suffix = context.slice(context.length - k);
      const table = this.counts[k];
      if (!table) {
        continue;
      }
      const nextCounts = table.get(joinContext(suffix));
      if (!nextCounts || nextCounts.size === 0) {
        continue;
      }
      const candidates = rankCandidates(nextCounts);
      return {
        token: candidates[0]?.token,
        confidence: candidates[0]?.probability ?? 0,
        candidates,
        backoffOrder: k,
      };
    }
    return { token: undefined, confidence: 0, candidates: [], backoffOrder: -1 };
  }

  serialize(): SerializedMovementPolicy {
    return {
      version: 1,
      backendId: this.backendId,
      order: this.order,
      counts: this.counts.map((table) => {
        const record: Record<string, Record<string, number>> = {};
        for (const [ctx, nextCounts] of table) {
          const inner: Record<string, number> = {};
          for (const [token, count] of nextCounts) {
            inner[token] = count;
          }
          record[ctx] = inner;
        }
        return record;
      }),
    };
  }
}

function rankCandidates(nextCounts: Map<string, number>): MovementCandidate[] {
  let total = 0;
  for (const count of nextCounts.values()) {
    total += count;
  }
  const candidates: MovementCandidate[] = [];
  for (const [token, count] of nextCounts) {
    candidates.push({ token, probability: total > 0 ? count / total : 0 });
  }
  // Deterministic ordering: probability desc, then token asc as tie-breaker.
  candidates.sort((a, b) => {
    if (b.probability !== a.probability) {
      return b.probability - a.probability;
    }
    return a.token < b.token ? -1 : a.token > b.token ? 1 : 0;
  });
  return candidates;
}

/**
 * Reference movement backend: a variable-order Markov model with stupid-backoff.
 * Dependency-free and deterministic — the cloud/CI stand-in for a real on-device
 * model, and a genuine learner in its own right for discrete movement streams.
 */
export class NgramMovementBackend implements MovementPolicyBackend {
  readonly id = "ngram-backoff";

  train(dataset: MovementDataset): MovementPolicy {
    const order = Math.max(1, Math.floor(dataset.order));
    const counts: Array<Map<string, Map<string, number>>> = [];
    for (let k = 0; k <= order; k += 1) {
      counts.push(new Map());
    }

    for (const tokens of dataset.sequences) {
      for (let i = 1; i < tokens.length; i += 1) {
        const next = tokens[i]!;
        const maxK = Math.min(order, i);
        for (let k = 0; k <= maxK; k += 1) {
          const context = tokens.slice(i - k, i);
          const key = joinContext(context);
          const table = counts[k]!;
          let nextCounts = table.get(key);
          if (!nextCounts) {
            nextCounts = new Map();
            table.set(key, nextCounts);
          }
          nextCounts.set(next, (nextCounts.get(next) ?? 0) + 1);
        }
      }
    }

    return new NgramMovementPolicy(this.id, order, counts);
  }
}

/** Rehydrate a persisted policy without retraining. */
export function deserializeMovementPolicy(serialized: SerializedMovementPolicy): MovementPolicy {
  const counts = serialized.counts.map((record) => {
    const table = new Map<string, Map<string, number>>();
    for (const [ctx, inner] of Object.entries(record)) {
      const nextCounts = new Map<string, number>();
      for (const [token, count] of Object.entries(inner)) {
        nextCounts.set(token, count);
      }
      table.set(ctx, nextCounts);
    }
    return table;
  });
  return new NgramMovementPolicy(serialized.backendId, serialized.order, counts);
}

/** Convenience: dataset-build → train, with a pluggable backend. */
export function trainMovementPolicy(
  sequences: MovementSequence[],
  options: { order?: number; backend?: MovementPolicyBackend } = {},
): { policy: MovementPolicy; dataset: MovementDataset } {
  const dataset = buildMovementDataset(sequences, { order: options.order });
  const backend = options.backend ?? new NgramMovementBackend();
  return { policy: backend.train(dataset), dataset };
}

/** Generalization eval report for a policy on held-out sequences. */
export type MovementEvalResult = {
  /** Number of next-movement predictions attempted. */
  totalPredictions: number;
  /** Top-1 correct predictions. */
  correct: number;
  /** Top-1 accuracy (0..1). */
  accuracy: number;
  /** Predictions whose true token was within the top-K candidates. */
  topKCorrect: number;
  topK: number;
  topKAccuracy: number;
  /** Mean confidence assigned to the top-1 candidate. */
  meanConfidence: number;
  /**
   * Predictions bucketed by the backoff order actually used. Entries with a
   * shorter order than the supplied context are generalized predictions.
   */
  byBackoffOrder: Record<number, { predictions: number; correct: number }>;
};

/**
 * Evaluate a policy's next-movement prediction fidelity on held-out sequences.
 * Walks each sequence position-by-position, conditioning only on the trailing
 * `order` tokens, and scores the prediction against the observed next token.
 */
export function evaluateMovementPolicy(
  policy: MovementPolicy,
  sequences: MovementSequence[],
  options: { topK?: number } = {},
): MovementEvalResult {
  const topK = Math.max(1, Math.floor(options.topK ?? 3));
  let totalPredictions = 0;
  let correct = 0;
  let topKCorrect = 0;
  let confidenceSum = 0;
  const byBackoffOrder: Record<number, { predictions: number; correct: number }> = {};

  for (const sequence of sequences) {
    const tokens = tokensOf(sequence);
    for (let i = 1; i < tokens.length; i += 1) {
      const start = Math.max(0, i - policy.order);
      const context = tokens.slice(start, i);
      const prediction = policy.predict(context);
      const actual = tokens[i]!;
      totalPredictions += 1;
      confidenceSum += prediction.confidence;

      const bucket = byBackoffOrder[prediction.backoffOrder] ?? { predictions: 0, correct: 0 };
      bucket.predictions += 1;

      const isTop1 = prediction.token === actual;
      if (isTop1) {
        correct += 1;
        bucket.correct += 1;
      }
      byBackoffOrder[prediction.backoffOrder] = bucket;

      const inTopK = prediction.candidates.slice(0, topK).some((candidate) => candidate.token === actual);
      if (inTopK) {
        topKCorrect += 1;
      }
    }
  }

  return {
    totalPredictions,
    correct,
    accuracy: totalPredictions > 0 ? correct / totalPredictions : 0,
    topKCorrect,
    topK,
    topKAccuracy: totalPredictions > 0 ? topKCorrect / totalPredictions : 0,
    meanConfidence: totalPredictions > 0 ? confidenceSum / totalPredictions : 0,
    byBackoffOrder,
  };
}

/**
 * Deterministic PRNG (mulberry32). Used only by the synthetic generator so that
 * tests are reproducible without relying on `Math.random` (which is unavailable
 * in some execution contexts and non-deterministic everywhere).
 */
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export type SyntheticMovementOptions = {
  /** How many sequences to emit. */
  count: number;
  /** PRNG seed for reproducibility (default 1). */
  seed?: number;
  /**
   * Workflow templates: each is an ordered list of movement tokens describing a
   * canonical task (e.g. open → navigate → type → save). Sampled per sequence.
   */
  workflows?: string[][];
  /**
   * Probability (0..1) of injecting a small mutation per step (skip / repeat /
   * swap), producing *related but novel* sequences that exercise generalization.
   */
  noise?: number;
  /** Base timestamp (ms) for the first event; each step advances by 1000ms. */
  startTs?: number;
};

const DEFAULT_WORKFLOWS: string[][] = [
  ["act:focus-window", "act:click", "act:type", "act:save"],
  ["act:focus-window", "act:scroll", "act:click", "act:copy", "act:paste"],
  ["act:open-menu", "act:click", "act:type", "act:submit"],
  ["obs:screen", "act:move-cursor", "act:click", "act:type", "act:save"],
];

/**
 * Generate synthetic movement sequences from workflow templates with optional
 * structured noise. Fully deterministic given a seed — the cloud stand-in for
 * real recorded OS input, used to validate the capture → train → generalize
 * loop and to build held-out generalization eval sets.
 */
export function generateSyntheticMovementSequences(
  options: SyntheticMovementOptions,
): MovementSequence[] {
  const rand = mulberry32(options.seed ?? 1);
  const workflows = options.workflows ?? DEFAULT_WORKFLOWS;
  const noise = Math.min(1, Math.max(0, options.noise ?? 0));
  const startTs = options.startTs ?? 0;
  const sequences: MovementSequence[] = [];

  for (let s = 0; s < options.count; s += 1) {
    const template = workflows[Math.floor(rand() * workflows.length)] ?? [];
    const tokens: string[] = [];
    for (const token of template) {
      if (noise > 0 && rand() < noise) {
        const mutation = Math.floor(rand() * 3);
        if (mutation === 0) {
          // skip this step
          continue;
        }
        if (mutation === 1) {
          // repeat the step
          tokens.push(token, token);
          continue;
        }
        // otherwise inject an extra observation before the step
        tokens.push("obs:screen", token);
        continue;
      }
      tokens.push(token);
    }
    const events: MovementEvent[] = tokens.map((token, index) => ({
      kind: token.startsWith("act:") ? "action" : "observation",
      token,
      label: token,
      ts: startTs + (s * 100 + index) * 1000,
    }));
    sequences.push({ id: `synthetic-${s}`, events });
  }

  return sequences;
}
