// In-process movement-policy learning + inference pipeline.
//
// The rest of the training subsystem (exporter, job-store, runner) prepares a
// reviewed dataset and emits a shell plan that hands off to an *external*,
// on-device trainer (mlx / axolotl on Apple Silicon). That handoff can only run
// on the user's real machine, so nothing in it is exercisable in the cloud.
//
// This module fills standing objective #2 parts (c) + (d) with a learner that
// runs *entirely in-process*: it tokenizes recorded movement actions into a
// discrete sequence, trains a small back-off n-gram policy over them, and can
// then (c) repeat a recorded movement sequence and (d) generalize to a new but
// related sequence by backing off to shorter shared context. The model backend
// is pluggable via `MovementModelBackend`, so a real on-device small model can
// drop in behind the same interface while tests keep using the deterministic
// Markov backend below.

import type { ReplayManifest, ReplayTimelineEvent } from "../capture/replay.js";
import type { TrajectoryAction, TrajectorySpan } from "../capture/trajectory.js";

/** One discrete movement, e.g. `device:tap:login` or `os:command-ran`. */
export type MovementToken = string;

export type MovementSequence = {
  id: string;
  tokens: MovementToken[];
};

export type MovementDataset = {
  sequences: MovementSequence[];
};

const CONTEXT_SEPARATOR = "";

// ---------------------------------------------------------------------------
// Tokenization: structured movement actions -> canonical discrete tokens.
// ---------------------------------------------------------------------------

function slug(value: string): string {
  return value
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
}

function readMetaString(
  metadata: Record<string, unknown> | undefined,
  key: string,
): string | undefined {
  const value = metadata?.[key];
  return typeof value === "string" && value.trim().length > 0 ? value : undefined;
}

/**
 * Canonicalize a single recorded action into a stable movement token.
 * Prefers structured metadata (gesture / direction / target / event) over the
 * free-text summary so that semantically identical movements collapse to the
 * same token regardless of phrasing.
 */
export function tokenizeAction(action: Pick<TrajectoryAction, "tool" | "summary" | "metadata">): MovementToken {
  const tool = slug(action.tool) || "action";
  const metadata = action.metadata;
  const verbSource =
    readMetaString(metadata, "gesture") ??
    readMetaString(metadata, "event") ??
    action.summary.trim().split(/\s+/)[0] ??
    "act";
  const verb = slug(verbSource) || "act";
  const qualifier = readMetaString(metadata, "direction") ?? readMetaString(metadata, "target");
  const parts = [tool, verb, ...(qualifier ? [slug(qualifier)] : [])].filter((part) => part.length > 0);
  return parts.join(":");
}

/** Ordered movement tokens for a single recorded trajectory span. */
export function tokenizeTrajectory(span: TrajectorySpan): MovementToken[] {
  return [...span.actions]
    .sort((a, b) => a.ts - b.ts)
    .map((action) => tokenizeAction(action));
}

/** Ordered movement tokens for the action events in a replay manifest. */
export function tokenizeReplayManifest(manifest: ReplayManifest): MovementToken[] {
  return manifest.events
    .filter((event): event is Extract<ReplayTimelineEvent, { kind: "action" }> => event.kind === "action")
    .slice()
    .sort((a, b) => a.ts - b.ts)
    .map((event) => tokenizeAction({ tool: event.tool, summary: event.summary }));
}

/** Assemble a training dataset from recorded trajectories and/or replay manifests. */
export function buildMovementDataset(sources: {
  trajectories?: TrajectorySpan[];
  manifests?: ReplayManifest[];
}): MovementDataset {
  const sequences: MovementSequence[] = [];
  for (const span of sources.trajectories ?? []) {
    const tokens = tokenizeTrajectory(span);
    if (tokens.length > 0) {
      sequences.push({ id: span.id, tokens });
    }
  }
  for (const manifest of sources.manifests ?? []) {
    const tokens = tokenizeReplayManifest(manifest);
    if (tokens.length > 0) {
      sequences.push({ id: `${manifest.sessionId}:replay`, tokens });
    }
  }
  return { sequences };
}

// ---------------------------------------------------------------------------
// Pluggable backend interface.
// ---------------------------------------------------------------------------

export type MovementTrainingOptions = {
  /** Maximum n-gram context length. Higher = more faithful recall, less back-off. */
  order?: number;
};

export type MovementCandidate = {
  token: MovementToken;
  probability: number;
};

export type MovementPrediction = {
  token: MovementToken;
  probability: number;
  /** Length of context that produced this prediction; 0 means unigram back-off. */
  matchedOrder: number;
  candidates: MovementCandidate[];
};

export type MovementGenerateOptions = {
  maxSteps?: number;
  stopToken?: MovementToken;
  /** Stop once the best prediction relies on context shorter than this. Default 1. */
  minMatchedOrder?: number;
};

export type MovementModelSnapshot = {
  version: 1;
  backend: string;
  order: number;
  vocabulary: MovementToken[];
  contexts: Array<{
    context: MovementToken[];
    total: number;
    next: Array<[MovementToken, number]>;
  }>;
};

export type TrainedMovementModel = {
  readonly backend: string;
  readonly order: number;
  readonly vocabulary: MovementToken[];
  predictNext(context: MovementToken[]): MovementPrediction | undefined;
  generate(prefix: MovementToken[], options?: MovementGenerateOptions): MovementToken[];
  serialize(): MovementModelSnapshot;
};

export interface MovementModelBackend {
  readonly name: string;
  train(dataset: MovementDataset, options?: MovementTrainingOptions): TrainedMovementModel;
}

// ---------------------------------------------------------------------------
// Deterministic Markov (back-off n-gram) backend.
// ---------------------------------------------------------------------------

type ContextTable = Map<string, Map<MovementToken, number>>;

function keyForContext(context: MovementToken[]): string {
  return context.join(CONTEXT_SEPARATOR);
}

function argmax(counts: Map<MovementToken, number>): MovementCandidate | undefined {
  let total = 0;
  for (const count of counts.values()) {
    total += count;
  }
  if (total === 0) {
    return undefined;
  }
  let bestToken: MovementToken | undefined;
  let bestCount = -1;
  for (const [token, count] of counts) {
    // Deterministic tie-break: higher count wins, then lexicographically smaller token.
    if (count > bestCount || (count === bestCount && (bestToken === undefined || token < bestToken))) {
      bestToken = token;
      bestCount = count;
    }
  }
  if (bestToken === undefined) {
    return undefined;
  }
  return { token: bestToken, probability: bestCount / total };
}

function rankCandidates(counts: Map<MovementToken, number>): MovementCandidate[] {
  let total = 0;
  for (const count of counts.values()) {
    total += count;
  }
  return [...counts.entries()]
    .map(([token, count]) => ({ token, probability: total === 0 ? 0 : count / total }))
    .sort((a, b) => (b.probability !== a.probability ? b.probability - a.probability : a.token < b.token ? -1 : 1));
}

class MarkovMovementModel implements TrainedMovementModel {
  constructor(
    readonly backend: string,
    readonly order: number,
    readonly vocabulary: MovementToken[],
    private readonly tables: ContextTable[],
  ) {}

  predictNext(context: MovementToken[]): MovementPrediction | undefined {
    const maxOrder = Math.min(this.order, context.length);
    for (let n = maxOrder; n >= 0; n -= 1) {
      const table = this.tables[n];
      if (!table) {
        continue;
      }
      const suffix = n === 0 ? [] : context.slice(context.length - n);
      const counts = table.get(keyForContext(suffix));
      if (!counts || counts.size === 0) {
        continue;
      }
      const best = argmax(counts);
      if (!best) {
        continue;
      }
      return {
        token: best.token,
        probability: best.probability,
        matchedOrder: n,
        candidates: rankCandidates(counts),
      };
    }
    return undefined;
  }

  generate(prefix: MovementToken[], options: MovementGenerateOptions = {}): MovementToken[] {
    const maxSteps = options.maxSteps ?? 256;
    const minMatchedOrder = options.minMatchedOrder ?? 1;
    const generated = [...prefix];
    for (let step = 0; step < maxSteps; step += 1) {
      const prediction = this.predictNext(generated);
      if (!prediction || prediction.matchedOrder < minMatchedOrder) {
        break;
      }
      if (options.stopToken !== undefined && prediction.token === options.stopToken) {
        break;
      }
      generated.push(prediction.token);
    }
    return generated;
  }

  serialize(): MovementModelSnapshot {
    const contexts: MovementModelSnapshot["contexts"] = [];
    for (let n = 0; n < this.tables.length; n += 1) {
      const table = this.tables[n];
      if (!table) {
        continue;
      }
      for (const [key, counts] of table) {
        const context = key.length === 0 ? [] : key.split(CONTEXT_SEPARATOR);
        let total = 0;
        for (const count of counts.values()) {
          total += count;
        }
        contexts.push({ context, total, next: [...counts.entries()] });
      }
    }
    return {
      version: 1,
      backend: this.backend,
      order: this.order,
      vocabulary: [...this.vocabulary],
      contexts,
    };
  }
}

export class MarkovMovementBackend implements MovementModelBackend {
  readonly name = "markov";

  train(dataset: MovementDataset, options: MovementTrainingOptions = {}): TrainedMovementModel {
    const order = Math.max(0, Math.floor(options.order ?? 3));
    const tables: ContextTable[] = Array.from({ length: order + 1 }, () => new Map());
    const vocabulary = new Set<MovementToken>();

    for (const sequence of dataset.sequences) {
      const tokens = sequence.tokens;
      for (const token of tokens) {
        vocabulary.add(token);
      }
      for (let i = 0; i < tokens.length; i += 1) {
        const next = tokens[i];
        if (next === undefined) {
          continue;
        }
        for (let n = 0; n <= order; n += 1) {
          if (i - n < 0) {
            break;
          }
          const context = tokens.slice(i - n, i);
          const table = tables[n];
          if (!table) {
            continue;
          }
          const key = keyForContext(context);
          const counts = table.get(key) ?? new Map<MovementToken, number>();
          counts.set(next, (counts.get(next) ?? 0) + 1);
          table.set(key, counts);
        }
      }
    }

    return new MarkovMovementModel(this.name, order, [...vocabulary].sort(), tables);
  }

  static fromSnapshot(snapshot: MovementModelSnapshot): TrainedMovementModel {
    const tables: ContextTable[] = Array.from({ length: snapshot.order + 1 }, () => new Map());
    for (const entry of snapshot.contexts) {
      const n = entry.context.length;
      if (n > snapshot.order) {
        continue;
      }
      const table = tables[n];
      if (!table) {
        continue;
      }
      table.set(keyForContext(entry.context), new Map(entry.next));
    }
    return new MarkovMovementModel(snapshot.backend, snapshot.order, [...snapshot.vocabulary], tables);
  }
}

// ---------------------------------------------------------------------------
// Generalization evaluation harness.
// ---------------------------------------------------------------------------

export type MovementEvaluation = {
  predictions: number;
  correct: number;
  accuracy: number;
  /** Coverage = fraction of positions where the model produced any prediction. */
  covered: number;
  coverage: number;
  /** Accuracy bucketed by the context order that produced each prediction. */
  byMatchedOrder: Record<number, { predictions: number; correct: number }>;
};

/**
 * Next-token accuracy over held-out sequences. Measures how well a model
 * generalizes: every position i>=1 is predicted from the true prefix tokens[0..i-1]
 * and compared against the recorded tokens[i].
 */
export function evaluateMovementModel(model: TrainedMovementModel, heldOut: MovementDataset): MovementEvaluation {
  let predictions = 0;
  let correct = 0;
  let covered = 0;
  const byMatchedOrder: Record<number, { predictions: number; correct: number }> = {};

  for (const sequence of heldOut.sequences) {
    for (let i = 1; i < sequence.tokens.length; i += 1) {
      predictions += 1;
      const expected = sequence.tokens[i];
      const prediction = model.predictNext(sequence.tokens.slice(0, i));
      if (!prediction) {
        continue;
      }
      covered += 1;
      const bucket = (byMatchedOrder[prediction.matchedOrder] ??= { predictions: 0, correct: 0 });
      bucket.predictions += 1;
      if (prediction.token === expected) {
        correct += 1;
        bucket.correct += 1;
      }
    }
  }

  return {
    predictions,
    correct,
    accuracy: predictions === 0 ? 0 : correct / predictions,
    covered,
    coverage: predictions === 0 ? 0 : covered / predictions,
    byMatchedOrder,
  };
}

// ---------------------------------------------------------------------------
// Synthetic movement-stream generator (deterministic; for cloud validation).
// ---------------------------------------------------------------------------

function mulberry32(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export type SyntheticMovementOptions = {
  /** Reusable motifs; each sequence is stitched from a random walk over these. */
  motifs?: MovementToken[][];
  sequences?: number;
  minMotifs?: number;
  maxMotifs?: number;
  seed?: number;
};

const DEFAULT_MOTIFS: MovementToken[][] = [
  ["os:focus-changed", "device:tap:search", "device:type:query"],
  ["device:tap:result", "os:window-opened", "device:scroll:down"],
  ["device:tap:compose", "device:type:body", "device:tap:send"],
  ["device:swipe:left", "device:tap:archive"],
];

/**
 * Produce a family of related movement sequences by stitching shared motifs in
 * varied orders. Sequences overlap in sub-patterns without being identical, so a
 * model trained on some and evaluated on the rest genuinely tests generalization.
 * Deterministic for a given seed.
 */
export function synthesizeMovementSequences(options: SyntheticMovementOptions = {}): MovementDataset {
  const motifs = options.motifs ?? DEFAULT_MOTIFS;
  const count = options.sequences ?? 8;
  const minMotifs = Math.max(1, options.minMotifs ?? 2);
  const maxMotifs = Math.max(minMotifs, options.maxMotifs ?? 4);
  const rng = mulberry32(options.seed ?? 1);
  const sequences: MovementSequence[] = [];

  for (let s = 0; s < count; s += 1) {
    const motifCount = minMotifs + Math.floor(rng() * (maxMotifs - minMotifs + 1));
    const tokens: MovementToken[] = [];
    for (let m = 0; m < motifCount; m += 1) {
      const motif = motifs[Math.floor(rng() * motifs.length)];
      if (motif) {
        tokens.push(...motif);
      }
    }
    sequences.push({ id: `synthetic-${s}`, tokens });
  }

  return { sequences };
}

/** Split a dataset into train/held-out partitions by a deterministic stride. */
export function splitMovementDataset(
  dataset: MovementDataset,
  holdoutEvery = 3,
): { train: MovementDataset; heldOut: MovementDataset } {
  const train: MovementSequence[] = [];
  const heldOut: MovementSequence[] = [];
  dataset.sequences.forEach((sequence, index) => {
    if (holdoutEvery > 0 && (index + 1) % holdoutEvery === 0) {
      heldOut.push(sequence);
    } else {
      train.push(sequence);
    }
  });
  return { train: { sequences: train }, heldOut: { sequences: heldOut } };
}
