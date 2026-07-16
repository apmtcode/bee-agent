// Local-movement learning: an in-process, deterministic model that trains on a
// recorded movement dataset, repeats the recorded movements, and generalizes to
// new-but-related movements.
//
// This is the on-device "train + inference + generalize" seam of the movement
// subsystem (standing objective #2c/#2d). The engine runs in the cloud with no
// access to a real machine, so the default backend here is a fully deterministic
// n-gram model that needs no native deps and can be validated with synthetic
// event streams. Real on-device backends (e.g. a small local transformer trained
// via MLX) implement the same `MovementModelBackend` interface and are swapped in
// through `MovementModelBackendRegistry`.

import type { TrajectoryAction, TrajectorySpan } from "../capture/trajectory.js";
import type { DeviceGestureKind } from "../capture/device-adapter.js";

/** A discrete movement token, e.g. `device:tap:send-button` or `device:swipe:up`. */
export type MovementToken = string;

/** Boundary markers so the model can learn where sequences start and end. */
export const MOVEMENT_START_TOKEN = "<s>" as const;
export const MOVEMENT_END_TOKEN = "</s>" as const;

const CONTEXT_SEPARATOR = "␟";

/** One ordered run of movements, derived from a single trajectory. */
export type MovementSequence = {
  id: string;
  trajectoryId?: string;
  tokens: MovementToken[];
};

/** A replayable, model-trainable dataset of movement sequences. */
export type MovementDataset = {
  version: 1;
  sequences: MovementSequence[];
  /** Sorted, de-duplicated set of every token that appears in `sequences`. */
  vocabulary: MovementToken[];
};

export type MovementModelConfig = {
  /** N-gram context length. 1 = bigram, 2 = trigram, … Higher = more literal. */
  order: number;
};

export const DEFAULT_MOVEMENT_MODEL_CONFIG: MovementModelConfig = { order: 2 };

/** A ranked next-token candidate produced by a trained model. */
export type MovementPrediction = {
  token: MovementToken;
  probability: number;
  count: number;
};

/** Serializable snapshot so a trained model can be persisted and reloaded. */
export type MovementModelSnapshot = {
  version: 1;
  backendId: string;
  order: number;
  vocabulary: MovementToken[];
  /** counts[k] maps a k-length context key -> { nextToken: count }. */
  counts: Array<Record<string, Record<MovementToken, number>>>;
};

/** A model that has been trained on a `MovementDataset`. */
export interface TrainedMovementModel {
  readonly backendId: string;
  readonly order: number;
  readonly vocabulary: MovementToken[];
  /** Ranked candidates for the movement that follows `context` (most-likely first). */
  predictNext(context: MovementToken[]): MovementPrediction[];
  /** Autonomously continue from `prompt` for up to `steps` movements (stops at end marker). */
  generate(prompt: MovementToken[], steps: number): MovementToken[];
  toSnapshot(): MovementModelSnapshot;
}

/** A pluggable training backend. The default is deterministic; real backends may not be. */
export interface MovementModelBackend {
  readonly id: string;
  train(dataset: MovementDataset, config?: Partial<MovementModelConfig>): TrainedMovementModel;
}

// ---------------------------------------------------------------------------
// Tokenization: TrajectorySpan actions -> movement tokens
// ---------------------------------------------------------------------------

function slugify(value: string): string {
  const slug = value
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
  return slug.length > 0 ? slug : "unknown";
}

/**
 * Reduce a captured action to a stable movement token. Prefers structured
 * gesture metadata (as emitted by the device/browser adapters) and falls back
 * to the human summary so any action shape still tokenizes.
 */
export function tokenizeAction(action: Pick<TrajectoryAction, "tool" | "summary" | "metadata">): MovementToken {
  const metadata = action.metadata ?? {};
  const gesture = typeof metadata.gesture === "string" ? metadata.gesture : undefined;
  const target = typeof metadata.target === "string" ? metadata.target : undefined;
  const direction = typeof metadata.direction === "string" ? metadata.direction : undefined;
  const verb = gesture ?? "act";
  const descriptor = target ?? direction ?? action.summary;
  return `${slugify(action.tool)}:${slugify(verb)}:${slugify(descriptor)}`;
}

/** Build one movement sequence from a trajectory's actions (ordered by timestamp). */
export function buildMovementSequence(trajectory: TrajectorySpan): MovementSequence {
  const actions = trajectory.review?.redactedActions
    ? trajectory.review.redactedActions.map((action) => ({ tool: action.tool, summary: action.summary, ts: action.ts }))
    : trajectory.actions;
  const ordered = [...actions].sort((a, b) => a.ts - b.ts);
  return {
    id: trajectory.id,
    trajectoryId: trajectory.id,
    tokens: ordered.map((action) => tokenizeAction(action)),
  };
}

/** Assemble a trainable dataset from a set of trajectories (empty sequences dropped). */
export function buildMovementDataset(trajectories: TrajectorySpan[]): MovementDataset {
  const sequences = trajectories.map((trajectory) => buildMovementSequence(trajectory)).filter((seq) => seq.tokens.length > 0);
  return {
    version: 1,
    sequences,
    vocabulary: collectVocabulary(sequences),
  };
}

/** Build a dataset directly from raw token sequences (useful for tests/tools). */
export function datasetFromTokenSequences(sequences: Array<{ id: string; tokens: MovementToken[]; trajectoryId?: string }>): MovementDataset {
  const kept: MovementSequence[] = sequences
    .filter((seq) => seq.tokens.length > 0)
    .map((seq) => ({ id: seq.id, tokens: [...seq.tokens], ...(seq.trajectoryId ? { trajectoryId: seq.trajectoryId } : {}) }));
  return { version: 1, sequences: kept, vocabulary: collectVocabulary(kept) };
}

function collectVocabulary(sequences: MovementSequence[]): MovementToken[] {
  const vocab = new Set<MovementToken>();
  for (const sequence of sequences) {
    for (const token of sequence.tokens) {
      vocab.add(token);
    }
  }
  return [...vocab].sort();
}

// ---------------------------------------------------------------------------
// Deterministic n-gram backend (default, cloud-safe)
// ---------------------------------------------------------------------------

function contextKey(context: MovementToken[]): string {
  return context.join(CONTEXT_SEPARATOR);
}

class NGramMovementModel implements TrainedMovementModel {
  readonly backendId: string;
  readonly order: number;
  readonly vocabulary: MovementToken[];
  /** counts[k]: Map<contextKey, Map<nextToken, count>> for context length k. */
  private readonly counts: Array<Map<string, Map<MovementToken, number>>>;

  constructor(params: {
    backendId: string;
    order: number;
    vocabulary: MovementToken[];
    counts: Array<Map<string, Map<MovementToken, number>>>;
  }) {
    this.backendId = params.backendId;
    this.order = params.order;
    this.vocabulary = params.vocabulary;
    this.counts = params.counts;
  }

  predictNext(context: MovementToken[]): MovementPrediction[] {
    // Stupid-backoff: use the longest suffix of `context` (up to `order`) that
    // was observed in training; fall back to shorter contexts, then unigram.
    const maxK = Math.min(this.order, context.length);
    for (let k = maxK; k >= 0; k -= 1) {
      const suffix = k === 0 ? [] : context.slice(context.length - k);
      const table = this.counts[k]?.get(contextKey(suffix));
      if (table && table.size > 0) {
        return rankTable(table);
      }
    }
    return [];
  }

  generate(prompt: MovementToken[], steps: number): MovementToken[] {
    // Seed context with start markers so the very first prediction is grounded.
    const context: MovementToken[] = [...Array(this.order).fill(MOVEMENT_START_TOKEN), ...prompt];
    const out: MovementToken[] = [];
    for (let i = 0; i < steps; i += 1) {
      const predictions = this.predictNext(context);
      const next = predictions[0]?.token;
      if (next === undefined || next === MOVEMENT_END_TOKEN) {
        break;
      }
      out.push(next);
      context.push(next);
    }
    return out;
  }

  toSnapshot(): MovementModelSnapshot {
    return {
      version: 1,
      backendId: this.backendId,
      order: this.order,
      vocabulary: [...this.vocabulary],
      counts: this.counts.map((level) => {
        const record: Record<string, Record<MovementToken, number>> = {};
        for (const [key, table] of level) {
          record[key] = Object.fromEntries(table);
        }
        return record;
      }),
    };
  }

  static fromSnapshot(snapshot: MovementModelSnapshot): NGramMovementModel {
    const counts = snapshot.counts.map((level) => {
      const map = new Map<string, Map<MovementToken, number>>();
      for (const [key, table] of Object.entries(level)) {
        map.set(key, new Map(Object.entries(table)));
      }
      return map;
    });
    return new NGramMovementModel({
      backendId: snapshot.backendId,
      order: snapshot.order,
      vocabulary: [...snapshot.vocabulary],
      counts,
    });
  }
}

function rankTable(table: Map<MovementToken, number>): MovementPrediction[] {
  const total = [...table.values()].reduce((sum, count) => sum + count, 0);
  return [...table.entries()]
    .map(([token, count]) => ({ token, count, probability: total > 0 ? count / total : 0 }))
    // Rank by count desc, then token asc for a fully deterministic ordering.
    .sort((a, b) => (b.count !== a.count ? b.count - a.count : a.token < b.token ? -1 : a.token > b.token ? 1 : 0));
}

/**
 * The default, deterministic training backend. Learns order-k transition counts
 * with start/end padding, so it both faithfully repeats recorded sequences and
 * generalizes across trajectories that share sub-movements.
 */
export class NGramMovementModelBackend implements MovementModelBackend {
  readonly id = "deterministic-ngram";

  train(dataset: MovementDataset, config?: Partial<MovementModelConfig>): TrainedMovementModel {
    const order = Math.max(0, Math.floor(config?.order ?? DEFAULT_MOVEMENT_MODEL_CONFIG.order));
    const counts: Array<Map<string, Map<MovementToken, number>>> = Array.from({ length: order + 1 }, () => new Map());

    for (const sequence of dataset.sequences) {
      const padded = [...Array(order).fill(MOVEMENT_START_TOKEN), ...sequence.tokens, MOVEMENT_END_TOKEN];
      for (let i = order; i < padded.length; i += 1) {
        const next = padded[i]!;
        for (let k = 0; k <= order; k += 1) {
          const context = padded.slice(i - k, i);
          const key = contextKey(context);
          const level = counts[k]!;
          const table = level.get(key) ?? new Map<MovementToken, number>();
          table.set(next, (table.get(next) ?? 0) + 1);
          level.set(key, table);
        }
      }
    }

    return new NGramMovementModel({ backendId: this.id, order, vocabulary: [...dataset.vocabulary], counts });
  }
}

export function loadMovementModelSnapshot(snapshot: MovementModelSnapshot): TrainedMovementModel {
  return NGramMovementModel.fromSnapshot(snapshot);
}

// ---------------------------------------------------------------------------
// Pluggable backend registry (seam for real on-device models)
// ---------------------------------------------------------------------------

export class MovementModelBackendRegistry {
  private readonly backends = new Map<string, MovementModelBackend>();

  constructor(backends: MovementModelBackend[] = [new NGramMovementModelBackend()]) {
    for (const backend of backends) {
      this.register(backend);
    }
  }

  register(backend: MovementModelBackend): void {
    this.backends.set(backend.id, backend);
  }

  get(id: string): MovementModelBackend | undefined {
    return this.backends.get(id);
  }

  require(id: string): MovementModelBackend {
    const backend = this.backends.get(id);
    if (!backend) {
      throw new Error(`unknown movement-model backend: ${id}`);
    }
    return backend;
  }

  list(): string[] {
    return [...this.backends.keys()].sort();
  }
}

/** Convenience registry pre-loaded with the deterministic default backend. */
export function createDefaultMovementModelRegistry(): MovementModelBackendRegistry {
  return new MovementModelBackendRegistry();
}

// ---------------------------------------------------------------------------
// Generalization evaluation harness
// ---------------------------------------------------------------------------

export type MovementGeneralizationMetrics = {
  backendId: string;
  order: number;
  trainSequenceCount: number;
  heldOutSequenceCount: number;
  /** Fraction of training sequences the model regenerates exactly (repeat fidelity). */
  replayFidelity: number;
  /** Teacher-forced top-1 next-movement accuracy on held-out sequences (generalization). */
  nextMovementAccuracy: number;
  heldOutPredictedSteps: number;
  heldOutCorrectSteps: number;
};

/**
 * Regenerate a sequence autonomously from its first movement and report whether
 * the model reproduces it exactly — the core "repeat the recorded movement" test.
 */
export function regenerateSequence(model: TrainedMovementModel, sequence: MovementSequence): MovementToken[] {
  if (sequence.tokens.length === 0) {
    return [];
  }
  const [first, ...rest] = sequence.tokens;
  return [first!, ...model.generate([first!], rest.length)];
}

function sequencesEqual(a: MovementToken[], b: MovementToken[]): boolean {
  return a.length === b.length && a.every((token, index) => token === b[index]);
}

/**
 * Train on `train` and measure two things: how faithfully the model repeats the
 * movements it saw (replayFidelity), and how well it predicts the next movement
 * on held-out but related sequences (nextMovementAccuracy).
 */
export function evaluateMovementGeneralization(params: {
  backend: MovementModelBackend;
  train: MovementDataset;
  heldOut: MovementDataset;
  config?: Partial<MovementModelConfig>;
}): MovementGeneralizationMetrics {
  const model = params.backend.train(params.train, params.config);

  let exactReplays = 0;
  for (const sequence of params.train.sequences) {
    if (sequencesEqual(regenerateSequence(model, sequence), sequence.tokens)) {
      exactReplays += 1;
    }
  }

  let predictedSteps = 0;
  let correctSteps = 0;
  for (const sequence of params.heldOut.sequences) {
    const padded = [...Array(model.order).fill(MOVEMENT_START_TOKEN), ...sequence.tokens, MOVEMENT_END_TOKEN];
    for (let i = model.order; i < padded.length; i += 1) {
      const context = padded.slice(0, i);
      const expected = padded[i]!;
      const predicted = model.predictNext(context)[0]?.token;
      predictedSteps += 1;
      if (predicted === expected) {
        correctSteps += 1;
      }
    }
  }

  return {
    backendId: model.backendId,
    order: model.order,
    trainSequenceCount: params.train.sequences.length,
    heldOutSequenceCount: params.heldOut.sequences.length,
    replayFidelity: params.train.sequences.length > 0 ? exactReplays / params.train.sequences.length : 1,
    nextMovementAccuracy: predictedSteps > 0 ? correctSteps / predictedSteps : 1,
    heldOutPredictedSteps: predictedSteps,
    heldOutCorrectSteps: correctSteps,
  };
}

// ---------------------------------------------------------------------------
// Synthetic movement-stream generator (cloud validation, no real OS input)
// ---------------------------------------------------------------------------

// Small, seeded PRNG so synthetic datasets are reproducible without Math.random.
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

type SyntheticStep = {
  gesture: DeviceGestureKind;
  targets?: string[];
  direction?: "up" | "down" | "left" | "right";
  optional?: boolean;
};

type MovementTaskTemplate = {
  name: string;
  steps: SyntheticStep[];
};

/**
 * Task templates share sub-movements (e.g. every task ends by tapping a primary
 * button), so a model trained on some variants can generalize to unseen ones.
 */
export const DEFAULT_SYNTHETIC_TASK_TEMPLATES: MovementTaskTemplate[] = [
  {
    name: "compose-message",
    steps: [
      { gesture: "tap", targets: ["compose-button", "new-message"] },
      { gesture: "type", targets: ["recipient-field"] },
      { gesture: "type", targets: ["message-body"] },
      { gesture: "scroll", direction: "down", optional: true },
      { gesture: "tap", targets: ["send-button"] },
    ],
  },
  {
    name: "file-search",
    steps: [
      { gesture: "shortcut", targets: ["open-search"] },
      { gesture: "type", targets: ["search-field"] },
      { gesture: "scroll", direction: "down", optional: true },
      { gesture: "tap", targets: ["first-result", "second-result"] },
    ],
  },
  {
    name: "settings-toggle",
    steps: [
      { gesture: "tap", targets: ["settings-gear"] },
      { gesture: "scroll", direction: "down", optional: true },
      { gesture: "tap", targets: ["notifications-row", "privacy-row"] },
      { gesture: "swipe", direction: "right" },
      { gesture: "tap", targets: ["save-button", "send-button"] },
    ],
  },
];

export type SyntheticMovementOptions = {
  seed: number;
  templates?: MovementTaskTemplate[];
  variantsPerTemplate?: number;
  sessionId?: string;
};

/**
 * Generate reproducible synthetic trajectories whose actions carry the same
 * gesture metadata the real device adapter emits, so the whole capture ->
 * dataset -> train -> replay pipeline can be exercised with no real OS input.
 */
export function generateSyntheticMovementTrajectories(options: SyntheticMovementOptions): TrajectorySpan[] {
  const templates = options.templates ?? DEFAULT_SYNTHETIC_TASK_TEMPLATES;
  const variants = Math.max(1, options.variantsPerTemplate ?? 4);
  const sessionId = options.sessionId ?? "synthetic-session";
  const random = mulberry32(options.seed);
  const trajectories: TrajectorySpan[] = [];

  for (const template of templates) {
    for (let variant = 0; variant < variants; variant += 1) {
      const actions: TrajectoryAction[] = [];
      let ts = 1;
      for (const step of template.steps) {
        if (step.optional && random() < 0.5) {
          continue;
        }
        const target = step.targets && step.targets.length > 0 ? step.targets[Math.floor(random() * step.targets.length)] : undefined;
        actions.push(buildSyntheticAction(step, target, ts));
        ts += 1;
      }
      trajectories.push({
        id: `${template.name}-${variant}`,
        sessionId,
        createdAt: `2026-01-01T00:00:${String(variant % 60).padStart(2, "0")}.000Z`,
        captureTier: "app",
        observations: [],
        actions,
        outcome: { status: "success", summary: `completed ${template.name}` },
      });
    }
  }

  return trajectories;
}

function buildSyntheticAction(step: SyntheticStep, target: string | undefined, ts: number): TrajectoryAction {
  const descriptor = target ?? step.direction ?? step.gesture;
  return {
    kind: "action",
    tool: "device",
    summary: `${step.gesture} ${descriptor}`,
    ts,
    metadata: {
      gesture: step.gesture,
      ...(target ? { target } : {}),
      ...(step.direction ? { direction: step.direction } : {}),
    },
  };
}

/** Deterministically split a dataset into train / held-out partitions. */
export function splitMovementDataset(
  dataset: MovementDataset,
  heldOutRatio: number,
): { train: MovementDataset; heldOut: MovementDataset } {
  const sorted = [...dataset.sequences].sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0));
  const heldOutCount = Math.min(sorted.length, Math.max(0, Math.round(sorted.length * heldOutRatio)));
  // Take held-out from the tail so train keeps at least one variant of each task.
  const heldOut = sorted.slice(sorted.length - heldOutCount);
  const train = sorted.slice(0, sorted.length - heldOutCount);
  return {
    train: { version: 1, sequences: train, vocabulary: collectVocabulary(train) },
    heldOut: { version: 1, sequences: heldOut, vocabulary: collectVocabulary(heldOut) },
  };
}
