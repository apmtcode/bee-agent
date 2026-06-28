/**
 * Pluggable local movement-model backend.
 *
 * Standing objective #2(c)/(d): post-train a *local* model on a recorded
 * movement dataset so it can (c) repeat the recorded movements and (d)
 * generalize to new-but-related movements.
 *
 * The real on-device training path (`runner.ts` / `execution-service.ts`)
 * generates shell plans for heavyweight runtimes (mlx / axolotl) that only run
 * on the user's Apple-silicon machine. None of that can execute — or be
 * tested — in the cloud. This module fills that gap with an *in-process*,
 * dependency-free, fully deterministic backend so the capture → dataset →
 * train → infer → eval loop can be exercised and regression-tested anywhere.
 *
 * The backend is pluggable: anything implementing {@link MovementModelBackend}
 * can be registered and selected by id, leaving a clean seam for a real
 * on-device small model to drop in later behind the same interface.
 */

import type { TrajectoryAction, TrajectorySpan } from "../capture/trajectory.js";
import type { ReplayTimelineEvent } from "../capture/replay.js";

// ---------------------------------------------------------------------------
// Movement dataset schema
// ---------------------------------------------------------------------------

/** A single discrete movement, represented as an opaque vocabulary token. */
export type MovementToken = string;

/** Boundary markers so the model learns where sequences start and end. */
export const MOVEMENT_START_TOKEN = "<start>" as const;
export const MOVEMENT_END_TOKEN = "<end>" as const;

export type MovementSequence = {
  /** Stable id (usually the source trajectory id) for traceability. */
  id: string;
  tokens: MovementToken[];
};

export type MovementDataset = {
  version: 1;
  sequences: MovementSequence[];
};

/**
 * Tokenize a recorded action into a movement token. Deterministic and stable:
 * the same action always maps to the same token so identical movements share
 * statistics (which is what lets the model both repeat and generalize).
 */
export function tokenizeAction(action: Pick<TrajectoryAction, "tool" | "summary">): MovementToken {
  return `${normalizeKey(action.tool)}:${normalizeKey(action.summary)}`;
}

function normalizeKey(value: string): string {
  return value
    .trim()
    .toLowerCase()
    .replace(/\s+/g, "-")
    .replace(/[^a-z0-9:_-]/g, "");
}

/** Build a movement dataset from trajectory spans (the captured action streams). */
export function datasetFromTrajectories(trajectories: TrajectorySpan[]): MovementDataset {
  const sequences: MovementSequence[] = trajectories
    .map((trajectory) => ({
      id: trajectory.id,
      tokens: [...trajectory.actions]
        .sort((a, b) => a.ts - b.ts)
        .map((action) => tokenizeAction(action)),
    }))
    .filter((sequence) => sequence.tokens.length > 0);
  return { version: 1, sequences };
}

/** Build a movement dataset from replay manifests (post-export timeline events). */
export function datasetFromReplayEvents(
  replays: { trajectoryIds: string[]; events: ReplayTimelineEvent[] }[],
): MovementDataset {
  const sequences: MovementSequence[] = [];
  for (const [index, replay] of replays.entries()) {
    const id = replay.trajectoryIds[0] ?? `replay-${index}`;
    const tokens = replay.events
      .filter((event): event is Extract<ReplayTimelineEvent, { kind: "action" }> => event.kind === "action")
      .sort((a, b) => a.ts - b.ts)
      .map((event) => tokenizeAction({ tool: event.tool, summary: event.summary }));
    if (tokens.length > 0) {
      sequences.push({ id, tokens });
    }
  }
  return { version: 1, sequences };
}

// ---------------------------------------------------------------------------
// Backend / model interfaces (the pluggable seam)
// ---------------------------------------------------------------------------

export type MovementTrainOptions = {
  /** Maximum context length (Markov order). Higher = more memorization. */
  order?: number;
  /** Stupid-backoff discount applied per backoff step. */
  backoffDiscount?: number;
};

export type MovementCandidate = {
  token: MovementToken;
  probability: number;
};

export type MovementPrediction = MovementCandidate & {
  /** Context length actually used after backoff (0 = unigram prior). */
  backoffOrder: number;
  /** Ranked alternatives for the same context (includes the chosen token). */
  alternatives: MovementCandidate[];
};

/** Serializable model state so a trained model can be persisted / restored. */
export type MovementModelSnapshot = {
  version: 1;
  backendId: string;
  order: number;
  backoffDiscount: number;
  vocabulary: MovementToken[];
  /** context (joined by ) -> { nextToken: count }. */
  counts: Record<string, Record<MovementToken, number>>;
};

export interface MovementModel {
  readonly backendId: string;
  readonly order: number;
  vocabulary(): MovementToken[];
  /** Highest-probability next token for a context, or undefined if untrained. */
  predictNext(context: MovementToken[]): MovementPrediction | undefined;
  /** Deterministically roll out a movement sequence from a seed prefix. */
  generate(seed?: MovementToken[], options?: MovementGenerateOptions): MovementToken[];
  snapshot(): MovementModelSnapshot;
}

export type MovementGenerateOptions = {
  maxSteps?: number;
};

export interface MovementModelBackend {
  readonly id: string;
  train(dataset: MovementDataset, options?: MovementTrainOptions): MovementModel;
  restore(snapshot: MovementModelSnapshot): MovementModel;
}

// ---------------------------------------------------------------------------
// Deterministic Markov backend (the in-cloud reference implementation)
// ---------------------------------------------------------------------------

const CONTEXT_SEPARATOR = "";
const DEFAULT_ORDER = 3;
const DEFAULT_BACKOFF_DISCOUNT = 0.4;

class MarkovMovementModel implements MovementModel {
  readonly backendId = "markov";

  constructor(
    readonly order: number,
    private readonly backoffDiscount: number,
    private readonly counts: Map<string, Map<MovementToken, number>>,
    private readonly vocab: MovementToken[],
  ) {}

  vocabulary(): MovementToken[] {
    return [...this.vocab];
  }

  predictNext(context: MovementToken[]): MovementPrediction | undefined {
    for (let used = Math.min(this.order, context.length); used >= 0; used -= 1) {
      const suffix = context.slice(context.length - used);
      const key = suffix.join(CONTEXT_SEPARATOR);
      const nextCounts = this.counts.get(key);
      if (!nextCounts || nextCounts.size === 0) {
        continue;
      }
      const total = sumValues(nextCounts);
      // Stupid-backoff weight: shorter contexts are discounted multiplicatively.
      const weight = this.backoffDiscount ** (this.order - used);
      const alternatives: MovementCandidate[] = [...nextCounts.entries()]
        .map(([token, count]) => ({ token, probability: (count / total) * weight }))
        .sort(compareCandidates);
      const best = alternatives[0];
      if (!best) {
        continue;
      }
      return { ...best, backoffOrder: used, alternatives };
    }
    return undefined;
  }

  generate(seed: MovementToken[] = [], options: MovementGenerateOptions = {}): MovementToken[] {
    const maxSteps = options.maxSteps ?? 256;
    const context: MovementToken[] = [MOVEMENT_START_TOKEN, ...seed];
    const output: MovementToken[] = [...seed];
    for (let step = 0; step < maxSteps; step += 1) {
      const prediction = this.predictNext(context);
      if (!prediction || prediction.token === MOVEMENT_END_TOKEN) {
        break;
      }
      output.push(prediction.token);
      context.push(prediction.token);
    }
    return output;
  }

  snapshot(): MovementModelSnapshot {
    const counts: Record<string, Record<MovementToken, number>> = {};
    for (const [context, nextCounts] of this.counts) {
      counts[context] = Object.fromEntries(nextCounts);
    }
    return {
      version: 1,
      backendId: this.backendId,
      order: this.order,
      backoffDiscount: this.backoffDiscount,
      vocabulary: [...this.vocab],
      counts,
    };
  }
}

export class MarkovMovementBackend implements MovementModelBackend {
  readonly id = "markov";

  train(dataset: MovementDataset, options: MovementTrainOptions = {}): MovementModel {
    const order = Math.max(1, options.order ?? DEFAULT_ORDER);
    const backoffDiscount = clampUnit(options.backoffDiscount ?? DEFAULT_BACKOFF_DISCOUNT);
    const counts = new Map<string, Map<MovementToken, number>>();
    const vocab = new Set<MovementToken>();

    for (const sequence of dataset.sequences) {
      const padded = [MOVEMENT_START_TOKEN, ...sequence.tokens, MOVEMENT_END_TOKEN];
      for (const token of sequence.tokens) {
        vocab.add(token);
      }
      for (let i = 1; i < padded.length; i += 1) {
        const next = padded[i]!;
        // Record every context length 0..order ending at position i-1.
        for (let used = 0; used <= order; used += 1) {
          if (i - used < 0) {
            break;
          }
          const context = padded.slice(i - used, i).join(CONTEXT_SEPARATOR);
          increment(counts, context, next);
        }
      }
    }

    return new MarkovMovementModel(order, backoffDiscount, counts, [...vocab].sort());
  }

  restore(snapshot: MovementModelSnapshot): MovementModel {
    const counts = new Map<string, Map<MovementToken, number>>();
    for (const [context, nextCounts] of Object.entries(snapshot.counts)) {
      counts.set(context, new Map(Object.entries(nextCounts)));
    }
    return new MarkovMovementModel(snapshot.order, snapshot.backoffDiscount, counts, [...snapshot.vocabulary]);
  }
}

// ---------------------------------------------------------------------------
// Pluggable backend registry
// ---------------------------------------------------------------------------

export type MovementBackendFactory = () => MovementModelBackend;

const backendRegistry = new Map<string, MovementBackendFactory>([
  ["markov", () => new MarkovMovementBackend()],
]);

export function registerMovementBackend(id: string, factory: MovementBackendFactory): void {
  backendRegistry.set(id, factory);
}

export function listMovementBackends(): string[] {
  return [...backendRegistry.keys()].sort();
}

export function createMovementBackend(id = "markov"): MovementModelBackend {
  const factory = backendRegistry.get(id);
  if (!factory) {
    throw new Error(`Unknown movement-model backend: ${id}. Known: ${listMovementBackends().join(", ")}`);
  }
  return factory();
}

// ---------------------------------------------------------------------------
// Evaluation harness (replay fidelity + generalization)
// ---------------------------------------------------------------------------

export type MovementEvalResult = {
  steps: number;
  correct: number;
  /** Teacher-forced next-token accuracy in [0, 1]. */
  accuracy: number;
  /** True when greedy generation reproduces the sequence exactly. */
  exactMatch: boolean;
};

/**
 * Replay fidelity: how well the model reproduces a *recorded* sequence.
 * Measured teacher-forced (predict next given the true prefix) plus a strict
 * end-to-end exact-match check via greedy generation from the start.
 */
export function evaluateReplayFidelity(model: MovementModel, sequence: MovementSequence): MovementEvalResult {
  const tokens = sequence.tokens;
  let correct = 0;
  const context: MovementToken[] = [MOVEMENT_START_TOKEN];
  for (const expected of tokens) {
    const prediction = model.predictNext(context);
    if (prediction?.token === expected) {
      correct += 1;
    }
    context.push(expected);
  }
  const generated = model.generate([], { maxSteps: tokens.length + 4 });
  return {
    steps: tokens.length,
    correct,
    accuracy: tokens.length === 0 ? 1 : correct / tokens.length,
    exactMatch: arraysEqual(generated, tokens),
  };
}

export type MovementGeneralizationResult = {
  sequences: number;
  /** Mean teacher-forced accuracy across held-out sequences. */
  meanAccuracy: number;
  /** Fraction of transitions whose full-order context was seen in training. */
  fullOrderCoverage: number;
  perSequence: (MovementEvalResult & { id: string })[];
};

/**
 * Generalization: accuracy on held-out, never-trained sequences. A pure
 * lookup table would score ~0 here; backoff lets the model still predict
 * plausible continuations from shorter known contexts.
 */
export function evaluateGeneralization(
  model: MovementModel,
  heldOut: MovementSequence[],
): MovementGeneralizationResult {
  const perSequence = heldOut.map((sequence) => ({
    id: sequence.id,
    ...evaluateReplayFidelity(model, sequence),
  }));
  let fullOrderHits = 0;
  let transitions = 0;
  for (const sequence of heldOut) {
    const context: MovementToken[] = [MOVEMENT_START_TOKEN];
    for (const expected of sequence.tokens) {
      const prediction = model.predictNext(context);
      transitions += 1;
      if (prediction && prediction.backoffOrder >= Math.min(model.order, context.length)) {
        fullOrderHits += 1;
      }
      context.push(expected);
    }
  }
  const meanAccuracy =
    perSequence.length === 0 ? 0 : perSequence.reduce((sum, r) => sum + r.accuracy, 0) / perSequence.length;
  return {
    sequences: heldOut.length,
    meanAccuracy,
    fullOrderCoverage: transitions === 0 ? 0 : fullOrderHits / transitions,
    perSequence,
  };
}

// ---------------------------------------------------------------------------
// Synthetic event-stream generator (cloud-safe stand-in for real OS capture)
// ---------------------------------------------------------------------------

export type SyntheticMovementOptions = {
  /** Number of trajectories to emit. */
  count: number;
  /** Deterministic seed; same seed + options => identical output. */
  seed?: number;
  /** Action vocabulary the synthetic tasks draw from. */
  tools?: string[];
};

/**
 * Generate deterministic, OS-free movement trajectories that share latent
 * structure (a small set of reusable "motifs"), so a model trained on some can
 * be meaningfully evaluated on held-out others. Used to exercise the
 * capture -> dataset -> train -> infer -> eval loop in environments with no
 * real input devices (e.g. the cloud self-evolution runner).
 */
export function synthesizeMovementTrajectories(options: SyntheticMovementOptions): TrajectorySpan[] {
  const tools = options.tools ?? ["mouse", "keyboard", "window", "scroll"];
  const motifs = buildMotifs(tools);
  let state = (options.seed ?? 1) >>> 0;
  const rand = () => {
    // Mulberry32 — deterministic PRNG, no Math.random (cloud-reproducible).
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };

  const trajectories: TrajectorySpan[] = [];
  for (let i = 0; i < options.count; i += 1) {
    const length = 2 + Math.floor(rand() * 2); // 2-3 motifs per trajectory
    const actions: TrajectoryAction[] = [];
    let ts = 0;
    for (let m = 0; m < length; m += 1) {
      const motif = motifs[Math.floor(rand() * motifs.length)]!;
      for (const step of motif) {
        ts += 10;
        actions.push({ kind: "action", tool: step.tool, summary: step.summary, ts });
      }
    }
    trajectories.push({
      id: `synthetic-${options.seed ?? 1}-${i}`,
      sessionId: `synthetic-session-${i}`,
      createdAt: new Date(0).toISOString(),
      captureTier: "full",
      observations: [],
      actions,
      outcome: { status: "success", summary: "synthetic", reward: 1 },
    });
  }
  return trajectories;
}

function buildMotifs(tools: string[]): { tool: string; summary: string }[][] {
  // Reusable multi-step movement patterns shared across trajectories.
  return [
    [
      { tool: tools[0] ?? "mouse", summary: "move to target" },
      { tool: tools[0] ?? "mouse", summary: "press primary" },
      { tool: tools[0] ?? "mouse", summary: "release primary" },
    ],
    [
      { tool: tools[1] ?? "keyboard", summary: "focus field" },
      { tool: tools[1] ?? "keyboard", summary: "type text" },
      { tool: tools[1] ?? "keyboard", summary: "submit enter" },
    ],
    [
      { tool: tools[2] ?? "window", summary: "activate window" },
      { tool: tools[3] ?? "scroll", summary: "scroll down" },
    ],
  ];
}

// ---------------------------------------------------------------------------
// Internals
// ---------------------------------------------------------------------------

function increment(counts: Map<string, Map<MovementToken, number>>, context: string, token: MovementToken): void {
  let nextCounts = counts.get(context);
  if (!nextCounts) {
    nextCounts = new Map();
    counts.set(context, nextCounts);
  }
  nextCounts.set(token, (nextCounts.get(token) ?? 0) + 1);
}

function sumValues(map: Map<MovementToken, number>): number {
  let total = 0;
  for (const value of map.values()) {
    total += value;
  }
  return total;
}

function compareCandidates(a: MovementCandidate, b: MovementCandidate): number {
  if (b.probability !== a.probability) {
    return b.probability - a.probability;
  }
  // Deterministic tiebreak so generation is reproducible.
  return a.token < b.token ? -1 : a.token > b.token ? 1 : 0;
}

function clampUnit(value: number): number {
  if (Number.isNaN(value)) {
    return DEFAULT_BACKOFF_DISCOUNT;
  }
  return Math.min(1, Math.max(0, value));
}

function arraysEqual(a: MovementToken[], b: MovementToken[]): boolean {
  if (a.length !== b.length) {
    return false;
  }
  return a.every((value, index) => value === b[index]);
}
