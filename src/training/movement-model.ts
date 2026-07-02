// Pluggable in-process movement-model backend for the local-movement learning
// subsystem (standing objective #2c/#2d).
//
// The training runner (`runner.ts`) emits launch scripts for *real* on-device
// training (MLX / axolotl on Apple Silicon), which cannot run in the cloud/CI.
// This module provides the complementary piece: a backend-agnostic interface
// plus a deterministic, dependency-free reference backend that can actually
// *train* on the recorded movement dataset and *predict* next movements —
// closing the train -> infer -> generalize loop so the pipeline is validatable
// with synthetic event streams (no real OS, no GPU).
//
// Real on-device backends (a small local model) implement the same
// `MovementModelBackend` interface; the deterministic backend is the default
// used by tests and as a safe fallback when no native backend is configured.

import type { ReplayManifest, ReplayTimelineEvent } from "../capture/replay.js";

/**
 * A single normalized movement event. This is the unit a movement model reasons
 * over — either an observation of environment state or a performed action
 * (mouse/keyboard/gesture/tool invocation). Derived from replay timeline events
 * but intentionally decoupled so backends never depend on capture internals.
 */
export type MovementEvent =
  | { kind: "observation"; source: string; summary: string; ts: number }
  | { kind: "action"; tool: string; summary: string; ts: number };

/** The action portion of a movement event (the thing a model predicts). */
export type MovementAction = { tool: string; summary: string };

/**
 * One supervised training example: the ordered context of events that preceded
 * an action, and the action that was taken next.
 */
export type MovementExample = {
  trajectoryId?: string;
  context: MovementEvent[];
  action: MovementAction & { ts: number };
};

/** A materialized dataset ready for a backend to consume. */
export type MovementDataset = {
  version: 1;
  examples: MovementExample[];
  /** Distinct action labels (`tool::summary`), sorted for determinism. */
  actionVocabulary: string[];
};

export type MovementTrainOptions = {
  /** Maximum n-gram order the backend considers (context tail length). */
  maxOrder?: number;
};

/** How a prediction was produced (useful for eval + debugging). */
export type MovementPredictionSource = "ngram" | "similarity" | "prior" | "none";

export type MovementPrediction = {
  action: MovementAction | undefined;
  label: string | undefined;
  /** 0..1 model confidence for the chosen action. */
  confidence: number;
  source: MovementPredictionSource;
  /** Other candidates the model considered, best-first. */
  alternatives: Array<{ label: string; score: number }>;
};

/** A serializable snapshot of a trained model (for persistence / inspection). */
export type MovementModelSnapshot = {
  backend: string;
  version: 1;
  maxOrder: number;
  exampleCount: number;
  actionVocabulary: string[];
};

/** A trained model: pure inference over a movement context. */
export interface MovementModel {
  readonly backend: string;
  predict(context: MovementEvent[]): MovementPrediction;
  serialize(): MovementModelSnapshot;
}

/** A pluggable training backend. Native backends implement this too. */
export interface MovementModelBackend {
  readonly name: string;
  train(dataset: MovementDataset, options?: MovementTrainOptions): Promise<MovementModel>;
}

// ---------------------------------------------------------------------------
// Dataset construction
// ---------------------------------------------------------------------------

/** Normalize a free-text summary so related movements collapse to one token. */
function normalizeText(value: string): string {
  return value.trim().toLowerCase().replace(/\s+/g, " ");
}

/** Stable label for an action target. */
export function actionLabel(action: MovementAction): string {
  return `${normalizeText(action.tool)}::${normalizeText(action.summary)}`;
}

/** Tokenize a single event into a stable string used for n-gram keys. */
function eventToken(event: MovementEvent): string {
  return event.kind === "observation"
    ? `obs:${normalizeText(event.source)}:${normalizeText(event.summary)}`
    : `act:${normalizeText(event.tool)}:${normalizeText(event.summary)}`;
}

/**
 * Bag of words over a context, used by the similarity fallback. Word-level
 * granularity lets the model generalize across *reworded* contexts (e.g. a
 * related task whose observations are phrased differently) where whole-event
 * tokens would share nothing.
 */
function contextWordSet(context: MovementEvent[]): Set<string> {
  const words = new Set<string>();
  for (const event of context) {
    const parts = event.kind === "observation" ? [event.source, event.summary] : [event.tool, event.summary];
    for (const part of parts) {
      for (const word of normalizeText(part).split(/[^a-z0-9]+/)) {
        if (word.length > 0) {
          words.add(word);
        }
      }
    }
  }
  return words;
}

/**
 * Convert a replay manifest's timeline into an ordered movement sequence.
 * Transcript events become coarse observations so surrounding context is kept
 * without leaking raw message content into the movement vocabulary.
 */
export function movementSequenceFromReplay(manifest: Pick<ReplayManifest, "events">): MovementEvent[] {
  return manifest.events.map((event) => replayEventToMovement(event));
}

function replayEventToMovement(event: ReplayTimelineEvent): MovementEvent {
  switch (event.kind) {
    case "action":
      return { kind: "action", tool: event.tool, summary: event.summary, ts: event.ts };
    case "observation":
      return { kind: "observation", source: event.source, summary: event.summary, ts: event.ts };
    case "transcript":
      return { kind: "observation", source: `transcript:${event.role}`, summary: event.content, ts: event.ts };
  }
}

/**
 * Build a supervised dataset from ordered movement sequences (one per
 * trajectory). Every action becomes a labeled example whose context is the
 * events that preceded it within its sequence.
 */
export function buildMovementDataset(
  sequences: Array<{ trajectoryId?: string; events: MovementEvent[] }>,
): MovementDataset {
  const examples: MovementExample[] = [];
  const vocabulary = new Set<string>();

  for (const sequence of sequences) {
    const context: MovementEvent[] = [];
    for (const event of sequence.events) {
      if (event.kind === "action") {
        examples.push({
          trajectoryId: sequence.trajectoryId,
          context: context.slice(),
          action: { tool: event.tool, summary: event.summary, ts: event.ts },
        });
        vocabulary.add(actionLabel(event));
      }
      context.push(event);
    }
  }

  return {
    version: 1,
    examples,
    actionVocabulary: [...vocabulary].sort(),
  };
}

/** Convenience: build a dataset directly from replay manifests. */
export function buildMovementDatasetFromReplays(
  replays: Array<Pick<ReplayManifest, "events" | "trajectoryIds">>,
): MovementDataset {
  return buildMovementDataset(
    replays.map((replay) => ({
      trajectoryId: replay.trajectoryIds[0],
      events: movementSequenceFromReplay(replay),
    })),
  );
}

// ---------------------------------------------------------------------------
// Deterministic reference backend
// ---------------------------------------------------------------------------

const DEFAULT_MAX_ORDER = 3;

type CandidateCounts = Map<string, number>;

/**
 * A dependency-free, fully deterministic movement backend.
 *
 * It combines two learners so it can both *repeat* recorded movements exactly
 * and *generalize* to new-but-related contexts:
 *
 *  1. A back-off n-gram transition model (orders `maxOrder..1`) over the event
 *     token stream — reproduces recorded sequences with high fidelity.
 *  2. A nearest-neighbour fallback over context token *sets* (Jaccard
 *     similarity) — generalizes to unseen contexts that resemble training ones.
 *  3. A global action prior as a last resort.
 *
 * No randomness is used anywhere; ties break lexicographically, so training and
 * inference are reproducible across machines and runs.
 */
export class DeterministicMovementBackend implements MovementModelBackend {
  readonly name = "deterministic-ngram";

  async train(dataset: MovementDataset, options?: MovementTrainOptions): Promise<MovementModel> {
    const maxOrder = Math.max(1, options?.maxOrder ?? DEFAULT_MAX_ORDER);
    return new DeterministicMovementModel(this.name, dataset, maxOrder);
  }
}

class DeterministicMovementModel implements MovementModel {
  /** order -> (context-key -> action-label -> count) */
  private readonly ngrams = new Map<number, Map<string, CandidateCounts>>();
  /** label -> concrete action (for reconstructing predictions) */
  private readonly labelToAction = new Map<string, MovementAction>();
  /** per-example context token sets, for the similarity fallback */
  private readonly neighbours: Array<{ tokens: Set<string>; label: string }> = [];
  /** global action frequency prior */
  private readonly prior: CandidateCounts = new Map();
  private readonly exampleCount: number;

  constructor(
    readonly backend: string,
    private readonly dataset: MovementDataset,
    private readonly maxOrder: number,
  ) {
    this.exampleCount = dataset.examples.length;
    for (let order = 1; order <= maxOrder; order += 1) {
      this.ngrams.set(order, new Map());
    }

    for (const example of dataset.examples) {
      const label = actionLabel(example.action);
      this.labelToAction.set(label, { tool: example.action.tool, summary: example.action.summary });
      increment(this.prior, label);

      const tokens = example.context.map(eventToken);
      this.neighbours.push({ tokens: contextWordSet(example.context), label });

      for (let order = 1; order <= maxOrder; order += 1) {
        if (tokens.length < order) {
          break;
        }
        const key = contextKey(tokens, order);
        const table = this.ngrams.get(order)!;
        let counts = table.get(key);
        if (!counts) {
          counts = new Map();
          table.set(key, counts);
        }
        increment(counts, label);
      }
    }
  }

  predict(context: MovementEvent[]): MovementPrediction {
    const tokens = context.map(eventToken);

    // 1. Back-off n-gram: try the longest context tail first.
    for (let order = Math.min(this.maxOrder, tokens.length); order >= 1; order -= 1) {
      const counts = this.ngrams.get(order)?.get(contextKey(tokens, order));
      if (counts && counts.size > 0) {
        return this.fromCounts(counts, "ngram");
      }
    }

    // 2. Nearest-neighbour generalization over context word sets.
    const similar = this.mostSimilar(contextWordSet(context));
    if (similar) {
      return {
        action: this.labelToAction.get(similar.label),
        label: similar.label,
        confidence: round(similar.score),
        source: "similarity",
        alternatives: [{ label: similar.label, score: round(similar.score) }],
      };
    }

    // 3. Global prior.
    if (this.prior.size > 0) {
      return this.fromCounts(this.prior, "prior");
    }

    return { action: undefined, label: undefined, confidence: 0, source: "none", alternatives: [] };
  }

  serialize(): MovementModelSnapshot {
    return {
      backend: this.backend,
      version: 1,
      maxOrder: this.maxOrder,
      exampleCount: this.exampleCount,
      actionVocabulary: this.dataset.actionVocabulary.slice(),
    };
  }

  private fromCounts(counts: CandidateCounts, source: MovementPredictionSource): MovementPrediction {
    const total = [...counts.values()].reduce((sum, count) => sum + count, 0);
    const ranked = [...counts.entries()].sort((a, b) => (b[1] - a[1]) || a[0].localeCompare(b[0]));
    const [bestLabel, bestCount] = ranked[0];
    return {
      action: this.labelToAction.get(bestLabel),
      label: bestLabel,
      confidence: total > 0 ? round(bestCount / total) : 0,
      source,
      alternatives: ranked.map(([label, count]) => ({ label, score: round(count / total) })),
    };
  }

  private mostSimilar(queryTokens: Set<string>): { label: string; score: number } | undefined {
    let best: { label: string; score: number } | undefined;
    for (const neighbour of this.neighbours) {
      const score = jaccard(queryTokens, neighbour.tokens);
      if (score <= 0) {
        continue;
      }
      // Deterministic tie-break: higher score wins, then lexical label order.
      if (!best || score > best.score || (score === best.score && neighbour.label.localeCompare(best.label) < 0)) {
        best = { label: neighbour.label, score };
      }
    }
    return best;
  }
}

function increment(counts: CandidateCounts, key: string): void {
  counts.set(key, (counts.get(key) ?? 0) + 1);
}

function contextKey(tokens: string[], order: number): string {
  return tokens.slice(tokens.length - order).join(" → ");
}

function jaccard(a: Set<string>, b: Set<string>): number {
  if (a.size === 0 && b.size === 0) {
    return 0;
  }
  let intersection = 0;
  for (const token of a) {
    if (b.has(token)) {
      intersection += 1;
    }
  }
  const union = a.size + b.size - intersection;
  return union === 0 ? 0 : intersection / union;
}

function round(value: number): number {
  return Math.round(value * 1000) / 1000;
}

// ---------------------------------------------------------------------------
// Train + evaluate orchestration
// ---------------------------------------------------------------------------

/** Fidelity metrics from replaying held-out sequences through a model. */
export type MovementEvalResult = {
  steps: number;
  /** Predicted action label exactly matched the recorded one. */
  exactMatches: number;
  /** Predicted tool matched the recorded tool (looser than exact). */
  toolMatches: number;
  /** exactMatches / steps. */
  fidelity: number;
  /** toolMatches / steps. */
  toolFidelity: number;
  /** Count of predictions by their production source. */
  bySource: Record<MovementPredictionSource, number>;
};

/**
 * Replay held-out sequences through a trained model and measure how faithfully
 * it reproduces / generalizes the recorded movements. Each recorded action is a
 * step: the model predicts from the true preceding context and is scored
 * against the actual next action.
 */
export function evaluateMovementModel(
  model: MovementModel,
  sequences: Array<{ trajectoryId?: string; events: MovementEvent[] }>,
): MovementEvalResult {
  const bySource: Record<MovementPredictionSource, number> = {
    ngram: 0,
    similarity: 0,
    prior: 0,
    none: 0,
  };
  let steps = 0;
  let exactMatches = 0;
  let toolMatches = 0;

  for (const sequence of sequences) {
    const context: MovementEvent[] = [];
    for (const event of sequence.events) {
      if (event.kind === "action") {
        steps += 1;
        const prediction = model.predict(context);
        bySource[prediction.source] += 1;
        if (prediction.label === actionLabel(event)) {
          exactMatches += 1;
        }
        if (prediction.action && normalizeText(prediction.action.tool) === normalizeText(event.tool)) {
          toolMatches += 1;
        }
      }
      context.push(event);
    }
  }

  return {
    steps,
    exactMatches,
    toolMatches,
    fidelity: steps > 0 ? round(exactMatches / steps) : 0,
    toolFidelity: steps > 0 ? round(toolMatches / steps) : 0,
    bySource,
  };
}

/**
 * High-level convenience: train a model on a dataset with the given backend
 * (deterministic by default) and return the model plus its self-replay fidelity
 * on the training data. A `holdout` set, when supplied, additionally measures
 * generalization to unseen-but-related sequences.
 */
export async function trainMovementModel(params: {
  dataset: MovementDataset;
  backend?: MovementModelBackend;
  options?: MovementTrainOptions;
  trainingSequences?: Array<{ trajectoryId?: string; events: MovementEvent[] }>;
  holdout?: Array<{ trajectoryId?: string; events: MovementEvent[] }>;
}): Promise<{
  model: MovementModel;
  replayFidelity?: MovementEvalResult;
  generalization?: MovementEvalResult;
}> {
  const backend = params.backend ?? new DeterministicMovementBackend();
  const model = await backend.train(params.dataset, params.options);
  return {
    model,
    replayFidelity: params.trainingSequences ? evaluateMovementModel(model, params.trainingSequences) : undefined,
    generalization: params.holdout ? evaluateMovementModel(model, params.holdout) : undefined,
  };
}
