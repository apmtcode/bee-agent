import type { ReplayTimelineEvent } from "../capture/replay.js";
import type { ReviewedExportManifest } from "./export-manifest.js";

/**
 * Pluggable local-model backend for the movement-learning subsystem.
 *
 * The reviewed-export → training-plan → on-device-training path (see
 * {@link ../training/runner.ts}) produces a real model artifact when bee-agent
 * runs on the user's machine. That path cannot execute in the cloud, so this
 * module defines the *interface* every model backend implements plus a
 * deterministic, dependency-free reference backend that runs anywhere.
 *
 * A backend answers two questions:
 *  - `train(dataset)`  — post-train on recorded movements so they can be repeated.
 *  - `predict(model, context)` — given the current context, emit the next
 *    movement, either recalled verbatim (repeat) or interpolated from the
 *    nearest recorded movement (generalize to new-but-related movements).
 *
 * A real on-device small model registers itself under a new name (see
 * {@link createMovementBackend}) and returns its own {@link TrainedMovementModel}
 * subtype; call sites depend only on this interface, never on a concrete class.
 */
export interface ModelBackend<TModel extends TrainedMovementModel = TrainedMovementModel> {
  /** Stable identifier, also stored on the produced model artifact. */
  readonly name: string;
  /** Post-train on a movement dataset, returning a serializable model artifact. */
  train(dataset: MovementDataset, config?: MovementTrainingConfig): Promise<TModel>;
  /** Predict the next movement for a context. Pure and deterministic. */
  predict(model: TModel, context: MovementQueryContext): MovementPrediction;
}

/** Base shape shared by every backend's serialized model artifact. */
export interface TrainedMovementModel {
  version: 1;
  backend: string;
  sampleCount: number;
}

/** One supervised example: preceding context → the movement that followed. */
export type MovementSample = {
  trajectoryId: string;
  /** Events that preceded the action, oldest-first (already capped). */
  contextEvents: MovementContextInput[];
  action: { tool: string; summary: string };
};

/** A replayable movement dataset derived from a reviewed export. */
export type MovementDataset = {
  version: 1;
  sampleCount: number;
  samples: MovementSample[];
};

/**
 * A single context event fed to a backend. A superset of the fields carried by
 * {@link ReplayTimelineEvent} so callers can synthesise *new* contexts (for
 * generalization) without owning a full replay timeline.
 */
export type MovementContextInput = {
  kind: "transcript" | "observation" | "action";
  summary?: string;
  source?: string;
  tool?: string;
  role?: string;
  content?: string;
};

export type MovementQueryContext = {
  events: MovementContextInput[];
};

export type MovementPredictionSource = "recall" | "generalized" | "fallback";

export type MovementPrediction = {
  tool: string;
  summary: string;
  /** Cosine similarity to the matched recorded context, 0..1. */
  confidence: number;
  source: MovementPredictionSource;
  matchedTrajectoryId?: string;
  matchedSampleIndex?: number;
};

export type MovementTrainingConfig = {
  /** How many trailing context events are featurized (default 4). */
  contextWindow?: number;
  /** Similarity at/above which a prediction counts as an exact recall (default 0.999). */
  recallThreshold?: number;
  /** Similarity above which a non-exact prediction counts as generalized (default 0). */
  generalizeThreshold?: number;
};

export type BuildMovementDatasetOptions = {
  /** How many preceding events to retain per sample (default 6). */
  datasetContextWindow?: number;
};

const DEFAULT_DATASET_CONTEXT_WINDOW = 6;
const DEFAULT_CONTEXT_WINDOW = 4;
const DEFAULT_RECALL_THRESHOLD = 0.999;
const DEFAULT_GENERALIZE_THRESHOLD = 0;

/**
 * Turn a reviewed export's replay timelines into supervised next-movement
 * samples: for every recorded `action` event, capture the window of events that
 * preceded it as the context and the action as the label.
 */
export function buildMovementDataset(
  manifest: ReviewedExportManifest,
  options: BuildMovementDatasetOptions = {},
): MovementDataset {
  const window = Math.max(0, options.datasetContextWindow ?? DEFAULT_DATASET_CONTEXT_WINDOW);
  const samples: MovementSample[] = [];

  for (const replay of manifest.replays) {
    const events = replay.events;
    for (let index = 0; index < events.length; index += 1) {
      const event = events[index];
      if (event.kind !== "action") {
        continue;
      }
      const start = window === 0 ? index : Math.max(0, index - window);
      const contextEvents = events.slice(start, index).map(toContextInput);
      samples.push({
        trajectoryId: event.trajectoryId,
        contextEvents,
        action: { tool: event.tool, summary: event.summary },
      });
    }
  }

  return { version: 1, sampleCount: samples.length, samples };
}

function toContextInput(event: ReplayTimelineEvent): MovementContextInput {
  switch (event.kind) {
    case "transcript":
      return { kind: "transcript", role: event.role, content: event.content };
    case "observation":
      return { kind: "observation", source: event.source, summary: event.summary };
    case "action":
      return { kind: "action", tool: event.tool, summary: event.summary };
  }
}

/** Serialized artifact for the deterministic nearest-neighbour backend. */
export interface NearestNeighborMovementModel extends TrainedMovementModel {
  backend: "deterministic-nn";
  contextWindow: number;
  recallThreshold: number;
  generalizeThreshold: number;
  /** The majority movement, returned when a query shares no tokens with training. */
  defaultAction?: { tool: string; summary: string };
  entries: NearestNeighborEntry[];
}

type NearestNeighborEntry = {
  trajectoryId: string;
  tokens: string[];
  tool: string;
  summary: string;
};

/**
 * Deterministic, dependency-free reference backend. It featurizes each context
 * as a bag of tokens and, at inference time, returns the recorded movement whose
 * context is most cosine-similar to the query — exact matches are "recall"
 * (repeat the movement), partial matches are "generalized" (a new-but-related
 * movement), and zero-overlap queries fall back to the majority movement.
 *
 * Chosen over a random/opaque model precisely so cloud/CI runs can assert the
 * repeat-and-generalize behaviour end-to-end without any OS access or training
 * dependency.
 */
export class DeterministicNearestNeighborBackend
  implements ModelBackend<NearestNeighborMovementModel>
{
  readonly name = "deterministic-nn";

  async train(
    dataset: MovementDataset,
    config: MovementTrainingConfig = {},
  ): Promise<NearestNeighborMovementModel> {
    const contextWindow = Math.max(0, config.contextWindow ?? DEFAULT_CONTEXT_WINDOW);
    const recallThreshold = config.recallThreshold ?? DEFAULT_RECALL_THRESHOLD;
    const generalizeThreshold = config.generalizeThreshold ?? DEFAULT_GENERALIZE_THRESHOLD;

    const entries: NearestNeighborEntry[] = dataset.samples.map((sample) => ({
      trajectoryId: sample.trajectoryId,
      tokens: featurizeContext(sample.contextEvents, contextWindow),
      tool: sample.action.tool,
      summary: sample.action.summary,
    }));

    return {
      version: 1,
      backend: "deterministic-nn",
      sampleCount: entries.length,
      contextWindow,
      recallThreshold,
      generalizeThreshold,
      defaultAction: computeMajorityAction(dataset.samples),
      entries,
    };
  }

  predict(model: NearestNeighborMovementModel, context: MovementQueryContext): MovementPrediction {
    if (model.entries.length === 0) {
      return { tool: "noop", summary: "", confidence: 0, source: "fallback" };
    }

    const queryTokens = featurizeContext(context.events, model.contextWindow);
    const queryVector = toTermCounts(queryTokens);

    let bestSimilarity = -1;
    let bestIndex = -1;
    for (let index = 0; index < model.entries.length; index += 1) {
      const similarity = cosineSimilarity(queryVector, toTermCounts(model.entries[index].tokens));
      // Strict `>` keeps the earliest entry on ties → stable and deterministic.
      if (similarity > bestSimilarity) {
        bestSimilarity = similarity;
        bestIndex = index;
      }
    }

    if (bestSimilarity <= 0) {
      const fallback = model.defaultAction ?? {
        tool: model.entries[0].tool,
        summary: model.entries[0].summary,
      };
      return { tool: fallback.tool, summary: fallback.summary, confidence: 0, source: "fallback" };
    }

    const matched = model.entries[bestIndex];
    const source: MovementPredictionSource =
      bestSimilarity >= model.recallThreshold
        ? "recall"
        : bestSimilarity > model.generalizeThreshold
          ? "generalized"
          : "fallback";

    return {
      tool: matched.tool,
      summary: matched.summary,
      confidence: bestSimilarity,
      source,
      matchedTrajectoryId: matched.trajectoryId,
      matchedSampleIndex: bestIndex,
    };
  }
}

export const MOVEMENT_BACKEND_NAMES = ["deterministic-nn"] as const;

export type MovementBackendName = (typeof MOVEMENT_BACKEND_NAMES)[number];

/**
 * Backend registry seam. Cloud/CI resolves the deterministic backend; a real
 * on-device backend adds a case here (and its own {@link TrainedMovementModel}
 * subtype) without touching any call site.
 */
export function createMovementBackend(
  name: MovementBackendName = "deterministic-nn",
): ModelBackend {
  switch (name) {
    case "deterministic-nn":
      return new DeterministicNearestNeighborBackend();
    default: {
      const exhaustive: never = name;
      throw new Error(`Unknown movement backend: ${String(exhaustive)}`);
    }
  }
}

function computeMajorityAction(
  samples: MovementSample[],
): { tool: string; summary: string } | undefined {
  const counts = new Map<string, { count: number; order: number; action: { tool: string; summary: string } }>();
  samples.forEach((sample, order) => {
    const key = `${sample.action.tool} ${sample.action.summary}`;
    const existing = counts.get(key);
    if (existing) {
      existing.count += 1;
    } else {
      counts.set(key, { count: 1, order, action: { ...sample.action } });
    }
  });

  let best: { count: number; order: number; action: { tool: string; summary: string } } | undefined;
  for (const entry of counts.values()) {
    // Higher count wins; earliest first-occurrence breaks ties → deterministic.
    if (!best || entry.count > best.count || (entry.count === best.count && entry.order < best.order)) {
      best = entry;
    }
  }
  return best?.action;
}

/** Tokenize the trailing `window` context events into a lower-cased bag of terms. */
export function featurizeContext(events: MovementContextInput[], window: number): string[] {
  const scoped = window <= 0 ? events : events.slice(-window);
  const tokens: string[] = [];
  for (const event of scoped) {
    tokens.push(`kind:${event.kind}`);
    pushTokens(tokens, event.summary);
    pushTokens(tokens, event.content);
    if (event.source) {
      tokens.push(`source:${event.source.toLowerCase()}`);
    }
    if (event.tool) {
      tokens.push(`tool:${event.tool.toLowerCase()}`);
    }
    if (event.role) {
      tokens.push(`role:${event.role.toLowerCase()}`);
    }
  }
  return tokens;
}

function pushTokens(tokens: string[], text: string | undefined): void {
  if (!text) {
    return;
  }
  for (const word of text.toLowerCase().split(/[^a-z0-9]+/)) {
    if (word) {
      tokens.push(word);
    }
  }
}

function toTermCounts(tokens: string[]): Map<string, number> {
  const counts = new Map<string, number>();
  for (const token of tokens) {
    counts.set(token, (counts.get(token) ?? 0) + 1);
  }
  return counts;
}

function cosineSimilarity(a: Map<string, number>, b: Map<string, number>): number {
  if (a.size === 0 || b.size === 0) {
    return 0;
  }
  let dot = 0;
  for (const [token, weight] of a) {
    const other = b.get(token);
    if (other !== undefined) {
      dot += weight * other;
    }
  }
  if (dot === 0) {
    return 0;
  }
  const magnitude = vectorMagnitude(a) * vectorMagnitude(b);
  return magnitude === 0 ? 0 : dot / magnitude;
}

function vectorMagnitude(vector: Map<string, number>): number {
  let sumOfSquares = 0;
  for (const weight of vector.values()) {
    sumOfSquares += weight * weight;
  }
  return Math.sqrt(sumOfSquares);
}
