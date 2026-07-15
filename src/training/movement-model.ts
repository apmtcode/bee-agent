import type { TrajectorySpan } from "../capture/trajectory.js";

/**
 * In-process, cloud-testable movement-learning backend.
 *
 * The `LocalAppleSiliconTrainingRunner` builds launch scripts that defer real
 * on-device SFT/RL to Apple-silicon hardware. This module provides the
 * complementary piece the objective calls for: a *pluggable model backend* that
 * can actually learn from a recorded movement dataset and predict the next
 * movement — entirely in-process, so the capture -> dataset -> train -> infer
 * pipeline can be validated with synthetic event streams in the cloud/CI.
 *
 * Two behaviours are required of a movement model:
 *   (c) repeat recorded movements  — exact-context prediction, and
 *   (d) generalize to new but related movements — via context backoff.
 * The reference `NgramMovementBackend` implements both with a deterministic,
 * dependency-free variable-order Markov model, so tests are reproducible. Real
 * on-device small models plug in behind the same {@link MovementModelBackend}
 * seam.
 */

/** A single event in a movement timeline, tokenized for the model. */
export type MovementToken = string;

/** One (context -> next action) training pair derived from a trajectory. */
export type MovementTrainingExample = {
  /** Ordered context tokens leading up to the action (most recent last). */
  context: MovementToken[];
  /** The action the model should learn to emit next. */
  action: MovementActionLabel;
};

/** The action a movement model predicts. */
export type MovementActionLabel = {
  tool: string;
  summary: string;
};

/** Result of asking a trained model what to do next. */
export type MovementPrediction = {
  action: MovementActionLabel;
  /** Share of observations at the matched context (0..1). */
  confidence: number;
  /** Length of the context suffix that produced the match (0 = global prior). */
  matchedOrder: number;
  /** True when the model generalized by backing off to a shorter context. */
  backoff: boolean;
};

export type MovementTrainingConfig = {
  /** Longest context suffix the model conditions on. Default 3. */
  maxOrder?: number;
};

/** A backend knows how to turn a dataset into a trained, queryable model. */
export interface MovementModelBackend {
  readonly id: string;
  train(
    examples: MovementTrainingExample[],
    config?: MovementTrainingConfig,
  ): Promise<TrainedMovementModel>;
}

/** A trained model can predict the next movement and be serialized to disk. */
export interface TrainedMovementModel {
  readonly backendId: string;
  readonly exampleCount: number;
  readonly maxOrder: number;
  /** Predict the next action given recent context; undefined if never trained. */
  predict(context: MovementToken[]): MovementPrediction | undefined;
  serialize(): string;
}

const DEFAULT_MAX_ORDER = 3;

/** Tokenize a trajectory observation for use as movement context. */
export function observationToken(source: string): MovementToken {
  return `o:${source}`;
}

/** Tokenize a trajectory action for use as context or as a prediction target. */
export function actionToken(tool: string): MovementToken {
  return `a:${tool}`;
}

/**
 * Flatten trajectory spans into ordered (context -> next action) examples.
 *
 * Observations and actions within a span are merged into a single timeline
 * ordered by timestamp (observations before actions on ties, matching the
 * replay manifest's ordering). For every action we emit one example whose
 * context is the up-to-`maxOrder` preceding event tokens *from the same span*
 * (movements do not leak across spans).
 */
export function extractMovementExamples(
  spans: TrajectorySpan[],
  config: MovementTrainingConfig = {},
): MovementTrainingExample[] {
  const maxOrder = normalizeMaxOrder(config.maxOrder);
  const examples: MovementTrainingExample[] = [];

  for (const span of spans) {
    const timeline = [
      ...span.observations.map((observation) => ({
        ts: observation.ts,
        order: 0,
        token: observationToken(observation.source),
        action: undefined as MovementActionLabel | undefined,
      })),
      ...span.actions.map((action) => ({
        ts: action.ts,
        order: 1,
        token: actionToken(action.tool),
        action: { tool: action.tool, summary: action.summary },
      })),
    ].sort((a, b) => (a.ts !== b.ts ? a.ts - b.ts : a.order - b.order));

    const tokens: MovementToken[] = [];
    for (const event of timeline) {
      if (event.action) {
        examples.push({
          context: tokens.slice(Math.max(0, tokens.length - maxOrder)),
          action: event.action,
        });
      }
      tokens.push(event.token);
    }
  }

  return examples;
}

type ToolStat = {
  tool: string;
  count: number;
  summaries: Map<string, number>;
};

type ContextBucket = Map<string, ToolStat>;

type SerializedModel = {
  version: 1;
  backendId: string;
  maxOrder: number;
  exampleCount: number;
  contexts: Array<{
    key: string;
    tools: Array<{ tool: string; count: number; summaries: Array<[string, number]> }>;
  }>;
};

/**
 * Variable-order Markov movement model with Katz-style backoff.
 *
 * Every training example is recorded at each context order from `maxOrder` down
 * to 0 (the global prior). Prediction tries the longest available context and
 * backs off to shorter ones — which is exactly what lets the model *repeat*
 * exact recorded movements (long context hits) while still *generalizing* to
 * new-but-related contexts (backoff to a shorter shared suffix).
 */
export class NgramMovementModel implements TrainedMovementModel {
  readonly backendId: string;
  readonly maxOrder: number;
  readonly exampleCount: number;
  private readonly contexts: Map<string, ContextBucket>;

  private constructor(params: {
    backendId: string;
    maxOrder: number;
    exampleCount: number;
    contexts: Map<string, ContextBucket>;
  }) {
    this.backendId = params.backendId;
    this.maxOrder = params.maxOrder;
    this.exampleCount = params.exampleCount;
    this.contexts = params.contexts;
  }

  static fromExamples(
    backendId: string,
    examples: MovementTrainingExample[],
    maxOrder: number,
  ): NgramMovementModel {
    const contexts = new Map<string, ContextBucket>();
    for (const example of examples) {
      const trimmed = example.context.slice(Math.max(0, example.context.length - maxOrder));
      for (let order = 0; order <= trimmed.length; order += 1) {
        const key = contextKey(trimmed.slice(trimmed.length - order));
        recordObservation(contexts, key, example.action);
      }
    }
    return new NgramMovementModel({
      backendId,
      maxOrder,
      exampleCount: examples.length,
      contexts,
    });
  }

  predict(context: MovementToken[]): MovementPrediction | undefined {
    if (this.exampleCount === 0) {
      return undefined;
    }
    const trimmed = context.slice(Math.max(0, context.length - this.maxOrder));
    const requestedOrder = trimmed.length;
    for (let order = requestedOrder; order >= 0; order -= 1) {
      const key = contextKey(trimmed.slice(trimmed.length - order));
      const bucket = this.contexts.get(key);
      if (!bucket || bucket.size === 0) {
        continue;
      }
      const best = argmaxTool(bucket);
      const total = totalCount(bucket);
      return {
        action: { tool: best.tool, summary: argmaxSummary(best) },
        confidence: total === 0 ? 0 : best.count / total,
        matchedOrder: order,
        backoff: order < requestedOrder,
      };
    }
    return undefined;
  }

  serialize(): string {
    const contexts: SerializedModel["contexts"] = [];
    for (const [key, bucket] of this.contexts) {
      contexts.push({
        key,
        tools: [...bucket.values()].map((stat) => ({
          tool: stat.tool,
          count: stat.count,
          summaries: [...stat.summaries.entries()],
        })),
      });
    }
    const payload: SerializedModel = {
      version: 1,
      backendId: this.backendId,
      maxOrder: this.maxOrder,
      exampleCount: this.exampleCount,
      contexts,
    };
    return JSON.stringify(payload);
  }

  static deserialize(serialized: string): NgramMovementModel {
    const parsed = JSON.parse(serialized) as SerializedModel;
    const contexts = new Map<string, ContextBucket>();
    for (const entry of parsed.contexts) {
      const bucket: ContextBucket = new Map();
      for (const stat of entry.tools) {
        bucket.set(stat.tool, {
          tool: stat.tool,
          count: stat.count,
          summaries: new Map(stat.summaries),
        });
      }
      contexts.set(entry.key, bucket);
    }
    return new NgramMovementModel({
      backendId: parsed.backendId,
      maxOrder: parsed.maxOrder,
      exampleCount: parsed.exampleCount,
      contexts,
    });
  }
}

/**
 * Deterministic reference backend. No native deps, no randomness — the same
 * dataset always yields the same model, so CI can assert exact predictions.
 */
export class NgramMovementBackend implements MovementModelBackend {
  readonly id = "ngram-backoff";

  async train(
    examples: MovementTrainingExample[],
    config: MovementTrainingConfig = {},
  ): Promise<TrainedMovementModel> {
    const maxOrder = normalizeMaxOrder(config.maxOrder);
    return NgramMovementModel.fromExamples(this.id, examples, maxOrder);
  }
}

/** Convenience: train a movement model directly from trajectory spans. */
export async function trainMovementModel(
  backend: MovementModelBackend,
  spans: TrajectorySpan[],
  config: MovementTrainingConfig = {},
): Promise<TrainedMovementModel> {
  const examples = extractMovementExamples(spans, config);
  return backend.train(examples, config);
}

function normalizeMaxOrder(value: number | undefined): number {
  if (value === undefined || !Number.isFinite(value)) {
    return DEFAULT_MAX_ORDER;
  }
  return Math.max(0, Math.floor(value));
}

function contextKey(tokens: MovementToken[]): string {
  return tokens.join(">");
}

function recordObservation(
  contexts: Map<string, ContextBucket>,
  key: string,
  action: MovementActionLabel,
): void {
  let bucket = contexts.get(key);
  if (!bucket) {
    bucket = new Map();
    contexts.set(key, bucket);
  }
  let stat = bucket.get(action.tool);
  if (!stat) {
    stat = { tool: action.tool, count: 0, summaries: new Map() };
    bucket.set(action.tool, stat);
  }
  stat.count += 1;
  stat.summaries.set(action.summary, (stat.summaries.get(action.summary) ?? 0) + 1);
}

function argmaxTool(bucket: ContextBucket): ToolStat {
  let best: ToolStat | undefined;
  for (const stat of bucket.values()) {
    if (
      !best ||
      stat.count > best.count ||
      (stat.count === best.count && stat.tool < best.tool)
    ) {
      best = stat;
    }
  }
  // bucket is non-empty at every call site.
  return best as ToolStat;
}

function argmaxSummary(stat: ToolStat): string {
  let bestSummary = "";
  let bestCount = -1;
  for (const [summary, count] of stat.summaries) {
    if (count > bestCount || (count === bestCount && summary < bestSummary)) {
      bestSummary = summary;
      bestCount = count;
    }
  }
  return bestSummary;
}

function totalCount(bucket: ContextBucket): number {
  let total = 0;
  for (const stat of bucket.values()) {
    total += stat.count;
  }
  return total;
}
