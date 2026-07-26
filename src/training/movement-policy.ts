import type { TrajectoryAction } from "../capture/trajectory.js";

/**
 * Local-movement learning: in-process, deterministic learn -> infer.
 *
 * bee-agent runs in Anthropic's cloud with no access to the user's real
 * machine, so the on-device recording/training in {@link ./runner.ts} (which
 * emits MLX/axolotl launch plans) cannot execute here. This module provides a
 * fully in-process, backend-pluggable policy that *can* run in the cloud/CI: it
 * learns transition statistics from recorded movement trajectories and predicts
 * the next movement. It powers two capabilities from standing objective #2:
 *   (c) "post-train a local model ... to repeat the recorded movements", and
 *   (d) "generalize to perform new but related movements".
 *
 * The default {@link MarkovMovementBackend} is a deterministic variable-order
 * back-off Markov model. Two abstraction channels give it generalization:
 *   - a *specific* channel keyed on tool|gesture|direction|target, and
 *   - a *general* channel keyed on tool|gesture|direction (target dropped).
 * On a context it has seen verbatim it replays the recorded next action; on a
 * novel-but-related context (e.g. the same gesture pattern over an unseen
 * target) the specific channel misses and it backs off to the general channel,
 * still predicting the right *kind* of movement. Everything is deterministic
 * (arg-max with lexicographic tie-break) so tests are reproducible.
 *
 * The {@link MovementPolicyBackend} interface is the seam for a real on-device
 * small model later; the Markov backend is the mock that keeps cloud/CI green.
 */

export type MovementFeature = {
  /** Full-fidelity token: tool|gesture|direction|target. */
  specific: string;
  /** Abstract token: tool|gesture|direction (target dropped) for back-off. */
  general: string;
  /** The originating tool, retained for reporting. */
  tool: string;
};

export type MovementSequence = {
  id: string;
  features: MovementFeature[];
};

export type MovementDataset = {
  version: 1;
  /** Maximum context window length the model may condition on. */
  order: number;
  sequences: MovementSequence[];
};

export type MovementPredictionLevel = "specific" | "general" | "fallback" | "unknown";

export type MovementPrediction = {
  /** Predicted next feature token (specific when available, else general). */
  next: string;
  /** Probability mass of the winning token within its channel, 0..1. */
  confidence: number;
  /** Which back-off channel produced the prediction. */
  level: MovementPredictionLevel;
  /** Context length (in features) the prediction actually conditioned on. */
  contextLength: number;
  /** Ranked runner-up tokens from the same channel. */
  alternatives: Array<{ token: string; probability: number }>;
};

/** Serializable trained artifact. Pluggable backends may extend the shape. */
export type MovementModel = {
  backend: string;
  order: number;
  /** context signature -> next specific token -> count. */
  specific: Record<string, Record<string, number>>;
  /** context signature -> next general token -> count. */
  general: Record<string, Record<string, number>>;
  /** Most frequent specific token overall, used as the last-resort fallback. */
  fallback: string | undefined;
  /** Distinct specific tokens observed, sorted. */
  vocabulary: string[];
  sampleCount: number;
};

export interface MovementPolicyBackend {
  readonly id: string;
  train(dataset: MovementDataset): Promise<MovementModel>;
  predict(model: MovementModel, context: MovementFeature[]): MovementPrediction;
}

const CONTEXT_SEPARATOR = " >> ";

function readStringMetadata(metadata: Record<string, unknown> | undefined, key: string): string | undefined {
  const value = metadata?.[key];
  return typeof value === "string" && value.length > 0 ? value : undefined;
}

/** Canonicalize a recorded action into a two-channel movement feature. */
export function encodeActionFeature(action: TrajectoryAction): MovementFeature {
  const gesture = readStringMetadata(action.metadata, "gesture");
  const direction = readStringMetadata(action.metadata, "direction");
  const target = readStringMetadata(action.metadata, "target");
  const generalParts = [action.tool, gesture, direction].filter((part): part is string => Boolean(part));
  const specificParts = target ? [...generalParts, `@${target}`] : [...generalParts];
  return {
    tool: action.tool,
    general: generalParts.join("|"),
    specific: specificParts.join("|"),
  };
}

export type MovementTrajectoryInput = {
  id: string;
  actions: TrajectoryAction[];
};

/**
 * Convert recorded trajectories into a windowed training dataset. Actions are
 * sorted by timestamp so out-of-order capture does not corrupt the sequence.
 */
export function buildMovementDataset(params: {
  trajectories: MovementTrajectoryInput[];
  order?: number;
}): MovementDataset {
  const order = Math.max(1, params.order ?? 2);
  const sequences: MovementSequence[] = params.trajectories.map((trajectory) => ({
    id: trajectory.id,
    features: [...trajectory.actions]
      .sort((a, b) => a.ts - b.ts)
      .map((action) => encodeActionFeature(action)),
  }));
  return { version: 1, order, sequences };
}

function signature(features: MovementFeature[], channel: "specific" | "general"): string {
  return features.map((feature) => feature[channel]).join(CONTEXT_SEPARATOR);
}

function increment(table: Record<string, Record<string, number>>, key: string, token: string): void {
  const row = (table[key] ??= {});
  row[token] = (row[token] ?? 0) + 1;
}

/** Deterministic variable-order back-off Markov policy. */
export class MarkovMovementBackend implements MovementPolicyBackend {
  readonly id = "markov-backoff-v1";

  async train(dataset: MovementDataset): Promise<MovementModel> {
    const specific: Record<string, Record<string, number>> = {};
    const general: Record<string, Record<string, number>> = {};
    const totals: Record<string, number> = {};
    let sampleCount = 0;

    for (const sequence of dataset.sequences) {
      const { features } = sequence;
      for (let i = 0; i < features.length; i += 1) {
        const next = features[i];
        if (!next) {
          continue;
        }
        totals[next.specific] = (totals[next.specific] ?? 0) + 1;
        // Record every context length from 1..order that fits before position i,
        // giving a variable-order model with natural back-off.
        const maxK = Math.min(dataset.order, i);
        for (let k = 1; k <= maxK; k += 1) {
          const context = features.slice(i - k, i);
          increment(specific, signature(context, "specific"), next.specific);
          increment(general, signature(context, "general"), next.general);
          sampleCount += 1;
        }
      }
    }

    const fallback = argmaxToken(totals);
    const vocabulary = Object.keys(totals).sort();
    return {
      backend: this.id,
      order: dataset.order,
      specific,
      general,
      fallback,
      vocabulary,
      sampleCount,
    };
  }

  predict(model: MovementModel, context: MovementFeature[]): MovementPrediction {
    const maxK = Math.min(model.order, context.length);
    // Longest-match first on the specific channel (verbatim replay), then the
    // same on the general channel (generalization), then global fallback.
    for (let k = maxK; k >= 1; k -= 1) {
      const window = context.slice(context.length - k);
      const row = model.specific[signature(window, "specific")];
      if (row) {
        return toPrediction(row, "specific", k);
      }
    }
    for (let k = maxK; k >= 1; k -= 1) {
      const window = context.slice(context.length - k);
      const row = model.general[signature(window, "general")];
      if (row) {
        return toPrediction(row, "general", k);
      }
    }
    if (model.fallback !== undefined) {
      return { next: model.fallback, confidence: 0, level: "fallback", contextLength: 0, alternatives: [] };
    }
    return { next: "", confidence: 0, level: "unknown", contextLength: 0, alternatives: [] };
  }
}

function argmaxToken(row: Record<string, number>): string | undefined {
  let best: string | undefined;
  let bestCount = -1;
  for (const token of Object.keys(row).sort()) {
    const count = row[token] ?? 0;
    if (count > bestCount) {
      best = token;
      bestCount = count;
    }
  }
  return best;
}

function toPrediction(
  row: Record<string, number>,
  level: "specific" | "general",
  contextLength: number,
): MovementPrediction {
  const total = Object.values(row).reduce((sum, count) => sum + count, 0);
  const ranked = Object.keys(row)
    .sort((a, b) => {
      const diff = (row[b] ?? 0) - (row[a] ?? 0);
      return diff !== 0 ? diff : a.localeCompare(b);
    })
    .map((token) => ({ token, probability: total > 0 ? (row[token] ?? 0) / total : 0 }));
  const top = ranked[0];
  return {
    next: top?.token ?? "",
    confidence: top?.probability ?? 0,
    level,
    contextLength,
    alternatives: ranked.slice(1),
  };
}

/**
 * Autoregressively roll the policy forward from a seed context to reproduce or
 * extend a movement sequence. Purely token-level (no OS actuation) — the real
 * device replay path lives in {@link ../capture/replay-service.ts}.
 */
export function rolloutMovementPolicy(params: {
  backend: MovementPolicyBackend;
  model: MovementModel;
  seed: MovementFeature[];
  steps: number;
}): Array<{ token: string; level: MovementPredictionLevel; confidence: number }> {
  const generated: Array<{ token: string; level: MovementPredictionLevel; confidence: number }> = [];
  const context: MovementFeature[] = [...params.seed];
  for (let step = 0; step < params.steps; step += 1) {
    const prediction = params.backend.predict(params.model, context);
    if (prediction.level === "unknown" || prediction.next === "") {
      break;
    }
    generated.push({ token: prediction.next, level: prediction.level, confidence: prediction.confidence });
    // Feed the prediction back in as both channels so back-off keeps working.
    context.push({ specific: prediction.next, general: prediction.next, tool: prediction.next.split("|")[0] ?? "" });
  }
  return generated;
}

export type MovementEvalResult = {
  /** Predictions attempted (positions with a non-empty preceding context). */
  total: number;
  /** Top-1 correct against the recorded next specific token. */
  correct: number;
  accuracy: number;
  /** Correct via the specific (verbatim) channel. */
  specificHits: number;
  /** Correct via the general (generalized) channel — the generalization signal. */
  generalHits: number;
  fallbackHits: number;
};

/**
 * Held-out generalization eval harness: for each recorded position, predict the
 * next feature from its preceding context and score against the ground truth.
 * A prediction is "correct" if it matches the recorded specific token, or if the
 * general-channel prediction matches the recorded general token (right kind of
 * movement over an unseen target) — the fidelity signal for objective #2(d).
 */
export function evaluateMovementPolicy(params: {
  backend: MovementPolicyBackend;
  model: MovementModel;
  heldOut: MovementSequence[];
}): MovementEvalResult {
  let total = 0;
  let correct = 0;
  let specificHits = 0;
  let generalHits = 0;
  let fallbackHits = 0;

  for (const sequence of params.heldOut) {
    for (let i = 1; i < sequence.features.length; i += 1) {
      const expected = sequence.features[i];
      if (!expected) {
        continue;
      }
      const context = sequence.features.slice(0, i);
      const prediction = params.backend.predict(params.model, context);
      total += 1;
      const specificMatch = prediction.level === "specific" && prediction.next === expected.specific;
      const generalMatch =
        (prediction.level === "general" || prediction.level === "specific") && prediction.next === expected.general;
      if (specificMatch) {
        correct += 1;
        specificHits += 1;
      } else if (generalMatch) {
        correct += 1;
        generalHits += 1;
      } else if (prediction.level === "fallback" && prediction.next === expected.specific) {
        correct += 1;
        fallbackHits += 1;
      }
    }
  }

  return {
    total,
    correct,
    accuracy: total > 0 ? correct / total : 0,
    specificHits,
    generalHits,
    fallbackHits,
  };
}
