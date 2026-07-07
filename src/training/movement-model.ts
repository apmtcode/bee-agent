import type { TrajectoryAction, TrajectorySpan } from "../capture/trajectory.js";
import type { ExportedReplayManifest, ReviewedExportManifest } from "./export-manifest.js";

/**
 * In-process, deterministic local-model backend for the movement-learning
 * subsystem (standing objective #2c/#2d).
 *
 * The {@link LocalAppleSiliconTrainingRunner} emits *real* on-device training
 * commands (mlx/axolotl) that only run on the user's machine. This module is
 * the complementary seam: a pluggable {@link MovementModelBackend} that can
 * train and infer entirely in-process, so the capture → dataset → train →
 * replay → generalize loop can be validated in the cloud/CI against synthetic
 * event streams. The default backend is a deterministic n-gram (Markov) model
 * with back-off; a real small on-device model can implement the same interface.
 */

/** A canonical string form of a single recorded movement/action. */
export type MovementToken = string;

/** An ordered sequence of movement tokens for one trajectory/session. */
export type MovementSequence = {
  id: string;
  tokens: MovementToken[];
};

/** A tokenized, replayable training dataset. */
export type MovementDataset = {
  sequences: MovementSequence[];
};

export type MovementPrediction = {
  /** The most likely next token, or undefined when nothing was learned. */
  token: MovementToken | undefined;
  /** Probability of {@link token} within the backed-off context distribution. */
  probability: number;
  /** Context order (n-gram length) the prediction backed off to; -1 if none. */
  order: number;
  /** Up to five candidate tokens, most-likely first (deterministic tie-break). */
  alternatives: Array<{ token: MovementToken; probability: number }>;
};

export interface MovementModel {
  /** Name of the backend that produced this model. */
  readonly backend: string;
  /** Highest context order the model can condition on. */
  readonly order: number;
  /** Predict the next movement token given a (possibly novel) context. */
  predictNext(context: MovementToken[]): MovementPrediction;
  /** Autoregressively continue from a seed until the end/step budget. */
  generate(seed: MovementToken[], maxSteps: number): MovementToken[];
  /** All movement tokens seen during training (excludes the end sentinel). */
  vocabulary(): MovementToken[];
}

export type MovementTrainingOptions = {
  /** Maximum n-gram context order. Defaults to the backend's configured order. */
  order?: number;
};

export interface MovementModelBackend {
  readonly name: string;
  train(dataset: MovementDataset, options?: MovementTrainingOptions): MovementModel;
}

/** Sentinel appended to each training sequence so the model learns when to stop. */
export const MOVEMENT_END_TOKEN = "<end>";

const SEP = "";

function keyFor(tokens: MovementToken[]): string {
  return `${tokens.length}${SEP}${tokens.join(SEP)}`;
}

function slug(value: string): string {
  return value
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 48);
}

function stringField(metadata: Record<string, unknown> | undefined, key: string): string | undefined {
  const value = metadata?.[key];
  return typeof value === "string" ? value : undefined;
}

/**
 * Derive a stable, structured movement token from an action. The `:`-joined
 * parts (tool → gesture → target/direction) let the Markov back-off generalize
 * over related movements that share a suffix of context.
 */
export function tokenizeAction(action: {
  tool: string;
  summary: string;
  metadata?: Record<string, unknown>;
}): MovementToken {
  const parts: string[] = [slug(action.tool) || "action"];
  const gesture = stringField(action.metadata, "gesture");
  if (gesture) {
    parts.push(slug(gesture));
  }
  const target = stringField(action.metadata, "target") ?? stringField(action.metadata, "direction");
  if (target) {
    parts.push(slug(target));
  }
  if (parts.length === 1) {
    const summary = slug(action.summary);
    if (summary) {
      parts.push(summary);
    }
  }
  return parts.join(":");
}

function sortByTs<T extends { ts: number }>(items: readonly T[]): T[] {
  return [...items].sort((a, b) => a.ts - b.ts);
}

/** Build a dataset from raw trajectory spans (richest source — uses metadata). */
export function datasetFromTrajectories(trajectories: readonly TrajectorySpan[]): MovementDataset {
  const sequences = trajectories
    .map((trajectory) => ({
      id: trajectory.id,
      tokens: sortByTs<TrajectoryAction>(trajectory.actions).map((action) => tokenizeAction(action)),
    }))
    .filter((sequence) => sequence.tokens.length > 0);
  return { sequences };
}

/** Build a dataset from exported replay manifests (the reviewed-export path). */
export function datasetFromReplays(replays: readonly ExportedReplayManifest[]): MovementDataset {
  const sequences = replays
    .map((replay) => {
      const actionEvents = replay.events.filter(
        (event): event is Extract<ExportedReplayManifest["events"][number], { kind: "action" }> =>
          event.kind === "action",
      );
      return {
        id: replay.trajectoryIds[0] ?? replay.sessionId,
        tokens: sortByTs(actionEvents).map((event) => tokenizeAction({ tool: event.tool, summary: event.summary })),
      };
    })
    .filter((sequence) => sequence.tokens.length > 0);
  return { sequences };
}

/** Convenience: build a movement dataset directly from a reviewed export. */
export function datasetFromReviewedExport(manifest: ReviewedExportManifest): MovementDataset {
  return datasetFromReplays(manifest.replays);
}

class MarkovMovementModel implements MovementModel {
  constructor(
    readonly backend: string,
    readonly order: number,
    private readonly transitions: Map<string, Map<MovementToken, number>>,
    private readonly vocab: MovementToken[],
  ) {}

  predictNext(context: MovementToken[]): MovementPrediction {
    const maxOrder = Math.min(this.order, context.length);
    for (let n = maxOrder; n >= 0; n--) {
      const suffix = context.slice(context.length - n);
      const distribution = this.transitions.get(keyFor(suffix));
      if (!distribution || distribution.size === 0) {
        continue;
      }
      let total = 0;
      for (const count of distribution.values()) {
        total += count;
      }
      const ranked = [...distribution.entries()]
        .sort((a, b) => b[1] - a[1] || (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0))
        .map(([token, count]) => ({ token, probability: count / total }));
      const top = ranked[0];
      return {
        token: top.token,
        probability: top.probability,
        order: n,
        alternatives: ranked.slice(0, 5),
      };
    }
    return { token: undefined, probability: 0, order: -1, alternatives: [] };
  }

  generate(seed: MovementToken[], maxSteps: number): MovementToken[] {
    const output = [...seed];
    for (let step = 0; step < maxSteps; step++) {
      const prediction = this.predictNext(output);
      if (!prediction.token || prediction.token === MOVEMENT_END_TOKEN) {
        break;
      }
      output.push(prediction.token);
    }
    return output;
  }

  vocabulary(): MovementToken[] {
    return [...this.vocab];
  }
}

/**
 * Deterministic n-gram (Markov) backend with back-off. Learns next-token
 * distributions for every context length 0..order and, at inference time,
 * conditions on the longest matching context suffix — which is what lets it
 * generalize to novel prefixes whose *tail* it has seen.
 */
export class MarkovMovementBackend implements MovementModelBackend {
  constructor(
    private readonly defaultOrder = 3,
    readonly name = "markov",
  ) {}

  train(dataset: MovementDataset, options?: MovementTrainingOptions): MovementModel {
    const order = Math.max(0, options?.order ?? this.defaultOrder);
    const transitions = new Map<string, Map<MovementToken, number>>();
    const vocab = new Set<MovementToken>();

    for (const sequence of dataset.sequences) {
      for (const token of sequence.tokens) {
        vocab.add(token);
      }
      const full = [...sequence.tokens, MOVEMENT_END_TOKEN];
      for (let i = 0; i < full.length; i++) {
        const next = full[i]!;
        const maxN = Math.min(order, i);
        for (let n = 0; n <= maxN; n++) {
          const key = keyFor(full.slice(i - n, i));
          let distribution = transitions.get(key);
          if (!distribution) {
            distribution = new Map();
            transitions.set(key, distribution);
          }
          distribution.set(next, (distribution.get(next) ?? 0) + 1);
        }
      }
    }

    return new MarkovMovementModel(this.name, order, transitions, [...vocab].sort());
  }
}

export type MovementBackendName = "markov" | "most-frequent";

/**
 * Backend registry — the pluggable seam. `markov` is the default n-gram model;
 * `most-frequent` is an order-0 control baseline (always predicts the globally
 * most common movement). A real on-device small-model backend can be registered
 * here without touching callers.
 */
export function createMovementBackend(
  name: MovementBackendName,
  options?: { order?: number },
): MovementModelBackend {
  switch (name) {
    case "markov":
      return new MarkovMovementBackend(options?.order ?? 3, "markov");
    case "most-frequent":
      return new MarkovMovementBackend(0, "most-frequent");
  }
}

export type MovementEvalResult = {
  backend: string;
  sequences: number;
  predictions: number;
  correct: number;
  accuracy: number;
  perSequence: Array<{ id: string; predictions: number; correct: number; accuracy: number }>;
};

/**
 * Teacher-forced next-token accuracy over held-out sequences — the
 * generalization eval harness. For each position it predicts from the true
 * prefix and compares to the observed next movement.
 */
export function evaluateMovementModel(
  model: MovementModel,
  heldOut: readonly MovementSequence[],
): MovementEvalResult {
  let predictions = 0;
  let correct = 0;
  const perSequence: MovementEvalResult["perSequence"] = [];

  for (const sequence of heldOut) {
    let sequencePredictions = 0;
    let sequenceCorrect = 0;
    for (let i = 1; i < sequence.tokens.length; i++) {
      const prediction = model.predictNext(sequence.tokens.slice(0, i));
      sequencePredictions++;
      if (prediction.token === sequence.tokens[i]) {
        sequenceCorrect++;
      }
    }
    perSequence.push({
      id: sequence.id,
      predictions: sequencePredictions,
      correct: sequenceCorrect,
      accuracy: sequencePredictions === 0 ? 0 : sequenceCorrect / sequencePredictions,
    });
    predictions += sequencePredictions;
    correct += sequenceCorrect;
  }

  return {
    backend: model.backend,
    sequences: heldOut.length,
    predictions,
    correct,
    accuracy: predictions === 0 ? 0 : correct / predictions,
    perSequence,
  };
}

export type ReplayFidelityResult = {
  matched: number;
  total: number;
  fidelity: number;
  generated: MovementToken[];
};

/**
 * Measure how faithfully the model reproduces (repeats) a recorded movement:
 * seed it with the first token, autoregressively generate, and compare
 * position-by-position to the original sequence.
 */
export function replayFidelity(model: MovementModel, sequence: MovementSequence): ReplayFidelityResult {
  if (sequence.tokens.length === 0) {
    return { matched: 0, total: 0, fidelity: 1, generated: [] };
  }
  const generated = model.generate([sequence.tokens[0]!], sequence.tokens.length * 2 + 4);
  let matched = 0;
  for (let i = 0; i < sequence.tokens.length; i++) {
    if (generated[i] === sequence.tokens[i]) {
      matched++;
    }
  }
  return {
    matched,
    total: sequence.tokens.length,
    fidelity: matched / sequence.tokens.length,
    generated,
  };
}
