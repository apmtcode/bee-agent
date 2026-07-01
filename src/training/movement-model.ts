import type { ExportedReplayManifest } from "./export-manifest.js";
import type { TrajectorySpan } from "../capture/trajectory.js";

/**
 * Movement-model subsystem (standing objective #2, parts c + d).
 *
 * This module makes the "post-train a local model on recorded movements and let
 * it generalize" capability *real and testable in the cloud*. The heavy on-device
 * runtimes (mlx / axolotl) are launched by {@link LocalAppleSiliconTrainingRunner}
 * on the user's machine; here we provide a pluggable backend seam plus a
 * deterministic in-process backend so the whole pipeline
 * (capture → dataset → train → infer → generalize) can be exercised with
 * synthetic data without any real OS input or GPU.
 *
 * A backend consumes a {@link MovementDataset} (ordered movement steps distilled
 * from reviewed trajectories/replays) and returns a {@link MovementModel} that
 * predicts the next movement given recent context. The reference
 * {@link NgramMovementBackend} learns a back-off n-gram over movement *channels*,
 * which is what lets it repeat recorded sequences exactly and generalise to new
 * but related sequences (an unseen full context still resolves via a shorter
 * matching suffix).
 */

export type MovementStepType = "observation" | "action";

/** The atomic unit a movement model learns over. */
export type MovementStep = {
  ts: number;
  type: MovementStepType;
  /** The observation source or the action tool — the "what kind of movement" key. */
  channel: string;
  summary: string;
};

/** An ordered run of movement steps distilled from a single trajectory/replay. */
export type MovementSequence = {
  id: string;
  steps: MovementStep[];
};

export type MovementDataset = {
  version: 1;
  sequences: MovementSequence[];
};

export type MovementPrediction = {
  step: { type: MovementStepType; channel: string; summary: string };
  /** Share of the matched context's observations that agreed with this step (0..1). */
  confidence: number;
  /** n-gram order actually used to make the prediction (0 = unconditional base rate). */
  matchedOrder: number;
  /** Total observations backing the matched context. */
  support: number;
  /**
   * True when the full available context (up to `maxOrder`) was not seen verbatim
   * in training and the model had to back off to a shorter suffix — i.e. the model
   * generalised rather than recalled.
   */
  generalized: boolean;
};

export type MovementTrainingOptions = {
  /** Longest context (in steps) the model conditions on. Defaults to 3. */
  maxOrder?: number;
};

export type MovementModelSnapshot = {
  version: 1;
  backendId: string;
  maxOrder: number;
  /** Per-order back-off tables: contextKey → list of [stepSignature, count]. */
  tables: Array<{
    order: number;
    entries: Array<[string, Array<[string, number]>]>;
  }>;
};

export interface MovementModel {
  readonly backendId: string;
  readonly maxOrder: number;
  /** Predict the movement most likely to follow `context`, or undefined if untrained. */
  predictNext(context: MovementStep[]): MovementPrediction | undefined;
  serialize(): MovementModelSnapshot;
}

export interface MovementModelBackend {
  readonly id: string;
  train(dataset: MovementDataset, options?: MovementTrainingOptions): MovementModel;
}

const CONTEXT_SEPARATOR = " > ";
const DEFAULT_MAX_ORDER = 3;

function channelToken(step: MovementStep): string {
  return `${step.type}:${step.channel}`;
}

function stepSignature(step: MovementStep): string {
  return JSON.stringify([step.type, step.channel, step.summary]);
}

function stepFromSignature(signature: string): MovementPrediction["step"] {
  const [type, channel, summary] = JSON.parse(signature) as [MovementStepType, string, string];
  return { type, channel, summary };
}

function contextKey(tokens: string[]): string {
  return tokens.join(CONTEXT_SEPARATOR);
}

/**
 * Deterministic back-off n-gram movement model.
 *
 * Standing in for a real on-device small model, it is fully reproducible (no clock
 * or randomness), serialisable to an artifact, and — via suffix back-off —
 * genuinely generalises to unseen-but-related movement sequences.
 */
export class NgramMovementModel implements MovementModel {
  readonly backendId: string;
  readonly maxOrder: number;
  /** order → contextKey → signature → count */
  private readonly tables: Map<number, Map<string, Map<string, number>>>;

  private constructor(
    backendId: string,
    maxOrder: number,
    tables: Map<number, Map<string, Map<string, number>>>,
  ) {
    this.backendId = backendId;
    this.maxOrder = maxOrder;
    this.tables = tables;
  }

  static train(
    backendId: string,
    dataset: MovementDataset,
    options?: MovementTrainingOptions,
  ): NgramMovementModel {
    const maxOrder = Math.max(0, Math.trunc(options?.maxOrder ?? DEFAULT_MAX_ORDER));
    const tables = new Map<number, Map<string, Map<string, number>>>();
    for (let order = 0; order <= maxOrder; order += 1) {
      tables.set(order, new Map());
    }

    for (const sequence of dataset.sequences) {
      const tokens = sequence.steps.map(channelToken);
      for (let index = 0; index < sequence.steps.length; index += 1) {
        const signature = stepSignature(sequence.steps[index]);
        const maxK = Math.min(maxOrder, index);
        for (let k = 0; k <= maxK; k += 1) {
          const key = contextKey(tokens.slice(index - k, index));
          const table = tables.get(k)!;
          let distribution = table.get(key);
          if (!distribution) {
            distribution = new Map();
            table.set(key, distribution);
          }
          distribution.set(signature, (distribution.get(signature) ?? 0) + 1);
        }
      }
    }

    return new NgramMovementModel(backendId, maxOrder, tables);
  }

  predictNext(context: MovementStep[]): MovementPrediction | undefined {
    const tokens = context.map(channelToken);
    const maxK = Math.min(this.maxOrder, tokens.length);
    for (let k = maxK; k >= 0; k -= 1) {
      const key = contextKey(tokens.slice(tokens.length - k));
      const distribution = this.tables.get(k)?.get(key);
      if (!distribution || distribution.size === 0) {
        continue;
      }
      let total = 0;
      for (const count of distribution.values()) {
        total += count;
      }
      // Deterministic argmax: highest count, ties broken by signature order.
      const best = [...distribution.entries()].sort((a, b) =>
        b[1] !== a[1] ? b[1] - a[1] : a[0] < b[0] ? -1 : 1,
      )[0];
      return {
        step: stepFromSignature(best[0]),
        confidence: best[1] / total,
        matchedOrder: k,
        support: total,
        generalized: k < maxK,
      };
    }
    return undefined;
  }

  serialize(): MovementModelSnapshot {
    const tables: MovementModelSnapshot["tables"] = [];
    for (let order = 0; order <= this.maxOrder; order += 1) {
      const table = this.tables.get(order);
      if (!table) {
        continue;
      }
      const entries: Array<[string, Array<[string, number]>]> = [...table.entries()]
        .sort((a, b) => (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0))
        .map(([key, distribution]) => [key, [...distribution.entries()]]);
      tables.push({ order, entries });
    }
    return { version: 1, backendId: this.backendId, maxOrder: this.maxOrder, tables };
  }

  static restore(snapshot: MovementModelSnapshot): NgramMovementModel {
    const tables = new Map<number, Map<string, Map<string, number>>>();
    for (let order = 0; order <= snapshot.maxOrder; order += 1) {
      tables.set(order, new Map());
    }
    for (const { order, entries } of snapshot.tables) {
      const table = tables.get(order) ?? new Map();
      for (const [key, distribution] of entries) {
        table.set(key, new Map(distribution));
      }
      tables.set(order, table);
    }
    return new NgramMovementModel(snapshot.backendId, snapshot.maxOrder, tables);
  }
}

/** The default, cloud-safe backend. Deterministic; requires no GPU or OS input. */
export class NgramMovementBackend implements MovementModelBackend {
  readonly id = "ngram-backoff";

  train(dataset: MovementDataset, options?: MovementTrainingOptions): MovementModel {
    return NgramMovementModel.train(this.id, dataset, options);
  }
}

/** Distil a movement sequence from a trajectory span (time-ordered obs + actions). */
export function movementSequenceFromTrajectory(trajectory: TrajectorySpan): MovementSequence {
  const steps: MovementStep[] = [
    ...trajectory.observations.map<MovementStep>((observation) => ({
      ts: observation.ts,
      type: "observation",
      channel: observation.source,
      summary: observation.summary,
    })),
    ...trajectory.actions.map<MovementStep>((action) => ({
      ts: action.ts,
      type: "action",
      channel: action.tool,
      summary: action.summary,
    })),
  ].sort((a, b) => a.ts - b.ts);
  return { id: trajectory.id, steps };
}

/** Distil a movement sequence from a reviewed replay manifest (ignores transcript). */
export function movementSequenceFromReplay(replay: ExportedReplayManifest): MovementSequence {
  const steps: MovementStep[] = replay.events
    .filter(
      (event): event is Extract<ExportedReplayManifest["events"][number], { kind: "observation" | "action" }> =>
        event.kind === "observation" || event.kind === "action",
    )
    .map((event) =>
      event.kind === "observation"
        ? { ts: event.ts, type: "observation", channel: event.source, summary: event.summary }
        : { ts: event.ts, type: "action", channel: event.tool, summary: event.summary },
    );
  return { id: replay.trajectoryIds[0] ?? replay.sessionId, steps };
}

export function movementDatasetFromTrajectories(trajectories: TrajectorySpan[]): MovementDataset {
  return {
    version: 1,
    sequences: trajectories.map(movementSequenceFromTrajectory).filter((sequence) => sequence.steps.length > 0),
  };
}

export function movementDatasetFromReplays(replays: ExportedReplayManifest[]): MovementDataset {
  return {
    version: 1,
    sequences: replays.map(movementSequenceFromReplay).filter((sequence) => sequence.steps.length > 0),
  };
}

export type MovementEvalResult = {
  totalPredictions: number;
  /** Predictions whose type+channel matched the held-out next movement (generalisation metric). */
  correct: number;
  /** Predictions that also matched the exact summary (recall metric). */
  exactCorrect: number;
  accuracy: number;
  exactAccuracy: number;
  /** Of `totalPredictions`, how many required back-off (the model generalised). */
  generalizedPredictions: number;
  /** Of those, how many were still correct. */
  generalizedCorrect: number;
  perSequence: Array<{ id: string; predictions: number; correct: number }>;
};

/**
 * Generalisation eval harness: replay each held-out sequence step-by-step, ask the
 * model to predict the next movement from the preceding steps, and score how often
 * it recovers the right *kind* of movement (type+channel) and the exact step.
 */
export function evaluateMovementModel(
  model: MovementModel,
  heldOut: MovementSequence[],
  options?: { minContext?: number },
): MovementEvalResult {
  const minContext = Math.max(0, options?.minContext ?? 1);
  let totalPredictions = 0;
  let correct = 0;
  let exactCorrect = 0;
  let generalizedPredictions = 0;
  let generalizedCorrect = 0;
  const perSequence: MovementEvalResult["perSequence"] = [];

  for (const sequence of heldOut) {
    let seqPredictions = 0;
    let seqCorrect = 0;
    for (let index = minContext; index < sequence.steps.length; index += 1) {
      const prediction = model.predictNext(sequence.steps.slice(0, index));
      if (!prediction) {
        continue;
      }
      const expected = sequence.steps[index];
      const kindMatch = prediction.step.type === expected.type && prediction.step.channel === expected.channel;
      const exactMatch = kindMatch && prediction.step.summary === expected.summary;
      totalPredictions += 1;
      seqPredictions += 1;
      if (kindMatch) {
        correct += 1;
        seqCorrect += 1;
      }
      if (exactMatch) {
        exactCorrect += 1;
      }
      if (prediction.generalized) {
        generalizedPredictions += 1;
        if (kindMatch) {
          generalizedCorrect += 1;
        }
      }
    }
    perSequence.push({ id: sequence.id, predictions: seqPredictions, correct: seqCorrect });
  }

  return {
    totalPredictions,
    correct,
    exactCorrect,
    accuracy: totalPredictions === 0 ? 0 : correct / totalPredictions,
    exactAccuracy: totalPredictions === 0 ? 0 : exactCorrect / totalPredictions,
    generalizedPredictions,
    generalizedCorrect,
    perSequence,
  };
}
