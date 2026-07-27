import path from "node:path";
import { readJsonFile, writeJsonAtomic } from "../shared/fs.js";
import type { TrajectorySpan } from "../capture/trajectory.js";
import type { ReplayTimelineEvent } from "../capture/replay.js";
import {
  DeterministicMovementBackend,
  MovementBackendRegistry,
  buildMovementDataset,
  movementDatasetFromReplays,
  movementStepToken,
  type LocalMovementModelBackend,
  type MovementDataset,
  type MovementPrediction,
  type MovementSequence,
  type MovementStep,
  type TrainMovementModelOptions,
  type TrainedMovementModel,
} from "./movement-model.js";

export type TrainMovementModelParams = TrainMovementModelOptions & {
  /** Backend id to fit with; defaults to the deterministic in-process backend. */
  backendId?: string;
};

export type MovementGeneralizationReport = {
  /** Held-out sequences that had enough steps to seed a prefix + expect a tail. */
  evaluated: number;
  /** Fraction of predicted steps whose canonical token matched the ground truth. */
  stepAccuracy: number;
  /** Fraction of evaluated sequences the model reproduced end-to-end. */
  sequenceAccuracy: number;
  /** Predicted steps that came from a backed-off (generalized) context. */
  generalizedSteps: number;
  perSequence: {
    trajectoryId: string;
    expected: number;
    matched: number;
    exact: boolean;
  }[];
};

/**
 * Orchestrates the local-movement learning loop end-to-end in-process:
 * dataset build -> backend train -> persist -> inference -> generalization eval.
 * The heavy on-device training path stays in {@link LocalAppleSiliconTrainingRunner};
 * this service is the cloud/CI-runnable seam that validates the pipeline and
 * powers real inference once a model exists.
 */
export class MovementLearningService {
  private readonly registry: MovementBackendRegistry;
  private readonly defaultBackendId: string;

  constructor(options: { registry?: MovementBackendRegistry; defaultBackend?: LocalMovementModelBackend } = {}) {
    const fallback = options.defaultBackend ?? new DeterministicMovementBackend();
    this.registry = options.registry ?? new MovementBackendRegistry([fallback]);
    this.defaultBackendId = fallback.id;
  }

  datasetFromTrajectories(trajectories: TrajectorySpan[]): MovementDataset {
    return buildMovementDataset(trajectories);
  }

  datasetFromReplays(replays: { trajectoryIds: string[]; events: ReplayTimelineEvent[] }[]): MovementDataset {
    return movementDatasetFromReplays(replays);
  }

  async train(dataset: MovementDataset, params: TrainMovementModelParams = {}): Promise<TrainedMovementModel> {
    const backend = this.registry.get(params.backendId ?? this.defaultBackendId);
    return await backend.train(dataset, { order: params.order });
  }

  async predict(
    model: TrainedMovementModel,
    prefix: MovementStep[],
    options: { maxSteps?: number; backendId?: string } = {},
  ): Promise<MovementPrediction> {
    const backend = this.registry.get(options.backendId ?? model.backendId ?? this.defaultBackendId);
    return await backend.predict(model, { prefix, maxSteps: options.maxSteps });
  }

  async saveModel(rootDir: string, relativePath: string, model: TrainedMovementModel): Promise<string> {
    const target = path.join(rootDir, relativePath);
    await writeJsonAtomic(target, model);
    return target;
  }

  async loadModel(rootDir: string, relativePath: string): Promise<TrainedMovementModel | undefined> {
    return await readJsonFile<TrainedMovementModel | undefined>(path.join(rootDir, relativePath), undefined);
  }

  /**
   * Generalization eval harness: seed each held-out sequence with a prefix and
   * measure how faithfully the model reconstructs the remaining movements. This
   * quantifies both repetition (exact reproduction) and generalization (steps
   * produced via backed-off contexts).
   */
  async evaluateGeneralization(
    model: TrainedMovementModel,
    heldOut: MovementSequence[],
    options: { prefixLength?: number; backendId?: string } = {},
  ): Promise<MovementGeneralizationReport> {
    const prefixLength = Math.max(0, options.prefixLength ?? 1);
    const perSequence: MovementGeneralizationReport["perSequence"] = [];
    let totalExpected = 0;
    let totalMatched = 0;
    let generalizedSteps = 0;
    let exactSequences = 0;

    for (const sequence of heldOut) {
      if (sequence.steps.length <= prefixLength) {
        continue;
      }
      const prefix = sequence.steps.slice(0, prefixLength);
      const expected = sequence.steps.slice(prefixLength);
      const prediction = await this.predict(model, prefix, {
        maxSteps: expected.length,
        backendId: options.backendId,
      });

      let matched = 0;
      for (let i = 0; i < expected.length; i += 1) {
        const predicted = prediction.steps[i];
        if (predicted && movementStepToken(predicted) === movementStepToken(expected[i]!)) {
          matched += 1;
        }
        if (predicted?.source === "generalized") {
          generalizedSteps += 1;
        }
      }

      const exact = matched === expected.length;
      if (exact) {
        exactSequences += 1;
      }
      totalExpected += expected.length;
      totalMatched += matched;
      perSequence.push({ trajectoryId: sequence.trajectoryId, expected: expected.length, matched, exact });
    }

    const evaluated = perSequence.length;
    return {
      evaluated,
      stepAccuracy: totalExpected > 0 ? totalMatched / totalExpected : 0,
      sequenceAccuracy: evaluated > 0 ? exactSequences / evaluated : 0,
      generalizedSteps,
      perSequence,
    };
  }
}
