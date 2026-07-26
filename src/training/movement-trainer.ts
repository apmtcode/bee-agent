import path from "node:path";
import { readJsonFile, writeJsonAtomic } from "../shared/fs.js";
import { movementTokenKey, type MovementDataset, type MovementSequence } from "./movement-dataset.js";
import {
  createMovementBackend,
  type MovementModel,
  type MovementModelBackend,
  type MovementTrainingOptions,
} from "./movement-model.js";

/**
 * End-to-end movement-learning pipeline: train a local model on a movement
 * dataset, persist it, and evaluate replay fidelity (objective: repeat recorded
 * movements) and generalization (objective: perform new-but-related movements).
 *
 * The backend is pluggable and defaults to the deterministic mock so this runs
 * in the cloud / CI. Real on-device backends implement the same interface.
 */

export type PersistedMovementModel = {
  version: 1;
  id: string;
  backend: string;
  model: MovementModel;
};

/** Next-token accuracy over recorded sequences — how faithfully a model replays. */
export type ReplayFidelityReport = {
  sequenceCount: number;
  predictions: number;
  correct: number;
  accuracy: number;
};

/** Top-1 / top-k accuracy over held-out related sequences — generalization. */
export type GeneralizationReport = {
  k: number;
  sequenceCount: number;
  predictions: number;
  top1Correct: number;
  topKCorrect: number;
  top1Accuracy: number;
  topKAccuracy: number;
  /** Share of predictions that required backoff (structural generalization). */
  backoffRate: number;
};

export type MovementTrainingResult = {
  id: string;
  modelPath: string;
  model: MovementModel;
  fidelity: ReplayFidelityReport;
};

/**
 * Measure how well `model` reproduces the recorded movements: for every
 * position i>=1 in each sequence, predict the next token from the prefix and
 * compare to ground truth.
 */
export function evaluateReplayFidelity(
  backend: MovementModelBackend,
  model: MovementModel,
  sequences: MovementSequence[],
): ReplayFidelityReport {
  let predictions = 0;
  let correct = 0;
  for (const sequence of sequences) {
    for (let i = 1; i < sequence.tokens.length; i += 1) {
      predictions += 1;
      const prediction = backend.predictNext(model, sequence.tokens.slice(0, i));
      if (prediction && prediction.key === movementTokenKey(sequence.tokens[i]!)) {
        correct += 1;
      }
    }
  }
  return {
    sequenceCount: sequences.length,
    predictions,
    correct,
    accuracy: predictions === 0 ? 0 : correct / predictions,
  };
}

/**
 * Measure generalization to held-out but related sequences using top-1 and
 * top-k next-token accuracy, plus how often the prediction had to back off to a
 * shorter context (the mechanism by which structure transfers to novel slots).
 */
export function evaluateGeneralization(
  backend: MovementModelBackend,
  model: MovementModel,
  sequences: MovementSequence[],
  k = 3,
): GeneralizationReport {
  let predictions = 0;
  let top1Correct = 0;
  let topKCorrect = 0;
  let backoffPredictions = 0;
  for (const sequence of sequences) {
    for (let i = 1; i < sequence.tokens.length; i += 1) {
      predictions += 1;
      const expected = movementTokenKey(sequence.tokens[i]!);
      const ranked = backend.rank(model, sequence.tokens.slice(0, i));
      const top = ranked[0];
      if (top?.backoff) {
        backoffPredictions += 1;
      }
      if (top && top.key === expected) {
        top1Correct += 1;
      }
      if (ranked.slice(0, k).some((prediction) => prediction.key === expected)) {
        topKCorrect += 1;
      }
    }
  }
  return {
    k,
    sequenceCount: sequences.length,
    predictions,
    top1Correct,
    topKCorrect,
    top1Accuracy: predictions === 0 ? 0 : top1Correct / predictions,
    topKAccuracy: predictions === 0 ? 0 : topKCorrect / predictions,
    backoffRate: predictions === 0 ? 0 : backoffPredictions / predictions,
  };
}

export class MovementModelTrainer {
  private readonly backend: MovementModelBackend;

  constructor(
    private readonly rootDir: string,
    backend: MovementModelBackend = createMovementBackend("markov"),
  ) {
    this.backend = backend;
  }

  private modelPath(id: string): string {
    return path.join(this.rootDir, "movement-models", `${id}.json`);
  }

  async train(
    id: string,
    dataset: MovementDataset,
    options?: MovementTrainingOptions,
  ): Promise<MovementTrainingResult> {
    const model = this.backend.train(dataset, options);
    const modelPath = this.modelPath(id);
    const persisted: PersistedMovementModel = {
      version: 1,
      id,
      backend: this.backend.name,
      model,
    };
    await writeJsonAtomic(modelPath, persisted);
    const fidelity = evaluateReplayFidelity(this.backend, model, dataset.sequences);
    return { id, modelPath, model, fidelity };
  }

  async loadModel(id: string): Promise<MovementModel | undefined> {
    const persisted = await readJsonFile<PersistedMovementModel | undefined>(this.modelPath(id), undefined);
    return persisted?.model;
  }

  evaluateReplay(model: MovementModel, sequences: MovementSequence[]): ReplayFidelityReport {
    return evaluateReplayFidelity(this.backend, model, sequences);
  }

  evaluateGeneralization(model: MovementModel, sequences: MovementSequence[], k = 3): GeneralizationReport {
    return evaluateGeneralization(this.backend, model, sequences, k);
  }
}
