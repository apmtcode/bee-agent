// Generalization eval harness for the local-movement learning subsystem
// (objective #2d). Measures two things the objective cares about:
//   1. Reproduction — can the trained model repeat movements it was trained on?
//   2. Generalization — does it predict the right next movement on held-out but
//      related sequences it never saw during training?
//
// Everything is deterministic (no RNG, no clock): the train/test split is a
// stride over the sequence list, and the backend itself is deterministic. That
// makes the harness a stable regression gate in the cloud/CI.

import type {
  MovementContext,
  MovementDataset,
  MovementModelBackend,
  MovementSequence,
  MovementStep,
  MovementTrainOptions,
  TrainedMovementModel,
} from "./movement-model.js";
import { movementToken } from "./movement-model.js";

export type GeneralizationSplit = {
  train: MovementSequence[];
  test: MovementSequence[];
};

export type GeneralizationEvalOptions = {
  /** Every `holdoutStride`-th sequence is held out for testing (default 3). */
  holdoutStride?: number;
  /** How many leading steps of a test sequence to reveal before predicting the rest (default 1). */
  seedSteps?: number;
  /** Training options forwarded to the backend. */
  train?: MovementTrainOptions;
};

export type GeneralizationReport = {
  backend: string;
  trainSequences: number;
  testSequences: number;
  /** Fraction of training sequences the model reproduces exactly from their seed. */
  reproductionRate: number;
  /** Teacher-forced next-step top-1 accuracy on held-out sequences. */
  heldOutStepAccuracy: number;
  /** Fraction of held-out predictions that required back-off to a broader context. */
  generalizationRate: number;
  /** Fraction of held-out sequences whose full continuation is generated exactly. */
  heldOutSequenceAccuracy: number;
  evaluatedSteps: number;
};

/** Deterministic stride split so results are reproducible run-to-run. */
export function splitForGeneralization(
  dataset: MovementDataset,
  holdoutStride = 3,
): GeneralizationSplit {
  const stride = Math.max(2, holdoutStride);
  const train: MovementSequence[] = [];
  const test: MovementSequence[] = [];
  dataset.sequences.forEach((sequence, index) => {
    if ((index + 1) % stride === 0) {
      test.push(sequence);
    } else {
      train.push(sequence);
    }
  });
  // Never leave the test set starved when the dataset is tiny.
  if (test.length === 0 && train.length > 1) {
    test.push(train.pop()!);
  }
  return { train, test };
}

export async function evaluateGeneralization(
  backend: MovementModelBackend,
  dataset: MovementDataset,
  options?: GeneralizationEvalOptions,
): Promise<GeneralizationReport> {
  const seedSteps = Math.max(1, options?.seedSteps ?? 1);
  const { train, test } = splitForGeneralization(dataset, options?.holdoutStride ?? 3);
  const model = await backend.train({ version: 1, sequences: train }, options?.train);

  const reproductionRate = train.length === 0 ? 0 : rate(train, (sequence) => reproduces(model, sequence, seedSteps));

  let correctSteps = 0;
  let evaluatedSteps = 0;
  let generalizedSteps = 0;
  let exactSequences = 0;

  for (const sequence of test) {
    if (sequence.steps.length <= seedSteps) {
      continue;
    }
    // Teacher-forced next-step accuracy: at each position feed the TRUE prefix.
    let sequenceExact = true;
    for (let i = seedSteps; i < sequence.steps.length; i += 1) {
      const context: MovementContext = {
        context: sequence.context,
        history: sequence.steps.slice(0, i),
      };
      const prediction = model.predictNext(context);
      evaluatedSteps += 1;
      if (prediction?.generalized) {
        generalizedSteps += 1;
      }
      if (prediction && movementToken(prediction.step) === movementToken(sequence.steps[i])) {
        correctSteps += 1;
      } else {
        sequenceExact = false;
      }
    }
    // Free-running generation accuracy: seed then roll out.
    const generated = model.generate(
      { context: sequence.context, history: sequence.steps.slice(0, seedSteps) },
      sequence.steps.length - seedSteps,
    );
    if (sequenceExact && sequencesMatch(generated, sequence.steps.slice(seedSteps))) {
      exactSequences += 1;
    }
  }

  const evaluatedSequences = test.filter((sequence) => sequence.steps.length > seedSteps).length;
  return {
    backend: backend.name,
    trainSequences: train.length,
    testSequences: test.length,
    reproductionRate,
    heldOutStepAccuracy: evaluatedSteps === 0 ? 0 : correctSteps / evaluatedSteps,
    generalizationRate: evaluatedSteps === 0 ? 0 : generalizedSteps / evaluatedSteps,
    heldOutSequenceAccuracy: evaluatedSequences === 0 ? 0 : exactSequences / evaluatedSequences,
    evaluatedSteps,
  };
}

function reproduces(model: TrainedMovementModel, sequence: MovementSequence, seedSteps: number): boolean {
  if (sequence.steps.length <= seedSteps) {
    return true;
  }
  const generated = model.generate(
    { context: sequence.context, history: sequence.steps.slice(0, seedSteps) },
    sequence.steps.length - seedSteps,
  );
  return sequencesMatch(generated, sequence.steps.slice(seedSteps));
}

function sequencesMatch(a: MovementStep[], b: MovementStep[]): boolean {
  if (a.length !== b.length) {
    return false;
  }
  return a.every((step, index) => movementToken(step) === movementToken(b[index]));
}

function rate<T>(items: T[], predicate: (item: T) => boolean): number {
  if (items.length === 0) {
    return 0;
  }
  let matched = 0;
  for (const item of items) {
    if (predicate(item)) {
      matched += 1;
    }
  }
  return matched / items.length;
}
