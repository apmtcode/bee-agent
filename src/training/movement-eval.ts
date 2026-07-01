import {
  MOVEMENT_END,
  MOVEMENT_START,
  buildMovementDataset,
  movementToken,
  type MovementDataset,
  type MovementSequence,
  type TrainedMovementModel,
} from "./movement-model.js";

/**
 * Generalization + fidelity eval harness for the movement-model subsystem.
 * Measures (a) how faithfully a trained model repeats a recorded movement
 * (replay fidelity) and (b) how well it predicts the next movement on held-out
 * but related trajectories (generalization). Also provides a seeded, fully
 * deterministic synthetic event-stream generator so the whole
 * capture -> dataset -> train -> replay pipeline can be validated in the cloud
 * without any real OS input.
 */

export type MovementDatasetSplit = {
  train: MovementDataset;
  test: MovementDataset;
};

/**
 * Deterministically split a dataset into train/test by index stride (every
 * `testStride`-th sequence becomes a test example). No randomness, so results
 * are reproducible across runs and machines.
 */
export function splitMovementDataset(dataset: MovementDataset, options?: { testStride?: number }): MovementDatasetSplit {
  const stride = Math.max(2, options?.testStride ?? 4);
  const train: MovementSequence[] = [];
  const test: MovementSequence[] = [];
  dataset.sequences.forEach((sequence, index) => {
    if (index % stride === stride - 1) {
      test.push(sequence);
    } else {
      train.push(sequence);
    }
  });
  return { train: buildMovementDataset(train), test: buildMovementDataset(test) };
}

/** Partition a dataset by an explicit hold-out trajectory-id set. */
export function partitionMovementDataset(dataset: MovementDataset, holdOutTrajectoryIds: Iterable<string>): MovementDatasetSplit {
  const holdOut = new Set(holdOutTrajectoryIds);
  const train: MovementSequence[] = [];
  const test: MovementSequence[] = [];
  for (const sequence of dataset.sequences) {
    (holdOut.has(sequence.trajectoryId) ? test : train).push(sequence);
  }
  return { train: buildMovementDataset(train), test: buildMovementDataset(test) };
}

export type NextActionAccuracy = {
  positions: number;
  top1Correct: number;
  topKCorrect: number;
  top1Accuracy: number;
  topKAccuracy: number;
  /** Positions where no context suffix matched at all (pure novelty). */
  unseenContexts: number;
};

/**
 * Next-action prediction accuracy over a set of sequences. For each position
 * the model predicts the next token from the preceding context; we score exact
 * top-1 matches and whether the true token appears in the top-K candidates.
 */
export function evaluateNextActionAccuracy(
  model: TrainedMovementModel,
  sequences: MovementSequence[],
  options?: { topK?: number },
): NextActionAccuracy {
  const topK = Math.max(1, options?.topK ?? 3);
  let positions = 0;
  let top1Correct = 0;
  let topKCorrect = 0;
  let unseenContexts = 0;

  for (const sequence of sequences) {
    const tokens = [MOVEMENT_START, ...sequence.steps.map((step) => step.token), MOVEMENT_END];
    for (let i = 1; i < tokens.length; i += 1) {
      const target = tokens[i];
      if (target === undefined) {
        continue;
      }
      const context = tokens.slice(0, i);
      const prediction = model.predictNext(context, { topK });
      positions += 1;
      if (prediction.matchedContextLength < 0) {
        unseenContexts += 1;
      }
      if (prediction.token === target) {
        top1Correct += 1;
      }
      if (prediction.candidates.some((candidate) => candidate.token === target)) {
        topKCorrect += 1;
      }
    }
  }

  return {
    positions,
    top1Correct,
    topKCorrect,
    top1Accuracy: positions === 0 ? 0 : top1Correct / positions,
    topKAccuracy: positions === 0 ? 0 : topKCorrect / positions,
    unseenContexts,
  };
}

export type ReplayFidelity = {
  trajectoryId: string;
  expected: string[];
  produced: string[];
  matchedPrefix: number;
  fidelity: number;
  exact: boolean;
};

/**
 * Replay fidelity: roll the model out from the sequence's first step and
 * measure how much of the recorded movement it reproduces. `fidelity` is the
 * length of the correctly reproduced leading prefix over the expected length.
 */
export function evaluateReplayFidelity(model: TrainedMovementModel, sequence: MovementSequence): ReplayFidelity {
  const expected = sequence.steps.map((step) => step.token);
  const seed = expected.length > 0 ? [MOVEMENT_START, expected[0]!] : [MOVEMENT_START];
  const produced = [expected[0], ...model.rollout(seed, { maxSteps: expected.length + 4 })].filter(
    (token): token is string => token !== undefined,
  );

  let matchedPrefix = 0;
  const limit = Math.min(expected.length, produced.length);
  for (let i = 0; i < limit; i += 1) {
    if (expected[i] === produced[i]) {
      matchedPrefix += 1;
    } else {
      break;
    }
  }

  return {
    trajectoryId: sequence.trajectoryId,
    expected,
    produced,
    matchedPrefix,
    fidelity: expected.length === 0 ? 1 : matchedPrefix / expected.length,
    exact: expected.length === produced.length && matchedPrefix === expected.length,
  };
}

export type ReplayFidelityReport = {
  sequences: number;
  exactMatches: number;
  meanFidelity: number;
  perSequence: ReplayFidelity[];
};

/** Aggregate replay fidelity across many sequences. */
export function evaluateDatasetReplayFidelity(model: TrainedMovementModel, sequences: MovementSequence[]): ReplayFidelityReport {
  const perSequence = sequences.map((sequence) => evaluateReplayFidelity(model, sequence));
  const exactMatches = perSequence.filter((result) => result.exact).length;
  const meanFidelity = perSequence.length === 0 ? 0 : perSequence.reduce((sum, result) => sum + result.fidelity, 0) / perSequence.length;
  return { sequences: perSequence.length, exactMatches, meanFidelity, perSequence };
}

/* -------------------------------------------------------------------------- */
/* Seeded synthetic movement generator                                        */
/* -------------------------------------------------------------------------- */

/** Deterministic 32-bit LCG — no Math.random, so streams are reproducible. */
class SeededRng {
  private state: number;
  constructor(seed: number) {
    this.state = (seed >>> 0) || 1;
  }
  next(): number {
    // Numerical Recipes LCG constants.
    this.state = (Math.imul(this.state, 1664525) + 1013904223) >>> 0;
    return this.state / 0xffffffff;
  }
  pick<T>(items: readonly T[]): T {
    return items[Math.floor(this.next() * items.length) % items.length]!;
  }
  int(maxExclusive: number): number {
    return Math.floor(this.next() * maxExclusive) % maxExclusive;
  }
}

/**
 * A movement "flow": an ordered list of stages, each offering interchangeable
 * variants. Related sequences share stages but pick different variants, so a
 * model trained on some can generalize to held-out variants of the same flow.
 */
type MovementFlow = {
  name: string;
  stages: Array<{ tool: string; variants: string[] }>;
};

const DEFAULT_FLOWS: MovementFlow[] = [
  {
    name: "compose-message",
    stages: [
      { tool: "os", variants: ["focused mail", "focused chat"] },
      { tool: "device", variants: ["tapped compose", "tapped new message"] },
      { tool: "device", variants: ["typed recipient", "typed subject"] },
      { tool: "device", variants: ["typed body", "typed reply"] },
      { tool: "device", variants: ["tapped send", "triggered send shortcut"] },
    ],
  },
  {
    name: "file-search",
    stages: [
      { tool: "os", variants: ["focused finder", "focused explorer"] },
      { tool: "device", variants: ["tapped search", "triggered search shortcut"] },
      { tool: "device", variants: ["typed query", "typed filename"] },
      { tool: "device", variants: ["scrolled down", "scrolled up"] },
      { tool: "device", variants: ["tapped open", "tapped result"] },
    ],
  },
  {
    name: "form-fill",
    stages: [
      { tool: "browser", variants: ["opened form", "opened checkout"] },
      { tool: "device", variants: ["typed name", "typed email"] },
      { tool: "device", variants: ["typed address", "typed phone"] },
      { tool: "device", variants: ["tapped continue", "tapped next"] },
      { tool: "device", variants: ["tapped submit", "tapped confirm"] },
    ],
  },
];

export type SyntheticMovementOptions = {
  seed?: number;
  sequenceCount?: number;
  flows?: MovementFlow[];
  /** Base timestamp; each step advances by ~1s deterministically. */
  startTs?: number;
};

/**
 * Generate a deterministic set of related synthetic movement sequences. Same
 * seed + options always yields byte-identical output, so it doubles as fixture
 * data for round-trip and generalization tests.
 */
export function generateSyntheticMovementSequences(options?: SyntheticMovementOptions): MovementSequence[] {
  const rng = new SeededRng(options?.seed ?? 1);
  const flows = options?.flows ?? DEFAULT_FLOWS;
  const count = Math.max(1, options?.sequenceCount ?? 12);
  const startTs = options?.startTs ?? 1_700_000_000_000;
  const sequences: MovementSequence[] = [];

  for (let index = 0; index < count; index += 1) {
    const flow = flows[index % flows.length]!;
    let ts = startTs + index * 60_000;
    const steps = flow.stages.map((stage) => {
      const summary = rng.pick(stage.variants);
      ts += 1_000 + rng.int(500);
      return {
        token: movementToken(stage.tool, summary),
        tool: stage.tool,
        summary,
        ts,
      };
    });
    sequences.push({
      trajectoryId: `synthetic-${flow.name}-${index}`,
      sessionId: `synthetic-session-${index % 3}`,
      steps,
    });
  }

  return sequences;
}

/** Convenience: synthetic sequences already assembled into a dataset. */
export function generateSyntheticMovementDataset(options?: SyntheticMovementOptions): MovementDataset {
  return buildMovementDataset(generateSyntheticMovementSequences(options));
}
