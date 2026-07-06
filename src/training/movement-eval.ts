import { buildTrajectorySpan, type TrajectoryAction, type TrajectorySpan } from "../capture/trajectory.js";
import {
  MOVEMENT_END_TOKEN,
  MOVEMENT_START_TOKEN,
  defaultMovementTokenizer,
  type MovementSequence,
  type MovementTokenizer,
  type TrainedMovementPolicy,
} from "./movement-policy.js";

/**
 * Generalization eval harness for movement policies. Measures how faithfully a
 * trained policy repeats and generalizes recorded movements on held-out
 * trajectories, plus a deterministic synthetic trajectory generator so the
 * capture -> dataset -> train -> infer loop can be validated without any real
 * OS input.
 */

export type PerTrajectoryEval = {
  trajectoryId: string;
  length: number;
  /** Fraction of next-token predictions that matched the recorded action. */
  nextTokenAccuracy: number;
  /** Fraction where the recorded action was within the top-k distribution. */
  topKAccuracy: number;
  /** Normalized longest-common-subsequence similarity of a rolled-out sequence. */
  sequenceSimilarity: number;
};

export type MovementPolicyEval = {
  trajectoryCount: number;
  predictionCount: number;
  nextTokenAccuracy: number;
  topKAccuracy: number;
  meanSequenceSimilarity: number;
  perTrajectory: PerTrajectoryEval[];
};

export type EvaluateMovementPolicyOptions = {
  tokenizer?: MovementTokenizer;
  /** Top-k window for topKAccuracy. Default 3. */
  topK?: number;
  /** How many leading tokens to seed rollouts with. Default 1. */
  seedLength?: number;
};

export function evaluateMovementPolicy(
  policy: TrainedMovementPolicy,
  heldOut: TrajectorySpan[],
  options: EvaluateMovementPolicyOptions = {},
): MovementPolicyEval {
  const tokenizer = options.tokenizer ?? defaultMovementTokenizer;
  const topK = Math.max(1, options.topK ?? 3);
  const seedLength = Math.max(0, options.seedLength ?? 1);

  const perTrajectory: PerTrajectoryEval[] = [];
  let totalPredictions = 0;
  let totalCorrect = 0;
  let totalTopK = 0;

  for (const trajectory of heldOut) {
    const sequence = toSequence(trajectory, tokenizer);
    if (sequence.length === 0) {
      continue;
    }
    const padded = [MOVEMENT_START_TOKEN, ...sequence, MOVEMENT_END_TOKEN];

    let correct = 0;
    let inTopK = 0;
    let predictions = 0;
    for (let i = 1; i < padded.length; i += 1) {
      const context = padded.slice(0, i);
      const expected = padded[i]!;
      const prediction = policy.predictNext(context);
      predictions += 1;
      if (prediction.token === expected) {
        correct += 1;
      }
      if (prediction.distribution.slice(0, topK).some((entry) => entry.token === expected)) {
        inTopK += 1;
      }
    }

    const seed = sequence.slice(0, Math.min(seedLength, sequence.length));
    const rollout = policy.generate(seed, sequence.length + 4);
    const similarity = lcsSimilarity(rollout, sequence);

    totalPredictions += predictions;
    totalCorrect += correct;
    totalTopK += inTopK;
    perTrajectory.push({
      trajectoryId: trajectory.id,
      length: sequence.length,
      nextTokenAccuracy: predictions === 0 ? 0 : correct / predictions,
      topKAccuracy: predictions === 0 ? 0 : inTopK / predictions,
      sequenceSimilarity: similarity,
    });
  }

  const meanSequenceSimilarity =
    perTrajectory.length === 0
      ? 0
      : perTrajectory.reduce((sum, entry) => sum + entry.sequenceSimilarity, 0) / perTrajectory.length;

  return {
    trajectoryCount: perTrajectory.length,
    predictionCount: totalPredictions,
    nextTokenAccuracy: totalPredictions === 0 ? 0 : totalCorrect / totalPredictions,
    topKAccuracy: totalPredictions === 0 ? 0 : totalTopK / totalPredictions,
    meanSequenceSimilarity,
    perTrajectory,
  };
}

function toSequence(trajectory: TrajectorySpan, tokenizer: MovementTokenizer): MovementSequence {
  return [...trajectory.actions].sort((a, b) => a.ts - b.ts).map((action) => tokenizer(action));
}

/** Normalized longest-common-subsequence similarity in [0, 1]. */
export function lcsSimilarity(a: readonly string[], b: readonly string[]): number {
  if (a.length === 0 && b.length === 0) {
    return 1;
  }
  if (a.length === 0 || b.length === 0) {
    return 0;
  }
  const rows = a.length + 1;
  const cols = b.length + 1;
  const table = new Array<number>(rows * cols).fill(0);
  for (let i = 1; i < rows; i += 1) {
    for (let j = 1; j < cols; j += 1) {
      if (a[i - 1] === b[j - 1]) {
        table[i * cols + j] = table[(i - 1) * cols + (j - 1)]! + 1;
      } else {
        table[i * cols + j] = Math.max(table[(i - 1) * cols + j]!, table[i * cols + (j - 1)]!);
      }
    }
  }
  const lcs = table[a.length * cols + b.length]!;
  return lcs / Math.max(a.length, b.length);
}

/**
 * Deterministic 32-bit PRNG (mulberry32). Seeded so synthetic datasets and their
 * eval results are reproducible across runs — no `Math.random`.
 */
export function createSeededRandom(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** A named workflow template: an ordered list of action tools ("movements"). */
export type MovementWorkflowTemplate = {
  name: string;
  steps: string[];
};

export type SyntheticTrajectoryOptions = {
  templates: MovementWorkflowTemplate[];
  /** Instances to emit per template. Default 4. */
  perTemplate?: number;
  /** Seed for reproducible perturbations. Default 1. */
  seed?: number;
  /** Probability of dropping a step (0..1). Default 0.15. */
  dropRate?: number;
  /** Probability of duplicating a step (0..1). Default 0.15. */
  repeatRate?: number;
  captureTier?: TrajectorySpan["captureTier"];
};

/**
 * Generate related-but-varied trajectories from workflow templates. Each
 * instance is the template's step sequence with small, seeded perturbations
 * (occasional drops/repeats), mirroring the natural variation in how a person
 * repeats a task. Related instances share structure, so a policy trained on some
 * can be evaluated for generalization on held-out ones.
 */
export function generateSyntheticTrajectories(options: SyntheticTrajectoryOptions): TrajectorySpan[] {
  const perTemplate = Math.max(1, options.perTemplate ?? 4);
  const dropRate = clamp01(options.dropRate ?? 0.15);
  const repeatRate = clamp01(options.repeatRate ?? 0.15);
  const random = createSeededRandom(options.seed ?? 1);
  const trajectories: TrajectorySpan[] = [];

  for (const template of options.templates) {
    for (let instance = 0; instance < perTemplate; instance += 1) {
      const steps: string[] = [];
      for (const step of template.steps) {
        if (random() < dropRate && steps.length > 0) {
          continue;
        }
        steps.push(step);
        if (random() < repeatRate) {
          steps.push(step);
        }
      }
      if (steps.length === 0) {
        steps.push(template.steps[0] ?? "noop");
      }
      const actions: TrajectoryAction[] = steps.map((tool, index) => ({
        kind: "action",
        tool,
        summary: `${template.name} step ${index + 1}`,
        ts: index + 1,
      }));
      trajectories.push(
        buildTrajectorySpan({
          id: `${template.name}-${instance}`,
          sessionId: `synthetic-${template.name}`,
          captureTier: options.captureTier ?? "operator",
          actions,
          outcome: { status: "success", summary: `${template.name} completed`, reward: 1 },
        }),
      );
    }
  }

  return trajectories;
}

function clamp01(value: number): number {
  if (Number.isNaN(value)) {
    return 0;
  }
  return Math.min(1, Math.max(0, value));
}
