import { buildMovementDataset } from "./movement-dataset.js";
import type { MovementDataset, MovementExample } from "./movement-dataset.js";
import type { MovementModelBackend, MovementQuery, TrainedMovementModel } from "./movement-backend.js";
import type { TrajectorySpan } from "../capture/trajectory.js";

/**
 * Generalization eval harness.
 *
 * Measures how well a trained movement model reproduces *held-out* movements —
 * the signal that it generalizes rather than merely memorizes. We split the
 * dataset's examples into train/holdout, train a backend on the train split,
 * then ask it to predict the action for each held-out context and compare
 * against ground truth.
 */

export type MovementMatchLevel = "exact" | "tool" | "miss";

export type MovementEvalCase = {
  expected: MovementExample;
  predictedSummary?: string;
  predictedTool?: string;
  predictedTarget?: string;
  match: MovementMatchLevel;
  generalized: boolean;
  confidence: number;
};

export type MovementGeneralizationReport = {
  trainCount: number;
  holdoutCount: number;
  /** Fraction of held-out cases where tool+gesture+target all matched. */
  exactAccuracy: number;
  /** Fraction where at least the tool matched. */
  toolAccuracy: number;
  /** Fraction of predictions the model flagged as generalized. */
  generalizationRate: number;
  cases: MovementEvalCase[];
};

function queryFromExample(example: MovementExample): MovementQuery {
  return {
    summary: example.context.summary,
    ...(example.context.app ? { app: example.context.app } : {}),
    ...(example.context.screen ? { screen: example.context.screen } : {}),
    ...(example.context.source ? { source: example.context.source } : {}),
  };
}

function classifyMatch(expected: MovementExample, prediction: ReturnType<TrainedMovementModel["predict"]>): MovementMatchLevel {
  if (!prediction) {
    return "miss";
  }
  const a = expected.action;
  const b = prediction.action;
  const toolMatch = a.tool === b.tool && (a.gesture ?? "") === (b.gesture ?? "");
  if (!toolMatch) {
    return "miss";
  }
  if ((a.target ?? "") === (b.target ?? "") && (a.direction ?? "") === (b.direction ?? "")) {
    return "exact";
  }
  return "tool";
}

/**
 * Deterministically split examples by index parity into a train majority and a
 * holdout minority. `holdoutEvery = 3` holds out every 3rd example.
 */
export function splitMovementDataset(
  dataset: MovementDataset,
  holdoutEvery = 3,
): { train: MovementDataset; holdout: MovementExample[] } {
  const train: MovementExample[] = [];
  const holdout: MovementExample[] = [];
  dataset.examples.forEach((example, position) => {
    if (holdoutEvery > 0 && position % holdoutEvery === holdoutEvery - 1) {
      holdout.push(example);
    } else {
      train.push(example);
    }
  });
  return {
    train: { ...dataset, examples: train },
    holdout,
  };
}

export async function evaluateMovementGeneralization(params: {
  backend: MovementModelBackend;
  trajectories: TrajectorySpan[];
  holdoutEvery?: number;
  requireApproved?: boolean;
  trainedAt?: string;
}): Promise<MovementGeneralizationReport> {
  const dataset = buildMovementDataset(params.trajectories, {
    requireApproved: params.requireApproved ?? false,
  });
  const { train, holdout } = splitMovementDataset(dataset, params.holdoutEvery ?? 3);
  const model = await params.backend.train(train, params.trainedAt ? { trainedAt: params.trainedAt } : {});

  const cases: MovementEvalCase[] = holdout.map((expected) => {
    const prediction = model.predict(queryFromExample(expected));
    const match = classifyMatch(expected, prediction);
    return {
      expected,
      predictedSummary: prediction?.action.summary,
      predictedTool: prediction?.action.tool,
      predictedTarget: prediction?.action.target,
      match,
      generalized: prediction?.generalized ?? false,
      confidence: prediction?.confidence ?? 0,
    };
  });

  const total = cases.length || 1;
  const exact = cases.filter((entry) => entry.match === "exact").length;
  const tool = cases.filter((entry) => entry.match !== "miss").length;
  const generalized = cases.filter((entry) => entry.generalized).length;

  return {
    trainCount: train.examples.length,
    holdoutCount: holdout.length,
    exactAccuracy: Number((exact / total).toFixed(4)),
    toolAccuracy: Number((tool / total).toFixed(4)),
    generalizationRate: Number((generalized / total).toFixed(4)),
    cases,
  };
}
