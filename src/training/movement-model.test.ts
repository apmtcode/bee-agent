import { describe, expect, it } from "vitest";
import type { TrajectorySpan } from "../capture/trajectory.js";
import {
  CountingMovementBackend,
  buildMovementDataset,
  evaluateMovementModel,
  movementSummaryToken,
  synthesizeMovementTrajectory,
  toMovementTimeline,
  trainMovementModel,
  type MovementModelBackend,
  type MovementSample,
  type MovementContext,
  type TrainedMovementModel,
} from "./movement-model.js";

function trajectory(id: string, steps: Parameters<typeof synthesizeMovementTrajectory>[0]["steps"], outcome?: TrajectorySpan["outcome"]): TrajectorySpan {
  return synthesizeMovementTrajectory({ id, steps, ...(outcome ? { outcome } : {}) });
}

describe("movementSummaryToken", () => {
  it("reduces a summary to a stable lowercase first token", () => {
    expect(movementSummaryToken("Open File main.ts")).toBe("open");
    expect(movementSummaryToken("  click!! button ")).toBe("click");
    expect(movementSummaryToken("")).toBe("<empty>");
  });
});

describe("toMovementTimeline", () => {
  it("merges observations and actions in ts order with observe-before-act tie-break", () => {
    const span = synthesizeMovementTrajectory({
      id: "t",
      steps: [{ observationSource: "editor", observationSummary: "file", tool: "type", actionSummary: "hi" }],
    });
    const timeline = toMovementTimeline(span);
    expect(timeline.map((event) => event.kind)).toEqual(["observation", "action"]);
    expect(timeline[0]!.ts).toBeLessThan(timeline[1]!.ts);
  });
});

describe("buildMovementDataset", () => {
  it("derives one (context -> action) sample per action with preceding context", () => {
    const span = trajectory("t1", [
      { observationSource: "editor", observationSummary: "file main.ts", tool: "type", actionSummary: "type hello" },
      { observationSource: "terminal", observationSummary: "prompt ready", tool: "run", actionSummary: "run tests" },
    ]);
    const samples = buildMovementDataset([span]);
    expect(samples).toHaveLength(2);
    expect(samples[0]!.context).toEqual<MovementContext>({
      lastObservationSource: "editor",
      lastActionTool: "<start>",
      lastObservationToken: "file",
    });
    expect(samples[0]!.action.tool).toBe("type");
    // Second action's context carries forward the first action as lastActionTool.
    expect(samples[1]!.context.lastActionTool).toBe("type");
    expect(samples[1]!.context.lastObservationSource).toBe("terminal");
  });

  it("weights samples by outcome reward and penalizes failures", () => {
    const rewarded = buildMovementDataset([
      trajectory("r", [{ observationSource: "a", observationSummary: "x", tool: "t", actionSummary: "s" }], {
        status: "success",
        summary: "ok",
        reward: 4,
      }),
    ]);
    const failed = buildMovementDataset([
      trajectory("f", [{ observationSource: "a", observationSummary: "x", tool: "t", actionSummary: "s" }], {
        status: "failure",
        summary: "bad",
      }),
    ]);
    expect(rewarded[0]!.weight).toBe(5); // reward + 1
    expect(failed[0]!.weight).toBe(0.25);
  });
});

describe("CountingMovementBackend — replay (objective c)", () => {
  it("replays the recorded next action for an exact context", async () => {
    const span = trajectory("t1", [
      { observationSource: "editor", observationSummary: "file main.ts", tool: "type", actionSummary: "type hello" },
    ]);
    const model = await trainMovementModel([span]);
    const prediction = model.predict({
      lastObservationSource: "editor",
      lastActionTool: "<start>",
      lastObservationToken: "file",
    });
    expect(prediction.tool).toBe("type");
    expect(prediction.summary).toBe("type hello");
    expect(prediction.source).toBe("exact");
    expect(prediction.backoffLevel).toBe(0);
    expect(prediction.confidence).toBe(1);
  });

  it("prefers the higher-reward action when the same context yields two actions", async () => {
    const lowReward = trajectory("low", [
      { observationSource: "editor", observationSummary: "file main.ts", tool: "delete", actionSummary: "delete line" },
    ], { status: "success", summary: "ok", reward: 0 });
    const highReward = trajectory("high", [
      { observationSource: "editor", observationSummary: "file main.ts", tool: "type", actionSummary: "type hello" },
    ], { status: "success", summary: "ok", reward: 9 });
    const model = await trainMovementModel([lowReward, highReward]);
    const prediction = model.predict({
      lastObservationSource: "editor",
      lastActionTool: "<start>",
      lastObservationToken: "file",
    });
    expect(prediction.tool).toBe("type");
    expect(prediction.confidence).toBeGreaterThan(0.5);
  });
});

describe("CountingMovementBackend — generalization (objective d)", () => {
  it("backs off to a coarser context for an unseen-but-related state", async () => {
    const span = trajectory("t1", [
      { observationSource: "editor", observationSummary: "file main.ts", tool: "type", actionSummary: "type hello" },
    ]);
    const model = await trainMovementModel([span]);
    // Same source + prior action, but a summary token never seen in training.
    const prediction = model.predict({
      lastObservationSource: "editor",
      lastActionTool: "<start>",
      lastObservationToken: "buffer",
    });
    expect(prediction.tool).toBe("type");
    expect(prediction.source).toBe("generalized");
    expect(prediction.backoffLevel).toBeGreaterThan(0);
  });

  it("falls back to the global prior when no context dimension matches", async () => {
    const span = trajectory("t1", [
      { observationSource: "editor", observationSummary: "file main.ts", tool: "type", actionSummary: "type hello" },
    ]);
    const model = await trainMovementModel([span]);
    const prediction = model.predict({
      lastObservationSource: "unknown-app",
      lastActionTool: "unknown-tool",
      lastObservationToken: "unknown",
    });
    expect(prediction.tool).toBe("type");
    expect(prediction.source).toBe("prior");
    expect(prediction.confidence).toBe(0);
  });
});

describe("evaluateMovementModel", () => {
  it("reports exact replay fidelity on the training trajectory itself", async () => {
    const span = trajectory("t1", [
      { observationSource: "editor", observationSummary: "file main.ts", tool: "type", actionSummary: "type hello" },
      { observationSource: "terminal", observationSummary: "prompt ready", tool: "run", actionSummary: "run tests" },
    ]);
    const model = await trainMovementModel([span]);
    const report = evaluateMovementModel(model, [span]);
    expect(report.sampleCount).toBe(2);
    expect(report.toolAccuracy).toBe(1);
    expect(report.exactAccuracy).toBe(1);
    expect(report.generalizationRate).toBe(0);
  });

  it("generalizes tool choice to a held-out but related trajectory", async () => {
    const train = trajectory("train", [
      { observationSource: "editor", observationSummary: "file main.ts", tool: "type", actionSummary: "type hello" },
    ]);
    const heldOut = trajectory("held", [
      // Same screen state family, unseen summary token + unseen action summary.
      { observationSource: "editor", observationSummary: "buffer scratch", tool: "type", actionSummary: "type world" },
    ]);
    const model = await trainMovementModel([train]);
    const report = evaluateMovementModel(model, [heldOut]);
    expect(report.toolAccuracy).toBe(1); // correct tool via generalization
    expect(report.exactAccuracy).toBe(0); // summary differs
    expect(report.generalizationRate).toBe(1);
    expect(report.generalizedToolAccuracy).toBe(1);
  });
});

describe("synthesizeMovementTrajectory", () => {
  it("produces deterministic, ts-ordered synthetic trajectories", () => {
    const a = synthesizeMovementTrajectory({
      id: "syn",
      startTs: 1000,
      stepMs: 5,
      steps: [
        { observationSource: "s", observationSummary: "obs", tool: "t", actionSummary: "act" },
      ],
    });
    const b = synthesizeMovementTrajectory({
      id: "syn",
      startTs: 1000,
      stepMs: 5,
      steps: [
        { observationSource: "s", observationSummary: "obs", tool: "t", actionSummary: "act" },
      ],
    });
    expect(a).toEqual(b);
    expect(a.observations[0]!.ts).toBe(1000);
    expect(a.actions[0]!.ts).toBe(1005);
  });
});

describe("pluggable backend seam", () => {
  it("accepts a custom MovementModelBackend implementing the interface", async () => {
    const alwaysNoop: MovementModelBackend = {
      name: "noop",
      train(dataset: MovementSample[]): TrainedMovementModel {
        return {
          backend: "noop",
          predict: () => ({ tool: "noop", summary: `${dataset.length}`, confidence: 1, source: "prior", backoffLevel: -1 }),
          toJSON: () => ({ backend: "noop" }),
        };
      },
    };
    const span = trajectory("t", [{ observationSource: "a", observationSummary: "b", tool: "c", actionSummary: "d" }]);
    const model = await trainMovementModel([span], alwaysNoop);
    expect(model.backend).toBe("noop");
    expect(model.predict({ lastObservationSource: "a", lastActionTool: "<start>", lastObservationToken: "b" }).tool).toBe(
      "noop",
    );
  });
});

describe("serialization", () => {
  it("produces an inspectable JSON snapshot of the learned policy", async () => {
    const span = trajectory("t1", [
      { observationSource: "editor", observationSummary: "file main.ts", tool: "type", actionSummary: "type hello" },
    ]);
    const model = await trainMovementModel([span]);
    const snapshot = model.toJSON() as { backend: string; version: number; levels: unknown[] };
    expect(snapshot.backend).toBe("counting");
    expect(snapshot.version).toBe(1);
    expect(snapshot.levels).toHaveLength(5);
    expect(JSON.stringify(snapshot)).toContain("type hello");
  });
});
