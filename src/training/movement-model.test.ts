import { describe, expect, it } from "vitest";
import { buildTrajectorySpan, type TrajectoryAction } from "../capture/trajectory.js";
import {
  NgramMovementBackend,
  buildMovementDataset,
  evaluateMovementModel,
  type MovementDataset,
  type MovementSequence,
  type MovementStep,
} from "./movement-model.js";

function step(tool: string, summary: string): MovementStep {
  return { tool, summary };
}

function action(tool: string, summary: string, ts: number): TrajectoryAction {
  return { kind: "action", tool, summary, ts };
}

function dataset(...sequences: MovementSequence[]): MovementDataset {
  return { version: 1, sequences };
}

describe("buildMovementDataset", () => {
  it("orders actions by timestamp and drops empty trajectories", () => {
    const spans = [
      buildTrajectorySpan({
        id: "traj-1",
        sessionId: "s1",
        actions: [action("mouse.click", "toolbar", 30), action("key.press", "cmd+s", 10)],
      }),
      buildTrajectorySpan({ id: "traj-empty", sessionId: "s1", actions: [] }),
    ];

    const result = buildMovementDataset(spans);

    expect(result.sequences).toHaveLength(1);
    expect(result.sequences[0]!.trajectoryId).toBe("traj-1");
    expect(result.sequences[0]!.steps.map((s) => s.summary)).toEqual(["cmd+s", "toolbar"]);
  });
});

describe("NgramMovementBackend", () => {
  it("repeats a memorized movement sequence exactly (replay)", async () => {
    const recorded = [
      step("window.focus", "editor"),
      step("mouse.click", "line-42"),
      step("key.press", "cmd+c"),
      step("key.press", "cmd+v"),
    ];
    const model = await new NgramMovementBackend().train(
      dataset({ trajectoryId: "t1", steps: recorded }),
    );

    const generated = model.generate([recorded[0]!], recorded.length - 1);

    expect(generated).toEqual(recorded.slice(1));
    const prediction = model.predictNext(recorded.slice(0, 2));
    expect(prediction?.step.summary).toBe("cmd+c");
    expect(prediction?.exact).toBe(true);
    expect(prediction?.probability).toBe(1);
  });

  it("generalizes to a new-but-related prefix via back-off", async () => {
    // Two trajectories teach: after a copy (cmd+c) the operator pastes (cmd+v).
    const model = await new NgramMovementBackend().train(
      dataset(
        {
          trajectoryId: "t1",
          steps: [step("mouse.click", "row-1"), step("key.press", "cmd+c"), step("key.press", "cmd+v")],
        },
        {
          trajectoryId: "t2",
          steps: [step("mouse.click", "row-2"), step("key.press", "cmd+c"), step("key.press", "cmd+v")],
        },
      ),
      { order: 2 },
    );

    // Unseen high-order prefix (new click target) but a familiar recent action.
    const prediction = model.predictNext([step("mouse.click", "row-99"), step("key.press", "cmd+c")]);

    expect(prediction?.step.summary).toBe("cmd+v");
    expect(prediction?.exact).toBe(false);
    expect(prediction?.contextUsed).toBeLessThan(2);
  });

  it("ranks the distribution by observed frequency", async () => {
    const model = await new NgramMovementBackend().train(
      dataset(
        { trajectoryId: "t1", steps: [step("key.press", "a"), step("key.press", "x")] },
        { trajectoryId: "t2", steps: [step("key.press", "a"), step("key.press", "x")] },
        { trajectoryId: "t3", steps: [step("key.press", "a"), step("key.press", "y")] },
      ),
      { order: 1 },
    );

    const dist = model.predictDistribution([step("key.press", "a")]);

    expect(dist.map((p) => p.step.summary)).toEqual(["x", "y"]);
    expect(dist[0]!.probability).toBeCloseTo(2 / 3, 5);
    expect(dist[1]!.probability).toBeCloseTo(1 / 3, 5);
  });

  it("round-trips through serialize/restore deterministically", async () => {
    const backend = new NgramMovementBackend();
    const data = dataset({
      trajectoryId: "t1",
      steps: [step("mouse.move", "up"), step("mouse.move", "up"), step("mouse.click", "target")],
    });
    const model = await backend.train(data, { order: 2 });
    const restored = backend.restore(model.serialize());

    const seed = [step("mouse.move", "up")];
    expect(restored.generate(seed, 3)).toEqual(model.generate(seed, 3));
    expect(restored.serialize()).toEqual(model.serialize());
  });

  it("returns no prediction for an empty model", async () => {
    const model = await new NgramMovementBackend().train(dataset());
    expect(model.predictNext([step("key.press", "a")])).toBeUndefined();
    expect(model.generate([], 5)).toEqual([]);
  });
});

describe("evaluateMovementModel", () => {
  it("scores perfect replay fidelity on the training sequence", async () => {
    const seq: MovementSequence = {
      trajectoryId: "t1",
      steps: [step("a", "1"), step("a", "2"), step("a", "3")],
    };
    const model = await new NgramMovementBackend().train(dataset(seq), { order: 3 });

    const report = evaluateMovementModel(model, [seq]);

    expect(report.predictions).toBe(3);
    expect(report.accuracy).toBe(1);
  });

  it("reports generalization on held-out but related sequences", async () => {
    const model = await new NgramMovementBackend().train(
      dataset(
        { trajectoryId: "t1", steps: [step("click", "a"), step("key", "copy"), step("key", "paste")] },
        { trajectoryId: "t2", steps: [step("click", "b"), step("key", "copy"), step("key", "paste")] },
      ),
      { order: 2 },
    );

    const report = evaluateMovementModel(model, [
      { trajectoryId: "held", steps: [step("click", "z"), step("key", "copy"), step("key", "paste")] },
    ]);

    // The paste-after-copy transition is predicted despite the unseen click target.
    expect(report.correct).toBeGreaterThan(0);
    expect(report.generalizedRate).toBeGreaterThan(0);
  });
});
