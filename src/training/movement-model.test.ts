import { describe, expect, it } from "vitest";
import { buildTrajectorySpan } from "../capture/trajectory.js";
import type { TrajectoryAction, TrajectoryObservation } from "../capture/trajectory.js";
import {
  NgramMovementModelBackend,
  buildMovementDataset,
  evaluateMovementModel,
  loadMovementModel,
  tokenizeAction,
  tokenizeObservation,
  type MovementSequence,
} from "./movement-model.js";

function action(tool: string, summary: string, ts: number, metadata?: Record<string, unknown>): TrajectoryAction {
  return { kind: "action", tool, summary, ts, ...(metadata ? { metadata } : {}) };
}

function observation(source: string, summary: string, ts: number, metadata?: Record<string, unknown>): TrajectoryObservation {
  return { kind: "observation", source, summary, ts, ...(metadata ? { metadata } : {}) };
}

describe("tokenizeAction", () => {
  it("builds a structured movement token from gesture metadata", () => {
    const token = tokenizeAction(
      action("device", "swiped up", 1, { gesture: "swipe", direction: "up", target: "Photo Grid" }),
    );
    expect(token).toBe("device:swipe:up:photo-grid");
  });

  it("falls back to the summary when no structured detail exists", () => {
    expect(tokenizeAction(action("browser", "Clicked Deploy!", 1))).toBe("browser:clicked-deploy");
  });

  it("reads the device-adapter's `kind` metadata key", () => {
    expect(tokenizeAction(action("device", "tapped", 1, { kind: "tap", target: "submit" }))).toBe("device:tap:submit");
  });
});

describe("tokenizeObservation", () => {
  it("captures window/app context", () => {
    expect(tokenizeObservation(observation("device", "Mail active", 1, { appName: "Mail" }))).toBe("obs:device:mail");
  });
});

describe("buildMovementDataset", () => {
  it("orders actions by timestamp and produces a sorted vocabulary", () => {
    const trajectory = buildTrajectorySpan({
      id: "t1",
      sessionId: "s1",
      actions: [
        action("device", "b", 30, { gesture: "type", target: "field" }),
        action("device", "a", 10, { gesture: "tap", target: "menu" }),
      ],
    });
    const dataset = buildMovementDataset([trajectory]);
    expect(dataset.sequences).toEqual([
      { id: "t1", tokens: ["device:tap:menu", "device:type:field"] },
    ]);
    expect(dataset.vocabulary).toEqual(["device:tap:menu", "device:type:field"]);
  });

  it("drops trajectories with no actions", () => {
    const empty = buildTrajectorySpan({ id: "t0", sessionId: "s1" });
    expect(buildMovementDataset([empty]).sequences).toEqual([]);
  });

  it("interleaves observations when requested", () => {
    const trajectory = buildTrajectorySpan({
      id: "t1",
      sessionId: "s1",
      observations: [observation("device", "Mail active", 5, { appName: "Mail" })],
      actions: [action("device", "tap", 10, { gesture: "tap", target: "compose" })],
    });
    const dataset = buildMovementDataset([trajectory], { includeObservations: true });
    expect(dataset.sequences[0]!.tokens).toEqual(["obs:device:mail", "device:tap:compose"]);
  });
});

function tap(target: string, ts: number): TrajectoryAction {
  return action("device", `tapped ${target}`, ts, { gesture: "tap", target });
}

describe("NgramMovementModelBackend", () => {
  it("repeats a memorized movement sequence exactly (full-order recall)", async () => {
    const trajectory = buildTrajectorySpan({
      id: "t1",
      sessionId: "s1",
      actions: [tap("open", 1), tap("compose", 2), tap("send", 3)],
    });
    const dataset = buildMovementDataset([trajectory]);
    const model = await new NgramMovementModelBackend().train(dataset, { order: 3 });

    const first = model.predict(["device:tap:open"]);
    expect(first?.token).toBe("device:tap:compose");
    expect(first?.backoff).toBe(false);

    // Roll the whole recorded movement forward from the seed.
    const generated = model.generate(["device:tap:open"], { steps: 5 });
    expect(generated).toEqual(["device:tap:compose", "device:tap:send"]);
  });

  it("generalizes to a novel-but-related context via backoff", async () => {
    // Two trajectories that share the suffix `compose -> send`.
    const a = buildTrajectorySpan({
      id: "a",
      sessionId: "s1",
      actions: [tap("inbox", 1), tap("compose", 2), tap("send", 3)],
    });
    const b = buildTrajectorySpan({
      id: "b",
      sessionId: "s2",
      actions: [tap("drafts", 1), tap("compose", 2), tap("send", 3)],
    });
    const model = await new NgramMovementModelBackend().train(buildMovementDataset([a, b]), { order: 3 });

    // Novel prefix never seen in full (`archive -> compose`), but the suffix
    // `compose` was: the model should still predict `send` by backing off.
    const prediction = model.predict(["device:tap:archive", "device:tap:compose"]);
    expect(prediction?.token).toBe("device:tap:send");
    expect(prediction?.backoff).toBe(true);
    expect(prediction?.order).toBe(1);
  });

  it("is deterministic under ties (breaks by count then token name)", async () => {
    const a = buildTrajectorySpan({ id: "a", sessionId: "s1", actions: [tap("home", 1), tap("beta", 2)] });
    const b = buildTrajectorySpan({ id: "b", sessionId: "s2", actions: [tap("home", 1), tap("alpha", 2)] });
    const model = await new NgramMovementModelBackend().train(buildMovementDataset([a, b]), { order: 2 });
    // Both continuations of `home` have count 1 => tie broken lexicographically.
    const p1 = model.predict(["device:tap:home"]);
    const p2 = model.predict(["device:tap:home"]);
    expect(p1?.token).toBe("device:tap:alpha");
    expect(p1).toEqual(p2);
    expect(p1?.candidates.map((candidate) => candidate.token)).toEqual(["device:tap:alpha", "device:tap:beta"]);
  });

  it("returns undefined for an empty untrained model", async () => {
    const model = await new NgramMovementModelBackend().train({ version: 1, sequences: [], vocabulary: [] });
    expect(model.predict(["anything"])).toBeUndefined();
    expect(model.generate(["anything"], { steps: 3 })).toEqual([]);
  });

  it("halts generation at a stop token", async () => {
    const trajectory = buildTrajectorySpan({
      id: "t1",
      sessionId: "s1",
      actions: [tap("open", 1), tap("compose", 2), tap("send", 3)],
    });
    const model = await new NgramMovementModelBackend().train(buildMovementDataset([trajectory]));
    const generated = model.generate(["device:tap:open"], { steps: 5, stopToken: "device:tap:send" });
    expect(generated).toEqual(["device:tap:compose"]);
  });

  it("round-trips through serialization without retraining", async () => {
    const trajectory = buildTrajectorySpan({
      id: "t1",
      sessionId: "s1",
      actions: [tap("open", 1), tap("compose", 2), tap("send", 3)],
    });
    const model = await new NgramMovementModelBackend().train(buildMovementDataset([trajectory]), { order: 2 });
    const restored = loadMovementModel(model.toJSON());
    expect(restored.order).toBe(2);
    expect(restored.backend).toBe("ngram-backoff");
    expect(restored.generate(["device:tap:open"], { steps: 5 })).toEqual([
      "device:tap:compose",
      "device:tap:send",
    ]);
  });
});

describe("evaluateMovementModel", () => {
  it("scores next-movement fidelity and flags generalized hits", async () => {
    const train = buildTrajectorySpan({
      id: "train",
      sessionId: "s1",
      actions: [tap("inbox", 1), tap("compose", 2), tap("send", 3)],
    });
    const model = await new NgramMovementModelBackend().train(buildMovementDataset([train]), { order: 3 });

    // Held-out sequence shares the `compose -> send` suffix but starts differently.
    const heldOut: MovementSequence[] = [
      { id: "eval", tokens: ["device:tap:archive", "device:tap:compose", "device:tap:send"] },
    ];
    const evaluation = evaluateMovementModel(model, heldOut);
    expect(evaluation.predictions).toBe(2);
    // `compose -> send` is predicted correctly via backoff; the first step
    // (`archive -> compose`) is not recoverable, so 1/2 correct, all generalized.
    expect(evaluation.correct).toBe(1);
    expect(evaluation.generalizedCorrect).toBe(1);
    expect(evaluation.accuracy).toBeCloseTo(0.5);
  });

  it("reports zero accuracy with no predictions on empty held-out data", async () => {
    const model = await new NgramMovementModelBackend().train({ version: 1, sequences: [], vocabulary: [] });
    expect(evaluateMovementModel(model, [])).toEqual({
      predictions: 0,
      correct: 0,
      accuracy: 0,
      generalizedCorrect: 0,
    });
  });
});
