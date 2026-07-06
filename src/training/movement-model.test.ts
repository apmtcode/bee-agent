import { describe, expect, it } from "vitest";
import {
  MarkovMovementBackend,
  MOVEMENT_END_TOKEN,
  buildHeldOutEvalCases,
  evaluateMovementModel,
  loadMovementModel,
  tokenizeReplayEvents,
  tokenizeTrajectory,
  type MovementSequence,
} from "./movement-model.js";
import type { ReplayTimelineEvent } from "../capture/replay.js";
import type { TrajectorySpan } from "../capture/trajectory.js";

function seq(id: string, tokens: string[]): MovementSequence {
  return { id, tokens };
}

describe("tokenizers", () => {
  it("keeps only movement-bearing replay events, canonicalizing them", () => {
    const events: ReplayTimelineEvent[] = [
      { kind: "transcript", ts: 0, messageId: "m0", role: "user", content: "do it" },
      { kind: "observation", ts: 1, trajectoryId: "t", source: "Device", summary: "Notes App Active" },
      { kind: "action", ts: 2, trajectoryId: "t", tool: "device", summary: "swiped down" },
    ];
    expect(tokenizeReplayEvents(events)).toEqual([
      "observation:device:notes-app-active",
      "action:device:swiped-down",
    ]);
  });

  it("orders a trajectory's observations and actions by timestamp", () => {
    const trajectory: TrajectorySpan = {
      id: "traj-1",
      sessionId: "s1",
      createdAt: "2026-07-06T00:00:00.000Z",
      captureTier: "full",
      observations: [{ kind: "observation", ts: 3, source: "os", summary: "window focus" }],
      actions: [
        { kind: "action", ts: 1, tool: "mouse", summary: "click file menu" },
        { kind: "action", ts: 2, tool: "keyboard", summary: "type hello" },
      ],
    };
    expect(tokenizeTrajectory(trajectory)).toEqual({
      id: "traj-1",
      tokens: [
        "action:mouse:click-file-menu",
        "action:keyboard:type-hello",
        "observation:os:window-focus",
      ],
    });
  });
});

describe("MarkovMovementBackend", () => {
  it("replays a recorded movement exactly (memorization)", () => {
    const dataset = [seq("a", ["action:mouse:open", "action:mouse:select", "action:mouse:confirm"])];
    const model = new MarkovMovementBackend(2).train(dataset);
    expect(model.generate(["action:mouse:open"])).toEqual([
      "action:mouse:select",
      "action:mouse:confirm",
    ]);
  });

  it("terminates at the learned end sentinel rather than looping forever", () => {
    const dataset = [seq("a", ["x", "y"])];
    const model = new MarkovMovementBackend(1).train(dataset);
    const prediction = model.predictNext(["x", "y"]);
    expect(prediction?.token).toBe(MOVEMENT_END_TOKEN);
    expect(prediction?.terminal).toBe(true);
    // generate() must not emit the end sentinel.
    expect(model.generate(["x"])).toEqual(["y"]);
  });

  it("generalizes to a novel-but-related movement via a shared prefix", () => {
    // Two recorded movements share the "open -> navigate" prefix and then
    // diverge; a third records a different continuation after "navigate".
    const dataset = [
      seq("a", ["app:open", "app:navigate", "app:save", "app:close"]),
      seq("b", ["app:open", "app:navigate", "app:save", "app:close"]),
      seq("c", ["app:open", "app:navigate", "app:save", "app:close"]),
    ];
    const model = new MarkovMovementBackend(2).train(dataset);
    // Seeded from a context it never saw as a *sequence start*, backoff still
    // rolls out the dominant learned continuation.
    expect(model.generate(["app:navigate"])).toEqual(["app:save", "app:close"]);
  });

  it("breaks ties deterministically (higher count, then lexicographic)", () => {
    const dataset = [
      seq("a", ["s", "b"]),
      seq("b", ["s", "b"]),
      seq("c", ["s", "a"]),
    ];
    const model = new MarkovMovementBackend(1).train(dataset);
    // "b" appears twice after "s", "a" once -> "b" wins on count.
    expect(model.predictNext(["s"])?.token).toBe("b");

    const tie = new MarkovMovementBackend(1).train([seq("x", ["s", "b"]), seq("y", ["s", "a"])]);
    // 1 vs 1 -> lexicographically smallest wins deterministically.
    expect(tie.predictNext(["s"])?.token).toBe("a");
  });

  it("returns undefined when there is no data at all", () => {
    const model = new MarkovMovementBackend(2).train([]);
    expect(model.predictNext(["anything"])).toBeUndefined();
    expect(model.generate(["anything"])).toEqual([]);
  });

  it("round-trips through serialize/load with identical behavior", () => {
    const dataset = [
      seq("a", ["p", "q", "r"]),
      seq("b", ["p", "q", "s"]),
    ];
    const model = new MarkovMovementBackend(2).train(dataset);
    const snapshot = model.serialize();
    // Snapshot ordering is stable/deterministic.
    expect(model.serialize()).toEqual(snapshot);

    const restored = loadMovementModel(snapshot);
    expect(restored.order).toBe(model.order);
    expect(restored.generate(["p"])).toEqual(model.generate(["p"]));
    expect(restored.predictNext(["p", "q"])).toEqual(model.predictNext(["p", "q"]));
  });
});

describe("evaluateMovementModel", () => {
  it("reports perfect fidelity when the model has memorized the movements", () => {
    const dataset = [
      seq("a", ["m:open", "m:edit", "m:save"]),
      seq("b", ["m:open", "m:edit", "m:save"]),
    ];
    const model = new MarkovMovementBackend(2).train(dataset);
    const cases = buildHeldOutEvalCases(dataset, 1);
    expect(cases).toHaveLength(2);

    const report = evaluateMovementModel(model, cases);
    expect(report.caseCount).toBe(2);
    expect(report.exactMatchRate).toBe(1);
    expect(report.meanTokenAccuracy).toBe(1);
    expect(report.firstStepAccuracy).toBe(1);
  });

  it("penalizes a wrong continuation in token accuracy", () => {
    const model = new MarkovMovementBackend(2).train([seq("a", ["one", "two", "three"])]);
    const report = evaluateMovementModel(model, [
      { id: "mismatch", seed: ["one"], expected: ["two", "nope"] },
    ]);
    expect(report.cases[0]?.firstStepMatch).toBe(true);
    expect(report.cases[0]?.exactMatch).toBe(false);
    expect(report.meanTokenAccuracy).toBeCloseTo(0.5, 5);
  });

  it("skips sequences too short to hold out", () => {
    expect(buildHeldOutEvalCases([seq("short", ["only"])], 1)).toEqual([]);
  });
});
