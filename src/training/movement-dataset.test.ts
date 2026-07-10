import { describe, expect, it } from "vitest";
import type { ReplayTimelineEvent } from "../capture/replay.js";
import {
  buildMovementDataset,
  movementActionKey,
  movementDatasetFromExport,
  splitMovementDataset,
} from "./movement-dataset.js";

function events(): ReplayTimelineEvent[] {
  return [
    { kind: "observation", ts: 3, trajectoryId: "t1", source: "device", summary: "editor active" },
    { kind: "action", ts: 4, trajectoryId: "t1", tool: "device", summary: "clicked run" },
    { kind: "transcript", ts: 1, messageId: "m1", role: "user", content: "please build" },
    { kind: "action", ts: 5, trajectoryId: "t1", tool: "device", summary: "typed npm build" },
  ];
}

describe("buildMovementDataset", () => {
  it("orders events by timestamp and emits one transition per action", () => {
    const dataset = buildMovementDataset([
      { sessionId: "s1", trajectoryIds: ["t1"], events: events() },
    ]);
    expect(dataset.transitionCount).toBe(2);
    const [sequence] = dataset.sequences;
    expect(sequence.id).toBe("t1");
    expect(sequence.events.map((event) => event.ts)).toEqual([1, 3, 4, 5]);
    expect(sequence.transitions.map((transition) => transition.action.summary)).toEqual([
      "clicked run",
      "typed npm build",
    ]);
    expect(sequence.transitions.map((transition) => transition.index)).toEqual([0, 1]);
  });

  it("folds the bounded window of preceding events into the context signature", () => {
    const dataset = buildMovementDataset(
      [{ sessionId: "s1", trajectoryIds: ["t1"], events: events() }],
      { contextWindow: 2 },
    );
    const [first, second] = dataset.sequences[0].transitions;
    // first action at ts=4 is preceded by transcript(ts1) + observation(ts3)
    expect(first.context).toContain("msg:user:please build");
    expect(first.context).toContain("obs:device:editor active");
    expect(first.backoffContext).toBe("obs:device:editor active");
    // second action's window is the observation + the first action
    expect(second.context).toContain("act:device");
  });

  it("uses a start sentinel when an action has no preceding events", () => {
    const dataset = buildMovementDataset([
      {
        sessionId: "s1",
        trajectoryIds: ["t1"],
        events: [{ kind: "action", ts: 1, trajectoryId: "t1", tool: "device", summary: "tapped" }],
      },
    ]);
    expect(dataset.sequences[0].transitions[0].context).toBe("<start>");
    expect(dataset.sequences[0].transitions[0].backoffContext).toBe("<start>");
  });

  it("falls back to a synthetic sequence id when no trajectory id is present", () => {
    const dataset = buildMovementDataset([{ sessionId: "s1", trajectoryIds: [], events: events() }]);
    expect(dataset.sequences[0].id).toBe("s1#0");
  });

  it("builds from a reviewed export manifest shape", () => {
    const dataset = movementDatasetFromExport({
      replays: [{ sessionId: "s1", trajectoryIds: ["t1"], eventCount: 4, events: events() }],
    });
    expect(dataset.transitionCount).toBe(2);
  });
});

describe("splitMovementDataset", () => {
  it("routes every Nth sequence to the holdout partition deterministically", () => {
    const replays = Array.from({ length: 6 }, (_, index) => ({
      sessionId: `s${index}`,
      trajectoryIds: [`t${index}`],
      events: events(),
    }));
    const dataset = buildMovementDataset(replays);
    const { train, holdout } = splitMovementDataset(dataset, 3);
    expect(holdout.sequences.map((sequence) => sequence.id)).toEqual(["t2", "t5"]);
    expect(train.sequences.map((sequence) => sequence.id)).toEqual(["t0", "t1", "t3", "t4"]);
    expect(train.transitionCount + holdout.transitionCount).toBe(dataset.transitionCount);
  });
});

describe("movementActionKey", () => {
  it("is stable and distinguishes tool + summary", () => {
    expect(movementActionKey({ tool: "device", summary: "click" })).toBe("device::click");
    expect(movementActionKey({ tool: "device", summary: "click" })).not.toBe(
      movementActionKey({ tool: "device", summary: "type" }),
    );
  });
});
