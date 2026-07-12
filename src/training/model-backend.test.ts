import { describe, expect, it } from "vitest";
import type { ReplayManifest } from "../capture/replay.js";
import { buildTrajectorySpan } from "../capture/trajectory.js";
import {
  MarkovMovementBackend,
  MOVEMENT_END_TOKEN,
  datasetFromReplays,
  datasetFromTrajectories,
  type MovementDataset,
} from "./model-backend.js";

function makeDataset(sequences: string[][]): MovementDataset {
  return {
    version: 1,
    sequences: sequences.map((tokens, index) => ({ id: `seq-${index}`, tokens })),
  };
}

describe("MarkovMovementBackend", () => {
  it("trains deterministically — identical datasets produce identical snapshots", async () => {
    const backend = new MarkovMovementBackend();
    const dataset = makeDataset([
      ["move:a", "click:b", "type:c"],
      ["move:a", "click:b", "scroll:d"],
    ]);

    const first = await backend.train(dataset, { order: 2 });
    const second = await backend.train(dataset, { order: 2 });

    expect(second.serialize()).toEqual(first.serialize());
  });

  it("(c) repeats a recorded movement — generating from the first token reproduces the trajectory", async () => {
    const backend = new MarkovMovementBackend();
    const recorded = ["open:app", "move:window", "click:save", "confirm:dialog"];
    const model = await backend.train(makeDataset([recorded]), { order: 3 });

    const replayed = model.generate([recorded[0]!]);

    expect([recorded[0]!, ...replayed]).toEqual(recorded);
  });

  it("predicts end-of-sequence after a fully-recorded trajectory", async () => {
    const backend = new MarkovMovementBackend();
    const recorded = ["a", "b", "c"];
    const model = await backend.train(makeDataset([recorded]), { order: 3 });

    const prediction = model.predictNext(recorded);

    expect(prediction.token).toBeNull();
  });

  it("(d) generalizes — an unseen but related prefix backs off to a plausible continuation", async () => {
    const backend = new MarkovMovementBackend();
    // Two related demos share the "…-> focus:field -> type:text" motif.
    const model = await backend.train(
      makeDataset([
        ["open:login", "focus:field", "type:text", "click:submit"],
        ["open:search", "focus:field", "type:text", "click:go"],
      ]),
      { order: 2 },
    );

    // A prefix ending in the shared "focus:field" the model has seen mid-stream,
    // but reached via a brand-new opener it never saw at full order.
    const prediction = model.predictNext(["open:settings", "focus:field"]);

    expect(prediction.token).toBe("type:text");
    expect(prediction.backedOff).toBe(true);
    expect(prediction.order).toBeLessThan(2);
  });

  it("falls back to the unigram distribution for a wholly unseen context", async () => {
    const backend = new MarkovMovementBackend();
    const model = await backend.train(
      makeDataset([
        ["step:x", "step:x", "step:x", "step:y"],
      ]),
      { order: 2 },
    );

    // "step:x" is by far the most common token, so a novel context predicts it.
    const prediction = model.predictNext(["totally:novel"]);

    expect(prediction.token).toBe("step:x");
    expect(prediction.order).toBe(0);
    expect(prediction.backedOff).toBe(true);
  });

  it("round-trips through serialize/restore with identical predictions", async () => {
    const backend = new MarkovMovementBackend();
    const dataset = makeDataset([
      ["a", "b", "c"],
      ["a", "b", "d"],
    ]);
    const model = await backend.train(dataset, { order: 2 });
    const snapshot = JSON.parse(JSON.stringify(model.serialize()));

    const restored = backend.restore(snapshot);

    expect(restored.order).toBe(model.order);
    expect(restored.predictNext(["a", "b"])).toEqual(model.predictNext(["a", "b"]));
    expect(restored.generate(["a"])).toEqual(model.generate(["a"]));
  });

  it("rejects snapshots from an incompatible backend", () => {
    const backend = new MarkovMovementBackend();
    expect(() =>
      backend.restore({
        version: 1,
        backendId: "some-other-backend",
        order: 1,
        tables: [],
        stats: { sequenceCount: 0, tokenCount: 0 },
      }),
    ).toThrow(/not compatible/);
  });

  it("does not emit the end sentinel as a generated movement", async () => {
    const backend = new MarkovMovementBackend();
    const model = await backend.train(makeDataset([["a", "b"]]), { order: 2 });
    const generated = model.generate(["a"], { maxSteps: 10 });
    expect(generated).not.toContain(MOVEMENT_END_TOKEN);
  });

  it("honors maxSteps when generating an unbounded loop", async () => {
    const backend = new MarkovMovementBackend();
    // A self-referential dataset would loop forever without the cap.
    const model = await backend.train(makeDataset([["loop", "loop", "loop", "loop"]]), { order: 1 });
    const generated = model.generate(["loop"], { maxSteps: 5, stopAtEnd: false });
    expect(generated).toHaveLength(5);
  });
});

describe("dataset builders", () => {
  it("builds a movement dataset from replay manifests, excluding transcript lines", () => {
    const replay: Pick<ReplayManifest, "sessionId" | "trajectoryIds" | "events"> = {
      sessionId: "sess-1",
      trajectoryIds: ["traj-1"],
      events: [
        { kind: "transcript", ts: 1, messageId: "m1", role: "user", content: "go" },
        { kind: "observation", ts: 2, trajectoryId: "traj-1", source: "screen", summary: "saw button" },
        { kind: "action", ts: 3, trajectoryId: "traj-1", tool: "mouse", summary: "click 10,20" },
      ],
    };

    const dataset = datasetFromReplays([replay]);

    expect(dataset.sequences).toEqual([
      { id: "traj-1", tokens: ["obs:screen:saw button", "act:mouse:click 10,20"] },
    ]);
  });

  it("builds a movement dataset from trajectory spans, ordered by timestamp", () => {
    const span = buildTrajectorySpan({
      id: "traj-2",
      sessionId: "sess-2",
      actions: [
        { kind: "action", tool: "keyboard", summary: "type b", ts: 20 },
        { kind: "action", tool: "keyboard", summary: "type a", ts: 10 },
      ],
    });

    const dataset = datasetFromTrajectories([span]);

    expect(dataset.sequences).toEqual([
      { id: "traj-2", tokens: ["act:keyboard:type a", "act:keyboard:type b"] },
    ]);
  });

  it("round-trips a replay dataset through training and reproduces the recorded movements", async () => {
    const replay: Pick<ReplayManifest, "sessionId" | "trajectoryIds" | "events"> = {
      sessionId: "sess-3",
      trajectoryIds: ["traj-3"],
      events: [
        { kind: "action", ts: 1, trajectoryId: "traj-3", tool: "mouse", summary: "move 0,0" },
        { kind: "action", ts: 2, trajectoryId: "traj-3", tool: "mouse", summary: "move 5,5" },
        { kind: "action", ts: 3, trajectoryId: "traj-3", tool: "mouse", summary: "click" },
      ],
    };
    const dataset = datasetFromReplays([replay]);
    const model = await new MarkovMovementBackend().train(dataset, { order: 3 });

    const start = dataset.sequences[0]!.tokens[0]!;
    expect([start, ...model.generate([start])]).toEqual(dataset.sequences[0]!.tokens);
  });
});
