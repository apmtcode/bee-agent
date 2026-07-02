import { describe, expect, it } from "vitest";
import { buildTrajectorySpan } from "../capture/trajectory.js";
import type { ReplayTimelineEvent } from "../capture/replay.js";
import {
  MarkovMovementBackend,
  MovementModelRegistry,
  buildMovementDataset,
  evaluateMovementModel,
  tokenizeReplayEvents,
  type MovementDataset,
} from "./movement-model.js";

function actionEvent(tool: string, summary: string, ts: number): ReplayTimelineEvent {
  return { kind: "action", ts, trajectoryId: "t", tool, summary };
}

describe("tokenizeReplayEvents", () => {
  it("emits action tokens only by default and normalizes summaries", () => {
    const tokens = tokenizeReplayEvents([
      { kind: "transcript", ts: 1, messageId: "m1", role: "user", content: "hi" },
      actionEvent("device", "Tapped  Login ", 2),
      { kind: "observation", ts: 3, trajectoryId: "t", source: "os", summary: "window" },
    ]);
    expect(tokens).toEqual(["action:device::tapped login"]);
  });

  it("can include observation context tokens", () => {
    const tokens = tokenizeReplayEvents(
      [{ kind: "observation", ts: 1, trajectoryId: "t", source: "os", summary: "Editor Focused" }],
      { include: ["observation", "action"] },
    );
    expect(tokens).toEqual(["observation:os::editor focused"]);
  });
});

describe("buildMovementDataset", () => {
  it("orders actions by timestamp and skips empty trajectories", () => {
    const dataset = buildMovementDataset([
      buildTrajectorySpan({
        id: "traj-1",
        sessionId: "s1",
        actions: [
          { kind: "action", tool: "device", summary: "click B", ts: 20 },
          { kind: "action", tool: "device", summary: "click A", ts: 10 },
        ],
      }),
      buildTrajectorySpan({ id: "traj-empty", sessionId: "s1" }),
    ]);
    expect(dataset.sequences).toHaveLength(1);
    expect(dataset.sequences[0]).toEqual({
      trajectoryId: "traj-1",
      tokens: ["action:device::click a", "action:device::click b"],
    });
  });
});

describe("MarkovMovementBackend", () => {
  const backend = new MarkovMovementBackend();

  it("repeats a recorded movement sequence exactly", () => {
    const dataset: MovementDataset = {
      version: 1,
      sequences: [{ tokens: ["a", "b", "c", "d"] }],
    };
    const model = backend.train(dataset, { order: 2 });
    expect(backend.generate(model, [])).toEqual(["a", "b", "c", "d"]);
  });

  it("is deterministic across retrains and serialization round-trips", () => {
    const dataset: MovementDataset = {
      version: 1,
      sequences: [{ tokens: ["open", "type", "save"] }, { tokens: ["open", "type", "send"] }],
    };
    const model1 = backend.train(dataset, { order: 2 });
    const model2 = backend.train(dataset, { order: 2 });
    expect(backend.serialize(model1)).toEqual(backend.serialize(model2));

    const restored = backend.deserialize(backend.serialize(model1));
    expect(backend.generate(restored, ["open"])).toEqual(backend.generate(model1, ["open"]));
  });

  it("generalizes to an unseen prefix via backoff", () => {
    // Trained: after "select" the recorded next move is always "confirm".
    const dataset: MovementDataset = {
      version: 1,
      sequences: [
        { tokens: ["login", "select", "confirm"] },
        { tokens: ["search", "select", "confirm"] },
      ],
    };
    const model = backend.train(dataset, { order: 3 });
    // Novel prefix "browse" was never seen, but the "select" suffix backs off
    // to the learned continuation.
    const prediction = backend.predict(model, ["browse", "select"]);
    expect(prediction.token).toBe("confirm");
    expect(prediction.backoffOrder).toBeLessThan(3);
    expect(prediction.probability).toBeGreaterThan(0);
  });

  it("predicts null (end-of-sequence) at a terminal context", () => {
    const model = backend.train({ version: 1, sequences: [{ tokens: ["a", "b"] }] }, { order: 2 });
    expect(backend.predict(model, ["a", "b"]).token).toBeNull();
  });

  it("returns an empty rollout for an empty dataset", () => {
    const model = backend.train({ version: 1, sequences: [] });
    expect(model.vocabulary).toEqual([]);
    expect(backend.generate(model, [])).toEqual([]);
    expect(backend.predict(model, []).token).toBeNull();
  });

  it("bounds runaway generation via maxLength", () => {
    // A self-looping movement (repeat -> repeat) would otherwise never end.
    const model = backend.train({ version: 1, sequences: [{ tokens: ["repeat", "repeat", "repeat"] }] }, { order: 1 });
    expect(backend.generate(model, [], { maxLength: 5 })).toHaveLength(5);
  });

  it("rejects a model serialized by a different backend", () => {
    expect(() => backend.deserialize(JSON.stringify({ backendId: "other" }))).toThrow(/backend mismatch/i);
  });
});

describe("evaluateMovementModel", () => {
  it("scores perfect next-token accuracy on the training sequence", () => {
    const backend = new MarkovMovementBackend();
    const sequences = [{ tokens: ["a", "b", "c"] }];
    const model = backend.train({ version: 1, sequences }, { order: 2 });
    const evaluation = evaluateMovementModel(backend, model, sequences);
    expect(evaluation.samples).toBe(3);
    expect(evaluation.accuracy).toBe(1);
  });

  it("scores partial accuracy on a related held-out sequence", () => {
    const backend = new MarkovMovementBackend();
    const model = backend.train(
      { version: 1, sequences: [{ tokens: ["a", "b", "c"] }] },
      { order: 2 },
    );
    const evaluation = evaluateMovementModel(backend, model, [{ tokens: ["a", "b", "z"] }]);
    // "a" and "b" are predicted correctly from the shared prefix; "z" is not.
    expect(evaluation.accuracy).toBeCloseTo(2 / 3, 5);
  });
});

describe("MovementModelRegistry", () => {
  it("resolves the default markov backend and reports availability", () => {
    const registry = new MovementModelRegistry();
    expect(registry.list()).toContain("markov");
    expect(registry.has("markov")).toBe(true);
    expect(registry.get("markov")).toBeInstanceOf(MarkovMovementBackend);
  });

  it("supports registering a pluggable custom backend", () => {
    const registry = new MovementModelRegistry();
    const custom = { ...new MarkovMovementBackend(), id: "on-device" } as MarkovMovementBackend;
    registry.register(custom);
    expect(registry.list()).toEqual(["markov", "on-device"]);
    expect(registry.get("on-device").id).toBe("on-device");
  });

  it("throws for an unknown backend id", () => {
    expect(() => new MovementModelRegistry().get("missing")).toThrow(/unknown movement-model backend/i);
  });
});
