import { describe, expect, it } from "vitest";
import {
  MARKOV_BACKEND_ID,
  MarkovMovementBackend,
  MovementModelRegistry,
  buildMovementDataset,
  createDefaultMovementModelRegistry,
  defaultMovementTokenizer,
  evaluateReplayFidelity,
  type MovementDataset,
} from "./movement-model.js";

// --- synthetic event-stream helpers (no OS access) ---------------------------

type SyntheticAction = { trajectoryId: string; ts: number; tool: string; summary: string };

function replayFromActions(sessionId: string, actions: SyntheticAction[]) {
  return {
    sessionId,
    events: actions.map((action) => ({
      kind: "action" as const,
      ts: action.ts,
      trajectoryId: action.trajectoryId,
      tool: action.tool,
      summary: action.summary,
    })),
  };
}

function gestureStream(trajectoryId: string, gestures: string[], startTs = 1000): SyntheticAction[] {
  return gestures.map((summary, index) => ({
    trajectoryId,
    ts: startTs + index * 10,
    tool: "device",
    summary,
  }));
}

describe("buildMovementDataset", () => {
  it("groups action events into per-trajectory token sequences ordered by ts", () => {
    const replay = replayFromActions("session-1", [
      { trajectoryId: "t1", ts: 30, tool: "device", summary: "tapped submit" },
      { trajectoryId: "t1", ts: 10, tool: "device", summary: "tapped field" },
      { trajectoryId: "t1", ts: 20, tool: "device", summary: "typed hello" },
      { trajectoryId: "t2", ts: 5, tool: "device", summary: "scrolled down" },
    ]);

    const dataset = buildMovementDataset([replay]);

    expect(dataset.sequences).toHaveLength(2);
    const t1 = dataset.sequences.find((s) => s.trajectoryId === "t1");
    expect(t1?.tokens).toEqual(["device::tapped field", "device::typed hello", "device::tapped submit"]);
    expect(dataset.vocabulary).toContain("device::scrolled down");
    // Vocabulary is sorted and de-duplicated.
    expect([...dataset.vocabulary].sort()).toEqual(dataset.vocabulary);
  });

  it("ignores non-action and untagged events", () => {
    const dataset = buildMovementDataset([
      {
        sessionId: "s",
        events: [
          { kind: "transcript", ts: 1, summary: "noise" },
          { kind: "observation", ts: 2, trajectoryId: "t", summary: "saw screen" },
          { kind: "action", ts: 3, tool: "device", summary: "tap" }, // no trajectoryId
          { kind: "action", ts: 4, trajectoryId: "t", tool: "device", summary: "tap real" },
        ],
      },
    ]);
    expect(dataset.sequences).toHaveLength(1);
    expect(dataset.sequences[0]?.tokens).toEqual(["device::tap real"]);
  });

  it("supports a custom tokenizer", () => {
    const replay = replayFromActions("s", [{ trajectoryId: "t", ts: 1, tool: "device", summary: "tap A" }]);
    const dataset = buildMovementDataset([replay], (event) => event.summary.toUpperCase());
    expect(dataset.sequences[0]?.tokens).toEqual(["TAP A"]);
  });
});

describe("MarkovMovementBackend replay fidelity", () => {
  it("reproduces a recorded movement sequence deterministically (objective 2c)", () => {
    const backend = new MarkovMovementBackend();
    const dataset = buildMovementDataset([
      replayFromActions("s", gestureStream("t1", ["open app", "tap field", "type query", "tap submit", "read result"])),
    ]);
    const model = backend.train(dataset);

    // Seeded with the first movement, generation reproduces the rest.
    const generated = backend.generate(model, ["device::open app"], 10);
    expect(generated).toEqual([
      "device::tap field",
      "device::type query",
      "device::tap submit",
      "device::read result",
    ]);

    // Next-token accuracy on the training sequence is perfect.
    const fidelity = evaluateReplayFidelity(backend, model, dataset.sequences);
    expect(fidelity.accuracy).toBe(1);
    expect(fidelity.totalPredictions).toBeGreaterThan(0);
  });

  it("generates from an empty seed using start-of-sequence statistics", () => {
    const backend = new MarkovMovementBackend();
    const dataset = buildMovementDataset([
      replayFromActions("s", gestureStream("t1", ["launch", "click", "confirm"])),
    ]);
    const model = backend.train(dataset);
    expect(backend.generate(model, [], 5)).toEqual(["device::launch", "device::click", "device::confirm"]);
  });
});

describe("MarkovMovementBackend generalization", () => {
  it("backs off to a shorter context for new-but-related movements (objective 2d)", () => {
    const backend = new MarkovMovementBackend();
    // Train: in every flow, "type query" is followed by "tap submit".
    const dataset = buildMovementDataset([
      replayFromActions("s1", gestureStream("t1", ["open app", "tap field", "type query", "tap submit"])),
      replayFromActions("s2", gestureStream("t2", ["open settings", "tap search", "type query", "tap submit"])),
    ]);
    const model = backend.train(dataset);

    // A never-seen prefix ("scroll list") followed by the known "type query":
    // the full n-gram was never recorded, so the model must back off.
    const prediction = backend.predictNext(model, ["device::scroll list", "device::type query"]);
    expect(prediction.token).toBe("device::tap submit");
    expect(prediction.backedOff).toBe(true);
    expect(prediction.contextOrder).toBeLessThan(2);
  });

  it("measurably generalizes to a held-out related trajectory", () => {
    const backend = new MarkovMovementBackend();
    const train = buildMovementDataset([
      replayFromActions("s1", gestureStream("t1", ["open", "search", "type", "submit", "wait", "read"])),
      replayFromActions("s2", gestureStream("t2", ["open", "search", "type", "submit", "wait", "read"])),
    ]);
    const model = backend.train(train);

    // Held-out trajectory shares the same sub-movements in the same order.
    const heldOut = buildMovementDataset([
      replayFromActions("s3", gestureStream("t3", ["open", "search", "type", "submit", "wait", "read"])),
    ]);
    const fidelity = evaluateReplayFidelity(backend, model, heldOut.sequences);
    expect(fidelity.accuracy).toBeGreaterThanOrEqual(0.8);
  });

  it("returns a null prediction when nothing was ever trained", () => {
    const backend = new MarkovMovementBackend();
    const empty: MovementDataset = { version: 1, sequences: [], vocabulary: [] };
    const model = backend.train(empty);
    const prediction = backend.predictNext(model, ["device::anything"]);
    expect(prediction.token).toBeNull();
    expect(prediction.ranked).toEqual([]);
    expect(backend.generate(model, [], 3)).toEqual([]);
  });
});

describe("MarkovMovementBackend serialization", () => {
  it("round-trips a model through serialize/deserialize", () => {
    const backend = new MarkovMovementBackend();
    const dataset = buildMovementDataset([
      replayFromActions("s", gestureStream("t1", ["a", "b", "c"])),
    ]);
    const model = backend.train(dataset);
    const restored = backend.deserialize(backend.serialize(model));

    expect(restored.backendId).toBe(MARKOV_BACKEND_ID);
    expect(backend.generate(restored, ["device::a"], 5)).toEqual(["device::b", "device::c"]);
  });

  it("rejects a serialized model from a different backend", () => {
    const backend = new MarkovMovementBackend();
    expect(() => backend.deserialize(JSON.stringify({ backendId: "other" }))).toThrow(/backend/);
  });
});

describe("MovementModelRegistry", () => {
  it("registers, retrieves and lists pluggable backends", () => {
    const registry = createDefaultMovementModelRegistry();
    expect(registry.list()).toContain(MARKOV_BACKEND_ID);
    expect(registry.require(MARKOV_BACKEND_ID).id).toBe(MARKOV_BACKEND_ID);
    expect(registry.get("missing")).toBeUndefined();
    expect(() => registry.require("missing")).toThrow(/No movement-model backend/);
  });

  it("allows a custom backend to be swapped in via the same interface", () => {
    const registry = new MovementModelRegistry();
    const stub = {
      id: "stub",
      train: () => ({ backendId: "stub" }),
      predictNext: () => ({ token: "x", probability: 1, contextOrder: 0, backedOff: false, ranked: [] }),
      generate: () => ["x"],
      serialize: () => "{}",
      deserialize: () => ({ backendId: "stub" }),
    };
    registry.register(stub);
    expect(registry.require("stub").generate({ backendId: "stub" }, [], 1)).toEqual(["x"]);
  });

  it("exposes the default tokenizer", () => {
    expect(defaultMovementTokenizer({ kind: "action", ts: 1, trajectoryId: "t", tool: "device", summary: "tap" })).toBe(
      "device::tap",
    );
  });
});
