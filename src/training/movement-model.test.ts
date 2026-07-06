import { describe, expect, it } from "vitest";
import type { ReplayManifest, ReplayTimelineEvent } from "../capture/replay.js";
import {
  NgramMovementBackend,
  buildMovementDataset,
  defaultMovementTokenizer,
  evaluateMovementModel,
  rolloutMovementSequence,
  type MovementSequence,
} from "./movement-model.js";

function action(trajectoryId: string, ts: number, tool: string, summary: string): ReplayTimelineEvent {
  return { kind: "action", ts, trajectoryId, tool, summary };
}

function replay(sessionId: string, events: ReplayTimelineEvent[]): ReplayManifest {
  const trajectoryIds = [...new Set(events.flatMap((e) => (e.kind === "action" ? [e.trajectoryId] : [])))];
  return { version: 1, sessionId, trajectoryIds, eventCount: events.length, events };
}

function seq(trajectoryId: string, tokens: string[]): MovementSequence {
  return { trajectoryId, sessionId: "s", tokens };
}

describe("buildMovementDataset", () => {
  it("groups action events per trajectory, orders by timestamp, and tokenizes", () => {
    const manifest = replay("s1", [
      action("t1", 30, "device", "scrolled down"),
      action("t1", 10, "device", "tapped Send"),
      action("t1", 20, "device", "typed message"),
      // a non-action event is ignored
      { kind: "observation", ts: 5, trajectoryId: "t1", source: "os", summary: "focused editor" },
    ]);

    const dataset = buildMovementDataset([manifest]);

    expect(dataset.sequences).toHaveLength(1);
    expect(dataset.sequences[0]!.tokens).toEqual(["device:tapped", "device:typed", "device:scrolled"]);
    expect(dataset.vocabulary).toEqual(["device:scrolled", "device:tapped", "device:typed"]);
  });

  it("keeps trajectories from different sessions separate and orders deterministically", () => {
    const dataset = buildMovementDataset([
      replay("b", [action("t2", 1, "device", "swiped up")]),
      replay("a", [action("t1", 1, "device", "tapped ok")]),
    ]);
    expect(dataset.sequences.map((s) => s.sessionId)).toEqual(["a", "b"]);
  });

  it("respects a custom tokenizer and minLength", () => {
    const dataset = buildMovementDataset(
      [
        replay("s", [action("t1", 1, "device", "tapped ok")]),
        replay("s", [action("t2", 1, "device", "a"), action("t2", 2, "device", "b")]),
      ],
      { tokenizer: (a) => a.tool, minLength: 2 },
    );
    expect(dataset.sequences).toHaveLength(1);
    expect(dataset.sequences[0]!.tokens).toEqual(["device", "device"]);
  });
});

describe("defaultMovementTokenizer", () => {
  it("falls back to the bare tool when the summary is empty", () => {
    expect(defaultMovementTokenizer({ kind: "action", ts: 0, trajectoryId: "t", tool: "device", summary: "  " })).toBe(
      "device",
    );
  });
});

describe("NgramMovementBackend training + prediction", () => {
  it("predicts the deterministic next movement from a learned context", () => {
    const backend = new NgramMovementBackend(2);
    const model = backend.train(buildMovementDataset([
      replay("s", [
        action("t1", 1, "app", "open editor"),
        action("t1", 2, "app", "edit file"),
        action("t1", 3, "app", "save file"),
      ]),
    ]));

    const prediction = backend.predict(model, ["app:open", "app:edit"]);
    expect(prediction.token).toBe("app:save");
    expect(prediction.probability).toBe(1);
    expect(prediction.backoffOrder).toBe(2);
  });

  it("backs off to a shorter context for an unseen prefix", () => {
    const backend = new NgramMovementBackend(2);
    const model = backend.train(buildMovementDataset([
      replay("s", [action("t1", 1, "app", "edit file"), action("t1", 2, "app", "save file")]),
    ]));

    // "app:open" was never seen, but the length-1 context "app:edit" was.
    const prediction = backend.predict(model, ["app:open", "app:edit"]);
    expect(prediction.token).toBe("app:save");
    expect(prediction.backoffOrder).toBe(1);
  });

  it("breaks ties deterministically by count then lexicographically", () => {
    const backend = new NgramMovementBackend(1);
    const model = backend.train({
      version: 1,
      vocabulary: ["a", "x", "z"],
      sequences: [seq("t1", ["a", "z"]), seq("t2", ["a", "x"])],
    });
    // Both "x" and "z" follow "a" once → lexicographically smaller wins.
    const prediction = backend.predict(model, ["a"]);
    expect(prediction.token).toBe("x");
    expect(prediction.candidates.map((c) => c.token)).toEqual(["x", "z"]);
  });

  it("returns an empty prediction for an untrained model", () => {
    const backend = new NgramMovementBackend();
    const model = backend.train({ version: 1, vocabulary: [], sequences: [] });
    expect(backend.predict(model, ["anything"]).token).toBeUndefined();
  });

  it("round-trips through JSON (serializable model)", () => {
    const backend = new NgramMovementBackend(2);
    const model = backend.train(buildMovementDataset([
      replay("s", [action("t1", 1, "app", "open x"), action("t1", 2, "app", "close x")]),
    ]));
    const revived = JSON.parse(JSON.stringify(model));
    expect(backend.predict(revived, ["app:open"]).token).toBe("app:close");
  });
});

describe("rolloutMovementSequence", () => {
  it("reproduces a recorded movement exactly from its seed (repeat)", () => {
    const backend = new NgramMovementBackend(2);
    const dataset = buildMovementDataset([
      replay("s", [
        action("t1", 1, "app", "open editor"),
        action("t1", 2, "app", "edit file"),
        action("t1", 3, "app", "save file"),
        action("t1", 4, "app", "close editor"),
      ]),
    ]);
    const model = backend.train(dataset);
    const rollout = rolloutMovementSequence(backend, model, { seed: ["app:open"] });
    expect(rollout).toEqual(["app:open", "app:edit", "app:save", "app:close"]);
  });

  it("starts from the learned start token when unseeded", () => {
    const backend = new NgramMovementBackend(2);
    const model = backend.train(buildMovementDataset([
      replay("s", [action("t1", 1, "app", "open editor"), action("t1", 2, "app", "save file")]),
    ]));
    expect(rolloutMovementSequence(backend, model)[0]).toBe("app:open");
  });

  it("honors maxLength and stopToken bounds", () => {
    const backend = new NgramMovementBackend(1);
    // A self-loop would run forever without the maxLength safety bound.
    const model = backend.train({ version: 1, vocabulary: ["a"], sequences: [seq("t1", ["a", "a", "a"])] });
    expect(rolloutMovementSequence(backend, model, { seed: ["a"], maxLength: 2 })).toEqual(["a", "a", "a"]);
    expect(rolloutMovementSequence(backend, model, { seed: ["a"], stopToken: "a" })).toEqual(["a"]);
  });
});

describe("evaluateMovementModel (generalization harness)", () => {
  it("scores perfect next-token accuracy on the training path", () => {
    const backend = new NgramMovementBackend(2);
    const dataset = buildMovementDataset([
      replay("s", [
        action("t1", 1, "app", "open editor"),
        action("t1", 2, "app", "edit file"),
        action("t1", 3, "app", "save file"),
      ]),
    ]);
    const model = backend.train(dataset);
    const result = evaluateMovementModel(backend, model, dataset.sequences);
    expect(result.accuracy).toBe(1);
    expect(result.correct).toBe(2);
    expect(result.total).toBe(2);
  });

  it("generalizes to a held-out related sequence via back-off", () => {
    const backend = new NgramMovementBackend(2);
    // Train on two related flows that share the "edit -> save" transition.
    const train = buildMovementDataset([
      replay("a", [
        action("t1", 1, "app", "open editor"),
        action("t1", 2, "app", "edit file"),
        action("t1", 3, "app", "save file"),
      ]),
      replay("b", [
        action("t2", 1, "app", "review file"),
        action("t2", 2, "app", "edit file"),
        action("t2", 3, "app", "save file"),
      ]),
    ]);
    const model = backend.train(train);
    // Held-out flow that follows the familiar "review -> edit -> save" path
    // (learned across the two training flows) then diverges into a movement the
    // model has never seen ("close").
    const heldOut: MovementSequence[] = [seq("h1", ["app:review", "app:edit", "app:save", "app:close"])];
    const result = evaluateMovementModel(backend, model, heldOut);
    // review->edit and edit->save generalize correctly; only the novel
    // save->close transition is a miss.
    expect(result.correct).toBe(2);
    expect(result.total).toBe(3);
    expect(result.accuracy).toBeCloseTo(2 / 3, 5);
  });

  it("returns zero accuracy for an empty held-out set", () => {
    const backend = new NgramMovementBackend();
    const model = backend.train({ version: 1, vocabulary: [], sequences: [] });
    expect(evaluateMovementModel(backend, model, []).accuracy).toBe(0);
  });
});
