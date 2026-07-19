import { describe, expect, it } from "vitest";
import { buildReplayManifest } from "../capture/replay.js";
import { buildTrajectorySpan } from "../capture/trajectory.js";
import type { TrajectoryAction } from "../capture/trajectory.js";
import {
  MarkovMovementBackend,
  buildMovementDatasetFromReplay,
  buildMovementDatasetFromTrajectories,
  createMovementModelBackend,
  movementTokenFromAction,
  serializeMovementToken,
  type MovementDataset,
  type MovementToken,
} from "./movement-model.js";

function seq(id: string, tokens: MovementToken[]) {
  return { id, tokens };
}

function action(tool: string, ts: number, metadata?: Record<string, unknown>): TrajectoryAction {
  return { kind: "action", tool, summary: tool, ts, ...(metadata ? { metadata } : {}) };
}

describe("serializeMovementToken", () => {
  it("encodes kind, target, and direction distinctly", () => {
    expect(serializeMovementToken({ kind: "tap" })).toBe("tap");
    expect(serializeMovementToken({ kind: "tap", target: "save" })).toBe("tap@save");
    expect(serializeMovementToken({ kind: "swipe", direction: "down" })).toBe("swipe^down");
    expect(serializeMovementToken({ kind: "type", target: "search", direction: "x" })).toBe("type@search^x");
  });
});

describe("dataset construction", () => {
  it("extracts gesture metadata from trajectory actions, ordered by timestamp", () => {
    const trajectory = buildTrajectorySpan({
      id: "t1",
      sessionId: "s1",
      actions: [
        action("device", 20, { gesture: "swipe", direction: "down" }),
        action("device", 10, { gesture: "tap", target: "menu" }),
      ],
    });
    const dataset = buildMovementDatasetFromTrajectories([trajectory]);
    expect(dataset.sequences).toHaveLength(1);
    expect(dataset.sequences[0].tokens).toEqual([
      { kind: "tap", target: "menu" },
      { kind: "swipe", direction: "down" },
    ]);
  });

  it("falls back to the action tool when no gesture metadata is present", () => {
    expect(movementTokenFromAction(action("Bash", 1))).toEqual({ kind: "Bash" });
  });

  it("drops trajectories with no actions", () => {
    const empty = buildTrajectorySpan({ id: "t2", sessionId: "s1", actions: [] });
    expect(buildMovementDatasetFromTrajectories([empty]).sequences).toHaveLength(0);
  });

  it("round-trips a dataset through a replay manifest", () => {
    // The replay manifest carries only tool + human summary, so the sequence is
    // recovered by parsing the device adapter's gesture summaries.
    const trajectory = buildTrajectorySpan({
      id: "t1",
      sessionId: "s1",
      actions: [
        { kind: "action", tool: "device", summary: "tapped menu", ts: 10 },
        { kind: "action", tool: "device", summary: "swiped down", ts: 20 },
      ],
    });
    const manifest = buildReplayManifest({ sessionId: "s1", transcript: [], trajectories: [trajectory] });
    const dataset = buildMovementDatasetFromReplay(manifest);
    expect(dataset.sequences[0].tokens).toEqual([
      { kind: "tap", target: "menu" },
      { kind: "swipe", direction: "down" },
    ]);
  });
});

describe("MarkovMovementBackend — repeat recorded movements (objective 2c)", () => {
  const dataset: MovementDataset = {
    version: 1,
    sequences: [
      seq("a", [
        { kind: "tap", target: "menu" },
        { kind: "tap", target: "settings" },
        { kind: "swipe", direction: "down" },
        { kind: "tap", target: "save" },
      ]),
    ],
  };

  it("predicts the exact next recorded movement", () => {
    const model = new MarkovMovementBackend().train(dataset, { maxOrder: 3 });
    const prediction = model.predictNext([
      { kind: "tap", target: "menu" },
      { kind: "tap", target: "settings" },
    ]);
    expect(prediction?.token).toEqual({ kind: "swipe", direction: "down" });
    expect(prediction?.source).toBe("exact");
    expect(prediction?.confidence).toBe(1);
  });

  it("regenerates the full recorded sequence from a seed", () => {
    const model = new MarkovMovementBackend().train(dataset, { maxOrder: 3 });
    const rollout = model.generate([{ kind: "tap", target: "menu" }], 3);
    expect(rollout).toEqual([
      { kind: "tap", target: "settings" },
      { kind: "swipe", direction: "down" },
      { kind: "tap", target: "save" },
    ]);
  });

  it("is deterministic across repeated training + inference", () => {
    const a = new MarkovMovementBackend().train(dataset).generate([{ kind: "tap", target: "menu" }], 3);
    const b = new MarkovMovementBackend().train(dataset).generate([{ kind: "tap", target: "menu" }], 3);
    expect(a).toEqual(b);
  });
});

describe("MarkovMovementBackend — backoff", () => {
  it("backs off to a shorter context when the full context is unseen", () => {
    const dataset: MovementDataset = {
      version: 1,
      sequences: [
        seq("a", [
          { kind: "tap", target: "a" },
          { kind: "tap", target: "b" },
        ]),
      ],
    };
    const model = new MarkovMovementBackend().train(dataset, { maxOrder: 3 });
    // Context ["tap@z", "tap@a"] was never seen at order 2, but "tap@a" -> "tap@b" was at order 1.
    const prediction = model.predictNext([
      { kind: "tap", target: "z" },
      { kind: "tap", target: "a" },
    ]);
    expect(prediction?.token).toEqual({ kind: "tap", target: "b" });
    expect(prediction?.source).toBe("backoff");
    expect(prediction?.order).toBe(1);
  });
});

describe("MarkovMovementBackend — generalize to related movements (objective 2d)", () => {
  it("predicts a related movement of the learned kind for an unseen target", () => {
    const dataset: MovementDataset = {
      version: 1,
      sequences: [
        seq("a", [
          { kind: "tap", target: "compose" },
          { kind: "type", target: "body" },
        ]),
        seq("b", [
          { kind: "tap", target: "reply" },
          { kind: "type", target: "body" },
        ]),
      ],
    };
    const model = new MarkovMovementBackend().train(dataset, { maxOrder: 2 });
    // "tap@forward" was never recorded, but after any tap the model learned a `type` follows.
    const prediction = model.predictNext([{ kind: "tap", target: "forward" }]);
    expect(prediction?.token.kind).toBe("type");
    expect(prediction?.source).toBe("generalized");
    expect(prediction?.confidence).toBeGreaterThan(0);
    expect(prediction?.confidence).toBeLessThan(1);
  });

  it("cold-starts from an empty context with the most frequent movement", () => {
    const dataset: MovementDataset = {
      version: 1,
      sequences: [
        seq("a", [{ kind: "tap", target: "home" }, { kind: "tap", target: "home" }]),
        seq("b", [{ kind: "swipe", direction: "up" }]),
      ],
    };
    const model = new MarkovMovementBackend().train(dataset);
    const prediction = model.predictNext([]);
    expect(prediction?.token).toEqual({ kind: "tap", target: "home" });
  });

  it("returns undefined for an unseen context with no generalization path", () => {
    const model = new MarkovMovementBackend().train({ version: 1, sequences: [] });
    expect(model.predictNext([{ kind: "tap" }])).toBeUndefined();
  });
});

describe("MarkovMovementBackend — serialization", () => {
  it("round-trips through JSON with identical predictions", () => {
    const dataset: MovementDataset = {
      version: 1,
      sequences: [
        seq("a", [
          { kind: "tap", target: "menu" },
          { kind: "swipe", direction: "down" },
          { kind: "tap", target: "save" },
        ]),
      ],
    };
    const model = new MarkovMovementBackend().train(dataset, { maxOrder: 3 });
    const snapshot = model.toJSON();
    const restored = MarkovMovementBackend.fromJSON(JSON.parse(JSON.stringify(snapshot)));

    const context = [{ kind: "tap", target: "menu" }, { kind: "swipe", direction: "down" }];
    expect(restored.predictNext(context)).toEqual(model.predictNext(context));
    expect(restored.generate([{ kind: "tap", target: "menu" }], 2)).toEqual(
      model.generate([{ kind: "tap", target: "menu" }], 2),
    );
    expect(snapshot.backendId).toBe("markov");
  });
});

describe("createMovementModelBackend", () => {
  it("resolves the markov backend by default", () => {
    expect(createMovementModelBackend().id).toBe("markov");
    expect(createMovementModelBackend("markov").id).toBe("markov");
  });
});
