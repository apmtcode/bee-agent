import { describe, expect, it } from "vitest";
import type { TrajectoryAction, TrajectorySpan } from "../capture/trajectory.js";
import {
  MarkovMovementBackend,
  MOVEMENT_END_TOKEN,
  MOVEMENT_START_TOKEN,
  buildMovementDataset,
  createMovementBackend,
  defaultMovementTokenizer,
  evaluateMovementModel,
  listMovementBackends,
  loadMovementModel,
  registerMovementBackend,
  tokenizeTrajectory,
  type MovementModel,
  type MovementModelBackend,
} from "./movement-model.js";

/** Build a deterministic synthetic movement trajectory from gesture kinds. */
function synthTrajectory(
  id: string,
  gestures: Array<{ kind: string; target?: string }>,
  startTs = 1_000,
): TrajectorySpan {
  const actions: TrajectoryAction[] = gestures.map((gesture, index) => ({
    kind: "action",
    tool: "device",
    summary: `${gesture.kind}${gesture.target ? ` ${gesture.target}` : ""}`,
    ts: startTs + index * 10,
    metadata: {
      gesture: gesture.kind,
      ...(gesture.target ? { target: gesture.target } : {}),
    },
  }));
  return {
    id,
    sessionId: `session-${id}`,
    createdAt: "2026-01-01T00:00:00.000Z",
    captureTier: "full",
    observations: [],
    actions,
  };
}

describe("movement tokenizer", () => {
  it("derives stable tokens from gesture metadata and orders by timestamp", () => {
    const trajectory = synthTrajectory("t1", [
      { kind: "tap", target: "compose" },
      { kind: "type", target: "subject" },
      { kind: "shortcut", target: "send" },
    ]);
    // Shuffle action order to prove tokenize sorts by ts.
    trajectory.actions.reverse();

    const sample = tokenizeTrajectory(trajectory);
    expect(sample.tokens).toEqual([
      MOVEMENT_START_TOKEN,
      "device:tap:compose",
      "device:type:subject",
      "device:shortcut:send",
      MOVEMENT_END_TOKEN,
    ]);
  });

  it("falls back to a summary slug when no gesture metadata is present", () => {
    const action: TrajectoryAction = {
      kind: "action",
      tool: "editor",
      summary: "Open Command Palette",
      ts: 5,
    };
    expect(defaultMovementTokenizer(action)).toBe("editor:open-command-palette");
  });
});

describe("MarkovMovementBackend replay (objective 2c)", () => {
  it("reproduces a single recorded trajectory exactly", () => {
    const trajectory = synthTrajectory("t1", [
      { kind: "tap", target: "inbox" },
      { kind: "swipe", target: "left" },
      { kind: "tap", target: "archive" },
    ]);
    const dataset = buildMovementDataset([trajectory]);
    const model = new MarkovMovementBackend().train(dataset);

    const replay = model.generate([MOVEMENT_START_TOKEN]);
    expect(replay).toEqual(["device:tap:inbox", "device:swipe:left", "device:tap:archive"]);
  });

  it("is deterministic across repeated training runs", () => {
    const trajectories = [
      synthTrajectory("a", [{ kind: "tap", target: "x" }, { kind: "type", target: "y" }]),
      synthTrajectory("b", [{ kind: "tap", target: "x" }, { kind: "scroll", target: "down" }]),
    ];
    const dataset = buildMovementDataset(trajectories);
    const first = new MarkovMovementBackend().train(dataset).serialize();
    const second = new MarkovMovementBackend().train(dataset).serialize();
    expect(first).toEqual(second);
  });
});

describe("MarkovMovementBackend generalization (objective 2d)", () => {
  it("composes a novel continuation from learned transitions", () => {
    // Two trajectories share the prefix tap:open -> type:query, then diverge.
    // A model trained on both, seeded with the shared prefix, should predict the
    // most-frequent continuation even though the exact full sequence below never
    // appeared verbatim in a single training sample.
    const trajectories = [
      synthTrajectory("a", [
        { kind: "tap", target: "open" },
        { kind: "type", target: "query" },
        { kind: "tap", target: "result" },
      ]),
      synthTrajectory("b", [
        { kind: "tap", target: "open" },
        { kind: "type", target: "query" },
        { kind: "tap", target: "result" },
      ]),
      synthTrajectory("c", [
        { kind: "tap", target: "open" },
        { kind: "type", target: "query" },
        { kind: "scroll", target: "down" },
      ]),
    ];
    const dataset = buildMovementDataset(trajectories);
    const model = new MarkovMovementBackend(2).train(dataset, { order: 2 });

    // Given the shared prefix, tap:result (count 2) beats scroll:down (count 1).
    const predictions = model.predictNext([MOVEMENT_START_TOKEN, "device:tap:open", "device:type:query"]);
    expect(predictions[0]?.token).toBe("device:tap:result");
    expect(predictions[0]?.count).toBe(2);
  });

  it("backs off to shorter contexts for unseen prefixes", () => {
    const dataset = buildMovementDataset([
      synthTrajectory("a", [
        { kind: "tap", target: "a" },
        { kind: "type", target: "b" },
        { kind: "tap", target: "c" },
      ]),
    ]);
    const model = new MarkovMovementBackend().train(dataset, { order: 2 });

    // This 2-token context never appeared, but the unigram "type:b" -> "tap:c" did.
    const predictions = model.predictNext(["<never>", "device:type:b"]);
    expect(predictions[0]?.token).toBe("device:tap:c");
  });
});

describe("serialization round-trip", () => {
  it("reloads to an identical model", () => {
    const dataset = buildMovementDataset([
      synthTrajectory("a", [{ kind: "tap", target: "x" }, { kind: "type", target: "y" }]),
    ]);
    const model = new MarkovMovementBackend().train(dataset);
    const snapshot = model.serialize();
    const reloaded = loadMovementModel(snapshot);

    expect(reloaded.serialize()).toEqual(snapshot);
    expect(reloaded.generate([MOVEMENT_START_TOKEN])).toEqual(model.generate([MOVEMENT_START_TOKEN]));
  });
});

describe("backend registry", () => {
  it("exposes the default markov backend and instantiates by id", () => {
    expect(listMovementBackends()).toContain("markov");
    expect(createMovementBackend().id).toBe("markov");
    expect(createMovementBackend("markov")).toBeInstanceOf(MarkovMovementBackend);
  });

  it("supports registering and selecting a custom backend (the on-device seam)", () => {
    const stubModel = { backendId: "stub" } as unknown as MovementModel;
    const stub: MovementModelBackend = { id: "stub", train: () => stubModel };
    registerMovementBackend("stub", () => stub);

    expect(listMovementBackends()).toContain("stub");
    expect(createMovementBackend("stub")).toBe(stub);
  });

  it("throws a helpful error for an unknown backend id", () => {
    expect(() => createMovementBackend("does-not-exist")).toThrow(/Unknown movement backend/);
  });
});

describe("generalization eval harness", () => {
  it("scores perfect replay on the training distribution", () => {
    const trajectories = [
      synthTrajectory("a", [{ kind: "tap", target: "x" }, { kind: "type", target: "y" }]),
      synthTrajectory("b", [{ kind: "tap", target: "x" }, { kind: "type", target: "y" }]),
    ];
    const dataset = buildMovementDataset(trajectories);
    const model = new MarkovMovementBackend().train(dataset);

    const result = evaluateMovementModel(model, dataset.samples);
    expect(result.topOneAccuracy).toBe(1);
    expect(result.exactSequenceMatch).toBe(1);
    expect(result.unpredicted).toBe(0);
  });

  it("reports imperfect fidelity on an unrelated held-out trajectory", () => {
    const trainDataset = buildMovementDataset([
      synthTrajectory("a", [{ kind: "tap", target: "x" }, { kind: "type", target: "y" }]),
    ]);
    const model = new MarkovMovementBackend().train(trainDataset);

    const heldOut = buildMovementDataset([
      synthTrajectory("z", [{ kind: "shortcut", target: "quit" }, { kind: "tap", target: "confirm" }]),
    ]);
    const result = evaluateMovementModel(model, heldOut.samples);
    expect(result.exactSequenceMatch).toBeLessThan(1);
  });
});
