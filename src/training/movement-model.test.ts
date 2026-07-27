import { describe, expect, it } from "vitest";
import {
  createMovementBackend,
  evaluateDatasetAccuracy,
  evaluateNextTokenAccuracy,
  generateMovementSequence,
  listMovementBackends,
  movementDatasetFromReplays,
  MOVEMENT_END_TOKEN,
  NgramMovementBackend,
  registerMovementBackend,
  tokenizeMovementEvent,
  tokenizeTrajectory,
  type MovementDataset,
  type MovementTrajectory,
} from "./movement-model.js";

function deployTrajectory(id: string): MovementTrajectory {
  return {
    id,
    events: [
      { kind: "observation", ts: 1, trajectoryId: id, source: "browser", summary: "opened dashboard" },
      { kind: "action", ts: 2, trajectoryId: id, tool: "mouse", summary: "clicked deploy button" },
      { kind: "observation", ts: 3, trajectoryId: id, source: "browser", summary: "confirm dialog shown" },
      { kind: "action", ts: 4, trajectoryId: id, tool: "keyboard", summary: "typed confirm" },
    ],
  };
}

describe("tokenizeMovementEvent", () => {
  it("maps actions and observations to distinct, normalized tokens", () => {
    expect(
      tokenizeMovementEvent({ kind: "action", ts: 1, trajectoryId: "t", tool: "mouse", summary: "  Clicked   Deploy " }),
    ).toBe("action:mouse:clicked deploy");
    expect(
      tokenizeMovementEvent({ kind: "observation", ts: 1, trajectoryId: "t", source: "browser", summary: "Opened Tab" }),
    ).toBe("obs:browser:opened tab");
    expect(
      tokenizeMovementEvent({ kind: "transcript", ts: 1, messageId: "m", role: "user", content: "hi there" }),
    ).toBe("msg:user");
  });

  it("appends the end marker when tokenizing a trajectory", () => {
    const tokens = tokenizeTrajectory(deployTrajectory("t"));
    expect(tokens.at(-1)).toBe(MOVEMENT_END_TOKEN);
    expect(tokens).toHaveLength(5);
  });
});

describe("NgramMovementBackend", () => {
  it("is deterministic: same dataset yields identical artifacts", async () => {
    const dataset: MovementDataset = { trajectories: [deployTrajectory("a"), deployTrajectory("b")] };
    const backend = new NgramMovementBackend();
    const first = await backend.train(dataset);
    const second = await backend.train(dataset);
    expect(second).toEqual(first);
    expect(first.backend).toBe("ngram-mock");
    expect(first.trajectoryCount).toBe(2);
    expect(first.vocabulary).toContain("action:mouse:clicked deploy button");
    expect(first.vocabulary).not.toContain(MOVEMENT_END_TOKEN);
  });

  it("reproduces a recorded movement chain from its first observation", async () => {
    const trajectory = deployTrajectory("a");
    const backend = new NgramMovementBackend();
    const model = await backend.train({ trajectories: [trajectory] });

    const seed = [tokenizeMovementEvent(trajectory.events[0]!)];
    const generated = generateMovementSequence(backend, model, seed);
    // The rest of the recorded token stream (minus the seed, minus the end marker).
    expect([seed[0], ...generated]).toEqual(tokenizeTrajectory(trajectory).slice(0, -1));
  });

  it("scores perfect next-token accuracy on trained data", async () => {
    const trajectory = deployTrajectory("a");
    const backend = new NgramMovementBackend();
    const model = await backend.train({ trajectories: [trajectory] });
    const result = evaluateNextTokenAccuracy(backend, model, trajectory);
    expect(result.accuracy).toBe(1);
    // 5 tokens (4 events + end marker); the opening token is the seed, so 4 scored.
    expect(result.steps).toBe(4);
    expect(result.correct).toBe(4);
  });

  it("generalizes via backoff to an unseen leading context", async () => {
    const backend = new NgramMovementBackend();
    const model = await backend.train({ trajectories: [deployTrajectory("a")] });

    // A context the model never saw as a full n-gram, but whose suffix it has.
    const novelSeed = ["obs:browser:some brand new banner", "obs:browser:confirm dialog shown"];
    const result = backend.infer(model, novelSeed);
    expect(result.prediction).toBe("action:keyboard:typed confirm");
    expect(result.backoffOrder).toBeLessThan(novelSeed.length);
  });

  it("returns an empty prediction for an empty model", async () => {
    const backend = new NgramMovementBackend();
    const model = await backend.train({ trajectories: [] });
    const result = backend.infer(model, ["anything"]);
    expect(result.prediction).toBeUndefined();
    expect(result.distribution).toEqual([]);
  });

  it("prefers the more frequent continuation deterministically", async () => {
    const shared = "obs:app:menu open";
    const build = (id: string, action: string): MovementTrajectory => ({
      id,
      events: [
        { kind: "observation", ts: 1, trajectoryId: id, source: "app", summary: "menu open" },
        { kind: "action", ts: 2, trajectoryId: id, tool: "mouse", summary: action },
      ],
    });
    const backend = new NgramMovementBackend();
    const model = await backend.train({
      trajectories: [build("a", "clicked settings"), build("b", "clicked settings"), build("c", "clicked help")],
    });
    const result = backend.infer(model, [shared]);
    expect(result.prediction).toBe("action:mouse:clicked settings");
    expect(result.probability).toBeCloseTo(2 / 3);
    expect(result.distribution.map((entry) => entry.token)).toEqual([
      "action:mouse:clicked settings",
      "action:mouse:clicked help",
    ]);
  });
});

describe("evaluateDatasetAccuracy", () => {
  it("aggregates per-trajectory accuracy into a mean", async () => {
    const backend = new NgramMovementBackend();
    const dataset: MovementDataset = { trajectories: [deployTrajectory("a"), deployTrajectory("b")] };
    const model = await backend.train(dataset);
    const evaluation = evaluateDatasetAccuracy(backend, model, dataset);
    expect(evaluation.meanAccuracy).toBe(1);
    expect(evaluation.perTrajectory).toHaveLength(2);
  });
});

describe("movementDatasetFromReplays", () => {
  it("derives trajectory ids from replay trajectoryIds", () => {
    const dataset = movementDatasetFromReplays([
      { trajectoryIds: ["traj-1", "traj-2"], events: deployTrajectory("x").events },
      { events: deployTrajectory("y").events },
    ]);
    expect(dataset.trajectories[0]!.id).toBe("traj-1+traj-2");
    expect(dataset.trajectories[1]!.id).toBe("replay-1");
  });
});

describe("backend registry", () => {
  it("exposes the mock backend by default and instantiates it", () => {
    expect(listMovementBackends()).toContain("ngram-mock");
    expect(createMovementBackend()).toBeInstanceOf(NgramMovementBackend);
    expect(createMovementBackend("ngram-mock").name).toBe("ngram-mock");
  });

  it("supports registering and resolving a custom backend", () => {
    registerMovementBackend("test-backend", () => new NgramMovementBackend());
    expect(listMovementBackends()).toContain("test-backend");
    expect(createMovementBackend("test-backend")).toBeInstanceOf(NgramMovementBackend);
  });

  it("throws a helpful error for an unknown backend", () => {
    expect(() => createMovementBackend("does-not-exist")).toThrow(/unknown movement backend/);
  });
});
