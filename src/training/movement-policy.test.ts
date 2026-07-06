import { describe, expect, it } from "vitest";
import { buildTrajectorySpan, type TrajectoryAction } from "../capture/trajectory.js";
import {
  MOVEMENT_END_TOKEN,
  MarkovMovementBackend,
  buildMovementDataset,
  createMovementPolicyBackend,
  deserializeMovementPolicy,
  registerMovementPolicyBackend,
  type MovementPolicyBackend,
} from "./movement-policy.js";

function actions(tools: string[]): TrajectoryAction[] {
  return tools.map((tool, index) => ({
    kind: "action",
    tool,
    summary: `${tool} #${index}`,
    ts: index + 1,
  }));
}

function trajectory(id: string, tools: string[]) {
  return buildTrajectorySpan({ id, sessionId: `sess-${id}`, actions: actions(tools) });
}

describe("buildMovementDataset", () => {
  it("tokenizes actions in timestamp order and builds ordered samples", () => {
    const traj = buildTrajectorySpan({
      id: "t1",
      sessionId: "s1",
      actions: [
        { kind: "action", tool: "click", summary: "b", ts: 30 },
        { kind: "action", tool: "focus", summary: "a", ts: 10 },
        { kind: "action", tool: "type", summary: "c", ts: 20 },
      ],
    });
    const dataset = buildMovementDataset([traj], { order: 2 });

    expect(dataset.sequences).toEqual([["focus", "type", "click"]]);
    expect(dataset.vocabulary).toEqual(["click", "focus", "type"]);
    // Padded = <start> focus type click <end>; every position after 0 is a sample.
    expect(dataset.samples).toContainEqual({ context: ["<start>"], next: "focus" });
    expect(dataset.samples).toContainEqual({ context: ["type", "click"], next: MOVEMENT_END_TOKEN });
    // Context is capped at `order`.
    expect(dataset.samples.every((sample) => sample.context.length <= 2)).toBe(true);
  });

  it("skips trajectories with no actions", () => {
    const dataset = buildMovementDataset([trajectory("empty", [])]);
    expect(dataset.sequences).toHaveLength(0);
    expect(dataset.samples).toHaveLength(0);
  });
});

describe("MarkovMovementBackend", () => {
  it("repeats a deterministically recorded movement sequence", async () => {
    const dataset = buildMovementDataset([
      trajectory("a", ["open", "search", "select", "confirm"]),
      trajectory("b", ["open", "search", "select", "confirm"]),
    ]);
    const policy = await new MarkovMovementBackend(3).train(dataset);

    // Seeded with the first movement, it reproduces the recorded rollout exactly.
    expect(policy.generate(["open"])).toEqual(["open", "search", "select", "confirm"]);
    const prediction = policy.predictNext(["<start>", "open", "search"]);
    expect(prediction.token).toBe("select");
    expect(prediction.confidence).toBeGreaterThan(0);
  });

  it("generalizes to a new-but-related context via backoff", async () => {
    // Trained only on sequences that always end "review -> submit".
    const dataset = buildMovementDataset([
      trajectory("a", ["login", "compose", "review", "submit"]),
      trajectory("b", ["draft", "compose", "review", "submit"]),
      trajectory("c", ["import", "compose", "review", "submit"]),
    ]);
    const policy = await new MarkovMovementBackend(2).train(dataset);

    // A context whose full order-2 prefix was never recorded ("edit" is unseen),
    // but "review" strongly predicts "submit" — backoff recovers it.
    const prediction = policy.predictNext(["edit", "review"]);
    expect(prediction.token).toBe("submit");
    expect(prediction.backoffOrder).toBeLessThan(2);
  });

  it("is deterministic in the face of ties (lexicographic tie-break)", async () => {
    const dataset = buildMovementDataset([trajectory("a", ["start", "zzz"]), trajectory("b", ["start", "aaa"])]);
    const policy = await new MarkovMovementBackend(1).train(dataset);
    // "start" -> {zzz:1, aaa:1}; equal probability, so lexicographically smallest wins.
    expect(policy.predictNext(["start"]).token).toBe("aaa");
  });

  it("round-trips through serialize/deserialize with identical inference", async () => {
    const dataset = buildMovementDataset([trajectory("a", ["open", "search", "select"])]);
    const policy = await new MarkovMovementBackend(3).train(dataset);
    const model = policy.serialize();
    const restored = deserializeMovementPolicy(model);

    expect(model.backendId).toBe("markov");
    expect(restored.generate(["open"])).toEqual(policy.generate(["open"]));
    expect(restored.predictNext(["open"]).token).toBe(policy.predictNext(["open"]).token);
  });

  it("terminates generation at <end> even under a large step budget", async () => {
    const dataset = buildMovementDataset([trajectory("a", ["one", "two"])]);
    const policy = await new MarkovMovementBackend(2).train(dataset);
    expect(policy.generate([], 1000)).toEqual(["one", "two"]);
  });
});

describe("createMovementPolicyBackend", () => {
  it("returns the markov backend for the default and mock ids", () => {
    expect(createMovementPolicyBackend().id).toBe("markov");
    expect(createMovementPolicyBackend("mock").id).toBe("markov");
  });

  it("throws for an unknown backend id", () => {
    expect(() => createMovementPolicyBackend("nope")).toThrow(/Unknown movement policy backend/);
  });

  it("supports registering a custom backend", () => {
    const custom: MovementPolicyBackend = {
      id: "custom",
      train: async () => {
        throw new Error("not used");
      },
    };
    registerMovementPolicyBackend("custom-test", () => custom);
    expect(createMovementPolicyBackend("custom-test")).toBe(custom);
  });
});
