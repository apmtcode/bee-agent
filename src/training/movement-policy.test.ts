import { describe, expect, it } from "vitest";
import { buildTrajectorySpan, type TrajectorySpan } from "../capture/trajectory.js";
import {
  buildMovementDataset,
  createMovementPolicyBackend,
  defaultMovementTokenizer,
  evaluateMovementPolicy,
  listMovementPolicyBackends,
  MarkovMovementBackend,
  MOVEMENT_BOS,
  MOVEMENT_EOS,
} from "./movement-policy.js";
import { generateSyntheticTrajectories } from "./synthetic-trajectories.js";

function span(id: string, actions: Array<[string, string, number]>): TrajectorySpan {
  return buildTrajectorySpan({
    id,
    sessionId: `s-${id}`,
    actions: actions.map(([tool, summary, ts]) => ({ kind: "action", tool, summary, ts })),
  });
}

describe("defaultMovementTokenizer", () => {
  it("abstracts coordinates so related movements share a token", () => {
    const a = defaultMovementTokenizer({ kind: "action", tool: "mouse.click", summary: "click at (120, 340)", ts: 0 });
    const b = defaultMovementTokenizer({ kind: "action", tool: "mouse.click", summary: "click at (88, 12)", ts: 0 });
    expect(a).toBe(b);
    expect(a).toBe("mouse.click|click at (#, #)");
  });
});

describe("buildMovementDataset", () => {
  it("frames sequences with BOS/EOS and emits sliding examples", () => {
    const dataset = buildMovementDataset([span("t1", [["a", "one", 1], ["b", "two", 2]])], { order: 2 });
    expect(dataset.sequences[0]!.tokens[0]).toBe(MOVEMENT_BOS);
    expect(dataset.sequences[0]!.tokens.at(-1)).toBe(MOVEMENT_EOS);
    // BOS->a, a->b, b->EOS
    expect(dataset.examples.map((example) => example.next)).toEqual(["a|one", "b|two", MOVEMENT_EOS]);
  });

  it("sorts actions by timestamp before framing", () => {
    const dataset = buildMovementDataset([span("t1", [["b", "two", 2], ["a", "one", 1]])], { includeEos: false });
    expect(dataset.sequences[0]!.tokens).toEqual([MOVEMENT_BOS, "a|one", "b|two"]);
  });

  it("supports an optional goal token after BOS", () => {
    const dataset = buildMovementDataset([span("t1", [["a", "one", 1]])], {
      goalToken: () => "goal:demo",
    });
    expect(dataset.sequences[0]!.tokens.slice(0, 2)).toEqual([MOVEMENT_BOS, "goal:demo"]);
  });
});

describe("MarkovMovementBackend", () => {
  it("reproduces a recorded movement sequence (objective: repeat)", () => {
    const recorded = span("t1", [["a", "one", 1], ["b", "two", 2], ["c", "three", 3]]);
    const dataset = buildMovementDataset([recorded], { order: 2 });
    const policy = new MarkovMovementBackend().train(dataset);
    expect(policy.generate()).toEqual(["a|one", "b|two", "c|three"]);
  });

  it("generalizes to a related-but-unseen continuation via backoff", () => {
    // Train one sequence; the transition b -> c is learned at order 1.
    const dataset = buildMovementDataset([span("t1", [["a", "s", 1], ["b", "s", 2], ["c", "s", 3]])], {
      order: 1,
    });
    const policy = new MarkovMovementBackend().train(dataset);
    // This full 3-token context was never seen, but backoff to the learned
    // order-1 context ["b|s"] still predicts c, so a novel prefix continues.
    const prediction = policy.predict([MOVEMENT_BOS, "x|s", "y|s", "b|s"]);
    expect(prediction?.token).toBe("c|s");
    expect(prediction?.matchedOrder).toBe(1);
  });

  it("uses a deterministic tie-break when counts are equal", () => {
    const dataset = buildMovementDataset(
      [span("t1", [["b", "beta", 1]]), span("t2", [["a", "alpha", 1]])],
      { order: 1, includeEos: false },
    );
    const policy = new MarkovMovementBackend().train(dataset);
    // From BOS both a|alpha and b|beta have count 1 -> lexicographically smaller wins.
    expect(policy.predict([MOVEMENT_BOS])?.token).toBe("a|alpha");
  });

  it("round-trips through serialize/load", () => {
    const dataset = buildMovementDataset([span("t1", [["a", "one", 1], ["b", "two", 2]])], { order: 2 });
    const policy = new MarkovMovementBackend().train(dataset);
    const restored = MarkovMovementBackend.load(policy.serialize());
    expect(restored.generate()).toEqual(policy.generate());
    expect(restored.order).toBe(policy.order);
  });
});

describe("createMovementPolicyBackend", () => {
  it("resolves the markov backend and lists registered backends", () => {
    expect(createMovementPolicyBackend("markov")).toBeInstanceOf(MarkovMovementBackend);
    expect(listMovementPolicyBackends()).toContain("markov");
  });

  it("throws for an unknown backend id", () => {
    expect(() => createMovementPolicyBackend("ggml-9000")).toThrow(/Unknown movement policy backend/);
  });
});

describe("evaluateMovementPolicy", () => {
  it("scores perfectly on the sequences it trained on", () => {
    const trajectories = generateSyntheticTrajectories({ family: "desktop-file-edit", count: 6, seed: 1 });
    const dataset = buildMovementDataset(trajectories, { order: 3 });
    const policy = new MarkovMovementBackend().train(dataset);
    const result = evaluateMovementPolicy(policy, dataset.sequences);
    expect(result.trajectoryCount).toBe(6);
    expect(result.nextTokenAccuracy).toBe(1);
    expect(result.rolloutFidelity).toBe(1);
  });

  it("generalizes to held-out related trajectories above chance", () => {
    // Train on low-novelty variants, evaluate on unseen novel-but-related ones.
    const train = generateSyntheticTrajectories({
      family: "web-form-submit",
      count: 30,
      seed: 100,
      noveltyRate: 0.3,
    });
    const heldOut = generateSyntheticTrajectories({
      family: "web-form-submit",
      count: 10,
      seed: 999,
      noveltyRate: 0.5,
    });
    const trainIds = new Set(train.map((trajectory) => trajectory.id));
    expect(heldOut.every((trajectory) => !trainIds.has(trajectory.id))).toBe(true);

    const dataset = buildMovementDataset(train, { order: 3 });
    const policy = new MarkovMovementBackend().train(dataset);
    const heldOutSequences = buildMovementDataset(heldOut, { order: 3 }).sequences;
    const result = evaluateMovementPolicy(policy, heldOutSequences);
    // The tool grammar is shared across the family, so a trained policy should
    // predict most held-out movements correctly despite never seeing them.
    expect(result.nextTokenAccuracy).toBeGreaterThan(0.7);
    expect(result.rolloutFidelity).toBeGreaterThan(0.5);
  });

  it("skips sequences shorter than the seed", () => {
    const result = evaluateMovementPolicy(
      new MarkovMovementBackend().train(buildMovementDataset([span("t1", [["a", "one", 1]])])),
      [{ trajectoryId: "empty", tokens: [MOVEMENT_BOS] }],
      { seedLength: 2 },
    );
    expect(result.trajectoryCount).toBe(0);
    expect(result.nextTokenAccuracy).toBe(0);
  });
});
