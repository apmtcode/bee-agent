import { describe, expect, it } from "vitest";
import {
  MarkovMovementBackend,
  MOVEMENT_END,
  buildMovementDataset,
  buildMovementSequence,
  defaultMovementTokenizer,
  evaluateMovementPolicy,
  generateSyntheticMovementDataset,
  movementSequenceFromTrajectory,
  type MovementSequence,
} from "./movement-policy.js";
import { buildTrajectorySpan } from "../capture/trajectory.js";

const backend = new MarkovMovementBackend();

describe("movement tokenization", () => {
  it("maps identical recorded actions to one token", () => {
    const a = defaultMovementTokenizer({ tool: "device", summary: "tapped Send" });
    const b = defaultMovementTokenizer({ tool: "device", summary: "tapped Send" });
    expect(a).toBe(b);
    expect(a).toBe("device·tapped Send");
  });

  it("derives a goal-labeled sequence from a trajectory span", () => {
    const span = buildTrajectorySpan({
      id: "t1",
      sessionId: "s1",
      observations: [{ kind: "observation", source: "device", summary: "Mail active", ts: 1 }],
      actions: [
        { kind: "action", tool: "device", summary: "tapped Compose", ts: 2 },
        { kind: "action", tool: "device", summary: "tapped Send", ts: 3 },
      ],
    });
    const sequence = movementSequenceFromTrajectory(span);
    expect(sequence.goal).toBe("device");
    expect(sequence.steps).toEqual(["device·tapped Compose", "device·tapped Send"]);
  });
});

describe("MarkovMovementBackend — repeat", () => {
  it("reproduces a recorded movement exactly via generate()", async () => {
    const recorded = ["device·tapped Compose", "device·typed subject", "device·tapped Send"];
    const dataset = buildMovementDataset([{ id: "s", goal: "compose", steps: recorded }]);
    const policy = await backend.train(dataset);

    const produced = policy.generate({ goal: "compose", history: [recorded[0]!] });
    expect([recorded[0], ...produced]).toEqual(recorded);
  });

  it("predicts the next step deterministically with backoff order metadata", async () => {
    const dataset = buildMovementDataset([
      { id: "a", goal: "g", steps: ["x", "y", "z"] },
      { id: "b", goal: "g", steps: ["x", "y", "z"] },
      { id: "c", goal: "g", steps: ["x", "y", "q"] },
    ]);
    const policy = await backend.train(dataset, { order: 2 });

    const prediction = policy.predictNext({ goal: "g", history: ["x", "y"] });
    expect(prediction.candidates[0]?.step).toBe("z");
    expect(prediction.candidates[0]?.support).toBe(2);
    expect(prediction.candidates[0]?.order).toBe(2);
    expect(prediction.candidates[0]?.goalMatched).toBe(true);
    // probability normalized within the chosen distribution (z:2, q:1)
    expect(prediction.candidates[0]?.probability).toBeCloseTo(2 / 3);
  });

  it("terminates generation at the END sentinel", async () => {
    const dataset = buildMovementDataset([{ id: "s", goal: "g", steps: ["only"] }]);
    const policy = await backend.train(dataset);
    const produced = policy.generate({ goal: "g", history: [] });
    expect(produced).toEqual(["only"]);
    expect(produced).not.toContain(MOVEMENT_END);
  });
});

describe("MarkovMovementBackend — generalize", () => {
  it("backs off to a shorter context for an unseen prefix", async () => {
    // 'a b c' seen; query prefix 'zz b' never seen at order 2 → back off to 'b' → c.
    const dataset = buildMovementDataset([
      { id: "1", goal: "g", steps: ["a", "b", "c"] },
      { id: "2", goal: "g", steps: ["a", "b", "c"] },
    ]);
    const policy = await backend.train(dataset, { order: 3 });

    const prediction = policy.predictNext({ goal: "g", history: ["zz", "b"] });
    expect(prediction.candidates[0]?.step).toBe("c");
    expect(prediction.candidates[0]?.order).toBeLessThan(2);
  });

  it("drops the goal label to transfer statistics across related goals", async () => {
    const dataset = buildMovementDataset([{ id: "1", goal: "known-goal", steps: ["open", "confirm"] }]);
    const policy = await backend.train(dataset);

    // A brand-new goal with no sequences of its own still gets a prediction via
    // the goal-agnostic backoff.
    const prediction = policy.predictNext({ goal: "unseen-goal", history: ["open"] });
    expect(prediction.candidates[0]?.step).toBe("confirm");
    expect(prediction.candidates[0]?.goalMatched).toBe(false);
  });

  it("returns no candidates for an empty policy", async () => {
    const policy = await backend.train(buildMovementDataset([]));
    expect(policy.predictNext({ history: [] }).candidates).toEqual([]);
    expect(policy.generate({ history: [] })).toEqual([]);
  });
});

describe("synthetic stream + evaluation harness", () => {
  it("is reproducible for a given seed", () => {
    const a = generateSyntheticMovementDataset({ seed: 42 });
    const b = generateSyntheticMovementDataset({ seed: 42 });
    expect(a).toEqual(b);
    const c = generateSyntheticMovementDataset({ seed: 7 });
    expect(c).not.toEqual(a);
  });

  it("trains on some goals and generalizes to held-out related sequences", () => {
    const goals = ["compose-email", "file-search", "checkout"];
    const train = generateSyntheticMovementDataset({ seed: 1, goals, sequencesPerGoal: 6, noise: 0.1 });
    const heldOut = generateSyntheticMovementDataset({ seed: 999, goals, sequencesPerGoal: 3, noise: 0.1 });

    return backend.train(train, { order: 3 }).then((policy) => {
      const result = evaluateMovementPolicy(policy, heldOut.sequences, { k: 3 });
      expect(result.total).toBeGreaterThan(0);
      // The skeleton is shared, so the policy should predict most next steps.
      expect(result.top1Accuracy).toBeGreaterThan(0.6);
      expect(result.topKAccuracy).toBeGreaterThanOrEqual(result.top1Accuracy);
    });
  });
});

describe("policy snapshot", () => {
  it("serializes vocabulary and contexts", async () => {
    const dataset = buildMovementDataset([{ id: "s", goal: "g", steps: ["a", "b"] }]);
    const policy = await backend.train(dataset, { order: 2 });
    const snapshot = policy.toJSON();
    expect(snapshot.backendId).toBe("markov-mock");
    expect(snapshot.order).toBe(2);
    expect(snapshot.vocabulary).toEqual(["a", "b"]);
    expect(snapshot.contexts.length).toBeGreaterThan(0);
  });

  it("buildMovementSequence honors a custom tokenizer and goal", () => {
    const sequence: MovementSequence = buildMovementSequence({
      id: "x",
      goal: "custom",
      actions: [{ tool: "device", summary: "tapped A" }],
      tokenizer: (action) => action.summary.toUpperCase(),
    });
    expect(sequence.goal).toBe("custom");
    expect(sequence.steps).toEqual(["TAPPED A"]);
  });
});
