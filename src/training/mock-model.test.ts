import { describe, expect, it } from "vitest";
import {
  inferMovementSequence,
  scoreSequenceLikelihood,
  tokenizeReplayEvent,
  trainMockMovementModel,
  type MovementSequence,
} from "./mock-model.js";

const recorded: MovementSequence[] = [
  { sessionId: "s1", tokens: ["action:open", "action:type", "action:click", "action:save"] },
  { sessionId: "s2", tokens: ["action:open", "action:type", "action:click", "action:save"] },
  { sessionId: "s3", tokens: ["action:open", "action:type", "action:submit"] },
];

describe("tokenizeReplayEvent", () => {
  it("namespaces tokens by kind and tool/source/role", () => {
    expect(tokenizeReplayEvent({ kind: "action", tool: "browser" })).toBe("action:browser");
    expect(tokenizeReplayEvent({ kind: "observation", source: "os" })).toBe("observation:os");
    expect(tokenizeReplayEvent({ kind: "transcript", role: "assistant" })).toBe("transcript:assistant");
    expect(tokenizeReplayEvent({ kind: "action" })).toBe("action:unknown");
  });
});

describe("trainMockMovementModel", () => {
  it("counts transitions and starts deterministically", () => {
    const model = trainMockMovementModel(recorded);
    expect(model.kind).toBe("markov-order-1");
    expect(model.sequenceCount).toBe(3);
    expect(model.tokens).toEqual([
      "action:click",
      "action:open",
      "action:save",
      "action:submit",
      "action:type",
    ]);
    expect(model.starts).toEqual({ "action:open": 3 });
    expect(model.transitions["action:open"]).toEqual({ "action:type": 3 });
    expect(model.transitions["action:type"]).toEqual({ "action:click": 2, "action:submit": 1 });
  });

  it("is a pure function of its input (same input -> identical model)", () => {
    expect(trainMockMovementModel(recorded)).toEqual(trainMockMovementModel(recorded));
  });

  it("ignores empty sequences and empty tokens", () => {
    const model = trainMockMovementModel([
      { tokens: [] },
      { tokens: ["", "action:x"] },
    ]);
    expect(model.sequenceCount).toBe(1);
    expect(model.starts).toEqual({ "action:x": 1 });
  });
});

describe("inferMovementSequence", () => {
  it("repeats the most-likely recorded movement", () => {
    const model = trainMockMovementModel(recorded);
    // action:type -> click (count 2) beats submit (count 1); click -> save.
    expect(inferMovementSequence(model, { start: "action:open" })).toEqual([
      "action:open",
      "action:type",
      "action:click",
      "action:save",
    ]);
  });

  it("defaults to the most common start token", () => {
    const model = trainMockMovementModel(recorded);
    expect(inferMovementSequence(model)[0]).toBe("action:open");
  });

  it("terminates on a repeated edge instead of looping forever", () => {
    const model = trainMockMovementModel([{ tokens: ["a", "b", "a", "b", "a"] }]);
    const out = inferMovementSequence(model, { start: "a", maxSteps: 100 });
    expect(out).toEqual(["a", "b", "a"]);
  });

  it("generalizes an unseen start by backing off to its namespace", () => {
    const model = trainMockMovementModel(recorded);
    // "action:resume" was never recorded, but generalize should still continue
    // with a plausible in-namespace movement rather than dead-ending.
    const generalized = inferMovementSequence(model, { start: "action:resume", generalize: true });
    expect(generalized.length).toBeGreaterThan(1);
    expect(generalized[0]).toBe("action:resume");
    expect(generalized[1]!.startsWith("action:")).toBe(true);

    const halted = inferMovementSequence(model, { start: "action:resume", generalize: false });
    expect(halted).toEqual(["action:resume"]);
  });

  it("respects maxSteps and empty models", () => {
    const model = trainMockMovementModel(recorded);
    expect(inferMovementSequence(model, { start: "action:open", maxSteps: 2 })).toEqual([
      "action:open",
      "action:type",
    ]);
    expect(inferMovementSequence(trainMockMovementModel([]))).toEqual([]);
  });
});

describe("scoreSequenceLikelihood", () => {
  it("scores recorded movements higher than unrelated ones", () => {
    const model = trainMockMovementModel(recorded);
    const onDistribution = scoreSequenceLikelihood(model, ["action:open", "action:type", "action:click"]);
    const offDistribution = scoreSequenceLikelihood(model, ["action:save", "action:open", "action:save"]);
    expect(onDistribution).toBeGreaterThan(offDistribution);
  });

  it("returns 0 for degenerate short sequences", () => {
    const model = trainMockMovementModel(recorded);
    expect(scoreSequenceLikelihood(model, ["action:open"])).toBe(0);
  });
});
