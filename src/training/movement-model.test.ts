import { describe, expect, it } from "vitest";
import {
  MarkovMovementBackend,
  MOVEMENT_END,
  evaluateMovementModel,
  tokenizeAction,
  tokenizeReplayManifest,
  tokenizeTrajectorySpan,
  type MovementSequence,
} from "./movement-model.js";
import {
  generateSyntheticMovementStream,
  splitMovementCorpus,
} from "./synthetic-movements.js";
import { buildTrajectorySpan } from "../capture/trajectory.js";

function seq(id: string, tokens: string[]): MovementSequence {
  return { id, tokens };
}

describe("MarkovMovementBackend", () => {
  it("reproduces a recorded movement sequence exactly (objective c)", async () => {
    const recorded = ["focus/window", "keyboard/type", "keyboard/submit", "pointer/click/result", MOVEMENT_END];
    const model = await new MarkovMovementBackend().train([seq("a", recorded)]);
    const generated = model.generate(recorded.slice(0, 1));
    expect(generated).toEqual(recorded.slice(1, recorded.length - 1));
  });

  it("predicts the highest-count next movement deterministically", async () => {
    const model = await new MarkovMovementBackend().train([
      seq("a", ["x", "y"]),
      seq("b", ["x", "y"]),
      seq("c", ["x", "z"]),
    ]);
    const prediction = model.predictNext(["x"]);
    expect(prediction.token).toBe("y");
    expect(prediction.order).toBe(1);
    expect(prediction.probability).toBeCloseTo(2 / 3, 5);
    expect(prediction.candidates.map((c) => c.token)).toEqual(["y", "z"]);
  });

  it("breaks count ties lexicographically for determinism", async () => {
    const model = await new MarkovMovementBackend().train([
      seq("a", ["x", "beta"]),
      seq("b", ["x", "alpha"]),
    ]);
    expect(model.predictNext(["x"]).token).toBe("alpha");
  });

  it("generalizes to a novel prefix by backing off to a shorter suffix (objective d)", async () => {
    // Train only on context "a b -> c". A novel long context "q a b" was never
    // seen at full order, but shares the "a b" suffix, so backoff still predicts c.
    const model = await new MarkovMovementBackend().train([seq("a", ["a", "b", "c"])], { maxOrder: 3 });
    const prediction = model.predictNext(["q", "a", "b"]);
    expect(prediction.token).toBe("c");
    expect(prediction.order).toBe(2); // backed off from 3 to the "a b" bigram
  });

  it("returns an empty prediction when nothing was learned", async () => {
    const model = await new MarkovMovementBackend().train([]);
    const prediction = model.predictNext(["anything"]);
    expect(prediction.token).toBeUndefined();
    expect(prediction.order).toBe(-1);
    expect(prediction.candidates).toEqual([]);
  });

  it("round-trips through serialize/deserialize", async () => {
    const dataset = [seq("a", ["x", "y", "z"]), seq("b", ["x", "y", "w"])];
    const model = await new MarkovMovementBackend().train(dataset, { maxOrder: 2 });
    const restored = MarkovMovementBackend.deserialize(model.serialize());
    expect(restored.maxOrder).toBe(2);
    expect(restored.predictNext(["x"])).toEqual(model.predictNext(["x"]));
    expect(restored.predictNext(["x", "y"])).toEqual(model.predictNext(["x", "y"]));
  });

  it("terminates generation at maxSteps even without an end marker", async () => {
    const model = await new MarkovMovementBackend().train([seq("loop", ["a", "a", "a", "a"])]);
    const generated = model.generate(["a"], 5);
    expect(generated.length).toBe(5);
    expect(generated.every((token) => token === "a")).toBe(true);
  });
});

describe("tokenizers", () => {
  it("derives a generalizable token from an action's gesture metadata", () => {
    const token = tokenizeAction({
      tool: "device",
      summary: "swiped up",
      metadata: { gesture: "swipe", direction: "up", target: "feed" },
    });
    expect(token).toBe("device/swipe/up");
  });

  it("includes targets only when asked", () => {
    const action = { tool: "device", summary: "tapped", metadata: { gesture: "tap", target: "submit-btn" } };
    expect(tokenizeAction(action)).toBe("device/tap");
    expect(tokenizeAction(action, { includeTargets: true })).toBe("device/tap/submit-btn");
  });

  it("falls back to the summary verb when there is no gesture", () => {
    expect(tokenizeAction({ tool: "shell", summary: "Ran build command" })).toBe("shell/ran");
  });

  it("tokenizes a trajectory span in timestamp order and appends the end marker", () => {
    const span = buildTrajectorySpan({
      id: "span-1",
      sessionId: "s1",
      actions: [
        { kind: "action", tool: "device", summary: "second", ts: 20, metadata: { gesture: "type" } },
        { kind: "action", tool: "device", summary: "first", ts: 10, metadata: { gesture: "tap" } },
      ],
    });
    const sequence = tokenizeTrajectorySpan(span);
    expect(sequence.tokens).toEqual(["device/tap", "device/type", MOVEMENT_END]);
  });

  it("tokenizes a replay manifest's action timeline", () => {
    const sequence = tokenizeReplayManifest({
      version: 1,
      sessionId: "s2",
      trajectoryIds: ["t1"],
      eventCount: 3,
      events: [
        { kind: "observation", ts: 1, trajectoryId: "t1", source: "os", summary: "focus" },
        { kind: "action", ts: 2, trajectoryId: "t1", tool: "keyboard", summary: "Type hello" },
        { kind: "action", ts: 3, trajectoryId: "t1", tool: "keyboard", summary: "Submit form" },
      ],
    });
    expect(sequence.tokens).toEqual(["keyboard/type", "keyboard/submit", MOVEMENT_END]);
  });
});

describe("synthetic movement stream", () => {
  it("is deterministic for a given seed", () => {
    const a = generateSyntheticMovementStream({ seed: 7, sequenceCount: 10 });
    const b = generateSyntheticMovementStream({ seed: 7, sequenceCount: 10 });
    expect(a).toEqual(b);
  });

  it("varies with the seed", () => {
    const a = generateSyntheticMovementStream({ seed: 1, sequenceCount: 10 });
    const b = generateSyntheticMovementStream({ seed: 2, sequenceCount: 10 });
    expect(a).not.toEqual(b);
  });

  it("respects the max length bound", () => {
    const sequences = generateSyntheticMovementStream({ seed: 3, sequenceCount: 20, maxLength: 5 });
    // maxLength bounds the walk; the appended end marker adds at most one token.
    expect(sequences.every((sequence) => sequence.tokens.length <= 6)).toBe(true);
  });

  it("terminates every walked sequence with the end marker", () => {
    const sequences = generateSyntheticMovementStream({ seed: 4, sequenceCount: 8 });
    expect(sequences.every((sequence) => sequence.tokens[sequence.tokens.length - 1] === MOVEMENT_END)).toBe(true);
  });
});

describe("generalization eval harness", () => {
  it("scores high next-movement accuracy on held-out related sequences", async () => {
    const corpus = generateSyntheticMovementStream({ seed: 11, sequenceCount: 120, maxLength: 16 });
    const { train, heldOut } = splitMovementCorpus(corpus, 0.25);
    expect(train.length).toBeGreaterThan(0);
    expect(heldOut.length).toBeGreaterThan(0);

    const model = await new MarkovMovementBackend().train(train, { maxOrder: 3 });
    const report = evaluateMovementModel(model, heldOut);

    expect(report.sequenceCount).toBe(heldOut.length);
    expect(report.predictedPositions).toBeGreaterThan(0);
    // The grammar is low-entropy, so a backoff model should predict most next
    // movements on unseen-but-related runs — this is the generalization signal.
    expect(report.nextTokenAccuracy).toBeGreaterThan(0.75);
    expect(report.nextTokenAccuracy).toBeLessThanOrEqual(1);
    expect(report.exactContextRate).toBeGreaterThanOrEqual(0);
    expect(report.exactContextRate).toBeLessThanOrEqual(1);
  });

  it("reports perfect reproduction on a deterministic corpus (objective c)", async () => {
    // Non-branching (deterministic) sequences: each context has exactly one
    // successor, so a trained model must reproduce them perfectly.
    const corpus = [
      seq("p", ["focus/window", "keyboard/type", "keyboard/submit", "pointer/click/result", MOVEMENT_END]),
      seq("q", ["focus/editor", "keyboard/open-file", "keyboard/type", "keyboard/save", MOVEMENT_END]),
    ];
    const model = await new MarkovMovementBackend().train(corpus, { maxOrder: 4 });
    const report = evaluateMovementModel(model, corpus);
    expect(report.nextTokenAccuracy).toBe(1);
    expect(report.exactSequenceRate).toBe(1);
  });

  it("keeps held-out accuracy no better than in-grammar training accuracy", async () => {
    const corpus = generateSyntheticMovementStream({ seed: 5, sequenceCount: 160, maxLength: 14 });
    const { train, heldOut } = splitMovementCorpus(corpus, 0.25);
    const model = await new MarkovMovementBackend().train(train, { maxOrder: 3 });
    const trainReport = evaluateMovementModel(model, train);
    const heldReport = evaluateMovementModel(model, heldOut);
    expect(trainReport.nextTokenAccuracy).toBeGreaterThan(0.75);
    expect(heldReport.nextTokenAccuracy).toBeLessThanOrEqual(trainReport.nextTokenAccuracy + 1e-9);
  });

  it("handles an empty held-out set without dividing by zero", async () => {
    const model = await new MarkovMovementBackend().train([seq("a", ["x", "y"])]);
    const report = evaluateMovementModel(model, []);
    expect(report).toEqual({
      sequenceCount: 0,
      predictedPositions: 0,
      nextTokenAccuracy: 0,
      exactContextRate: 0,
      exactSequenceRate: 0,
    });
  });
});
