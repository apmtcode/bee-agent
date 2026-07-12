import { describe, expect, it } from "vitest";
import { buildTrajectorySpan, type TrajectoryAction } from "../capture/trajectory.js";
import {
  MarkovMovementBackend,
  createDefaultMovementBackend,
  deserializeMovementModel,
  evaluateMovementModel,
  tokenizeAction,
  tokenizeActions,
  tokenizeTrajectory,
  type MovementSequence,
} from "./movement-model.js";

function gesture(kind: string, extra: Record<string, unknown>, ts: number): TrajectoryAction {
  return {
    kind: "action",
    tool: "device",
    summary: `${kind} action`,
    ts,
    metadata: { gesture: kind, ...extra },
  };
}

/**
 * Synthetic movement-stream generator: deterministically produces sequences
 * following a simple grammar so capture→dataset→train→infer can be validated
 * without any real OS input. Every "open" flow ends by saving.
 */
function syntheticSequences(count: number): MovementSequence[] {
  // Distinct first tokens so a single-token seed uniquely identifies each flow.
  const templates = [
    ["app:open-menu", "device:tap:file", "device:tap:open", "device:type:filename", "device:tap:confirm"],
    ["app:save-menu", "device:tap:file", "device:tap:save", "device:shortcut:cmd-s"],
    ["nav:swipe", "device:swipe:left", "device:tap:back"],
  ];
  const sequences: MovementSequence[] = [];
  for (let index = 0; index < count; index += 1) {
    const template = templates[index % templates.length]!;
    sequences.push({ id: `syn-${index}`, tokens: [...template] });
  }
  return sequences;
}

describe("movement tokenization", () => {
  it("derives structured tokens from gesture metadata", () => {
    expect(tokenizeAction(gesture("tap", { target: "Save Button" }, 1))).toBe("device:tap:save-button");
    expect(tokenizeAction(gesture("swipe", { direction: "left" }, 1))).toBe("device:swipe:left");
    expect(tokenizeAction(gesture("shortcut", { target: "cmd+s" }, 1))).toBe("device:shortcut:cmd-s");
  });

  it("falls back to a summary slug when no gesture metadata is present", () => {
    const action: TrajectoryAction = { kind: "action", tool: "editor", summary: "Format document", ts: 5 };
    expect(tokenizeAction(action)).toBe("editor:format");
  });

  it("orders tokens by timestamp", () => {
    const actions: TrajectoryAction[] = [
      gesture("tap", { target: "b" }, 30),
      gesture("tap", { target: "a" }, 10),
      gesture("tap", { target: "c" }, 20),
    ];
    expect(tokenizeActions(actions)).toEqual(["device:tap:a", "device:tap:c", "device:tap:b"]);
  });

  it("tokenizes a trajectory span, honoring redacted actions", () => {
    const span = buildTrajectorySpan({
      id: "t1",
      sessionId: "s1",
      actions: [gesture("tap", { target: "menu" }, 1), gesture("type", { target: "field" }, 2)],
    });
    expect(tokenizeTrajectory(span).tokens).toEqual(["device:tap:menu", "device:type:field"]);
  });
});

describe("MarkovMovementBackend — replay of recorded movements", () => {
  it("reproduces a recorded sequence exactly via greedy generation", () => {
    const sequence: MovementSequence = {
      id: "rec",
      tokens: ["device:tap:menu", "device:tap:file", "device:tap:open", "device:tap:confirm"],
    };
    const model = new MarkovMovementBackend().train([sequence], { order: 2 });
    const replayed = model.generate([sequence.tokens[0]!]);
    expect(replayed).toEqual(sequence.tokens);
  });

  it("terminates naturally at the learned end boundary", () => {
    const model = new MarkovMovementBackend().train(
      [{ id: "a", tokens: ["x", "y", "z"] }],
      { order: 2 },
    );
    // maxLength is generous; generation must stop at the learned end, not run on.
    expect(model.generate(["x"], { maxLength: 50 })).toEqual(["x", "y", "z"]);
  });

  it("achieves perfect replay fidelity on its own training set", () => {
    const sequences = syntheticSequences(9);
    const model = createDefaultMovementBackend().train(sequences, { order: 3 });
    const evaluation = evaluateMovementModel(model, sequences);
    expect(evaluation.replayFidelity).toBe(1);
    expect(evaluation.nextTokenAccuracy).toBe(1);
  });
});

describe("MarkovMovementBackend — generalization to related movements", () => {
  it("predicts a plausible continuation for an unseen-but-related context via backoff", () => {
    // Train on flows that always follow "file" with "save", and separately show
    // an "open" flow. A novel prefix ending in "file" should still predict a
    // learned successor of "file" even though this exact full context is unseen.
    const model = new MarkovMovementBackend().train(
      [
        { id: "a", tokens: ["menu", "file", "save", "done"] },
        { id: "b", tokens: ["toolbar", "file", "save", "done"] },
      ],
      { order: 3 },
    );
    // Unseen 3-gram context ["home","panel","file"] backs off to "file" -> "save".
    const prediction = model.predictNext(["home", "panel", "file"])[0];
    expect(prediction?.token).toBe("save");
    expect(prediction?.backoffOrder).toBe(1);
  });

  it("generalizes across held-out related trajectories above chance", () => {
    const all = syntheticSequences(12);
    // Split so every flow template appears in both train and held-out sets.
    const train = all.slice(0, 9);
    const heldOut = all.slice(9);
    const model = createDefaultMovementBackend().train(train, { order: 3 });
    const evaluation = evaluateMovementModel(model, heldOut);
    // Held-out sequences are fresh instances of learned flows, so the model
    // should predict nearly every next token correctly (chance is far lower).
    expect(evaluation.nextTokenAccuracy).toBeGreaterThan(0.9);
    expect(evaluation.sequenceCount).toBe(heldOut.length);
  });

  it("returns no prediction when there is no training data at all", () => {
    const model = new MarkovMovementBackend().train([], { order: 2 });
    expect(model.predictNext(["anything"])).toEqual([]);
    expect(model.generate(["anything"])).toEqual(["anything"]);
    expect(model.vocabulary).toEqual([]);
  });
});

describe("model serialization", () => {
  it("round-trips through JSON with identical predictions", () => {
    const sequences = syntheticSequences(6);
    const model = new MarkovMovementBackend().train(sequences, { order: 3 });
    const restored = deserializeMovementModel(model.toJSON());

    expect(restored.order).toBe(model.order);
    expect(restored.vocabulary).toEqual(model.vocabulary);
    const context = ["device:tap:menu", "device:tap:file"];
    expect(restored.predictNext(context)).toEqual(model.predictNext(context));
    expect(restored.generate(["device:tap:menu"])).toEqual(model.generate(["device:tap:menu"]));
  });

  it("does not leak the internal end sentinel into the public vocabulary", () => {
    const model = new MarkovMovementBackend().train([{ id: "a", tokens: ["one", "two"] }]);
    expect(model.vocabulary).toEqual(["one", "two"]);
  });
});

describe("evaluation harness", () => {
  it("scores probabilities that sum to one for a given context", () => {
    const model = new MarkovMovementBackend().train(
      [
        { id: "a", tokens: ["start", "left"] },
        { id: "b", tokens: ["start", "right"] },
        { id: "c", tokens: ["start", "left"] },
      ],
      { order: 1 },
    );
    const predictions = model.predictNext(["start"]);
    const total = predictions.reduce((sum, prediction) => sum + prediction.probability, 0);
    expect(total).toBeCloseTo(1, 10);
    // "left" seen twice, "right" once → left ranks first with probability 2/3.
    expect(predictions[0]).toMatchObject({ token: "left", probability: 2 / 3 });
  });

  it("reports zero fidelity for degenerate single-token sequences", () => {
    const model = new MarkovMovementBackend().train([{ id: "a", tokens: ["x", "y"] }]);
    const evaluation = evaluateMovementModel(model, [{ id: "short", tokens: ["x"] }]);
    expect(evaluation.perSequence[0]).toEqual({ id: "short", nextTokenAccuracy: 0, replayFidelity: 0 });
  });
});
