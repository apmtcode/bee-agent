import { describe, expect, it } from "vitest";
import {
  NgramMovementModelBackend,
  MOVEMENT_END,
  evaluateMovementModel,
  generateSyntheticMovementSequences,
  sequencesFromReplayEvents,
  sequencesFromTrajectories,
  tokenizeAction,
  type MovementSequence,
} from "./movement-model.js";
import { buildTrajectorySpan } from "../capture/trajectory.js";

function seq(id: string, tokens: string[]): MovementSequence {
  return { id, tokens };
}

describe("NgramMovementModelBackend", () => {
  it("repeats a recorded movement sequence exactly (objective 2c)", () => {
    const backend = new NgramMovementModelBackend();
    const recorded = seq("t1", ["device:tap:menu", "device:tap:settings", "device:swipe:down", "device:tap:save"]);
    const model = backend.train([recorded], { order: 3 });

    // Deterministically replay from the start; must reproduce the recording.
    expect(model.generate()).toEqual(recorded.tokens);
  });

  it("predicts the recorded continuation for a seen context", () => {
    const backend = new NgramMovementModelBackend();
    const model = backend.train([seq("t1", ["a", "b", "c", "d"])], { order: 3 });

    const prediction = model.predictNext(["b", "c"]);
    expect(prediction?.token).toBe("d");
    expect(prediction?.contextOrder).toBe(2);
    expect(prediction?.probability).toBeCloseTo(1);
  });

  it("generalizes to an unseen-but-related context via backoff (objective 2d)", () => {
    const backend = new NgramMovementModelBackend();
    // "open then save" is a strong pattern; the exact trigram is never seen.
    const model = backend.train(
      [
        seq("t1", ["open", "edit", "save"]),
        seq("t2", ["open", "review", "save"]),
        seq("t3", ["open", "annotate", "save"]),
      ],
      { order: 3 },
    );

    // ["reopen","edit"] was never observed as a trigram, but "edit"->"save" is a
    // strong learned bigram, so the model backs off one level and still predicts
    // the related "save" for the novel context.
    const prediction = model.predictNext(["reopen", "edit"]);
    expect(prediction).toBeDefined();
    expect(prediction?.token).toBe("save");
    expect(prediction?.contextOrder).toBe(1); // backed off from trigram to bigram
  });

  it("terminates generation at the end sentinel", () => {
    const backend = new NgramMovementModelBackend();
    const model = backend.train([seq("t1", ["x", "y"])], { order: 2 });
    const generated = model.generate([], 100);
    expect(generated).toEqual(["x", "y"]);
    expect(generated).not.toContain(MOVEMENT_END);
  });

  it("is deterministic and serializable across identical training runs", () => {
    const backend = new NgramMovementModelBackend();
    const data = [seq("t1", ["a", "b", "a", "c"]), seq("t2", ["a", "b", "d"])];
    const first = backend.train(data, { order: 3 }).serialize();
    const second = backend.train(data, { order: 3 }).serialize();
    expect(first).toEqual(second);
    expect(first.vocabulary).toEqual([...first.vocabulary].sort());
  });

  it("breaks ties deterministically by frequency then token order", () => {
    const backend = new NgramMovementModelBackend();
    // After "a": "b" appears twice, "c" once -> "b" wins on frequency.
    const model = backend.train([seq("t1", ["a", "b"]), seq("t2", ["a", "b"]), seq("t3", ["a", "c"])], { order: 2 });
    const prediction = model.predictNext(["a"]);
    expect(prediction?.token).toBe("b");
    expect(prediction?.candidates.map((candidate) => candidate.token)).toEqual(["b", "c"]);
  });
});

describe("tokenization", () => {
  it("encodes gesture/direction/target metadata into a movement token", () => {
    const trajectory = buildTrajectorySpan({
      id: "t1",
      sessionId: "s1",
      actions: [
        {
          kind: "action",
          tool: "device",
          summary: "swiped down",
          ts: 2,
          metadata: { gesture: "swipe", direction: "down", target: "feed" },
        },
        {
          kind: "action",
          tool: "device",
          summary: "tapped Submit",
          ts: 1,
          metadata: { gesture: "tap", target: "Submit" },
        },
      ],
    });

    const [sequence] = sequencesFromTrajectories([trajectory]);
    // Sorted by ts: tap(1) before swipe(2).
    expect(sequence?.tokens).toEqual(["device:tap:Submit", "device:swipe:down:feed"]);
  });

  it("falls back to the summary when no gesture metadata is present", () => {
    const token = tokenizeAction({ kind: "action", tool: "shell", summary: "ran build", ts: 0 });
    expect(token).toBe("shell:ran build");
  });

  it("builds sequences from replay-timeline action events grouped by trajectory", () => {
    const sequences = sequencesFromReplayEvents([
      { kind: "observation", ts: 0, trajectoryId: "t1", source: "device", summary: "focus" },
      { kind: "action", ts: 3, trajectoryId: "t1", tool: "device", summary: "tap b" },
      { kind: "action", ts: 1, trajectoryId: "t1", tool: "device", summary: "tap a" },
      { kind: "action", ts: 2, trajectoryId: "t2", tool: "device", summary: "swipe" },
    ]);
    const t1 = sequences.find((sequence) => sequence.id === "t1");
    expect(t1?.tokens).toEqual(["device:tap a", "device:tap b"]);
    expect(sequences.find((sequence) => sequence.id === "t2")?.tokens).toEqual(["device:swipe"]);
  });
});

describe("evaluateMovementModel + synthetic generator", () => {
  it("achieves perfect fidelity replaying a single recorded trajectory", () => {
    const backend = new NgramMovementModelBackend();
    // A single recorded path has an unambiguous prefix tree (including the very
    // first move), so the deterministic model reproduces it with 100% fidelity.
    const train = [
      seq("t1", ["launch:mail", "compose", "type:subject", "type:body", "review", "send"]),
    ];
    const model = backend.train(train, { order: 3 });
    const result = evaluateMovementModel(model, train);
    expect(result.accuracy).toBe(1);
    expect(result.predictions).toBeGreaterThan(0);
    expect(model.generate()).toEqual(train[0]!.tokens);
  });

  it("fits training data at least as well as held-out perturbed sequences (2c/2d)", () => {
    const backend = new NgramMovementModelBackend();
    const templates = [
      ["launch:mail", "compose", "type:subject", "type:body", "review", "send"],
      ["launch:browser", "navigate", "click:link", "scroll:down", "read", "bookmark"],
    ];
    const train = generateSyntheticMovementSequences({ templates, count: 40, seed: 11 });
    // Held-out set uses a different seed + perturbation -> new-but-related.
    const heldOut = generateSyntheticMovementSequences({
      templates,
      count: 20,
      seed: 99,
      perturbationRate: 0.5,
    });
    const model = backend.train(train, { order: 3 });

    const trainEval = evaluateMovementModel(model, train);
    const heldOutEval = evaluateMovementModel(model, heldOut);

    // Perturbation only removes information, so held-out never beats training.
    expect(trainEval.accuracy).toBeGreaterThanOrEqual(heldOutEval.accuracy);
    // The only source of training error is first-move ambiguity across templates.
    expect(trainEval.accuracy).toBeGreaterThan(0.8);
    // Chance for this vocabulary is well under 0.5; require real generalization.
    expect(heldOutEval.accuracy).toBeGreaterThan(0.5);
    // Some held-out predictions must have relied on backoff (order < 2).
    const usedBackoff = Object.entries(heldOutEval.backoffHistogram)
      .filter(([order]) => Number(order) < 2)
      .reduce((sum, [, count]) => sum + count, 0);
    expect(usedBackoff).toBeGreaterThan(0);
  });

  it("generates deterministic synthetic streams for a fixed seed", () => {
    const templates = [["a", "b", "c"], ["a", "d"]];
    const first = generateSyntheticMovementSequences({ templates, count: 10, seed: 42, perturbationRate: 0.4 });
    const second = generateSyntheticMovementSequences({ templates, count: 10, seed: 42, perturbationRate: 0.4 });
    expect(first).toEqual(second);
    expect(first).toHaveLength(10);
  });
});
