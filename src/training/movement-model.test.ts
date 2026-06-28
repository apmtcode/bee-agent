import { describe, expect, it } from "vitest";
import { buildReplayManifest } from "../capture/replay.js";
import { buildTrajectorySpan } from "../capture/trajectory.js";
import type { TrajectoryAction } from "../capture/trajectory.js";
import {
  actionToMovementToken,
  buildMovementDataset,
  createMovementBackend,
  evaluateNextActionAccuracy,
  templatizeSummary,
} from "./movement-model.js";

function action(tool: string, summary: string, ts: number): TrajectoryAction {
  return { kind: "action", tool, summary, ts };
}

/**
 * Synthetic event-stream generator: build a replay manifest for one trajectory
 * from a list of (tool, summary) movements without touching any real OS input.
 */
function syntheticReplay(trajectoryId: string, sessionId: string, moves: Array<[string, string]>) {
  const span = buildTrajectorySpan({
    id: trajectoryId,
    sessionId,
    actions: moves.map(([tool, summary], index) => action(tool, summary, index + 1)),
  });
  return buildReplayManifest({ sessionId, transcript: [], trajectories: [span] });
}

describe("movement tokenization", () => {
  it("templatizes concrete coordinates and indices to '#'", () => {
    expect(templatizeSummary("Click at (12, 40)")).toBe("click at (#, #)");
    expect(templatizeSummary("move  to   880,5")).toBe("move to #,#");
    expect(actionToMovementToken({ tool: "mouse", summary: "Click at (12, 40)" })).toBe("mouse|click at (#, #)");
  });

  it("maps the same movement at different positions to the same token", () => {
    const a = actionToMovementToken({ tool: "mouse", summary: "click at (12, 40)" });
    const b = actionToMovementToken({ tool: "mouse", summary: "click at (880, 5)" });
    expect(a).toBe(b);
  });
});

describe("buildMovementDataset", () => {
  it("produces one timestamp-ordered action sequence per trajectory", () => {
    const replay = syntheticReplay("traj-1", "sess-1", [
      ["mouse", "move to (0, 0)"],
      ["mouse", "click at (10, 10)"],
      ["keyboard", "type hello"],
    ]);
    const dataset = buildMovementDataset([replay]);
    expect(dataset.sequences).toHaveLength(1);
    expect(dataset.sequences[0].tokens).toEqual([
      "mouse|move to (#, #)",
      "mouse|click at (#, #)",
      "keyboard|type hello",
    ]);
    // Vocabulary is sorted + de-duplicated.
    expect(dataset.vocabulary).toEqual([...dataset.vocabulary].sort());
  });

  it("ignores non-action events", () => {
    const span = buildTrajectorySpan({
      id: "traj-1",
      sessionId: "sess-1",
      observations: [{ kind: "observation", source: "screen", summary: "window opened", ts: 1 }],
      actions: [action("mouse", "click at (1, 2)", 2)],
    });
    const replay = buildReplayManifest({ sessionId: "sess-1", transcript: [], trajectories: [span] });
    const dataset = buildMovementDataset([replay]);
    expect(dataset.sequences[0].tokens).toEqual(["mouse|click at (#, #)"]);
  });
});

describe("NgramMovementBackend", () => {
  const drag: Array<[string, string]> = [
    ["mouse", "move to (0, 0)"],
    ["mouse", "press left"],
    ["mouse", "move to (50, 50)"],
    ["mouse", "release left"],
  ];

  it("repeats a recorded movement sequence by greedy generation", () => {
    const dataset = buildMovementDataset([syntheticReplay("traj-1", "sess-1", drag)]);
    const model = createMovementBackend("ngram").train(dataset);
    const seed = [dataset.sequences[0].tokens[0]];
    const generated = model.generate(seed, drag.length - 1);
    expect(generated).toEqual(dataset.sequences[0].tokens);
  });

  it("generalizes to the same movement performed at new coordinates", () => {
    // Train on a drag at one set of positions...
    const dataset = buildMovementDataset([syntheticReplay("traj-1", "sess-1", drag)]);
    const model = createMovementBackend("ngram").train(dataset);

    // ...and evaluate on a *new* drag at entirely different coordinates. Because
    // coordinates are templatized away, the held-out sequence shares the learned
    // token transitions, so next-action prediction is perfect despite never
    // having seen these exact positions.
    const heldOut = buildMovementDataset([
      syntheticReplay("traj-2", "sess-2", [
        ["mouse", "move to (900, 12)"],
        ["mouse", "press left"],
        ["mouse", "move to (123, 456)"],
        ["mouse", "release left"],
      ]),
    ]);
    const evaluation = evaluateNextActionAccuracy(model, heldOut.sequences);
    expect(evaluation.predictions).toBe(3);
    expect(evaluation.accuracy).toBe(1);
  });

  it("backs off to a lower-order context for an unseen prefix", () => {
    const dataset = buildMovementDataset([
      syntheticReplay("a", "s", [
        ["mouse", "press left"],
        ["mouse", "move to (1, 1)"],
        ["mouse", "release left"],
      ]),
    ]);
    const model = createMovementBackend("ngram", { order: 2 }).train(dataset);
    // Context "release left" was never followed by anything in training, but the
    // bigram "press left" -> "move" and unigram priors still let the model
    // predict via backoff rather than returning null.
    const prediction = model.predictNext(["keyboard|type foo", "mouse|press left"]);
    expect(prediction.token).toBe("mouse|move to (#, #)");
    expect(prediction.order).toBeLessThanOrEqual(2);
  });

  it("returns a null prediction from an empty model", () => {
    const model = createMovementBackend("ngram").train({ version: 1, sequences: [], vocabulary: [] });
    expect(model.predictNext(["anything"]).token).toBeNull();
    expect(model.generate(["seed"], 5)).toEqual(["seed"]);
  });

  it("round-trips through a JSON snapshot deterministically", () => {
    const dataset = buildMovementDataset([syntheticReplay("traj-1", "sess-1", drag)]);
    const backend = createMovementBackend("ngram");
    const model = backend.train(dataset);
    const snapshot = JSON.parse(JSON.stringify(model.toJSON()));
    const restored = backend.restore(snapshot);
    const seed = [dataset.sequences[0].tokens[0]];
    expect(restored.generate(seed, drag.length - 1)).toEqual(model.generate(seed, drag.length - 1));
  });

  it("prefers the more frequent transition with a deterministic tie-break", () => {
    // "click" follows "focus" twice and "scroll" once -> argmax is "click".
    const dataset = buildMovementDataset([
      syntheticReplay("a", "s", [["ui", "focus field"], ["ui", "click button"]]),
      syntheticReplay("b", "s", [["ui", "focus field"], ["ui", "click button"]]),
      syntheticReplay("c", "s", [["ui", "focus field"], ["ui", "scroll down"]]),
    ]);
    const model = createMovementBackend("ngram").train(dataset);
    const prediction = model.predictNext(["ui|focus field"]);
    expect(prediction.token).toBe("ui|click button");
    expect(prediction.probability).toBeCloseTo(2 / 3, 5);
    expect(prediction.candidates.map((candidate) => candidate.token)).toEqual([
      "ui|click button",
      "ui|scroll down",
    ]);
  });
});

describe("createMovementBackend", () => {
  it("throws on an unknown backend name", () => {
    // @ts-expect-error — exercising the runtime guard for an invalid name.
    expect(() => createMovementBackend("does-not-exist")).toThrow(/unknown movement backend/);
  });
});
