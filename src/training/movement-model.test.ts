import { describe, expect, it } from "vitest";
import { buildTrajectorySpan, type TrajectoryAction } from "../capture/trajectory.js";
import {
  MarkovMovementBackend,
  MovementModel,
  evaluateGeneralization,
  tokenizeAction,
  tokenizeTrajectory,
  type MovementSequence,
} from "./movement-model.js";
import { generateSyntheticSequences } from "./synthetic-movements.js";

function action(partial: Partial<TrajectoryAction> & { tool: string }): TrajectoryAction {
  return { kind: "action", summary: "", ts: 0, ...partial };
}

describe("tokenizeAction", () => {
  it("prefers structured gesture metadata over the summary", () => {
    expect(
      tokenizeAction(action({ tool: "device", summary: "swiped left", metadata: { gesture: "swipe", direction: "left" } })),
    ).toBe("device:swipe:left");
  });

  it("collapses phrasing differences to the same token", () => {
    const a = tokenizeAction(action({ tool: "device", summary: "tapped the OK button", metadata: { gesture: "tap", target: "ok" } }));
    const b = tokenizeAction(action({ tool: "device", summary: "clicked OK", metadata: { gesture: "tap", target: "OK" } }));
    expect(a).toBe(b);
  });

  it("falls back to a summary slug for untagged actions", () => {
    expect(tokenizeAction(action({ tool: "shell", summary: "run npm test" }))).toBe("shell:run-npm-test");
  });

  it("orders trajectory actions by timestamp", () => {
    const span = buildTrajectorySpan({
      id: "t1",
      sessionId: "s",
      actions: [
        action({ tool: "device", ts: 30, metadata: { gesture: "tap" } }),
        action({ tool: "device", ts: 10, metadata: { gesture: "scroll", direction: "down" } }),
      ],
    });
    expect(tokenizeTrajectory(span)).toEqual(["device:scroll:down", "device:tap"]);
  });
});

describe("MarkovMovementBackend", () => {
  const sequences: MovementSequence[] = [
    ["a", "b", "c"],
    ["a", "b", "c"],
    ["a", "b", "d"],
  ];

  it("predicts the most frequent continuation deterministically", () => {
    const model = new MovementModel({ order: 2 });
    model.trainFromSequences(sequences);
    expect(model.predictNext(["a", "b"])).toBe("c");
    const ranked = model.rankNext(["a", "b"]);
    expect(ranked[0]).toEqual({ token: "c", probability: 2 / 3 });
    expect(ranked[1]).toEqual({ token: "d", probability: 1 / 3 });
  });

  it("backs off to shorter contexts for unseen prefixes", () => {
    const model = new MovementModel({ order: 2 });
    model.trainFromSequences(sequences);
    // "z b" was never seen at order 2, but "b" -> "c"/"d" was seen at order 1.
    expect(model.predictNext(["z", "b"])).toBe("c");
  });

  it("generates a deterministic greedy continuation", () => {
    const model = new MovementModel({ order: 2 });
    model.trainFromSequences(sequences);
    expect(model.generate(["a"], 2)).toEqual(["b", "c"]);
  });

  it("round-trips through serialize/deserialize losslessly", () => {
    const model = new MovementModel({ order: 2 });
    model.trainFromSequences(sequences);
    const state = JSON.parse(JSON.stringify(model.serialize()));

    const reloaded = new MovementModel({ order: 2 });
    reloaded.load(state);
    expect(reloaded.rankNext(["a", "b"])).toEqual(model.rankNext(["a", "b"]));
  });

  it("rejects a foreign policy state on deserialize", () => {
    expect(() => new MarkovMovementBackend().deserialize({ backendId: "other" })).toThrow(/invalid markov/);
  });

  it("throws when predicting before training", () => {
    expect(() => new MovementModel().predictNext(["a"])).toThrow(/not been trained/);
  });
});

describe("evaluateGeneralization", () => {
  it("scores perfect recall on the training distribution", () => {
    const model = new MovementModel({ order: 2 });
    const seqs: MovementSequence[] = [
      ["x", "y", "z"],
      ["x", "y", "z"],
    ];
    const policy = model.trainFromSequences(seqs);
    const report = evaluateGeneralization(policy, seqs);
    expect(report.nextTokenAccuracy).toBe(1);
    expect(report.coverage).toBe(1);
    expect(report.fullSequenceMatches).toBe(2);
  });

  it("generalizes to held-out sequences built from learned sub-phrases", () => {
    // Train on synthetic motifs, hold out a disjoint synthetic seed, and assert
    // the model reproduces most held-out movements it never saw verbatim.
    const train = generateSyntheticSequences({ seed: 7, count: 40 });
    const heldOut = generateSyntheticSequences({ seed: 999, count: 12 });

    const model = new MovementModel({ order: 3 });
    const policy = model.trainFromSequences(train);
    const report = evaluateGeneralization(policy, heldOut);

    expect(report.sequencesEvaluated).toBeGreaterThan(0);
    expect(report.evaluated).toBeGreaterThan(0);
    // Motifs share structure, so a back-off model should generalize strongly.
    expect(report.nextTokenAccuracy).toBeGreaterThan(0.6);
    expect(report.coverage).toBeGreaterThanOrEqual(report.nextTokenAccuracy);
  });

  it("ignores sequences too short to score", () => {
    const model = new MovementModel();
    const policy = model.trainFromSequences([["a", "b"]]);
    const report = evaluateGeneralization(policy, [["solo"]]);
    expect(report.evaluated).toBe(0);
    expect(report.nextTokenAccuracy).toBe(0);
  });
});
