import { describe, expect, it } from "vitest";
import type { ReplayManifest } from "../capture/replay.js";
import { buildTrajectorySpan } from "../capture/trajectory.js";
import {
  MarkovMovementBackend,
  MOVEMENT_EOS,
  evaluateMovementModel,
  extractMovementSequences,
  extractMovementSequencesFromReplays,
  tokenizeMovement,
  type MovementSequence,
} from "./movement-model.js";

describe("tokenizeMovement", () => {
  it("produces a stable, normalized token from tool + summary", () => {
    expect(tokenizeMovement({ tool: "device", summary: "tapped Login!" })).toBe("device:tapped-login");
    // Casing / trailing punctuation collapse to the same movement.
    expect(tokenizeMovement({ tool: "device", summary: "Tapped login" })).toBe(
      tokenizeMovement({ tool: "device", summary: "tapped login" }),
    );
  });

  it("folds gesture/direction metadata into the token", () => {
    expect(
      tokenizeMovement({
        tool: "device",
        summary: "swiped up",
        metadata: { gesture: "swipe", direction: "up" },
      }),
    ).toBe("device:swipe:up:swiped-up");
  });
});

describe("sequence extraction", () => {
  it("orders actions by timestamp and drops empty trajectories", () => {
    const withActions = buildTrajectorySpan({
      id: "t1",
      sessionId: "s1",
      actions: [
        { kind: "action", tool: "device", summary: "second", ts: 20 },
        { kind: "action", tool: "device", summary: "first", ts: 10 },
      ],
    });
    const empty = buildTrajectorySpan({ id: "t2", sessionId: "s1" });
    const sequences = extractMovementSequences([withActions, empty]);
    expect(sequences).toHaveLength(1);
    expect(sequences[0]?.tokens).toEqual(["device:first", "device:second"]);
  });

  it("groups replay-manifest actions by trajectory", () => {
    const manifest: ReplayManifest = {
      version: 1,
      sessionId: "s9",
      trajectoryIds: ["a", "b"],
      eventCount: 3,
      events: [
        { kind: "action", ts: 1, trajectoryId: "a", tool: "device", summary: "open app" },
        { kind: "observation", ts: 2, trajectoryId: "a", source: "device", summary: "ignored" },
        { kind: "action", ts: 3, trajectoryId: "b", tool: "device", summary: "tap submit" },
      ],
    };
    const sequences = extractMovementSequencesFromReplays([manifest]);
    expect(sequences.map((sequence) => sequence.trajectoryId)).toEqual(["a", "b"]);
    expect(sequences[0]?.tokens).toEqual(["device:open-app"]);
  });
});

const RECORDED: MovementSequence[] = [
  { trajectoryId: "r1", tokens: ["device:focus-app", "device:tap-search", "device:type-query", "device:tap-result"] },
  { trajectoryId: "r2", tokens: ["device:focus-app", "device:tap-search", "device:type-query", "device:tap-result"] },
  { trajectoryId: "r3", tokens: ["device:focus-app", "device:tap-menu", "device:tap-settings"] },
];

describe("MarkovMovementBackend", () => {
  it("memorizes a recorded movement run and replays it from its start", () => {
    const backend = new MarkovMovementBackend();
    const model = backend.train({ sequences: RECORDED }, { order: 3 });
    const generated = model.generate(["device:focus-app", "device:tap-search"]);
    expect(generated).toEqual(["device:type-query", "device:tap-result"]);
  });

  it("generalizes to a related but unseen prefix via shorter shared context", () => {
    const backend = new MarkovMovementBackend();
    const model = backend.train({ sequences: RECORDED }, { order: 3 });
    // "device:tap-search" only ever precedes "device:type-query" in training,
    // even though this exact full prefix path was never seen as a unit.
    const [top] = model.predictNext(["device:tap-search"]);
    expect(top?.token).toBe("device:type-query");
  });

  it("predictNext returns normalized, deterministically-ordered probabilities", () => {
    const backend = new MarkovMovementBackend();
    const model = backend.train({ sequences: RECORDED }, { order: 2 });
    const predictions = model.predictNext(["device:focus-app"], 5);
    const total = predictions.reduce((sum, prediction) => sum + prediction.probability, 0);
    expect(total).toBeCloseTo(1, 6);
    // Two distinct continuations of focus-app were recorded: tap-search, tap-menu.
    expect(predictions.map((prediction) => prediction.token)).toContain("device:tap-search");
    expect(predictions.map((prediction) => prediction.token)).toContain("device:tap-menu");
    // Identical input yields identical ordering (no randomness).
    expect(model.predictNext(["device:focus-app"], 5)).toEqual(predictions);
  });

  it("serializes and reloads to an identical model", () => {
    const backend = new MarkovMovementBackend();
    const model = backend.train({ sequences: RECORDED }, { order: 3 });
    const serialized = model.serialize();
    const json = JSON.stringify(serialized);
    const reloaded = backend.load(JSON.parse(json));
    expect(reloaded.predictNext(["device:focus-app"], 5)).toEqual(model.predictNext(["device:focus-app"], 5));
    expect(reloaded.generate(["device:focus-app", "device:tap-search"])).toEqual(
      model.generate(["device:focus-app", "device:tap-search"]),
    );
    // serialize() is order-stable -> round-trips byte-identically.
    expect(JSON.stringify(reloaded.serialize())).toBe(json);
  });

  it("knows when to stop (predicts EOS after a recorded run ends)", () => {
    const backend = new MarkovMovementBackend();
    const model = backend.train({ sequences: RECORDED }, { order: 3 });
    const [top] = model.predictNext(["device:focus-app", "device:tap-menu", "device:tap-settings"]);
    expect(top?.token).toBe(MOVEMENT_EOS);
  });
});

describe("evaluateMovementModel", () => {
  it("scores higher next-movement accuracy than a context-free unigram baseline", () => {
    // "open" is ambiguous given only its immediate predecessor (it leads to
    // "report" on the dashboard path and "profile" on the settings path); a
    // unigram cannot disambiguate, but order>=2 context can.
    const train: MovementSequence[] = [
      { trajectoryId: "a", tokens: ["login", "dashboard", "open", "report", "export"] },
      { trajectoryId: "b", tokens: ["login", "settings", "open", "profile", "save"] },
    ];
    // Held-out but related: same vocabulary/structure, unseen as a whole run.
    const heldOut: MovementSequence[] = [
      { trajectoryId: "h", tokens: ["login", "dashboard", "open", "report", "export"] },
    ];

    const backend = new MarkovMovementBackend();
    const contextModel = backend.train({ sequences: train }, { order: 3 });
    const unigramModel = backend.train({ sequences: train }, { order: 1 });

    const contextEval = evaluateMovementModel(contextModel, heldOut, 1);
    const unigramEval = evaluateMovementModel(unigramModel, heldOut, 1);

    expect(contextEval.predictions).toBe(heldOut[0]!.tokens.length + 1); // + EOS
    expect(contextEval.top1Accuracy).toBeGreaterThan(unigramEval.top1Accuracy);
    expect(contextEval.top1Accuracy).toBeGreaterThanOrEqual(0.8);
    expect(contextEval.topKAccuracy).toBeGreaterThanOrEqual(contextEval.top1Accuracy);
  });
});
