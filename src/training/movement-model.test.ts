import { describe, expect, it } from "vitest";
import { buildReplayManifest } from "../capture/replay.js";
import { buildTrajectorySpan, type TrajectoryAction } from "../capture/trajectory.js";
import {
  MOVEMENT_END,
  MovementBackendRegistry,
  NgramMovementBackend,
  datasetFromReplayManifests,
  datasetFromTrajectories,
  evaluateGeneralization,
  generateSyntheticMovementDataset,
  loadMovementModel,
  movementTokenFromAction,
  sequenceFromTrajectory,
  type MovementSequence,
} from "./movement-model.js";

function action(tool: string, ts: number, metadata: Record<string, unknown>, summary = "did thing"): TrajectoryAction {
  return { kind: "action", tool, summary, ts, metadata };
}

describe("movement tokenization", () => {
  it("derives a canonical token from structured action metadata", () => {
    expect(
      movementTokenFromAction(action("device", 1, { gesture: "tap", target: "Save Button" })),
    ).toBe("device/tap/save-button");
    expect(
      movementTokenFromAction(action("device", 1, { gesture: "swipe", direction: "up" })),
    ).toBe("device/swipe/up");
  });

  it("falls back to a summary slug when metadata is sparse", () => {
    expect(movementTokenFromAction(action("keyboard", 1, {}, "Pressed Cmd S"))).toBe(
      "keyboard/act/pressed-cmd-s",
    );
  });

  it("orders trajectory actions by timestamp", () => {
    const trajectory = buildTrajectorySpan({
      id: "t1",
      sessionId: "s1",
      actions: [
        action("device", 30, { gesture: "type", target: "field" }),
        action("device", 10, { gesture: "tap", target: "menu" }),
      ],
    });
    const sequence = sequenceFromTrajectory(trajectory);
    expect(sequence.tokens).toEqual(["device/tap/menu", "device/type/field"]);
  });
});

describe("NgramMovementBackend training + inference", () => {
  const dataset = generateSyntheticMovementDataset({
    motifs: [
      { id: "save", tokens: ["app/focus/editor", "key/shortcut/cmd-s", "app/toast/saved"] },
      { id: "close", tokens: ["app/focus/editor", "key/shortcut/cmd-w", "app/window/closed"] },
    ],
    repeats: 3,
  });

  it("reproduces a recorded movement from its seed (repeats memorized movements)", async () => {
    const model = await new NgramMovementBackend().train(dataset, { order: 2 });
    const generated = model.generate(["app/focus/editor", "key/shortcut/cmd-s"], 10);
    expect(generated).toEqual(["app/focus/editor", "key/shortcut/cmd-s", "app/toast/saved"]);
  });

  it("predicts the most-likely next token with backoff transparency", async () => {
    const model = await new NgramMovementBackend().train(dataset, { order: 2 });
    const prediction = model.predictNext(["app/focus/editor", "key/shortcut/cmd-s"]);
    expect(prediction?.token).toBe("app/toast/saved");
    expect(prediction?.order).toBe(2);
    expect(prediction?.probability).toBeGreaterThan(0.9);
  });

  it("is deterministic across retrains", async () => {
    const a = await new NgramMovementBackend().train(dataset, { order: 2 });
    const b = await new NgramMovementBackend().train(dataset, { order: 2 });
    expect(a.snapshot()).toEqual(b.snapshot());
    expect(a.generate([], 8)).toEqual(b.generate([], 8));
  });

  it("terminates generation at the end sentinel", async () => {
    const model = await new NgramMovementBackend().train(dataset, { order: 2 });
    const generated = model.generate([], 20);
    expect(generated).not.toContain(MOVEMENT_END);
    expect(generated.length).toBeLessThan(20);
  });
});

describe("generalization to novel-but-related movements", () => {
  it("scores held-out compound sequences well via backoff (objective 2d)", async () => {
    // Train only on the base motifs; hold out the interleaved compounds the
    // model has never seen as whole sequences.
    const motifs = [
      { id: "open", tokens: ["app/focus/finder", "app/open/project", "app/focus/editor"] },
      { id: "run", tokens: ["app/focus/editor", "key/shortcut/cmd-r", "app/panel/output"] },
    ];
    const train = generateSyntheticMovementDataset({ motifs, repeats: 4 });
    const withCompounds = generateSyntheticMovementDataset({ motifs, repeats: 1, interleave: true });
    const heldOut: MovementSequence[] = withCompounds.sequences.filter((s) => s.id.includes("+"));

    const model = await new NgramMovementBackend().train(train, { order: 2 });
    const eval_ = evaluateGeneralization(model, heldOut);

    expect(heldOut.length).toBeGreaterThan(0);
    // The compound "open then run" shares the pivot token `app/focus/editor`,
    // so a generalizing model predicts the continuation it never saw as a whole.
    expect(eval_.nextTokenAccuracy).toBeGreaterThan(0.6);
    expect(eval_.meanLogProb).toBeGreaterThan(-2);
  });

  it("a memorize-only baseline (empty training) does not generalize", async () => {
    const model = await new NgramMovementBackend().train({ sequences: [] }, { order: 2 });
    const heldOut: MovementSequence[] = [{ id: "x", tokens: ["a/act/one", "a/act/two"] }];
    const eval_ = evaluateGeneralization(model, heldOut);
    expect(eval_.nextTokenAccuracy).toBe(0);
  });
});

describe("snapshot round-trip", () => {
  it("reloads a trained model that predicts identically", async () => {
    const dataset = generateSyntheticMovementDataset({
      motifs: [{ id: "m", tokens: ["a/act/one", "a/act/two", "a/act/three"] }],
      repeats: 2,
    });
    const model = await new NgramMovementBackend().train(dataset, { order: 2 });
    const reloaded = loadMovementModel(model.snapshot());
    expect(reloaded.predictNext(["a/act/one"])?.token).toBe(
      model.predictNext(["a/act/one"])?.token,
    );
    expect(reloaded.generate([], 8)).toEqual(model.generate([], 8));
  });
});

describe("dataset adapters", () => {
  it("builds a dataset from trajectories", () => {
    const dataset = datasetFromTrajectories([
      buildTrajectorySpan({
        id: "t1",
        sessionId: "s1",
        actions: [action("device", 1, { gesture: "tap", target: "menu" })],
      }),
      buildTrajectorySpan({ id: "t2", sessionId: "s1", actions: [] }),
    ]);
    expect(dataset.sequences).toHaveLength(1);
    expect(dataset.sequences[0].tokens).toEqual(["device/tap/menu"]);
  });

  it("builds a dataset from replay manifests", () => {
    const trajectory = buildTrajectorySpan({
      id: "t1",
      sessionId: "s1",
      actions: [
        action("device", 10, { gesture: "tap", target: "menu" }, "tapped menu"),
        action("device", 20, { gesture: "type", target: "field" }, "typed field"),
      ],
    });
    const manifest = buildReplayManifest({ sessionId: "s1", transcript: [], trajectories: [trajectory] });
    const dataset = datasetFromReplayManifests([manifest]);
    expect(dataset.sequences).toHaveLength(1);
    expect(dataset.sequences[0].tokens).toEqual(["device/act/tapped-menu", "device/act/typed-field"]);
  });
});

describe("MovementBackendRegistry", () => {
  it("registers the ngram backend by default and is extensible", () => {
    const registry = new MovementBackendRegistry();
    expect(registry.list()).toContain("ngram");
    expect(registry.get("ngram")).toBeInstanceOf(NgramMovementBackend);
    registry.register({ name: "custom", train: async () => registry.get("ngram")!.train({ sequences: [] }) });
    expect(registry.list()).toContain("custom");
  });
});
