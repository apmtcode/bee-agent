import { describe, expect, it } from "vitest";
import type { ExportedReplayManifest } from "./export-manifest.js";
import {
  DeterministicMovementBackend,
  MovementModelRegistry,
  datasetFromReplays,
  evaluateMovementModel,
  type MovementSequence,
  type MovementTrainingDataset,
} from "./movement-model.js";

function seq(trajectoryId: string, tokens: MovementSequence["tokens"]): MovementSequence {
  return { trajectoryId, tokens };
}

const OPEN_SEQ: MovementSequence = seq("t1", [
  { kind: "observation", source: "window", summary: "editor focused" },
  { kind: "action", tool: "keyboard", summary: "press cmd+s" },
  { kind: "observation", source: "window", summary: "save dialog" },
  { kind: "action", tool: "mouse", summary: "click save" },
]);

describe("DeterministicMovementBackend", () => {
  it("repeats a recorded movement exactly (exact-match replay)", async () => {
    const model = await new DeterministicMovementBackend().train({ sequences: [OPEN_SEQ] });

    const prediction = model.predictNext([{ kind: "observation", source: "window", summary: "editor focused" }]);
    expect(prediction).toBeDefined();
    expect(prediction?.action).toEqual({ tool: "keyboard", summary: "press cmd+s" });
    expect(prediction?.generalized).toBe(false);
    expect(prediction?.via).toBe("exact");
    expect(prediction?.confidence).toBe(1);
    expect(prediction?.support).toBe(1);
  });

  it("generalizes to a new-but-related observation via coarse back-off", async () => {
    const model = await new DeterministicMovementBackend().train({ sequences: [OPEN_SEQ] });

    // Same source ("window") but a summary never seen in training.
    const prediction = model.predictNext([{ kind: "observation", source: "window", summary: "brand new window state" }]);
    expect(prediction?.generalized).toBe(true);
    expect(prediction?.via).toBe("coarse");
    // Coarse table for source "window" saw two actions once each; determinism
    // breaks the tie by action key ("keyboard press cmd+s" < "mouse click save").
    expect(prediction?.action).toEqual({ tool: "keyboard", summary: "press cmd+s" });
  });

  it("falls back to the global action prior for an unknown source", async () => {
    const model = await new DeterministicMovementBackend().train({ sequences: [OPEN_SEQ] });
    const prediction = model.predictNext([{ kind: "observation", source: "totally-unknown", summary: "x" }]);
    expect(prediction?.via).toBe("global");
    expect(prediction?.generalized).toBe(true);
  });

  it("returns undefined when trained on an empty dataset", async () => {
    const model = await new DeterministicMovementBackend().train({ sequences: [] });
    expect(model.predictNext([{ kind: "observation", source: "a", summary: "b" }])).toBeUndefined();
    expect(model.metrics()).toEqual({
      sequenceCount: 0,
      transitionCount: 0,
      distinctStates: 0,
      distinctActions: 0,
    });
  });

  it("weights the most frequent transition and reports confidence", async () => {
    const dataset: MovementTrainingDataset = {
      sequences: [
        seq("a", [
          { kind: "observation", source: "menu", summary: "open" },
          { kind: "action", tool: "mouse", summary: "click file" },
        ]),
        seq("b", [
          { kind: "observation", source: "menu", summary: "open" },
          { kind: "action", tool: "mouse", summary: "click file" },
        ]),
        seq("c", [
          { kind: "observation", source: "menu", summary: "open" },
          { kind: "action", tool: "keyboard", summary: "type f" },
        ]),
      ],
    };
    const model = await new DeterministicMovementBackend().train(dataset);
    const prediction = model.predictNext([{ kind: "observation", source: "menu", summary: "open" }]);
    expect(prediction?.action).toEqual({ tool: "mouse", summary: "click file" });
    expect(prediction?.support).toBe(2);
    expect(prediction?.confidence).toBeCloseTo(2 / 3);
  });

  it("records metrics reflecting the trained transitions", async () => {
    const model = await new DeterministicMovementBackend().train({ sequences: [OPEN_SEQ] });
    const metrics = model.metrics();
    expect(metrics.sequenceCount).toBe(1);
    expect(metrics.transitionCount).toBe(2);
    expect(metrics.distinctActions).toBe(2);
  });
});

describe("serialize / restore", () => {
  it("round-trips a trained model through a snapshot", async () => {
    const backend = new DeterministicMovementBackend();
    const model = await backend.train({ sequences: [OPEN_SEQ] });
    const snapshot = model.serialize();

    const restored = backend.restore(snapshot);
    const context = [{ kind: "observation", source: "window", summary: "save dialog" } as const];
    expect(restored.predictNext(context)).toEqual(model.predictNext(context));
    expect(restored.metrics()).toEqual(model.metrics());
  });

  it("serialize returns an independent copy (no shared mutable state)", async () => {
    const model = await new DeterministicMovementBackend().train({ sequences: [OPEN_SEQ] });
    const first = model.serialize();
    first.global["injected"] = { tool: "x", summary: "y", count: 99 };
    const second = model.serialize();
    expect(second.global["injected"]).toBeUndefined();
  });
});

describe("MovementModelRegistry", () => {
  it("selects the deterministic backend by default and lists it", () => {
    const registry = new MovementModelRegistry();
    expect(registry.list()).toContain("deterministic-markov");
    expect(registry.has("deterministic-markov")).toBe(true);
    expect(registry.get("deterministic-markov").name).toBe("deterministic-markov");
  });

  it("throws for an unknown backend name", () => {
    const registry = new MovementModelRegistry();
    expect(() => registry.get("mlx-real")).toThrow(/Unknown movement-model backend/);
  });

  it("restores a snapshot via the backend named in it", async () => {
    const registry = new MovementModelRegistry();
    const model = await registry.get("deterministic-markov").train({ sequences: [OPEN_SEQ] });
    const restored = registry.restore(model.serialize());
    expect(restored.backend).toBe("deterministic-markov");
  });

  it("accepts a pluggable custom backend", () => {
    const registry = new MovementModelRegistry([]);
    const custom = new DeterministicMovementBackend();
    registry.register(custom);
    expect(registry.has("deterministic-markov")).toBe(true);
  });
});

describe("datasetFromReplays", () => {
  it("groups observation/action events per trajectory, ordered by ts, dropping transcript", () => {
    const replays: ExportedReplayManifest[] = [
      {
        sessionId: "s1",
        trajectoryIds: ["t1"],
        eventCount: 4,
        events: [
          { kind: "transcript", ts: 1, messageId: "m1", role: "user", content: "do it" },
          { kind: "action", ts: 30, trajectoryId: "t1", tool: "mouse", summary: "click save" },
          { kind: "observation", ts: 10, trajectoryId: "t1", source: "window", summary: "editor focused" },
          { kind: "action", ts: 20, trajectoryId: "t1", tool: "keyboard", summary: "press cmd+s" },
        ],
      },
    ];
    const dataset = datasetFromReplays(replays);
    expect(dataset.sequences).toHaveLength(1);
    expect(dataset.sequences[0].tokens).toEqual([
      { kind: "observation", source: "window", summary: "editor focused" },
      { kind: "action", tool: "keyboard", summary: "press cmd+s" },
      { kind: "action", tool: "mouse", summary: "click save" },
    ]);
  });

  it("produces a dataset a backend can train on end to end", async () => {
    const replays: ExportedReplayManifest[] = [
      {
        sessionId: "s1",
        trajectoryIds: ["t1"],
        eventCount: 2,
        events: [
          { kind: "observation", ts: 1, trajectoryId: "t1", source: "window", summary: "editor focused" },
          { kind: "action", ts: 2, trajectoryId: "t1", tool: "keyboard", summary: "press cmd+s" },
        ],
      },
    ];
    const model = await new DeterministicMovementBackend().train(datasetFromReplays(replays));
    const prediction = model.predictNext([{ kind: "observation", source: "window", summary: "editor focused" }]);
    expect(prediction?.action.tool).toBe("keyboard");
  });
});

describe("evaluateMovementModel", () => {
  it("scores perfect accuracy when replaying the training sequence", async () => {
    const model = await new DeterministicMovementBackend().train({ sequences: [OPEN_SEQ] });
    const evaluation = evaluateMovementModel(model, [OPEN_SEQ]);
    expect(evaluation.total).toBe(2);
    expect(evaluation.correct).toBe(2);
    expect(evaluation.accuracy).toBe(1);
    expect(evaluation.generalizedHits).toBe(0);
  });

  it("measures generalization on a held-out but related trajectory", async () => {
    // Train on one save-flow; hold out a related flow with the same sources/tools
    // but different summaries. Coarse back-off should recover the right action.
    const train: MovementSequence = seq("train", [
      { kind: "observation", source: "editor", summary: "file A open" },
      { kind: "action", tool: "keyboard", summary: "save" },
    ]);
    const held: MovementSequence = seq("held", [
      { kind: "observation", source: "editor", summary: "file B open" },
      { kind: "action", tool: "keyboard", summary: "save" },
    ]);
    const model = await new DeterministicMovementBackend().train({ sequences: [train] });
    const evaluation = evaluateMovementModel(model, [held]);
    expect(evaluation.accuracy).toBe(1);
    expect(evaluation.generalizedHits).toBe(1);
  });

  it("reports zero accuracy on an empty held-out set", async () => {
    const model = await new DeterministicMovementBackend().train({ sequences: [OPEN_SEQ] });
    expect(evaluateMovementModel(model, [])).toEqual({
      correct: 0,
      total: 0,
      accuracy: 0,
      generalizedHits: 0,
    });
  });
});
