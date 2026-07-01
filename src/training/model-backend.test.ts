import { describe, expect, it } from "vitest";
import type { ExportedReplayManifest } from "./export-manifest.js";
import {
  MOVEMENT_EOS,
  MovementBackendRegistry,
  NGramMovementBackend,
  buildMovementDataset,
  createDefaultMovementBackendRegistry,
  extractMovementSequence,
  loadMovementModel,
  tokenizeMovementEvent,
} from "./model-backend.js";

function actionReplay(
  trajectoryId: string,
  actions: Array<{ tool: string; summary: string; ts: number }>,
): Pick<ExportedReplayManifest, "events"> {
  return {
    events: actions.map((action) => ({
      kind: "action" as const,
      ts: action.ts,
      trajectoryId,
      tool: action.tool,
      summary: action.summary,
    })),
  };
}

describe("tokenizeMovementEvent", () => {
  it("reduces action events to stable normalized tokens", () => {
    expect(
      tokenizeMovementEvent({ kind: "action", ts: 1, trajectoryId: "t", tool: "Mouse.Move", summary: "  To Toolbar " }),
    ).toBe("mouse.move:to-toolbar");
  });

  it("ignores non-action events", () => {
    expect(
      tokenizeMovementEvent({ kind: "observation", ts: 1, trajectoryId: "t", source: "screen", summary: "x" }),
    ).toBeUndefined();
    expect(
      tokenizeMovementEvent({ kind: "transcript", ts: 1, messageId: "m", role: "user", content: "hi" }),
    ).toBeUndefined();
  });
});

describe("buildMovementDataset", () => {
  it("extracts ordered movement sequences and EOS-terminates them", () => {
    const replay = actionReplay("t1", [
      { tool: "mouse", summary: "b", ts: 20 },
      { tool: "mouse", summary: "a", ts: 10 },
    ]);
    expect(extractMovementSequence(replay)).toEqual(["mouse:a", "mouse:b"]);

    const dataset = buildMovementDataset([replay], { order: 2 });
    expect(dataset.sequences).toEqual([["mouse:a", "mouse:b", MOVEMENT_EOS]]);
    expect(dataset.vocabulary).toEqual(["mouse:a", "mouse:b"]);
    // one sample per position (including predicting EOS from the final token)
    expect(dataset.samples).toHaveLength(3);
  });

  it("drops replays with no action events", () => {
    const dataset = buildMovementDataset([{ events: [] }]);
    expect(dataset.sequences).toHaveLength(0);
    expect(dataset.samples).toHaveLength(0);
  });
});

describe("NGramMovementBackend — repetition", () => {
  it("reproduces a recorded movement sequence exactly", async () => {
    const replay = actionReplay("t1", [
      { tool: "mouse", summary: "open-menu", ts: 1 },
      { tool: "mouse", summary: "hover-file", ts: 2 },
      { tool: "click", summary: "save", ts: 3 },
    ]);
    const dataset = buildMovementDataset([replay], { order: 3 });
    const model = await new NGramMovementBackend().train(dataset);

    const generated = model.generate(["mouse:open-menu"]);
    expect(generated).toEqual(["mouse:hover-file", "click:save"]);

    // A deterministic single-step prediction with full probability mass.
    const prediction = model.predictNext(["mouse:open-menu"]);
    expect(prediction?.token).toBe("mouse:hover-file");
    expect(prediction?.probability).toBe(1);
  });

  it("falls back to the unigram prior for a wholly unseen context", async () => {
    const dataset = buildMovementDataset([actionReplay("t1", [{ tool: "a", summary: "x", ts: 1 }])]);
    const model = await new NGramMovementBackend().train(dataset);
    // No transition keyed by "totally-unseen" exists, so prediction backs off
    // all the way to the unigram distribution rather than crashing.
    const prediction = model.predictNext(["totally-unseen"]);
    expect(prediction).toBeDefined();
    expect(prediction?.order).toBe(0);
  });

  it("returns undefined when the model has no data at all", async () => {
    const model = await new NGramMovementBackend().train(buildMovementDataset([]));
    expect(model.predictNext(["anything"])).toBeUndefined();
  });
});

describe("NGramMovementBackend — generalization via back-off", () => {
  it("predicts a sensible next move for a novel prefix sharing a suffix", async () => {
    // Two recorded flows share the sub-move "focus-search" -> "type-query".
    const training = [
      actionReplay("t1", [
        { tool: "key", summary: "cmd-l", ts: 1 },
        { tool: "focus", summary: "search", ts: 2 },
        { tool: "type", summary: "query", ts: 3 },
      ]),
      actionReplay("t2", [
        { tool: "menu", summary: "edit", ts: 1 },
        { tool: "focus", summary: "search", ts: 2 },
        { tool: "type", summary: "query", ts: 3 },
      ]),
    ];
    const dataset = buildMovementDataset(training, { order: 3 });
    const model = await new NGramMovementBackend().train(dataset);

    // Novel prefix never seen verbatim; longest matching suffix "focus:search"
    // backs off to the shared transition and predicts "type:query".
    const prediction = model.predictNext(["mouse:new-thing", "focus:search"]);
    expect(prediction?.token).toBe("type:query");
    expect(prediction?.order).toBe(1); // matched via the 1-token suffix, not the full context
  });
});

describe("serialization", () => {
  it("round-trips a trained model to an artifact and back", async () => {
    const dataset = buildMovementDataset([
      actionReplay("t1", [
        { tool: "mouse", summary: "a", ts: 1 },
        { tool: "mouse", summary: "b", ts: 2 },
      ]),
    ]);
    const model = await new NGramMovementBackend().train(dataset);
    const restored = loadMovementModel(model.serialize());

    expect(restored.serialize()).toEqual(model.serialize());
    expect(restored.generate(["mouse:a"])).toEqual(model.generate(["mouse:a"]));
  });
});

describe("MovementBackendRegistry", () => {
  it("resolves the default mock backend and lists ids", () => {
    const registry = createDefaultMovementBackendRegistry();
    expect(registry.has("ngram-mock")).toBe(true);
    expect(registry.list()).toContain("ngram-mock");
    expect(registry.create("ngram-mock")).toBeInstanceOf(NGramMovementBackend);
  });

  it("throws a helpful error for an unregistered backend", () => {
    const registry = new MovementBackendRegistry();
    expect(() => registry.create("mlx")).toThrowError(/unknown movement backend "mlx"/);
  });

  it("supports registering a custom on-device backend under a new id", () => {
    const registry = createDefaultMovementBackendRegistry();
    registry.register("mlx", () => new NGramMovementBackend({ id: "mlx", order: 5 }));
    const backend = registry.create("mlx");
    expect(backend.id).toBe("mlx");
  });
});
