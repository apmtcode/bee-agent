import { describe, expect, it } from "vitest";
import {
  MarkovMovementBackend,
  MOVEMENT_END_TOKEN,
  MovementBackendRegistry,
  buildMovementDatasetFromExport,
  buildMovementDatasetFromReplays,
  defaultMovementBackendRegistry,
  deserializeMovementModel,
  type MovementDataset,
} from "./movement-model.js";
import type { ExportedReplayManifest, ReviewedExportManifest } from "./export-manifest.js";

function actionEvent(tool: string, ts: number): ExportedReplayManifest["events"][number] {
  return { kind: "action", ts, trajectoryId: "t1", tool, summary: `${tool} step` };
}

const openFileFlow: MovementDataset = {
  samples: [
    { id: "a", tokens: ["focus-window", "click-menu", "click-open", "type-path", "press-enter"] },
    { id: "b", tokens: ["focus-window", "click-menu", "click-open", "type-path", "press-enter"] },
    { id: "c", tokens: ["focus-window", "click-menu", "click-save", "type-path", "press-enter"] },
  ],
};

describe("MarkovMovementBackend", () => {
  it("trains a serializable model that captures the recorded transitions", async () => {
    const backend = new MarkovMovementBackend();
    const model = await backend.train(openFileFlow, { order: 2, trainedAt: "2026-07-14T00:00:00Z" });

    expect(model.backend).toBe("markov");
    expect(model.order).toBe(2);
    expect(model.sampleCount).toBe(3);
    expect(model.tokenCount).toBe(15);
    expect(model.vocab).toContain("click-open");
    expect(model.vocab).toContain(MOVEMENT_END_TOKEN);
    // Model is plain JSON — round-trips through serialization.
    expect(deserializeMovementModel(JSON.parse(JSON.stringify(model)))).toEqual(model);
  });

  it("predicts the most likely next movement from a known context", async () => {
    const backend = new MarkovMovementBackend();
    const model = await backend.train(openFileFlow, { order: 2 });

    const ranked = backend.predictNext(model, ["focus-window", "click-menu"]);
    // "click-open" (2/3) should beat "click-save" (1/3).
    expect(ranked[0]!.token).toBe("click-open");
    expect(ranked[0]!.probability).toBeGreaterThan(ranked[1]!.probability);
    expect(ranked[0]!.contextUsed).toBe(2);
  });

  it("generalizes to unseen prefixes via stupid-backoff", async () => {
    const backend = new MarkovMovementBackend();
    const model = await backend.train(openFileFlow, { order: 3 });

    // This 2-token history was never observed as a full order-3 context, so the
    // model must back off to a shorter history rather than return uniform noise.
    const ranked = backend.predictNext(model, ["never-seen-token", "click-menu"]);
    expect(ranked[0]!.token).toBe("click-open");
    expect(ranked[0]!.contextUsed).toBeLessThan(3);
    expect(ranked[0]!.contextUsed).toBeGreaterThan(0);
  });

  it("generates a full movement sequence terminating at END", async () => {
    const backend = new MarkovMovementBackend();
    const model = await backend.train(openFileFlow, { order: 2 });

    const generated = backend.generate(model, ["focus-window"], 32);
    expect(generated[0]).toBe("click-menu");
    expect(generated).not.toContain(MOVEMENT_END_TOKEN);
    // The dominant learned path is the open-file flow.
    expect(generated).toEqual(["click-menu", "click-open", "type-path", "press-enter"]);
  });

  it("is deterministic across repeated training and inference", async () => {
    const backend = new MarkovMovementBackend();
    const a = await backend.train(openFileFlow, { order: 2 });
    const b = await backend.train(openFileFlow, { order: 2 });
    expect(a).toEqual(b);
    expect(backend.generate(a, ["focus-window"])).toEqual(backend.generate(b, ["focus-window"]));
  });

  it("scores high replay fidelity on held-out but related movements", async () => {
    const backend = new MarkovMovementBackend();
    const model = await backend.train(openFileFlow, { order: 2 });

    const heldOut: MovementDataset = {
      samples: [{ id: "held", tokens: ["focus-window", "click-menu", "click-open", "type-path", "press-enter"] }],
    };
    const evalResult = backend.evaluate(model, heldOut);
    expect(evalResult.predictions).toBe(6); // 5 tokens + END
    expect(evalResult.accuracy).toBeGreaterThan(0.8);
    expect(evalResult.perplexity).toBeGreaterThan(0);
    expect(Number.isFinite(evalResult.perplexity)).toBe(true);
  });

  it("handles an empty dataset without throwing", async () => {
    const backend = new MarkovMovementBackend();
    const model = await backend.train({ samples: [] });
    expect(model.sampleCount).toBe(0);
    expect(backend.generate(model, [])).toEqual([]);
    expect(backend.evaluate(model, { samples: [] }).accuracy).toBe(0);
  });
});

describe("MovementBackendRegistry", () => {
  it("provides the markov backend by default", () => {
    expect(defaultMovementBackendRegistry.has("markov")).toBe(true);
    expect(defaultMovementBackendRegistry.names()).toContain("markov");
    expect(defaultMovementBackendRegistry.create("markov")).toBeInstanceOf(MarkovMovementBackend);
  });

  it("supports registering a pluggable custom backend", () => {
    const registry = new MovementBackendRegistry();
    const stub = {
      name: "on-device-stub",
      train: async () => ({
        version: 1 as const,
        backend: "on-device-stub",
        order: 1,
        smoothing: 0,
        vocab: [],
        transitions: {},
        sampleCount: 0,
        tokenCount: 0,
      }),
      predictNext: () => [],
      generate: () => [],
      evaluate: () => ({ sampleCount: 0, predictions: 0, correct: 0, accuracy: 0, perplexity: 0 }),
    };
    registry.register("on-device-stub", () => stub);
    expect(registry.create("on-device-stub").name).toBe("on-device-stub");
  });

  it("throws a helpful error for an unknown backend", () => {
    const registry = new MovementBackendRegistry();
    expect(() => registry.create("does-not-exist")).toThrow(/Unknown movement backend/);
  });
});

describe("dataset adapters", () => {
  it("builds a movement dataset from replay action events only", () => {
    const replays: ExportedReplayManifest[] = [
      {
        sessionId: "s1",
        trajectoryIds: ["t1"],
        eventCount: 3,
        events: [
          { kind: "transcript", ts: 1, messageId: "m1", role: "user", content: "do it" },
          actionEvent("click", 2),
          { kind: "observation", ts: 3, trajectoryId: "t1", source: "screen", summary: "opened" },
          actionEvent("type", 4),
        ],
      },
    ];
    const dataset = buildMovementDatasetFromReplays(replays);
    expect(dataset.samples).toHaveLength(1);
    expect(dataset.samples[0]!.tokens).toEqual(["click", "type"]);
  });

  it("drops replays with no action events and honors a custom tokenizer", () => {
    const replays: ExportedReplayManifest[] = [
      { sessionId: "empty", trajectoryIds: [], eventCount: 0, events: [] },
      { sessionId: "s2", trajectoryIds: ["t1"], eventCount: 1, events: [actionEvent("scroll", 1)] },
    ];
    const dataset = buildMovementDatasetFromReplays(replays, (action) => `${action.tool}:${action.summary}`);
    expect(dataset.samples).toHaveLength(1);
    expect(dataset.samples[0]!.tokens).toEqual(["scroll:scroll step"]);
  });

  it("builds a dataset directly from a reviewed export manifest", () => {
    const manifest = {
      replays: [
        { sessionId: "s3", trajectoryIds: ["t1"], eventCount: 1, events: [actionEvent("drag", 1)] },
      ],
    } as unknown as ReviewedExportManifest;
    const dataset = buildMovementDatasetFromExport(manifest);
    expect(dataset.samples[0]!.tokens).toEqual(["drag"]);
  });
});

describe("deserializeMovementModel", () => {
  it("rejects malformed models", () => {
    expect(() => deserializeMovementModel(null)).toThrow(/not an object/);
    expect(() => deserializeMovementModel({ version: 2 })).toThrow(/missing required fields/);
  });
});
