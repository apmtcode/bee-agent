import { describe, expect, it } from "vitest";
import { buildMovementDataset } from "./movement-dataset.js";
import { generateSyntheticTrajectories } from "./synthetic-movements.js";
import { NgramMovementBackend, NGRAM_BACKEND_ID, createDefaultMovementBackend } from "./ngram-backend.js";
import { MovementModelBackendRegistry } from "./model-backend.js";

const backend = new NgramMovementBackend();

describe("NgramMovementBackend training + replay", () => {
  it("reproduces a single recorded movement sequence exactly", async () => {
    const dataset = buildMovementDataset(
      generateSyntheticTrajectories({ seed: 7, count: 1, lengthRange: [4, 4] }),
    );
    const recorded = dataset.sequences[0].tokens.map((token) => token.symbol);

    const model = await backend.train(dataset, { order: 3 });
    // Generation from the start sentinel replays the recorded movements.
    expect(model.generate()).toEqual(recorded);
  });

  it("predicts the next movement from a known context", async () => {
    const dataset = buildMovementDataset(
      generateSyntheticTrajectories({ seed: 7, count: 1, lengthRange: [4, 4] }),
    );
    const symbols = dataset.sequences[0].tokens.map((token) => token.symbol);
    const model = await backend.train(dataset, { order: 2 });

    const prediction = model.predict(["<s>", symbols[0]]);
    expect(prediction?.symbol).toBe(symbols[1]);
    expect(prediction?.probability).toBeGreaterThan(0);
  });
});

describe("NgramMovementBackend generalization", () => {
  it("backs off to shorter context for unseen prefixes", async () => {
    // Two sequences share a common suffix but diverge on their prefix.
    const dataset = {
      version: 1 as const,
      sequences: [
        {
          trajectoryId: "a",
          tokens: [
            { type: "action" as const, symbol: "action:mouse:move", channel: "mouse", summary: "move" },
            { type: "action" as const, symbol: "action:mouse:click", channel: "mouse", summary: "click" },
            { type: "action" as const, symbol: "action:window:submit", channel: "window", summary: "submit" },
          ],
        },
        {
          trajectoryId: "b",
          tokens: [
            { type: "action" as const, symbol: "action:keyboard:type", channel: "keyboard", summary: "type" },
            { type: "action" as const, symbol: "action:mouse:click", channel: "mouse", summary: "click" },
            { type: "action" as const, symbol: "action:window:submit", channel: "window", summary: "submit" },
          ],
        },
      ],
      vocabulary: ["action:keyboard:type", "action:mouse:click", "action:mouse:move", "action:window:submit"],
      representatives: {},
    };
    const model = await backend.train(dataset, { order: 2 });

    // A novel high-order context (never seen as a bigram) still generalizes:
    // "click" is always followed by "submit", so backoff recovers it.
    const prediction = model.predict(["action:unknown:foo", "action:mouse:click"]);
    expect(prediction?.symbol).toBe("action:window:submit");
    expect(prediction?.order).toBeLessThanOrEqual(1);
  });

  it("returns no prediction when nothing was learned", async () => {
    const model = await backend.train(
      { version: 1, sequences: [], vocabulary: [], representatives: {} },
      { order: 2 },
    );
    expect(model.predict(["anything"])).toBeUndefined();
    expect(model.generate()).toEqual([]);
  });
});

describe("NgramMovementModel serialization", () => {
  it("round-trips through serialize/load with identical behavior", async () => {
    const dataset = buildMovementDataset(
      generateSyntheticTrajectories({ seed: 11, count: 3, lengthRange: [3, 5] }),
    );
    const model = await backend.train(dataset, { order: 2 });
    const restored = backend.load(model.serialize());

    for (const sequence of dataset.sequences) {
      const symbols = sequence.tokens.map((token) => token.symbol);
      expect(restored.generate(["<s>", symbols[0]])).toEqual(model.generate(["<s>", symbols[0]]));
    }
    expect(restored.serialize()).toEqual(model.serialize());
    expect(restored.order).toBe(model.order);
  });
});

describe("MovementModelBackendRegistry", () => {
  it("registers and resolves backends by id", () => {
    const registry = new MovementModelBackendRegistry().register(createDefaultMovementBackend());
    expect(registry.has(NGRAM_BACKEND_ID)).toBe(true);
    expect(registry.get(NGRAM_BACKEND_ID).id).toBe(NGRAM_BACKEND_ID);
    expect(registry.list()).toEqual([NGRAM_BACKEND_ID]);
  });

  it("rejects duplicate registration and unknown lookups", () => {
    const registry = new MovementModelBackendRegistry().register(createDefaultMovementBackend());
    expect(() => registry.register(createDefaultMovementBackend())).toThrow(/already registered/);
    expect(() => registry.get("does-not-exist")).toThrow(/unknown movement backend/);
  });
});
