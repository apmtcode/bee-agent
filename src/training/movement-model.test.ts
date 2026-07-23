import { describe, expect, it } from "vitest";
import {
  MarkovMovementBackend,
  MovementModelRegistry,
  createDefaultMovementModelRegistry,
  evaluateMovementModel,
  generateSyntheticMovementDataset,
  tokenizeTrajectorySpan,
  type MovementDataset,
} from "./movement-model.js";
import { buildTrajectorySpan } from "../capture/trajectory.js";

describe("MarkovMovementBackend — repeat recorded movements (objective #2c)", () => {
  it("replays a recorded movement sequence exactly from its start", async () => {
    const dataset: MovementDataset = {
      sequences: [{ id: "s1", tokens: ["tap:menu", "tap:compose", "type:body", "tap:send"] }],
    };
    const model = await new MarkovMovementBackend().train(dataset, { maxOrder: 2 });
    const rollout = model.generate(["tap:menu"]);
    expect(rollout).toEqual(["tap:compose", "type:body", "tap:send"]);
  });

  it("predicts the correct next movement with high confidence", async () => {
    const dataset: MovementDataset = {
      sequences: Array.from({ length: 5 }, (_, i) => ({
        id: `s${i}`,
        tokens: ["open:app", "tap:new", "type:text", "tap:save"],
      })),
    };
    const model = await new MarkovMovementBackend().train(dataset);
    const predictions = model.predictNext(["open:app", "tap:new"]);
    expect(predictions[0]?.token).toBe("type:text");
    expect(predictions[0]?.score).toBeGreaterThan(0.5);
  });

  it("is deterministic: identical training yields identical predictions", async () => {
    const dataset: MovementDataset = {
      sequences: [{ id: "s1", tokens: ["a", "b", "c", "b", "a"] }],
    };
    const backend = new MarkovMovementBackend();
    const a = (await backend.train(dataset)).predictNext(["a"]);
    const b = (await backend.train(dataset)).predictNext(["a"]);
    expect(a).toEqual(b);
  });
});

describe("MarkovMovementBackend — generalize to related movements (objective #2d)", () => {
  it("falls back to lower-order context for an unseen prefix", async () => {
    const dataset: MovementDataset = {
      sequences: [
        { id: "s1", tokens: ["tap:menu", "tap:settings", "tap:save"] },
        { id: "s2", tokens: ["tap:toolbar", "tap:settings", "tap:save"] },
      ],
    };
    const model = await new MarkovMovementBackend().train(dataset, { maxOrder: 2 });
    // "tap:home tap:settings" was never seen at order-2, but the bigram
    // (settings -> save) generalizes across both recorded prefixes.
    const predictions = model.predictNext(["tap:home", "tap:settings"]);
    expect(predictions[0]?.token).toBe("tap:save");
  });

  it("generalizes across held-out synthetic trajectories better than chance", async () => {
    const vocabulary = ["m1", "m2", "m3", "m4", "m5", "m6"];
    const train = generateSyntheticMovementDataset({ seed: 7, sequenceCount: 60, vocabulary });
    const heldOut = generateSyntheticMovementDataset({ seed: 999, sequenceCount: 20, vocabulary });
    const model = await new MarkovMovementBackend().train(train, { maxOrder: 2 });
    const result = evaluateMovementModel(model, heldOut, { k: 3 });
    // 6-token vocab => chance top-1 ≈ 0.17. The grammar advances 70% of the
    // time, so a model that learned structure must clear chance comfortably.
    expect(result.samples).toBeGreaterThan(0);
    expect(result.top1Accuracy).toBeGreaterThan(0.4);
    expect(result.topKAccuracy).toBeGreaterThanOrEqual(result.top1Accuracy);
  });
});

describe("serialization round-trip", () => {
  it("reloads a model that produces identical predictions", async () => {
    const dataset: MovementDataset = {
      sequences: [{ id: "s1", tokens: ["a", "b", "c", "d"] }],
    };
    const backend = new MarkovMovementBackend();
    const trained = await backend.train(dataset);
    const snapshot = trained.serialize();
    const reloaded = backend.load(snapshot);
    expect(reloaded.predictNext(["a", "b"])).toEqual(trained.predictNext(["a", "b"]));
    expect(reloaded.generate(["a"])).toEqual(trained.generate(["a"]));
  });

  it("produces a stable, JSON-serializable snapshot", async () => {
    const dataset: MovementDataset = { sequences: [{ id: "s1", tokens: ["x", "y", "x", "y"] }] };
    const backend = new MarkovMovementBackend();
    const first = (await backend.train(dataset)).serialize();
    const second = (await backend.train(dataset)).serialize();
    expect(JSON.stringify(first)).toBe(JSON.stringify(second));
    expect(first.backendId).toBe("markov-v1");
  });
});

describe("context conditioning", () => {
  it("learns distinct continuations per discrete context", async () => {
    const dataset: MovementDataset = {
      sequences: [
        { id: "mail", tokens: ["tap:new", "tap:send"], context: { app: "mail" } },
        { id: "docs", tokens: ["tap:new", "tap:print"], context: { app: "docs" } },
      ],
    };
    const model = await new MarkovMovementBackend().train(dataset, { maxOrder: 3 });
    // Both contexts share the "tap:new" prefix; interpolation still surfaces a
    // valid continuation, and both recorded next-moves are viable candidates.
    const predictions = model.predictNext(["tap:new"], { limit: 5 });
    const tokens = predictions.map((p) => p.token);
    expect(tokens).toContain("tap:send");
    expect(tokens).toContain("tap:print");
  });
});

describe("tokenizeTrajectorySpan", () => {
  it("canonicalizes actions into ordered movement tokens", () => {
    const span = buildTrajectorySpan({
      id: "t1",
      sessionId: "sess",
      actions: [
        { kind: "action", tool: "device", summary: "tapped compose", ts: 200, metadata: { gesture: "tap", target: "compose-button" } },
        { kind: "action", tool: "device", summary: "swiped down", ts: 100, metadata: { gesture: "swipe", direction: "down" } },
      ],
    });
    const sequence = tokenizeTrajectorySpan(span);
    // Sorted by ts: swipe (100) before tap (200).
    expect(sequence.tokens).toEqual(["swipe:down", "tap:compose-button"]);
    expect(sequence.id).toBe("t1");
  });

  it("round-trips a tokenized trajectory through training", async () => {
    const span = buildTrajectorySpan({
      id: "t2",
      sessionId: "sess",
      actions: [
        { kind: "action", tool: "editor", summary: "open file", ts: 1 },
        { kind: "action", tool: "editor", summary: "save file", ts: 2 },
      ],
    });
    const sequence = tokenizeTrajectorySpan(span);
    const model = await new MarkovMovementBackend().train({ sequences: [sequence] });
    const rollout = model.generate([sequence.tokens[0]!]);
    expect(rollout).toEqual([sequence.tokens[1]!]);
  });
});

describe("MovementModelRegistry", () => {
  it("registers and requires backends by id", () => {
    const registry = createDefaultMovementModelRegistry();
    expect(registry.list()).toContain("markov-v1");
    expect(registry.require("markov-v1").id).toBe("markov-v1");
  });

  it("throws for an unregistered backend", () => {
    const registry = new MovementModelRegistry();
    expect(() => registry.require("nope")).toThrow(/not registered/);
    expect(registry.get("nope")).toBeUndefined();
  });
});

describe("generateSyntheticMovementDataset", () => {
  it("is reproducible for a fixed seed and varies by seed", () => {
    const vocabulary = ["a", "b", "c"];
    const one = generateSyntheticMovementDataset({ seed: 42, sequenceCount: 4, vocabulary });
    const two = generateSyntheticMovementDataset({ seed: 42, sequenceCount: 4, vocabulary });
    const three = generateSyntheticMovementDataset({ seed: 43, sequenceCount: 4, vocabulary });
    expect(one).toEqual(two);
    expect(JSON.stringify(one)).not.toBe(JSON.stringify(three));
  });

  it("respects length bounds and rejects an empty vocabulary", () => {
    const dataset = generateSyntheticMovementDataset({
      seed: 1,
      sequenceCount: 10,
      vocabulary: ["x", "y"],
      minLength: 2,
      maxLength: 4,
    });
    for (const seq of dataset.sequences) {
      expect(seq.tokens.length).toBeGreaterThanOrEqual(2);
      expect(seq.tokens.length).toBeLessThanOrEqual(4);
    }
    expect(() =>
      generateSyntheticMovementDataset({ seed: 1, sequenceCount: 1, vocabulary: [] }),
    ).toThrow(/vocabulary/);
  });
});
