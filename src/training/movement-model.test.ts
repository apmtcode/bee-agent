import { describe, expect, it } from "vitest";
import type { TrajectorySpan } from "../capture/trajectory.js";
import {
  buildMovementDataset,
  buildMovementSequence,
  createDefaultMovementModelRegistry,
  datasetFromTokenSequences,
  evaluateMovementGeneralization,
  generateSyntheticMovementTrajectories,
  loadMovementModelSnapshot,
  MovementModelBackendRegistry,
  NGramMovementModelBackend,
  regenerateSequence,
  splitMovementDataset,
  tokenizeAction,
  type MovementModelBackend,
} from "./movement-model.js";

function trajectory(id: string, actions: Array<{ tool: string; summary: string; ts: number; metadata?: Record<string, unknown> }>): TrajectorySpan {
  return {
    id,
    sessionId: "s1",
    createdAt: "2026-01-01T00:00:00.000Z",
    captureTier: "app",
    observations: [],
    actions: actions.map((action) => ({ kind: "action", ...action })),
  };
}

describe("tokenizeAction", () => {
  it("prefers structured gesture metadata", () => {
    expect(tokenizeAction({ tool: "device", summary: "tapped Send", metadata: { gesture: "tap", target: "Send Button" } })).toBe(
      "device:tap:send-button",
    );
    expect(tokenizeAction({ tool: "device", summary: "scrolled", metadata: { gesture: "scroll", direction: "down" } })).toBe(
      "device:scroll:down",
    );
  });

  it("falls back to the summary when no metadata is present", () => {
    expect(tokenizeAction({ tool: "browser", summary: "Clicked link" })).toBe("browser:act:clicked-link");
  });
});

describe("buildMovementDataset", () => {
  it("orders actions by timestamp and collects a sorted vocabulary", () => {
    const seq = buildMovementSequence(
      trajectory("t1", [
        { tool: "device", summary: "b", ts: 20, metadata: { gesture: "tap", target: "b" } },
        { tool: "device", summary: "a", ts: 10, metadata: { gesture: "tap", target: "a" } },
      ]),
    );
    expect(seq.tokens).toEqual(["device:tap:a", "device:tap:b"]);

    const dataset = buildMovementDataset([
      trajectory("t1", [{ tool: "device", summary: "a", ts: 1, metadata: { gesture: "tap", target: "a" } }]),
      trajectory("empty", []),
    ]);
    expect(dataset.sequences).toHaveLength(1);
    expect(dataset.vocabulary).toEqual(["device:tap:a"]);
  });
});

describe("NGramMovementModelBackend", () => {
  const backend = new NGramMovementModelBackend();

  it("repeats every distinct recorded sequence exactly", () => {
    const dataset = datasetFromTokenSequences([
      { id: "q1", tokens: ["a", "b", "c"] },
      { id: "q2", tokens: ["p", "x", "y", "z"] },
      { id: "q3", tokens: ["m", "n"] },
    ]);
    const model = backend.train(dataset, { order: 2 });
    for (const sequence of dataset.sequences) {
      expect(regenerateSequence(model, sequence)).toEqual(sequence.tokens);
    }
  });

  it("predicts ranked candidates with stupid-backoff for unseen contexts", () => {
    const dataset = datasetFromTokenSequences([
      { id: "q1", tokens: ["tap", "type", "send"] },
      { id: "q2", tokens: ["tap", "type", "cancel"] },
      { id: "q3", tokens: ["tap", "type", "send"] },
    ]);
    const model = backend.train(dataset, { order: 2 });

    // Seen bigram context "tap,type" -> "send" (count 2) beats "cancel" (count 1).
    const seen = model.predictNext(["tap", "type"]);
    expect(seen[0]?.token).toBe("send");
    expect(seen[0]!.probability).toBeGreaterThan(seen[1]!.probability);

    // Unseen longer context backs off to the known suffix and still predicts.
    const backedOff = model.predictNext(["zzz", "tap", "type"]);
    expect(backedOff[0]?.token).toBe("send");
  });

  it("round-trips through a snapshot without changing predictions", () => {
    const dataset = datasetFromTokenSequences([{ id: "q1", tokens: ["a", "b", "c"] }]);
    const model = backend.train(dataset, { order: 2 });
    const restored = loadMovementModelSnapshot(model.toSnapshot());
    expect(restored.order).toBe(model.order);
    expect(restored.predictNext(["a", "b"])).toEqual(model.predictNext(["a", "b"]));
    expect(restored.generate(["a"], 5)).toEqual(model.generate(["a"], 5));
  });
});

describe("synthetic movement generation", () => {
  it("is reproducible for a given seed", () => {
    const a = generateSyntheticMovementTrajectories({ seed: 7, variantsPerTemplate: 5 });
    const b = generateSyntheticMovementTrajectories({ seed: 7, variantsPerTemplate: 5 });
    expect(JSON.stringify(a)).toBe(JSON.stringify(b));
    // Different seed diverges.
    const c = generateSyntheticMovementTrajectories({ seed: 99, variantsPerTemplate: 5 });
    expect(JSON.stringify(c)).not.toBe(JSON.stringify(a));
  });

  it("emits device-adapter-shaped action metadata", () => {
    const [first] = generateSyntheticMovementTrajectories({ seed: 1, variantsPerTemplate: 1 });
    expect(first!.actions[0]!.tool).toBe("device");
    expect(first!.actions[0]!.metadata).toHaveProperty("gesture");
  });
});

describe("evaluateMovementGeneralization", () => {
  it("repeats training movements and generalizes to unseen related ones", () => {
    const backend = new NGramMovementModelBackend();
    // Held-out sequences come from a different seed: never-seen variants of the
    // very same task grammar, so accuracy above chance is genuine generalization.
    const train = buildMovementDataset(generateSyntheticMovementTrajectories({ seed: 42, variantsPerTemplate: 12, sessionId: "train" }));
    const heldOut = buildMovementDataset(generateSyntheticMovementTrajectories({ seed: 4242, variantsPerTemplate: 4, sessionId: "held" }));

    const metrics = evaluateMovementGeneralization({ backend, train, heldOut, config: { order: 2 } });
    expect(metrics.backendId).toBe("deterministic-ngram");
    expect(metrics.heldOutPredictedSteps).toBeGreaterThan(0);
    // Generalizes: predicts the next movement on unseen but related sequences well.
    expect(metrics.nextMovementAccuracy).toBeGreaterThan(0.6);
    // Some exact-repeat capacity survives even on a noisy multi-variant set,
    // where greedy regeneration collapses colliding contexts to the dominant
    // path (exact-repeat on non-colliding sequences is covered above).
    expect(metrics.replayFidelity).toBeGreaterThan(0.2);
    expect(metrics.replayFidelity).toBeLessThanOrEqual(1);
  });

  it("a higher-order model is at least as literal as a unigram baseline", () => {
    const backend = new NGramMovementModelBackend();
    const dataset = buildMovementDataset(generateSyntheticMovementTrajectories({ seed: 5, variantsPerTemplate: 6 }));
    const { train, heldOut } = splitMovementDataset(dataset, 0.34);
    const baseline = evaluateMovementGeneralization({ backend, train, heldOut, config: { order: 0 } });
    const contextual = evaluateMovementGeneralization({ backend, train, heldOut, config: { order: 2 } });
    expect(contextual.nextMovementAccuracy).toBeGreaterThanOrEqual(baseline.nextMovementAccuracy);
  });
});

describe("MovementModelBackendRegistry", () => {
  it("ships the deterministic backend by default and supports custom backends", () => {
    const registry = createDefaultMovementModelRegistry();
    expect(registry.list()).toContain("deterministic-ngram");
    expect(registry.require("deterministic-ngram").id).toBe("deterministic-ngram");
    expect(() => registry.require("missing")).toThrow(/unknown movement-model backend/);

    const fake: MovementModelBackend = {
      id: "fake",
      train: (dataset) => new NGramMovementModelBackend().train(dataset),
    };
    const custom = new MovementModelBackendRegistry([fake]);
    expect(custom.list()).toEqual(["fake"]);
    expect(custom.get("fake")?.id).toBe("fake");
  });
});
