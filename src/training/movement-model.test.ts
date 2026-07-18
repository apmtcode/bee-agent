import { describe, expect, it } from "vitest";
import {
  MarkovMovementBackend,
  MovementModelRegistry,
  createDefaultMovementModelRegistry,
  evaluateMovementModel,
  generateSyntheticMovementDataset,
  tokenizeTrajectoryAction,
  trajectoriesToMovementDataset,
  type MovementDataset,
} from "./movement-model.js";
import type { TrajectoryAction, TrajectorySpan } from "../capture/trajectory.js";

function dataset(sequences: Array<{ id: string; tokens: string[] }>): MovementDataset {
  return { version: 1, sequences };
}

describe("MarkovMovementBackend", () => {
  it("repeats a recorded movement chain given its opening context", () => {
    const backend = new MarkovMovementBackend();
    const tokens = ["tap:menu", "tap:search", "type:query", "tap:submit"];
    const model = backend.train(dataset([{ id: "a", tokens }]));

    // Seed with the first movement; the high-order chain reproduces the rest.
    const continuation = backend.generate(model, [tokens[0]!]);
    expect(continuation).toEqual(tokens.slice(1));

    // And it stops (does not run to maxSteps) once the recording ends.
    expect(continuation.length).toBe(tokens.length - 1);
  });

  it("generalizes to an unseen prefix by backing off to a known suffix", () => {
    const backend = new MarkovMovementBackend();
    // Every flow ends `type:query -> tap:submit`, reached from varied prefixes.
    const model = backend.train(
      dataset([
        { id: "a", tokens: ["tap:menu", "type:query", "tap:submit"] },
        { id: "b", tokens: ["scroll:down", "type:query", "tap:submit"] },
        { id: "c", tokens: ["tap:search", "type:query", "tap:submit"] },
      ]),
    );

    // A prefix never seen verbatim, but ending in the learned motif token.
    const prediction = backend.predictNext(model, ["swipe:left", "type:query"]);
    expect(prediction.token).toBe("tap:submit");
    expect(prediction.order).toBeGreaterThanOrEqual(1);
    expect(prediction.confidence).toBeGreaterThan(0);
  });

  it("is deterministic across repeated training and prediction", () => {
    const backend = new MarkovMovementBackend();
    const data = generateSyntheticMovementDataset({ count: 6, seed: 7 });
    const first = backend.generate(backend.train(data), ["tap:menu"]);
    const second = backend.generate(backend.train(data), ["tap:menu"]);
    expect(first).toEqual(second);
  });

  it("reports order -1 and no token for an empty model", () => {
    const backend = new MarkovMovementBackend();
    const model = backend.train(dataset([]));
    const prediction = backend.predictNext(model, ["tap:menu"]);
    expect(prediction.order).toBe(-1);
    expect(prediction.token).toBeUndefined();
    expect(prediction.stop).toBe(false);
  });

  it("falls back to the unconditional distribution when context is unknown", () => {
    const backend = new MarkovMovementBackend();
    const model = backend.train(dataset([{ id: "a", tokens: ["tap:menu", "tap:menu", "tap:search"] }]));
    // A context whose tokens never appear -> order 0 backoff.
    const prediction = backend.predictNext(model, ["totally:novel"]);
    expect(prediction.order).toBe(0);
    expect(prediction.token).toBe("tap:menu"); // most frequent token
  });
});

describe("tokenizeTrajectoryAction", () => {
  it("prefers gesture + target metadata", () => {
    const action: TrajectoryAction = {
      kind: "action",
      tool: "device",
      summary: "tapped Submit",
      ts: 1,
      metadata: { gesture: "tap", target: "Submit Button" },
    };
    expect(tokenizeTrajectoryAction(action)).toBe("tap:submit-button");
  });

  it("falls back to tool + summary when no gesture metadata", () => {
    const action: TrajectoryAction = {
      kind: "action",
      tool: "Bash",
      summary: "ran npm test",
      ts: 1,
    };
    expect(tokenizeTrajectoryAction(action)).toBe("bash:ran-npm-test");
  });
});

describe("trajectoriesToMovementDataset", () => {
  it("orders actions by timestamp and labels by outcome", () => {
    const span: TrajectorySpan = {
      id: "span-1",
      sessionId: "s1",
      createdAt: "2026-07-18T00:00:00.000Z",
      captureTier: "full",
      observations: [],
      actions: [
        { kind: "action", tool: "device", summary: "b", ts: 20, metadata: { gesture: "swipe", direction: "up" } },
        { kind: "action", tool: "device", summary: "a", ts: 10, metadata: { gesture: "tap", target: "Menu" } },
      ],
      outcome: { status: "success", summary: "done" },
    };
    const built = trajectoriesToMovementDataset([span]);
    expect(built.sequences).toHaveLength(1);
    expect(built.sequences[0]!.tokens).toEqual(["tap:menu", "swipe:up"]);
    expect(built.sequences[0]!.label).toBe("success");
  });
});

describe("evaluateMovementModel", () => {
  it("scores high on held-out sequences drawn from the same structured corpus", () => {
    const backend = new MarkovMovementBackend();
    const all = generateSyntheticMovementDataset({ count: 12, seed: 3 });
    const train = { version: 1 as const, sequences: all.sequences.slice(0, 9) };
    const test = all.sequences.slice(9);
    const model = backend.train(train);
    const report = evaluateMovementModel(backend, model, test);
    expect(report.sequences).toBe(test.length);
    expect(report.predictions).toBeGreaterThan(0);
    // The shared closing motif guarantees meaningful predictive fidelity.
    expect(report.accuracy).toBeGreaterThan(0.3);
  });

  it("reaches perfect accuracy when evaluating on the training sequence", () => {
    const backend = new MarkovMovementBackend();
    const tokens = ["tap:menu", "type:query", "tap:submit"];
    const model = backend.train(dataset([{ id: "a", tokens }]));
    const report = evaluateMovementModel(backend, model, [{ id: "a", tokens }]);
    expect(report.accuracy).toBe(1);
  });
});

describe("generateSyntheticMovementDataset", () => {
  it("is deterministic for a fixed seed and varies by seed", () => {
    const a = generateSyntheticMovementDataset({ count: 5, seed: 42 });
    const b = generateSyntheticMovementDataset({ count: 5, seed: 42 });
    const c = generateSyntheticMovementDataset({ count: 5, seed: 43 });
    expect(a).toEqual(b);
    expect(a).not.toEqual(c);
  });

  it("ends every sequence with the shared closing motif", () => {
    const built = generateSyntheticMovementDataset({ count: 6, seed: 9 });
    for (const sequence of built.sequences) {
      expect(sequence.tokens.slice(-2)).toEqual(["type:query", "tap:submit"]);
    }
  });
});

describe("MovementModelRegistry", () => {
  it("registers and requires backends by name", () => {
    const registry = new MovementModelRegistry().register(new MarkovMovementBackend());
    expect(registry.list()).toEqual(["markov-mock"]);
    expect(registry.require("markov-mock").name).toBe("markov-mock");
    expect(() => registry.require("nope")).toThrow(/unknown movement backend/);
  });

  it("default registry ships the mock backend", () => {
    const registry = createDefaultMovementModelRegistry();
    expect(registry.get("markov-mock")).toBeDefined();
  });
});
