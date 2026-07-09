import { describe, expect, it } from "vitest";
import { buildTrajectorySpan } from "../capture/trajectory.js";
import {
  MOVEMENT_EOS,
  MarkovMovementBackend,
  buildMovementDatasetFromSpans,
  createMovementModelBackend,
  movementActionToken,
  type MovementDataset,
} from "./movement-model.js";

const dataset: MovementDataset = {
  version: 1,
  sequences: [
    { id: "s1", tokens: ["a", "b", "c", "d"] },
    { id: "s2", tokens: ["a", "b", "c", "e"] },
    { id: "s3", tokens: ["a", "b", "c", "d"] },
  ],
};

describe("MarkovMovementBackend", () => {
  it("repeats a recorded sequence by rolling out from its seed", async () => {
    const backend = new MarkovMovementBackend({ order: 2 });
    const model = await backend.train({
      version: 1,
      sequences: [{ id: "only", tokens: ["x", "y", "z", "w"] }],
    });
    expect(backend.generate(model, ["x"])).toEqual(["y", "z", "w"]);
    // Generation terminates at the learned end-of-sequence sentinel.
    expect(backend.generate(model, ["x", "y", "z", "w"])).toEqual([]);
  });

  it("predicts the most-likely next token deterministically", async () => {
    const backend = new MarkovMovementBackend({ order: 2 });
    const model = await backend.train(dataset);
    // After "a b c", d appears twice and e once -> d wins with p = 2/3.
    const prediction = backend.predictNext(model, ["a", "b", "c"]);
    expect(prediction.token).toBe("d");
    expect(prediction.probability).toBeCloseTo(2 / 3, 10);
    expect(prediction.alternatives.map((alt) => alt.token)).toEqual(["d", "e"]);
  });

  it("generalizes to an unseen context via back-off", async () => {
    const backend = new MarkovMovementBackend({ order: 3 });
    const model = await backend.train(dataset);
    // "z b c" was never seen at order 3, but "b c" -> d/e is known at order 2.
    const prediction = backend.predictNext(model, ["z", "b", "c"]);
    expect(prediction.token).toBe("d");
    expect(prediction.contextOrder).toBeLessThan(3);
    expect(prediction.contextOrder).toBeGreaterThan(0);
  });

  it("falls back to the unigram distribution for a fully novel context", async () => {
    const backend = new MarkovMovementBackend({ order: 2 });
    const model = await backend.train(dataset);
    const prediction = backend.predictNext(model, ["totally", "novel"]);
    // Unigram: "a"(3), "b"(3), "c"(3), "d"(2), "e"(1), plus EOS(3).
    expect(prediction.contextOrder).toBe(0);
    expect(prediction.token).not.toBeNull();
  });

  it("returns a null prediction for an empty model", async () => {
    const backend = new MarkovMovementBackend();
    const model = await backend.train({ version: 1, sequences: [] });
    const prediction = backend.predictNext(model, ["a"]);
    expect(prediction.token).toBeNull();
    expect(prediction.alternatives).toEqual([]);
  });

  it("produces a JSON round-trippable model", async () => {
    const backend = new MarkovMovementBackend({ order: 2 });
    const model = await backend.train(dataset);
    const restored = JSON.parse(JSON.stringify(model));
    expect(restored.vocabulary).toEqual(["a", "b", "c", "d", "e"]);
    expect(restored.tokenCount).toBe(12);
    expect(restored.sequenceCount).toBe(3);
    // Prediction is identical whether from the live or the deserialized model.
    expect(backend.predictNext(restored, ["a", "b", "c"]).token).toBe(
      backend.predictNext(model, ["a", "b", "c"]).token,
    );
  });

  it("does not emit BOS/EOS sentinels as generated tokens", async () => {
    const backend = new MarkovMovementBackend({ order: 2 });
    const model = await backend.train(dataset);
    const generated = backend.generate(model, [], { maxSteps: 20 });
    expect(generated).not.toContain(MOVEMENT_EOS);
    expect(generated.length).toBeLessThanOrEqual(20);
  });
});

describe("dataset helpers", () => {
  it("derives stable tokens from action tool + summary", () => {
    expect(movementActionToken({ tool: "Device", summary: "  Tapped   Submit  " })).toBe(
      "device:tapped submit",
    );
  });

  it("builds a dataset from trajectory spans in timestamp order", () => {
    const span = buildTrajectorySpan({
      id: "traj-1",
      sessionId: "sess-1",
      actions: [
        { kind: "action", tool: "keyboard", summary: "type hello", ts: 30 },
        { kind: "action", tool: "mouse", summary: "click field", ts: 10 },
      ],
    });
    const built = buildMovementDatasetFromSpans([span]);
    expect(built.sequences).toEqual([
      { id: "traj-1", tokens: ["mouse:click field", "keyboard:type hello"] },
    ]);
  });

  it("drops spans below the minimum token threshold", () => {
    const empty = buildTrajectorySpan({ id: "t0", sessionId: "s", actions: [] });
    const one = buildTrajectorySpan({
      id: "t1",
      sessionId: "s",
      actions: [{ kind: "action", tool: "mouse", summary: "click", ts: 1 }],
    });
    expect(buildMovementDatasetFromSpans([empty, one], { minTokens: 1 }).sequences).toHaveLength(1);
    expect(buildMovementDatasetFromSpans([empty, one], { minTokens: 2 }).sequences).toHaveLength(0);
  });
});

describe("createMovementModelBackend", () => {
  it("creates a markov backend by default", () => {
    expect(createMovementModelBackend().id).toBe("markov-2");
    expect(createMovementModelBackend("markov", { order: 3 }).id).toBe("markov-3");
  });
});
