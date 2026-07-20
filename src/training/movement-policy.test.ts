import { describe, expect, it } from "vitest";
import { buildTrajectorySpan } from "../capture/trajectory.js";
import {
  MarkovMovementBackend,
  defaultActionToken,
  sequenceFromTrajectorySpan,
  type MovementDataset,
} from "./movement-policy.js";

function dataset(sequences: string[][]): MovementDataset {
  return { sequences: sequences.map((tokens, i) => ({ id: `s${i}`, tokens })) };
}

describe("MarkovMovementBackend", () => {
  it("repeats a single recorded movement sequence exactly from its seed", () => {
    const backend = new MarkovMovementBackend();
    const model = backend.train(dataset([["a", "b", "c", "d"]]));
    expect(model.generate(["a"])).toEqual(["a", "b", "c", "d"]);
  });

  it("predicts the most likely next movement given context", () => {
    const backend = new MarkovMovementBackend();
    // "open" is followed by "type" twice and "quit" once → type wins.
    const model = backend.train(
      dataset([
        ["open", "type", "save"],
        ["open", "type", "close"],
        ["open", "quit"],
      ]),
    );
    const prediction = model.predict(["open"]);
    expect(prediction.token).toBe("type");
    expect(prediction.probability).toBeCloseTo(2 / 3, 5);
    // Distribution is sorted by probability descending.
    expect(prediction.distribution[0]?.token).toBe("type");
  });

  it("predicts end-of-sequence as a null token", () => {
    const backend = new MarkovMovementBackend();
    const model = backend.train(dataset([["a", "b"]]));
    expect(model.predict(["a", "b"]).token).toBeNull();
  });

  it("backs off to shorter contexts for novel-but-related prefixes", () => {
    const backend = new MarkovMovementBackend();
    const model = backend.train(
      dataset([
        ["x", "focus", "type", "save"],
        ["y", "focus", "type", "save"],
      ]),
      { maxOrder: 3 },
    );
    // "z focus" was never seen, but "focus" → "type" was: backoff generalizes.
    const prediction = model.predict(["z", "focus"]);
    expect(prediction.token).toBe("type");
    expect(prediction.order).toBeLessThan(2);
  });

  it("is deterministic across repeated generations", () => {
    const backend = new MarkovMovementBackend();
    const model = backend.train(dataset([["a", "b", "c"], ["a", "b", "d"]]));
    const first = model.generate(["a"]);
    const second = model.generate(["a"]);
    expect(first).toEqual(second);
  });

  it("caps generation length to avoid runaway loops", () => {
    const backend = new MarkovMovementBackend();
    // A self-loop: "a" is always followed by "a".
    const model = backend.train(dataset([["a", "a", "a", "a", "a", "a"]]), { maxOrder: 1 });
    const generated = model.generate(["a"], { maxLength: 5 });
    expect(generated.length).toBeLessThanOrEqual(6); // seed + maxLength
  });

  it("serializes and restores to an identical model", () => {
    const backend = new MarkovMovementBackend();
    const model = backend.train(
      dataset([
        ["open", "type", "save"],
        ["open", "type", "close"],
      ]),
      { maxOrder: 2 },
    );
    const restored = backend.restore(model.serialize());
    expect(restored.predict(["open"]).token).toBe(model.predict(["open"]).token);
    expect(restored.generate(["open"])).toEqual(model.generate(["open"]));
    // Round-trip is stable.
    expect(restored.serialize()).toEqual(model.serialize());
  });

  it("returns a null prediction for an untrained model", () => {
    const backend = new MarkovMovementBackend();
    const model = backend.train(dataset([]));
    expect(model.predict(["anything"]).token).toBeNull();
  });
});

describe("sequenceFromTrajectorySpan", () => {
  it("tokenizes recorded actions in chronological order", () => {
    const span = buildTrajectorySpan({
      id: "span-1",
      sessionId: "sess-1",
      actions: [
        { kind: "action", tool: "editor", summary: "save file", ts: 20 },
        { kind: "action", tool: "device", summary: "tapped Submit", ts: 10, metadata: { gesture: "tap", target: "Submit" } },
      ],
    });
    const sequence = sequenceFromTrajectorySpan(span);
    expect(sequence.tokens).toEqual(["device:tap:submit", "editor:save"]);
  });

  it("default tokenizer is stable for the same action shape", () => {
    const a = defaultActionToken({ kind: "action", tool: "os", summary: "focused Editor window", ts: 1 });
    const b = defaultActionToken({ kind: "action", tool: "os", summary: "focused Terminal window", ts: 2 });
    expect(a).toBe(b);
    expect(a).toBe("os:focused");
  });
});
