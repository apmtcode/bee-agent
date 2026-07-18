import { describe, expect, it } from "vitest";
import { NgramMovementBackend } from "./ngram-backend.js";
import type { MovementDataset, MovementEvent } from "./movement-model.js";
import { movementToken } from "./movement-model.js";
import { generateSyntheticMovementDataset } from "./synthetic-movements.js";
import { evaluateMovementModel } from "./movement-eval.js";

function seq(id: string, events: MovementEvent[]): MovementDataset["sequences"][number] {
  return { id, events };
}

const login: MovementEvent[] = [
  { ts: 1, kind: "focus", target: "login" },
  { ts: 2, kind: "click", target: "user" },
  { ts: 3, kind: "type", target: "user", value: "a" },
  { ts: 4, kind: "click", target: "submit" },
];

describe("NgramMovementBackend training + inference", () => {
  it("repeats a memorised sequence exactly from its seed", () => {
    const backend = new NgramMovementBackend();
    const model = backend.train({ version: 1, sequences: [seq("a", login), seq("b", login)] }, { order: 3 });

    const generated = model.generate([login[0]!], { maxSteps: 10 });
    expect(generated.map(movementToken)).toEqual(login.slice(1).map(movementToken));
  });

  it("predicts the correct next movement given a prefix", () => {
    const backend = new NgramMovementBackend();
    const model = backend.train({ version: 1, sequences: [seq("a", login)] }, { order: 3 });

    const prediction = model.predictNext(login.slice(0, 2));
    expect(prediction?.token).toBe(movementToken(login[2]!));
    expect(prediction?.probability).toBeGreaterThan(0);
    expect(prediction?.contextOrderUsed).toBeGreaterThan(0);
  });

  it("generalises to an unseen prefix via backoff instead of failing", () => {
    const backend = new NgramMovementBackend();
    // Two flows share the "click submit -> END" ending after distinct starts.
    const flowA: MovementEvent[] = [
      { ts: 1, kind: "focus", target: "a" },
      { ts: 2, kind: "click", target: "submit" },
    ];
    const flowB: MovementEvent[] = [
      { ts: 1, kind: "focus", target: "b" },
      { ts: 2, kind: "click", target: "submit" },
    ];
    const model = backend.train({ version: 1, sequences: [seq("a", flowA), seq("b", flowB)] }, { order: 2 });

    // Novel context: focus target "c" was never seen — backoff should still fire.
    const prediction = model.predictNext([{ ts: 1, kind: "focus", target: "c" }]);
    expect(prediction).toBeDefined();
    expect(prediction!.contextOrderUsed).toBeLessThan(2);
  });

  it("is deterministic across repeated predictions (stable tie-break)", () => {
    const backend = new NgramMovementBackend();
    const model = backend.train({ version: 1, sequences: [seq("a", login), seq("b", login)] }, { order: 2 });
    const first = model.predictNext(login.slice(0, 1));
    const second = model.predictNext(login.slice(0, 1));
    expect(first).toEqual(second);
  });
});

describe("NgramMovementModel serialization", () => {
  it("round-trips through an artifact and reproduces predictions", () => {
    const backend = new NgramMovementBackend({ now: () => new Date("2026-07-18T00:00:00.000Z") });
    const model = backend.train({ version: 1, sequences: [seq("a", login)] }, { order: 3 });
    const artifact = model.toArtifact();

    expect(artifact.backend).toBe("ngram-backoff");
    expect(artifact.trainedAt).toBe("2026-07-18T00:00:00.000Z");
    expect(artifact.eventCount).toBe(login.length);
    expect(artifact.vocabulary).toContain(movementToken(login[0]!));

    const reloaded = backend.load(artifact);
    expect(reloaded.predictNext(login.slice(0, 2))).toEqual(model.predictNext(login.slice(0, 2)));
    expect(reloaded.generate([login[0]!]).map(movementToken)).toEqual(
      model.generate([login[0]!]).map(movementToken),
    );
  });

  it("rejects an artifact from a foreign backend", () => {
    const backend = new NgramMovementBackend();
    const artifact = backend.train({ version: 1, sequences: [seq("a", login)] }).toArtifact();
    expect(() => backend.load({ ...artifact, backend: "some-other-backend" })).toThrow();
  });
});

describe("end-to-end synthetic train + eval", () => {
  it("achieves high replay fidelity on held-in synthetic data", () => {
    const dataset = generateSyntheticMovementDataset({ seed: 7, sequenceCount: 24 });
    const backend = new NgramMovementBackend();
    const model = backend.train(dataset, { order: 4 });

    const result = evaluateMovementModel(model, dataset.sequences);
    expect(result.nextTokenAccuracy).toBeGreaterThan(0.9);
    expect(result.replayFidelity).toBe(1);
    expect(result.averageTokenOverlap).toBeGreaterThan(0.9);
  });

  it("generalises to held-out sequences of the same programs", () => {
    const train = generateSyntheticMovementDataset({ seed: 1, sequenceCount: 30 });
    const heldOut = generateSyntheticMovementDataset({ seed: 999, sequenceCount: 10 });
    const backend = new NgramMovementBackend();
    const model = backend.train(train, { order: 4 });

    const result = evaluateMovementModel(model, heldOut.sequences);
    // Different sampling/timestamps, same underlying programs → strong transfer.
    expect(result.averageTokenOverlap).toBeGreaterThan(0.8);
    expect(result.nextTokenAccuracy).toBeGreaterThan(0.8);
  });
});
