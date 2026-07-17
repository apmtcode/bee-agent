import { describe, expect, it } from "vitest";
import {
  MarkovMovementBackend,
  MovementModelInference,
  MOVEMENT_END,
  evaluateMovementModel,
  type MovementSequence,
} from "./movement-model.js";

const fixedNow = () => new Date("2026-07-17T00:00:00.000Z");

function seq(trajectoryId: string, tools: string[]): MovementSequence {
  return { trajectoryId, steps: tools.map((tool, index) => ({ tool, summary: `${tool}#${index}` })) };
}

describe("MarkovMovementBackend", () => {
  it("produces a serializable artifact that round-trips through JSON", async () => {
    const backend = new MarkovMovementBackend();
    const artifact = await backend.train({
      jobId: "job-1",
      mode: "sft",
      order: 2,
      now: fixedNow,
      sequences: [seq("t1", ["a", "b", "c"])],
    });

    expect(artifact.backendId).toBe("markov-ngram");
    expect(artifact.jobId).toBe("job-1");
    expect(artifact.order).toBe(2);
    expect(artifact.vocabulary).toEqual(["a", "b", "c"]);
    expect(artifact.sequenceCount).toBe(1);
    expect(artifact.stepCount).toBe(3);
    expect(artifact.trainedAt).toBe("2026-07-17T00:00:00.000Z");

    const restored = JSON.parse(JSON.stringify(artifact));
    expect(restored).toEqual(artifact);
  });

  it("clamps the order into [1, 8]", async () => {
    const backend = new MarkovMovementBackend();
    const low = await backend.train({ jobId: "j", mode: "sft", order: 0, sequences: [seq("t", ["a"])] });
    const high = await backend.train({ jobId: "j", mode: "sft", order: 99, sequences: [seq("t", ["a"])] });
    expect(low.order).toBe(1);
    expect(high.order).toBe(8);
  });
});

describe("MovementModelInference — repeat", () => {
  it("reproduces a recorded movement sequence from an empty seed", async () => {
    const backend = new MarkovMovementBackend();
    const recorded = ["window.focus", "mouse.move", "mouse.click", "app.launch"];
    const artifact = await backend.train({
      jobId: "job",
      mode: "sft",
      order: 2,
      sequences: [seq("t1", recorded)],
    });

    const inference = new MovementModelInference(artifact);
    expect(inference.generate({ maxSteps: 20 })).toEqual(recorded);
  });

  it("terminates decoding at the END sentinel", async () => {
    const backend = new MarkovMovementBackend();
    const artifact = await backend.train({ jobId: "j", mode: "sft", sequences: [seq("t", ["a", "b"])] });
    const inference = new MovementModelInference(artifact);
    // With context ["a","b"], the only observed continuation is END → no token.
    expect(inference.predictNext(["a", "b"])?.token).toBe(MOVEMENT_END);
    expect(inference.generate({ seed: ["a", "b"] })).toEqual([]);
  });
});

describe("MovementModelInference — generalize via backoff", () => {
  it("predicts the more frequent continuation when contexts collide", async () => {
    const backend = new MarkovMovementBackend();
    const artifact = await backend.train({
      jobId: "job",
      mode: "sft",
      order: 2,
      sequences: [
        seq("t1", ["a", "b", "c"]),
        seq("t2", ["a", "b", "c"]),
        seq("t3", ["a", "b", "d"]),
      ],
    });
    const inference = new MovementModelInference(artifact);
    const prediction = inference.predictNext(["a", "b"]);
    expect(prediction?.token).toBe("c");
    expect(prediction?.order).toBe(2);
    expect(prediction?.probability).toBeCloseTo(2 / 3);
  });

  it("backs off to a shorter history for an unseen context", async () => {
    const backend = new MarkovMovementBackend();
    const artifact = await backend.train({
      jobId: "job",
      mode: "sft",
      order: 2,
      sequences: [
        seq("t1", ["x", "y", "z"]),
        seq("t2", ["p", "y", "z"]),
      ],
    });
    const inference = new MovementModelInference(artifact);
    // "q" was never seen, but bigram context ["y"] → "z" is known.
    const prediction = inference.predictNext(["q", "y"]);
    expect(prediction?.token).toBe("z");
    expect(prediction?.order).toBeLessThan(2);
  });

  it("returns undefined for an empty model", async () => {
    const backend = new MarkovMovementBackend();
    const artifact = await backend.train({ jobId: "j", mode: "sft", sequences: [] });
    const inference = new MovementModelInference(artifact);
    expect(inference.predictNext(["anything"])).toBeUndefined();
    expect(inference.generate()).toEqual([]);
  });
});

describe("evaluateMovementModel", () => {
  it("reports perfect fidelity on the training distribution", async () => {
    const backend = new MarkovMovementBackend();
    const sequences = [seq("t1", ["a", "b", "c"]), seq("t2", ["a", "b", "c"])];
    const artifact = await backend.train({ jobId: "j", mode: "sft", order: 2, sequences });
    const inference = new MovementModelInference(artifact);

    const report = evaluateMovementModel(inference, sequences);
    expect(report.sequences).toBe(2);
    expect(report.steps).toBe(6);
    expect(report.accuracy).toBe(1);
  });

  it("still scores related held-out sequences above zero via generalization", async () => {
    const backend = new MarkovMovementBackend();
    const train = [
      seq("t1", ["open", "type", "save"]),
      seq("t2", ["open", "type", "save"]),
      seq("t3", ["open", "click", "save"]),
    ];
    const artifact = await backend.train({ jobId: "j", mode: "sft", order: 2, sequences: train });
    const inference = new MovementModelInference(artifact);

    // Held-out but related: starts the same, ends the same.
    const heldOut = [seq("h1", ["open", "type", "save"])];
    const report = evaluateMovementModel(inference, heldOut);
    expect(report.accuracy).toBeGreaterThan(0.5);
  });
});
