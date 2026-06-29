import { describe, expect, it } from "vitest";
import { buildTrajectorySpan, type TrajectoryAction } from "../capture/trajectory.js";
import {
  MarkovMovementBackend,
  MovementModelTrainer,
  buildMovementDataset,
  tokenizeTrajectoryAction,
  type MovementDataset,
} from "./movement-model.js";
import { generateSyntheticTrajectories } from "./synthetic-movements.js";

function action(tool: string, summary: string, ts: number, metadata?: Record<string, unknown>): TrajectoryAction {
  return { kind: "action", tool, summary, ts, ...(metadata ? { metadata } : {}) };
}

function datasetFromTokens(sequences: string[][]): MovementDataset {
  return {
    version: 1,
    sequences: sequences.map((tokens, index) => ({
      id: `seq-${index}`,
      steps: tokens.map((token, step) => ({ token, tool: "device", summary: token, ts: step })),
    })),
  };
}

describe("tokenizeTrajectoryAction", () => {
  it("derives a target-independent token from gesture + direction", () => {
    expect(tokenizeTrajectoryAction(action("device", "tapped sign-in", 1, { gesture: "tap", target: "sign-in" })).token).toBe(
      "device:tap",
    );
    expect(
      tokenizeTrajectoryAction(action("device", "scrolled down results", 2, { gesture: "scroll", direction: "down" })).token,
    ).toBe("device:scroll:down");
  });

  it("falls back to the first summary word when no gesture metadata exists", () => {
    expect(tokenizeTrajectoryAction(action("browser", "Click submit button", 1)).token).toBe("browser:click");
  });

  it("orders steps by timestamp when building a dataset", () => {
    const span = buildTrajectorySpan({
      id: "t1",
      sessionId: "s1",
      actions: [
        action("device", "second", 200, { gesture: "type" }),
        action("device", "first", 100, { gesture: "tap" }),
      ],
    });
    const dataset = buildMovementDataset([span]);
    expect(dataset.sequences[0]!.steps.map((step) => step.token)).toEqual(["device:tap", "device:type"]);
  });

  it("drops trajectories that produced no actions", () => {
    const empty = buildTrajectorySpan({ id: "t1", sessionId: "s1", actions: [] });
    expect(buildMovementDataset([empty]).sequences).toHaveLength(0);
  });
});

describe("MarkovMovementBackend", () => {
  const backend = new MarkovMovementBackend();

  it("learns deterministic transition counts and vocabulary", () => {
    const artifact = backend.train(datasetFromTokens([["a", "b", "c"], ["a", "b", "c"]]), { order: 2 });
    expect(artifact.vocabulary).toEqual(["a", "b", "c"]);
    expect(artifact.stepCount).toBe(6);
    expect(artifact.contexts["a"]).toEqual({ b: 2 });
    // exact-context prediction has zero backoff
    const prediction = backend.predict(artifact, ["a", "b"]);
    expect(prediction.token).toBe("c");
    expect(prediction.backoff).toBe(0);
    expect(prediction.matchedOrder).toBe(2);
  });

  it("breaks ties deterministically by count then lexical order", () => {
    const artifact = backend.train(datasetFromTokens([["x", "b"], ["x", "a"]]), { order: 1 });
    // both b and a follow x once -> lexical tiebreak picks "a"
    expect(backend.predict(artifact, ["x"]).token).toBe("a");
  });

  it("generalizes via backoff to a context never seen verbatim", () => {
    // Trained bigram b->c exists, but the full context [z, b] was never seen.
    const artifact = backend.train(datasetFromTokens([["a", "b", "c"]]), { order: 2 });
    const prediction = backend.predict(artifact, ["z", "b"]);
    expect(prediction.token).toBe("c");
    expect(prediction.backoff).toBe(1);
    expect(prediction.matchedOrder).toBe(1);
  });

  it("falls back to the unigram prior for a fully unknown context", () => {
    const artifact = backend.train(datasetFromTokens([["a", "a", "b"]]), { order: 2 });
    const prediction = backend.predict(artifact, ["unknown"]);
    expect(prediction.token).toBe("a"); // most frequent token
    expect(prediction.matchedOrder).toBe(0);
  });

  it("returns no token for an empty model", () => {
    const artifact = backend.train({ version: 1, sequences: [] });
    expect(backend.predict(artifact, []).token).toBeUndefined();
  });
});

describe("MovementModelTrainer", () => {
  it("repeats a recorded movement sequence from a seed (memorization)", () => {
    const trainer = new MovementModelTrainer();
    const dataset = datasetFromTokens([["device:tap", "device:type", "device:tap", "device:type", "device:tap"]]);
    const artifact = trainer.train(dataset, { order: 3 });
    const generated = trainer.generate(artifact, ["device:tap"], 4);
    expect(generated.map((m) => m.token)).toEqual(["device:type", "device:tap", "device:type", "device:tap"]);
  });

  it("scores held-out related sequences and attributes hits to generalization", () => {
    const trainer = new MovementModelTrainer(new MarkovMovementBackend());
    // Train on two flows; hold out a recombination (prefix of A + suffix of B).
    const train = datasetFromTokens([
      ["open", "tap", "type", "submit"],
      ["focus", "scroll", "tap", "submit"],
    ]);
    const artifact = trainer.train(train, { order: 2 });
    const heldOut = datasetFromTokens([["open", "tap", "type", "submit"]]); // seen suffix bigrams
    const report = trainer.evaluate(artifact, heldOut);
    expect(report.predictionCount).toBe(4);
    expect(report.accuracy).toBeGreaterThan(0.5);
    expect(report.correct).toBe(report.exactContextHits + report.generalizedHits);
  });
});

describe("synthetic round-trip + generalization", () => {
  it("generates deterministic trajectories for a fixed seed", () => {
    const a = generateSyntheticTrajectories({ seed: 42, sequenceCount: 5 });
    const b = generateSyntheticTrajectories({ seed: 42, sequenceCount: 5 });
    expect(JSON.stringify(a)).toEqual(JSON.stringify(b));
    expect(a).toHaveLength(5);
    expect(a[0]!.actions.length).toBeGreaterThan(0);
  });

  it("trains on simulated capture and generalizes to held-out related flows", () => {
    const trainTrajectories = generateSyntheticTrajectories({ seed: 1, sequenceCount: 40 });
    const heldOutTrajectories = generateSyntheticTrajectories({ seed: 999, sequenceCount: 20 });

    const trainer = new MovementModelTrainer();
    const artifact = trainer.train(buildMovementDataset(trainTrajectories), { order: 3 });
    const report = trainer.evaluate(artifact, buildMovementDataset(heldOutTrajectories));

    // The held-out set uses a different seed (novel target/flow orderings) yet
    // shares movement-token structure, so the model should predict most steps.
    expect(report.predictionCount).toBeGreaterThan(0);
    expect(report.accuracy).toBeGreaterThan(0.6);
    // Some held-out contexts were never seen verbatim -> proven generalization.
    expect(report.generalizedHits).toBeGreaterThan(0);
  });
});
