import { describe, expect, it } from "vitest";
import { buildReplayManifest } from "../capture/replay.js";
import { buildTrajectorySpan } from "../capture/trajectory.js";
import type { TrajectoryAction } from "../capture/trajectory.js";
import {
  buildMovementDataset,
  buildMovementSequencesFromReplays,
  buildMovementSequencesFromTrajectories,
  createMovementModelBackend,
  DeterministicMarkovBackend,
  evaluateMovementModel,
  loadMovementModel,
} from "./movement-model.js";

function action(tool: string, ts: number, summary = tool): TrajectoryAction {
  return { kind: "action", tool, summary, ts };
}

describe("buildMovementDataset", () => {
  it("windows sequences into orders 0..maxOrder and collects a sorted vocabulary", () => {
    const dataset = buildMovementDataset([["a", "b", "c"]], 2);
    expect(dataset.vocabulary).toEqual(["a", "b", "c"]);
    expect(dataset.examples).toEqual([
      { context: [], next: "a" },
      { context: ["a"], next: "b" },
      { context: ["a", "b"], next: "c" },
    ]);
    // sequences are copied, not aliased.
    expect(dataset.sequences).toEqual([["a", "b", "c"]]);
  });

  it("rejects a negative maxOrder", () => {
    expect(() => buildMovementDataset([["a"]], -1)).toThrow(/maxOrder/);
  });
});

describe("DeterministicMarkovBackend", () => {
  it("reproduces a demonstrated movement sequence exactly from its seed", async () => {
    const sequence = ["mouse.move", "mouse.down", "mouse.move", "mouse.up", "key.press"];
    const dataset = buildMovementDataset([sequence], 3);
    const model = await new DeterministicMarkovBackend().train(dataset);

    const rolledOut = model.generate([sequence[0]!], sequence.length - 1);
    expect([sequence[0], ...rolledOut]).toEqual(sequence);
  });

  it("is deterministic: identical datasets yield identical predictions", async () => {
    const dataset = buildMovementDataset([["a", "b", "a", "c"]], 2);
    const backend = new DeterministicMarkovBackend();
    const first = (await backend.train(dataset)).predict(["a"]);
    const second = (await backend.train(dataset)).predict(["a"]);
    expect(first).toEqual(second);
  });

  it("breaks count ties by first-observed order", async () => {
    // After context "x": "b" appears first, then "a" — both once. First wins.
    const dataset = buildMovementDataset([["x", "b"], ["x", "a"]], 1);
    const model = await new DeterministicMarkovBackend().train(dataset);
    const prediction = model.predict(["x"]);
    expect(prediction.token).toBe("b");
    expect(prediction.candidates.map((candidate) => candidate.token)).toEqual(["b", "a"]);
    expect(prediction.confidence).toBeCloseTo(0.5);
  });

  it("prefers the highest-order context and reports its confidence", async () => {
    // "open" -> "type" always; but after "close open" -> "save".
    const dataset = buildMovementDataset(
      [
        ["open", "type"],
        ["open", "type"],
        ["close", "open", "save"],
      ],
      2,
    );
    const model = await new DeterministicMarkovBackend().train(dataset);

    const highOrder = model.predict(["close", "open"]);
    expect(highOrder.token).toBe("save");
    expect(highOrder.order).toBe(2);
    expect(highOrder.backedOff).toBe(false);

    const lowOrder = model.predict(["open"]);
    expect(lowOrder.token).toBe("type");
    expect(lowOrder.order).toBe(1);
  });

  it("generalizes to an unseen-but-related context via stupid backoff", async () => {
    // The bigram "scroll -> click" is well attested; a novel 2-token context
    // ending in "scroll" has never been seen, so the model must back off.
    const dataset = buildMovementDataset(
      [
        ["scroll", "click"],
        ["scroll", "click"],
        ["scroll", "click"],
      ],
      2,
    );
    const model = await new DeterministicMarkovBackend().train(dataset);
    const prediction = model.predict(["hover", "scroll"]);
    expect(prediction.token).toBe("click");
    expect(prediction.order).toBe(1);
    expect(prediction.backedOff).toBe(true);
  });

  it("falls back to the global next-token distribution for an unknown context", async () => {
    const dataset = buildMovementDataset([["a", "b", "b", "c"]], 2);
    const model = await new DeterministicMarkovBackend().train(dataset);
    // "b" is the single most frequent token overall -> unigram fallback.
    const prediction = model.predict(["totally-unseen"]);
    expect(prediction.token).toBe("b");
    expect(prediction.order).toBe(0);
    expect(prediction.backedOff).toBe(true);
  });

  it("returns an empty prediction for an empty model", async () => {
    const model = await new DeterministicMarkovBackend().train(buildMovementDataset([], 2));
    const prediction = model.predict(["a"]);
    expect(prediction.token).toBeUndefined();
    expect(prediction.confidence).toBe(0);
    expect(prediction.candidates).toEqual([]);
  });

  it("stops generating when the model has no continuation", async () => {
    const model = await new DeterministicMarkovBackend().train(buildMovementDataset([], 2));
    expect(model.generate(["seed"], 5)).toEqual([]);
  });
});

describe("serialization", () => {
  it("round-trips a model through serialize/load with identical predictions", async () => {
    const dataset = buildMovementDataset([["a", "b", "c", "a", "b", "d"]], 3);
    const model = await new DeterministicMarkovBackend().train(dataset);
    const restored = loadMovementModel(model.serialize());

    for (const context of [[], ["a"], ["a", "b"], ["c", "a", "b"], ["zzz"]]) {
      expect(restored.predict(context)).toEqual(model.predict(context));
    }
    expect(restored.serialize()).toEqual(model.serialize());
  });

  it("serializes entries in a stable, sorted order", async () => {
    const model = await new DeterministicMarkovBackend().train(buildMovementDataset([["b", "a"], ["a", "b"]], 1));
    const orderOne = model.serialize().orders.find((entry) => entry.order === 1);
    const keys = orderOne?.entries.map((entry) => entry.key) ?? [];
    expect(keys).toEqual([...keys].sort());
  });
});

describe("backend registry", () => {
  it("creates the deterministic backend by default", () => {
    expect(createMovementModelBackend().id).toBe("deterministic-markov");
  });

  it("throws for an unknown backend kind", () => {
    // @ts-expect-error intentionally passing an invalid kind.
    expect(() => createMovementModelBackend("nope")).toThrow(/unknown movement-model backend/);
  });
});

describe("sequence extraction from capture artifacts", () => {
  it("derives token sequences from trajectory actions in timestamp order", () => {
    const trajectory = buildTrajectorySpan({
      id: "t1",
      sessionId: "s1",
      actions: [action("key.press", 30), action("mouse.move", 10), action("mouse.up", 20)],
    });
    expect(buildMovementSequencesFromTrajectories([trajectory])).toEqual([
      ["mouse.move", "mouse.up", "key.press"],
    ]);
  });

  it("derives token sequences from replay-manifest action events", () => {
    const trajectory = buildTrajectorySpan({
      id: "t1",
      sessionId: "s1",
      observations: [{ kind: "observation", source: "screen", summary: "window", ts: 5 }],
      actions: [action("mouse.move", 10), action("mouse.down", 20)],
    });
    const manifest = buildReplayManifest({ sessionId: "s1", transcript: [], trajectories: [trajectory] });
    expect(buildMovementSequencesFromReplays([manifest])).toEqual([["mouse.move", "mouse.down"]]);
  });

  it("honours a custom tokenizer that folds in the summary", () => {
    const trajectory = buildTrajectorySpan({
      id: "t1",
      sessionId: "s1",
      actions: [action("key.press", 10, "Enter")],
    });
    const sequences = buildMovementSequencesFromTrajectories([trajectory], (a) => `${a.tool}:${a.summary}`);
    expect(sequences).toEqual([["key.press:Enter"]]);
  });
});

describe("evaluateMovementModel", () => {
  it("scores next-token accuracy and a generalization rate on held-out windows", async () => {
    // Train on repeated drag gestures.
    const train = buildMovementDataset(
      [
        ["down", "move", "up"],
        ["down", "move", "up"],
      ],
      2,
    );
    const model = await new DeterministicMarkovBackend().train(train);

    const heldOut = [
      { context: ["down"], next: "move" }, // exact bigram match
      { context: ["hover", "down"], next: "move" }, // requires backoff -> generalized
      { context: ["down", "move"], next: "up" }, // exact trigram match
    ];
    const result = evaluateMovementModel(model, heldOut);
    expect(result.total).toBe(3);
    expect(result.correct).toBe(3);
    expect(result.accuracy).toBe(1);
    expect(result.generalizedCorrect).toBe(1);
    expect(result.generalizationRate).toBeCloseTo(1 / 3);
  });

  it("returns zeroed metrics for an empty held-out set", async () => {
    const model = await new DeterministicMarkovBackend().train(buildMovementDataset([["a"]], 1));
    expect(evaluateMovementModel(model, [])).toEqual({
      total: 0,
      correct: 0,
      accuracy: 0,
      generalizedCorrect: 0,
      generalizationRate: 0,
    });
  });
});
