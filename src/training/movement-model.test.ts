import { describe, expect, it } from "vitest";
import type { TrajectoryAction, TrajectorySpan } from "../capture/trajectory.js";
import {
  MOVEMENT_END_TOKEN,
  MOVEMENT_START_TOKEN,
  NgramMovementBackend,
  buildMovementDataset,
  buildMovementSequences,
  buildMovementSequencesFromReplay,
  createMovementBackend,
  deriveMovementStep,
  encodeMovementToken,
  evaluateMovementModel,
  generateMovement,
  generateSyntheticMovementSequences,
  trainMovementModel,
} from "./movement-model.js";

function action(tool: string, ts: number, metadata: Record<string, unknown>, summary = "did thing"): TrajectoryAction {
  return { kind: "action", tool, summary, ts, metadata };
}

function trajectory(id: string, actions: TrajectoryAction[]): Pick<TrajectorySpan, "id" | "actions"> {
  return { id, actions };
}

describe("deriveMovementStep", () => {
  it("normalizes device gesture metadata into a canonical token", () => {
    const step = deriveMovementStep(action("device", 1, { gesture: "tap", target: "Send Button", direction: "up" }));
    expect(step.tool).toBe("device");
    expect(step.action).toBe("tap");
    expect(step.target).toBe("send-button");
    expect(step.direction).toBe("up");
    expect(step.token).toBe("device.tap@send-button^up");
  });

  it("reads the browser 'action' metadata key", () => {
    const step = deriveMovementStep(action("browser", 1, { action: "click", target: "results" }));
    expect(step.token).toBe("browser.click@results");
  });

  it("falls back to the summary's first word when metadata is absent", () => {
    const step = deriveMovementStep({ tool: "os", summary: "focused editor window" });
    expect(step.action).toBe("focused");
    expect(step.token).toBe("os.focused");
  });
});

describe("encodeMovementToken", () => {
  it("is stable and target/direction sensitive", () => {
    expect(encodeMovementToken({ tool: "device", action: "tap" })).toBe("device.tap");
    expect(encodeMovementToken({ tool: "device", action: "tap", target: "a" })).toBe("device.tap@a");
    expect(encodeMovementToken({ tool: "device", action: "swipe", direction: "left" })).toBe("device.swipe^left");
  });
});

describe("dataset construction", () => {
  it("orders steps by timestamp and builds a sorted vocabulary", () => {
    const sequences = buildMovementSequences([
      trajectory("t1", [
        action("device", 30, { gesture: "send", target: "x" }),
        action("device", 10, { gesture: "tap", target: "x" }),
        action("device", 20, { gesture: "type", target: "x" }),
      ]),
    ]);
    expect(sequences[0].steps.map((s) => s.action)).toEqual(["tap", "type", "send"]);

    const dataset = buildMovementDataset(sequences);
    expect(dataset.vocabulary).toEqual([...dataset.vocabulary].sort());
    expect(dataset.vocabulary).toContain("device.tap@x");
  });

  it("groups replay action events per trajectory", () => {
    const sequences = buildMovementSequencesFromReplay([
      { kind: "transcript", ts: 0, messageId: "m", role: "user", content: "hi" },
      { kind: "action", ts: 2, trajectoryId: "t1", tool: "device", summary: "tapped x" },
      { kind: "action", ts: 1, trajectoryId: "t1", tool: "device", summary: "focused x" },
      { kind: "observation", ts: 3, trajectoryId: "t1", source: "device", summary: "seen" },
    ]);
    expect(sequences).toHaveLength(1);
    expect(sequences[0].steps.map((s) => s.action)).toEqual(["focused", "tapped"]);
  });
});

describe("NgramMovementBackend training + replay", () => {
  it("repeats a recorded movement verbatim from the start", async () => {
    const trajectories = [
      trajectory("t1", [
        action("os", 1, { action: "focus", target: "editor" }),
        action("browser", 2, { action: "click", target: "search" }),
        action("browser", 3, { action: "type", target: "search" }),
        action("browser", 4, { action: "submit", target: "search" }),
      ]),
    ];
    const { model } = await trainMovementModel(trajectories, { config: { order: 3 } });
    const backend = new NgramMovementBackend();
    const rollout = generateMovement(backend, model, [], 16);
    expect(rollout.map((step) => step.token)).toEqual([
      "os.focus@editor",
      "browser.click@search",
      "browser.type@search",
      "browser.submit@search",
    ]);
  });

  it("predicts the trained continuation with full confidence and no backoff", async () => {
    const { model } = await trainMovementModel(
      [trajectory("t1", [action("device", 1, { gesture: "tap", target: "a" }), action("device", 2, { gesture: "tap", target: "b" })])],
      { config: { order: 2 } },
    );
    const backend = new NgramMovementBackend();
    const prediction = backend.predict(model, [MOVEMENT_START_TOKEN, "device.tap@a"]);
    expect(prediction.token).toBe("device.tap@b");
    expect(prediction.backoffOrder).toBe(2);
    expect(prediction.confidence).toBeCloseTo(1, 5);
  });

  it("emits <end> after the final recorded step", async () => {
    const { model } = await trainMovementModel(
      [trajectory("t1", [action("device", 1, { gesture: "tap", target: "only" })])],
      { config: { order: 1 } },
    );
    const backend = new NgramMovementBackend();
    const prediction = backend.predict(model, [MOVEMENT_START_TOKEN, "device.tap@only"]);
    expect(prediction.token).toBe(MOVEMENT_END_TOKEN);
  });

  it("backs off to a shorter context for unseen prefixes (generalization)", async () => {
    // Two flows share the tail "type -> submit"; the model must predict submit
    // after type even from a start context it never saw paired with that type.
    const { model } = await trainMovementModel(
      [
        trajectory("a", [
          action("browser", 1, { action: "click", target: "login" }),
          action("browser", 2, { action: "type", target: "field" }),
          action("browser", 3, { action: "submit", target: "form" }),
        ]),
        trajectory("b", [
          action("browser", 1, { action: "navigate", target: "home" }),
          action("browser", 2, { action: "type", target: "field" }),
          action("browser", 3, { action: "submit", target: "form" }),
        ]),
      ],
      { config: { order: 3 } },
    );
    const backend = new NgramMovementBackend();
    // Unseen 3-gram context, but "type@field -> submit@form" generalizes via backoff.
    const prediction = backend.predict(model, ["browser.scroll@page", "browser.type@field"]);
    expect(prediction.token).toBe("browser.submit@form");
    expect(prediction.backoffOrder).toBeLessThan(2);
  });

  it("returns <end> with zero confidence for an empty model", async () => {
    const backend = new NgramMovementBackend();
    const model = await backend.train(buildMovementDataset([]), { order: 2 });
    const prediction = backend.predict(model, ["anything"]);
    expect(prediction.token).toBe(MOVEMENT_END_TOKEN);
    expect(prediction.confidence).toBe(0);
    expect(prediction.distribution).toHaveLength(0);
  });
});

describe("evaluateMovementModel", () => {
  it("achieves perfect in-sample accuracy on deterministic training data", async () => {
    const trajectories = [
      trajectory("t1", [
        action("os", 1, { action: "focus", target: "editor" }),
        action("browser", 2, { action: "click", target: "search" }),
        action("browser", 3, { action: "submit", target: "search" }),
      ]),
    ];
    const { model, dataset, trainEval } = await trainMovementModel(trajectories, { config: { order: 3 } });
    expect(trainEval.accuracy).toBe(1);
    expect(trainEval.predictions).toBeGreaterThan(0);

    const backend = createMovementBackend();
    const reeval = evaluateMovementModel(backend, model, dataset.sequences);
    expect(reeval.correct).toBe(reeval.predictions);
  });

  it("generalizes above chance to held-out sequences from the same patterns", async () => {
    const train = generateSyntheticMovementSequences({ seed: 7, sequenceCount: 40 });
    const heldOut = generateSyntheticMovementSequences({ seed: 99, sequenceCount: 20 });
    const backend = createMovementBackend();
    const dataset = buildMovementDataset(train);
    const model = await backend.train(dataset, { order: 2 });
    const evalResult = evaluateMovementModel(backend, model, heldOut);
    // Held-out data shares the generator's patterns, so next-step accuracy is high
    // despite the sequences never appearing in training.
    expect(evalResult.accuracy).toBeGreaterThan(0.5);
    expect(evalResult.predictions).toBeGreaterThan(0);
  });
});

describe("generateSyntheticMovementSequences", () => {
  it("is deterministic for a fixed seed and varies with the seed", () => {
    const a = generateSyntheticMovementSequences({ seed: 3, sequenceCount: 5 });
    const b = generateSyntheticMovementSequences({ seed: 3, sequenceCount: 5 });
    const c = generateSyntheticMovementSequences({ seed: 4, sequenceCount: 5 });
    expect(a).toEqual(b);
    expect(JSON.stringify(a)).not.toEqual(JSON.stringify(c));
  });

  it("round-trips through dataset build without losing tokens", () => {
    const sequences = generateSyntheticMovementSequences({ seed: 1, sequenceCount: 8 });
    const dataset = buildMovementDataset(sequences);
    const seen = new Set(dataset.sequences.flatMap((s) => s.steps.map((step) => step.token)));
    for (const token of dataset.vocabulary) {
      expect(seen.has(token)).toBe(true);
    }
  });
});

describe("createMovementBackend", () => {
  it("returns the n-gram backend and round-trips a serialized model", async () => {
    const backend = createMovementBackend("ngram-movement");
    expect(backend.name).toBe("ngram-movement");
    const { model } = await trainMovementModel(
      [trajectory("t1", [action("device", 1, { gesture: "tap", target: "x" })])],
      { backend, config: { order: 2 } },
    );
    const restored = JSON.parse(JSON.stringify(model));
    const prediction = backend.predict(restored, [MOVEMENT_START_TOKEN]);
    expect(prediction.token).toBe("device.tap@x");
  });
});
