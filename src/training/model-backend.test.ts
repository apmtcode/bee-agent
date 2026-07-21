import { describe, expect, it } from "vitest";
import type { ReviewedExportManifest } from "./export-manifest.js";
import {
  DEFAULT_MOVEMENT_MODEL_ORDER,
  MarkovMovementBackend,
  deriveMovementDataset,
  scoreReplayFidelity,
  type MovementDataset,
} from "./model-backend.js";

function actionEvent(trajectoryId: string, tool: string, ts: number) {
  return { kind: "action" as const, ts, trajectoryId, tool, summary: `${tool} @${ts}` };
}

function manifestWithReplays(
  replays: ReviewedExportManifest["replays"],
): Pick<ReviewedExportManifest, "replays"> {
  return { replays };
}

describe("deriveMovementDataset", () => {
  it("extracts ordered action sequences per trajectory and drops non-actions", () => {
    const manifest = manifestWithReplays([
      {
        sessionId: "s1",
        trajectoryIds: ["t1"],
        eventCount: 4,
        events: [
          { kind: "transcript", ts: 0, messageId: "m0", role: "user", content: "go" },
          actionEvent("t1", "mouse.move", 1),
          { kind: "observation", ts: 2, trajectoryId: "t1", source: "screen", summary: "..." },
          actionEvent("t1", "mouse.click", 3),
          actionEvent("t1", "key.press", 4),
        ],
      },
    ]);

    const dataset = deriveMovementDataset(manifest);
    expect(dataset.sequences).toHaveLength(1);
    expect(dataset.sequences[0]!.tokens).toEqual(["action:mouse.move", "action:mouse.click", "action:key.press"]);
    expect(dataset.vocabulary).toEqual(["action:key.press", "action:mouse.click", "action:mouse.move"]);
  });

  it("splits events from the same replay into separate per-trajectory sequences", () => {
    const manifest = manifestWithReplays([
      {
        sessionId: "s1",
        trajectoryIds: ["t1", "t2"],
        eventCount: 3,
        events: [actionEvent("t1", "a", 1), actionEvent("t2", "b", 2), actionEvent("t1", "c", 3)],
      },
    ]);
    const dataset = deriveMovementDataset(manifest);
    const byTrajectory = Object.fromEntries(dataset.sequences.map((s) => [s.trajectoryId, s.tokens]));
    expect(byTrajectory["t1"]).toEqual(["action:a", "action:c"]);
    expect(byTrajectory["t2"]).toEqual(["action:b"]);
  });
});

function datasetOf(sequences: string[][]): MovementDataset {
  const vocabulary = [...new Set(sequences.flat())].sort();
  return {
    version: 1,
    sequences: sequences.map((tokens, index) => ({
      trajectoryId: `t${index}`,
      sessionId: "s",
      tokens,
    })),
    vocabulary,
  };
}

describe("MarkovMovementBackend", () => {
  it("reproduces a recorded movement sequence exactly (repeat)", async () => {
    const sequence = ["move", "click", "type", "enter", "move", "click"];
    const model = await new MarkovMovementBackend().train(datasetOf([sequence]));

    // Given the recorded seed, autoregressive rollout replays the rest.
    const rollout = model.generate([sequence[0]!], sequence.length - 1);
    expect([sequence[0], ...rollout]).toEqual(sequence);

    const fidelity = scoreReplayFidelity(model, sequence);
    expect(fidelity.accuracy).toBe(1);
  });

  it("is deterministic: identical input yields identical predictions", async () => {
    const dataset = datasetOf([
      ["a", "b", "c"],
      ["a", "b", "d"],
      ["a", "b", "c"],
    ]);
    const first = await new MarkovMovementBackend().train(dataset);
    const second = await new MarkovMovementBackend().train(dataset);
    expect(first.predictNext(["a", "b"])).toEqual(second.predictNext(["a", "b"]));
  });

  it("ranks the most frequent continuation first with correct probabilities", async () => {
    const model = await new MarkovMovementBackend().train(
      datasetOf([
        ["a", "b", "c"],
        ["a", "b", "c"],
        ["a", "b", "d"],
      ]),
    );
    const prediction = model.predictNext(["a", "b"]);
    expect(prediction.token).toBe("c");
    expect(prediction.order).toBe(2);
    expect(prediction.probability).toBeCloseTo(2 / 3);
    expect(prediction.candidates.map((c) => c.token)).toEqual(["c", "d"]);
  });

  it("generalizes to an unseen context by backing off to a seen suffix", async () => {
    // The bigram (x -> y) is never seen, but the suffix "b" reliably precedes "c".
    const model = await new MarkovMovementBackend().train(
      datasetOf([
        ["a", "b", "c"],
        ["z", "b", "c"],
      ]),
    );
    const prediction = model.predictNext(["totally", "new", "b"]);
    expect(prediction.token).toBe("c");
    // Backed off from order 3 down to the seen unigram-suffix context "b" (order 1).
    expect(prediction.order).toBe(1);
  });

  it("falls back to the unigram prior for a wholly unseen context", async () => {
    const model = await new MarkovMovementBackend().train(
      datasetOf([
        ["a", "a", "a", "b"],
      ]),
    );
    const prediction = model.predictNext(["nothing", "matches"]);
    expect(prediction.order).toBe(0);
    expect(prediction.token).toBe("a"); // most frequent token overall
  });

  it("returns an empty prediction for an empty model", async () => {
    const model = await new MarkovMovementBackend().train(datasetOf([]));
    expect(model.predictNext(["x"]).token).toBeUndefined();
    expect(model.generate(["x"], 5)).toEqual([]);
  });

  it("respects a custom order and defaults otherwise", async () => {
    const dataset = datasetOf([["a", "b", "c", "d", "e"]]);
    const defaulted = await new MarkovMovementBackend().train(dataset);
    expect(defaulted.order).toBe(DEFAULT_MOVEMENT_MODEL_ORDER);
    const custom = await new MarkovMovementBackend().train(dataset, { order: 1 });
    expect(custom.order).toBe(1);
  });

  it("round-trips through serialize/restore with identical behavior", async () => {
    const dataset = datasetOf([
      ["open", "select", "copy"],
      ["open", "select", "paste"],
      ["open", "select", "copy"],
    ]);
    const backend = new MarkovMovementBackend();
    const model = await backend.train(dataset);
    const restored = backend.restore(JSON.parse(JSON.stringify(model.serialize())));

    expect(restored.order).toBe(model.order);
    expect(restored.vocabulary()).toEqual(model.vocabulary());
    for (const context of [["open", "select"], ["select"], ["unseen"]]) {
      expect(restored.predictNext(context)).toEqual(model.predictNext(context));
    }
  });
});

describe("scoreReplayFidelity generalization harness", () => {
  it("measures partial accuracy on a held-out but related sequence", async () => {
    // Train on movements that establish move->click->type; hold out a variant.
    const model = await new MarkovMovementBackend().train(
      datasetOf([
        ["move", "click", "type"],
        ["move", "click", "type"],
        ["move", "click", "submit"],
      ]),
    );
    // Held-out target shares structure but diverges at the last step.
    const fidelity = scoreReplayFidelity(model, ["move", "click", "type"]);
    expect(fidelity.total).toBe(2);
    expect(fidelity.matches).toBe(2); // move->click and click->type both learned
    expect(fidelity.accuracy).toBe(1);
  });

  it("reports zero accuracy when nothing is scorable", async () => {
    const model = await new MarkovMovementBackend().train(datasetOf([["a", "b"]]));
    expect(scoreReplayFidelity(model, ["a"]).accuracy).toBe(0);
    expect(scoreReplayFidelity(model, []).accuracy).toBe(0);
  });
});
