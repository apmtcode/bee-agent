import { describe, expect, it } from "vitest";
import {
  MarkovMovementBackend,
  buildMovementDataset,
  decodeMovementToken,
  encodeMovementToken,
  evaluateMovementModel,
  generateSyntheticMovementDataset,
  movementStepFromAction,
  type MovementStep,
  type SyntheticMovementWorkflow,
} from "./movement-model.js";

describe("movement token codec", () => {
  it("round-trips a full step", () => {
    const step: MovementStep = { tool: "device", action: "swipe", target: "list", direction: "down" };
    expect(decodeMovementToken(encodeMovementToken(step))).toEqual(step);
  });

  it("round-trips a minimal step and omits empty fields", () => {
    const step: MovementStep = { tool: "os", action: "focus-changed" };
    const token = encodeMovementToken(step);
    expect(token).toBe("os|focus-changed||");
    expect(decodeMovementToken(token)).toEqual(step);
  });

  it("escapes separators embedded in fields", () => {
    const step: MovementStep = { tool: "device", action: "type", target: "a|b" };
    expect(decodeMovementToken(encodeMovementToken(step)).target).toBe("a/b");
  });
});

describe("movementStepFromAction", () => {
  it("prefers gesture metadata", () => {
    expect(
      movementStepFromAction({
        tool: "device",
        summary: "tapped submit-button",
        ts: 1,
        metadata: { gesture: "tap", target: "submit-button" },
      }),
    ).toEqual({ tool: "device", action: "tap", target: "submit-button" });
  });

  it("falls back to the first summary word", () => {
    expect(
      movementStepFromAction({ tool: "browser", summary: "clicked login link", ts: 1 }),
    ).toEqual({ tool: "browser", action: "clicked" });
  });

  it("reads os event metadata", () => {
    expect(
      movementStepFromAction({
        tool: "os",
        summary: "focused Editor",
        ts: 1,
        metadata: { event: "focus-changed", windowTitle: "Editor" },
      }),
    ).toEqual({ tool: "os", action: "focus-changed", target: "Editor" });
  });
});

describe("buildMovementDataset", () => {
  it("sorts actions by ts and drops empty trajectories", () => {
    const dataset = buildMovementDataset([
      {
        id: "t1",
        actions: [
          { tool: "device", summary: "b", ts: 20, metadata: { gesture: "swipe", direction: "up" } },
          { tool: "device", summary: "a", ts: 10, metadata: { gesture: "tap", target: "x" } },
        ],
      },
      { id: "empty", actions: [] },
    ]);
    expect(dataset.sequences).toHaveLength(1);
    expect(dataset.sequences[0].tokens).toEqual(["device|tap|x|", "device|swipe||up"]);
  });
});

// A small synthetic "vocabulary" of workflows used across the model tests.
const OPEN_EDIT_SAVE: SyntheticMovementWorkflow = {
  id: "open-edit-save",
  steps: [
    { tool: "os", action: "focus-changed", target: "editor" },
    { tool: "device", action: "type", target: "body" },
    { tool: "device", action: "shortcut", target: "save" },
  ],
};
const OPEN_EDIT_CLOSE: SyntheticMovementWorkflow = {
  id: "open-edit-close",
  steps: [
    { tool: "os", action: "focus-changed", target: "editor" },
    { tool: "device", action: "type", target: "body" },
    { tool: "device", action: "shortcut", target: "close" },
  ],
};
const BROWSE: SyntheticMovementWorkflow = {
  id: "browse",
  steps: [
    { tool: "browser", action: "navigate", target: "docs" },
    { tool: "browser", action: "scroll", direction: "down" },
    { tool: "browser", action: "click", target: "link" },
  ],
};

describe("MarkovMovementBackend — repeat recorded movements (obj 2c)", () => {
  it("reproduces a recorded sequence exactly from its first movement", async () => {
    const dataset = generateSyntheticMovementDataset({ workflows: [OPEN_EDIT_SAVE] });
    const model = await new MarkovMovementBackend({ maxOrder: 3 }).train(dataset);
    const seed = dataset.sequences[0].tokens.slice(0, 1);
    const generated = model.generate(seed, 5);
    expect([seed[0], ...generated]).toEqual(dataset.sequences[0].tokens);
  });

  it("predicts the next movement with full confidence for a unique context", async () => {
    const dataset = generateSyntheticMovementDataset({ workflows: [OPEN_EDIT_SAVE] });
    const model = await new MarkovMovementBackend().train(dataset);
    const prediction = model.predictNext(["os|focus-changed|editor|"]);
    expect(prediction.token).toBe("device|type|body|");
    expect(prediction.confidence).toBe(1);
  });

  it("scores perfectly on training data of distinguishable workflows", async () => {
    // Two workflows with distinct openings, so every context is unambiguous.
    const dataset = generateSyntheticMovementDataset({
      workflows: [OPEN_EDIT_SAVE, BROWSE],
    });
    const model = await new MarkovMovementBackend({ maxOrder: 3 }).train(dataset);
    const result = evaluateMovementModel(model, dataset.sequences);
    expect(result.nextTokenAccuracy).toBe(1);
    expect(result.exactReplayRate).toBe(1);
    expect(result.sequenceCount).toBe(2);
  });
});

describe("MarkovMovementBackend — generalize via backoff (obj 2d)", () => {
  it("predicts a plausible next movement for an unseen prefix using shorter context", async () => {
    // Train on two workflows that share a common prefix but diverge on the last
    // step. A brand-new first step the model never saw still yields a sensible
    // continuation via backoff to the shared middle transition.
    const training = generateSyntheticMovementDataset({
      workflows: [OPEN_EDIT_SAVE, OPEN_EDIT_CLOSE],
    });
    const model = await new MarkovMovementBackend({ maxOrder: 2 }).train(training);

    // Held-out sequence: a novel opening app the model never saw, then the
    // familiar "type body" movement. The order-2 context (newApp -> type) was
    // never observed, but order-1 (type -> shortcut) was — so backoff predicts.
    const heldOut = [
      "os|focus-changed|newapp|",
      "device|type|body|",
    ];
    const prediction = model.predictNext(heldOut);
    expect(prediction.token).toBeDefined();
    expect(prediction.order).toBeLessThan(model.maxOrder);
    // Both training workflows follow "type body" with a device|shortcut move.
    expect(decodeMovementToken(prediction.token!).action).toBe("shortcut");
  });

  it("reports backoff usage in eval metrics on related held-out data", async () => {
    const training = generateSyntheticMovementDataset({ workflows: [OPEN_EDIT_SAVE] });
    const model = await new MarkovMovementBackend({ maxOrder: 3 }).train(training);
    const heldOut = generateSyntheticMovementDataset({ workflows: [OPEN_EDIT_CLOSE] });
    const result = evaluateMovementModel(model, heldOut.sequences);
    // The shared prefix is reproduced; only the final divergent step misses.
    expect(result.correctSteps).toBeGreaterThan(0);
    expect(result.nextTokenAccuracy).toBeGreaterThan(0);
    expect(result.nextTokenAccuracy).toBeLessThan(1);
  });
});

describe("MarkovMovementModel — determinism & persistence", () => {
  it("produces identical snapshots across independent training runs", async () => {
    const dataset = generateSyntheticMovementDataset({
      workflows: [OPEN_EDIT_SAVE, OPEN_EDIT_CLOSE],
      repetitionsPerWorkflow: 2,
    });
    const backend = new MarkovMovementBackend({ maxOrder: 3 });
    const a = await backend.train(dataset);
    const b = await backend.train(dataset);
    expect(a.snapshot()).toEqual(b.snapshot());
  });

  it("restores from a snapshot with identical predictions", async () => {
    const dataset = generateSyntheticMovementDataset({
      workflows: [OPEN_EDIT_SAVE, OPEN_EDIT_CLOSE],
    });
    const backend = new MarkovMovementBackend();
    const model = await backend.train(dataset);
    const restored = backend.restore(model.snapshot());
    const context = ["os|focus-changed|editor|", "device|type|body|"];
    expect(restored.predictNext(context)).toEqual(model.predictNext(context));
    expect(restored.vocabulary).toEqual(model.vocabulary);
  });

  it("does not loop forever on ambiguous unigram backoff", async () => {
    const dataset = generateSyntheticMovementDataset({
      workflows: [
        { id: "ab", steps: [{ tool: "t", action: "a" }, { tool: "t", action: "b" }] },
        { id: "ba", steps: [{ tool: "t", action: "b" }, { tool: "t", action: "a" }] },
      ],
    });
    const model = await new MarkovMovementBackend({ maxOrder: 1 }).train(dataset);
    const generated = model.generate(["t|a||"], 100);
    expect(generated.length).toBeLessThan(100);
  });
});

describe("evaluateMovementModel", () => {
  it("returns zeroed metrics when there is nothing to evaluate", async () => {
    const model = await new MarkovMovementBackend().train({ version: 1, sequences: [] });
    const result = evaluateMovementModel(model, [{ id: "single", tokens: ["only|move||"] }]);
    expect(result.sequenceCount).toBe(0);
    expect(result.nextTokenAccuracy).toBe(0);
    expect(result.exactReplayRate).toBe(0);
  });
});

describe("generateSyntheticMovementDataset", () => {
  it("labels repetitions deterministically", () => {
    const dataset = generateSyntheticMovementDataset({
      workflows: [OPEN_EDIT_SAVE],
      repetitionsPerWorkflow: 3,
    });
    expect(dataset.sequences.map((s) => s.id)).toEqual([
      "open-edit-save#0",
      "open-edit-save#1",
      "open-edit-save#2",
    ]);
  });
});
