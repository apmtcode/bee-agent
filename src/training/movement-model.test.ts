import { describe, expect, it } from "vitest";
import {
  MarkovMovementBackend,
  createMovementModelBackend,
  evaluateMovementModel,
  movementDatasetFromReplays,
  type MovementDataset,
  type MovementSequence,
} from "./movement-model.js";
import type { ReplayManifest } from "../capture/replay.js";

/** A small, deterministic synthetic dataset: two shared-prefix "deploy" flows. */
const dataset: MovementDataset = {
  version: 1,
  sequences: [
    {
      id: "seq-a",
      steps: [
        { tool: "mouse.move", target: "toolbar" },
        { tool: "mouse.click", target: "deploy-button" },
        { tool: "key.press", target: "enter" },
      ],
    },
    {
      id: "seq-b",
      steps: [
        { tool: "mouse.move", target: "toolbar" },
        { tool: "mouse.click", target: "deploy-button" },
        { tool: "key.press", target: "enter" },
      ],
    },
    {
      id: "seq-c",
      steps: [
        { tool: "mouse.move", target: "sidebar" },
        { tool: "mouse.click", target: "settings" },
      ],
    },
  ],
};

describe("MarkovMovementBackend", () => {
  it("reproduces a memorized movement sequence exactly (objective 2c: repeat)", () => {
    const model = new MarkovMovementBackend().train(dataset, { order: 3 });
    const generated = model.generate();
    // START -> highest-count first step is the toolbar/deploy flow (2 of 3 seqs).
    expect(generated).toEqual([
      { tool: "mouse.move", target: "toolbar" },
      { tool: "mouse.click", target: "deploy-button" },
      { tool: "key.press", target: "enter" },
    ]);
  });

  it("predicts the next step for an exact prefix with full confidence", () => {
    const model = new MarkovMovementBackend().train(dataset, { order: 3 });
    const prediction = model.predictNext([
      { tool: "mouse.move", target: "toolbar" },
      { tool: "mouse.click", target: "deploy-button" },
    ]);
    expect(prediction.step).toEqual({ tool: "key.press", target: "enter" });
    expect(prediction.confidence).toBe(1);
    expect(prediction.generalized).toBe(false);
    expect(prediction.matchedOrder).toBe(3);
  });

  it("generalizes to a novel-but-related prefix via backoff (objective 2d)", () => {
    const model = new MarkovMovementBackend().train(dataset, { order: 3 });
    // This exact 2-step prefix never appears in training, but its suffix
    // ("mouse.click deploy-button") does — the model should back off and still
    // predict the correct continuation.
    const prediction = model.predictNext([
      { tool: "key.press", target: "cmd-k" },
      { tool: "mouse.click", target: "deploy-button" },
    ]);
    expect(prediction.step).toEqual({ tool: "key.press", target: "enter" });
    expect(prediction.generalized).toBe(true);
    expect(prediction.matchedOrder).toBeLessThan(3);
  });

  it("terminates generation with a STOP sentinel and caps at maxSteps", () => {
    const model = new MarkovMovementBackend().train(dataset, { order: 2 });
    const generated = model.generate([{ tool: "mouse.move", target: "sidebar" }], { maxSteps: 10 });
    // Seed is excluded from the returned tail; the settings flow ends after click.
    expect(generated).toEqual([{ tool: "mouse.click", target: "settings" }]);
  });

  it("is deterministic across repeated training and prediction", () => {
    const backend = new MarkovMovementBackend();
    const a = backend.train(dataset, { order: 3 }).generate();
    const b = backend.train(dataset, { order: 3 }).generate();
    expect(a).toEqual(b);
  });

  it("round-trips through serialize/load with identical behavior", () => {
    const backend = new MarkovMovementBackend();
    const model = backend.train(dataset, { order: 3 });
    const restored = backend.load(model.serialize());
    expect(restored.order).toBe(model.order);
    expect(restored.stats).toEqual(model.stats);
    expect(restored.generate()).toEqual(model.generate());
  });

  it("exposes vocabulary/step stats", () => {
    const model = new MarkovMovementBackend().train(dataset, { order: 3 });
    expect(model.stats.sequenceCount).toBe(3);
    expect(model.stats.stepCount).toBe(8);
    // 5 distinct movement tokens: toolbar-move, deploy-click, enter, sidebar-move, settings-click.
    expect(model.stats.vocabularySize).toBe(5);
  });
});

describe("createMovementModelBackend", () => {
  it("returns the markov backend by default", () => {
    expect(createMovementModelBackend().name).toBe("markov");
    expect(createMovementModelBackend("markov").name).toBe("markov");
  });

  it("throws for an unregistered backend kind", () => {
    // @ts-expect-error deliberately passing an invalid kind
    expect(() => createMovementModelBackend("mlx")).toThrow(/Register it in createMovementModelBackend/);
  });
});

describe("movementDatasetFromReplays", () => {
  it("derives one sequence per trajectory from action events only", () => {
    const replays: Pick<ReplayManifest, "trajectoryIds" | "events">[] = [
      {
        trajectoryIds: ["traj-1"],
        events: [
          { kind: "observation", ts: 1, trajectoryId: "traj-1", source: "browser", summary: "opened deploy" },
          { kind: "action", ts: 2, trajectoryId: "traj-1", tool: "browser.click", summary: "deploy" },
          { kind: "transcript", ts: 3, messageId: "m1", role: "assistant", content: "done" },
          { kind: "action", ts: 4, trajectoryId: "traj-1", tool: "key.press", summary: "enter" },
        ],
      },
    ];
    const derived = movementDatasetFromReplays(replays);
    expect(derived.sequences).toEqual([
      {
        id: "traj-1",
        steps: [
          { tool: "browser.click", target: "deploy" },
          { tool: "key.press", target: "enter" },
        ],
      },
    ]);
  });

  it("drops trajectories that contributed no action steps", () => {
    const replays: Pick<ReplayManifest, "trajectoryIds" | "events">[] = [
      {
        trajectoryIds: ["traj-empty"],
        events: [
          { kind: "observation", ts: 1, trajectoryId: "traj-empty", source: "os", summary: "focus" },
        ],
      },
    ];
    expect(movementDatasetFromReplays(replays).sequences).toEqual([]);
  });
});

describe("evaluateMovementModel", () => {
  it("scores near-perfect next-step accuracy on the training distribution", () => {
    const model = new MarkovMovementBackend().train(dataset, { order: 3 });
    const report = evaluateMovementModel(model, dataset.sequences);
    expect(report.stepCount).toBe(8);
    // 7/8: every step is reproduced except seq-c's cold-start opener, which is a
    // minority first move (1 of 3 sequences) and legitimately loses the argmax
    // from the empty START context to the majority "toolbar" branch.
    expect(report.correct).toBe(7);
    expect(report.accuracy).toBeCloseTo(0.875, 5);
  });

  it("measures generalization on a held-out related sequence", () => {
    const model = new MarkovMovementBackend().train(dataset, { order: 3 });
    const heldOut: MovementSequence[] = [
      {
        id: "held-1",
        steps: [
          // Novel opening (cold command-palette shortcut), then the memorized
          // toolbar->deploy->enter flow the model should still complete once its
          // context window overlaps the training data.
          { tool: "key.press", target: "cmd-k" },
          { tool: "mouse.move", target: "toolbar" },
          { tool: "mouse.click", target: "deploy-button" },
          { tool: "key.press", target: "enter" },
        ],
      },
    ];
    const report = evaluateMovementModel(model, heldOut);
    // Once the prefix overlaps the memorized flow, deploy-button and enter are
    // both predicted correctly — generalization from a novel opener.
    expect(report.correct).toBeGreaterThanOrEqual(2);
    expect(report.accuracy).toBeGreaterThanOrEqual(0.5);
    expect(report.perSequence[0]?.sequenceId).toBe("held-1");
  });
});
