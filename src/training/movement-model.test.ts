import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { buildReplayManifest } from "../capture/replay.js";
import { buildTrajectorySpan } from "../capture/trajectory.js";
import {
  MovementModel,
  NGramMovementBackend,
  sequenceFromReplayManifest,
  sequenceFromTrajectory,
  sequencesFromTrajectories,
  type MovementSequence,
} from "./movement-model.js";

function step(tool: string, summary = `${tool}-detail`): { tool: string; summary: string } {
  return { tool, summary };
}

// A small, repeatable "open app -> type -> save" workflow used as the recorded
// movement pattern the model should learn and then reproduce / generalize.
function workflow(app: string): MovementSequence {
  return [
    step("window.focus", `focus:${app}`),
    step("mouse.click", `click:${app}-menu`),
    step("keyboard.type", `type:${app}-body`),
    step("keyboard.shortcut", "shortcut:save"),
  ];
}

describe("NGramMovementBackend", () => {
  it("rejects invalid order", () => {
    expect(() => new NGramMovementBackend(0)).toThrow(/positive integer/);
    expect(() => new NGramMovementBackend(1.5)).toThrow(/positive integer/);
  });

  it("learns transition frequencies and predicts the next tool", () => {
    const model = new MovementModel();
    model.train([workflow("editor"), workflow("editor"), workflow("notes")]);

    const prediction = model.predictNext([step("window.focus"), step("mouse.click")]);
    expect(prediction.step?.tool).toBe("keyboard.type");
    expect(prediction.confidence).toBeGreaterThan(0);
    expect(prediction.distribution[0]?.tool).toBe("keyboard.type");
    // distribution is a normalized probability simplex
    const mass = prediction.distribution.reduce((sum, entry) => sum + entry.probability, 0);
    expect(mass).toBeCloseTo(1, 10);
  });

  it("predicts a representative (most frequent) summary for the tool", () => {
    const model = new MovementModel();
    model.train([
      [step("a", "common"), step("b", "common")],
      [step("a", "common"), step("b", "common")],
      [step("a", "common"), step("b", "rare")],
    ]);
    const prediction = model.predictNext([step("a", "common")]);
    expect(prediction.step?.tool).toBe("b");
    expect(prediction.step?.summary).toBe("common");
  });

  it("backs off to a shorter related context for unseen prefixes (generalization)", () => {
    const model = new MovementModel({ backend: new NGramMovementBackend(2) });
    model.train([workflow("editor"), workflow("editor")]);

    // The 2-gram (keyboard.type -> mouse.click) never occurs in the workflow
    // (type is always followed by a shortcut), so this exact context was never
    // seen. But the trailing tool "mouse.click" reliably precedes
    // "keyboard.type", so back-off to the shorter context still yields the
    // right next movement.
    const prediction = model.predictNext([
      step("keyboard.type", "type:terminal"),
      step("mouse.click", "click:terminal-tab"),
    ]);
    expect(prediction.step?.tool).toBe("keyboard.type");
    expect(prediction.backoffOrder).toBeLessThan(2);
  });

  it("returns an empty prediction before training has any data", () => {
    const model = new MovementModel();
    model.train([]);
    const prediction = model.predictNext([step("a")]);
    expect(prediction.step).toBeNull();
    expect(prediction.confidence).toBe(0);
    expect(prediction.distribution).toEqual([]);
  });

  it("throws when predicting before training", () => {
    const model = new MovementModel();
    expect(() => model.predictNext([step("a")])).toThrow(/not been trained/);
  });
});

describe("MovementModel.generate", () => {
  it("reproduces the recorded movement workflow from a seed and stops naturally", () => {
    const model = new MovementModel();
    model.train([workflow("editor"), workflow("editor"), workflow("editor")]);

    const rollout = model.generate({ seed: [step("window.focus", "focus:editor")], maxSteps: 16 });
    const tools = [step("window.focus"), ...rollout.steps].map((s) => s.tool);
    expect(tools).toEqual([
      "window.focus",
      "mouse.click",
      "keyboard.type",
      "keyboard.shortcut",
    ]);
    expect(rollout.stoppedNaturally).toBe(true);
    expect(rollout.meanConfidence).toBeGreaterThan(0);
  });

  it("honors the maxSteps cap", () => {
    const model = new MovementModel();
    // A cycle a->b->a->b... never predicts END, so generation hits the cap.
    model.train([[step("a"), step("b"), step("a"), step("b"), step("a"), step("b")]]);
    const rollout = model.generate({ seed: [step("a")], maxSteps: 5 });
    expect(rollout.steps).toHaveLength(5);
    expect(rollout.stoppedNaturally).toBe(false);
  });
});

describe("MovementModel.evaluate", () => {
  it("scores high next-tool accuracy on held-out but related trajectories", () => {
    const model = new MovementModel({ backend: new NGramMovementBackend(2) });
    model.train([workflow("editor"), workflow("notes"), workflow("browser")]);

    // Held-out app the model never trained on, but same movement grammar.
    const result = model.evaluate([workflow("spreadsheet")]);
    expect(result.sequences).toBe(1);
    expect(result.predictions).toBe(4);
    expect(result.accuracy).toBeGreaterThanOrEqual(0.75);
    expect(result.perSequence[0]?.accuracy).toBeGreaterThanOrEqual(0.75);
  });

  it("reports NaN meanBackoffOrder when nothing is correct", () => {
    const model = new MovementModel();
    model.train([[step("x"), step("y")]]);
    const result = model.evaluate([[step("p"), step("q")]]);
    expect(result.correct).toBe(0);
    expect(Number.isNaN(result.meanBackoffOrder)).toBe(true);
  });
});

describe("sequence adapters", () => {
  it("builds a time-ordered sequence from a trajectory's actions", () => {
    const trajectory = buildTrajectorySpan({
      id: "t1",
      sessionId: "s1",
      actions: [
        { kind: "action", tool: "keyboard.type", summary: "second", ts: 200 },
        { kind: "action", tool: "mouse.click", summary: "first", ts: 100 },
      ],
    });
    expect(sequenceFromTrajectory(trajectory)).toEqual([
      step("mouse.click", "first"),
      step("keyboard.type", "second"),
    ]);
  });

  it("drops empty trajectories when batching", () => {
    const withActions = buildTrajectorySpan({
      id: "t1",
      sessionId: "s1",
      actions: [{ kind: "action", tool: "a", summary: "x", ts: 1 }],
    });
    const empty = buildTrajectorySpan({ id: "t2", sessionId: "s1" });
    expect(sequencesFromTrajectories([withActions, empty])).toHaveLength(1);
  });

  it("extracts the action timeline from a replay manifest", () => {
    const trajectory = buildTrajectorySpan({
      id: "t1",
      sessionId: "s1",
      observations: [{ kind: "observation", source: "screen", summary: "ignored", ts: 50 }],
      actions: [
        { kind: "action", tool: "mouse.click", summary: "first", ts: 100 },
        { kind: "action", tool: "keyboard.type", summary: "second", ts: 200 },
      ],
    });
    const manifest = buildReplayManifest({ sessionId: "s1", transcript: [], trajectories: [trajectory] });
    expect(sequenceFromReplayManifest(manifest)).toEqual([
      step("mouse.click", "first"),
      step("keyboard.type", "second"),
    ]);
  });
});

describe("MovementModel persistence", () => {
  let dir: string;
  beforeEach(async () => {
    dir = await fs.mkdtemp(path.join(os.tmpdir(), "movement-model-"));
  });
  afterEach(async () => {
    await fs.rm(dir, { recursive: true, force: true });
  });

  it("round-trips a trained model through save/load with identical predictions", async () => {
    const original = new MovementModel();
    original.train([workflow("editor"), workflow("notes")]);
    const file = path.join(dir, "model.json");
    await original.save(file);

    const restored = new MovementModel();
    await restored.load(file);
    expect(restored.trained).toBe(true);

    const context = [step("window.focus"), step("mouse.click")];
    expect(restored.predictNext(context)).toEqual(original.predictNext(context));
  });

  it("rejects loading a model for a different backend", async () => {
    const file = path.join(dir, "model.json");
    await fs.writeFile(file, JSON.stringify({ backend: "transformer", state: {} }));
    const model = new MovementModel();
    await expect(model.load(file)).rejects.toThrow(/no saved ngram movement model/);
  });
});
