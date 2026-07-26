import { describe, expect, it } from "vitest";
import {
  HardwareTrainingBackend,
  MockLocalTrainingBackend,
  TrainingBackendRegistry,
  TrainingBackendUnavailableError,
  buildMovementDataset,
  createDefaultTrainingBackendRegistry,
  evaluateMovementModel,
  inferMovement,
  normalizeObservation,
  tokenizeObservation,
  type MovementDataset,
  type MovementSample,
} from "./backend.js";
import type { ReviewedExportManifest } from "./export-manifest.js";

function manifestWith(replays: ReviewedExportManifest["replays"]): ReviewedExportManifest {
  return {
    version: 1,
    createdAt: "2026-01-01T00:00:00.000Z",
    reviewedBy: "operator",
    purpose: "local movement fine-tuning",
    targetPlatform: "apple-silicon",
    modes: ["sft"],
    rawCaptureIncluded: false,
    promotedSkills: [],
    executableSkills: [],
    executableSkillRuns: [],
    memories: [],
    trajectories: [],
    replays,
  };
}

describe("buildMovementDataset", () => {
  it("pairs each action with the most recent preceding observation in its trajectory", () => {
    const dataset = buildMovementDataset(
      manifestWith([
        {
          sessionId: "s1",
          trajectoryIds: ["t1"],
          eventCount: 4,
          events: [
            { kind: "observation", ts: 1, trajectoryId: "t1", source: "screen", summary: "Save button visible top right" },
            { kind: "action", ts: 2, trajectoryId: "t1", tool: "mouse", summary: "click 900 40" },
            { kind: "observation", ts: 3, trajectoryId: "t1", source: "screen", summary: "Dialog asks for file name" },
            { kind: "action", ts: 4, trajectoryId: "t1", tool: "keyboard", summary: "type report" },
          ],
        },
      ]),
    );

    expect(dataset.samples).toEqual<MovementSample[]>([
      { trajectoryId: "t1", observation: "Save button visible top right", source: "screen", tool: "mouse", action: "click 900 40" },
      { trajectoryId: "t1", observation: "Dialog asks for file name", source: "screen", tool: "keyboard", action: "type report" },
    ]);
  });

  it("falls back to the last transcript message when no observation precedes the action", () => {
    const dataset = buildMovementDataset(
      manifestWith([
        {
          sessionId: "s1",
          trajectoryIds: ["t1"],
          eventCount: 2,
          events: [
            { kind: "transcript", ts: 1, messageId: "m1", role: "user", content: "open the settings menu" },
            { kind: "action", ts: 2, trajectoryId: "t1", tool: "mouse", summary: "click settings" },
          ],
        },
      ]),
    );

    expect(dataset.samples).toEqual<MovementSample[]>([
      { trajectoryId: "t1", observation: "open the settings menu", tool: "mouse", action: "click settings" },
    ]);
  });
});

describe("normalizeObservation / tokenizeObservation", () => {
  it("normalizes case, punctuation, and whitespace deterministically", () => {
    expect(normalizeObservation("  Save Button, top-right! ")).toBe("save button top right");
    expect(tokenizeObservation("Save the SAVE button.")).toEqual(["button", "save", "the"]);
  });
});

describe("MockLocalTrainingBackend", () => {
  const dataset: MovementDataset = {
    version: 1,
    samples: [
      { trajectoryId: "t1", observation: "Save button visible top right", source: "screen", tool: "mouse", action: "click" },
      { trajectoryId: "t1", observation: "Save button visible top right", source: "screen", tool: "mouse", action: "click" },
      { trajectoryId: "t1", observation: "Save button visible top right", source: "screen", tool: "keyboard", action: "ctrl+s" },
      { trajectoryId: "t2", observation: "Dialog asks for file name", source: "screen", tool: "keyboard", action: "type report" },
    ],
  };

  it("learns a deterministic policy table keyed by normalized observation, resolving ties by support", async () => {
    const model = await new MockLocalTrainingBackend().train(dataset);

    expect(model.backendId).toBe("mock");
    expect(model.sampleCount).toBe(4);
    // Two unique observation keys; sorted lexicographically.
    expect(model.policies.map((p) => p.observationKey)).toEqual([
      "dialog asks for file name",
      "save button visible top right",
    ]);
    // "mouse click" has support 2 vs "keyboard ctrl+s" support 1 → wins.
    const savePolicy = model.policies.find((p) => p.observationKey === "save button visible top right");
    expect(savePolicy).toMatchObject({ tool: "mouse", action: "click", support: 2 });
  });

  it("is fully deterministic across repeated training runs", async () => {
    const a = await new MockLocalTrainingBackend().train(dataset);
    const b = await new MockLocalTrainingBackend().train(dataset);
    expect(a).toEqual(b);
  });

  it("infers an exact recorded movement with full confidence", async () => {
    const model = await new MockLocalTrainingBackend().train(dataset);
    const prediction = inferMovement(model, { summary: "Save button visible TOP right" });
    expect(prediction).toEqual({ tool: "mouse", action: "click", confidence: 1, match: "exact" });
  });

  it("generalizes a new-but-related observation to the nearest known movement", async () => {
    const model = await new MockLocalTrainingBackend().train(dataset);
    // Never-seen phrasing, but shares most tokens with the Save observation.
    const prediction = inferMovement(model, { summary: "The save button is visible on the right" });
    expect(prediction.match).toBe("generalized");
    expect(prediction.tool).toBe("mouse");
    expect(prediction.action).toBe("click");
    expect(prediction.neighborObservation).toBe("save button visible top right");
    expect(prediction.confidence).toBeGreaterThan(0.34);
    expect(prediction.confidence).toBeLessThan(1);
  });

  it("returns a 'none' match when nothing is related enough", async () => {
    const model = await new MockLocalTrainingBackend().train(dataset);
    const prediction = inferMovement(model, { summary: "completely unrelated quantum banana" });
    expect(prediction).toEqual({ tool: "", action: "", confidence: 0, match: "none" });
  });
});

describe("evaluateMovementModel", () => {
  it("scores reproduction accuracy and match-kind breakdown", async () => {
    const dataset: MovementDataset = {
      version: 1,
      samples: [
        { trajectoryId: "t1", observation: "click the blue submit button", tool: "mouse", action: "click submit" },
        { trajectoryId: "t2", observation: "focus the search field", tool: "mouse", action: "click search" },
      ],
    };
    const model = await new MockLocalTrainingBackend().train(dataset);

    const held: MovementSample[] = [
      // exact
      { trajectoryId: "t1", observation: "click the blue submit button", tool: "mouse", action: "click submit" },
      // generalized (shares tokens with the submit sample)
      { trajectoryId: "t3", observation: "click the submit button now", tool: "mouse", action: "click submit" },
    ];

    const result = evaluateMovementModel(model, held);
    expect(result.total).toBe(2);
    expect(result.exact).toBe(1);
    expect(result.generalized).toBe(1);
    expect(result.none).toBe(0);
    expect(result.correct).toBe(2);
    expect(result.accuracy).toBe(1);
  });

  it("reports zero accuracy for an empty sample set without dividing by zero", async () => {
    const model = await new MockLocalTrainingBackend().train({ version: 1, samples: [] });
    expect(evaluateMovementModel(model, [])).toEqual({
      total: 0,
      exact: 0,
      generalized: 0,
      none: 0,
      correct: 0,
      accuracy: 0,
    });
  });
});

describe("TrainingBackendRegistry", () => {
  it("creates the deterministic mock backend by default", async () => {
    const registry = createDefaultTrainingBackendRegistry();
    expect(registry.list()).toEqual(["axolotl", "mlx", "mock"]);
    const backend = registry.create("mock");
    expect(backend.id).toBe("mock");
    const model = await backend.train({ version: 1, samples: [] });
    expect(model.backendId).toBe("mock");
  });

  it("surfaces on-device backends as unavailable in the cloud, loudly", async () => {
    const registry = createDefaultTrainingBackendRegistry();
    const backend = registry.create("mlx");
    expect(backend).toBeInstanceOf(HardwareTrainingBackend);
    await expect(backend.train({ version: 1, samples: [] })).rejects.toBeInstanceOf(TrainingBackendUnavailableError);
  });

  it("throws a helpful error for an unknown backend", () => {
    const registry = new TrainingBackendRegistry();
    expect(() => registry.create("nope")).toThrowError(/Unknown training backend "nope"/);
  });

  it("supports registering a custom pluggable backend", async () => {
    const registry = new TrainingBackendRegistry().register("custom", () => ({
      id: "custom",
      async train() {
        return { version: 1 as const, backendId: "custom", sampleCount: 0, policies: [] };
      },
    }));
    expect(registry.has("custom")).toBe(true);
    const model = await registry.create("custom").train({ version: 1, samples: [] });
    expect(model.backendId).toBe("custom");
  });
});
