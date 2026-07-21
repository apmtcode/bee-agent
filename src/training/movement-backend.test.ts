import { describe, expect, it } from "vitest";
import type { ReviewedExportManifest } from "./export-manifest.js";
import {
  buildMovementDataset,
  MockMovementPolicy,
  MockMovementTrainingBackend,
  MovementBackendRegistry,
  type MovementDataset,
} from "./movement-backend.js";

function exportWithReplays(replays: ReviewedExportManifest["replays"]): ReviewedExportManifest {
  return {
    version: 1,
    createdAt: "2026-07-21T00:00:00.000Z",
    reviewedBy: "reviewer",
    purpose: "test",
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

/** A synthetic "open app → type → save" movement trajectory. */
function openTypeSaveReplay(trajectoryId: string, sessionId: string): ReviewedExportManifest["replays"][number] {
  return {
    sessionId,
    trajectoryIds: [trajectoryId],
    eventCount: 6,
    events: [
      { kind: "observation", ts: 1, trajectoryId, source: "screen", summary: "editor window is focused" },
      { kind: "action", ts: 2, trajectoryId, tool: "mouse.click", summary: "click the file menu" },
      { kind: "observation", ts: 3, trajectoryId, source: "screen", summary: "menu is open" },
      { kind: "action", ts: 4, trajectoryId, tool: "keyboard.type", summary: "type document body" },
      { kind: "action", ts: 5, trajectoryId, tool: "keyboard.shortcut", summary: "press save shortcut" },
    ],
  };
}

describe("buildMovementDataset", () => {
  it("extracts ordered action sequences and observation links per trajectory", () => {
    const manifest = exportWithReplays([openTypeSaveReplay("traj-1", "sess-1")]);
    const dataset = buildMovementDataset(manifest);

    expect(dataset.sequences).toHaveLength(1);
    expect(dataset.sequences[0]!.trajectoryId).toBe("traj-1");
    expect(dataset.sequences[0]!.steps.map((step) => step.tool)).toEqual([
      "mouse.click",
      "keyboard.type",
      "keyboard.shortcut",
    ]);

    // Each action pairs with its most recent preceding observation.
    expect(dataset.observationLinks).toEqual([
      { observation: "editor window is focused", tool: "mouse.click", summary: "click the file menu" },
      { observation: "menu is open", tool: "keyboard.type", summary: "type document body" },
      { observation: "menu is open", tool: "keyboard.shortcut", summary: "press save shortcut" },
    ]);
  });

  it("splits a multi-trajectory replay into separate sequences", () => {
    const manifest = exportWithReplays([
      {
        sessionId: "sess",
        trajectoryIds: ["a", "b"],
        eventCount: 4,
        events: [
          { kind: "action", ts: 1, trajectoryId: "a", tool: "mouse.move", summary: "move to a" },
          { kind: "action", ts: 2, trajectoryId: "b", tool: "mouse.move", summary: "move to b" },
          { kind: "action", ts: 3, trajectoryId: "a", tool: "mouse.click", summary: "click a" },
        ],
      },
    ]);
    const dataset = buildMovementDataset(manifest);
    expect(dataset.sequences.map((sequence) => sequence.trajectoryId).sort()).toEqual(["a", "b"]);
    const seqA = dataset.sequences.find((sequence) => sequence.trajectoryId === "a");
    expect(seqA!.steps.map((step) => step.tool)).toEqual(["mouse.move", "mouse.click"]);
  });
});

describe("MockMovementTrainingBackend", () => {
  it("replays a recorded trajectory exactly", async () => {
    const dataset = buildMovementDataset(exportWithReplays([openTypeSaveReplay("traj-1", "sess-1")]));
    const policy = await new MockMovementTrainingBackend().train({ jobId: "job-1", dataset });

    expect(policy.trajectoryIds()).toEqual(["traj-1"]);
    const replayed = policy.replay("traj-1");
    expect(replayed.map((step) => step.tool)).toEqual(["mouse.click", "keyboard.type", "keyboard.shortcut"]);
    expect(replayed.map((step) => step.summary)).toEqual([
      "click the file menu",
      "type document body",
      "press save shortcut",
    ]);
  });

  it("throws for an unknown trajectory", async () => {
    const dataset = buildMovementDataset(exportWithReplays([openTypeSaveReplay("traj-1", "sess-1")]));
    const policy = await new MockMovementTrainingBackend().train({ jobId: "job-1", dataset });
    expect(() => policy.replay("nope")).toThrow(/Unknown trajectory/);
  });

  it("generates the learned sequence from a cold start", async () => {
    const dataset = buildMovementDataset(exportWithReplays([openTypeSaveReplay("traj-1", "sess-1")]));
    const policy = await new MockMovementTrainingBackend().train({ jobId: "job-1", dataset });

    const generated = policy.generate({}, { maxSteps: 10 });
    // With a single deterministic trajectory, rollout reproduces the sequence
    // then halts at the learned END transition.
    expect(generated.map((step) => step.tool)).toEqual(["mouse.click", "keyboard.type", "keyboard.shortcut"]);
  });

  it("continues a partial sequence via the transition table", async () => {
    const dataset = buildMovementDataset(exportWithReplays([openTypeSaveReplay("traj-1", "sess-1")]));
    const policy = await new MockMovementTrainingBackend().train({ jobId: "job-1", dataset });

    const prediction = policy.predictNext({ recentTools: ["mouse.click"] });
    expect(prediction.tool).toBe("keyboard.type");
    expect(prediction.source).toBe("transition");
    expect(prediction.confidence).toBeGreaterThan(0);
  });

  it("generalizes to a new-but-related observation via nearest match", async () => {
    // Train on two related trajectories so the observation index has coverage.
    const dataset = buildMovementDataset(
      exportWithReplays([openTypeSaveReplay("traj-1", "sess-1"), openTypeSaveReplay("traj-2", "sess-2")]),
    );
    const policy = await new MockMovementTrainingBackend().train({ jobId: "job-1", dataset });

    // A state the model never saw verbatim but that overlaps "menu is open".
    const prediction = policy.predictNext({ observation: "the menu is now open and visible" });
    expect(prediction.source).toBe("nearest-observation");
    expect(["keyboard.type", "keyboard.shortcut"]).toContain(prediction.tool);
    expect(prediction.confidence).toBeGreaterThan(0);
  });

  it("prefers the majority next-tool when transitions diverge", async () => {
    // From "start" the model sees clickA twice and moveB once → clickA wins.
    const dataset: MovementDataset = {
      sequences: [
        { trajectoryId: "t1", steps: [step("t1", "clickA", 0), step("t1", "done", 1)] },
        { trajectoryId: "t2", steps: [step("t2", "clickA", 0), step("t2", "done", 1)] },
        { trajectoryId: "t3", steps: [step("t3", "moveB", 0), step("t3", "done", 1)] },
      ],
      observationLinks: [],
    };
    const policy = await new MockMovementTrainingBackend().train({ jobId: "job-x", dataset });
    const prediction = policy.predictNext({});
    expect(prediction.tool).toBe("clickA");
    expect(prediction.confidence).toBeCloseTo(2 / 3, 2);
  });

  it("is deterministic across repeated training runs", async () => {
    const dataset = buildMovementDataset(
      exportWithReplays([openTypeSaveReplay("traj-1", "sess-1"), openTypeSaveReplay("traj-2", "sess-2")]),
    );
    const backend = new MockMovementTrainingBackend();
    const a = (await backend.train({ jobId: "job", dataset })).serialize();
    const b = (await backend.train({ jobId: "job", dataset })).serialize();
    expect(a).toEqual(b);
  });

  it("round-trips through serialization", async () => {
    const dataset = buildMovementDataset(exportWithReplays([openTypeSaveReplay("traj-1", "sess-1")]));
    const policy = await new MockMovementTrainingBackend().train({ jobId: "job-1", dataset });
    const restored = MockMovementPolicy.fromSerialized(policy.serialize());

    expect(restored.replay("traj-1")).toEqual(policy.replay("traj-1"));
    expect(restored.predictNext({ recentTools: ["mouse.click"] })).toEqual(
      policy.predictNext({ recentTools: ["mouse.click"] }),
    );
    // Mutating the restored snapshot must not affect the original.
    const snapshot = policy.serialize();
    snapshot.sequences[0]!.steps[0]!.tool = "mutated";
    expect(policy.replay("traj-1")[0]!.tool).toBe("mouse.click");
  });
});

describe("MovementBackendRegistry", () => {
  it("registers the mock backend by default and resolves it", () => {
    const registry = new MovementBackendRegistry();
    expect(registry.ids()).toContain("mock-ngram");
    expect(registry.require("mock-ngram")).toBeInstanceOf(MockMovementTrainingBackend);
  });

  it("throws for an unregistered backend id", () => {
    const registry = new MovementBackendRegistry();
    expect(() => registry.require("does-not-exist")).toThrow(/No movement backend/);
    expect(registry.get("does-not-exist")).toBeUndefined();
  });

  it("accepts additional backends", () => {
    const registry = new MovementBackendRegistry([]);
    expect(registry.ids()).toEqual([]);
    registry.register(new MockMovementTrainingBackend());
    expect(registry.ids()).toEqual(["mock-ngram"]);
  });
});

function step(trajectoryId: string, tool: string, ts: number) {
  return { trajectoryId, tool, summary: `${tool} summary`, ts };
}
