import { describe, expect, it } from "vitest";
import type { ReviewedExportManifest } from "./export-manifest.js";
import {
  DEFAULT_MOVEMENT_BACKEND_ID,
  DeterministicMockBackend,
  MovementBackendRegistry,
  buildMovementDataset,
  evaluateMovementPolicy,
  tokenizeMovementText,
  type MovementDataset,
} from "./movement-backend.js";

function replay(
  trajectoryId: string,
  events: Array<
    | { kind: "observation"; ts: number; summary: string }
    | { kind: "transcript"; ts: number; role: "user"; content: string }
    | { kind: "action"; ts: number; tool: string; summary: string }
  >,
): ReviewedExportManifest["replays"][number] {
  return {
    sessionId: `session-${trajectoryId}`,
    trajectoryIds: [trajectoryId],
    eventCount: events.length,
    events: events.map((event) =>
      event.kind === "observation"
        ? { kind: "observation", ts: event.ts, trajectoryId, source: "os", summary: event.summary }
        : event.kind === "transcript"
          ? { kind: "transcript", ts: event.ts, messageId: `m-${event.ts}`, role: event.role, content: event.content }
          : { kind: "action", ts: event.ts, trajectoryId, tool: event.tool, summary: event.summary },
    ),
  };
}

function manifestWith(replays: ReviewedExportManifest["replays"]): ReviewedExportManifest {
  return {
    version: 1,
    createdAt: "2026-01-01T00:00:00.000Z",
    reviewedBy: "tester",
    purpose: "unit-test",
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

describe("tokenizeMovementText", () => {
  it("keeps dotted filenames as single tokens and lowercases", () => {
    expect(tokenizeMovementText("Open File Report.TXT")).toEqual(["open", "file", "report.txt"]);
  });
});

describe("buildMovementDataset", () => {
  it("splits context and ordered actions from replay timelines", () => {
    const manifest = manifestWith([
      replay("t1", [
        { kind: "observation", ts: 1, summary: "open file report.txt" },
        { kind: "action", ts: 2, tool: "type", summary: "type report.txt" },
        { kind: "action", ts: 3, tool: "key", summary: "press enter" },
      ]),
    ]);
    const dataset = buildMovementDataset(manifest);
    expect(dataset.examples).toHaveLength(1);
    const [example] = dataset.examples;
    expect(example.contextTokens).toEqual(["open", "file", "report.txt"]);
    expect(example.actions.map((a) => a.tool)).toEqual(["type", "key"]);
  });

  it("drops replays with no actions", () => {
    const manifest = manifestWith([
      replay("t1", [{ kind: "observation", ts: 1, summary: "just looking" }]),
    ]);
    expect(buildMovementDataset(manifest).examples).toHaveLength(0);
  });
});

const trainingDataset: MovementDataset = buildMovementDataset(
  manifestWith([
    replay("open-file", [
      { kind: "observation", ts: 1, summary: "open file report.txt" },
      { kind: "action", ts: 2, tool: "type", summary: "type report.txt" },
      { kind: "action", ts: 3, tool: "key", summary: "press enter" },
    ]),
    replay("close-window", [
      { kind: "observation", ts: 1, summary: "close window editor" },
      { kind: "action", ts: 2, tool: "click", summary: "click close editor" },
    ]),
  ]),
);

describe("DeterministicMockBackend", () => {
  it("reproduces a recorded movement exactly for a matching goal", async () => {
    const backend = new DeterministicMockBackend();
    const artifact = await backend.train(trainingDataset);
    const policy = backend.load(artifact);

    const prediction = policy.predict({ goal: "open file report.txt" });
    expect(prediction.matchedTrajectoryId).toBe("open-file");
    expect(prediction.confidence).toBe(1);
    expect(prediction.generalized).toBe(false);
    expect(prediction.actions.map((a) => a.summary)).toEqual(["type report.txt", "press enter"]);
  });

  it("generalizes to a new-but-related movement via slot substitution", async () => {
    const backend = new DeterministicMockBackend();
    const artifact = await backend.train(trainingDataset);
    const policy = backend.load(artifact);

    const prediction = policy.predict({ goal: "open file budget.csv" });
    expect(prediction.matchedTrajectoryId).toBe("open-file");
    expect(prediction.generalized).toBe(true);
    expect(prediction.substitutions).toEqual([{ from: "report.txt", to: "budget.csv" }]);
    expect(prediction.actions.map((a) => a.summary)).toEqual(["type budget.csv", "press enter"]);
    // The non-parameter action is untouched.
    expect(prediction.actions[1].tool).toBe("key");
  });

  it("routes to the closest exemplar by structural skeleton", async () => {
    const backend = new DeterministicMockBackend();
    const artifact = await backend.train(trainingDataset);
    const policy = backend.load(artifact);

    const prediction = policy.predict({ goal: "close window terminal" });
    expect(prediction.matchedTrajectoryId).toBe("close-window");
    expect(prediction.actions.map((a) => a.summary)).toEqual(["click close terminal"]);
  });

  it("returns no actions below the configured confidence floor", async () => {
    const backend = new DeterministicMockBackend();
    const artifact = await backend.train(trainingDataset, { minConfidence: 0.9 });
    const policy = backend.load(artifact);

    const prediction = policy.predict({ goal: "unrelated goal about nothing" });
    expect(prediction.matchedTrajectoryId).toBeNull();
    expect(prediction.actions).toHaveLength(0);
  });

  it("produces a JSON-serializable artifact", async () => {
    const backend = new DeterministicMockBackend();
    const artifact = await backend.train(trainingDataset);
    const roundTripped = JSON.parse(JSON.stringify(artifact));
    expect(roundTripped.backendId).toBe(DEFAULT_MOVEMENT_BACKEND_ID);
    const policy = backend.load(roundTripped);
    expect(policy.predict({ goal: "open file report.txt" }).confidence).toBe(1);
  });
});

describe("evaluateMovementPolicy", () => {
  it("scores perfect fidelity when replaying the training examples", async () => {
    const backend = new DeterministicMockBackend();
    const artifact = await backend.train(trainingDataset);
    const policy = backend.load(artifact);
    const result = evaluateMovementPolicy(policy, trainingDataset.examples);
    expect(result.count).toBe(2);
    expect(result.exactMatchRate).toBe(1);
    expect(result.meanToolAccuracy).toBe(1);
    expect(result.meanSummaryF1).toBe(1);
  });

  it("measures generalization on held-out related trajectories", async () => {
    const backend = new DeterministicMockBackend();
    const artifact = await backend.train(trainingDataset);
    const policy = backend.load(artifact);
    const heldOut = buildMovementDataset(
      manifestWith([
        replay("open-notes", [
          { kind: "observation", ts: 1, summary: "open file notes.md" },
          { kind: "action", ts: 2, tool: "type", summary: "type notes.md" },
          { kind: "action", ts: 3, tool: "key", summary: "press enter" },
        ]),
      ]),
    ).examples;
    const result = evaluateMovementPolicy(policy, heldOut);
    expect(result.exactMatchRate).toBe(1);
  });

  it("returns zeroed metrics for an empty held-out set", async () => {
    const backend = new DeterministicMockBackend();
    const policy = backend.load(await backend.train(trainingDataset));
    expect(evaluateMovementPolicy(policy, [])).toEqual({
      count: 0,
      exactMatchRate: 0,
      meanToolAccuracy: 0,
      meanSummaryF1: 0,
    });
  });
});

describe("MovementBackendRegistry", () => {
  it("seeds the deterministic mock by default and creates it", () => {
    const registry = new MovementBackendRegistry();
    expect(registry.has(DEFAULT_MOVEMENT_BACKEND_ID)).toBe(true);
    expect(registry.ids()).toContain(DEFAULT_MOVEMENT_BACKEND_ID);
    expect(registry.create().id).toBe(DEFAULT_MOVEMENT_BACKEND_ID);
  });

  it("supports registering an alternate backend and throws on unknown ids", () => {
    const registry = new MovementBackendRegistry(false);
    registry.register("stub", () => new DeterministicMockBackend());
    expect(registry.ids()).toEqual(["stub"]);
    expect(() => registry.create("does-not-exist")).toThrow(/unknown movement backend/);
  });
});
