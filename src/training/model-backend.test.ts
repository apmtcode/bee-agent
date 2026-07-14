import { describe, expect, it } from "vitest";
import {
  DeterministicNearestNeighborBackend,
  buildMovementDataset,
  createMovementBackend,
  featurizeContext,
  type MovementDataset,
} from "./model-backend.js";
import type { ReviewedExportManifest } from "./export-manifest.js";

function manifestWith(replays: ReviewedExportManifest["replays"]): ReviewedExportManifest {
  return {
    version: 1,
    createdAt: "2026-01-01T00:00:00.000Z",
    reviewedBy: "operator",
    purpose: "movement learning",
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

// A tiny synthetic movement stream: two recorded trajectories, each an
// observation followed by the action that repeated it.
const manifest = manifestWith([
  {
    sessionId: "sess-1",
    trajectoryIds: ["traj-open"],
    eventCount: 2,
    events: [
      { kind: "observation", ts: 1, trajectoryId: "traj-open", source: "browser", summary: "open deploy dialog" },
      { kind: "action", ts: 2, trajectoryId: "traj-open", tool: "mouse", summary: "click deploy button" },
    ],
  },
  {
    sessionId: "sess-2",
    trajectoryIds: ["traj-save"],
    eventCount: 2,
    events: [
      { kind: "observation", ts: 1, trajectoryId: "traj-save", source: "editor", summary: "unsaved changes banner" },
      { kind: "action", ts: 2, trajectoryId: "traj-save", tool: "keyboard", summary: "press save shortcut" },
    ],
  },
]);

describe("buildMovementDataset", () => {
  it("emits one supervised sample per recorded action with its preceding context", () => {
    const dataset = buildMovementDataset(manifest);
    expect(dataset.sampleCount).toBe(2);
    expect(dataset.samples[0]).toEqual({
      trajectoryId: "traj-open",
      contextEvents: [{ kind: "observation", source: "browser", summary: "open deploy dialog" }],
      action: { tool: "mouse", summary: "click deploy button" },
    });
    expect(dataset.samples[1].action).toEqual({ tool: "keyboard", summary: "press save shortcut" });
  });

  it("respects the dataset context window", () => {
    const longReplay = manifestWith([
      {
        sessionId: "sess-x",
        trajectoryIds: ["traj-x"],
        eventCount: 4,
        events: [
          { kind: "observation", ts: 1, trajectoryId: "traj-x", source: "a", summary: "one" },
          { kind: "observation", ts: 2, trajectoryId: "traj-x", source: "b", summary: "two" },
          { kind: "observation", ts: 3, trajectoryId: "traj-x", source: "c", summary: "three" },
          { kind: "action", ts: 4, trajectoryId: "traj-x", tool: "mouse", summary: "go" },
        ],
      },
    ]);
    const dataset = buildMovementDataset(longReplay, { datasetContextWindow: 2 });
    expect(dataset.samples[0].contextEvents).toHaveLength(2);
    expect(dataset.samples[0].contextEvents.map((event) => event.summary)).toEqual(["two", "three"]);
  });
});

describe("DeterministicNearestNeighborBackend", () => {
  it("recalls the exact recorded movement for a seen context (repeat)", async () => {
    const backend = new DeterministicNearestNeighborBackend();
    const model = await backend.train(buildMovementDataset(manifest));

    const prediction = backend.predict(model, {
      events: [{ kind: "observation", source: "browser", summary: "open deploy dialog" }],
    });

    expect(prediction.source).toBe("recall");
    expect(prediction.confidence).toBeGreaterThanOrEqual(0.999);
    expect(prediction).toMatchObject({ tool: "mouse", summary: "click deploy button", matchedTrajectoryId: "traj-open" });
  });

  it("generalizes to a new-but-related context via the nearest recorded movement", async () => {
    const backend = new DeterministicNearestNeighborBackend();
    const model = await backend.train(buildMovementDataset(manifest));

    // Unseen phrasing that shares tokens with the "deploy" trajectory only.
    const prediction = backend.predict(model, {
      events: [{ kind: "observation", source: "browser", summary: "reopen the deploy panel" }],
    });

    expect(prediction.source).toBe("generalized");
    expect(prediction.confidence).toBeGreaterThan(0);
    expect(prediction.confidence).toBeLessThan(0.999);
    expect(prediction.tool).toBe("mouse");
    expect(prediction.matchedTrajectoryId).toBe("traj-open");
  });

  it("falls back to the majority movement when a query shares no tokens", async () => {
    // Separate replays so each action's context is only its own observation.
    const majorityManifest = manifestWith([
      {
        sessionId: "sess-a",
        trajectoryIds: ["a"],
        eventCount: 2,
        events: [
          { kind: "observation", ts: 1, trajectoryId: "a", source: "s", summary: "alpha" },
          { kind: "action", ts: 2, trajectoryId: "a", tool: "mouse", summary: "common move" },
        ],
      },
      {
        sessionId: "sess-b",
        trajectoryIds: ["b"],
        eventCount: 2,
        events: [
          { kind: "observation", ts: 1, trajectoryId: "b", source: "s", summary: "beta" },
          { kind: "action", ts: 2, trajectoryId: "b", tool: "mouse", summary: "common move" },
        ],
      },
      {
        sessionId: "sess-c",
        trajectoryIds: ["c"],
        eventCount: 2,
        events: [
          { kind: "observation", ts: 1, trajectoryId: "c", source: "s", summary: "gamma" },
          { kind: "action", ts: 2, trajectoryId: "c", tool: "keyboard", summary: "rare move" },
        ],
      },
    ]);
    const backend = new DeterministicNearestNeighborBackend();
    const model = await backend.train(buildMovementDataset(majorityManifest));

    // Different event kind, tool, and words → zero token overlap with training.
    const prediction = backend.predict(model, {
      events: [{ kind: "action", tool: "clipboard", summary: "zzz totally unrelated" }],
    });

    expect(prediction.source).toBe("fallback");
    expect(prediction.confidence).toBe(0);
    expect(prediction).toMatchObject({ tool: "mouse", summary: "common move" });
  });

  it("returns a noop for an empty model", async () => {
    const backend = new DeterministicNearestNeighborBackend();
    const model = await backend.train({ version: 1, sampleCount: 0, samples: [] } satisfies MovementDataset);
    const prediction = backend.predict(model, { events: [] });
    expect(prediction).toEqual({ tool: "noop", summary: "", confidence: 0, source: "fallback" });
  });

  it("is deterministic across repeated inference", async () => {
    const backend = new DeterministicNearestNeighborBackend();
    const model = await backend.train(buildMovementDataset(manifest));
    const context = { events: [{ kind: "observation" as const, source: "editor", summary: "changes banner" }] };
    const first = backend.predict(model, context);
    const second = backend.predict(model, context);
    expect(first).toEqual(second);
  });

  it("produces a JSON-serializable model artifact", async () => {
    const backend = new DeterministicNearestNeighborBackend();
    const model = await backend.train(buildMovementDataset(manifest));
    const roundTripped = JSON.parse(JSON.stringify(model));
    expect(roundTripped).toEqual(model);
    const prediction = backend.predict(roundTripped, {
      events: [{ kind: "observation", source: "browser", summary: "open deploy dialog" }],
    });
    expect(prediction.source).toBe("recall");
  });
});

describe("createMovementBackend", () => {
  it("resolves the deterministic backend by default and by name", () => {
    expect(createMovementBackend().name).toBe("deterministic-nn");
    expect(createMovementBackend("deterministic-nn").name).toBe("deterministic-nn");
  });
});

describe("featurizeContext", () => {
  it("tokenizes only the trailing window of events", () => {
    const tokens = featurizeContext(
      [
        { kind: "observation", summary: "first event" },
        { kind: "observation", summary: "second event" },
      ],
      1,
    );
    expect(tokens).toContain("second");
    expect(tokens).not.toContain("first");
  });
});
