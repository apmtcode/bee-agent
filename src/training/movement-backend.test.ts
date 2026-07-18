import { describe, expect, it } from "vitest";
import type { ReplayManifest } from "../capture/replay.js";
import type { TrajectorySpan } from "../capture/trajectory.js";
import {
  DEFAULT_MOVEMENT_TOKENIZER,
  MockMovementBackend,
  buildMovementDataset,
  buildMovementDatasetFromTrajectories,
  evaluateMovementFidelity,
  rolloutMovements,
} from "./movement-backend.js";

function syntheticTrajectory(id: string, steps: Array<["obs" | "act", string, string]>): TrajectorySpan {
  const observations: TrajectorySpan["observations"] = [];
  const actions: TrajectorySpan["actions"] = [];
  steps.forEach(([kind, key, summary], index) => {
    if (kind === "obs") {
      observations.push({ kind: "observation", source: key, summary, ts: index });
    } else {
      actions.push({ kind: "action", tool: key, summary, ts: index });
    }
  });
  return {
    id,
    sessionId: `session-${id}`,
    createdAt: "2026-07-18T00:00:00.000Z",
    captureTier: "full",
    observations,
    actions,
  };
}

// A repeated "open menu -> tap Save" movement pattern.
function menuSaveTrajectory(id: string): TrajectorySpan {
  return syntheticTrajectory(id, [
    ["obs", "device", "editor active"],
    ["act", "device", "swiped up"],
    ["obs", "device", "menu open"],
    ["act", "device", "tapped Save"],
  ]);
}

describe("buildMovementDataset", () => {
  it("emits one sample per action with a bounded preceding context", () => {
    const dataset = buildMovementDatasetFromTrajectories([menuSaveTrajectory("t1")], {
      contextWindow: 2,
    });
    expect(dataset.samples).toHaveLength(2);
    // First action ("swiped up") has one preceding observation.
    expect(dataset.samples[0]?.context).toEqual(["obs:device"]);
    expect(dataset.samples[0]?.action).toEqual({ tool: "device", summary: "swiped up" });
    // Second action ("tapped Save") is preceded by obs, act, obs — capped to 2.
    expect(dataset.samples[1]?.context).toEqual(["act:device", "obs:device"]);
    expect(dataset.samples[1]?.action).toEqual({ tool: "device", summary: "tapped Save" });
    expect(dataset.vocabulary).toContain("obs:device");
    expect(dataset.vocabulary).toContain("act:device");
  });

  it("orders trajectory events by timestamp when building replays", () => {
    const dataset = buildMovementDatasetFromTrajectories([menuSaveTrajectory("t1")]);
    expect(dataset.samples.map((sample) => sample.action.summary)).toEqual([
      "swiped up",
      "tapped Save",
    ]);
  });
});

describe("MockMovementBackend train + infer", () => {
  it("reproduces recorded movements from a matching context (objective 2c)", async () => {
    const backend = new MockMovementBackend();
    const dataset = buildMovementDatasetFromTrajectories([
      menuSaveTrajectory("t1"),
      menuSaveTrajectory("t2"),
      menuSaveTrajectory("t3"),
    ]);
    const model = await backend.train(dataset);

    const afterMenu = await backend.infer(model, ["act:device", "obs:device"]);
    expect(afterMenu?.action).toEqual({ tool: "device", summary: "tapped Save" });
    expect(afterMenu?.confidence).toBe(1);
    expect(afterMenu?.matchedContextLength).toBe(2);
  });

  it("generalizes to a new-but-related context via backoff (objective 2d)", async () => {
    const backend = new MockMovementBackend();
    // Train on menu->Save. Query a context whose *full* prefix was never seen
    // (different leading observation) but whose recent suffix matches.
    const dataset = buildMovementDatasetFromTrajectories([
      menuSaveTrajectory("t1"),
      menuSaveTrajectory("t2"),
    ]);
    const model = await backend.train(dataset, { contextWindow: 3 });

    const novelContext = ["obs:browser", "act:device", "obs:device"];
    const prediction = await backend.infer(model, novelContext);
    expect(prediction?.action.summary).toBe("tapped Save");
    // It could not match the full 3-token prefix, so it backed off.
    expect(prediction?.matchedContextLength).toBeLessThan(3);
    expect(prediction?.matchedContextLength).toBeGreaterThan(0);
  });

  it("falls back to the unconditional prior when no context matches", async () => {
    const backend = new MockMovementBackend();
    const dataset = buildMovementDatasetFromTrajectories([menuSaveTrajectory("t1")]);
    const model = await backend.train(dataset);
    const prediction = await backend.infer(model, ["obs:totally-unseen"]);
    expect(prediction).toBeDefined();
    expect(prediction?.matchedContextLength).toBe(0);
  });

  it("returns undefined for an empty model", async () => {
    const backend = new MockMovementBackend();
    const model = await backend.train({ version: 1, contextWindow: 2, samples: [], vocabulary: [] });
    expect(await backend.infer(model, ["obs:device"])).toBeUndefined();
  });

  it("is deterministic: identical input yields byte-identical models", async () => {
    const backend = new MockMovementBackend();
    const dataset = buildMovementDatasetFromTrajectories([menuSaveTrajectory("a"), menuSaveTrajectory("b")]);
    const first = await backend.train(dataset);
    const second = await backend.train(dataset);
    expect(JSON.stringify(first)).toBe(JSON.stringify(second));
  });

  it("breaks ties deterministically by action key", async () => {
    const backend = new MockMovementBackend();
    // Same context, two equally-frequent actions -> lower action key wins.
    const replays: ReplayManifest[] = [
      {
        version: 1,
        sessionId: "s",
        trajectoryIds: ["t"],
        eventCount: 4,
        events: [
          { kind: "observation", ts: 0, trajectoryId: "t", source: "device", summary: "x" },
          { kind: "action", ts: 1, trajectoryId: "t", tool: "device", summary: "zebra" },
          { kind: "observation", ts: 2, trajectoryId: "t", source: "device", summary: "x" },
          { kind: "action", ts: 3, trajectoryId: "t", tool: "device", summary: "alpha" },
        ],
      },
    ];
    const dataset = buildMovementDataset(replays, { contextWindow: 1 });
    const model = await backend.train(dataset);
    const prediction = await backend.infer(model, ["obs:device"]);
    expect(prediction?.action.summary).toBe("alpha");
  });
});

describe("rolloutMovements", () => {
  it("autoregressively repeats a learned movement sequence", async () => {
    const backend = new MockMovementBackend();
    const dataset = buildMovementDatasetFromTrajectories([
      menuSaveTrajectory("t1"),
      menuSaveTrajectory("t2"),
    ]);
    const model = await backend.train(dataset, { contextWindow: 2 });
    const steps = await rolloutMovements(backend, model, ["obs:device"], { maxSteps: 4 });
    expect(steps.length).toBeGreaterThan(0);
    expect(steps[0]?.action.summary).toBe("swiped up");
    expect(steps.map((step) => step.step)).toEqual(steps.map((_, index) => index));
  });

  it("honours stopOnRepeat as a loop guard", async () => {
    const backend = new MockMovementBackend();
    // A trajectory where the unconditional prior is a single dominant action.
    const dataset = buildMovementDatasetFromTrajectories([
      syntheticTrajectory("t", [
        ["obs", "device", "idle"],
        ["act", "device", "poll"],
      ]),
    ]);
    const model = await backend.train(dataset, { contextWindow: 1 });
    const steps = await rolloutMovements(backend, model, [], { maxSteps: 10, stopOnRepeat: true });
    expect(steps.length).toBeLessThanOrEqual(2);
  });
});

describe("evaluateMovementFidelity", () => {
  it("scores perfect fidelity on the training distribution", async () => {
    const backend = new MockMovementBackend();
    const dataset = buildMovementDatasetFromTrajectories([
      menuSaveTrajectory("t1"),
      menuSaveTrajectory("t2"),
      menuSaveTrajectory("t3"),
    ]);
    const model = await backend.train(dataset);
    const report = await evaluateMovementFidelity(backend, model, dataset);
    expect(report.total).toBe(6);
    expect(report.accuracy).toBe(1);
    expect(report.averageMatchedContextLength).toBeGreaterThan(0);
  });

  it("generalizes to a held-out but related trajectory above chance", async () => {
    const backend = new MockMovementBackend();
    const train = buildMovementDatasetFromTrajectories([
      menuSaveTrajectory("t1"),
      menuSaveTrajectory("t2"),
      menuSaveTrajectory("t3"),
    ]);
    const model = await backend.train(train, { contextWindow: 3 });

    // Held-out trajectory: same menu->Save movement reached via a new preamble.
    const heldOut = buildMovementDatasetFromTrajectories([
      syntheticTrajectory("held", [
        ["obs", "browser", "tab active"],
        ["act", "device", "swiped up"],
        ["obs", "device", "menu open"],
        ["act", "device", "tapped Save"],
      ]),
    ]);
    const report = await evaluateMovementFidelity(backend, model, heldOut);
    expect(report.accuracy).toBeGreaterThanOrEqual(0.5);
  });
});

describe("DEFAULT_MOVEMENT_TOKENIZER", () => {
  it("is coarse on context but keeps full action summaries in the dataset", () => {
    expect(DEFAULT_MOVEMENT_TOKENIZER.observation({
      kind: "observation",
      ts: 0,
      trajectoryId: "t",
      source: "device",
      summary: "very specific detail",
    })).toBe("obs:device");
    expect(DEFAULT_MOVEMENT_TOKENIZER.transcript({
      kind: "transcript",
      ts: 0,
      messageId: "m",
      role: "assistant",
      content: "hi",
    })).toBe("msg:assistant");
  });
});
