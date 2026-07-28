import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import type { ReplayTimelineEvent } from "../capture/replay.js";
import type { TrajectorySpan } from "../capture/trajectory.js";
import {
  DETERMINISTIC_BACKEND_NAME,
  DeterministicMovementBackend,
  buildMovementDatasetFromReplays,
  buildMovementDatasetFromTrajectories,
  evaluateMovementFidelity,
  getMovementBackend,
  listMovementBackends,
  loadMovementModel,
  readMovementModel,
  saveMovementModel,
  type MovementDataset,
} from "./movement-model.js";

const tempDirs: string[] = [];

async function makeTempDir(): Promise<string> {
  const dir = await fs.mkdtemp(path.join(os.tmpdir(), "movement-model-"));
  tempDirs.push(dir);
  return dir;
}

afterEach(async () => {
  await Promise.all(tempDirs.splice(0).map((dir) => fs.rm(dir, { recursive: true, force: true })));
});

function actionEvent(trajectoryId: string, tool: string, ts: number): ReplayTimelineEvent {
  return { kind: "action", ts, trajectoryId, tool, summary: `${tool}@${ts}` };
}

/** Two recorded "drag" movements: move -> down -> move -> up. */
const dragReplays = [
  {
    trajectoryIds: ["traj-1"],
    events: [
      actionEvent("traj-1", "mouse.move", 10),
      actionEvent("traj-1", "mouse.down", 20),
      actionEvent("traj-1", "mouse.move", 30),
      actionEvent("traj-1", "mouse.up", 40),
    ] as ReplayTimelineEvent[],
  },
  {
    trajectoryIds: ["traj-2"],
    events: [
      actionEvent("traj-2", "mouse.move", 5),
      actionEvent("traj-2", "mouse.down", 15),
      actionEvent("traj-2", "mouse.move", 25),
      actionEvent("traj-2", "mouse.up", 35),
    ] as ReplayTimelineEvent[],
  },
];

describe("movement dataset construction", () => {
  it("builds one ordered action sequence per trajectory from replays", () => {
    const dataset = buildMovementDatasetFromReplays([
      {
        trajectoryIds: ["a", "b"],
        events: [
          actionEvent("b", "key.press", 100),
          actionEvent("a", "mouse.move", 10),
          actionEvent("a", "mouse.down", 5), // out of order on purpose
          { kind: "observation", ts: 1, trajectoryId: "a", source: "screen", summary: "noise" },
        ] as ReplayTimelineEvent[],
      },
    ]);
    const a = dataset.sequences.find((s) => s.id === "a");
    const b = dataset.sequences.find((s) => s.id === "b");
    // Observation events are ignored; actions are sorted by ts.
    expect(a?.tokens).toEqual(["mouse.down", "mouse.move"]);
    expect(b?.tokens).toEqual(["key.press"]);
    expect(a?.source).toBe("replay");
  });

  it("builds sequences from trajectory spans and drops empty ones", () => {
    const trajectories: TrajectorySpan[] = [
      {
        id: "t1",
        sessionId: "s1",
        createdAt: "2026-01-01T00:00:00.000Z",
        captureTier: "full",
        observations: [],
        actions: [
          { kind: "action", tool: "key.press", summary: "a", ts: 2 },
          { kind: "action", tool: "key.release", summary: "a", ts: 1 },
        ],
      },
      {
        id: "t2",
        sessionId: "s1",
        createdAt: "2026-01-01T00:00:00.000Z",
        captureTier: "full",
        observations: [],
        actions: [],
      },
    ];
    const dataset = buildMovementDatasetFromTrajectories(trajectories);
    expect(dataset.sequences).toHaveLength(1);
    expect(dataset.sequences[0]?.tokens).toEqual(["key.release", "key.press"]);
  });
});

describe("deterministic movement backend", () => {
  it("is registered as the default backend", () => {
    expect(listMovementBackends()).toContain(DETERMINISTIC_BACKEND_NAME);
    expect(getMovementBackend(DETERMINISTIC_BACKEND_NAME)).toBeInstanceOf(DeterministicMovementBackend);
  });

  it("throws for an unknown backend", () => {
    expect(() => getMovementBackend("nope")).toThrow(/unknown movement model backend/);
  });

  it("reproduces recorded movements exactly (objective 2c)", async () => {
    const dataset = buildMovementDatasetFromReplays(dragReplays);
    const model = await new DeterministicMovementBackend().train(dataset, { order: 2 });
    // Seeded with the first movement, a rollout replays the recorded drag.
    const rollout = model.rollout({ seed: ["mouse.move"] });
    expect(rollout).toEqual(["mouse.move", "mouse.down", "mouse.move", "mouse.up"]);

    const fidelity = evaluateMovementFidelity(model, dataset.sequences);
    expect(fidelity.exactReproductionRate).toBe(1);
    expect(fidelity.nextTokenAccuracy).toBe(1);
  });

  it("predicts the correct next movement from a known prefix", async () => {
    const dataset = buildMovementDatasetFromReplays(dragReplays);
    const model = await new DeterministicMovementBackend().train(dataset, { order: 2 });
    const prediction = model.predictNext(["mouse.down", "mouse.move"]);
    expect(prediction.token).toBe("mouse.up");
    expect(prediction.probability).toBeGreaterThan(0.5);
    // The full distribution is normalized (sums to ~1).
    const mass = prediction.distribution.reduce((sum, entry) => sum + entry.probability, 0);
    expect(mass).toBeCloseTo(1, 5);
  });

  it("generalizes to a new but related movement via backoff (objective 2d)", async () => {
    // Train a click grammar: down -> up always. A never-seen prefix that still
    // ends in "mouse.down" should predict "mouse.up" by backing off to order 1.
    const dataset: MovementDataset = {
      version: 1,
      sequences: [
        { id: "c1", tokens: ["mouse.down", "mouse.up"], source: "synthetic" },
        { id: "c2", tokens: ["mouse.down", "mouse.up"], source: "synthetic" },
        { id: "c3", tokens: ["mouse.down", "mouse.up"], source: "synthetic" },
      ],
    };
    const model = await new DeterministicMovementBackend().train(dataset, { order: 3 });
    const prediction = model.predictNext(["key.press", "scroll", "mouse.down"]);
    expect(prediction.token).toBe("mouse.up");
    // The exact 3-gram context was never seen, so it backed off to a shorter one.
    expect(prediction.backoffOrder).toBeLessThan(3);
    expect(prediction.backoffOrder).toBeGreaterThanOrEqual(1);
  });

  it("is deterministic: same dataset + options yields identical rollouts", async () => {
    const dataset = buildMovementDatasetFromReplays(dragReplays);
    const backend = new DeterministicMovementBackend();
    const a = await backend.train(dataset, { order: 2 });
    const b = await backend.train(dataset, { order: 2 });
    expect(a.rollout({ seed: ["mouse.move"] })).toEqual(b.rollout({ seed: ["mouse.move"] }));
  });

  it("round-trips through serialize/load with identical behaviour", async () => {
    const dataset = buildMovementDatasetFromReplays(dragReplays);
    const model = await new DeterministicMovementBackend().train(dataset, { order: 2 });
    const restored = loadMovementModel(model.serialize());
    expect(restored.backend).toBe(DETERMINISTIC_BACKEND_NAME);
    expect(restored.rollout({ seed: ["mouse.move"] })).toEqual(model.rollout({ seed: ["mouse.move"] }));
    expect(restored.predictNext(["mouse.down"]).token).toBe(model.predictNext(["mouse.down"]).token);
  });

  it("persists a trained model to disk and reloads it", async () => {
    const dir = await makeTempDir();
    const file = path.join(dir, "artifacts", "movement-model.json");
    const dataset = buildMovementDatasetFromReplays(dragReplays);
    const model = await new DeterministicMovementBackend().train(dataset, { order: 2 });
    await saveMovementModel(file, model);

    const reloaded = await readMovementModel(file);
    expect(reloaded).toBeDefined();
    expect(reloaded?.rollout({ seed: ["mouse.move"] })).toEqual(["mouse.move", "mouse.down", "mouse.move", "mouse.up"]);
    expect(await readMovementModel(path.join(dir, "missing.json"))).toBeUndefined();
  });

  it("handles an empty dataset without crashing", async () => {
    const model = await new DeterministicMovementBackend().train({ version: 1, sequences: [] });
    const report = evaluateMovementFidelity(model, []);
    expect(report.sequenceCount).toBe(0);
    expect(report.nextTokenAccuracy).toBe(0);
    expect(model.rollout()).toEqual([]);
  });
});

describe("generalization eval harness", () => {
  it("scores fidelity on held-out related sequences", async () => {
    // Train on drags; evaluate on a held-out drag (same grammar) -> high score.
    const train = buildMovementDatasetFromReplays(dragReplays);
    const model = await new DeterministicMovementBackend().train(train, { order: 2 });
    const heldOut: MovementDataset = {
      version: 1,
      sequences: [{ id: "held", tokens: ["mouse.move", "mouse.down", "mouse.move", "mouse.up"], source: "synthetic" }],
    };
    const report = evaluateMovementFidelity(model, heldOut.sequences);
    expect(report.exactReproductionRate).toBe(1);
    expect(report.perSequence[0]?.reproduced).toBe(true);
    expect(report.nextTokenAccuracy).toBeGreaterThan(0.9);
  });
});
