import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import type { ReplayManifest } from "../capture/replay.js";
import type { TrajectorySpan } from "../capture/trajectory.js";
import {
  MOVEMENT_END_TOKEN,
  MOVEMENT_START_TOKEN,
  MarkovMovementBackend,
  MovementModelTrainer,
  type MovementDataset,
  movementDatasetFromReplays,
  movementDatasetFromTrajectories,
  movementSequenceFromReplay,
  tokenizeAction,
} from "./movement-model.js";

const CLICK = tokenizeAction("device", "tapped submit");
const TYPE = tokenizeAction("device", "typed into field");
const SCROLL = tokenizeAction("device", "scrolled down");
const SAVE = tokenizeAction("device", "triggered save shortcut");

function datasetFrom(sequences: string[][]): MovementDataset {
  return { version: 1, sequences };
}

describe("tokenizeAction", () => {
  it("produces stable, normalized tokens", () => {
    expect(tokenizeAction("Device", "Tapped  Submit!")).toBe("device:tapped-submit");
    expect(tokenizeAction("device", "tapped submit")).toBe(tokenizeAction("device", "tapped submit"));
  });
});

describe("MarkovMovementBackend", () => {
  it("repeats a single recorded movement run deterministically (objective 2c)", async () => {
    const backend = new MarkovMovementBackend(2);
    const recorded = [CLICK, TYPE, SCROLL, SAVE];
    const snapshot = await backend.train(datasetFrom([recorded]));

    const generated = await backend.generate(snapshot, [], { maxSteps: 16 });
    expect(generated).toEqual(recorded);
  });

  it("predicts the most likely next movement from context", async () => {
    const backend = new MarkovMovementBackend(2);
    // CLICK is followed by TYPE twice and by SCROLL once → argmax is TYPE.
    const snapshot = await backend.train(
      datasetFrom([
        [CLICK, TYPE],
        [CLICK, TYPE],
        [CLICK, SCROLL],
      ]),
    );

    const prediction = await backend.predictNext(snapshot, [CLICK]);
    expect(prediction.token).toBe(TYPE);
    expect(prediction.probability).toBeCloseTo(2 / 3, 10);
    expect(prediction.distribution.map((c) => c.token)).toEqual([TYPE, SCROLL]);
    // Probabilities form a valid distribution.
    const mass = prediction.distribution.reduce((sum, c) => sum + c.probability, 0);
    expect(mass).toBeCloseTo(1, 10);
  });

  it("backs off to shorter contexts for new-but-related movements (objective 2d)", async () => {
    const backend = new MarkovMovementBackend(2);
    // Order-2 context [SCROLL, CLICK] was never recorded, but CLICK→TYPE was.
    const snapshot = await backend.train(
      datasetFrom([
        [CLICK, TYPE, SAVE],
        [TYPE, CLICK, TYPE],
      ]),
    );

    const prediction = await backend.predictNext(snapshot, [SCROLL, CLICK]);
    // Falls back to order-1 (the CLICK→ distribution) instead of failing.
    expect(prediction.backoffOrder).toBe(1);
    expect(prediction.token).toBe(TYPE);
  });

  it("uses the start sentinel to model how runs begin and end", async () => {
    const backend = new MarkovMovementBackend(1);
    const snapshot = await backend.train(datasetFrom([[CLICK, SAVE]]));

    const first = await backend.predictNext(snapshot, [MOVEMENT_START_TOKEN]);
    expect(first.token).toBe(CLICK);
    const last = await backend.predictNext(snapshot, [SAVE]);
    expect(last.token).toBe(MOVEMENT_END_TOKEN);
  });

  it("generalizes to a novel run by generating a plausible continuation", async () => {
    const backend = new MarkovMovementBackend(2);
    const snapshot = await backend.train(
      datasetFrom([
        [CLICK, TYPE, SAVE],
        [CLICK, TYPE, SAVE],
        [SCROLL, CLICK, TYPE, SAVE],
      ]),
    );

    // Seed with a context the model never recorded verbatim.
    const generated = await backend.generate(snapshot, [SCROLL, CLICK], { maxSteps: 8 });
    expect(generated.length).toBeGreaterThan(0);
    expect(generated).not.toContain(MOVEMENT_START_TOKEN);
    expect(generated).not.toContain(MOVEMENT_END_TOKEN);
    // Learned that TYPE→SAVE follows, so it completes the related run.
    expect(generated).toContain(SAVE);
  });

  it("seeded sampling is reproducible and greedy generation is deterministic", async () => {
    const backend = new MarkovMovementBackend(2);
    const snapshot = await backend.train(
      datasetFrom([
        [CLICK, TYPE],
        [CLICK, SCROLL],
        [CLICK, TYPE],
      ]),
    );

    const a = await backend.generate(snapshot, [], { seed: 42, maxSteps: 8 });
    const b = await backend.generate(snapshot, [], { seed: 42, maxSteps: 8 });
    expect(a).toEqual(b);

    const greedy1 = await backend.generate(snapshot, [], { maxSteps: 8 });
    const greedy2 = await backend.generate(snapshot, [], { maxSteps: 8 });
    expect(greedy1).toEqual(greedy2);
  });

  it("rejects invalid order and incompatible snapshots", async () => {
    expect(() => new MarkovMovementBackend(0)).toThrow(/positive integer/);
    const snapshot = await new MarkovMovementBackend().train(datasetFrom([[CLICK]]));
    const foreign = { ...snapshot, backendId: "other" };
    await expect(new MarkovMovementBackend().predictNext(foreign, [CLICK])).rejects.toThrow(/not compatible/);
  });
});

describe("dataset adapters", () => {
  it("builds sequences from replay manifests and drops empty runs", () => {
    const manifest: ReplayManifest = {
      version: 1,
      sessionId: "s1",
      trajectoryIds: ["t1"],
      eventCount: 3,
      events: [
        { kind: "observation", ts: 1, trajectoryId: "t1", source: "device", summary: "app active" },
        { kind: "action", ts: 2, trajectoryId: "t1", tool: "device", summary: "tapped submit" },
        { kind: "action", ts: 3, trajectoryId: "t1", tool: "device", summary: "typed into field" },
      ],
    };
    expect(movementSequenceFromReplay(manifest)).toEqual([CLICK, TYPE]);

    const empty: ReplayManifest = { ...manifest, events: [], eventCount: 0 };
    const dataset = movementDatasetFromReplays([manifest, empty]);
    expect(dataset.sequences).toEqual([[CLICK, TYPE]]);
  });

  it("builds sequences from trajectory spans", () => {
    const span: TrajectorySpan = {
      id: "t1",
      sessionId: "s1",
      createdAt: "2026-07-24T00:00:00.000Z",
      captureTier: "operator",
      observations: [],
      actions: [
        { kind: "action", tool: "device", summary: "tapped submit", ts: 1 },
        { kind: "action", tool: "device", summary: "triggered save shortcut", ts: 2 },
      ],
    };
    expect(movementDatasetFromTrajectories([span]).sequences).toEqual([[CLICK, SAVE]]);
  });
});

describe("MovementModelTrainer", () => {
  let rootDir: string;

  beforeEach(async () => {
    rootDir = await mkdtemp(path.join(tmpdir(), "bee-movement-"));
  });

  afterEach(async () => {
    await rm(rootDir, { recursive: true, force: true });
  });

  it("trains, persists, reloads, and infers with the same result", async () => {
    const trainer = new MovementModelTrainer(rootDir);
    expect(trainer.backendId).toBe("markov-backoff");

    const dataset = datasetFrom([
      [CLICK, TYPE, SAVE],
      [CLICK, TYPE, SAVE],
    ]);
    const snapshot = await trainer.trainAndSave(dataset, "models/movement.json");
    expect(snapshot.createdAtSequences).toBe(2);

    const reloaded = await trainer.load("models/movement.json");
    expect(reloaded).toEqual(snapshot);

    const fromDisk = await trainer.predictNext(reloaded!, [CLICK]);
    const fromMemory = await trainer.predictNext(snapshot, [CLICK]);
    expect(fromDisk).toEqual(fromMemory);
    expect(fromDisk.token).toBe(TYPE);
  });

  it("returns undefined when no snapshot has been saved", async () => {
    const trainer = new MovementModelTrainer(rootDir);
    expect(await trainer.load("models/missing.json")).toBeUndefined();
  });
});
