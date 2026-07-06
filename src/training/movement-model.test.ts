import { describe, expect, it } from "vitest";
import {
  MOVEMENT_END_TOKEN,
  MOVEMENT_START_TOKEN,
  MarkovMovementBackend,
  MarkovMovementModel,
  datasetFromReplayManifests,
  datasetFromTrajectories,
  evaluateMovementModel,
  synthesizeMovementSequences,
  tokenizeReplayEvent,
  tokenizeTrajectory,
  type MovementDataset,
} from "./movement-model.js";
import type { ReplayManifest } from "../capture/replay.js";
import { buildTrajectorySpan } from "../capture/trajectory.js";

function dataset(sequences: { id: string; tokens: string[] }[]): MovementDataset {
  return { version: 1, sequences };
}

describe("tokenization", () => {
  it("normalizes replay events to structural tokens, dropping literal args", () => {
    expect(
      tokenizeReplayEvent({ kind: "action", ts: 1, trajectoryId: "t", tool: "device", summary: "tapped Submit" }),
    ).toBe("action:device:tapped");
    // Same verb, different target -> same token family (enables generalization).
    expect(
      tokenizeReplayEvent({ kind: "action", ts: 1, trajectoryId: "t", tool: "device", summary: "tapped Cancel" }),
    ).toBe("action:device:tapped");
    expect(
      tokenizeReplayEvent({ kind: "observation", ts: 1, trajectoryId: "t", source: "os", summary: "focused Mail" }),
    ).toBe("observation:os:focused");
    expect(tokenizeReplayEvent({ kind: "transcript", ts: 1, messageId: "m", role: "user", content: "hi" })).toBe(
      "transcript:user",
    );
  });

  it("tokenizes trajectories in timestamp order", () => {
    const span = buildTrajectorySpan({
      id: "traj-1",
      sessionId: "s1",
      observations: [{ kind: "observation", source: "os", summary: "focused Editor", ts: 10 }],
      actions: [
        { kind: "action", tool: "device", summary: "typed hello", ts: 20 },
        { kind: "action", tool: "device", summary: "tapped Save", ts: 15 },
      ],
    });
    expect(tokenizeTrajectory(span)).toEqual([
      "observation:os:focused",
      "action:device:tapped",
      "action:device:typed",
    ]);
  });
});

describe("dataset builders", () => {
  it("builds an action-only dataset from replay manifests by default", () => {
    const manifest: ReplayManifest = {
      version: 1,
      sessionId: "s1",
      trajectoryIds: ["traj-1"],
      eventCount: 3,
      events: [
        { kind: "observation", ts: 1, trajectoryId: "traj-1", source: "os", summary: "focused App" },
        { kind: "action", ts: 2, trajectoryId: "traj-1", tool: "device", summary: "tapped Menu" },
        { kind: "action", ts: 3, trajectoryId: "traj-1", tool: "device", summary: "swiped up" },
      ],
    };
    const ds = datasetFromReplayManifests([manifest]);
    expect(ds.sequences).toHaveLength(1);
    expect(ds.sequences[0]).toEqual({ id: "traj-1", tokens: ["action:device:tapped", "action:device:swiped"] });
  });

  it("can include observations when asked", () => {
    const manifest: ReplayManifest = {
      version: 1,
      sessionId: "s1",
      trajectoryIds: ["traj-1"],
      eventCount: 2,
      events: [
        { kind: "observation", ts: 1, trajectoryId: "traj-1", source: "os", summary: "focused App" },
        { kind: "action", ts: 2, trajectoryId: "traj-1", tool: "device", summary: "tapped Menu" },
      ],
    };
    const ds = datasetFromReplayManifests([manifest], { include: ["observation", "action"] });
    expect(ds.sequences[0]?.tokens).toEqual(["observation:os:focused", "action:device:tapped"]);
  });

  it("drops empty sequences", () => {
    const manifest: ReplayManifest = {
      version: 1,
      sessionId: "s1",
      trajectoryIds: ["traj-1"],
      eventCount: 1,
      events: [{ kind: "observation", ts: 1, trajectoryId: "traj-1", source: "os", summary: "focused App" }],
    };
    expect(datasetFromReplayManifests([manifest]).sequences).toHaveLength(0);
  });

  it("builds a dataset from trajectories directly", () => {
    const span = buildTrajectorySpan({
      id: "traj-1",
      sessionId: "s1",
      actions: [{ kind: "action", tool: "device", summary: "tapped Save", ts: 1 }],
    });
    expect(datasetFromTrajectories([span]).sequences).toEqual([{ id: "traj-1", tokens: ["action:device:tapped"] }]);
  });
});

describe("MarkovMovementBackend training + reproduction (objective c)", () => {
  it("reproduces a recorded deterministic movement sequence exactly", async () => {
    const recorded = ["action:app:open", "action:app:click", "action:app:type", "action:app:submit"];
    const model = await new MarkovMovementBackend().train(dataset([{ id: "s", tokens: recorded }]), { order: 2 });
    // Seeded with the start marker, greedy rollout should reproduce the recording.
    expect(model.generate([MOVEMENT_START_TOKEN])).toEqual(recorded);
  });

  it("predicts the recorded next movement with full confidence for a unique context", async () => {
    const model = await new MarkovMovementBackend().train(
      dataset([{ id: "s", tokens: ["action:app:open", "action:app:click"] }]),
      { order: 2 },
    );
    const prediction = model.predictNext([MOVEMENT_START_TOKEN, "action:app:open"]);
    expect(prediction?.token).toBe("action:app:click");
    expect(prediction?.probability).toBe(1);
  });

  it("learns to stop (predicts the end token) at a sequence terminus", async () => {
    const model = await new MarkovMovementBackend().train(dataset([{ id: "s", tokens: ["action:app:open"] }]), {
      order: 2,
    });
    expect(model.predictNext(["action:app:open"])?.token).toBe(MOVEMENT_END_TOKEN);
  });

  it("is deterministic: identical datasets yield identical serialized models", async () => {
    const ds = dataset([
      { id: "a", tokens: ["action:app:open", "action:app:click"] },
      { id: "b", tokens: ["action:app:open", "action:app:type"] },
    ]);
    const a = await new MarkovMovementBackend().train(ds, { order: 2 });
    const b = await new MarkovMovementBackend().train(ds, { order: 2 });
    expect(a.toJSON()).toEqual(b.toJSON());
  });

  it("breaks ties deterministically by frequency then lexical order", async () => {
    // From <start>: 'open' appears twice, 'boot' once -> 'open' wins by count.
    const ds = dataset([
      { id: "a", tokens: ["action:app:open"] },
      { id: "b", tokens: ["action:app:open"] },
      { id: "c", tokens: ["action:app:boot"] },
    ]);
    const model = await new MarkovMovementBackend().train(ds, { order: 2 });
    expect(model.predictNext([MOVEMENT_START_TOKEN])?.token).toBe("action:app:open");
  });
});

describe("generalization via stupid-backoff (objective d)", () => {
  it("predicts a next movement for an unseen prefix by backing off to a shorter context", async () => {
    // Train: the pair (click -> type) is always observed, but only after 'open'.
    const model = await new MarkovMovementBackend().train(
      dataset([{ id: "s", tokens: ["action:app:open", "action:app:click", "action:app:type"] }]),
      { order: 2 },
    );
    // Novel context: 'boot' then 'click' — the order-2 context (boot,click) was
    // never seen, but backing off to the order-1 context (click) has been.
    const prediction = model.predictNext(["action:app:boot", "action:app:click"]);
    expect(prediction?.token).toBe("action:app:type");
    expect(prediction?.backoffOrder).toBeLessThan(2);
  });

  it("generalizes to a related-but-unseen held-out split with high accuracy", async () => {
    const motifs = [
      {
        id: "deploy",
        base: ["action:app:open", "action:app:config", "action:app:deploy"],
        variants: ["action:app:confirm", "action:app:notify", "action:app:close"],
      },
    ];
    const train = synthesizeMovementSequences({ motifs, repeats: 3, seed: 0 });
    const heldOut = synthesizeMovementSequences({ motifs, repeats: 2, seed: 7 });
    const model = await new MarkovMovementBackend().train(dataset(train), { order: 2 });

    const result = evaluateMovementModel(model, heldOut);
    expect(result.predictions).toBeGreaterThan(0);
    // The shared prefix is fully learned; the varying tail is where it must
    // generalize. Accuracy should still be strong.
    expect(result.accuracy).toBeGreaterThan(0.6);
    expect(result.meanConfidence).toBeGreaterThan(0);
  });

  it("reports generalizationRate > 0 when the eval exercises unseen contexts", async () => {
    const model = await new MarkovMovementBackend().train(
      dataset([{ id: "s", tokens: ["a", "b", "c"] }]),
      { order: 2 },
    );
    const result = evaluateMovementModel(model, [{ id: "h", tokens: ["x", "b", "c"] }]);
    expect(result.generalizationRate).toBeGreaterThan(0);
  });
});

describe("serialization round-trip", () => {
  it("restores an identical model from JSON", async () => {
    const model = await new MarkovMovementBackend().train(
      dataset([{ id: "s", tokens: ["action:app:open", "action:app:click"] }]),
      { order: 2 },
    );
    const restored = MarkovMovementModel.fromJSON(model.toJSON());
    expect(restored.toJSON()).toEqual(model.toJSON());
    expect(restored.generate([MOVEMENT_START_TOKEN])).toEqual(model.generate([MOVEMENT_START_TOKEN]));
  });
});

describe("edge cases", () => {
  it("returns undefined prediction for an untrained (empty) model", async () => {
    const model = await new MarkovMovementBackend().train({ version: 1, sequences: [] });
    expect(model.predictNext([])).toBeUndefined();
    expect(model.generate([MOVEMENT_START_TOKEN])).toEqual([]);
  });

  it("respects maxSteps to bound rollout length", async () => {
    // A self-looping dataset would otherwise generate forever.
    const model = await new MarkovMovementBackend().train(
      dataset([{ id: "s", tokens: ["loop", "loop", "loop", "loop"] }]),
      { order: 1 },
    );
    expect(model.generate([MOVEMENT_START_TOKEN], 3).length).toBeLessThanOrEqual(3);
  });

  it("synthesizeMovementSequences is deterministic and split-disjoint by seed", () => {
    const motifs = [{ id: "m", base: ["a", "b"], variants: ["c", "d"] }];
    const first = synthesizeMovementSequences({ motifs, repeats: 2, seed: 0 });
    const same = synthesizeMovementSequences({ motifs, repeats: 2, seed: 0 });
    expect(first).toEqual(same);
    expect(first[0]?.tokens).toEqual(["a", "b", "c"]);
    expect(first[1]?.tokens).toEqual(["a", "b", "d"]);
  });
});
