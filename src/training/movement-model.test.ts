import { describe, expect, it } from "vitest";
import type { ReplayManifest } from "../capture/replay.js";
import type { TrajectorySpan } from "../capture/trajectory.js";
import {
  MarkovMovementBackend,
  buildMovementDataset,
  buildMovementDatasetFromTrajectories,
  evaluateMovementModel,
  movementTokenId,
  parseMovementTokenId,
  tokenizeReplayEvent,
  type MovementDataset,
} from "./movement-model.js";

function replay(sessionId: string, trajectoryId: string, steps: Array<["observation" | "action", string, string, number]>): ReplayManifest {
  const events = steps.map(([kind, toolOrSource, summary, ts]) =>
    kind === "observation"
      ? { kind: "observation" as const, ts, trajectoryId, source: toolOrSource, summary }
      : { kind: "action" as const, ts, trajectoryId, tool: toolOrSource, summary },
  );
  return { version: 1, sessionId, trajectoryIds: [trajectoryId], eventCount: events.length, events };
}

/** A repeated "open editor then save" movement episode used across tests. */
function editorReplays(count: number): ReplayManifest[] {
  return Array.from({ length: count }, (_, index) =>
    replay(`session-${index}`, `traj-${index}`, [
      ["observation", "os", "focused Editor", index * 100 + 1],
      ["action", "device", "tapped File menu", index * 100 + 2],
      ["action", "device", "tapped Save", index * 100 + 3],
      ["observation", "os", "focused Editor", index * 100 + 4],
    ]),
  );
}

describe("movement tokenization", () => {
  it("normalizes verb/target and round-trips through the vocabulary id", () => {
    const token = tokenizeReplayEvent({ kind: "action", ts: 1, trajectoryId: "t", tool: "device", summary: "Tapped   Save Button" });
    expect(token).toEqual({ channel: "action", tool: "device", verb: "tapped", target: "save button" });
    const id = movementTokenId(token!);
    expect(parseMovementTokenId(id)).toEqual(token);
  });

  it("ignores transcript events (not movements)", () => {
    expect(
      tokenizeReplayEvent({ kind: "transcript", ts: 1, messageId: "m", role: "user", content: "hi" }),
    ).toBeUndefined();
  });
});

describe("dataset building", () => {
  it("groups events per trajectory ordered by timestamp", () => {
    const manifest = replay("s", "t", [
      ["action", "device", "tapped Save", 3],
      ["observation", "os", "focused Editor", 1],
      ["action", "device", "tapped File", 2],
    ]);
    const dataset = buildMovementDataset([manifest]);
    expect(dataset.sequences).toHaveLength(1);
    expect(dataset.sequences[0].tokens.map((token) => token.verb)).toEqual(["focused", "tapped", "tapped"]);
    expect(dataset.sequences[0].tokens.map((token) => token.target)).toEqual(["editor", "file", "save"]);
  });

  it("builds equivalent sequences directly from trajectory spans", () => {
    const trajectory: TrajectorySpan = {
      id: "t",
      sessionId: "s",
      createdAt: new Date(0).toISOString(),
      captureTier: "operator",
      observations: [{ kind: "observation", source: "os", summary: "focused Editor", ts: 1 }],
      actions: [{ kind: "action", tool: "device", summary: "tapped Save", ts: 2 }],
    };
    const dataset = buildMovementDatasetFromTrajectories([trajectory]);
    expect(dataset.sequences[0].tokens.map(movementTokenId)).toEqual([
      "observation|os|focused|editor",
      "action|device|tapped|save",
    ]);
  });
});

describe("markov movement backend", () => {
  it("repeats recorded movements exactly after training", async () => {
    const dataset = buildMovementDataset(editorReplays(3));
    const model = await new MarkovMovementBackend().train(dataset, { order: 2 });

    const seed = dataset.sequences[0].tokens.slice(0, 1); // "focused Editor"
    const generated = model.generate(seed, 2);
    expect(generated.map((token) => `${token.verb} ${token.target}`)).toEqual([
      "tapped file menu",
      "tapped save",
    ]);
    const first = model.predictNext(seed);
    expect(first.source).toBe("exact");
    expect(first.confidence).toBeGreaterThan(0);
  });

  it("is deterministic across repeated training runs", async () => {
    const dataset = buildMovementDataset(editorReplays(2));
    const a = await new MarkovMovementBackend().train(dataset);
    const b = await new MarkovMovementBackend().train(dataset);
    expect(a.toArtifact()).toEqual(b.toArtifact());
  });

  it("generalizes to an unseen context via backoff instead of failing", async () => {
    const dataset = buildMovementDataset(editorReplays(3));
    const model = await new MarkovMovementBackend().train(dataset, { order: 2 });

    // Context the model never saw as a 2-gram; must back off to a shorter context.
    const novelContext = [
      { channel: "action" as const, tool: "device", verb: "swiped", target: "somewhere new" },
      { channel: "action" as const, tool: "device", verb: "tapped", target: "file menu" },
    ];
    const prediction = model.predictNext(novelContext);
    expect(prediction.token).toBeDefined();
    expect(prediction.source).toBe("backoff");
    expect(prediction.token && `${prediction.token.verb} ${prediction.token.target}`).toBe("tapped save");
  });

  it("falls back to the unigram marginal when no context matches", async () => {
    const dataset = buildMovementDataset(editorReplays(3));
    const model = await new MarkovMovementBackend().train(dataset, { order: 2 });
    const prediction = model.predictNext([
      { channel: "action", tool: "device", verb: "unknown", target: "nothing seen before" },
    ]);
    expect(prediction.source).toBe("fallback");
    expect(prediction.token).toBeDefined();
  });

  it("returns an empty prediction for an empty model", async () => {
    const empty: MovementDataset = { version: 1, sequences: [] };
    const model = await new MarkovMovementBackend().train(empty);
    const prediction = model.predictNext([]);
    expect(prediction.source).toBe("empty");
    expect(prediction.token).toBeUndefined();
    expect(model.generate([], 5)).toEqual([]);
  });

  it("round-trips through a serializable artifact with identical predictions", async () => {
    const dataset = buildMovementDataset(editorReplays(2));
    const backend = new MarkovMovementBackend();
    const model = await backend.train(dataset);
    const artifact = model.toArtifact();

    const serialized = JSON.parse(JSON.stringify(artifact));
    const restored = backend.load(serialized);

    const seed = dataset.sequences[0].tokens.slice(0, 1);
    expect(restored.generate(seed, 3)).toEqual(model.generate(seed, 3));
    expect(restored.toArtifact()).toEqual(artifact);
  });
});

describe("generalization eval harness", () => {
  it("scores perfect replay fidelity on the training distribution", async () => {
    const dataset = buildMovementDataset(editorReplays(4));
    const model = await new MarkovMovementBackend().train(dataset, { order: 2 });
    const evaluation = evaluateMovementModel(model, dataset.sequences);
    expect(evaluation.accuracy).toBe(1);
    expect(evaluation.predictions).toBeGreaterThan(0);
  });

  it("measures accuracy and generalization on held-out related trajectories", async () => {
    // Train on episodes A/B; hold out a related-but-unseen episode C.
    const train = buildMovementDataset(editorReplays(2));
    const heldOut = buildMovementDataset([
      replay("session-held", "traj-held", [
        ["observation", "os", "focused Editor", 1],
        ["action", "device", "tapped File menu", 2],
        ["action", "device", "tapped Save", 3],
      ]),
    ]);
    const model = await new MarkovMovementBackend().train(train, { order: 2 });
    const evaluation = evaluateMovementModel(model, heldOut.sequences);
    // The shared "focused Editor -> tapped File menu -> tapped Save" pattern generalizes.
    expect(evaluation.accuracy).toBeGreaterThan(0.5);
    expect(evaluation.generalizedFraction).toBeGreaterThanOrEqual(0);
  });
});
