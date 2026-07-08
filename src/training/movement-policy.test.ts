import { describe, expect, it } from "vitest";
import {
  MarkovMovementBackend,
  MOVEMENT_END_TOKEN,
  createMovementBackend,
  evaluateMovementFidelity,
  movementSequenceFromReplay,
  movementSequenceFromTrajectory,
  type MovementDataset,
} from "./movement-policy.js";
import type { ReplayManifest } from "../capture/replay.js";
import type { TrajectorySpan } from "../capture/trajectory.js";

function dataset(sequences: Array<{ id: string; tokens: string[]; reward?: number }>): MovementDataset {
  return { sequences };
}

describe("MarkovMovementBackend", () => {
  it("repeats a recorded movement exactly from a seed prefix", async () => {
    const backend = new MarkovMovementBackend();
    const policy = await backend.train(
      dataset([{ id: "t1", tokens: ["move:a", "click:b", "type:c", "submit:d"] }]),
    );

    const generation = policy.generate(["move:a"], { maxSteps: 16 });
    expect(generation.tokens).toEqual(["move:a", "click:b", "type:c", "submit:d"]);
    expect(generation.generated).toEqual(["click:b", "type:c", "submit:d"]);
    expect(generation.stopped).toBe("end");
  });

  it("generalizes to a new-but-related prefix via backoff", async () => {
    // Two recorded flows that share a common suffix behaviour: after "focus:field"
    // the movement is always "type:text" then "press:enter".
    const policy = await createMovementBackend("markov", { order: 2 }).train(
      dataset([
        { id: "login", tokens: ["open:login", "focus:field", "type:text", "press:enter"] },
        { id: "search", tokens: ["open:search", "focus:field", "type:text", "press:enter"] },
      ]),
    );

    // A prefix never seen verbatim ("open:settings") but ending in a known
    // context should still predict the learned continuation.
    const generation = policy.generate(["open:settings", "focus:field"], { maxSteps: 8 });
    expect(generation.generated).toEqual(["type:text", "press:enter"]);
  });

  it("ranks candidates with a deterministic tie-break", async () => {
    const policy = await new MarkovMovementBackend().train(
      dataset([
        { id: "a", tokens: ["start", "z"] },
        { id: "b", tokens: ["start", "a"] },
      ]),
    );
    const ranked = policy.rankNext(["start"]);
    // Equal counts → alphabetical tie-break ("a" before "z").
    expect(ranked.map((c) => c.token)).toEqual(["a", "z"]);
    expect(ranked[0]?.probability).toBeCloseTo(0.5);
  });

  it("reports metadata and excludes the end sentinel from vocab", async () => {
    const policy = await new MarkovMovementBackend().train(
      dataset([{ id: "a", tokens: ["x", "y", "z"] }]),
    );
    expect(policy.metadata.sequenceCount).toBe(1);
    expect(policy.metadata.vocabSize).toBe(3);
    expect(policy.metadata.kind).toBe("markov");
    expect(policy.metadata.transitionCount).toBeGreaterThan(0);
  });

  it("weights sequences by reward", async () => {
    const policy = await new MarkovMovementBackend().train(
      dataset([
        { id: "reinforced", tokens: ["start", "good"], reward: 4 },
        { id: "neutral", tokens: ["start", "bad"] },
      ]),
    );
    // The rewarded continuation should win despite equal sequence counts.
    expect(policy.predictNext(["start"])?.token).toBe("good");
  });

  it("round-trips through serialize/load", async () => {
    const backend = new MarkovMovementBackend();
    const policy = await backend.train(
      dataset([{ id: "t", tokens: ["a", "b", "c"] }]),
    );
    const restored = backend.load(policy.serialize());
    expect(restored.generate(["a"]).tokens).toEqual(["a", "b", "c"]);
    expect(restored.metadata.vocabSize).toBe(policy.metadata.vocabSize);
  });

  it("stops with no-prediction on an empty model", async () => {
    const policy = await new MarkovMovementBackend().train(dataset([]));
    const generation = policy.generate(["anything"]);
    expect(generation.stopped).toBe("no-prediction");
    expect(generation.generated).toEqual([]);
  });

  it("exposes the end sentinel constant", () => {
    expect(MOVEMENT_END_TOKEN).toBeTypeOf("string");
  });
});

describe("movement extraction helpers", () => {
  it("extracts action tokens from a replay manifest by default", () => {
    const manifest: ReplayManifest = {
      version: 1,
      sessionId: "s1",
      trajectoryIds: ["t1"],
      eventCount: 3,
      events: [
        { kind: "observation", ts: 1, trajectoryId: "t1", source: "screen", summary: "" },
        { kind: "action", ts: 2, trajectoryId: "t1", tool: "mouse.move", summary: "" },
        { kind: "action", ts: 3, trajectoryId: "t1", tool: "mouse.click", summary: "" },
      ],
    };
    const sequence = movementSequenceFromReplay(manifest);
    expect(sequence.tokens).toEqual(["action:mouse.move", "action:mouse.click"]);
  });

  it("includes observations when requested", () => {
    const manifest: ReplayManifest = {
      version: 1,
      sessionId: "s1",
      trajectoryIds: ["t1"],
      eventCount: 2,
      events: [
        { kind: "observation", ts: 1, trajectoryId: "t1", source: "screen", summary: "" },
        { kind: "action", ts: 2, trajectoryId: "t1", tool: "mouse.move", summary: "" },
      ],
    };
    const sequence = movementSequenceFromReplay(manifest, { include: ["observation", "action"] });
    expect(sequence.tokens).toEqual(["observation:screen", "action:mouse.move"]);
  });

  it("extracts action tokens and reward from a trajectory span", () => {
    const trajectory: TrajectorySpan = {
      id: "t1",
      sessionId: "s1",
      createdAt: "2026-07-08T00:00:00.000Z",
      captureTier: "operator",
      observations: [],
      actions: [
        { kind: "action", tool: "key.press", summary: "", ts: 1 },
        { kind: "action", tool: "key.release", summary: "", ts: 2 },
      ],
      outcome: { status: "success", summary: "done", reward: 2 },
    };
    const sequence = movementSequenceFromTrajectory(trajectory);
    expect(sequence.tokens).toEqual(["action:key.press", "action:key.release"]);
    expect(sequence.reward).toBe(2);
  });
});

describe("evaluateMovementFidelity", () => {
  it("scores full fidelity for learned sequences and partial for novel ones", async () => {
    const train = dataset([
      { id: "flow", tokens: ["a", "b", "c", "d"] },
    ]);
    const policy = await createMovementBackend("markov", { order: 3 }).train(train);

    const held = [
      { id: "flow", tokens: ["a", "b", "c", "d"] }, // seen → fidelity 1
      { id: "novel", tokens: ["a", "x", "y"] }, // diverges immediately → 0
    ];
    const result = evaluateMovementFidelity(policy, held, { seedLength: 1 });
    expect(result.perSequence.find((s) => s.id === "flow")?.fidelity).toBe(1);
    expect(result.perSequence.find((s) => s.id === "novel")?.fidelity).toBe(0);
    expect(result.meanFidelity).toBeCloseTo(0.5);
  });
});

describe("createMovementBackend", () => {
  it("throws on an unknown backend kind", () => {
    expect(() => createMovementBackend("mlx" as never)).toThrow(/Unknown movement backend/);
  });
});
