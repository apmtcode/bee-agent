import { describe, expect, it } from "vitest";
import type { ReplayManifest } from "../capture/replay.js";
import {
  MarkovMovementBackend,
  buildMovementDataset,
  trainMovementModelFromReplays,
  type MovementModelBackend,
  type MovementDataset,
} from "./movement-model.js";

function manifest(sessionId: string, trajectoryId: string, steps: Array<
  | { obs: string }
  | { tool: string; summary: string }
>): ReplayManifest {
  const events = steps.map((step, index) =>
    "obs" in step
      ? ({ kind: "observation", ts: index + 1, trajectoryId, source: step.obs, summary: `saw ${step.obs}` } as const)
      : ({ kind: "action", ts: index + 1, trajectoryId, tool: step.tool, summary: step.summary } as const),
  );
  return {
    version: 1,
    sessionId,
    trajectoryIds: [trajectoryId],
    eventCount: events.length,
    events,
  };
}

describe("buildMovementDataset", () => {
  it("groups actions by trajectory in timestamp order and keeps observation context", () => {
    const dataset = buildMovementDataset([
      manifest("s1", "t1", [
        { obs: "editor" },
        { tool: "mouse", summary: "move to file tree" },
        { tool: "mouse", summary: "click main.ts" },
        { tool: "keyboard", summary: "type edit" },
      ]),
    ]);

    expect(dataset.sequences).toHaveLength(1);
    const sequence = dataset.sequences[0]!;
    expect(sequence.trajectoryId).toBe("t1");
    expect(sequence.contextSources).toEqual(["editor"]);
    expect(sequence.actions).toEqual([
      { tool: "mouse", summary: "move to file tree" },
      { tool: "mouse", summary: "click main.ts" },
      { tool: "keyboard", summary: "type edit" },
    ]);
    // Vocabulary is sorted + de-duplicated over the composite token key.
    expect(dataset.vocabulary).toHaveLength(3);
  });

  it("orders actions by timestamp even when replay events arrive out of order", () => {
    const dataset = buildMovementDataset([
      {
        version: 1,
        sessionId: "s1",
        trajectoryIds: ["t1"],
        eventCount: 2,
        events: [
          { kind: "action", ts: 5, trajectoryId: "t1", tool: "keyboard", summary: "second" },
          { kind: "action", ts: 2, trajectoryId: "t1", tool: "mouse", summary: "first" },
        ],
      },
    ]);
    expect(dataset.sequences[0]!.actions.map((a) => a.summary)).toEqual(["first", "second"]);
  });

  it("ignores transcript events (they are not movements)", () => {
    const dataset = buildMovementDataset([
      {
        version: 1,
        sessionId: "s1",
        trajectoryIds: ["t1"],
        eventCount: 2,
        events: [
          { kind: "transcript", ts: 1, messageId: "m1", role: "user", content: "hi" },
          { kind: "action", ts: 2, trajectoryId: "t1", tool: "mouse", summary: "click" },
        ],
      },
    ]);
    expect(dataset.sequences).toHaveLength(1);
    expect(dataset.sequences[0]!.actions).toHaveLength(1);
  });
});

describe("MarkovMovementBackend", () => {
  const recorded = manifest("s1", "t1", [
    { obs: "editor" },
    { tool: "mouse", summary: "open palette" },
    { tool: "keyboard", summary: "type build" },
    { tool: "keyboard", summary: "press enter" },
  ]);

  it("reproduces a recorded movement sequence exactly (repeat)", async () => {
    const { model } = await trainMovementModelFromReplays([recorded]);
    const generated = model.generate();
    expect(generated.terminatedByModel).toBe(true);
    expect(generated.actions).toEqual([
      { tool: "mouse", summary: "open palette" },
      { tool: "keyboard", summary: "type build" },
      { tool: "keyboard", summary: "press enter" },
    ]);
  });

  it("scores full replay fidelity on the training sequence", async () => {
    const { dataset, model } = await trainMovementModelFromReplays([recorded]);
    const report = model.scoreFidelity(dataset.sequences[0]!.actions);
    expect(report.fidelity).toBe(1);
    expect(report.matchedSteps).toBe(3);
    expect(report.mismatches).toHaveLength(0);
  });

  it("generalizes to a novel context via backoff (perform new but related movement)", async () => {
    // Two trajectories share the prefix [A, B]; after B, "D" is more frequent
    // than "C". A context ending in B that was never seen at full order still
    // predicts the most frequent shorter-context continuation.
    const dataset: MovementDataset = buildMovementDataset([
      manifest("s1", "t1", [
        { tool: "mouse", summary: "A" },
        { tool: "mouse", summary: "B" },
        { tool: "mouse", summary: "C" },
      ]),
      manifest("s2", "t2", [
        { tool: "key", summary: "X" },
        { tool: "mouse", summary: "B" },
        { tool: "mouse", summary: "D" },
      ]),
      manifest("s3", "t3", [
        { tool: "key", summary: "Y" },
        { tool: "mouse", summary: "B" },
        { tool: "mouse", summary: "D" },
      ]),
    ]);
    const model = await new MarkovMovementBackend(2).train(dataset);

    // Novel prefix [Z, B]: the bigram (Z,B) was never seen, so the model backs
    // off to the unigram-context "after B" and picks the most frequent: D.
    const prediction = model.predictNext([
      { tool: "mouse", summary: "Z" },
      { tool: "mouse", summary: "B" },
    ]);
    expect(prediction?.kind).toBe("action");
    if (prediction?.kind === "action") {
      expect(prediction.summary).toBe("D");
      expect(prediction.backoffOrder).toBeGreaterThan(0);
    }
  });

  it("makes deterministic tie-broken predictions", async () => {
    const { model } = await trainMovementModelFromReplays([recorded]);
    const first = model.predictNext([]);
    const second = model.predictNext([]);
    expect(first).toEqual(second);
    expect(first?.kind).toBe("action");
  });

  it("round-trips through JSON serialization", async () => {
    const { model } = await trainMovementModelFromReplays([recorded]);
    const serialized = model.toJSON();
    expect(serialized.backendId).toBe("markov-movement-v1");
    expect(serialized.order).toBe(2);
    expect(serialized.transitions.length).toBeGreaterThan(0);
    // Serialization is stable/sorted, so it can be diffed across runs.
    expect(JSON.stringify(model.toJSON())).toBe(JSON.stringify(model.toJSON()));
  });

  it("honours the maxSteps guard against non-terminating rollout", async () => {
    // A self-loop: after "spin" the most frequent next action is "spin" again.
    const dataset = buildMovementDataset([
      manifest("s1", "t1", [
        { tool: "mouse", summary: "spin" },
        { tool: "mouse", summary: "spin" },
        { tool: "mouse", summary: "spin" },
      ]),
    ]);
    const model = await new MarkovMovementBackend(1).train(dataset);
    const generated = model.generate({ maxSteps: 4 });
    expect(generated.steps).toBeLessThanOrEqual(4);
  });

  it("accepts an alternative pluggable backend implementation", async () => {
    const constantBackend: MovementModelBackend = {
      id: "constant-test-backend",
      async train() {
        return {
          backendId: "constant-test-backend",
          order: 0,
          vocabulary: [],
          predictNext: () => ({ kind: "action", tool: "noop", summary: "noop", probability: 1, backoffOrder: 0 }),
          generate: () => ({ actions: [], terminatedByModel: true, steps: 0 }),
          scoreFidelity: (reference) => ({
            referenceLength: reference.length,
            matchedSteps: 0,
            fidelity: reference.length === 0 ? 1 : 0,
            mismatches: [],
          }),
          toJSON: () => ({ version: 1, backendId: "constant-test-backend", order: 0, vocabulary: [], transitions: [] }),
        };
      },
    };
    const { model } = await trainMovementModelFromReplays([recorded], constantBackend);
    expect(model.backendId).toBe("constant-test-backend");
    expect(model.predictNext([])).toEqual({
      kind: "action",
      tool: "noop",
      summary: "noop",
      probability: 1,
      backoffOrder: 0,
    });
  });
});
