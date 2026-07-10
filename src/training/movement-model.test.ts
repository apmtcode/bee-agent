import { describe, expect, it } from "vitest";
import type { ReplayTimelineEvent } from "../capture/replay.js";
import { buildMovementDataset, splitMovementDataset } from "./movement-dataset.js";
import {
  DeterministicMarkovMovementBackend,
  MovementModelTrainer,
} from "./movement-model.js";

/**
 * Synthetic movement generator: a workflow of `open editor -> run build -> read
 * output`, repeatable so training sees the same transitions many times. This
 * doubles as the synthetic event-stream generator the roadmap calls for.
 */
function buildWorkflow(sessionId: string, trajectoryId: string, base: number): ReplayTimelineEvent[] {
  return [
    { kind: "observation", ts: base + 1, trajectoryId, source: "device", summary: "editor active" },
    { kind: "action", ts: base + 2, trajectoryId, tool: "device", summary: "opened terminal" },
    { kind: "observation", ts: base + 3, trajectoryId, source: "device", summary: "terminal focused" },
    { kind: "action", ts: base + 4, trajectoryId, tool: "device", summary: "typed npm build" },
    { kind: "observation", ts: base + 5, trajectoryId, source: "device", summary: "build output shown" },
    { kind: "action", ts: base + 6, trajectoryId, tool: "device", summary: "scrolled output" },
  ];
}

function replays(count: number) {
  return Array.from({ length: count }, (_, index) => ({
    sessionId: `s${index}`,
    trajectoryIds: [`t${index}`],
    events: buildWorkflow(`s${index}`, `t${index}`, index * 100),
  }));
}

describe("DeterministicMarkovMovementBackend", () => {
  it("trains a serializable model and reproduces recorded movements exactly", async () => {
    const dataset = buildMovementDataset(replays(3));
    const backend = new DeterministicMarkovMovementBackend();
    const trainer = new MovementModelTrainer(backend);

    const model = await trainer.train(dataset, { label: "editor-build-flow" });
    expect(model.backendId).toBe("markov-v1");
    expect(model.label).toBe("editor-build-flow");
    expect(model.trainedTransitions).toBe(dataset.transitionCount);
    expect(model.vocabulary).toContain("device::typed npm build");
    // model params must round-trip through JSON (on-device persistence contract)
    expect(() => JSON.parse(JSON.stringify(model.parameters))).not.toThrow();

    const report = trainer.evaluate(model, dataset);
    expect(report.totalTransitions).toBe(dataset.transitionCount);
    expect(report.fidelity).toBe(1);
    expect(report.exactMatches).toBe(dataset.transitionCount);
    expect(report.misses).toBe(0);
  });

  it("is deterministic: identical training yields identical parameters", async () => {
    const dataset = buildMovementDataset(replays(2));
    const backend = new DeterministicMarkovMovementBackend();
    const a = await backend.train(dataset);
    const b = await backend.train(dataset);
    expect(JSON.stringify(a.parameters)).toBe(JSON.stringify(b.parameters));
    expect(a.vocabulary).toEqual(b.vocabulary);
  });

  it("breaks ties lexicographically and reports calibrated confidence", async () => {
    // Same context ("<start>" -> first action) leads to two competing actions;
    // "aaa" appears twice, "zzz" once, so argmax picks "aaa" with 2/3 confidence.
    const dataset = buildMovementDataset([
      { sessionId: "s1", trajectoryIds: ["t1"], events: [{ kind: "action", ts: 1, trajectoryId: "t1", tool: "device", summary: "aaa" }] },
      { sessionId: "s2", trajectoryIds: ["t2"], events: [{ kind: "action", ts: 1, trajectoryId: "t2", tool: "device", summary: "aaa" }] },
      { sessionId: "s3", trajectoryIds: ["t3"], events: [{ kind: "action", ts: 1, trajectoryId: "t3", tool: "device", summary: "zzz" }] },
    ]);
    const backend = new DeterministicMarkovMovementBackend();
    const model = await backend.train(dataset);
    const session = backend.load(model);
    const prediction = session.predictNext("<start>");
    expect(prediction?.action.summary).toBe("aaa");
    expect(prediction?.confidence).toBeCloseTo(2 / 3);
    expect(prediction?.backoff).toBe(false);
  });

  it("generalizes to a related-but-unseen context via backoff", async () => {
    const dataset = buildMovementDataset(replays(3));
    const backend = new DeterministicMarkovMovementBackend();
    const model = await backend.train(dataset);
    const session = backend.load(model);

    // A novel full-window context whose most-recent event ("terminal focused")
    // WAS seen in training: exact match misses, backoff recovers the learned
    // "typed npm build" action.
    const prediction = session.predictFromEvents([
      { kind: "transcript", ts: 1, messageId: "m", role: "user", content: "some never-before-seen instruction" },
      { kind: "observation", ts: 2, trajectoryId: "t", source: "device", summary: "terminal focused" },
    ]);
    expect(prediction).toBeDefined();
    expect(prediction?.backoff).toBe(true);
    expect(prediction?.action.summary).toBe("typed npm build");
  });

  it("returns undefined when nothing in the model matches", async () => {
    const dataset = buildMovementDataset(replays(1));
    const backend = new DeterministicMarkovMovementBackend();
    const model = await backend.train(dataset);
    const session = backend.load(model);
    expect(
      session.predictFromEvents([
        { kind: "observation", ts: 1, trajectoryId: "t", source: "totally", summary: "unknown state" },
      ]),
    ).toBeUndefined();
  });

  it("refuses to load a model trained by a different backend", async () => {
    const backend = new DeterministicMarkovMovementBackend();
    expect(() =>
      backend.load({
        backendId: "other",
        version: 1,
        contextWindow: 2,
        trainedTransitions: 0,
        vocabulary: [],
        parameters: { kind: "some-other-backend" },
      }),
    ).toThrow(/cannot load/);
  });
});

describe("MovementModelTrainer generalization eval", () => {
  it("measures fidelity on a held-out split of related trajectories", async () => {
    const dataset = buildMovementDataset(replays(6));
    const { train, holdout } = splitMovementDataset(dataset, 3);
    const trainer = new MovementModelTrainer(new DeterministicMarkovMovementBackend());
    const model = await trainer.train(train);

    const report = trainer.evaluate(model, holdout);
    // Held-out sequences follow the same workflow, so the model reproduces every
    // action it never trained on — full generalization on this repeated flow.
    expect(report.totalTransitions).toBe(holdout.transitionCount);
    expect(report.fidelity).toBe(1);
    expect(report.perSequence.length).toBe(holdout.sequences.length);
    expect(report.perSequence.every((entry) => entry.exactMatches + entry.backoffMatches === entry.transitions)).toBe(true);
  });
});
