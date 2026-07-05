import { describe, expect, it } from "vitest";
import type { ReplayManifest, ReplayTimelineEvent } from "../capture/replay.js";
import {
  MovementBackendRegistry,
  NGramMovementBackend,
  TrainedMovementModel,
  buildMovementDataset,
  evaluateReplayFidelity,
  tokenizeReplayEvent,
  tokenizeReplayManifest,
  type MovementSample,
} from "./movement-backend.js";

/** Build a synthetic replay manifest from an ordered list of action tool names. */
function actionReplay(sessionId: string, trajectoryId: string, tools: string[]): ReplayManifest {
  const events: ReplayTimelineEvent[] = tools.map((tool, index) => ({
    kind: "action",
    ts: index,
    trajectoryId,
    tool,
    summary: `${tool} step ${index}`,
  }));
  return {
    version: 1,
    sessionId,
    trajectoryIds: [trajectoryId],
    eventCount: events.length,
    events,
  };
}

describe("tokenization", () => {
  it("maps each event kind to a stable token", () => {
    expect(tokenizeReplayEvent({ kind: "action", ts: 0, trajectoryId: "t", tool: "click", summary: "" })).toBe(
      "action:click",
    );
    expect(
      tokenizeReplayEvent({ kind: "observation", ts: 0, trajectoryId: "t", source: "screen", summary: "" }),
    ).toBe("observation:screen");
    expect(
      tokenizeReplayEvent({ kind: "transcript", ts: 0, messageId: "m", role: "assistant", content: "" }),
    ).toBe("transcript:assistant");
  });

  it("builds a de-duplicated, sorted vocabulary from replays", () => {
    const dataset = buildMovementDataset([
      actionReplay("s1", "t1", ["focus", "click", "type"]),
      actionReplay("s2", "t2", ["focus", "scroll"]),
    ]);
    expect(dataset.samples).toHaveLength(2);
    expect(dataset.vocabulary).toEqual([
      "action:click",
      "action:focus",
      "action:scroll",
      "action:type",
    ]);
    expect(tokenizeReplayManifest(actionReplay("s3", "t3", ["a"])).tokens).toEqual(["action:a"]);
  });

  it("drops empty replays from the dataset", () => {
    const dataset = buildMovementDataset([actionReplay("s1", "t1", [])]);
    expect(dataset.samples).toHaveLength(0);
    expect(dataset.vocabulary).toEqual([]);
  });
});

describe("NGramMovementBackend training + replay", () => {
  it("reproduces a recorded movement exactly from its first step", async () => {
    const recorded = ["focus", "click", "type", "submit"];
    const dataset = buildMovementDataset([actionReplay("s1", "t1", recorded)]);
    const model = await new NGramMovementBackend().train(dataset);

    const continuation = model.generate(["action:focus"], 10);
    expect(["action:focus", ...continuation]).toEqual(recorded.map((tool) => `action:${tool}`));
  });

  it("generates from an empty prompt using the learned start distribution", async () => {
    const dataset = buildMovementDataset([actionReplay("s1", "t1", ["focus", "click"])]);
    const model = await new NGramMovementBackend().train(dataset);
    expect(model.generate([], 10)).toEqual(["action:focus", "action:click"]);
  });

  it("is deterministic: identical datasets yield identical predictions", async () => {
    const dataset = buildMovementDataset([actionReplay("s1", "t1", ["focus", "click", "type"])]);
    const backend = new NGramMovementBackend();
    const first = await backend.train(dataset);
    const second = await backend.train(dataset);
    expect(first.generate([], 10)).toEqual(second.generate([], 10));
    expect(first.predictNext(["action:focus"])).toEqual(second.predictNext(["action:focus"]));
  });

  it("breaks ties lexicographically for reproducibility", async () => {
    // After "focus" the model has seen "click" and "apply" once each; the
    // lexicographically smaller token must win deterministically.
    const dataset = buildMovementDataset([
      actionReplay("s1", "t1", ["focus", "click"]),
      actionReplay("s2", "t2", ["focus", "apply"]),
    ]);
    const model = await new NGramMovementBackend().train(dataset, { order: 1 });
    expect(model.predictNext(["action:focus"])?.token).toBe("action:apply");
  });
});

describe("generalization", () => {
  it("composes a novel path from overlapping sub-sequences", async () => {
    // Trajectory A: open -> menu -> save.  Trajectory B: edit -> menu -> export.
    // Neither contains "open -> menu -> export", but an order-1 model should
    // generalize the shared "menu" pivot to reach export.
    const dataset = buildMovementDataset([
      actionReplay("sA", "tA", ["open", "menu", "save"]),
      actionReplay("sB", "tB", ["edit", "menu", "export"]),
    ]);
    const model = await new NGramMovementBackend().train(dataset, { order: 1 });

    // From the "menu" pivot the next step is decided by the order-1 context.
    const next = model.predictNext(["action:menu"]);
    expect(next).toBeDefined();
    // "export" < "save" lexicographically, and both were seen once, so the model
    // deterministically produces the cross-trajectory continuation.
    expect(next?.token).toBe("action:export");
    expect(["action:open", ...model.generate(["action:open"], 10)]).toEqual([
      "action:open",
      "action:menu",
      "action:export",
    ]);
  });

  it("backs off to shorter contexts for unseen prefixes", async () => {
    const dataset = buildMovementDataset([actionReplay("s1", "t1", ["focus", "click", "type"])]);
    const model = await new NGramMovementBackend().train(dataset, { order: 2 });
    // "scroll" was never observed, but the model still predicts via unigram/backoff.
    const prediction = model.predictNext(["action:scroll"]);
    expect(prediction).toBeDefined();
    expect(prediction!.order).toBeLessThan(2);
  });
});

describe("serialization", () => {
  it("round-trips a trained model through JSON without changing behaviour", async () => {
    const dataset = buildMovementDataset([actionReplay("s1", "t1", ["focus", "click", "type"])]);
    const backend = new NGramMovementBackend();
    const model = await backend.train(dataset);

    const restored = TrainedMovementModel.fromJSON(JSON.parse(JSON.stringify(model.toJSON())));
    expect(restored.order).toBe(model.order);
    expect(restored.vocabulary).toEqual(model.vocabulary);
    expect(restored.generate([], 10)).toEqual(model.generate([], 10));

    const viaBackend = backend.load(model.toJSON());
    expect(viaBackend.generate(["action:focus"], 10)).toEqual(model.generate(["action:focus"], 10));
  });
});

describe("MovementBackendRegistry", () => {
  it("resolves the default ngram backend and lists ids", () => {
    const registry = new MovementBackendRegistry();
    expect(registry.list()).toEqual(["ngram"]);
    expect(registry.get("ngram")).toBeInstanceOf(NGramMovementBackend);
  });

  it("throws on an unknown backend id", () => {
    const registry = new MovementBackendRegistry();
    expect(() => registry.get("mystery")).toThrow(/Unknown movement backend/);
  });

  it("accepts a custom pluggable backend", async () => {
    const registry = new MovementBackendRegistry([new NGramMovementBackend()]);
    const custom = {
      id: "constant",
      async train() {
        return TrainedMovementModel.fromJSON({
          version: 1 as const,
          backendId: "constant",
          order: 1,
          vocabulary: [],
          transitions: [],
        });
      },
      load: (serialized: Parameters<NGramMovementBackend["load"]>[0]) =>
        TrainedMovementModel.fromJSON(serialized),
    };
    registry.register(custom);
    expect(registry.list()).toContain("constant");
    expect(registry.get("constant").id).toBe("constant");
  });
});

describe("evaluateReplayFidelity", () => {
  it("scores perfect fidelity on a memorized sample", async () => {
    const recorded = ["focus", "click", "type"];
    const dataset = buildMovementDataset([actionReplay("s1", "t1", recorded)]);
    const model = await new NGramMovementBackend().train(dataset);

    const heldOut: MovementSample = tokenizeReplayManifest(actionReplay("s1", "t1", recorded));
    const report = evaluateReplayFidelity(model, [heldOut]);
    expect(report.sampleCount).toBe(1);
    expect(report.accuracy).toBe(1);
    expect(report.exactReplayRate).toBe(1);
    expect(report.correct).toBe(report.predictions);
  });

  it("reports partial fidelity on a related-but-unseen sample", async () => {
    const dataset = buildMovementDataset([actionReplay("s1", "t1", ["focus", "click", "type"])]);
    const model = await new NGramMovementBackend().train(dataset, { order: 2 });

    const related = tokenizeReplayManifest(actionReplay("s2", "t2", ["focus", "click", "submit"]));
    const report = evaluateReplayFidelity(model, [related]);
    // The shared prefix "focus -> click" is predicted correctly; the divergent
    // final step is not — so fidelity is strictly between 0 and 1.
    expect(report.accuracy).toBeGreaterThan(0);
    expect(report.accuracy).toBeLessThan(1);
    expect(report.exactReplayRate).toBe(0);
  });

  it("returns neutral fidelity when there is nothing to score", () => {
    const model = TrainedMovementModel.fromJSON({
      version: 1,
      backendId: "ngram",
      order: 2,
      vocabulary: [],
      transitions: [],
    });
    const report = evaluateReplayFidelity(model, []);
    expect(report).toEqual({
      sampleCount: 0,
      predictions: 0,
      correct: 0,
      accuracy: 1,
      exactReplayRate: 0,
    });
  });
});
