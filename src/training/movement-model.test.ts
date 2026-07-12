import { describe, expect, it } from "vitest";
import type { ReplayTimelineEvent } from "../capture/replay.js";
import {
  buildMovementDataset,
  createDefaultMovementModelRegistry,
  evaluateMovementModel,
  MarkovMovementBackend,
  MemorizingMovementBackend,
  restoreMovementModel,
  tokenizeReplayEvent,
  type ReplaySource,
} from "./movement-model.js";
import {
  DEFAULT_MOVEMENT_GRAMMAR,
  generateSyntheticMovementStreams,
} from "./synthetic-movements.js";

function action(ts: number, tool: string, summary: string): ReplayTimelineEvent {
  return { kind: "action", ts, trajectoryId: "t1", tool, summary };
}

function observation(ts: number, source: string, summary: string): ReplayTimelineEvent {
  return { kind: "observation", ts, trajectoryId: "t1", source, summary };
}

describe("movement dataset", () => {
  it("tokenizes action events and skips observations by default", () => {
    const source: ReplaySource[] = [
      {
        id: "s1",
        events: [observation(0, "os", "focus"), action(1, "pointer.click", "Click File")],
      },
    ];
    const dataset = buildMovementDataset(source);
    expect(dataset.sequences).toHaveLength(1);
    expect(dataset.sequences[0]!.steps.map((step) => step.kind)).toEqual(["action"]);
    // Summary is normalized (lowercased, collapsed) inside the token.
    expect(dataset.sequences[0]!.steps[0]!.token).toBe("action:pointer.click:click file");
    expect(dataset.vocab).toEqual(["action:pointer.click:click file"]);
  });

  it("includes observations when requested and orders steps by ts", () => {
    const source: ReplaySource[] = [
      {
        id: "s1",
        events: [action(5, "keyboard.type", "type"), observation(1, "ui", "ready")],
      },
    ];
    const dataset = buildMovementDataset(source, { includeObservations: true });
    const kinds = dataset.sequences[0]!.steps.map((step) => step.kind);
    expect(kinds).toEqual(["observation", "action"]);
  });

  it("tokenizeReplayEvent returns undefined for skipped observations", () => {
    expect(tokenizeReplayEvent(observation(0, "os", "x"))).toBeUndefined();
    expect(tokenizeReplayEvent(observation(0, "os", "x"), { includeObservations: true })).toBeDefined();
  });
});

describe("MarkovMovementBackend", () => {
  const repeated: ReplaySource[] = [
    { id: "a", events: [action(0, "open", "a"), action(1, "nav", "b"), action(2, "save", "c")] },
    { id: "b", events: [action(0, "open", "a"), action(1, "nav", "b"), action(2, "save", "c")] },
  ];

  it("repeats an exactly-recorded movement sequence via generate()", async () => {
    const dataset = buildMovementDataset(repeated);
    const model = await new MarkovMovementBackend().train(dataset, { order: 2 });
    const seed = ["action:open:a"];
    const rollout = model.generate(seed, 2);
    expect(rollout).toEqual(["action:nav:b", "action:save:c"]);
  });

  it("predicts ranked candidates with normalized probabilities", async () => {
    const dataset = buildMovementDataset(repeated);
    const model = await new MarkovMovementBackend().train(dataset, { order: 2 });
    const predictions = model.predictNext(["action:open:a"]);
    expect(predictions[0]!.token).toBe("action:nav:b");
    const total = predictions.reduce((sum, p) => sum + p.probability, 0);
    expect(total).toBeCloseTo(1, 6);
  });

  it("generalizes to an unseen prefix via backoff", async () => {
    // Two motifs sharing a common continuation after "nav".
    const dataset = buildMovementDataset([
      { id: "a", events: [action(0, "open", "a"), action(1, "nav", "b"), action(2, "save", "c")] },
      { id: "b", events: [action(0, "start", "z"), action(1, "nav", "b"), action(2, "save", "c")] },
    ]);
    const model = await new MarkovMovementBackend().train(dataset, { order: 2 });
    // Context ["action:nav:b"] with a preceding token never seen in that pair
    // still predicts the shared continuation through lower-order backoff.
    const predictions = model.predictNext(["action:unseen:q", "action:nav:b"]);
    expect(predictions[0]!.token).toBe("action:save:c");
  });

  it("returns no predictions for an empty model", async () => {
    const model = await new MarkovMovementBackend().train(buildMovementDataset([]), { order: 2 });
    expect(model.predictNext(["anything"])).toEqual([]);
    expect(model.generate(["anything"], 3)).toEqual([]);
  });

  it("is deterministic across repeated training + generation", async () => {
    const dataset = buildMovementDataset(repeated);
    const a = await new MarkovMovementBackend().train(dataset, { order: 2 });
    const b = await new MarkovMovementBackend().train(dataset, { order: 2 });
    expect(a.generate(["action:open:a"], 3)).toEqual(b.generate(["action:open:a"], 3));
  });
});

describe("snapshot round-trip", () => {
  it("restores an equivalent model from a snapshot", async () => {
    const dataset = buildMovementDataset([
      { id: "a", events: [action(0, "open", "a"), action(1, "nav", "b"), action(2, "save", "c")] },
    ]);
    const model = await new MarkovMovementBackend().train(dataset, { order: 2 });
    const restored = restoreMovementModel(model.snapshot());
    expect(restored.vocabSize).toBe(model.vocabSize);
    expect(restored.order).toBe(model.order);
    expect(restored.generate(["action:open:a"], 2)).toEqual(model.generate(["action:open:a"], 2));
  });
});

describe("MovementModelBackendRegistry", () => {
  it("exposes reference backends with markov as default", () => {
    const registry = createDefaultMovementModelRegistry();
    expect(registry.list()).toEqual(["markov", "memorizing"]);
    expect(registry.getDefault().id).toBe("markov");
    expect(registry.get("memorizing").id).toBe("memorizing");
    expect(registry.has("nope")).toBe(false);
    expect(() => registry.get("nope")).toThrow(/Unknown movement model backend/);
  });
});

describe("evaluateMovementModel", () => {
  it("scores a Markov model above the memorizing baseline on held-out related streams", async () => {
    const train = generateSyntheticMovementStreams({ seed: 1, streamCount: 6, stepsPerStream: 10 });
    const heldOut = generateSyntheticMovementStreams({ seed: 999, streamCount: 3, stepsPerStream: 10 });

    const trainDataset = buildMovementDataset(train);
    const heldOutDataset = buildMovementDataset(heldOut);

    const markov = await new MarkovMovementBackend().train(trainDataset, { order: 2 });
    const memorizing = await new MemorizingMovementBackend().train(trainDataset, { order: 2 });

    const markovReport = evaluateMovementModel(markov, heldOutDataset.sequences, { topK: 3 });
    const memorizingReport = evaluateMovementModel(memorizing, heldOutDataset.sequences, { topK: 3 });

    expect(markovReport.samples).toBeGreaterThan(0);
    // The generalizing model should predict real held-out movements more often
    // and abstain far less than the pure-memorization baseline.
    expect(markovReport.top1Accuracy).toBeGreaterThan(memorizingReport.top1Accuracy);
    expect(markovReport.abstained).toBeLessThanOrEqual(memorizingReport.abstained);
    expect(markovReport.topKAccuracy).toBeGreaterThanOrEqual(markovReport.top1Accuracy);
  });
});

describe("generateSyntheticMovementStreams", () => {
  it("is reproducible for a given seed and varies across seeds", () => {
    const a = generateSyntheticMovementStreams({ seed: 42, streamCount: 2, stepsPerStream: 6 });
    const b = generateSyntheticMovementStreams({ seed: 42, streamCount: 2, stepsPerStream: 6 });
    const c = generateSyntheticMovementStreams({ seed: 7, streamCount: 2, stepsPerStream: 6 });
    expect(a).toEqual(b);
    expect(a).not.toEqual(c);
  });

  it("emits ordered events drawn from the default grammar", () => {
    const [stream] = generateSyntheticMovementStreams({ seed: 3, streamCount: 1, stepsPerStream: 5 });
    expect(stream!.events.length).toBeGreaterThan(0);
    const tsValues = stream!.events.map((event) => event.ts);
    const sorted = [...tsValues].sort((x, y) => x - y);
    expect(tsValues).toEqual(sorted);
    const tools = new Set(DEFAULT_MOVEMENT_GRAMMAR.motifs.map((motif) => motif.tool));
    for (const event of stream!.events) {
      if (event.kind === "action") {
        expect(tools.has(event.tool)).toBe(true);
      }
    }
  });
});
