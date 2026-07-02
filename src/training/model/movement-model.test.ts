import { describe, expect, it } from "vitest";
import {
  MockMovementBackend,
  MovementModelRegistry,
  loadMovementModel,
  movementEventToken,
  type MovementDataset,
  type MovementEvent,
} from "./movement-model.js";

function seq(id: string, tokens: Array<[MovementEvent["kind"], string, string]>, startTs = 1000): {
  trajectoryId: string;
  events: MovementEvent[];
  outcome: "success";
} {
  let ts = startTs;
  const events = tokens.map(([kind, key, summary]) => {
    ts += 100;
    return kind === "action"
      ? ({ kind, tool: key, summary, ts } as MovementEvent)
      : ({ kind, source: key, summary, ts } as MovementEvent);
  });
  return { trajectoryId: id, events, outcome: "success" };
}

const dataset: MovementDataset = {
  version: 1,
  sequences: [
    seq("a", [
      ["observation", "window", "focus editor"],
      ["action", "mouse.click", "click File"],
      ["action", "mouse.click", "click Open"],
      ["action", "keyboard.type", "type report.md"],
      ["action", "keyboard.press", "press Enter"],
    ]),
    seq("b", [
      ["observation", "window", "focus editor"],
      ["action", "mouse.click", "click File"],
      ["action", "mouse.click", "click Open"],
      ["action", "keyboard.type", "type notes.txt"],
      ["action", "keyboard.press", "press Enter"],
    ]),
  ],
};

describe("MockMovementBackend", () => {
  it("trains a model with sensible stats", async () => {
    const model = await new MockMovementBackend().train(dataset);
    const stats = model.stats();
    expect(stats.backendId).toBe("mock-markov");
    expect(stats.sequenceCount).toBe(2);
    expect(stats.eventCount).toBe(10);
    expect(stats.vocabularySize).toBeGreaterThan(0);
    expect(stats.transitionCount).toBeGreaterThan(0);
  });

  it("replays a recorded movement: predicts the true next action", async () => {
    const model = await new MockMovementBackend().train(dataset);
    // After focus->click File->click Open, the recorded next tool is keyboard.type.
    const prefix = dataset.sequences[0].events.slice(0, 3);
    const prediction = model.predictNext(prefix);
    expect(prediction).toBeDefined();
    expect(prediction?.event.kind).toBe("action");
    expect(prediction?.event.tool).toBe("keyboard.type");
    expect(prediction?.confidence).toBeGreaterThan(0);
  });

  it("generates a full rollout that terminates at the learned END", async () => {
    const model = await new MockMovementBackend().train(dataset);
    const seed = dataset.sequences[0].events.slice(0, 1);
    const rollout = model.generate(seed, { maxSteps: 20 });
    // Should reproduce the shared prefix and stop (no runaway).
    expect(rollout.length).toBeGreaterThan(0);
    expect(rollout.length).toBeLessThan(20);
    const tokens = rollout.map(movementEventToken);
    expect(tokens).toContain("action:mouse.click");
    expect(tokens).toContain("action:keyboard.type");
  });

  it("assigns strictly increasing timestamps during generation", async () => {
    const model = await new MockMovementBackend().train(dataset);
    const seed = dataset.sequences[0].events.slice(0, 1);
    const rollout = model.generate(seed);
    for (let i = 1; i < rollout.length; i += 1) {
      expect(rollout[i].ts).toBeGreaterThan(rollout[i - 1].ts);
    }
  });

  it("generalizes to an unseen prefix via back-off", async () => {
    const model = await new MockMovementBackend().train(dataset);
    // A prefix that never appears at high order but whose last token (click) is known.
    const novelPrefix: MovementEvent[] = [
      { kind: "action", tool: "mouse.click", summary: "click somewhere new", ts: 5000 },
    ];
    const prediction = model.predictNext(novelPrefix);
    expect(prediction).toBeDefined();
    // Backed off to a lower order (< maxOrder) but still produced a learned transition.
    expect(prediction?.order).toBeLessThanOrEqual(1);
  });

  it("is deterministic across repeated training + inference", async () => {
    const backend = new MockMovementBackend();
    const first = await backend.train(dataset);
    const second = await backend.train(dataset);
    const seed = dataset.sequences[0].events.slice(0, 1);
    expect(first.generate(seed).map(movementEventToken)).toEqual(
      second.generate(seed).map(movementEventToken),
    );
  });

  it("round-trips through serialize/load with identical behaviour", async () => {
    const model = await new MockMovementBackend().train(dataset);
    const restored = loadMovementModel(model.serialize());
    const seed = dataset.sequences[0].events.slice(0, 1);
    expect(restored.generate(seed).map(movementEventToken)).toEqual(
      model.generate(seed).map(movementEventToken),
    );
    expect(restored.stats().vocabularySize).toBe(model.stats().vocabularySize);
  });

  it("honours includeOutcomes filtering", async () => {
    const mixed: MovementDataset = {
      version: 1,
      sequences: [
        { ...dataset.sequences[0], outcome: "success" },
        { ...dataset.sequences[1], outcome: "failure" },
      ],
    };
    const model = await new MockMovementBackend().train(mixed, { includeOutcomes: ["success"] });
    expect(model.stats().sequenceCount).toBe(1);
  });
});

describe("MovementModelRegistry", () => {
  it("registers the mock backend by default and resolves it", () => {
    const registry = new MovementModelRegistry();
    expect(registry.has("mock-markov")).toBe(true);
    expect(registry.resolve("mock-markov")).toBeInstanceOf(MockMovementBackend);
    expect(registry.list()).toContain("mock-markov");
  });

  it("supports registering an alternative backend", () => {
    const registry = new MovementModelRegistry();
    registry.register(new MockMovementBackend("on-device-mlx"));
    expect(registry.has("on-device-mlx")).toBe(true);
    expect(registry.list()).toEqual(["mock-markov", "on-device-mlx"]);
  });
});
