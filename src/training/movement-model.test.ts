import { describe, expect, it } from "vitest";
import {
  MarkovMovementBackend,
  MovementModelBackendRegistry,
  buildMovementDatasetFromReplays,
  buildMovementDatasetFromTrajectories,
  createDefaultMovementBackendRegistry,
  defaultMovementTokenizer,
  detailedMovementTokenizer,
  evaluateMovementModel,
  loadMovementModel,
  type MovementDataset,
  type MovementModel,
  type MovementModelBackend,
} from "./movement-model.js";
import { buildTrajectorySpan, type TrajectoryAction } from "../capture/trajectory.js";
import type { ReplayTimelineEvent } from "../capture/replay.js";

function action(tool: string, summary: string, ts: number): TrajectoryAction {
  return { kind: "action", tool, summary, ts };
}

describe("MarkovMovementBackend", () => {
  it("repeats a recorded movement chain exactly (memorization)", () => {
    const dataset: MovementDataset = {
      sequences: [{ id: "s1", tokens: ["focus", "open", "type", "click", "save", "end"] }],
    };
    const model = new MarkovMovementBackend().train(dataset, { order: 2 });

    const continuation = model.generate(["focus"], { stopTokens: ["end"] });
    expect(continuation).toEqual(["open", "type", "click", "save"]);
  });

  it("predicts with full context order when the exact context was seen", () => {
    const dataset: MovementDataset = {
      sequences: [{ id: "s1", tokens: ["a", "x", "y", "z", "end"] }],
    };
    const model = new MarkovMovementBackend().train(dataset, { order: 2 });

    const prediction = model.predictNext(["a", "x"]);
    expect(prediction?.token).toBe("y");
    expect(prediction?.contextOrder).toBe(2);
    expect(prediction?.probability).toBe(1);
  });

  it("generalizes to a new-but-related context by backing off to a shorter context", () => {
    const dataset: MovementDataset = {
      sequences: [
        { id: "a", tokens: ["a", "x", "y", "z", "end"] },
        { id: "b", tokens: ["b", "x", "y", "z", "end"] },
      ],
    };
    const model = new MarkovMovementBackend().train(dataset, { order: 2 });

    // "c" was never seen, so the order-2 context ["c","x"] is absent; the model
    // backs off to the order-1 context ["x"] it learned from both sequences.
    const prediction = model.predictNext(["c", "x"]);
    expect(prediction?.token).toBe("y");
    expect(prediction?.contextOrder).toBe(1);
  });

  it("ranks alternatives deterministically by count then token", () => {
    const dataset: MovementDataset = {
      sequences: [
        { id: "1", tokens: ["m", "b"] },
        { id: "2", tokens: ["m", "b"] },
        { id: "3", tokens: ["m", "a"] },
      ],
    };
    const model = new MarkovMovementBackend().train(dataset, { order: 1 });
    const prediction = model.predictNext(["m"]);
    expect(prediction?.token).toBe("b"); // count 2 > count 1
    expect(prediction?.alternatives[0]?.token).toBe("a");
  });

  it("returns undefined for an empty model", () => {
    const model = new MarkovMovementBackend().train({ sequences: [] });
    expect(model.predictNext(["anything"])).toBeUndefined();
  });

  it("round-trips through serialization", () => {
    const dataset: MovementDataset = {
      sequences: [{ id: "s1", tokens: ["p", "q", "r", "q", "s"] }],
    };
    const model = new MarkovMovementBackend().train(dataset, { order: 2 });
    const restored = loadMovementModel(model.toJSON());

    for (const context of [["p"], ["q"], ["r", "q"], ["p", "q"]]) {
      expect(restored.predictNext(context)?.token).toBe(model.predictNext(context)?.token);
    }
    expect(restored.toJSON()).toEqual(model.toJSON());
  });
});

describe("movement dataset builders", () => {
  it("builds a dataset from trajectory actions, sorted by timestamp", () => {
    const trajectory = buildTrajectorySpan({
      id: "traj-1",
      sessionId: "sess-1",
      actions: [action("click", "button", 30), action("focus", "window", 10), action("type", "text", 20)],
    });
    const dataset = buildMovementDatasetFromTrajectories([trajectory]);
    expect(dataset.sequences).toEqual([{ id: "traj-1", tokens: ["focus", "type", "click"] }]);
  });

  it("groups replay action events per trajectory and ignores non-action events", () => {
    const events: ReplayTimelineEvent[] = [
      { kind: "transcript", ts: 1, messageId: "m1", role: "user", content: "go" },
      { kind: "action", ts: 5, trajectoryId: "t1", tool: "open", summary: "app" },
      { kind: "observation", ts: 6, trajectoryId: "t1", source: "os", summary: "opened" },
      { kind: "action", ts: 7, trajectoryId: "t1", tool: "type", summary: "hello" },
      { kind: "action", ts: 8, trajectoryId: "t2", tool: "scroll", summary: "down" },
    ];
    const dataset = buildMovementDatasetFromReplays([{ events }]);
    expect(dataset.sequences).toEqual([
      { id: "t1", tokens: ["open", "type"] },
      { id: "t2", tokens: ["scroll"] },
    ]);
  });

  it("supports a detailed tokenizer that distinguishes movement variants", () => {
    expect(defaultMovementTokenizer({ tool: "type", summary: "Hello world" })).toBe("type");
    expect(detailedMovementTokenizer({ tool: "type", summary: "Hello world" })).toBe("type:hello");
  });
});

describe("evaluateMovementModel", () => {
  it("scores next-movement accuracy and reports generalization via context order", () => {
    const train: MovementDataset = {
      sequences: [
        { id: "a", tokens: ["login", "nav", "type", "submit"] },
        { id: "b", tokens: ["login", "nav", "type", "submit"] },
      ],
    };
    const model = new MarkovMovementBackend().train(train, { order: 2 });

    // Held-out sequence shares the learned nav→type→submit tail.
    const heldOut: MovementDataset = {
      sequences: [{ id: "c", tokens: ["login", "nav", "type", "submit"] }],
    };
    const result = evaluateMovementModel(model, heldOut);

    expect(result.sequenceCount).toBe(1);
    expect(result.positionCount).toBe(4);
    expect(result.nextTokenAccuracy).toBeGreaterThan(0.5);
    expect(result.averageContextOrder).toBeGreaterThan(0);
    expect(result.perSequence[0]?.id).toBe("c");
  });

  it("reports zero accuracy against an empty model without throwing", () => {
    const model = new MarkovMovementBackend().train({ sequences: [] });
    const result = evaluateMovementModel(model, { sequences: [{ id: "x", tokens: ["a", "b"] }] });
    expect(result.nextTokenAccuracy).toBe(0);
    expect(result.positionCount).toBe(0);
  });
});

describe("MovementModelBackendRegistry", () => {
  it("registers the default Markov backend", () => {
    const registry = createDefaultMovementBackendRegistry();
    expect(registry.list()).toContain("markov-ngram");
    expect(registry.get("markov-ngram")).toBeInstanceOf(MarkovMovementBackend);
  });

  it("is pluggable — a custom backend satisfies the interface and swaps in", () => {
    const constant: MovementModelBackend = {
      id: "constant-mock",
      train(): MovementModel {
        return {
          backendId: "constant-mock",
          order: 0,
          predictNext: () => ({ token: "noop", probability: 1, contextOrder: 0, alternatives: [] }),
          generate: () => ["noop"],
          toJSON: () => ({ version: 1, backendId: "constant-mock", order: 0, transitions: {} }),
        };
      },
    };
    const registry = new MovementModelBackendRegistry().register(constant);
    const model = registry.get("constant-mock")!.train({ sequences: [] });
    expect(model.predictNext([])?.token).toBe("noop");
    expect(registry.list()).toEqual(["constant-mock"]);
  });
});
