import { describe, expect, it } from "vitest";
import type { ReplayTimelineEvent } from "../capture/replay.js";
import {
  MovementModelBackendRegistry,
  NgramMovementBackend,
  buildMovementDataset,
  type MovementDataset,
  type MovementSequence,
} from "./movement-model.js";

function seq(id: string, tokens: string[]): MovementSequence {
  return {
    id,
    steps: tokens.map((token, index) => ({
      token,
      tool: token,
      summary: `${token} #${index}`,
      ts: index,
    })),
  };
}

function dataset(...sequences: MovementSequence[]): MovementDataset {
  return { version: 1, sequences };
}

describe("NgramMovementBackend", () => {
  it("repeats a recorded movement sequence from its seed", () => {
    const recorded = ["focus", "click", "type", "submit", "screenshot"];
    const model = new NgramMovementBackend().train(dataset(seq("t1", recorded)), { order: 3 });

    const continuation = model.generate([recorded[0]!]);

    expect(continuation).toEqual(recorded.slice(1));
  });

  it("generalizes to a novel-but-related prefix via back-off", () => {
    // Two related recordings that share the suffix pattern "type -> submit".
    const model = new NgramMovementBackend().train(
      dataset(
        seq("a", ["open", "focus", "type", "submit"]),
        seq("b", ["scroll", "focus", "type", "submit"]),
      ),
      { order: 2 },
    );

    // "hover" was never seen before "focus", but the model has learned that
    // "focus" is followed by "type" and "type" by "submit". Back-off should
    // still produce the shared continuation.
    const prediction = model.predictNext(["hover", "focus"]);
    expect(prediction?.token).toBe("type");
    // It backed off from the unseen order-2 context to a shorter one.
    expect(prediction?.backoffOrder).toBeLessThan(2);

    const rollout = model.generate(["hover", "focus"], { maxSteps: 5 });
    expect(rollout).toEqual(["type", "submit"]);
  });

  it("prefers the higher-order context when it is trusted (faithful repeat over generalization)", () => {
    // After "b" the next token is context-dependent: "x" then "1", but "y" then "2".
    const model = new NgramMovementBackend().train(
      dataset(seq("s1", ["x", "b", "1"]), seq("s2", ["y", "b", "2"])),
      { order: 2 },
    );

    expect(model.predictNext(["x", "b"])?.token).toBe("1");
    expect(model.predictNext(["y", "b"])?.token).toBe("2");
    // Order-2 context is used, not the ambiguous order-1 "b".
    expect(model.predictNext(["x", "b"])?.backoffOrder).toBe(2);
  });

  it("is fully deterministic: identical dataset yields identical model and output", () => {
    const data = dataset(
      seq("a", ["p", "q", "r", "q", "p"]),
      seq("b", ["q", "r", "p", "r"]),
    );
    const first = new NgramMovementBackend().train(data, { order: 3 });
    const second = new NgramMovementBackend().train(data, { order: 3 });

    expect(first.serialize()).toEqual(second.serialize());
    expect(first.generate(["p"])).toEqual(second.generate(["p"]));
  });

  it("breaks prediction ties by lexical token order (reproducible)", () => {
    // From the empty context, "a" and "b" are equally frequent; "a" wins.
    const model = new NgramMovementBackend().train(dataset(seq("a", ["a"]), seq("b", ["b"])), {
      order: 1,
    });
    expect(model.predictNext([])?.token).toBe("a");
  });

  it("round-trips through serialize/deserialize", () => {
    const backend = new NgramMovementBackend();
    const model = backend.train(dataset(seq("a", ["one", "two", "three", "two", "one"])), { order: 2 });
    const restored = backend.deserialize(model.serialize());

    expect(restored.serialize()).toEqual(model.serialize());
    expect(restored.generate(["one"])).toEqual(model.generate(["one"]));
    expect(restored.stats().vocabularySize).toBe(model.stats().vocabularySize);
  });

  it("reports coherent training stats", () => {
    const model = new NgramMovementBackend().train(
      dataset(seq("a", ["click", "type"]), seq("b", ["click", "scroll"])),
      { order: 2 },
    );
    const stats = model.stats();
    expect(stats.backendId).toBe("ngram-backoff");
    expect(stats.sequenceCount).toBe(2);
    expect(stats.stepCount).toBe(4);
    expect(stats.vocabularySize).toBe(3); // click, type, scroll
    expect(stats.transitionCount).toBeGreaterThan(0);
  });

  it("halts idle loops via stopOnRepeat", () => {
    // A degenerate recording that would otherwise self-loop on "idle".
    const model = new NgramMovementBackend().train(dataset(seq("a", ["idle", "idle", "idle", "idle"])), {
      order: 1,
    });
    const rollout = model.generate(["idle"], { maxSteps: 100, stopOnRepeat: 2 });
    expect(rollout.length).toBeLessThan(100);
  });

  it("handles an empty dataset without throwing", () => {
    const model = new NgramMovementBackend().train(dataset(), { order: 2 });
    expect(model.predictNext(["anything"])).toBeUndefined();
    expect(model.generate(["anything"])).toEqual([]);
    expect(model.stats().stepCount).toBe(0);
  });

  it("clamps a non-positive order to at least 1", () => {
    const model = new NgramMovementBackend().train(dataset(seq("a", ["m", "n"])), { order: 0 });
    expect(model.order).toBe(1);
    expect(model.generate(["m"])).toEqual(["n"]);
  });
});

describe("MovementModelBackendRegistry", () => {
  it("resolves the default ngram backend", () => {
    const registry = new MovementModelBackendRegistry();
    expect(registry.list()).toContain("ngram-backoff");
    expect(registry.require("ngram-backoff").id).toBe("ngram-backoff");
  });

  it("throws for an unknown backend and supports registering custom ones", () => {
    const registry = new MovementModelBackendRegistry();
    expect(() => registry.require("mlx")).toThrow(/Unknown movement-model backend/);

    const stub = new NgramMovementBackend();
    Object.defineProperty(stub, "id", { value: "custom" });
    registry.register(stub);
    expect(registry.get("custom")?.id).toBe("custom");
  });
});

describe("buildMovementDataset", () => {
  it("extracts action sequences from replay manifests in timeline order", () => {
    const events: ReplayTimelineEvent[] = [
      { kind: "transcript", ts: 0, messageId: "m0", role: "user", content: "go" },
      { kind: "action", ts: 3, trajectoryId: "t", tool: "submit", summary: "submit" },
      { kind: "observation", ts: 1, trajectoryId: "t", source: "screen", summary: "form" },
      { kind: "action", ts: 2, trajectoryId: "t", tool: "type", summary: "type" },
    ];
    const built = buildMovementDataset([{ sessionId: "s1", events }]);

    expect(built.sequences).toHaveLength(1);
    expect(built.sequences[0]!.steps.map((s) => s.token)).toEqual(["type", "submit"]);
  });

  it("skips replays without action events and supports a custom tokenizer", () => {
    const observationOnly: ReplayTimelineEvent[] = [
      { kind: "observation", ts: 0, trajectoryId: "t", source: "screen", summary: "idle" },
    ];
    const withActions: ReplayTimelineEvent[] = [
      { kind: "action", ts: 0, trajectoryId: "t", tool: "click", summary: "click ok" },
    ];
    const built = buildMovementDataset(
      [
        { sessionId: "empty", events: observationOnly },
        { sessionId: "s2", events: withActions },
      ],
      { tokenize: (event) => `${event.tool}:${event.summary}` },
    );

    expect(built.sequences.map((s) => s.id)).toEqual(["s2"]);
    expect(built.sequences[0]!.steps[0]!.token).toBe("click:click ok");
  });

  it("feeds end-to-end into a trained model that repeats the captured movement", () => {
    const events: ReplayTimelineEvent[] = ["focus", "type", "submit"].map((tool, index) => ({
      kind: "action",
      ts: index,
      trajectoryId: "t",
      tool,
      summary: tool,
    }));
    const built = buildMovementDataset([{ sessionId: "s", events }]);
    const model = new NgramMovementBackend().train(built, { order: 3 });

    expect(model.generate(["focus"])).toEqual(["type", "submit"]);
  });
});
