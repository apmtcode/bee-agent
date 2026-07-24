import { describe, expect, it } from "vitest";
import type { ReplayTimelineEvent } from "../capture/replay.js";
import {
  MovementModelRegistry,
  NgramMovementBackend,
  buildMovementExamples,
  createDefaultMovementModelRegistry,
  evaluateMovementModel,
  slugifyMovement,
  tokenizeReplayEvents,
  trainMovementModelFromReplays,
  type MovementModelBackend,
  type TaggedMovementToken,
} from "./movement-model.js";

/** Build a replay-style action/observation event stream from a compact spec. */
function replay(steps: Array<{ obs?: string; act?: string; tool?: string; source?: string }>): ReplayTimelineEvent[] {
  const events: ReplayTimelineEvent[] = [];
  let ts = 0;
  for (const step of steps) {
    ts += 1;
    if (step.obs) {
      events.push({ kind: "observation", ts, trajectoryId: "t1", source: step.source ?? "os", summary: step.obs });
    }
    if (step.act) {
      ts += 1;
      events.push({ kind: "action", ts, trajectoryId: "t1", tool: step.tool ?? "device", summary: step.act });
    }
  }
  return events;
}

describe("slugifyMovement", () => {
  it("produces stable comparable slugs", () => {
    expect(slugifyMovement("Tapped Submit Button!")).toBe("tapped-submit-button");
    expect(slugifyMovement("   ")).toBe("unknown");
    expect(slugifyMovement("scrolled DOWN")).toBe(slugifyMovement("Scrolled down"));
  });
});

describe("tokenizeReplayEvents", () => {
  it("tags observations and actions and skips transcript by default", () => {
    const events: ReplayTimelineEvent[] = [
      { kind: "transcript", ts: 1, messageId: "m1", role: "user", content: "hi" },
      { kind: "observation", ts: 2, trajectoryId: "t1", source: "os", summary: "editor focused" },
      { kind: "action", ts: 3, trajectoryId: "t1", tool: "keyboard", summary: "typed hello" },
    ];
    expect(tokenizeReplayEvents(events).map((token) => token.kind)).toEqual(["observation", "action"]);
    expect(tokenizeReplayEvents(events, { includeTranscript: true }).map((token) => token.kind)).toEqual([
      "transcript",
      "observation",
      "action",
    ]);
    expect(tokenizeReplayEvents(events)[1].token).toBe("act:keyboard:typed-hello");
  });
});

describe("buildMovementExamples", () => {
  it("targets each action with a bounded preceding context", () => {
    const tokens = tokenizeReplayEvents(replay([{ obs: "a", act: "one" }, { obs: "b", act: "two" }, { obs: "c", act: "three" }]));
    const examples = buildMovementExamples(tokens, 2);
    expect(examples).toHaveLength(3);
    // Context is bounded to `order` most-recent tokens.
    expect(examples[2].context.length).toBeLessThanOrEqual(2);
    expect(examples[0].action).toBe("act:device:one");
  });
});

describe("NgramMovementBackend", () => {
  it("repeats recorded movements exactly (repeat fidelity)", () => {
    const events = replay([
      { obs: "form open", act: "tap name field", tool: "device" },
      { obs: "name field focused", act: "type name", tool: "keyboard" },
      { obs: "name entered", act: "tap submit", tool: "device" },
    ]);
    const model = trainMovementModelFromReplays({ replays: [{ events }] }, { order: 3 });
    const tokens = tokenizeReplayEvents(events);
    const evaluation = evaluateMovementModel(model, [tokens]);
    expect(evaluation.accuracy).toBe(1);
    expect(evaluation.correct).toBe(evaluation.totalActions);
    // Every hit came from an exact-length context match, not backoff.
    expect(evaluation.bySource.exact.correct).toBe(evaluation.totalActions);
  });

  it("generalizes to a new-but-related movement via context backoff", () => {
    // Two demonstrations that share the tail pattern "focus a field, then type
    // creds" but differ in how they got there.
    const trainA = replay([
      { obs: "login form", act: "tap field", tool: "device" },
      { obs: "field focused", act: "type creds", tool: "keyboard" },
    ]);
    const trainB = replay([
      { obs: "search page", act: "tap field", tool: "device" },
      { obs: "field focused", act: "type creds", tool: "keyboard" },
    ]);
    const model = trainMovementModelFromReplays({ replays: [{ events: trainA }, { events: trainB }] }, { order: 3 });

    // Held-out: the leading observation ("email box") was never seen, so the
    // full-order context cannot match — but the shared 2-token suffix
    // (tap field -> field focused -> type creds) has been learned. Backoff
    // recovers the right movement despite the novel surrounding context.
    const heldOut: TaggedMovementToken[] = tokenizeReplayEvents(
      replay([{ obs: "email box", act: "tap field", tool: "device" }, { obs: "field focused" }]),
    );
    const prediction = model.predict(heldOut.map((token) => token.token));
    expect(prediction).toBeDefined();
    expect(prediction?.action).toBe("act:keyboard:type-creds");
    // The novel full context forced a shorter-suffix (backoff) match, not exact.
    expect(prediction?.source).toBe("backoff");
    expect(prediction?.confidence).toBeGreaterThan(0);
  });

  it("falls back to the prior for a wholly unseen context", () => {
    const model = trainMovementModelFromReplays({ replays: [{ events: replay([{ obs: "x", act: "only action" }]) }] });
    const prediction = model.predict(["obs:never:seen"]);
    expect(prediction?.source).toBe("prior");
    expect(prediction?.action).toBe("act:device:only-action");
  });

  it("returns undefined when nothing was learned", () => {
    const model = new NgramMovementBackend().train([]);
    expect(model.predict(["anything"])).toBeUndefined();
    expect(model.vocabulary).toEqual([]);
  });

  it("generate() rolls out a deterministic recorded sequence", () => {
    const events = replay([
      { obs: "start", act: "step one" },
      { act: "step two" },
      { act: "step three" },
    ]);
    const model = trainMovementModelFromReplays({ replays: [{ events }] }, { order: 2 });
    const rollout = model.generate(["obs:os:start"], 3);
    expect(rollout[0]).toBe("act:device:step-one");
    expect(rollout).toEqual(model.generate(["obs:os:start"], 3)); // deterministic
  });

  it("serialize/load round-trips to an identical policy", () => {
    const events = replay([{ obs: "a", act: "one" }, { obs: "b", act: "two" }]);
    const backend = new NgramMovementBackend();
    const original = trainMovementModelFromReplays({ replays: [{ events }] }, { order: 2 });
    const restored = backend.load(original.serialize());
    const context = ["obs:os:a"];
    expect(restored.serialize()).toEqual(original.serialize());
    expect(restored.predict(context)?.action).toBe(original.predict(context)?.action);
  });

  it("chooses the most frequent action deterministically on ties", () => {
    // "z-action" and "a-action" both follow the same context once each; the
    // lexically-smaller token must win the tie for reproducibility.
    const model = new NgramMovementBackend().train(
      [
        { context: ["ctx"], action: "z-action" },
        { context: ["ctx"], action: "a-action" },
      ],
      { order: 1 },
    );
    expect(model.predict(["ctx"])?.action).toBe("a-action");
    // A clear majority overrides the tie-break.
    const skewed = new NgramMovementBackend().train(
      [
        { context: ["ctx"], action: "z-action" },
        { context: ["ctx"], action: "z-action" },
        { context: ["ctx"], action: "a-action" },
      ],
      { order: 1 },
    );
    const prediction = skewed.predict(["ctx"]);
    expect(prediction?.action).toBe("z-action");
    expect(prediction?.confidence).toBeCloseTo(2 / 3);
  });
});

describe("MovementModelRegistry", () => {
  it("exposes the default n-gram backend and supports pluggable registration", () => {
    const registry = createDefaultMovementModelRegistry();
    expect(registry.list()).toContain("ngram-backoff");
    expect(registry.get("ngram-backoff")).toBeInstanceOf(NgramMovementBackend);

    const stub: MovementModelBackend = {
      id: "mock-onnx",
      train: () => registry.get("ngram-backoff").train([]),
      load: (serialized) => registry.get("ngram-backoff").load(serialized),
    };
    registry.register(stub);
    expect(registry.has("mock-onnx")).toBe(true);
    expect(() => registry.get("missing")).toThrow(/Unknown movement-model backend/);
  });

  it("routes trainMovementModelFromReplays through the selected backend", () => {
    const registry = new MovementModelRegistry([new NgramMovementBackend()]);
    const model = trainMovementModelFromReplays(
      { replays: [{ events: replay([{ obs: "a", act: "one" }]) }] },
      { registry, backendId: "ngram-backoff" },
    );
    expect(model.backendId).toBe("ngram-backoff");
  });
});
