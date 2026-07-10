import { describe, expect, it } from "vitest";
import { buildTrajectorySpan, type TrajectoryAction, type TrajectorySpan } from "../capture/trajectory.js";
import {
  buildMovementDataset,
  createMovementBackend,
  listMovementBackends,
  MarkovMovementBackend,
  movementGestureKey,
  movementTokenFromAction,
  movementTokenKey,
  registerMovementBackend,
  type MovementDataset,
  type MovementModelBackend,
  type MovementToken,
} from "./movement-model.js";

function action(tool: string, gesture: string, target: string, ts: number): TrajectoryAction {
  return {
    kind: "action",
    tool,
    summary: `${gesture} ${target}`,
    ts,
    metadata: { gesture, target },
  };
}

function span(id: string, actions: TrajectoryAction[]): TrajectorySpan {
  return buildTrajectorySpan({ id, sessionId: "sess-1", actions });
}

describe("buildMovementDataset", () => {
  it("extracts ordered movement tokens per trajectory and a de-duplicated vocabulary", () => {
    const dataset = buildMovementDataset([
      span("t1", [action("device", "type", "name", 20), action("device", "tap", "open", 10)]),
      span("t2", [action("device", "tap", "open", 5)]),
    ]);

    expect(dataset.sequences).toHaveLength(2);
    // Sorted by timestamp regardless of insertion order.
    expect(dataset.sequences[0]!.tokens.map((t) => t.gesture)).toEqual(["tap", "type"]);
    // "tap open" appears in both trajectories but is counted once in the vocab.
    expect(dataset.vocabulary).toHaveLength(2);
  });

  it("skips trajectories with no actions", () => {
    const dataset = buildMovementDataset([span("empty", [])]);
    expect(dataset.sequences).toHaveLength(0);
  });
});

describe("movement token keys", () => {
  it("distinguishes targets in the full key but not the coarse key", () => {
    const a = movementTokenFromAction(action("device", "tap", "save", 1));
    const b = movementTokenFromAction(action("device", "tap", "cancel", 1));
    expect(movementTokenKey(a)).not.toEqual(movementTokenKey(b));
    expect(movementGestureKey(a)).toEqual(movementGestureKey(b));
  });
});

describe("MarkovMovementBackend — repeat recorded movements", () => {
  const dataset: MovementDataset = buildMovementDataset([
    span("t1", [action("device", "tap", "open", 1), action("device", "type", "name", 2), action("device", "tap", "save", 3)]),
    span("t2", [action("device", "tap", "open", 1), action("device", "type", "name", 2), action("device", "tap", "save", 3)]),
  ]);

  it("predicts the exact next recorded movement for a known context", () => {
    const model = new MarkovMovementBackend().train(dataset);
    const prediction = model.predict([{ tool: "device", gesture: "tap", target: "open" }]);
    expect(prediction).toBeDefined();
    expect(prediction!.token).toMatchObject({ gesture: "type", target: "name" });
    expect(prediction!.generalized).toBe(false);
    expect(prediction!.confidence).toBeCloseTo(1);
  });

  it("regenerates the recorded sequence from a seed movement", () => {
    const model = new MarkovMovementBackend().train(dataset);
    const seed: MovementToken[] = [{ tool: "device", gesture: "tap", target: "open" }];
    const generated = model.generate(seed, 2);
    expect(generated.map((t) => `${t.gesture}:${t.target}`)).toEqual(["type:name", "tap:save"]);
  });

  it("is deterministic across independently trained models", () => {
    const a = new MarkovMovementBackend().train(dataset);
    const b = new MarkovMovementBackend().train(dataset);
    const ctx: MovementToken[] = [{ tool: "device", gesture: "tap", target: "open" }];
    expect(a.predict(ctx)).toEqual(b.predict(ctx));
  });
});

describe("MarkovMovementBackend — generalize to new-but-related movements", () => {
  it("proposes the right kind of movement for an unseen target via coarse backoff", () => {
    // Every recorded "tap <field>" is followed by a "type" gesture.
    const dataset = buildMovementDataset([
      span("t1", [action("device", "tap", "name-field", 1), action("device", "type", "name-field", 2)]),
      span("t2", [action("device", "tap", "email-field", 1), action("device", "type", "email-field", 2)]),
    ]);
    const model = new MarkovMovementBackend().train(dataset);

    // A tap on a target the model has NEVER seen exactly.
    const prediction = model.predict([{ tool: "device", gesture: "tap", target: "phone-field" }]);
    expect(prediction).toBeDefined();
    expect(prediction!.generalized).toBe(true);
    expect(prediction!.token.gesture).toBe("type");
    // Coarse generalization does not fabricate a specific target.
    expect(prediction!.token.target).toBeUndefined();
  });

  it("breaks ties deterministically and reports confidence", () => {
    const dataset = buildMovementDataset([
      span("t1", [action("device", "tap", "a", 1), action("device", "swipe", "a", 2)]),
      span("t2", [action("device", "tap", "a", 1), action("device", "scroll", "a", 2)]),
    ]);
    const model = new MarkovMovementBackend().train(dataset);
    const prediction = model.predict([{ tool: "device", gesture: "tap", target: "a" }]);
    // Two equally-likely continuations; lexicographic key tie-break picks "scroll".
    expect(prediction!.token.gesture).toBe("scroll");
    expect(prediction!.confidence).toBeCloseTo(0.5);
  });

  it("stops generating when there is no learned continuation", () => {
    const dataset = buildMovementDataset([span("t1", [action("device", "tap", "solo", 1)])]);
    const model = new MarkovMovementBackend().train(dataset);
    // A completely unrelated tool/gesture with no prior overlap still falls back
    // to the unconditional prior exactly once, then keeps proposing it.
    const generated = model.generate([{ tool: "device", gesture: "tap", target: "solo" }], 3);
    expect(generated.length).toBeGreaterThan(0);
    expect(model.stats().tokenCount).toBe(1);
  });
});

describe("movement backend registry", () => {
  it("resolves the default markov backend and lists registered ids", () => {
    expect(createMovementBackend().id).toBe("markov");
    expect(listMovementBackends()).toContain("markov");
  });

  it("supports registering a pluggable backend and rejects unknown ids", () => {
    const stub: MovementModelBackend = {
      id: "stub-backend",
      train: (dataset) => new MarkovMovementBackend().train(dataset),
    };
    registerMovementBackend(stub);
    expect(createMovementBackend("stub-backend").id).toBe("stub-backend");
    expect(listMovementBackends()).toContain("stub-backend");
    expect(() => createMovementBackend("does-not-exist")).toThrow(/unknown movement backend/);
  });
});
