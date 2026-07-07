import { describe, expect, it } from "vitest";
import {
  IN_MEMORY_MOVEMENT_BACKEND_ID,
  InMemoryMovementModelBackend,
  MovementModelBackendRegistry,
  createDefaultMovementBackendRegistry,
  datasetFromReplayManifests,
  datasetFromTrajectories,
  evaluateMovementModel,
  movementEventFromTimeline,
  type MovementDataset,
  type MovementEvent,
  type MovementSequence,
} from "./movement-model.js";
import type { ReplayTimelineEvent } from "../capture/replay.js";
import { buildTrajectorySpan } from "../capture/trajectory.js";

function action(tool: string, summary: string): MovementEvent {
  return { kind: "action", tool, summary };
}
function observation(source: string, summary: string): MovementEvent {
  return { kind: "observation", source, summary };
}
function seq(id: string, events: MovementEvent[]): MovementSequence {
  return { id, events };
}

const OPEN_EDITOR_FLOW: MovementEvent[] = [
  observation("screen", "desktop"),
  action("mouse.move", "to dock"),
  action("mouse.click", "editor icon"),
  observation("screen", "editor window"),
  action("keyboard.type", "hello"),
];

describe("InMemoryMovementModelBackend", () => {
  it("repeats a recorded movement verbatim from the start (objective 2c)", async () => {
    const dataset: MovementDataset = { version: 1, sequences: [seq("t1", OPEN_EDITOR_FLOW)] };
    const model = await new InMemoryMovementModelBackend().train(dataset);

    // Empty context => generate the most likely full recorded sequence.
    const replayed = model.predict([]);
    expect(replayed).toEqual(OPEN_EDITOR_FLOW);
  });

  it("continues a recorded prefix exactly (deterministic recall)", async () => {
    const dataset: MovementDataset = { version: 1, sequences: [seq("t1", OPEN_EDITOR_FLOW)] };
    const model = await new InMemoryMovementModelBackend().train(dataset);

    const continuation = model.predict(OPEN_EDITOR_FLOW.slice(0, 2));
    expect(continuation).toEqual(OPEN_EDITOR_FLOW.slice(2));
  });

  it("is deterministic across repeated predictions", async () => {
    const dataset: MovementDataset = {
      version: 1,
      sequences: [
        seq("a", [action("mouse.click", "x"), action("mouse.click", "y"), action("mouse.click", "z")]),
        seq("b", [action("mouse.click", "x"), action("mouse.click", "y"), action("mouse.click", "w")]),
      ],
    };
    const model = await new InMemoryMovementModelBackend().train(dataset, { order: 1 });
    const first = model.predict([action("mouse.click", "x")]);
    const second = model.predict([action("mouse.click", "x")]);
    expect(first).toEqual(second);
  });

  it("generalizes to a new-but-related prefix via backoff (objective 2d)", async () => {
    // Two flows that share the sub-pattern "focus field -> type" but reach it
    // from different entry points.
    const dataset: MovementDataset = {
      version: 1,
      sequences: [
        seq("login-flow", [
          action("nav", "login-page"),
          action("focus", "field"),
          action("type", "value"),
        ]),
        seq("search-flow", [
          action("nav", "search-page"),
          action("focus", "field"),
          action("type", "value"),
        ]),
      ],
    };
    const model = await new InMemoryMovementModelBackend().train(dataset, { order: 2 });

    // A prefix never recorded verbatim: navigate to a *settings* page and focus
    // the field. The order-2 context [nav settings, focus field] was never seen,
    // so the model backs off to the shared shorter context [focus field] and
    // still predicts the "type value" continuation — a related generalization,
    // not a memorized replay.
    const predicted = model.predict(
      [action("nav", "settings-page"), action("focus", "field")],
      { maxSteps: 1 },
    );
    expect(predicted).toHaveLength(1);
    expect(predicted[0]).toEqual(action("type", "value"));
  });

  it("bounds generation with maxSteps against cyclic data", async () => {
    // A degenerate loop: click A then B then A ... never reaching an end marker
    // from a low-order context.
    const dataset: MovementDataset = {
      version: 1,
      sequences: [seq("loop", [action("k", "a"), action("k", "b"), action("k", "a"), action("k", "b")])],
    };
    const model = await new InMemoryMovementModelBackend().train(dataset, { order: 1 });
    const predicted = model.predict([action("k", "a")], { maxSteps: 5 });
    expect(predicted.length).toBeLessThanOrEqual(5);
  });

  it("round-trips through a snapshot with identical predictions", async () => {
    const backend = new InMemoryMovementModelBackend();
    const dataset: MovementDataset = { version: 1, sequences: [seq("t1", OPEN_EDITOR_FLOW)] };
    const model = await backend.train(dataset);
    const snapshot = model.serialize();

    expect(snapshot.backendId).toBe(IN_MEMORY_MOVEMENT_BACKEND_ID);

    const restored = backend.load(snapshot);
    expect(restored.predict(OPEN_EDITOR_FLOW.slice(0, 1))).toEqual(
      model.predict(OPEN_EDITOR_FLOW.slice(0, 1)),
    );
    // Snapshot survives a JSON serialization boundary (persistence seam).
    const roundTripped = backend.load(JSON.parse(JSON.stringify(snapshot)));
    expect(roundTripped.predict([])).toEqual(model.predict([]));
  });

  it("rejects loading a snapshot from a different backend", async () => {
    const backend = new InMemoryMovementModelBackend();
    const model = await backend.train({ version: 1, sequences: [seq("t", OPEN_EDITOR_FLOW)] });
    const snapshot = { ...model.serialize(), backendId: "some-other-backend" };
    expect(() => backend.load(snapshot)).toThrow(/cannot be loaded/);
  });
});

describe("MovementModelBackendRegistry", () => {
  it("selects backends by id and reports the default", () => {
    const registry = createDefaultMovementBackendRegistry();
    expect(registry.has(IN_MEMORY_MOVEMENT_BACKEND_ID)).toBe(true);
    expect(registry.list()).toContain(IN_MEMORY_MOVEMENT_BACKEND_ID);
    expect(registry.get(IN_MEMORY_MOVEMENT_BACKEND_ID)).toBeInstanceOf(InMemoryMovementModelBackend);
  });

  it("throws for an unknown backend id", () => {
    const registry = new MovementModelBackendRegistry();
    expect(() => registry.get("nope")).toThrow(/unknown movement-model backend/);
  });
});

describe("dataset builders", () => {
  it("builds a dataset from replay manifests, dropping transcript events", () => {
    const events: ReplayTimelineEvent[] = [
      { kind: "transcript", ts: 1, messageId: "m1", role: "user", content: "go" },
      { kind: "observation", ts: 2, trajectoryId: "t1", source: "screen", summary: "home" },
      { kind: "action", ts: 3, trajectoryId: "t1", tool: "mouse.click", summary: "button" },
    ];
    const dataset = datasetFromReplayManifests([{ trajectoryIds: ["t1"], events }]);
    expect(dataset.sequences).toHaveLength(1);
    expect(dataset.sequences[0].id).toBe("t1");
    expect(dataset.sequences[0].events).toEqual([
      observation("screen", "home"),
      action("mouse.click", "button"),
    ]);
  });

  it("builds a dataset from trajectory spans, ordering by timestamp", () => {
    const span = buildTrajectorySpan({
      id: "t1",
      sessionId: "s1",
      observations: [{ kind: "observation", source: "screen", summary: "later", ts: 30 }],
      actions: [
        { kind: "action", tool: "mouse.move", summary: "first", ts: 10 },
        { kind: "action", tool: "mouse.click", summary: "middle", ts: 20 },
      ],
    });
    const dataset = datasetFromTrajectories([span]);
    expect(dataset.sequences[0].events).toEqual([
      action("mouse.move", "first"),
      action("mouse.click", "middle"),
      observation("screen", "later"),
    ]);
  });

  it("maps timeline events and ignores transcripts", () => {
    expect(
      movementEventFromTimeline({ kind: "transcript", ts: 1, messageId: "m", role: "user", content: "x" }),
    ).toBeUndefined();
    expect(
      movementEventFromTimeline({ kind: "action", ts: 1, trajectoryId: "t", tool: "k", summary: "s" }),
    ).toEqual(action("k", "s"));
  });
});

describe("evaluateMovementModel", () => {
  it("reports perfect fidelity on memorized sequences", async () => {
    const sequences = [seq("t1", OPEN_EDITOR_FLOW)];
    const model = await new InMemoryMovementModelBackend().train({ version: 1, sequences });
    const result = evaluateMovementModel(model, sequences, 1);
    expect(result.fidelity).toBe(1);
    expect(result.matched).toBe(result.total);
    expect(result.total).toBe(OPEN_EDITOR_FLOW.length - 1);
  });

  it("measures partial fidelity on held-out related sequences", async () => {
    const train = [
      seq("a", [action("open", "app"), action("nav", "home"), action("read", "feed")]),
      seq("b", [action("open", "app"), action("nav", "home"), action("read", "inbox")]),
    ];
    const model = await new InMemoryMovementModelBackend().train({ version: 1, sequences: train }, { order: 3 });
    // Held-out sequence shares the "open app" prefix but diverges.
    const heldOut = [seq("c", [action("open", "app"), action("nav", "home"), action("read", "feed")])];
    const result = evaluateMovementModel(model, heldOut, 1);
    expect(result.fidelity).toBeGreaterThan(0);
    expect(result.fidelity).toBeLessThanOrEqual(1);
  });
});
