import { describe, expect, it } from "vitest";
import type { ReplayManifest, ReplayTimelineEvent } from "../capture/replay.js";
import type { TrajectorySpan } from "../capture/trajectory.js";
import {
  MOVEMENT_BOS,
  MarkovMovementBackend,
  MovementModelRegistry,
  buildMovementDatasetFromReplays,
  buildMovementDatasetFromTrajectories,
  movementSequenceFidelity,
  tokenizeMovementEvents,
  type MovementDataset,
} from "./movement-model.js";

function replay(sessionId: string, events: ReplayTimelineEvent[]): ReplayManifest {
  return {
    version: 1,
    sessionId,
    trajectoryIds: [sessionId],
    eventCount: events.length,
    events,
  };
}

function action(ts: number, tool: string, summary: string): ReplayTimelineEvent {
  return { kind: "action", ts, trajectoryId: "t", tool, summary };
}

describe("movement tokenizer", () => {
  it("canonicalizes actions and observations and can drop transcript", () => {
    const events: ReplayTimelineEvent[] = [
      { kind: "transcript", ts: 1, messageId: "m1", role: "user", content: "go" },
      { kind: "observation", ts: 2, trajectoryId: "t", source: "device", summary: "  Editor   ACTIVE " },
      action(3, "device", "Swiped Down"),
    ];
    expect(tokenizeMovementEvents(events)).toEqual([
      "obs:device:editor active",
      "act:device:swiped down",
    ]);
    expect(tokenizeMovementEvents(events, { includeTranscript: true })).toEqual([
      "msg:user",
      "obs:device:editor active",
      "act:device:swiped down",
    ]);
    expect(tokenizeMovementEvents(events, { includeObservations: false })).toEqual([
      "act:device:swiped down",
    ]);
  });
});

describe("MarkovMovementBackend — repeat recorded movements", () => {
  it("reproduces a recorded sequence via greedy generation", () => {
    const events = [action(1, "device", "tap compose"), action(2, "device", "type subject"), action(3, "device", "tap send")];
    const dataset = buildMovementDatasetFromReplays([replay("s1", events)]);
    const model = new MarkovMovementBackend().train(dataset);
    const generated = model.generate();
    expect(generated).toEqual([
      "act:device:tap compose",
      "act:device:type subject",
      "act:device:tap send",
    ]);
    expect(movementSequenceFidelity(dataset.sequences[0].tokens, generated)).toBe(1);
  });

  it("predicts the next movement from a recorded context", () => {
    const events = [action(1, "device", "open app"), action(2, "device", "scroll down"), action(3, "device", "tap item")];
    const dataset = buildMovementDatasetFromReplays([replay("s1", events)]);
    const model = new MarkovMovementBackend().train(dataset);
    expect(model.predictNext(["act:device:open app"])).toBe("act:device:scroll down");
    // End of a learned sequence => model stops (EOS surfaces as undefined).
    expect(model.predictNext(["act:device:scroll down", "act:device:tap item"])).toBeUndefined();
  });
});

describe("MarkovMovementBackend — generalize to related movements", () => {
  it("backs off to shorter context for an unseen high-order prefix", () => {
    // Two sequences that share the transition "focus field" -> "type text",
    // but reached "focus field" through different prior actions.
    const a = [action(1, "device", "open form"), action(2, "device", "focus field"), action(3, "device", "type text")];
    const b = [action(1, "device", "reopen form"), action(2, "device", "focus field"), action(3, "device", "type text")];
    const dataset = buildMovementDatasetFromReplays([replay("a", a), replay("b", b)]);
    const model = new MarkovMovementBackend({ order: 3 }).train(dataset);

    // A novel prior action never seen before "focus field": no order-2 context
    // matches, so the model backs off to the order-1 context and still predicts
    // the correct generalized continuation.
    const novelContext = ["act:device:navigate settings", "act:device:focus field"];
    expect(model.predictNext(novelContext)).toBe("act:device:type text");
  });

  it("orders candidates by frequency with a deterministic tie-break", () => {
    // "loop" is followed by "step" three times and "stop" once => argmax picks "step".
    const seqs = [
      [action(1, "d", "loop"), action(2, "d", "step")],
      [action(1, "d", "loop"), action(2, "d", "step")],
      [action(1, "d", "loop"), action(2, "d", "step")],
      [action(1, "d", "loop"), action(2, "d", "stop")],
    ].map((events, i) => replay(`s${i}`, events));
    const model = new MarkovMovementBackend().train(buildMovementDatasetFromReplays(seqs));
    expect(model.predictNext(["act:d:loop"])).toBe("act:d:step");
  });
});

describe("serialization + registry", () => {
  it("round-trips a trained model through serialize/deserialize", () => {
    const events = [action(1, "device", "a"), action(2, "device", "b"), action(3, "device", "c")];
    const dataset = buildMovementDatasetFromReplays([replay("s1", events)]);
    const model = new MarkovMovementBackend().train(dataset);
    const restored = new MarkovMovementBackend().deserialize(model.serialize());
    expect(restored.generate()).toEqual(model.generate());
    expect(restored.backend).toBe("markov");
  });

  it("resolves backends by their discriminator via the registry", () => {
    const registry = new MovementModelRegistry();
    expect(registry.list()).toContain("markov");
    const model = registry.get("markov").train({ version: 1, sequences: [{ id: "x", tokens: ["act:d:go"] }] });
    const restored = registry.deserialize(model.serialize());
    expect(restored.generate()).toEqual(["act:d:go"]);
    expect(() => registry.get("nonexistent")).toThrow(/unknown movement-model backend/);
  });
});

describe("dataset builders", () => {
  it("builds sequences from trajectories ordered by timestamp", () => {
    const trajectory: TrajectorySpan = {
      id: "t1",
      sessionId: "s1",
      createdAt: "2026-07-27T00:00:00.000Z",
      captureTier: "full",
      observations: [{ kind: "observation", source: "device", summary: "editor", ts: 10 }],
      actions: [
        { kind: "action", tool: "device", summary: "second", ts: 30 },
        { kind: "action", tool: "device", summary: "first", ts: 20 },
      ],
    };
    const dataset: MovementDataset = buildMovementDatasetFromTrajectories([trajectory]);
    expect(dataset.sequences[0].tokens).toEqual([
      "obs:device:editor",
      "act:device:first",
      "act:device:second",
    ]);
  });

  it("ignores empty trajectories and preserves the BOS constant", () => {
    const empty: TrajectorySpan = {
      id: "t0",
      sessionId: "s0",
      createdAt: "2026-07-27T00:00:00.000Z",
      captureTier: "off",
      observations: [],
      actions: [],
    };
    expect(buildMovementDatasetFromTrajectories([empty]).sequences).toHaveLength(0);
    expect(MOVEMENT_BOS).toBe("␂");
  });
});
