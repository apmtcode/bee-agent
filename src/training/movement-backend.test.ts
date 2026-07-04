import { describe, expect, it } from "vitest";
import type { ReplayManifest, ReplayTimelineEvent } from "../capture/replay.js";
import {
  canonicalMovementToken,
  deriveMovementSequence,
  movementContextKey,
  rolloutMovements,
  type MovementSequence,
} from "./movement-backend.js";
import { MarkovMovementBackend } from "./markov-movement-backend.js";

function action(ts: number, tool: string, summary: string): ReplayTimelineEvent {
  return { kind: "action", ts, trajectoryId: "traj-1", tool, summary };
}

function manifest(events: ReplayTimelineEvent[]): ReplayManifest {
  return {
    version: 1,
    sessionId: "session-1",
    trajectoryIds: ["traj-1"],
    eventCount: events.length,
    events,
  };
}

function steps(...pairs: Array<[string, string]>): MovementSequence {
  return pairs.map(([tool, summary]) => ({
    token: canonicalMovementToken(tool, summary),
    tool,
    summary,
  }));
}

describe("canonicalMovementToken", () => {
  it("collapses similar movements to a shared token so related movements generalize", () => {
    expect(canonicalMovementToken("mouse", "Click login button")).toBe("mouse:click");
    expect(canonicalMovementToken("mouse", "click submit button")).toBe("mouse:click");
    expect(canonicalMovementToken("keyboard", "Type password")).toBe("keyboard:type");
  });

  it("normalizes whitespace and empty summaries", () => {
    expect(canonicalMovementToken("  Window Focus  ", "")).toBe("window-focus");
  });
});

describe("deriveMovementSequence", () => {
  it("extracts only action events, ts-ordered, and drops transcript/observations", () => {
    const replay = manifest([
      { kind: "transcript", ts: 5, messageId: "m1", role: "user", content: "hi" },
      action(30, "mouse", "click submit"),
      { kind: "observation", ts: 10, trajectoryId: "traj-1", source: "os", summary: "window opened" },
      action(10, "mouse", "move to field"),
      action(20, "keyboard", "type email"),
    ]);

    const sequence = deriveMovementSequence(replay);
    expect(sequence.map((step) => step.token)).toEqual([
      "mouse:move",
      "keyboard:type",
      "mouse:click",
    ]);
    expect(sequence[0]?.metadata).toMatchObject({ trajectoryId: "traj-1", ts: 10 });
  });
});

describe("MarkovMovementBackend", () => {
  it("reproduces a recorded movement sequence from a seed (replay)", async () => {
    const backend = new MarkovMovementBackend(2);
    // A deterministic recorded routine: move -> type -> click -> submit.
    const recorded = steps(
      ["mouse", "move to field"],
      ["keyboard", "type email"],
      ["mouse", "click submit"],
      ["keyboard", "enter confirm"],
    );
    const model = await backend.train([recorded]);

    const replayed = rolloutMovements(backend, model, recorded.slice(0, 1), { maxSteps: 3 });
    expect(replayed.map((step) => step.token)).toEqual([
      "keyboard:type",
      "mouse:click",
      "keyboard:enter",
    ]);
  });

  it("generalizes to an unseen-but-related context via backoff", async () => {
    const backend = new MarkovMovementBackend(2);
    // Two routines that both end a "type" with a "click".
    const model = await backend.train([
      steps(["mouse", "move here"], ["keyboard", "type name"], ["mouse", "click ok"]),
      steps(["window", "focus app"], ["keyboard", "type note"], ["mouse", "click save"]),
    ]);

    // Novel order-2 context [scroll:page, keyboard:type]: scroll never preceded
    // a type in training, but the order-1 keyboard:type -> mouse:click is known.
    const novelContext = steps(["scroll", "page down"], ["keyboard", "type something new"]);
    const prediction = backend.predictNext(model, novelContext);

    expect(prediction).toBeDefined();
    expect(prediction?.token).toBe("mouse:click");
    // Backed off from order 2 to order 1 — this is the generalization signal.
    expect(prediction?.contextOrderUsed).toBe(1);
  });

  it("is deterministic and breaks probability ties lexicographically", async () => {
    const backend = new MarkovMovementBackend(1);
    const model = await backend.train([
      steps(["mouse", "click start"], ["mouse", "zebra move"]),
      steps(["mouse", "click start"], ["mouse", "alpha move"]),
    ]);
    // From mouse:click, both mouse:zebra and mouse:alpha follow once each (tie).
    const first = backend.predictNext(model, steps(["mouse", "click start"]));
    const second = backend.predictNext(model, steps(["mouse", "click start"]));
    expect(first?.token).toBe("mouse:alpha");
    expect(first).toEqual(second);
    expect(first?.candidates).toHaveLength(2);
  });

  it("falls back to the unigram distribution for a fully novel context", async () => {
    const backend = new MarkovMovementBackend(2);
    // Train on a single repeated token so the unigram fallback is well defined.
    const trained = await backend.train([steps(["mouse", "click a"], ["mouse", "click b"])]);
    const prediction = backend.predictNext(trained, steps(["keyboard", "type x"]));
    expect(prediction).toBeDefined();
    expect(prediction?.token).toBe("mouse:click");
    expect(prediction?.contextOrderUsed).toBe(0);
    expect(trained.vocabulary).toEqual(["mouse:click"]);
  });

  it("produces a plain-JSON model artifact that survives serialization", async () => {
    const backend = new MarkovMovementBackend(2);
    const model = await backend.train([
      steps(["mouse", "move a"], ["keyboard", "type b"], ["mouse", "click c"]),
    ]);
    const roundTripped = JSON.parse(JSON.stringify(model));
    expect(roundTripped).toEqual(model);
    expect(model.backendId).toBe("markov-v1");
    expect(model.trainedStepCount).toBe(3);
    expect(model.trainedSequenceCount).toBe(1);

    // A backend can predict against a deserialized model unchanged.
    const prediction = backend.predictNext(roundTripped, steps(["mouse", "move a"]));
    expect(prediction?.token).toBe("keyboard:type");
  });

  it("returns undefined when the model has no data", async () => {
    const backend = new MarkovMovementBackend();
    const empty = await backend.train([]);
    expect(backend.predictNext(empty, steps(["mouse", "click"]))).toBeUndefined();
    expect(empty.vocabulary).toEqual([]);
  });
});

describe("movementContextKey", () => {
  it("joins tokens with the reserved separator", () => {
    const key = movementContextKey(steps(["mouse", "click a"], ["keyboard", "type b"]));
    expect(key).toBe("mouse:click␟keyboard:type");
  });
});
