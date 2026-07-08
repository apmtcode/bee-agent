import { describe, expect, it } from "vitest";
import type { ReplayTimelineEvent } from "../capture/replay.js";
import {
  MarkovMovementBackend,
  MarkovMovementModel,
  MOVEMENT_END_TOKEN,
  MovementModelBackendRegistry,
  evaluateReplayFidelity,
  toMovementSequence,
  tokenizeReplayEvents,
  type MovementSequence,
} from "./model-backend.js";

function action(tool: string, ts: number): ReplayTimelineEvent {
  return { kind: "action", ts, trajectoryId: "t", tool, summary: `did ${tool}` };
}

function observation(source: string, ts: number): ReplayTimelineEvent {
  return { kind: "observation", ts, trajectoryId: "t", source, summary: `saw ${source}` };
}

function seq(id: string, tokens: string[]): MovementSequence {
  return { id, tokens: [...tokens, MOVEMENT_END_TOKEN] };
}

describe("tokenizeReplayEvents", () => {
  it("keeps movement events and drops transcript chatter", () => {
    const events: ReplayTimelineEvent[] = [
      { kind: "transcript", ts: 0, messageId: "m1", role: "user", content: "open the app" },
      observation("Window Focus", 1),
      action("Mouse Click", 2),
    ];
    expect(tokenizeReplayEvents(events)).toEqual(["observe:window-focus", "action:mouse-click"]);
  });

  it("appends the end sentinel when building a training sequence", () => {
    const sequence = toMovementSequence("traj-1", [action("Click", 1)]);
    expect(sequence.tokens).toEqual(["action:click", MOVEMENT_END_TOKEN]);
  });
});

describe("MarkovMovementBackend", () => {
  it("repeats a recorded movement exactly after training", async () => {
    const recorded = ["action:focus", "action:click", "action:type", "action:submit"];
    const model = await new MarkovMovementBackend().train([seq("traj-1", recorded)]);
    const replay = model.generate([recorded[0]!], { maxSteps: 10 });
    expect(replay).toEqual(recorded.slice(1));
  });

  it("stops generating at the end sentinel by default", async () => {
    const model = await new MarkovMovementBackend().train([seq("a", ["action:open", "action:close"])]);
    const replay = model.generate(["action:open"]);
    expect(replay).toEqual(["action:close"]);
    expect(replay).not.toContain(MOVEMENT_END_TOKEN);
  });

  it("ranks next-token predictions by observed frequency", async () => {
    // After "action:click": twice -> "action:type", once -> "action:drag".
    const dataset = [
      seq("a", ["action:click", "action:type"]),
      seq("b", ["action:click", "action:type"]),
      seq("c", ["action:click", "action:drag"]),
    ];
    const model = await new MarkovMovementBackend().train(dataset, { order: 1 });
    const predictions = model.predict(["action:click"]);
    expect(predictions[0]).toMatchObject({ token: "action:type" });
    expect(predictions[0]!.probability).toBeCloseTo(2 / 3);
    expect(predictions[1]).toMatchObject({ token: "action:drag" });
  });

  it("generalizes to an unseen context via backoff", async () => {
    // The model never saw "action:scroll" preceding "action:click", but the
    // unigram/backoff distribution still predicts the globally-common next move.
    const dataset = [
      seq("a", ["action:focus", "action:click", "action:type"]),
      seq("b", ["action:hover", "action:click", "action:type"]),
    ];
    const model = await new MarkovMovementBackend().train(dataset, { order: 2 });
    const next = model.predictNext(["action:scroll", "action:click"]);
    expect(next).toBe("action:type");
  });

  it("returns no prediction from an untrained/empty model", async () => {
    const model = await new MarkovMovementBackend().train([]);
    expect(model.predictNext(["action:click"])).toBeUndefined();
    expect(model.generate(["action:click"])).toEqual([]);
  });

  it("round-trips through serialize/deserialize", async () => {
    const recorded = ["action:focus", "action:click", "action:submit"];
    const model = await new MarkovMovementBackend().train([seq("t", recorded)]);
    const restored = MarkovMovementModel.deserialize(model.serialize());
    expect(restored.backend).toBe("markov-backoff");
    expect(restored.generate([recorded[0]!])).toEqual(recorded.slice(1));
    expect(restored.serialize()).toEqual(model.serialize());
  });
});

describe("evaluateReplayFidelity", () => {
  it("reports perfect fidelity when the model reproduces the sequence", async () => {
    const recorded = ["action:a", "action:b", "action:c"];
    const model = await new MarkovMovementBackend().train([seq("t", recorded)]);
    const result = evaluateReplayFidelity(model, seq("t", recorded), 1);
    expect(result.fidelity).toBe(1);
    expect(result.predicted).toEqual(["action:b", "action:c"]);
  });

  it("measures partial fidelity on a held-out but related sequence", async () => {
    const model = await new MarkovMovementBackend().train(
      [
        seq("a", ["action:open", "action:read", "action:close"]),
        seq("b", ["action:open", "action:read", "action:close"]),
      ],
      { order: 2 },
    );
    // Held-out sequence diverges at the end; the common prefix should still replay.
    const held = seq("held", ["action:open", "action:read", "action:save"]);
    const result = evaluateReplayFidelity(model, held, 1);
    expect(result.matchedPrefix).toBeGreaterThanOrEqual(1);
    expect(result.fidelity).toBeGreaterThan(0);
    expect(result.fidelity).toBeLessThanOrEqual(1);
  });
});

describe("MovementModelBackendRegistry", () => {
  it("registers the markov backend by default and looks it up by name", () => {
    const registry = new MovementModelBackendRegistry();
    expect(registry.get("markov-backoff")?.info.inProcess).toBe(true);
    expect(registry.list().map((info) => info.name)).toContain("markov-backoff");
    expect(registry.get("nonexistent")).toBeUndefined();
  });
});
