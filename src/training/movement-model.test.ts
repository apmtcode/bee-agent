import { describe, expect, it } from "vitest";
import { buildTrajectorySpan, type TrajectorySpan } from "../capture/trajectory.js";
import {
  actionToken,
  extractMovementExamples,
  NgramMovementBackend,
  NgramMovementModel,
  observationToken,
  trainMovementModel,
} from "./movement-model.js";

function span(
  id: string,
  events: Array<{ ts: number } & ({ source: string } | { tool: string; summary: string })>,
): TrajectorySpan {
  const observations = events
    .filter((event): event is { ts: number; source: string } => "source" in event)
    .map((event) => ({ kind: "observation" as const, source: event.source, summary: event.source, ts: event.ts }));
  const actions = events
    .filter((event): event is { ts: number; tool: string; summary: string } => "tool" in event)
    .map((event) => ({ kind: "action" as const, tool: event.tool, summary: event.summary, ts: event.ts }));
  return buildTrajectorySpan({ id, sessionId: "s", observations, actions });
}

describe("extractMovementExamples", () => {
  it("emits one example per action with the preceding context, ordered by ts", () => {
    const examples = extractMovementExamples([
      span("a", [
        { ts: 1, source: "window" },
        { ts: 2, tool: "app.open", summary: "open" },
        { ts: 3, tool: "pointer.click", summary: "click" },
      ]),
    ]);

    expect(examples).toEqual([
      { context: [observationToken("window")], action: { tool: "app.open", summary: "open" } },
      {
        context: [observationToken("window"), actionToken("app.open")],
        action: { tool: "pointer.click", summary: "click" },
      },
    ]);
  });

  it("truncates context to maxOrder and never leaks across spans", () => {
    const examples = extractMovementExamples(
      [
        span("a", [
          { ts: 1, tool: "one", summary: "1" },
          { ts: 2, tool: "two", summary: "2" },
          { ts: 3, tool: "three", summary: "3" },
        ]),
        span("b", [{ ts: 1, tool: "solo", summary: "s" }]),
      ],
      { maxOrder: 1 },
    );

    // "three" sees only "two" (order 1), and span b's action starts fresh.
    expect(examples.at(-2)).toEqual({
      context: [actionToken("two")],
      action: { tool: "three", summary: "3" },
    });
    expect(examples.at(-1)).toEqual({ context: [], action: { tool: "solo", summary: "s" } });
  });
});

describe("NgramMovementBackend", () => {
  it("repeats a recorded movement via exact-context match", async () => {
    const backend = new NgramMovementBackend();
    const model = await trainMovementModel(backend, [
      span("a", [
        { ts: 1, source: "window" },
        { ts: 2, tool: "app.open", summary: "open editor" },
        { ts: 3, tool: "pointer.click", summary: "click document" },
      ]),
    ]);

    const prediction = model.predict([observationToken("window"), actionToken("app.open")]);
    expect(prediction).toBeDefined();
    expect(prediction?.action.tool).toBe("pointer.click");
    expect(prediction?.action.summary).toBe("click document");
    expect(prediction?.backoff).toBe(false);
    expect(prediction?.matchedOrder).toBe(2);
    expect(prediction?.confidence).toBe(1);
  });

  it("generalizes to an unseen context by backing off to a shorter suffix", async () => {
    const backend = new NgramMovementBackend();
    // After typing, the recorded movement is always "save".
    const model = await trainMovementModel(backend, [
      span("a", [
        { ts: 1, tool: "keyboard.type", summary: "type" },
        { ts: 2, tool: "keyboard.save", summary: "save" },
      ]),
      span("b", [
        { ts: 1, tool: "keyboard.type", summary: "type" },
        { ts: 2, tool: "keyboard.save", summary: "save" },
      ]),
    ]);

    // Novel longer context never seen verbatim; the model backs off from the
    // unseen 2-token context to the shared 1-token suffix ("type") and still
    // predicts the recorded next movement — this is the generalization path.
    const prediction = model.predict([actionToken("pointer.drag"), actionToken("keyboard.type")]);
    expect(prediction?.action.tool).toBe("keyboard.save");
    expect(prediction?.backoff).toBe(true);
    expect(prediction?.matchedOrder).toBe(1);
  });

  it("falls back to the global prior when no context matches", async () => {
    const backend = new NgramMovementBackend();
    const model = await trainMovementModel(backend, [
      span("a", [
        { ts: 1, tool: "common", summary: "c" },
        { ts: 2, tool: "common", summary: "c" },
        { ts: 3, tool: "rare", summary: "r" },
      ]),
    ]);

    const prediction = model.predict([actionToken("totally-unseen-token")]);
    expect(prediction?.action.tool).toBe("common");
    expect(prediction?.matchedOrder).toBe(0);
    expect(prediction?.backoff).toBe(true);
  });

  it("returns undefined when trained on no examples", async () => {
    const backend = new NgramMovementBackend();
    const model = await backend.train([]);
    expect(model.exampleCount).toBe(0);
    expect(model.predict([actionToken("x")])).toBeUndefined();
  });

  it("breaks ties deterministically by tool name", async () => {
    const backend = new NgramMovementBackend();
    const model = await trainMovementModel(backend, [
      span("a", [{ ts: 1, tool: "zzz", summary: "z" }]),
      span("b", [{ ts: 1, tool: "aaa", summary: "a" }]),
    ]);
    // Global prior: both count 1 -> lexicographically smallest wins.
    expect(model.predict([])?.action.tool).toBe("aaa");
  });
});

describe("NgramMovementModel serialization", () => {
  it("round-trips to an identical model", async () => {
    const backend = new NgramMovementBackend();
    const model = await trainMovementModel(backend, [
      span("a", [
        { ts: 1, source: "window" },
        { ts: 2, tool: "app.open", summary: "open" },
        { ts: 3, tool: "pointer.click", summary: "click" },
      ]),
    ]);

    const restored = NgramMovementModel.deserialize(model.serialize());
    expect(restored.serialize()).toBe(model.serialize());
    expect(restored.exampleCount).toBe(model.exampleCount);
    expect(restored.predict([observationToken("window"), actionToken("app.open")])?.action.tool).toBe(
      "pointer.click",
    );
  });
});
