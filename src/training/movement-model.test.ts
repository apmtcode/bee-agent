import { describe, expect, it } from "vitest";
import type { ReplayManifest, ReplayTimelineEvent } from "../capture/replay.js";
import {
  MockMovementModelBackend,
  buildMovementDataset,
  contextSignature,
  restoreMovementModel,
  type MovementContext,
} from "./movement-model.js";

function manifest(sessionId: string, events: ReplayTimelineEvent[]): ReplayManifest {
  return {
    version: 1,
    sessionId,
    trajectoryIds: [sessionId],
    eventCount: events.length,
    events,
  };
}

function obs(ts: number, source: string, summary: string): ReplayTimelineEvent {
  return { kind: "observation", ts, trajectoryId: "t", source, summary };
}

function act(ts: number, tool: string, summary: string): ReplayTimelineEvent {
  return { kind: "action", ts, trajectoryId: "t", tool, summary };
}

function ctx(...texts: string[]): MovementContext {
  return { events: texts.map((text) => ({ kind: "observation" as const, text })) };
}

// A recorded "open the settings panel" movement: observe the menu, then click.
const openSettings = manifest("session-open-settings", [
  obs(1, "screen", "top menu bar visible with gear icon"),
  act(2, "mouse.move", "move cursor to gear icon at 1180,20"),
  act(3, "mouse.click", "click gear icon to open settings panel"),
]);

// A recorded "save the document" movement.
const saveDoc = manifest("session-save-doc", [
  obs(1, "screen", "document editor focused with unsaved changes"),
  act(2, "keyboard.press", "press cmd+s to save the document"),
]);

describe("buildMovementDataset", () => {
  it("derives one (context -> next action) example per recorded action", () => {
    const dataset = buildMovementDataset([openSettings]);
    expect(dataset.examples).toHaveLength(2);

    // First action: context is just the preceding observation.
    const [first, second] = dataset.examples;
    expect(first?.action).toEqual({ tool: "mouse.move", summary: "move cursor to gear icon at 1180,20" });
    expect(first?.context.events.map((event) => event.kind)).toEqual(["observation"]);
    expect(first?.sourceId).toBe("session-open-settings");

    // Second action's context includes the observation AND the first action.
    expect(second?.action.tool).toBe("mouse.click");
    expect(second?.context.events.map((event) => event.kind)).toEqual(["observation", "action"]);
  });

  it("respects the context window size", () => {
    const dataset = buildMovementDataset([openSettings], { windowSize: 1 });
    const clickExample = dataset.examples.find((example) => example.action.tool === "mouse.click");
    // Window of 1 keeps only the immediately preceding event (the move action).
    expect(clickExample?.context.events).toHaveLength(1);
    expect(clickExample?.context.events[0]?.kind).toBe("action");
  });
});

describe("MockMovementModelBackend", () => {
  it("recalls the exact recorded action for a seen context (repeat movement)", async () => {
    const backend = new MockMovementModelBackend();
    const model = await backend.train(buildMovementDataset([openSettings]));

    // At inference the recorder feeds back the same observation encoding
    // ("<source>: <summary>") the training pair was built from.
    const seen = ctx("screen: top menu bar visible with gear icon");
    const prediction = model.predict(seen);
    expect(prediction.source).toBe("recall");
    expect(prediction.confidence).toBe(1);
    expect(prediction.action).toEqual({ tool: "mouse.move", summary: "move cursor to gear icon at 1180,20" });
  });

  it("generalizes to a new but related context (generalize movement)", async () => {
    const backend = new MockMovementModelBackend();
    const model = await backend.train(buildMovementDataset([openSettings]), { similarityThreshold: 0.2 });

    // Novel phrasing, overlapping vocabulary ("menu", "gear", "icon", "visible").
    const related = ctx("a gear icon is visible in the menu");
    const prediction = model.predict(related);
    expect(prediction.source).toBe("generalized");
    expect(prediction.confidence).toBeGreaterThan(0);
    expect(prediction.confidence).toBeLessThan(1);
    expect(prediction.action.tool).toBe("mouse.move");
  });

  it("falls back to the global prior for an unrelated context", async () => {
    const backend = new MockMovementModelBackend();
    // saveDoc has a single action, making cmd+s the global most-frequent action.
    const model = await backend.train(buildMovementDataset([saveDoc]));

    const unrelated = ctx("completely different terrain elevation mountains rivers");
    const prediction = model.predict(unrelated);
    expect(prediction.source).toBe("fallback");
    expect(prediction.confidence).toBe(0);
    expect(prediction.action.tool).toBe("keyboard.press");
  });

  it("ranks the most frequently recorded action first for a signature", async () => {
    const backend = new MockMovementModelBackend();
    // Same context, but the "click" action recorded twice vs "double-click" once.
    const repeated = manifest("session-repeat", [
      obs(1, "screen", "button labeled submit"),
      act(2, "mouse.click", "click submit"),
      obs(3, "screen", "button labeled submit"),
      act(4, "mouse.click", "click submit"),
      obs(5, "screen", "button labeled submit"),
      act(6, "mouse.doubleclick", "double click submit"),
    ]);
    const model = await backend.train(buildMovementDataset([repeated], { windowSize: 1 }));
    const prediction = model.predict(ctx("screen: button labeled submit"));
    expect(prediction.source).toBe("recall");
    expect(prediction.action.summary).toBe("click submit");
  });

  it("produces a deterministic, serializable artifact that round-trips", async () => {
    const backend = new MockMovementModelBackend();
    const dataset = buildMovementDataset([openSettings, saveDoc]);

    const first = await backend.train(dataset);
    const second = await backend.train(dataset);
    // Training is deterministic.
    expect(second.serialize()).toEqual(first.serialize());

    // Serialize -> JSON -> restore preserves predictions exactly.
    const json = JSON.stringify(first.serialize());
    const restored = restoreMovementModel(JSON.parse(json));
    const probe = ctx("top menu bar visible with gear icon");
    expect(restored.predict(probe)).toEqual(first.predict(probe));
    expect(restored.exampleCount).toBe(first.exampleCount);
    expect(restored.backendId).toBe("mock-frequency-v1");
  });

  it("keeps an empty dataset safe (no fallback, degrades to noop)", async () => {
    const backend = new MockMovementModelBackend();
    const model = await backend.train({ examples: [] });
    expect(model.exampleCount).toBe(0);
    const prediction = model.predict(ctx("anything"));
    expect(prediction.source).toBe("fallback");
    expect(prediction.action.tool).toBe("noop");
  });
});

describe("contextSignature", () => {
  it("is order-independent and stop-word tolerant", () => {
    expect(contextSignature(ctx("open the settings panel"))).toBe(contextSignature(ctx("panel settings open")));
  });
});
