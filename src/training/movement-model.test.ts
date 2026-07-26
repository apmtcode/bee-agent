import { describe, expect, it } from "vitest";
import {
  NGramMovementBackend,
  movementFromDeviceInput,
  movementTokenKey,
  movementsFromReplayEvents,
  rolloutMovements,
  type MovementDataset,
  type MovementEvent,
} from "./movement-model.js";
import type { ReplayTimelineEvent } from "../capture/replay.js";
import type { DeviceCaptureInput } from "../capture/device-adapter.js";

function seq(id: string, events: MovementEvent[]) {
  return { id, events };
}

const recorded: MovementDataset = {
  sequences: [
    seq("a", [
      { context: "mail", channel: "os", action: "focus-changed", target: "mail" },
      { context: "mail", channel: "device", action: "scroll:down", target: "content" },
      { context: "mail", channel: "device", action: "tap", target: "compose" },
      { context: "mail", channel: "tool", action: "compose", target: "mail:compose" },
    ]),
    seq("b", [
      { context: "mail", channel: "os", action: "focus-changed", target: "mail" },
      { context: "mail", channel: "device", action: "scroll:down", target: "content" },
      { context: "mail", channel: "device", action: "tap", target: "compose" },
      { context: "mail", channel: "tool", action: "compose", target: "mail:compose" },
    ]),
  ],
};

describe("NGramMovementBackend", () => {
  it("reproduces a recorded movement sequence exactly via rollout", async () => {
    const backend = new NGramMovementBackend();
    const model = await backend.train(recorded, { order: 3 });

    const seed = [recorded.sequences[0].events[0]];
    const produced = rolloutMovements(backend, model, seed, 3);

    expect(produced.map(movementTokenKey)).toEqual(
      recorded.sequences[0].events.slice(1).map(movementTokenKey),
    );
  });

  it("predicts the recorded continuation with an exact-context (ngram) hit", async () => {
    const backend = new NGramMovementBackend();
    const model = await backend.train(recorded, { order: 3 });

    const context = recorded.sequences[0].events.slice(0, 2);
    const prediction = backend.predict(model, context);

    expect(prediction.level).toMatch(/^ngram-/);
    expect(prediction.confidence).toBeGreaterThan(0);
    expect(prediction.event?.action).toBe("tap");
  });

  it("generalizes to an unseen context via the feature backoff layer", async () => {
    const backend = new NGramMovementBackend();
    const model = await backend.train(recorded, { order: 3 });

    // A never-recorded exact context (scroll:up) but a known app+channel feature.
    const unseenContext: MovementEvent[] = [
      { context: "mail", channel: "os", action: "focus-changed", target: "mail" },
      { context: "mail", channel: "device", action: "scroll:up", target: "content" },
    ];
    const prediction = backend.predict(model, unseenContext);

    expect(prediction.level).toBe("feature");
    expect(prediction.event).toBeDefined();
    // The only movement seen after a mail/device movement is the tap-to-compose.
    expect(prediction.event?.action).toBe("tap");
  });

  it("returns an empty prediction for an untrained model", async () => {
    const backend = new NGramMovementBackend();
    const model = await backend.train({ sequences: [] });
    const prediction = backend.predict(model, [
      { context: "x", channel: "device", action: "tap" },
    ]);
    expect(prediction.level).toBe("empty");
    expect(prediction.event).toBeUndefined();
  });

  it("produces a JSON-serializable artifact that round-trips", async () => {
    const backend = new NGramMovementBackend();
    const model = await backend.train(recorded, { order: 2 });
    const reloaded = JSON.parse(JSON.stringify(model));

    const context = recorded.sequences[0].events.slice(0, 2);
    expect(backend.predict(reloaded, context).event?.action).toBe(
      backend.predict(model, context).event?.action,
    );
  });
});

describe("movement adapters", () => {
  it("maps replay timeline events into movements", () => {
    const events: ReplayTimelineEvent[] = [
      { kind: "observation", ts: 1, trajectoryId: "t1", source: "os", summary: "focused editor" },
      { kind: "action", ts: 2, trajectoryId: "t1", tool: "device", summary: "tapped save" },
    ];
    const movements = movementsFromReplayEvents(events);
    expect(movements).toHaveLength(2);
    expect(movements[0]).toMatchObject({ channel: "os", action: "os", context: "t1" });
    expect(movements[1]).toMatchObject({ channel: "tool", action: "device", target: "tapped save" });
  });

  it("maps a device gesture capture into a movement with direction encoded", () => {
    const input: DeviceCaptureInput = {
      sessionId: "s1",
      deviceId: "d1",
      platform: "macos",
      appId: "browser",
      appName: "Browser",
      visibleIndicator: true,
      ts: 1,
      gesture: { kind: "swipe", direction: "left", ts: 1, target: "gallery" },
    };
    expect(movementFromDeviceInput(input)).toEqual({
      context: "browser",
      channel: "device",
      action: "swipe:left",
      target: "gallery",
    });
  });

  it("returns undefined for a device capture with no gesture", () => {
    const input: DeviceCaptureInput = {
      sessionId: "s1",
      deviceId: "d1",
      platform: "macos",
      appId: "browser",
      appName: "Browser",
      visibleIndicator: true,
      ts: 1,
    };
    expect(movementFromDeviceInput(input)).toBeUndefined();
  });
});
