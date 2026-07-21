import { describe, expect, it } from "vitest";
import {
  NearestNeighborMovementBackend,
  createDefaultMovementBackendRegistry,
  createMovementDataset,
  evaluateMovementPolicy,
  movementEndpoint,
  movementEndpointError,
  parseMovementDataset,
  serializeMovementDataset,
  synthesizeDemonstration,
  synthesizeMovement,
  type MovementDemonstration,
  type MovementTask,
} from "./movement-model.js";

describe("synthesizeMovement", () => {
  it("produces a monotonic, replayable event stream ending at the target", () => {
    const task: MovementTask = {
      gesture: "click",
      start: { x: 0, y: 0 },
      target: { x: 100, y: 40 },
    };
    const events = synthesizeMovement(task, { steps: 4, stepMs: 10 });

    expect(events[0]).toMatchObject({ kind: "pointer-move", x: 0, y: 0 });
    expect(events.at(-1)).toMatchObject({ kind: "pointer-up", x: 100, y: 40 });
    for (let i = 1; i < events.length; i += 1) {
      expect(events[i].ts).toBeGreaterThanOrEqual(events[i - 1].ts);
    }
    expect(movementEndpoint(events)).toEqual({ x: 100, y: 40 });
  });

  it("emits key events for typed text", () => {
    const events = synthesizeMovement(
      { gesture: "type", start: { x: 0, y: 0 }, target: { x: 10, y: 10 }, text: "hi" },
      { steps: 1 },
    );
    const keys = events.filter((event) => event.kind === "key");
    expect(keys).toHaveLength(4); // down/up for h, down/up for i
    expect(keys.map((event) => (event.kind === "key" ? event.key : ""))).toEqual(["h", "h", "i", "i"]);
  });
});

describe("movement dataset format", () => {
  it("round-trips through serialize/parse", () => {
    const dataset = createMovementDataset([
      synthesizeDemonstration("d1", { gesture: "click", start: { x: 0, y: 0 }, target: { x: 50, y: 50 } }),
    ]);
    const restored = parseMovementDataset(serializeMovementDataset(dataset));
    expect(restored).toEqual(dataset);
  });

  it("rejects malformed datasets", () => {
    expect(() => parseMovementDataset(JSON.stringify({ version: 2 }))).toThrow(/invalid movement dataset/);
  });
});

describe("NearestNeighborMovementBackend", () => {
  function trainingDemos(): MovementDemonstration[] {
    // Curved demonstrations toward a few known targets — the "recorded style".
    return [
      synthesizeDemonstration(
        "d-right",
        { gesture: "click", start: { x: 0, y: 0 }, target: { x: 200, y: 0 } },
        { steps: 8, curvature: 0.2 },
      ),
      synthesizeDemonstration(
        "d-down",
        { gesture: "click", start: { x: 0, y: 0 }, target: { x: 0, y: 200 } },
        { steps: 8, curvature: 0.2 },
      ),
    ];
  }

  it("reproduces a seen movement's endpoint exactly", () => {
    const policy = new NearestNeighborMovementBackend().train(createMovementDataset(trainingDemos()));
    const task: MovementTask = { gesture: "click", start: { x: 0, y: 0 }, target: { x: 200, y: 0 } };
    expect(movementEndpointError(policy.predict(task), task)).toBeCloseTo(0, 6);
  });

  it("generalizes to an UNSEEN but related target with zero endpoint error", () => {
    const policy = new NearestNeighborMovementBackend().train(createMovementDataset(trainingDemos()));
    // A target the model never saw, between/beyond the demonstrated ones.
    const heldOut: MovementTask = { gesture: "click", start: { x: 10, y: 5 }, target: { x: 320, y: 25 } };

    const events = policy.predict(heldOut);
    expect(events.length).toBeGreaterThan(2);
    // Endpoint reproduced precisely via the frame transform.
    expect(movementEndpointError(events, heldOut)).toBeCloseTo(0, 6);
    // Path starts at the requested start, not the demonstrated one.
    expect(events[0]).toMatchObject({ kind: "pointer-move", x: 10, y: 5 });
  });

  it("carries recorded motion shape (curvature) into the generalized movement", () => {
    const policy = new NearestNeighborMovementBackend().train(createMovementDataset(trainingDemos()));
    const heldOut: MovementTask = { gesture: "click", start: { x: 0, y: 0 }, target: { x: 400, y: 0 } };
    const result = evaluateMovementPolicy(policy, [heldOut]);
    // A straight retarget would have ~0 shape error; the curved demos transfer bow.
    expect(result.meanShapeError).toBeGreaterThan(0.01);
    expect(result.meanEndpointError).toBeCloseTo(0, 6);
  });

  it("generalizes typed text to new content", () => {
    const demos = [
      synthesizeDemonstration(
        "type-demo",
        { gesture: "type", start: { x: 0, y: 0 }, target: { x: 100, y: 100 }, text: "ab" },
        { steps: 4 },
      ),
    ];
    const policy = new NearestNeighborMovementBackend().train(createMovementDataset(demos));
    const events = policy.predict({
      gesture: "type",
      start: { x: 0, y: 0 },
      target: { x: 150, y: 80 },
      text: "xyz",
    });
    const typed = events
      .filter((event) => event.kind === "key" && event.down)
      .map((event) => (event.kind === "key" ? event.key : ""));
    expect(typed).toEqual(["x", "y", "z"]);
  });

  it("falls back to a straight synthesized movement with no demonstrations", () => {
    const policy = new NearestNeighborMovementBackend().train(createMovementDataset([]));
    const task: MovementTask = { gesture: "click", start: { x: 0, y: 0 }, target: { x: 30, y: 30 } };
    expect(policy.demonstrationCount).toBe(0);
    expect(movementEndpointError(policy.predict(task), task)).toBeCloseTo(0, 6);
  });
});

describe("evaluateMovementPolicy", () => {
  it("reports near-zero endpoint error across a held-out eval set", () => {
    const demos = [
      synthesizeDemonstration("a", { gesture: "click", start: { x: 0, y: 0 }, target: { x: 100, y: 0 } }, { steps: 6 }),
      synthesizeDemonstration("b", { gesture: "click", start: { x: 0, y: 0 }, target: { x: 0, y: 100 } }, { steps: 6 }),
    ];
    const policy = new NearestNeighborMovementBackend().train(createMovementDataset(demos));
    const heldOut: MovementTask[] = [
      { gesture: "click", start: { x: 5, y: 5 }, target: { x: 250, y: 40 } },
      { gesture: "click", start: { x: 20, y: 0 }, target: { x: 60, y: 300 } },
      { gesture: "click", start: { x: 0, y: 0 }, target: { x: 175, y: 175 } },
    ];
    const result = evaluateMovementPolicy(policy, heldOut);
    expect(result.count).toBe(3);
    expect(result.meanEndpointError).toBeCloseTo(0, 5);
    expect(result.maxEndpointError).toBeLessThan(1e-3);
  });
});

describe("MovementBackendRegistry", () => {
  it("exposes the default backend by id and is pluggable", () => {
    const registry = createDefaultMovementBackendRegistry();
    expect(registry.list()).toContain("nearest-neighbor-mock");
    const backend = registry.get("nearest-neighbor-mock");
    expect(backend).toBeDefined();
    const policy = backend!.train(createMovementDataset([]));
    expect(policy.backendId).toBe("nearest-neighbor-mock");
  });
});
