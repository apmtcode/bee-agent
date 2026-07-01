import { describe, expect, it } from "vitest";
import {
  MarkovMovementBackend,
  MovementBackendRegistry,
  extractMovementTrajectory,
  rolloutMovements,
  type MovementTrajectory,
} from "./movement-policy.js";
import { buildTrajectorySpan } from "../capture/trajectory.js";

function flow(id: string, app: string, steps: Array<[string, string?]>): MovementTrajectory {
  return {
    id,
    app,
    events: steps.map(([gesture, target], index) => ({
      app,
      gesture: gesture as MovementTrajectory["events"][number]["gesture"],
      ...(target ? { target } : {}),
      ts: index,
    })),
  };
}

describe("MarkovMovementBackend", () => {
  it("learns to repeat a recorded movement flow exactly (replay fidelity)", () => {
    const trajectory = flow("t1", "mail", [
      ["tap", "compose"],
      ["type", "subject"],
      ["type", "body"],
      ["tap", "send"],
    ]);
    const model = new MarkovMovementBackend().train([trajectory], { order: 3 });

    const rollout = rolloutMovements(model, { app: "mail" });
    expect(rollout.map((event) => `${event.gesture}:${event.target}`)).toEqual([
      "tap:compose",
      "type:subject",
      "type:body",
      "tap:send",
    ]);
  });

  it("terminates rollouts at the learned end of the flow", () => {
    const trajectory = flow("t1", "editor", [
      ["shortcut", "save"],
      ["tap", "close"],
    ]);
    const model = new MarkovMovementBackend().train([trajectory], { order: 2 });
    const rollout = rolloutMovements(model, { app: "editor" }, { maxSteps: 20 });
    expect(rollout).toHaveLength(2);
  });

  it("predicts termination from the full recorded context", () => {
    const trajectory = flow("t1", "mail", [
      ["tap", "compose"],
      ["tap", "send"],
    ]);
    const model = new MarkovMovementBackend().train([trajectory], { order: 3 });
    const prediction = model.predict({ app: "mail", history: trajectory.events });
    expect(prediction.done).toBe(true);
  });

  it("generalizes movement shape to novel targets via the gesture channel", () => {
    // Train on the same gesture flow with several different target sets.
    const training: MovementTrajectory[] = [
      flow("a", "mail", [["tap", "compose-1"], ["type", "subj-1"], ["tap", "send-1"]]),
      flow("b", "mail", [["tap", "compose-2"], ["type", "subj-2"], ["tap", "send-2"]]),
      flow("c", "mail", [["tap", "compose-3"], ["type", "subj-3"], ["tap", "send-3"]]),
    ];
    const model = new MarkovMovementBackend().train(training, { order: 3 });

    // A held-out flow with entirely new targets — tokens never seen.
    const novelHistory = flow("z", "mail", [["tap", "compose-99"]]).events;
    const prediction = model.predict({ app: "mail", history: novelHistory });
    // Exact token is novel, but the gesture shape (type after tap) generalizes.
    expect(prediction.gesture).toBe("type");
  });

  it("round-trips through serialize/load", () => {
    const trajectory = flow("t1", "mail", [["tap", "compose"], ["tap", "send"]]);
    const backend = new MarkovMovementBackend();
    const model = backend.train([trajectory], { order: 2 });
    const restored = backend.load(model.serialize());

    const original = rolloutMovements(model, { app: "mail" });
    const reloaded = rolloutMovements(restored, { app: "mail" });
    expect(reloaded).toEqual(original);
  });

  it("is deterministic across identical trainings", () => {
    const trajectory = flow("t1", "mail", [["tap", "a"], ["type", "b"], ["tap", "c"]]);
    const first = new MarkovMovementBackend().train([trajectory], { order: 3 }).serialize();
    const second = new MarkovMovementBackend().train([trajectory], { order: 3 }).serialize();
    expect(second).toEqual(first);
  });
});

describe("MovementBackendRegistry", () => {
  it("provides the markov backend by default and is pluggable", () => {
    const registry = new MovementBackendRegistry();
    expect(registry.list()).toContain("markov");
    expect(registry.get("markov").id).toBe("markov");
    expect(() => registry.get("nope")).toThrow(/unknown movement backend/);
  });

  it("accepts a custom registered backend", () => {
    const registry = new MovementBackendRegistry([]);
    const backend = new MarkovMovementBackend();
    registry.register(backend);
    expect(registry.get("markov")).toBe(backend);
  });
});

describe("extractMovementTrajectory", () => {
  it("projects device-capture action spans into movement events", () => {
    const span = buildTrajectorySpan({
      id: "span-1",
      sessionId: "session-1",
      actions: [
        {
          kind: "action",
          tool: "device",
          summary: "tapped compose",
          ts: 1,
          metadata: { gesture: "tap", target: "compose", appId: "mail" },
        },
        {
          kind: "action",
          tool: "device",
          summary: "scrolled down",
          ts: 2,
          metadata: { gesture: "scroll", direction: "down", appId: "mail" },
        },
        {
          kind: "action",
          tool: "shell",
          summary: "ran a non-movement action",
          ts: 3,
          metadata: { command: "ls" },
        },
      ],
    });

    const trajectory = extractMovementTrajectory(span);
    expect(trajectory.app).toBe("mail");
    expect(trajectory.events).toHaveLength(2);
    expect(trajectory.events[0]).toMatchObject({ gesture: "tap", target: "compose" });
    expect(trajectory.events[1]).toMatchObject({ gesture: "scroll", direction: "down" });
  });
});
