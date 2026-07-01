import { describe, expect, it } from "vitest";
import {
  MarkovMovementBackend,
  evaluateMovementModel,
  movementSequenceFromReplay,
  movementSequenceFromTrajectory,
  movementToken,
  type MovementSequence,
} from "./movement-model.js";
import type { ReplayTimelineEvent } from "../capture/replay.js";
import type { TrajectorySpan } from "../capture/trajectory.js";

/** Deterministic synthetic "login" flow used as recorded ground truth. */
function loginSequence(id: string, context = "app:login"): MovementSequence {
  return {
    id,
    context,
    steps: [
      { gesture: "tap", target: "username" },
      { gesture: "type", target: "username", value: "text" },
      { gesture: "tap", target: "password" },
      { gesture: "type", target: "password", value: "secret" },
      { gesture: "tap", target: "submit" },
    ],
  };
}

describe("MarkovMovementBackend", () => {
  it("repeats a recorded movement sequence exactly after training (objective 2c)", async () => {
    const backend = new MarkovMovementBackend();
    const recorded = loginSequence("s1");
    const model = await backend.train({ sequences: [recorded] });

    const result = await backend.infer(model, { context: recorded.context, maxSteps: 16 });

    expect(result.steps.map(movementToken)).toEqual(recorded.steps.map(movementToken));
    expect(result.steps.map((step) => step.target)).toEqual(recorded.steps.map((step) => step.target));
    expect(result.backedOff).toBe(false);
    expect(result.confidence).toBe(1);
  });

  it("terminates on its own via the learned end token", async () => {
    const backend = new MarkovMovementBackend();
    const model = await backend.train({ sequences: [loginSequence("s1")] });
    // maxSteps far larger than the sequence: it must stop at the recorded length.
    const result = await backend.infer(model, { context: "app:login", maxSteps: 100 });
    expect(result.steps).toHaveLength(5);
  });

  it("generalizes to a related target via overrides while preserving structure (objective 2d)", async () => {
    const backend = new MarkovMovementBackend();
    const model = await backend.train({ sequences: [loginSequence("s1")] });

    const result = await backend.infer(model, {
      context: "app:login",
      targetOverrides: { submit: "continue" },
      maxSteps: 16,
    });

    // Same gesture structure, but the final target is the substituted element.
    expect(result.steps.map(movementToken)).toEqual(loginSequence("s1").steps.map(movementToken));
    expect(result.steps.at(-1)?.target).toBe("continue");
  });

  it("continues from a seed of steps", async () => {
    const backend = new MarkovMovementBackend();
    const model = await backend.train({ sequences: [loginSequence("s1")] });

    const result = await backend.infer(model, {
      context: "app:login",
      seed: [{ gesture: "tap", target: "username" }],
      maxSteps: 16,
    });

    expect(result.steps).toHaveLength(5);
    expect(result.steps[1]?.gesture).toBe("type");
  });

  it("backs off to lower-order/global transitions on an unseen context (objective 2d)", async () => {
    const backend = new MarkovMovementBackend();
    // Train on the labelled context, then infer under a *novel* context that
    // shares the same tokens: the model must still produce a plausible flow.
    const model = await backend.train({ sequences: [loginSequence("s1", "app:login")] });

    const result = await backend.infer(model, {
      context: "app:signup-modal",
      seed: [{ gesture: "tap", target: "username" }],
      maxSteps: 16,
    });

    expect(result.steps.length).toBeGreaterThan(1);
    expect(result.backedOff).toBe(true);
    expect(result.confidence).toBeLessThan(1);
  });

  it("learns direction-carrying gestures without token collision", async () => {
    const backend = new MarkovMovementBackend();
    const swipe: MovementSequence = {
      id: "s-swipe",
      context: "app:gallery",
      steps: [
        { gesture: "swipe", direction: "left" },
        { gesture: "swipe", direction: "up" },
        { gesture: "tap", target: "photo" },
      ],
    };
    const model = await backend.train({ sequences: [swipe] });
    const result = await backend.infer(model, { context: "app:gallery", maxSteps: 8 });

    expect(result.steps.map(movementToken)).toEqual(swipe.steps.map(movementToken));
    expect(result.steps[0]).toMatchObject({ gesture: "swipe", direction: "left" });
    expect(result.steps[1]).toMatchObject({ gesture: "swipe", direction: "up" });
  });

  it("is deterministic across repeated inference runs", async () => {
    const backend = new MarkovMovementBackend();
    const model = await backend.train({
      sequences: [loginSequence("a"), loginSequence("b"), loginSequence("c")],
    });
    const first = await backend.infer(model, { context: "app:login", maxSteps: 16 });
    const second = await backend.infer(model, { context: "app:login", maxSteps: 16 });
    expect(second.steps).toEqual(first.steps);
  });
});

describe("evaluateMovementModel", () => {
  it("scores perfect fidelity on held-out copies of trained flows", async () => {
    const backend = new MarkovMovementBackend();
    const train = [loginSequence("t1"), loginSequence("t2")];
    const model = await backend.train({ sequences: train });

    const evalResult = await evaluateMovementModel(backend, model, [loginSequence("held")]);

    expect(evalResult.sequences).toBe(1);
    expect(evalResult.tokenFidelity).toBe(1);
    expect(evalResult.exactMatch).toBe(1);
  });

  it("reports partial fidelity when the held-out flow diverges", async () => {
    const backend = new MarkovMovementBackend();
    const model = await backend.train({ sequences: [loginSequence("t1")] });

    const divergent: MovementSequence = {
      id: "d1",
      context: "app:login",
      steps: [
        { gesture: "tap", target: "username" },
        { gesture: "type", target: "username" },
        { gesture: "scroll", direction: "down" }, // unseen continuation
      ],
    };

    const evalResult = await evaluateMovementModel(backend, model, [divergent]);
    expect(evalResult.tokenFidelity).toBeGreaterThan(0);
    expect(evalResult.tokenFidelity).toBeLessThan(1);
    expect(evalResult.exactMatch).toBe(0);
  });

  it("returns a neutral score for an empty held-out set", async () => {
    const backend = new MarkovMovementBackend();
    const model = await backend.train({ sequences: [loginSequence("t1")] });
    const evalResult = await evaluateMovementModel(backend, model, []);
    expect(evalResult).toEqual({ sequences: 0, tokenFidelity: 1, exactMatch: 1, meanConfidence: 1 });
  });
});

describe("dataset adapters", () => {
  it("extracts a movement sequence from a captured trajectory span", () => {
    const trajectory: TrajectorySpan = {
      id: "traj-1",
      sessionId: "sess-1",
      createdAt: "2026-07-01T00:00:00.000Z",
      captureTier: "app",
      observations: [
        {
          kind: "observation",
          source: "device",
          summary: "Photos active",
          ts: 1,
          metadata: { appName: "Photos" },
        },
      ],
      actions: [
        {
          kind: "action",
          tool: "device",
          summary: "swiped left",
          ts: 2,
          metadata: { gesture: "swipe", direction: "left" },
        },
        {
          kind: "action",
          tool: "device",
          summary: "tapped photo",
          ts: 3,
          metadata: { gesture: "tap", target: "photo" },
        },
      ],
    };

    const sequence = movementSequenceFromTrajectory(trajectory);
    expect(sequence.context).toBe("device:Photos");
    expect(sequence.steps).toEqual([
      { gesture: "swipe", direction: "left" },
      { gesture: "tap", target: "photo" },
    ]);
  });

  it("extracts a movement sequence from a replay timeline", () => {
    const events: ReplayTimelineEvent[] = [
      { kind: "observation", ts: 1, trajectoryId: "t", source: "device", summary: "app active" },
      { kind: "action", ts: 2, trajectoryId: "t", tool: "device", summary: "tapped submit" },
      { kind: "transcript", ts: 3, messageId: "m", role: "assistant", content: "done" },
    ];
    const sequence = movementSequenceFromReplay("seq-1", events, "app:form");
    expect(sequence.context).toBe("app:form");
    expect(sequence.steps).toEqual([{ gesture: "device", target: "submit" }]);
  });

  it("round-trips capture -> dataset -> train -> replay for a trajectory", async () => {
    const backend = new MarkovMovementBackend();
    const trajectory: TrajectorySpan = {
      id: "traj-rt",
      sessionId: "sess",
      createdAt: "2026-07-01T00:00:00.000Z",
      captureTier: "app",
      observations: [
        { kind: "observation", source: "device", summary: "Editor", ts: 1, metadata: { appName: "Editor" } },
      ],
      actions: [
        { kind: "action", tool: "device", summary: "tapped file", ts: 2, metadata: { gesture: "tap", target: "file" } },
        { kind: "action", tool: "device", summary: "typed", ts: 3, metadata: { gesture: "type", target: "file", valueSummary: "code" } },
        { kind: "action", tool: "device", summary: "tapped save", ts: 4, metadata: { gesture: "tap", target: "save" } },
      ],
    };
    const sequence = movementSequenceFromTrajectory(trajectory);
    const model = await backend.train({ sequences: [sequence] });
    const result = await backend.infer(model, { context: sequence.context, maxSteps: 16 });

    expect(result.steps).toEqual(sequence.steps);
  });
});
