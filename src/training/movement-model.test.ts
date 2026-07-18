import { describe, expect, it } from "vitest";
import type { ReplayManifest, ReplayTimelineEvent } from "../capture/replay.js";
import {
  DEFAULT_MOVEMENT_BACKENDS,
  MarkovMovementBackend,
  buildMovementDataset,
  evaluateMovementModel,
  movementTokenForEvent,
  resolveMovementBackend,
} from "./movement-model.js";

// --- synthetic event-stream generator (no real OS input needed) ------------

let ts = 0;

function action(trajectoryId: string, tool: string): ReplayTimelineEvent {
  return { kind: "action", ts: (ts += 1), trajectoryId, tool, summary: `${tool} move` };
}

function observation(trajectoryId: string, source: string): ReplayTimelineEvent {
  return { kind: "observation", ts: (ts += 1), trajectoryId, source, summary: `saw ${source}` };
}

function replay(sessionId: string, trajectoryId: string, events: ReplayTimelineEvent[]): ReplayManifest {
  return {
    version: 1,
    sessionId,
    trajectoryIds: [trajectoryId],
    eventCount: events.length,
    events,
  };
}

/**
 * A family of related "open app, focus field, type, submit" movement traces.
 * They share structure but differ in the concrete field observed, so a model
 * that learns the *shape* can generalize across them.
 */
function openAndSubmitTrace(sessionId: string, field: string): ReplayManifest {
  const id = `${sessionId}-traj`;
  return replay(sessionId, id, [
    action(id, "window.focus"),
    observation(id, `field.${field}`),
    action(id, "keyboard.type"),
    action(id, "mouse.click"),
    action(id, "form.submit"),
  ]);
}

describe("movement tokenization", () => {
  it("reduces each event kind to a stable shape token", () => {
    expect(movementTokenForEvent({ kind: "action", ts: 1, trajectoryId: "t", tool: "mouse.click", summary: "" })).toBe(
      "act:mouse.click",
    );
    expect(
      movementTokenForEvent({ kind: "observation", ts: 1, trajectoryId: "t", source: "field.email", summary: "" }),
    ).toBe("obs:field.email");
    expect(movementTokenForEvent({ kind: "transcript", ts: 1, messageId: "m", role: "user", content: "hi" })).toBe(
      "msg:user",
    );
  });
});

describe("buildMovementDataset", () => {
  it("builds one sequence per non-empty replay with a sorted vocabulary", () => {
    const dataset = buildMovementDataset([
      openAndSubmitTrace("s1", "email"),
      replay("s-empty", "empty", []),
    ]);
    expect(dataset.sequences).toHaveLength(1);
    expect(dataset.sequences[0]!.tokens).toEqual([
      "act:window.focus",
      "obs:field.email",
      "act:keyboard.type",
      "act:mouse.click",
      "act:form.submit",
    ]);
    expect(dataset.vocabulary).toEqual([...dataset.vocabulary].sort());
  });
});

describe("MarkovMovementBackend — repeat recorded movements (objective 2c)", () => {
  it("regenerates a recorded trajectory from its first move", () => {
    const dataset = buildMovementDataset([openAndSubmitTrace("s1", "email")]);
    const model = new MarkovMovementBackend().train(dataset);
    const generated = model.generate(["act:window.focus"], 10);
    expect(generated).toEqual([
      "obs:field.email",
      "act:keyboard.type",
      "act:mouse.click",
      "act:form.submit",
    ]);
  });

  it("predicts the single most-likely next move for a context", () => {
    const dataset = buildMovementDataset([openAndSubmitTrace("s1", "email")]);
    const model = new MarkovMovementBackend().train(dataset);
    expect(model.predictNext(["act:window.focus", "obs:field.email"])).toBe("act:keyboard.type");
  });
});

describe("MarkovMovementBackend — generalize to new-but-related moves (objective 2d)", () => {
  it("composes learned transitions into a novel trajectory never recorded verbatim", () => {
    // Train on two related traces; each ends the same way after typing.
    const dataset = buildMovementDataset([
      openAndSubmitTrace("s1", "email"),
      openAndSubmitTrace("s2", "password"),
    ]);
    const model = new MarkovMovementBackend(1).train(dataset);

    // Seed with a field the model saw, then let it complete the shared tail.
    const generated = model.generate(["obs:field.email"], 10);
    expect(generated).toEqual(["act:keyboard.type", "act:mouse.click", "act:form.submit"]);

    // Order-1 model generalizes: "act:keyboard.type" always leads to the shared
    // click->submit tail regardless of which field preceded it.
    expect(model.generate(["act:keyboard.type"], 10)).toEqual(["act:mouse.click", "act:form.submit"]);
  });

  it("transfers the shared action tail to a held-out trajectory with a novel field", () => {
    const train = buildMovementDataset([
      openAndSubmitTrace("s1", "email"),
      openAndSubmitTrace("s2", "password"),
      openAndSubmitTrace("s3", "username"),
    ]);
    const model = new MarkovMovementBackend(1).train(train);

    // Held-out trace uses a field the model has NEVER seen ("search"). The
    // shared type->click->submit action tail is learned from every trained
    // trace, so it transfers; the field-specific observation (and the move that
    // immediately follows a token the model has never seen) are inherently not
    // predictable from prior recordings.
    const heldOut = buildMovementDataset([openAndSubmitTrace("s4", "search")]);
    const evaluation = evaluateMovementModel(model, heldOut.sequences);

    expect(evaluation.sequenceCount).toBe(1);
    expect(evaluation.predictions).toBe(4);
    // 2 of 4 next-token predictions transfer (the click and submit moves); far
    // above the ~1/vocab chance rate for a multi-token vocabulary.
    expect(evaluation.correct).toBe(2);
    expect(evaluation.accuracy).toBeCloseTo(0.5);
    expect(evaluation.accuracy).toBeGreaterThan(1 / heldOut.vocabulary.length);
  });
});

describe("model serialization", () => {
  it("round-trips through a JSON artifact with identical behavior", () => {
    const dataset = buildMovementDataset([
      openAndSubmitTrace("s1", "email"),
      openAndSubmitTrace("s2", "password"),
    ]);
    const model = new MarkovMovementBackend(2).train(dataset);
    const artifact = model.serialize();
    const json = JSON.parse(JSON.stringify(artifact));

    const reloaded = MarkovMovementBackend.deserialize(json);
    expect(reloaded.backendId).toBe(model.backendId);
    expect(reloaded.generate(["act:window.focus"], 10)).toEqual(model.generate(["act:window.focus"], 10));
    expect(artifact.trainedSequences).toBe(2);
    expect(artifact.trainedTokens).toBeGreaterThan(0);
  });

  it("serialize returns a defensive copy of transitions", () => {
    const dataset = buildMovementDataset([openAndSubmitTrace("s1", "email")]);
    const model = new MarkovMovementBackend().train(dataset);
    const first = model.serialize();
    const firstKey = Object.keys(first.transitions)[0]!;
    first.transitions[firstKey]!["injected"] = 999;
    const second = model.serialize();
    expect(second.transitions[firstKey]!["injected"]).toBeUndefined();
  });
});

describe("backend registry", () => {
  it("resolves the default markov backend by id and by default", () => {
    expect(resolveMovementBackend().id).toBe("markov-movement@1");
    expect(resolveMovementBackend("markov-movement@1").id).toBe("markov-movement@1");
    expect(Object.keys(DEFAULT_MOVEMENT_BACKENDS)).toContain("markov-movement@1");
  });

  it("throws for an unknown backend id", () => {
    expect(() => resolveMovementBackend("neural-on-device@99")).toThrow(/Unknown movement model backend/);
  });
});
