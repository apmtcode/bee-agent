import { describe, expect, it } from "vitest";
import type { TrajectorySpan } from "../capture/trajectory.js";
import {
  MOVEMENT_START_TOKEN,
  MovementPolicyModel,
  NgramMovementBackend,
  createDefaultMovementModel,
  decodeMovementToken,
  encodeMovementStep,
  tokenizeSequence,
  trajectoryToMovementSequence,
  type MovementSequence,
} from "./movement-model.js";

function seq(id: string, steps: MovementSequence["steps"]): MovementSequence {
  return { id, steps };
}

const OPEN_MAIL: MovementSequence = seq("open-mail", [
  { gesture: "tap", target: "mail-app" },
  { gesture: "tap", target: "search-field" },
  { gesture: "type", target: "search-field", valueSummary: "invoice" },
  { gesture: "tap", target: "row-1" },
]);

describe("movement token codec", () => {
  it("round-trips a step through encode/decode", () => {
    const step = { gesture: "swipe" as const, direction: "left" as const, target: "card" };
    expect(decodeMovementToken(encodeMovementStep(step))).toEqual({
      gesture: "swipe",
      direction: "left",
      target: "card",
    });
  });

  it("normalizes casing and whitespace for stable tokens", () => {
    expect(encodeMovementStep({ gesture: "tap", target: "  Mail-App " })).toBe(
      encodeMovementStep({ gesture: "tap", target: "mail-app" }),
    );
  });

  it("wraps sequences with start/end sentinels", () => {
    const tokens = tokenizeSequence(OPEN_MAIL);
    expect(tokens[0]).toBe(MOVEMENT_START_TOKEN);
    expect(tokens.at(-1)).toBe("end");
    expect(tokens).toHaveLength(OPEN_MAIL.steps.length + 2);
  });
});

describe("NgramMovementBackend training + prediction", () => {
  it("produces JSON-serializable weights tagged by backend", () => {
    const weights = new NgramMovementBackend().train([OPEN_MAIL]);
    expect(weights.backend).toBe("ngram");
    expect(weights.sequenceCount).toBe(1);
    expect(() => JSON.parse(JSON.stringify(weights))).not.toThrow();
    expect(weights.vocabulary).toContain(MOVEMENT_START_TOKEN);
  });

  it("predicts the recorded next step with probability 1 for a unique context", () => {
    const model = createDefaultMovementModel([OPEN_MAIL]);
    const first = model.predict([MOVEMENT_START_TOKEN])[0]!;
    expect(decodeMovementToken(first.token)).toEqual({ gesture: "tap", target: "mail-app" });
    expect(first.probability).toBe(1);
  });
});

describe("replay fidelity", () => {
  it("greedy rollout reproduces a memorized sequence exactly", () => {
    const model = createDefaultMovementModel([OPEN_MAIL]);
    expect(model.generate()).toEqual(OPEN_MAIL.steps);
  });

  it("is deterministic across repeated rollouts", () => {
    const model = createDefaultMovementModel([OPEN_MAIL, OPEN_MAIL]);
    expect(model.generate()).toEqual(model.generate());
  });

  it("terminates at the end sentinel rather than running to maxSteps", () => {
    const model = createDefaultMovementModel([OPEN_MAIL]);
    expect(model.generate({ maxSteps: 100 })).toHaveLength(OPEN_MAIL.steps.length);
  });
});

describe("generalization via backoff", () => {
  it("predicts a plausible continuation for an unseen high-order context", () => {
    // Two tasks share the "tap search-field -> type" shape but differ in app.
    const a = seq("a", [
      { gesture: "tap", target: "app-a" },
      { gesture: "tap", target: "search-field" },
      { gesture: "type", target: "search-field", valueSummary: "x" },
    ]);
    const b = seq("b", [
      { gesture: "tap", target: "app-b" },
      { gesture: "tap", target: "search-field" },
      { gesture: "type", target: "search-field", valueSummary: "y" },
    ]);
    const model = MovementPolicyModel.train(new NgramMovementBackend(), [a, b], { order: 2 });

    // Novel prefix (app-c) never seen, but the "search-field" suffix has been.
    const context = tokenizeSequence(seq("probe", [
      { gesture: "tap", target: "app-c" },
      { gesture: "tap", target: "search-field" },
    ])).slice(0, -1); // drop trailing end sentinel
    const prediction = model.predict(context)[0];
    expect(prediction).toBeDefined();
    expect(decodeMovementToken(prediction!.token).gesture).toBe("type");
  });
});

describe("trajectoryToMovementSequence", () => {
  it("reconstructs the movement stream from recorded action metadata", () => {
    const span: TrajectorySpan = {
      id: "span-1",
      sessionId: "s",
      createdAt: "2026-07-27T00:00:00.000Z",
      captureTier: "full",
      observations: [],
      actions: [
        {
          kind: "action",
          tool: "device",
          summary: "tapped mail-app",
          ts: 20,
          metadata: { gesture: "tap", target: "mail-app" },
        },
        {
          kind: "action",
          tool: "device",
          summary: "scrolled down",
          ts: 10,
          metadata: { gesture: "scroll", direction: "down" },
        },
      ],
      outcome: { status: "success", summary: "opened mail" },
    };
    const sequence = trajectoryToMovementSequence(span);
    // Sorted by ts, so scroll (ts 10) precedes tap (ts 20).
    expect(sequence.steps).toEqual([
      { gesture: "scroll", direction: "down" },
      { gesture: "tap", target: "mail-app" },
    ]);
    expect(sequence.goal).toBe("opened mail");
  });
});
