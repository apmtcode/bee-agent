import { describe, expect, it } from "vitest";
import type { ReplayTimelineEvent } from "../capture/replay.js";
import type { TrajectorySpan } from "../capture/trajectory.js";
import {
  MOVEMENT_END_TOKEN,
  NgramMovementBackend,
  NgramMovementModel,
  buildMovementDataset,
  buildMovementSequence,
  buildTrajectoryMovementSequence,
  evaluateNextTokenAccuracy,
  tokenizeReplayEvent,
  type MovementSequence,
} from "./movement-model.js";

function actionEvent(tool: string, ts: number): ReplayTimelineEvent {
  return { kind: "action", ts, trajectoryId: "t", tool, summary: `${tool}@${ts}` };
}

function seq(id: string, tokens: string[]): MovementSequence {
  return { id, tokens: [...tokens, MOVEMENT_END_TOKEN] };
}

describe("tokenization + dataset building", () => {
  it("derives stable tokens per event kind", () => {
    expect(tokenizeReplayEvent(actionEvent("click", 1))).toBe("action:click");
    expect(
      tokenizeReplayEvent({ kind: "observation", ts: 1, trajectoryId: "t", source: "screen", summary: "s" }),
    ).toBe("observation:screen");
    expect(
      tokenizeReplayEvent({ kind: "transcript", ts: 1, messageId: "m", role: "user", content: "c" }),
    ).toBe("transcript:user");
  });

  it("appends a terminal token to built sequences", () => {
    const sequence = buildMovementSequence("s", [actionEvent("open", 1), actionEvent("type", 2)]);
    expect(sequence.tokens).toEqual(["action:open", "action:type", MOVEMENT_END_TOKEN]);
  });

  it("builds a dataset from replay manifests and drops empty ones", () => {
    const dataset = buildMovementDataset([
      { sessionId: "a", events: [actionEvent("open", 1), actionEvent("save", 2)] },
      { sessionId: "b", events: [] },
    ]);
    expect(dataset).toHaveLength(1);
    expect(dataset[0]!.id).toBe("a#0");
    expect(dataset[0]!.tokens).toEqual(["action:open", "action:save", MOVEMENT_END_TOKEN]);
  });

  it("orders a trajectory movement sequence by timestamp", () => {
    const trajectory = {
      id: "traj",
      actions: [
        { kind: "action", tool: "save", summary: "", ts: 30 },
        { kind: "action", tool: "open", summary: "", ts: 10 },
        { kind: "action", tool: "type", summary: "", ts: 20 },
      ],
    } as unknown as TrajectorySpan;
    const sequence = buildTrajectoryMovementSequence(trajectory);
    expect(sequence.tokens).toEqual(["action:open", "action:type", "action:save", MOVEMENT_END_TOKEN]);
  });
});

describe("reproduction (objective 2c)", () => {
  it("reproduces a recorded movement sequence exactly from a seed", () => {
    const training = [seq("recorded", ["action:open", "action:type", "action:save"])];
    const model = new NgramMovementBackend().train(training, { order: 2 });
    const generated = model.generate(["action:open"]);
    expect(generated).toEqual(["action:type", "action:save"]);
  });

  it("scores 100% next-token accuracy on its own training data", () => {
    const training = [
      seq("a", ["action:open", "action:type", "action:save"]),
      seq("b", ["action:focus", "action:type", "action:close"]),
    ];
    const model = NgramMovementModel.train(training, { order: 2 });
    const evaluation = evaluateNextTokenAccuracy(model, training);
    expect(evaluation.accuracy).toBe(1);
    expect(evaluation.correct).toBe(evaluation.predictions);
  });
});

describe("generalization (objective 2d)", () => {
  it("predicts a learned continuation for an unseen but related prefix via backoff", () => {
    // Both training sequences share the bigram (click -> submit).
    const training = [
      seq("a", ["action:scroll", "action:click", "action:submit"]),
      seq("b", ["action:focus", "action:click", "action:submit"]),
    ];
    const model = NgramMovementModel.train(training, { order: 2 });

    // Novel prefix: "hover" was never seen, so the order-2 context "hover click"
    // has no counts and the model must back off to the order-1 "click" context.
    const predictions = model.predict(["action:hover", "action:click"]);
    expect(predictions[0]!.token).toBe("action:submit");
    expect(predictions[0]!.order).toBe(1);
  });

  it("falls back to the unigram distribution for a fully unknown context", () => {
    const training = [seq("a", ["action:open", "action:open", "action:save"])];
    const model = NgramMovementModel.train(training, { order: 2 });
    const predictions = model.predict(["action:totally-unseen"]);
    // "action:open" is the most frequent token overall -> unigram argmax.
    expect(predictions[0]!.token).toBe("action:open");
    expect(predictions[0]!.order).toBe(0);
  });
});

describe("determinism + persistence", () => {
  it("is deterministic across repeated training and generation", () => {
    const training = [
      seq("a", ["action:open", "action:type", "action:save"]),
      seq("b", ["action:open", "action:type", "action:close"]),
    ];
    const first = NgramMovementModel.train(training, { order: 2 }).generate(["action:open"]);
    const second = NgramMovementModel.train(training, { order: 2 }).generate(["action:open"]);
    expect(first).toEqual(second);
  });

  it("breaks probability ties by lexical token order for stable output", () => {
    // After "open", "close" and "zoom" are equally likely; "close" wins lexically.
    const training = [
      seq("a", ["action:open", "action:close"]),
      seq("b", ["action:open", "action:zoom"]),
    ];
    const model = NgramMovementModel.train(training, { order: 2 });
    const predictions = model.predict(["action:open"]);
    expect(predictions[0]!.token).toBe("action:close");
    expect(predictions[0]!.probability).toBeCloseTo(0.5);
  });

  it("round-trips through serialization with identical predictions", () => {
    const training = [
      seq("a", ["action:open", "action:type", "action:save"]),
      seq("b", ["action:focus", "action:type", "action:close"]),
    ];
    const model = NgramMovementModel.train(training, { order: 2 });
    const restored = NgramMovementModel.fromSerialized(JSON.parse(JSON.stringify(model.serialize())));

    expect(restored.order).toBe(model.order);
    expect(restored.generate(["action:open"])).toEqual(model.generate(["action:open"]));
    expect(restored.predict(["action:type"])).toEqual(model.predict(["action:type"]));
  });

  it("stops generation at maxSteps even without a terminal token", () => {
    // A self-loop sequence: after "spin" the only continuation is "spin".
    const model = NgramMovementModel.train([{ id: "loop", tokens: ["action:spin", "action:spin"] }], {
      order: 1,
    });
    const generated = model.generate(["action:spin"], { maxSteps: 3 });
    expect(generated).toEqual(["action:spin", "action:spin", "action:spin"]);
  });
});
