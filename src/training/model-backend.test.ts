import { describe, expect, it } from "vitest";
import type { ReplayTimelineEvent } from "../capture/replay.js";
import {
  MarkovMovementBackend,
  buildMovementDataset,
  evaluateReplayFidelity,
  tokenizeReplayEvent,
  type MovementDataset,
  type MovementSequence,
  type ReplayLike,
} from "./model-backend.js";

function actionEvent(tool: string, ts: number, trajectoryId = "traj-1"): ReplayTimelineEvent {
  return { kind: "action", ts, trajectoryId, tool, summary: `${tool} @ ${ts}` };
}

function replay(trajectoryId: string, tools: string[]): ReplayLike {
  return {
    trajectoryIds: [trajectoryId],
    events: tools.map((tool, index) => actionEvent(tool, index, trajectoryId)),
  };
}

function seq(sourceId: string, tokens: string[]): MovementSequence {
  return { sourceId, tokens };
}

describe("tokenizeReplayEvent", () => {
  it("canonicalises each event kind to a stable token", () => {
    expect(tokenizeReplayEvent(actionEvent("click", 0))).toBe("action:click");
    expect(
      tokenizeReplayEvent({ kind: "observation", ts: 1, trajectoryId: "t", source: "screen", summary: "s" }),
    ).toBe("obs:screen");
    expect(
      tokenizeReplayEvent({ kind: "transcript", ts: 2, messageId: "m", role: "assistant", content: "c" }),
    ).toBe("msg:assistant");
  });
});

describe("buildMovementDataset", () => {
  it("derives one action-only sequence per replay", () => {
    const dataset = buildMovementDataset([
      replay("a", ["click", "type", "click"]),
      replay("b", ["scroll"]),
    ]);
    expect(dataset.sequences).toEqual([
      seq("a", ["action:click", "action:type", "action:click"]),
      seq("b", ["action:scroll"]),
    ]);
  });

  it("skips replays with no matching events and honours the kinds filter", () => {
    const mixed: ReplayLike = {
      trajectoryIds: ["m"],
      events: [
        { kind: "transcript", ts: 0, messageId: "x", role: "user", content: "hi" },
        actionEvent("click", 1, "m"),
      ],
    };
    expect(buildMovementDataset([mixed]).sequences).toEqual([seq("m", ["action:click"])]);
    expect(buildMovementDataset([mixed], { kinds: ["transcript"] }).sequences).toEqual([
      seq("m", ["msg:user"]),
    ]);
    // A replay that contributes zero tokens under the filter is dropped.
    expect(buildMovementDataset([replay("z", ["click"])], { kinds: ["observation"] }).sequences).toEqual(
      [],
    );
  });
});

describe("MarkovMovementBackend train + repeat (objective 2c)", () => {
  const backend = new MarkovMovementBackend();

  it("records vocabulary, counts, and provenance", () => {
    const dataset: MovementDataset = {
      version: 1,
      sequences: [seq("a", ["open", "focus", "type", "submit"])],
    };
    const model = backend.train(dataset, { trainedAt: "2026-07-06T00:00:00.000Z", order: 2 });
    expect(model.backend).toBe("markov-backoff");
    expect(model.order).toBe(2);
    expect(model.sequenceCount).toBe(1);
    expect(model.tokenCount).toBe(4);
    expect(model.vocabulary).toEqual(["focus", "open", "submit", "type"]);
    expect(model.trainedAt).toBe("2026-07-06T00:00:00.000Z");
  });

  it("replays a memorised movement sequence exactly (deterministic argmax)", () => {
    const dataset: MovementDataset = {
      version: 1,
      sequences: [seq("a", ["open", "focus", "type", "submit", "close"])],
    };
    const model = backend.train(dataset);
    // From the first movement it should reproduce the rest of the recording.
    expect(backend.generate(model, ["open"], 4)).toEqual(["focus", "type", "submit", "close"]);
  });

  it("predicts the most frequent continuation first", () => {
    const dataset: MovementDataset = {
      version: 1,
      sequences: [
        seq("a", ["menu", "save"]),
        seq("b", ["menu", "save"]),
        seq("c", ["menu", "quit"]),
      ],
    };
    const model = backend.train(dataset);
    const ranked = backend.predictNext(model, ["menu"]);
    expect(ranked[0]).toMatchObject({ token: "save" });
    expect(ranked[0]?.probability).toBeCloseTo(2 / 3, 6);
    expect(ranked.map((p) => p.token)).toEqual(["save", "quit"]);
  });
});

describe("MarkovMovementBackend generalisation via backoff (objective 2d)", () => {
  const backend = new MarkovMovementBackend(3);

  it("predicts for an unseen context by backing off to a shorter one", () => {
    const dataset: MovementDataset = {
      version: 1,
      sequences: [
        seq("a", ["open", "focus", "type", "submit"]),
        seq("b", ["copy", "focus", "type", "submit"]),
      ],
    };
    const model = backend.train(dataset);
    // The exact 3-gram ["scroll","drag","focus"] was never seen, but the
    // suffix ["focus"] was — backoff should still predict "type".
    const ranked = backend.predictNext(model, ["scroll", "drag", "focus"]);
    expect(ranked[0]?.token).toBe("type");
    // The prediction was made at a backed-off (order < 3) context.
    expect(ranked[0]?.order).toBeLessThan(3);
  });

  it("falls back to the unconditional distribution for a fully unseen context", () => {
    const dataset: MovementDataset = {
      version: 1,
      sequences: [seq("a", ["click", "click", "click", "type"])],
    };
    const model = backend.train(dataset);
    const ranked = backend.predictNext(model, ["totally-unseen-token"]);
    // "click" dominates the unconditional counts, so it leads even with no
    // matching context at all.
    expect(ranked[0]?.token).toBe("click");
    expect(ranked[0]?.order).toBe(0);
  });

  it("returns no prediction when the model has never seen a real movement", () => {
    const model = backend.train({ version: 1, sequences: [] });
    expect(backend.predictNext(model, ["anything"])).toEqual([]);
    expect(backend.generate(model, ["anything"], 5)).toEqual([]);
  });
});

describe("evaluateReplayFidelity", () => {
  const backend = new MarkovMovementBackend(3);

  it("scores perfect fidelity on the training sequence", () => {
    const train: MovementDataset = {
      version: 1,
      sequences: [seq("a", ["a", "b", "c", "d", "e"])],
    };
    const model = backend.train(train);
    const report = evaluateReplayFidelity(backend, model, train.sequences);
    expect(report.predictions).toBe(4);
    expect(report.correct).toBe(4);
    expect(report.accuracy).toBe(1);
  });

  it("generalises above chance to a held-out related sequence", () => {
    // Train on a repeated workflow, hold out a variant that shares transitions.
    const train: MovementDataset = {
      version: 1,
      sequences: [
        seq("a", ["login", "dashboard", "report", "export", "logout"]),
        seq("b", ["login", "dashboard", "report", "export", "logout"]),
      ],
    };
    const model = backend.train(train);
    const heldOut: MovementSequence[] = [seq("c", ["login", "dashboard", "report", "export"])];
    const report = evaluateReplayFidelity(backend, model, heldOut);
    expect(report.sequences).toBe(1);
    expect(report.predictions).toBe(3);
    // Every transition in the held-out variant was learned from training.
    expect(report.accuracy).toBe(1);
  });

  it("reports zero accuracy with no scorable positions", () => {
    const model = backend.train({ version: 1, sequences: [seq("a", ["x"])] });
    const report = evaluateReplayFidelity(backend, model, [seq("b", ["x"])]);
    expect(report.predictions).toBe(0);
    expect(report.accuracy).toBe(0);
  });
});
