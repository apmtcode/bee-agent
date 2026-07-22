import { describe, expect, it } from "vitest";
import {
  MarkovMovementBackend,
  MarkovMovementModel,
  actionToMovementToken,
  buildMovementDataset,
  encodeMovementToken,
  trainMovementModel,
  type MovementToken,
} from "./movement-model.js";
import type { TrajectoryAction, TrajectorySpan } from "../capture/trajectory.js";

function action(tool: string, summary: string, ts: number, metadata?: Record<string, unknown>): TrajectoryAction {
  return { kind: "action", tool, summary, ts, ...(metadata ? { metadata } : {}) };
}

function trajectory(id: string, actions: TrajectoryAction[]): TrajectorySpan {
  return {
    id,
    sessionId: `session-${id}`,
    createdAt: "2026-01-01T00:00:00.000Z",
    captureTier: "full",
    observations: [],
    actions,
  };
}

/** A "open editor -> type -> save" gesture flow, repeated so it dominates counts. */
function openTypeSaveTrajectory(id: string): TrajectorySpan {
  return trajectory(id, [
    action("device", "tapped editor", 1, { gesture: "tap", target: "editor" }),
    action("device", "typed into body", 2, { gesture: "type", target: "body" }),
    action("device", "triggered save", 3, { gesture: "shortcut", target: "save" }),
  ]);
}

describe("movement token derivation", () => {
  it("prefers structured metadata over summary parsing", () => {
    const token = actionToMovementToken(
      action("device", "swiped left", 1, { gesture: "swipe", direction: "left", target: "carousel" }),
    );
    expect(token).toEqual({ tool: "device", gesture: "swipe", direction: "left", target: "carousel" });
    expect(encodeMovementToken(token)).toBe("device:swipe:>left:@carousel");
  });

  it("falls back to parsing the human summary when metadata is absent", () => {
    const token = actionToMovementToken(action("device", "typed into body", 1));
    expect(token.gesture).toBe("type");
    expect(token.target).toBe("body");
  });

  it("normalizes past-tense verbs to canonical gestures", () => {
    expect(actionToMovementToken(action("mouse", "clicked submit", 1)).gesture).toBe("tap");
    expect(actionToMovementToken(action("device", "scrolled down", 1)).gesture).toBe("scroll");
  });
});

describe("buildMovementDataset", () => {
  it("orders actions by timestamp and drops empty trajectories", () => {
    const dataset = buildMovementDataset([
      trajectory("out-of-order", [
        action("device", "typed into body", 5, { gesture: "type", target: "body" }),
        action("device", "tapped editor", 1, { gesture: "tap", target: "editor" }),
      ]),
      trajectory("empty", []),
    ]);
    expect(dataset.sequences).toHaveLength(1);
    expect(dataset.sequences[0].tokens.map((t) => t.gesture)).toEqual(["tap", "type"]);
  });

  it("uses redacted actions when a review redacted the trajectory", () => {
    const base = openTypeSaveTrajectory("redacted");
    base.review = {
      status: "approved",
      reviewedAt: "2026-01-01T00:00:00.000Z",
      reviewedBy: "operator",
      redactedActions: [
        { ts: 1, tool: "device", summary: "tapped editor" },
        { ts: 2, tool: "device", summary: "triggered save" },
      ],
    };
    const dataset = buildMovementDataset([base]);
    expect(dataset.sequences[0].tokens.map((t) => t.gesture)).toEqual(["tap", "shortcut"]);
  });
});

describe("MarkovMovementBackend — repeat recorded movements (objective 2c)", () => {
  it("reproduces the recorded continuation from a seen prefix", () => {
    const model = trainMovementModel([
      openTypeSaveTrajectory("a"),
      openTypeSaveTrajectory("b"),
    ]);
    const seed: MovementToken[] = [{ tool: "device", gesture: "tap", target: "editor" }];
    const rollout = model.generate(seed, 2);
    expect(rollout.map((t) => t.gesture)).toEqual(["type", "shortcut"]);
    expect(rollout[1].target).toBe("save");
  });

  it("is deterministic — identical training yields identical predictions", () => {
    const a = trainMovementModel([openTypeSaveTrajectory("a")]);
    const b = trainMovementModel([openTypeSaveTrajectory("a")]);
    const seed: MovementToken[] = [{ tool: "device", gesture: "tap", target: "editor" }];
    expect(a.generate(seed, 3)).toEqual(b.generate(seed, 3));
  });

  it("prefers the most frequent successor when contexts compete", () => {
    // "tap editor" is followed by "type" 3x and by "scroll" 1x.
    const model = trainMovementModel([
      openTypeSaveTrajectory("a"),
      openTypeSaveTrajectory("b"),
      openTypeSaveTrajectory("c"),
      trajectory("noise", [
        action("device", "tapped editor", 1, { gesture: "tap", target: "editor" }),
        action("device", "scrolled down", 2, { gesture: "scroll", direction: "down" }),
      ]),
    ]);
    const prediction = model.predictNext([{ tool: "device", gesture: "tap", target: "editor" }]);
    expect(prediction?.token.gesture).toBe("type");
    expect(prediction?.confidence).toBeCloseTo(3 / 4);
  });
});

describe("MarkovMovementBackend — generalize to related movements (objective 2d)", () => {
  it("predicts via back-off for an unseen prefix that shares a suffix", () => {
    // Train: [openApp, tap editor, type]. Query with a *different* preamble
    // (openSettings) but the same recent suffix (tap editor) — a higher-order
    // context [openSettings, tap editor] was never seen, so the model backs off
    // to [tap editor] and still predicts "type".
    const model = trainMovementModel(
      [
        trajectory("train", [
          action("device", "tapped launcher", 1, { gesture: "tap", target: "launcher" }),
          action("device", "tapped editor", 2, { gesture: "tap", target: "editor" }),
          action("device", "typed into body", 3, { gesture: "type", target: "body" }),
        ]),
      ],
      { order: 3 },
    );
    const novelContext: MovementToken[] = [
      { tool: "device", gesture: "tap", target: "settings" },
      { tool: "device", gesture: "tap", target: "editor" },
    ];
    const prediction = model.predictNext(novelContext);
    expect(prediction?.token.gesture).toBe("type");
    // It matched a *shorter* context than the full query — that is generalization.
    expect(prediction?.matchedOrder).toBeLessThan(novelContext.length);
  });

  it("falls back to the unigram distribution for a fully novel context", () => {
    const model = trainMovementModel([openTypeSaveTrajectory("a"), openTypeSaveTrajectory("b")]);
    const prediction = model.predictNext([{ tool: "mouse", gesture: "wiggle", target: "nowhere" }]);
    expect(prediction).toBeDefined();
    expect(prediction?.matchedOrder).toBe(0);
  });

  it("returns undefined from an empty model", () => {
    const model = new MarkovMovementBackend().train({ sequences: [] });
    expect(model.predictNext([])).toBeUndefined();
    expect(model.generate([], 5)).toEqual([]);
  });
});

describe("model artifact serialization", () => {
  it("round-trips through JSON and preserves predictions", () => {
    const model = trainMovementModel([openTypeSaveTrajectory("a"), openTypeSaveTrajectory("b")]);
    const json = JSON.parse(JSON.stringify(model.toJSON()));
    const restored = MarkovMovementModel.fromJSON(json);
    const seed: MovementToken[] = [{ tool: "device", gesture: "tap", target: "editor" }];
    expect(restored.generate(seed, 2)).toEqual(model.generate(seed, 2));
    expect(json.backend).toBe("markov-backoff");
    expect(json.version).toBe(1);
  });
});
