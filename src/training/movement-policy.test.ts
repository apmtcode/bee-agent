import { describe, expect, it } from "vitest";
import { buildReplayManifest } from "../capture/replay.js";
import { buildTrajectorySpan } from "../capture/trajectory.js";
import {
  MOVEMENT_END_TOKEN,
  NgramMovementPolicyBackend,
  extractMovementExample,
  extractMovementExamplesFromReplay,
  tokenizeMovement,
  type MovementExample,
} from "./movement-policy.js";

function example(trajectoryId: string, tokens: string[]): MovementExample {
  return { trajectoryId, tokens };
}

describe("tokenizeMovement", () => {
  it("normalizes tool and summary into a stable token", () => {
    expect(tokenizeMovement({ tool: "  Click ", summary: "Save   Document " })).toBe("click::save document");
  });

  it("never collides with the end sentinel", () => {
    expect(tokenizeMovement({ tool: "x", summary: "y" })).not.toBe(MOVEMENT_END_TOKEN);
  });
});

describe("NgramMovementPolicyBackend", () => {
  const backend = new NgramMovementPolicyBackend();

  it("repeats a recorded movement sequence exactly via generate()", () => {
    const tokens = ["focus::a", "click::b", "type::c", "click::save"];
    const model = backend.train([example("t1", tokens)], { maxOrder: 3 });
    expect(model.generate(tokens.slice(0, 1))).toEqual(tokens);
  });

  it("predicts the next movement from the exact context", () => {
    const tokens = ["focus::a", "click::b", "type::c"];
    const model = backend.train([example("t1", tokens)], { maxOrder: 2 });
    const prediction = model.predict(["focus::a"]);
    expect(prediction.token).toBe("click::b");
    expect(prediction.source).toBe("exact");
    expect(prediction.probability).toBeGreaterThan(0);
  });

  it("generalizes to an unseen context by backing off to a shorter suffix", () => {
    // Two trajectories where "click::b" is always followed by "type::c",
    // regardless of what came before.
    const model = backend.train(
      [example("t1", ["focus::a", "click::b", "type::c"]), example("t2", ["press::x", "click::b", "type::c"])],
      { maxOrder: 3 },
    );
    // A novel prefix the model never saw, but ending in the familiar "click::b".
    const prediction = model.predict(["scroll::z", "click::b"]);
    expect(prediction.token).toBe("type::c");
    expect(prediction.source).toBe("backoff");
    expect(prediction.order).toBe(1);
  });

  it("ranks candidates by frequency in predictTopK", () => {
    const model = backend.train(
      [
        example("t1", ["click::b", "type::c"]),
        example("t2", ["click::b", "type::c"]),
        example("t3", ["click::b", "press::d"]),
      ],
      { maxOrder: 1 },
    );
    const ranked = model.predictTopK(["click::b"], 2);
    expect(ranked.map((entry) => entry.token)).toEqual(["type::c", "press::d"]);
    expect(ranked[0]!.probability).toBeCloseTo(2 / 3, 5);
  });

  it("falls back to a global unigram when no suffix matches", () => {
    const model = backend.train([example("t1", ["focus::a", "focus::a", "click::b"])], { maxOrder: 2 });
    const prediction = model.predict(["totally::unseen"]);
    expect(prediction.source).toBe("global");
    expect(prediction.token).toBe("focus::a"); // most frequent token overall
  });

  it("reports no prediction for an empty model", () => {
    const model = backend.train([], { maxOrder: 2 });
    expect(model.predict(["anything"]).source).toBe("none");
    expect(model.generate(["anything"])).toEqual(["anything"]);
  });

  it("serializes and reloads to an identical model", () => {
    const tokens = ["focus::a", "click::b", "type::c", "click::save"];
    const trained = backend.train([example("t1", tokens), example("t2", ["focus::a", "click::save"])], { maxOrder: 3 });
    const reloaded = backend.load(trained.serialize());

    expect(reloaded.maxOrder).toBe(trained.maxOrder);
    expect(reloaded.vocabulary()).toEqual(trained.vocabulary());
    expect(reloaded.generate(tokens.slice(0, 1))).toEqual(trained.generate(tokens.slice(0, 1)));
    expect(reloaded.predict(["focus::a"])).toEqual(trained.predict(["focus::a"]));
    // Round-trips again to the same bytes.
    expect(reloaded.serialize()).toEqual(trained.serialize());
  });

  it("excludes the end sentinel from the vocabulary", () => {
    const model = backend.train([example("t1", ["focus::a", "click::b"])], { maxOrder: 2 });
    expect(model.vocabulary()).toEqual(["click::b", "focus::a"]);
  });
});

describe("extractMovementExample", () => {
  it("orders a trajectory's actions by timestamp", () => {
    const span = buildTrajectorySpan({
      id: "tr1",
      sessionId: "s1",
      actions: [
        { kind: "action", tool: "click", summary: "save", ts: 30 },
        { kind: "action", tool: "focus", summary: "window", ts: 10 },
        { kind: "action", tool: "type", summary: "title", ts: 20 },
      ],
    });
    expect(extractMovementExample(span).tokens).toEqual(["focus::window", "type::title", "click::save"]);
  });

  it("extracts examples from a replay manifest's action events", () => {
    const span = buildTrajectorySpan({
      id: "tr1",
      sessionId: "s1",
      observations: [{ kind: "observation", source: "screen", summary: "ignored", ts: 5 }],
      actions: [
        { kind: "action", tool: "focus", summary: "window", ts: 10 },
        { kind: "action", tool: "click", summary: "save", ts: 20 },
      ],
    });
    const replay = buildReplayManifest({ sessionId: "s1", transcript: [], trajectories: [span] });
    const examples = extractMovementExamplesFromReplay(replay);
    expect(examples).toHaveLength(1);
    expect(examples[0]!.tokens).toEqual(["focus::window", "click::save"]);
  });
});
