import { describe, expect, it } from "vitest";
import { buildTrajectorySpan, type TrajectoryAction, type TrajectorySpan } from "../capture/trajectory.js";
import {
  actionToken,
  buildMovementDataset,
  evaluateMovementModel,
  NGramMovementBackend,
  tokenizeTrajectory,
} from "./movement-model.js";

function action(tool: string, summary: string, ts: number): TrajectoryAction {
  return { kind: "action", tool, summary, ts };
}

function span(id: string, actions: TrajectoryAction[]): TrajectorySpan {
  return buildTrajectorySpan({ id, sessionId: "s1", actions });
}

/** Deterministic synthetic movement trajectory (no real OS input). */
function syntheticSpan(id: string, tools: Array<[string, string]>, startTs = 1000): TrajectorySpan {
  return span(
    id,
    tools.map(([tool, summary], index) => action(tool, summary, startTs + index * 10)),
  );
}

describe("tokenizeTrajectory", () => {
  it("orders tokens by timestamp and slugifies tool + summary", () => {
    const sequence = tokenizeTrajectory(
      span("t1", [action("device", "Tapped Submit Button", 30), action("device", "Focus Field", 10)]),
    );
    expect(sequence.tokens).toEqual(["device:focus-field", "device:tapped-submit-button"]);
  });

  it("can include observations interleaved with actions", () => {
    const base = span("t2", [action("device", "tap", 20)]);
    const withObs: TrajectorySpan = {
      ...base,
      observations: [{ kind: "observation", source: "os", summary: "focused editor", ts: 10 }],
    };
    const sequence = tokenizeTrajectory(withObs, { includeObservations: true });
    expect(sequence.tokens).toEqual(["obs:os:focused-editor", "device:tap"]);
  });

  it("filters actions by tool when requested", () => {
    const sequence = tokenizeTrajectory(
      span("t3", [action("device", "tap", 10), action("browser", "click", 20)]),
      { tools: ["browser"] },
    );
    expect(sequence.tokens).toEqual(["browser:click"]);
  });
});

describe("buildMovementDataset", () => {
  it("produces a sorted, de-duplicated vocabulary and drops empty sequences", () => {
    const dataset = buildMovementDataset([
      syntheticSpan("a", [["device", "b"], ["device", "a"]]),
      syntheticSpan("b", [["device", "a"]]),
      span("empty", []),
    ]);
    expect(dataset.sequences.map((s) => s.id)).toEqual(["a", "b"]);
    expect(dataset.vocabulary).toEqual(["device:a", "device:b"]);
  });
});

describe("NGramMovementBackend — repeat recorded movements", () => {
  it("replays a recorded sequence verbatim from its seed", () => {
    const recorded = syntheticSpan("rec", [
      ["os", "focus editor"],
      ["device", "type hello"],
      ["device", "save file"],
      ["device", "run tests"],
    ]);
    const dataset = buildMovementDataset([recorded]);
    const model = new NGramMovementBackend({ order: 3 }).train(dataset);

    const seed = [actionToken("os", "focus editor")];
    const rollout = model.generate(seed, 10);
    expect(rollout).toEqual([
      actionToken("device", "type hello"),
      actionToken("device", "save file"),
      actionToken("device", "run tests"),
    ]);
  });

  it("marks an exactly-seen context prediction as source=exact", () => {
    const dataset = buildMovementDataset([
      syntheticSpan("rec", [["a", "1"], ["b", "2"], ["c", "3"]]),
    ]);
    const model = new NGramMovementBackend({ order: 3 }).train(dataset);
    const prediction = model.predictNext([actionToken("a", "1"), actionToken("b", "2")]);
    expect(prediction.token).toBe(actionToken("c", "3"));
    expect(prediction.source).toBe("exact");
    expect(prediction.confidence).toBeGreaterThan(0);
  });

  it("memorizes non-conflicting training data with perfect eval accuracy", () => {
    const dataset = buildMovementDataset([
      syntheticSpan("x", [["a", "1"], ["b", "2"], ["c", "3"]]),
      syntheticSpan("y", [["d", "1"], ["e", "2"], ["f", "3"]]),
    ]);
    const model = new NGramMovementBackend({ order: 3 }).train(dataset);
    const evaluation = evaluateMovementModel(model, dataset.sequences);
    expect(evaluation.accuracy).toBe(1);
    expect(evaluation.correct).toBe(evaluation.predictions);
  });
});

describe("NGramMovementBackend — generalize to related movements", () => {
  it("backs off to a shorter context for an unseen prefix", () => {
    // Both trajectories teach "open -> edit" and "edit -> save"; a novel
    // combination the model never saw as a full trigram should still predict
    // the learned continuation via bigram backoff.
    const dataset = buildMovementDataset([
      syntheticSpan("t1", [["app", "open project"], ["app", "edit file"], ["app", "save"]]),
      syntheticSpan("t2", [["app", "search"], ["app", "edit file"], ["app", "save"]]),
    ]);
    const model = new NGramMovementBackend({ order: 3 }).train(dataset);

    // Novel two-token context never seen together, but "edit file" -> "save"
    // is a strong bigram.
    const prediction = model.predictNext([actionToken("app", "run"), actionToken("app", "edit file")]);
    expect(prediction.token).toBe(actionToken("app", "save"));
    expect(prediction.source).toBe("backoff");
  });

  it("falls back to the unigram prior when no context matches", () => {
    const dataset = buildMovementDataset([
      syntheticSpan("t1", [["app", "a"], ["app", "common"], ["app", "common"]]),
    ]);
    const model = new NGramMovementBackend({ order: 3 }).train(dataset);
    const prediction = model.predictNext([actionToken("zzz", "never seen")]);
    expect(prediction.token).toBe(actionToken("app", "common"));
    expect(prediction.source).toBe("prior");
  });

  it("returns a null prediction for an untrained model", () => {
    const model = new NGramMovementBackend().train({ version: 1, sequences: [], vocabulary: [] });
    const prediction = model.predictNext([actionToken("a", "b")]);
    expect(prediction.token).toBeNull();
    expect(prediction.source).toBe("unknown");
    expect(prediction.confidence).toBe(0);
  });
});

describe("NGramMovementBackend — serialization is a pluggable, lossless seam", () => {
  it("restore() reproduces identical predictions", () => {
    const dataset = buildMovementDataset([
      syntheticSpan("t1", [["a", "1"], ["b", "2"], ["c", "3"]]),
      syntheticSpan("t2", [["a", "1"], ["b", "2"], ["d", "4"]]),
    ]);
    const backend = new NGramMovementBackend({ order: 3 });
    const model = backend.train(dataset);
    const serialized = model.serialize();
    expect(serialized.backend).toBe("ngram");
    expect(serialized.order).toBe(3);

    const restored = backend.restore(serialized);
    for (const sequence of dataset.sequences) {
      for (let i = 1; i < sequence.tokens.length; i++) {
        const context = sequence.tokens.slice(0, i);
        expect(restored.predictNext(context)).toEqual(model.predictNext(context));
      }
    }
  });

  it("breaks ties deterministically (lexicographic on equal counts)", () => {
    // Context "start" is followed once by each of two tokens; the model must
    // always pick the lexicographically smaller one.
    const dataset = buildMovementDataset([
      syntheticSpan("t1", [["app", "start"], ["app", "zeta"]]),
      syntheticSpan("t2", [["app", "start"], ["app", "alpha"]]),
    ]);
    const model = new NGramMovementBackend({ order: 2 }).train(dataset);
    const prediction = model.predictNext([actionToken("app", "start")]);
    expect(prediction.token).toBe(actionToken("app", "alpha"));
  });
});
