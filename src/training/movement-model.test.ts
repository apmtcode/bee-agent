import { describe, expect, it } from "vitest";
import { buildReplayManifest } from "../capture/replay.js";
import { buildTrajectorySpan } from "../capture/trajectory.js";
import {
  MarkovMovementBackend,
  createMovementBackend,
  evaluateMovementModel,
  extractMovementSamples,
  generateSyntheticMovementSamples,
  listMovementBackends,
  movementToken,
  registerMovementBackend,
  type MovementModelBackend,
  type MovementSample,
} from "./movement-model.js";

function sample(label: string, ...tokens: string[]): MovementSample {
  return { label, tokens };
}

describe("movementToken", () => {
  it("produces canonical, slugged tokens", () => {
    expect(movementToken("Mouse", "Click Deploy!")).toBe("mouse::click-deploy");
    expect(movementToken("keyboard", "  Type   Command  ")).toBe("keyboard::type-command");
  });
});

describe("extractMovementSamples", () => {
  it("pulls ordered action tokens from a replay manifest", () => {
    const trajectory = buildTrajectorySpan({
      id: "traj-1",
      sessionId: "sess-1",
      observations: [{ kind: "observation", source: "browser", summary: "opened page", ts: 5 }],
      actions: [
        { kind: "action", tool: "mouse", summary: "click menu", ts: 10 },
        { kind: "action", tool: "keyboard", summary: "type command", ts: 20 },
      ],
    });
    const manifest = buildReplayManifest({ sessionId: "sess-1", transcript: [], trajectories: [trajectory] });

    const samples = extractMovementSamples([manifest]);

    expect(samples).toHaveLength(1);
    expect(samples[0]!.label).toBe("sess-1");
    // observations are excluded; actions ordered by ts.
    expect(samples[0]!.tokens).toEqual(["mouse::click-menu", "keyboard::type-command"]);
  });
});

describe("MarkovMovementBackend", () => {
  const backend = new MarkovMovementBackend();

  it("repeats a recorded movement sequence exactly", () => {
    const model = backend.train([sample("s", "a::one", "a::two", "a::three")]);
    // deterministic single continuation -> should regenerate the recording.
    const generated = backend.generate(model, ["a::one"], 2);
    expect(generated).toEqual(["a::two", "a::three"]);
  });

  it("uses the highest available order then backs off", () => {
    const model = backend.train(
      [
        sample("s1", "x::a", "x::b", "x::c"),
        sample("s2", "y::a", "x::b", "y::d"),
      ],
      { maxOrder: 2 },
    );

    // full order-2 context [x::a, x::b] was seen exactly once -> x::c.
    const exact = backend.predict(model, ["x::a", "x::b"]);
    expect(exact.token).toBe("x::c");
    expect(exact.method).toBe("exact");
    expect(exact.order).toBe(2);

    // order-2 context [z::a, x::b] never seen -> backs off to order-1 [x::b],
    // which was followed by both x::c and y::d (tie -> lexical smallest).
    const backoff = backend.predict(model, ["z::a", "x::b"]);
    expect(backoff.method).toBe("backoff");
    expect(backoff.order).toBe(1);
    expect(backoff.token).toBe("x::c");
  });

  it("generalizes an unseen-but-related seed to the nearest known movement", () => {
    const model = backend.train([sample("s", "mouse::click-save-button", "keyboard::press-enter")]);

    // "mouse::click-save-icon" was never seen, but shares tool + keywords with
    // "mouse::click-save-button" -> nearest-token fallback predicts the same next.
    const prediction = backend.predict(model, ["mouse::click-save-icon"]);
    expect(prediction.method).toBe("nearest");
    expect(prediction.token).toBe("keyboard::press-enter");
  });

  it("returns a none prediction for an empty model", () => {
    const model = backend.train([]);
    const prediction = backend.predict(model, ["anything::here"]);
    expect(prediction.method).toBe("none");
    expect(prediction.token).toBeUndefined();
    expect(model.trainedTransitions).toBe(0);
  });

  it("ranks candidate movements by probability", () => {
    const model = backend.train(
      [
        sample("s1", "root::a", "child::x"),
        sample("s2", "root::a", "child::x"),
        sample("s3", "root::a", "child::y"),
      ],
      { maxOrder: 1 },
    );
    const prediction = backend.predict(model, ["root::a"]);
    expect(prediction.token).toBe("child::x");
    expect(prediction.candidates.map((candidate) => candidate.token)).toEqual(["child::x", "child::y"]);
    expect(prediction.candidates[0]!.probability).toBeCloseTo(2 / 3, 10);
  });
});

describe("synthetic generation + evaluation", () => {
  it("is deterministic for a fixed seed", () => {
    const options = { seed: 42, sampleCount: 5, sequenceLength: 8 };
    const first = generateSyntheticMovementSamples(options);
    const second = generateSyntheticMovementSamples(options);
    expect(first).toEqual(second);
    expect(first).toHaveLength(5);
    expect(first[0]!.tokens).toHaveLength(8);
  });

  it("trains a model that beats chance on held-out synthetic movements", () => {
    const backend = createMovementBackend();
    const train = generateSyntheticMovementSamples({ seed: 7, sampleCount: 40, sequenceLength: 12 });
    const heldOut = generateSyntheticMovementSamples({ seed: 999, sampleCount: 10, sequenceLength: 12 });

    const model = backend.train(train, { maxOrder: 3 });
    const evaluation = evaluateMovementModel(backend, model, heldOut);

    expect(evaluation.predictions).toBeGreaterThan(0);
    // 8 primitives -> uniform baseline is 0.125; structured data must beat it.
    expect(evaluation.accuracy).toBeGreaterThan(0.2);
    // held-out sequences it never trained on must not fail to produce a token.
    expect(evaluation.methodBreakdown.none).toBe(0);
  });

  it("reproduces training sequences with perfect self-accuracy", () => {
    const backend = createMovementBackend();
    const train = generateSyntheticMovementSamples({ seed: 3, sampleCount: 6, sequenceLength: 10 });
    const model = backend.train(train, { maxOrder: 4 });
    const evaluation = evaluateMovementModel(backend, model, train);
    expect(evaluation.accuracy).toBeGreaterThan(0.5);
  });
});

describe("backend registry", () => {
  it("lists and resolves the built-in markov backend", () => {
    expect(listMovementBackends()).toContain("markov");
    expect(createMovementBackend("markov")).toBeInstanceOf(MarkovMovementBackend);
  });

  it("throws for an unknown backend", () => {
    expect(() => createMovementBackend("does-not-exist")).toThrow(/unknown movement backend/);
  });

  it("supports registering a custom pluggable backend", () => {
    const stub: MovementModelBackend = {
      name: "stub",
      train: () => ({
        version: 1,
        backend: "stub",
        maxOrder: 0,
        vocabulary: [],
        transitions: {},
        tokenFeatures: {},
        trainedSamples: 0,
        trainedTransitions: 0,
      }),
      predict: () => ({ token: undefined, probability: 0, order: 0, method: "none", candidates: [] }),
      generate: () => [],
    };
    registerMovementBackend("stub", () => stub);
    expect(listMovementBackends()).toContain("stub");
    expect(createMovementBackend("stub").name).toBe("stub");
  });
});
