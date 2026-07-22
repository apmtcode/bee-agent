import { describe, expect, it } from "vitest";
import { MarkovMovementBackend } from "./backends/markov-movement-backend.js";
import {
  buildMovementDataset,
  evaluateGeneralization,
  evaluateReplayFidelity,
  mergeMovementDatasets,
  tokenizeReplayEvent,
  tokenizeTrajectory,
  type MovementDataset,
} from "./movement-model.js";
import {
  generateSyntheticCorpus,
  generateSyntheticReplay,
} from "./synthetic-movements.js";
import type { ReplayManifest } from "../capture/replay.js";
import type { TrajectorySpan } from "../capture/trajectory.js";

// Distinct-verb macro: every movement is unique, so order-1 context already
// determines the next movement — a cleanly learnable recorded sequence.
const GRAMMAR = {
  verbs: ["focus.window", "mouse.move", "mouse.click", "keyboard.type", "keyboard.enter"],
  observationSources: ["screen"],
};

// Repeated-verb macro: `mouse.click` recurs, so the local pattern
// `obs, mouse.click, obs` precedes two different next movements. Only a longer
// context can disambiguate — exercises the model's order sensitivity.
const REPEATED_GRAMMAR = {
  verbs: ["focus.window", "mouse.click", "keyboard.type", "mouse.click", "keyboard.enter"],
  observationSources: ["screen"],
};

describe("movement tokenization", () => {
  it("encodes movement verbs by kind, lowercased and space-normalized", () => {
    expect(
      tokenizeReplayEvent({ kind: "action", ts: 0, trajectoryId: "t", tool: "Mouse Click", summary: "" }),
    ).toBe("act:mouse_click");
    expect(
      tokenizeReplayEvent({ kind: "observation", ts: 0, trajectoryId: "t", source: "Screen", summary: "" }),
    ).toBe("obs:screen");
    expect(
      tokenizeReplayEvent({ kind: "transcript", ts: 0, messageId: "m", role: "assistant", content: "" }),
    ).toBe("msg:assistant");
  });

  it("orders trajectory tokens by timestamp with observations before same-ts actions", () => {
    const trajectory: TrajectorySpan = {
      id: "t",
      sessionId: "s",
      createdAt: "2026-01-01T00:00:00.000Z",
      captureTier: "full",
      observations: [{ kind: "observation", ts: 10, source: "screen", summary: "" }],
      actions: [
        { kind: "action", ts: 10, tool: "mouse.click", summary: "" },
        { kind: "action", ts: 5, tool: "focus.window", summary: "" },
      ],
    };
    expect(tokenizeTrajectory(trajectory)).toEqual(["act:focus.window", "obs:screen", "act:mouse.click"]);
  });
});

describe("buildMovementDataset", () => {
  it("keeps only movement events (drops transcript chatter) and empties", () => {
    const replay: ReplayManifest = {
      version: 1,
      sessionId: "s1",
      trajectoryIds: ["t"],
      eventCount: 3,
      events: [
        { kind: "transcript", ts: 0, messageId: "m", role: "user", content: "hi" },
        { kind: "observation", ts: 1, trajectoryId: "t", source: "screen", summary: "" },
        { kind: "action", ts: 2, trajectoryId: "t", tool: "mouse.click", summary: "" },
      ],
    };
    const empty: ReplayManifest = {
      version: 1,
      sessionId: "s2",
      trajectoryIds: [],
      eventCount: 1,
      events: [{ kind: "transcript", ts: 0, messageId: "m", role: "user", content: "hi" }],
    };
    const dataset = buildMovementDataset([replay, empty]);
    expect(dataset.sequences).toEqual([{ sessionId: "s1", tokens: ["obs:screen", "act:mouse.click"] }]);
  });

  it("merges datasets without aliasing token arrays", () => {
    const a: MovementDataset = { version: 1, sequences: [{ sessionId: "a", tokens: ["act:x"] }] };
    const b: MovementDataset = { version: 1, sequences: [{ sessionId: "b", tokens: ["act:y"] }] };
    const merged = mergeMovementDatasets(a, b);
    merged.sequences[0]!.tokens.push("mutated");
    expect(a.sequences[0]!.tokens).toEqual(["act:x"]);
    expect(merged.sequences).toHaveLength(2);
  });
});

describe("MarkovMovementBackend — repeat recorded movements", () => {
  it("greedily reproduces a recorded movement sequence exactly", () => {
    const replay = generateSyntheticReplay({ sessionId: "s", grammar: GRAMMAR });
    const dataset = buildMovementDataset([replay]);
    const model = new MarkovMovementBackend().train(dataset, { order: 3 });

    const fidelity = evaluateReplayFidelity(model, dataset.sequences[0]!);
    expect(fidelity.exactMatch).toBe(true);
    expect(fidelity.tokenAccuracy).toBe(1);
  });

  it("captures a repeated-verb macro exactly once the order is high enough", () => {
    const replay = generateSyntheticReplay({ sessionId: "s", grammar: REPEATED_GRAMMAR });
    const dataset = buildMovementDataset([replay]);
    const backend = new MarkovMovementBackend();

    // Order 3 is too short to disambiguate the two `mouse.click` states...
    const shallow = evaluateReplayFidelity(backend.train(dataset, { order: 3 }), dataset.sequences[0]!);
    expect(shallow.exactMatch).toBe(false);

    // ...but a longer context reaches the distinguishing earlier movement.
    const deep = evaluateReplayFidelity(backend.train(dataset, { order: 5 }), dataset.sequences[0]!);
    expect(deep.exactMatch).toBe(true);
  });

  it("survives a serialize/load round-trip with identical behaviour", () => {
    const replay = generateSyntheticReplay({ sessionId: "s", grammar: GRAMMAR });
    const dataset = buildMovementDataset([replay]);
    const backend = new MarkovMovementBackend();
    const model = backend.train(dataset, { order: 3 });

    const restored = backend.load(JSON.parse(JSON.stringify(model.serialize())));
    expect(restored.generate([])).toEqual(model.generate([]));
    expect(restored.order).toBe(model.order);
  });
});

describe("MarkovMovementBackend — generalize to related movements", () => {
  it("predicts continuations for held-out but related streams via back-off", () => {
    const training = generateSyntheticCorpus({
      grammar: GRAMMAR,
      count: 6,
      seed: 11,
      dropoutRate: 0.15,
      sessionPrefix: "train",
    });
    const heldOut = generateSyntheticCorpus({
      grammar: GRAMMAR,
      count: 3,
      seed: 999,
      dropoutRate: 0.15,
      sessionPrefix: "eval",
    });

    const model = new MarkovMovementBackend().train(buildMovementDataset(training), { order: 3 });
    const result = evaluateGeneralization(model, buildMovementDataset(heldOut).sequences);

    // The model has never seen these exact streams, yet the shared grammar means
    // it should predict the next movement correctly a strong majority of the time.
    expect(result.evaluatedPredictions).toBeGreaterThan(0);
    expect(result.nextTokenAccuracy).toBeGreaterThan(0.6);
  });

  it("backs off to a shorter context when the full context is unseen", () => {
    const dataset = buildMovementDataset([generateSyntheticReplay({ sessionId: "s", grammar: GRAMMAR })]);
    const model = new MarkovMovementBackend().train(dataset, { order: 3 });

    // A context the training data never contained verbatim.
    const prediction = model.predictNext(["act:never.seen.before", "act:also.novel"]);
    expect(prediction.token).toBeTruthy();
    expect(prediction.conditionedOrder).toBeLessThan(2);
  });

  it("returns EOS for an empty model so generation always terminates", () => {
    const model = new MarkovMovementBackend().train({ version: 1, sequences: [] });
    expect(model.generate([])).toEqual([]);
    expect(model.predictNext(["act:x"]).token).toBe("<eos>");
  });
});

describe("synthetic streams", () => {
  it("are deterministic for a fixed seed and vary with dropout", () => {
    const a = generateSyntheticReplay({ sessionId: "s", grammar: GRAMMAR, seed: 42, dropoutRate: 0.3 });
    const b = generateSyntheticReplay({ sessionId: "s", grammar: GRAMMAR, seed: 42, dropoutRate: 0.3 });
    expect(a.events).toEqual(b.events);

    const full = generateSyntheticReplay({ sessionId: "s", grammar: GRAMMAR, dropoutRate: 0 });
    // With observations interleaved, every verb yields one obs + one action.
    expect(full.events.filter((event) => event.kind === "action")).toHaveLength(GRAMMAR.verbs.length);
  });
});
