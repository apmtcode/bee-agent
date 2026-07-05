import { describe, expect, it } from "vitest";
import {
  MOVEMENT_END_TOKEN,
  MOVEMENT_START_TOKEN,
  buildMovementDatasetFromReplays,
  buildMovementDatasetFromTrajectories,
  createMovementBackend,
  movementTokenFromAction,
  normalizeTargetClass,
  splitMovementDataset,
  type MovementDataset,
} from "./movement-model.js";
import { MarkovMovementBackend, MarkovMovementModel } from "./markov-backend.js";
import { MovementTrainingService, evaluateMovementModel } from "./movement-training-service.js";
import { buildTrajectorySpan } from "../capture/trajectory.js";

function dataset(sequences: Array<{ id: string; tokens: string[] }>): MovementDataset {
  const vocab = new Set<string>([MOVEMENT_END_TOKEN]);
  for (const sequence of sequences) {
    for (const token of sequence.tokens) {
      vocab.add(token);
    }
  }
  return { version: 1, sequences, vocabulary: [...vocab].sort() };
}

describe("movement tokenizer", () => {
  it("derives a coarse tool:verb:target token and drops specific ids", () => {
    const token = movementTokenFromAction({
      tool: "device",
      summary: "tapped Submit",
      metadata: { gesture: "tap", target: "button-42" },
    });
    expect(token).toBe("device:tap:button");
  });

  it("collapses ids so distinct instances share a token (generalization)", () => {
    const a = movementTokenFromAction({ tool: "device", summary: "typed", metadata: { gesture: "type", target: "field-3" } });
    const b = movementTokenFromAction({ tool: "device", summary: "typed", metadata: { gesture: "type", target: "field-7" } });
    expect(a).toBe(b);
    expect(a).toBe("device:type:field");
  });

  it("falls back to the summary verb when no gesture metadata is present", () => {
    expect(movementTokenFromAction({ tool: "browser", summary: "clicked login" })).toBe("browser:clicked:login");
  });

  it("normalizeTargetClass strips numeric/hex suffixes", () => {
    expect(normalizeTargetClass("row-000abc12")).toBe("row");
    expect(normalizeTargetClass("Menu Item 3")).toBe("menu");
    expect(normalizeTargetClass(undefined)).toBeUndefined();
  });
});

describe("dataset builders", () => {
  it("builds one sequence per trajectory, ordered by timestamp", () => {
    const trajectory = buildTrajectorySpan({
      id: "t1",
      sessionId: "s1",
      actions: [
        { kind: "action", tool: "device", summary: "swiped up", ts: 30, metadata: { gesture: "swipe", direction: "up" } },
        { kind: "action", tool: "device", summary: "tapped Menu", ts: 10, metadata: { gesture: "tap", target: "menu" } },
      ],
    });
    const built = buildMovementDatasetFromTrajectories([trajectory]);
    expect(built.sequences).toHaveLength(1);
    expect(built.sequences[0].tokens).toEqual(["device:tap:menu", "device:swipe:up"]);
    expect(built.vocabulary).toContain(MOVEMENT_END_TOKEN);
  });

  it("builds sequences from replay action events, ignoring transcript/observation", () => {
    const built = buildMovementDatasetFromReplays([
      {
        sessionId: "s1",
        trajectoryIds: ["t1"],
        eventCount: 3,
        events: [
          { kind: "transcript", ts: 1, messageId: "m1", role: "user", content: "go" },
          { kind: "observation", ts: 2, trajectoryId: "t1", source: "os", summary: "focused App" },
          { kind: "action", ts: 3, trajectoryId: "t1", tool: "device", summary: "tapped Send", },
        ],
      },
    ]);
    expect(built.sequences[0].tokens).toEqual(["device:tapped:send"]);
  });
});

describe("markov backend training + replay", () => {
  it("reproduces a single recorded movement exactly (replay fidelity)", () => {
    const model = new MarkovMovementBackend().train(
      dataset([{ id: "seq", tokens: ["a", "b", "c", "d"] }]),
    );
    expect(model.generate()).toEqual(["a", "b", "c", "d"]);
    const prediction = model.predictNext([MOVEMENT_START_TOKEN, "a"]);
    expect(prediction?.token).toBe("b");
  });

  it("predicts the end sentinel after the final recorded token", () => {
    const model = new MarkovMovementBackend().train(dataset([{ id: "seq", tokens: ["a", "b"] }]));
    expect(model.predictNext([MOVEMENT_START_TOKEN, "a", "b"])?.token).toBe(MOVEMENT_END_TOKEN);
  });

  it("is deterministic — same dataset yields identical generation", () => {
    const data = dataset([
      { id: "1", tokens: ["open", "search", "type", "enter"] },
      { id: "2", tokens: ["open", "search", "type", "enter"] },
    ]);
    const first = new MarkovMovementBackend().train(data).generate();
    const second = new MarkovMovementBackend().train(data).generate();
    expect(first).toEqual(second);
    expect(first).toEqual(["open", "search", "type", "enter"]);
  });

  it("generalizes: recombines learned transitions into a novel-but-related path", () => {
    // Two recorded movements share the middle step "b". A bigram model can
    // reach a path (x -> b -> c) that was never recorded verbatim.
    const model = new MarkovMovementBackend({ order: 1 }).train(
      dataset([
        { id: "1", tokens: ["a", "b", "c"] },
        { id: "2", tokens: ["x", "b", "d"] },
      ]),
    );
    const generalized = model.generate({ seed: [MOVEMENT_START_TOKEN, "x"] });
    // From x -> b (learned), then b -> {c,d} tie broken lexically to c: a path
    // (x, b, c) never seen together in training.
    expect(generalized).toEqual(["b", "c"]);
  });

  it("assigns higher probability to related held-out sequences than to unrelated ones", () => {
    const model = new MarkovMovementBackend({ order: 2 }).train(
      dataset([
        { id: "1", tokens: ["open", "nav", "click", "save"] },
        { id: "2", tokens: ["open", "nav", "click", "close"] },
      ]),
    ) as MarkovMovementModel;
    const related = model.scoreSequence(["open", "nav", "click", "save"]);
    const unrelated = model.scoreSequence(["zzz", "qqq", "www"]);
    expect(related).toBeGreaterThan(unrelated);
  });

  it("round-trips through serialization", () => {
    const backend = new MarkovMovementBackend({ order: 2 });
    const data = dataset([{ id: "1", tokens: ["a", "b", "c"] }]);
    const trained = backend.train(data);
    const serialized = JSON.parse(JSON.stringify(trained.toJSON()));
    const loaded = backend.load(serialized);
    expect(loaded.generate()).toEqual(trained.generate());
    expect(loaded.scoreSequence(["a", "b", "c"])).toBeCloseTo(trained.scoreSequence(["a", "b", "c"]));
  });

  it("returns undefined prediction for an empty (untrained) model", () => {
    const model = new MarkovMovementBackend().train({ version: 1, sequences: [], vocabulary: [] });
    expect(model.predictNext([MOVEMENT_START_TOKEN])).toBeUndefined();
    expect(model.generate()).toEqual([]);
  });
});

describe("createMovementBackend registry", () => {
  it("creates the markov backend by default", () => {
    expect(createMovementBackend().kind).toBe("markov");
    expect(createMovementBackend("markov", { order: 3 }).kind).toBe("markov");
  });
});

describe("MovementTrainingService + eval harness", () => {
  it("trains from trajectories and reports perfect replay fidelity", () => {
    const trajectory = buildTrajectorySpan({
      id: "t1",
      sessionId: "s1",
      actions: [
        { kind: "action", tool: "device", summary: "tapped Compose", ts: 1, metadata: { gesture: "tap", target: "compose" } },
        { kind: "action", tool: "device", summary: "typed body", ts: 2, metadata: { gesture: "type", target: "body" } },
        { kind: "action", tool: "device", summary: "tapped Send", ts: 3, metadata: { gesture: "tap", target: "send" } },
      ],
    });
    const result = new MovementTrainingService().trainFromTrajectories([trajectory]);
    expect(result.replayEvaluation.exactMatchRate).toBe(1);
    expect(result.replayEvaluation.meanTokenAccuracy).toBe(1);
    expect(result.model.generate()).toEqual(["device:tap:compose", "device:type:body", "device:tap:send"]);
  });

  it("produces a generalization evaluation over a held-out split", () => {
    const sequences = Array.from({ length: 6 }, (_, index) => ({
      id: `seq-${index}`,
      tokens: ["open", "nav", "click", index % 2 === 0 ? "save" : "close"],
    }));
    const result = new MovementTrainingService("markov").trainFromDataset(dataset(sequences), {
      holdoutEvery: 3,
    });
    expect(result.generalizationEvaluation).toBeDefined();
    expect(result.generalizationEvaluation?.sequenceCount).toBeGreaterThan(0);
    // Held-out sequences share the learned "open nav click" prefix, so mean
    // log-prob should be finite and well above negative infinity.
    expect(Number.isFinite(result.generalizationEvaluation?.meanLogProb ?? -Infinity)).toBe(true);
    expect(result.generalizationEvaluation?.meanTokenAccuracy).toBeGreaterThan(0.5);
  });

  it("evaluateMovementModel handles an empty dataset without dividing by zero", () => {
    const model = createMovementBackend().train({ version: 1, sequences: [], vocabulary: [] });
    const evaluation = evaluateMovementModel(model, { version: 1, sequences: [], vocabulary: [] });
    expect(evaluation.sequenceCount).toBe(0);
    expect(evaluation.exactMatchRate).toBe(0);
  });

  it("splitMovementDataset holds out every Nth sequence deterministically", () => {
    const data = dataset(Array.from({ length: 5 }, (_, i) => ({ id: `s${i}`, tokens: ["a"] })));
    const { train, holdout } = splitMovementDataset(data, 2);
    expect(holdout.sequences.map((s) => s.id)).toEqual(["s1", "s3"]);
    expect(train.sequences.map((s) => s.id)).toEqual(["s0", "s2", "s4"]);
  });
});
