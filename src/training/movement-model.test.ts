import { describe, expect, it } from "vitest";
import type { ReplayTimelineEvent } from "../capture/replay.js";
import type { TrajectorySpan } from "../capture/trajectory.js";
import {
  MarkovMovementBackend,
  buildMovementDataset,
  defaultMovementTokenizer,
  evaluateMovementModel,
  movementSequencesFromReplays,
  movementSequencesFromTrajectories,
  resolveMovementBackend,
  synthesizeMovementSequences,
  trainMovementModel,
} from "./movement-model.js";

/** Build a token the same way the default tokenizer does, for expectations. */
function tok(tool: string, summary: string): string {
  return defaultMovementTokenizer({ tool, summary });
}

function action(tool: string, summary: string, ts: number): TrajectorySpan["actions"][number] {
  return { kind: "action", tool, summary, ts };
}

function trajectory(id: string, actions: TrajectorySpan["actions"]): TrajectorySpan {
  return {
    id,
    sessionId: "session",
    createdAt: "2026-01-01T00:00:00.000Z",
    captureTier: "app",
    observations: [],
    actions,
  };
}

describe("movement tokenization", () => {
  it("normalizes summaries and sorts actions by timestamp", () => {
    const sequences = movementSequencesFromTrajectories([
      trajectory("t1", [action("click", "  Save  Button ", 20), action("type", "Hello", 10)]),
    ]);
    expect(sequences[0].tokens).toEqual([tok("type", "Hello"), tok("click", "Save Button")]);
  });

  it("extracts only action events from replay manifests, in timestamp order", () => {
    const events: ReplayTimelineEvent[] = [
      { kind: "action", ts: 3, trajectoryId: "t", tool: "click", summary: "b" },
      { kind: "transcript", ts: 1, messageId: "m", role: "user", content: "hi" },
      { kind: "observation", ts: 2, trajectoryId: "t", source: "os", summary: "x" },
      { kind: "action", ts: 1, trajectoryId: "t", tool: "open", summary: "a" },
    ];
    const sequences = movementSequencesFromReplays([{ trajectoryIds: ["t"], events }]);
    expect(sequences[0].tokens).toEqual([tok("open", "a"), tok("click", "b")]);
  });
});

describe("MarkovMovementBackend — repeat recorded movements", () => {
  it("reproduces a recorded trajectory exactly given its prefix", () => {
    const seq = movementSequencesFromTrajectories([
      trajectory("t1", [
        action("open", "menu", 1),
        action("click", "file", 2),
        action("click", "save", 3),
        action("press", "enter", 4),
      ]),
    ]);
    const model = trainMovementModel(buildMovementDataset(seq), { order: 2 });
    const rollout = model.generate(seq[0].tokens.slice(0, 1), 3);
    expect([...seq[0].tokens.slice(0, 1), ...rollout]).toEqual(seq[0].tokens);
  });

  it("predicts the highest-order continuation and reports the order used", () => {
    const seq = movementSequencesFromTrajectories([
      trajectory("t1", [action("a", "1", 1), action("b", "2", 2), action("c", "3", 3)]),
    ]);
    const model = trainMovementModel(buildMovementDataset(seq), { order: 2 });
    const prediction = model.predictNext([tok("a", "1"), tok("b", "2")]);
    expect(prediction.token).toBe(tok("c", "3"));
    expect(prediction.order).toBe(2);
    expect(prediction.probability).toBe(1);
  });
});

describe("MarkovMovementBackend — generalize via backoff", () => {
  it("produces a plausible next movement for an unseen high-order context", () => {
    // Two related tasks that share the "open menu -> click X -> click save" shape.
    const seq = movementSequencesFromTrajectories([
      trajectory("t1", [action("open", "menu", 1), action("click", "file", 2), action("click", "save", 3)]),
      trajectory("t2", [action("open", "menu", 1), action("click", "edit", 2), action("click", "save", 3)]),
    ]);
    const model = trainMovementModel(buildMovementDataset(seq), { order: 2 });
    // Novel context: "open menu -> click view" was never recorded, but the
    // order-1 context "click view" backs off to the learned "-> click save".
    const prediction = model.predictNext([tok("open", "menu"), tok("click", "view")]);
    expect(prediction.token).toBe(tok("click", "save"));
    expect(prediction.order).toBeLessThan(2); // came from backoff, not exact recall
  });

  it("falls back to the unigram most-common movement with no usable context", () => {
    const seq = movementSequencesFromTrajectories([
      trajectory("t1", [action("x", "1", 1), action("x", "1", 2), action("y", "2", 3)]),
    ]);
    const model = trainMovementModel(buildMovementDataset(seq), { order: 2 });
    const prediction = model.predictNext(["totally-unknown-context"]);
    expect(prediction.order).toBe(0);
    expect(prediction.token).toBe(tok("x", "1")); // most frequent overall
  });

  it("returns an empty prediction for an untrained model", () => {
    const model = new MarkovMovementBackend().train(buildMovementDataset([]));
    expect(model.predictNext(["anything"])).toEqual({ token: undefined, probability: 0, order: -1, candidates: [] });
  });
});

describe("determinism and persistence", () => {
  it("is deterministic across retrains and tie-breaks stably", () => {
    const dataset = buildMovementDataset(
      movementSequencesFromTrajectories([
        trajectory("t1", [action("start", "s", 1), action("a", "1", 2)]),
        trajectory("t2", [action("start", "s", 1), action("b", "2", 2)]),
      ]),
    );
    const a = trainMovementModel(dataset, { order: 1 });
    const b = trainMovementModel(dataset, { order: 1 });
    expect(a.serialize()).toEqual(b.serialize());
    // The two continuations are tied after "start"; lexicographic tie-break wins.
    expect(a.predictNext([tok("start", "s")]).token).toBe(tok("a", "1"));
  });

  it("round-trips through serialize/load with identical predictions", () => {
    const dataset = buildMovementDataset(
      movementSequencesFromTrajectories([
        trajectory("t1", [action("open", "app", 1), action("click", "go", 2), action("done", "ok", 3)]),
      ]),
    );
    const backend = resolveMovementBackend("markov");
    const model = backend.train(dataset, { order: 2 });
    const reloaded = backend.load(model.serialize());
    expect(reloaded.serialize()).toEqual(model.serialize());
    expect(reloaded.predictNext([tok("open", "app"), tok("click", "go")]).token).toBe(tok("done", "ok"));
  });

  it("rejects unknown backends", () => {
    expect(() => resolveMovementBackend("nope")).toThrow(/unknown movement backend/);
  });
});

describe("generalization eval harness", () => {
  it("measures perfect recall when held-out equals training", () => {
    const sequences = synthesizeMovementSequences({
      // Distinct first tokens so each prefix uniquely determines its continuation.
      templates: [
        { id: "save", tokens: ["launch-editor", "click-file", "click-save", "press-enter"] },
        { id: "quit", tokens: ["open-menu", "click-file", "click-quit", "press-enter"] },
      ],
      countPerTemplate: 3,
    });
    const model = trainMovementModel(buildMovementDataset(sequences), { order: 3 });
    const result = evaluateMovementModel(model, sequences);
    expect(result.accuracy).toBe(1);
    expect(result.exactMatchSequences).toBe(sequences.length);
  });

  it("generalizes to perturbed held-out trajectories well above chance", () => {
    const templates = [
      { id: "save", tokens: ["open-menu", "click-file", "click-save", "press-enter"] },
      { id: "quit", tokens: ["open-menu", "click-edit", "click-quit", "press-enter"] },
    ];
    const train = synthesizeMovementSequences({ templates, countPerTemplate: 4, seed: 7 });
    const heldOut = synthesizeMovementSequences({ templates, countPerTemplate: 4, seed: 42, perturb: true });
    const model = trainMovementModel(buildMovementDataset(train), { order: 2 });
    const result = evaluateMovementModel(model, heldOut);
    // Vocabulary is 6 tokens, so chance top-1 is ~0.17; backoff should beat it comfortably.
    expect(result.accuracy).toBeGreaterThan(0.5);
    expect(result.totalPredictions).toBeGreaterThan(0);
    const backoffTotal = Object.entries(result.byOrder)
      .filter(([order]) => Number(order) < 2)
      .reduce((sum, [, bucket]) => sum + bucket.total, 0);
    expect(backoffTotal).toBeGreaterThan(0); // generalization exercised the backoff path
  });
});

describe("tokenizer contract", () => {
  it("default tokenizer joins tool and normalized summary", () => {
    const token = defaultMovementTokenizer({ tool: "Click", summary: "  Multi   Word " });
    expect(token).toBe(`Clickmulti word`);
  });
});
