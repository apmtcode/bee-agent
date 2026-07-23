import { describe, expect, it } from "vitest";
import { buildTrajectorySpan, type TrajectorySpan } from "../capture/trajectory.js";
import { buildReplayManifest } from "../capture/replay.js";
import {
  MarkovMovementBackend,
  MovementModelTrainer,
  MOVEMENT_END_TOKEN,
  MOVEMENT_START_TOKEN,
  actionToken,
  buildMovementDataset,
  buildMovementDatasetFromReplays,
  evaluateNextTokenAccuracy,
  splitMovementDataset,
  tokenizeTrajectory,
} from "./movement-model.js";

/** Synthetic trajectory factory — no real OS input, deterministic timestamps. */
function trajectory(id: string, tools: string[], reward?: number): TrajectorySpan {
  return buildTrajectorySpan({
    id,
    sessionId: `session-${id}`,
    actions: tools.map((tool, index) => ({
      kind: "action" as const,
      tool,
      summary: `${tool} step ${index}`,
      ts: index + 1,
    })),
    ...(reward !== undefined
      ? { outcome: { status: "success" as const, summary: "ok", reward } }
      : {}),
  });
}

describe("movement dataset tokenization", () => {
  it("frames action sequences with sentinels in timestamp order", () => {
    const tokens = tokenizeTrajectory(trajectory("t1", ["click", "type", "submit"]));
    expect(tokens).toEqual([
      MOVEMENT_START_TOKEN,
      actionToken("click"),
      actionToken("type"),
      actionToken("submit"),
      MOVEMENT_END_TOKEN,
    ]);
  });

  it("orders interleaved observations and actions by timestamp when requested", () => {
    const span = buildTrajectorySpan({
      id: "t2",
      sessionId: "s2",
      observations: [{ kind: "observation", source: "screen", summary: "focus", ts: 2 }],
      actions: [
        { kind: "action", tool: "click", summary: "c", ts: 1 },
        { kind: "action", tool: "type", summary: "t", ts: 3 },
      ],
    });
    const tokens = tokenizeTrajectory(span, { includeObservations: true });
    expect(tokens).toEqual([
      MOVEMENT_START_TOKEN,
      actionToken("click"),
      "obs:screen",
      actionToken("type"),
      MOVEMENT_END_TOKEN,
    ]);
  });

  it("builds a sorted vocabulary and drops empty sequences", () => {
    const dataset = buildMovementDataset([
      trajectory("t1", ["click", "type"]),
      trajectory("empty", []),
    ]);
    expect(dataset.sequences).toHaveLength(1);
    expect(dataset.vocabulary).toEqual([...dataset.vocabulary].sort());
    expect(dataset.vocabulary).toContain(actionToken("click"));
  });

  it("derives movement tokens from replay manifests", () => {
    const span = trajectory("t1", ["click", "type"]);
    const replay = buildReplayManifest({ sessionId: "s1", transcript: [], trajectories: [span] });
    const dataset = buildMovementDatasetFromReplays([replay]);
    expect(dataset.sequences[0]?.tokens).toEqual([
      MOVEMENT_START_TOKEN,
      actionToken("click"),
      actionToken("type"),
      MOVEMENT_END_TOKEN,
    ]);
  });
});

describe("MarkovMovementBackend training + replay", () => {
  it("replays a recorded movement deterministically via greedy generation", async () => {
    const backend = new MarkovMovementBackend();
    const dataset = buildMovementDataset([trajectory("t1", ["click", "type", "submit"])]);
    const model = await backend.train(dataset, { order: 1, trainedAt: "2026-07-23T00:00:00Z" });

    expect(model.trainedAt).toBe("2026-07-23T00:00:00Z");
    const result = backend.generate(model, [MOVEMENT_START_TOKEN]);
    expect(result.stoppedReason).toBe("end");
    expect(result.generated).toEqual([actionToken("click"), actionToken("type"), actionToken("submit")]);
  });

  it("returns a ranked, normalized probability distribution", async () => {
    const backend = new MarkovMovementBackend();
    // From "click": type appears twice, submit once -> type is argmax.
    const dataset = buildMovementDataset([
      trajectory("a", ["click", "type"]),
      trajectory("b", ["click", "type"]),
      trajectory("c", ["click", "submit"]),
    ]);
    const model = await backend.train(dataset, { order: 1 });
    const prediction = backend.predict(model, [actionToken("click")]);

    expect(prediction.next).toBe(actionToken("type"));
    const total = prediction.candidates.reduce((sum, c) => sum + c.probability, 0);
    expect(total).toBeCloseTo(1, 10);
    expect(prediction.candidates[0]?.probability).toBeCloseTo(2 / 3, 10);
  });

  it("reports no candidates for an unknown context", async () => {
    const backend = new MarkovMovementBackend();
    const model = await backend.train(buildMovementDataset([trajectory("t1", ["click"])]), { order: 1 });
    const prediction = backend.predict(model, [actionToken("never-seen")]);
    expect(prediction.candidates).toEqual([]);
    expect(prediction.next).toBeUndefined();
  });
});

describe("generalization to new-but-related movements", () => {
  it("recombines learned transitions into an unseen sequence", async () => {
    const backend = new MarkovMovementBackend();
    // No single trajectory is [scroll, click, type, submit]. The model must
    // stitch scroll->click (only from B) with click->type->submit (only from A).
    // Two copies of A make click->type outweigh B's terminal click->END so the
    // greedy path is unambiguous.
    const dataset = buildMovementDataset([
      trajectory("a1", ["click", "type", "submit"]),
      trajectory("a2", ["click", "type", "submit"]),
      trajectory("b", ["scroll", "click"]),
    ]);
    const model = await backend.train(dataset, { order: 1 });

    const result = backend.generate(model, [MOVEMENT_START_TOKEN, actionToken("scroll")]);
    expect(result.sequence).toEqual([
      MOVEMENT_START_TOKEN,
      actionToken("scroll"),
      actionToken("click"),
      actionToken("type"),
      actionToken("submit"),
    ]);
    expect(result.stoppedReason).toBe("end");
  });

  it("halts at maxSteps without an END transition", async () => {
    const backend = new MarkovMovementBackend();
    // A pure cycle click->drag->click... never emits END.
    const dataset = buildMovementDataset([trajectory("loop", ["click", "drag", "click", "drag", "click"])]);
    const model = await backend.train(dataset, { order: 1 });
    const result = backend.generate(model, [actionToken("click")], { maxSteps: 5 });
    expect(result.generated).toHaveLength(5);
    expect(result.stoppedReason).toBe("max-steps");
  });

  it("samples reproducibly from a seed", async () => {
    const backend = new MarkovMovementBackend();
    const dataset = buildMovementDataset([
      trajectory("a", ["click", "type"]),
      trajectory("b", ["click", "submit"]),
      trajectory("c", ["click", "drag"]),
    ]);
    const model = await backend.train(dataset, { order: 1 });
    const first = backend.generate(model, [actionToken("click")], { sample: true, seed: 42, maxSteps: 1 });
    const second = backend.generate(model, [actionToken("click")], { sample: true, seed: 42, maxSteps: 1 });
    expect(first.generated).toEqual(second.generated);
    expect(first.generated).toHaveLength(1);
  });
});

describe("eval harness", () => {
  it("scores perfect next-token accuracy on memorized data", async () => {
    const backend = new MarkovMovementBackend();
    const dataset = buildMovementDataset([trajectory("t1", ["click", "type", "submit"])]);
    const model = await backend.train(dataset, { order: 1 });
    const result = evaluateNextTokenAccuracy(backend, model, dataset);
    expect(result.accuracy).toBe(1);
    expect(result.predictedCount).toBe(model.tokenCount - dataset.sequences.length * model.order);
  });

  it("generalizes above chance to a held-out partition", async () => {
    const backend = new MarkovMovementBackend();
    // A family of related "form-fill" trajectories sharing click->type->submit.
    const trajectories = Array.from({ length: 20 }, (_, index) =>
      trajectory(`form-${index}`, index % 2 === 0 ? ["click", "type", "submit"] : ["focus", "click", "type", "submit"]),
    );
    const dataset = buildMovementDataset(trajectories);
    const { train, holdout } = splitMovementDataset(dataset, 0.3);

    expect(train.sequences.length).toBeGreaterThan(0);
    expect(holdout.sequences.length).toBeGreaterThan(0);

    const model = await backend.train(train, { order: 1 });
    const evalResult = evaluateNextTokenAccuracy(backend, model, holdout);
    // Transitions learned on the train split predict the held-out family well.
    expect(evalResult.accuracy).toBeGreaterThan(0.5);
  });

  it("splits deterministically by sequence id", () => {
    const dataset = buildMovementDataset(
      Array.from({ length: 30 }, (_, index) => trajectory(`t-${index}`, ["click", "type"])),
    );
    const a = splitMovementDataset(dataset, 0.25);
    const b = splitMovementDataset(dataset, 0.25);
    expect(a.holdout.sequences.map((s) => s.id)).toEqual(b.holdout.sequences.map((s) => s.id));
  });
});

describe("MovementModelTrainer orchestrator", () => {
  it("wraps a pluggable backend end-to-end", async () => {
    const trainer = new MovementModelTrainer();
    expect(trainer.backendId).toBe("markov");

    const dataset = buildMovementDataset([
      trajectory("a", ["open", "click", "type", "save"]),
      trajectory("b", ["open", "click", "save"]),
    ]);
    const model = await trainer.train(dataset, { order: 1 });
    const generated = trainer.generate(model, [MOVEMENT_START_TOKEN]);
    expect(generated.sequence[0]).toBe(MOVEMENT_START_TOKEN);
    expect(generated.sequence).toContain(actionToken("open"));

    const evalResult = trainer.evaluate(model, dataset);
    expect(evalResult.accuracy).toBeGreaterThan(0);
  });

  it("honors a custom backend implementation", async () => {
    const trainer = new MovementModelTrainer({
      id: "stub",
      train: async () => ({
        version: 1,
        backend: "stub",
        order: 1,
        vocabulary: [],
        transitions: [],
        sequenceCount: 0,
        tokenCount: 0,
      }),
      predict: () => ({ context: [], candidates: [] }),
      generate: (_model, prompt) => ({ prompt, generated: [], sequence: prompt, stoppedReason: "no-transition" }),
    });
    expect(trainer.backendId).toBe("stub");
    const model = await trainer.train(buildMovementDataset([trajectory("t", ["click"])]));
    expect(model.backend).toBe("stub");
  });
});
