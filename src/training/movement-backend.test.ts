import { describe, expect, it } from "vitest";
import { buildReplayManifest } from "../capture/replay.js";
import { buildTrajectorySpan, type TrajectoryAction } from "../capture/trajectory.js";
import {
  MarkovMovementBackend,
  MOVEMENT_END_TOKEN,
  buildMovementDatasetFromReplays,
  buildMovementDatasetFromTrajectories,
  evaluateMovementModel,
  type MovementDataset,
} from "./movement-backend.js";

function action(tool: string, summary: string, ts: number): TrajectoryAction {
  return { kind: "action", tool, summary, ts };
}

describe("MarkovMovementBackend", () => {
  it("reproduces a single recorded movement exactly (replay)", async () => {
    const backend = new MarkovMovementBackend();
    const dataset: MovementDataset = {
      version: 1,
      samples: [{ id: "m1", tokens: ["mouse.move(10,10)", "mouse.click", "key.type('hi')"] }],
    };

    const model = await backend.train(dataset, { order: 2 });
    const replay = backend.generate(model);

    expect(replay).toEqual(["mouse.move(10,10)", "mouse.click", "key.type('hi')"]);
    expect(replay).not.toContain(MOVEMENT_END_TOKEN);
  });

  it("generalizes to a novel-but-related movement via backoff", async () => {
    const backend = new MarkovMovementBackend();
    // Two related movements: both open a menu then choose an item, differing in
    // the opening step. The shared suffix "menu.open -> item.select" is learned.
    const dataset: MovementDataset = {
      version: 1,
      samples: [
        { id: "a", tokens: ["hotkey.cmd+n", "menu.open", "item.select"] },
        { id: "b", tokens: ["button.click", "menu.open", "item.select"] },
      ],
    };

    const model = await backend.train(dataset, { order: 1 });

    // A context ending in "menu.open" was never *started* this way in training as
    // a full-order match, but the order-1 backoff still predicts "item.select".
    const predictions = backend.predict(model, ["some.novel.start", "menu.open"]);
    expect(predictions[0]?.token).toBe("item.select");
  });

  it("weights higher-reward movements more heavily", async () => {
    const backend = new MarkovMovementBackend();
    const dataset: MovementDataset = {
      version: 1,
      samples: [
        { id: "low", tokens: ["start", "safe.path"], reward: 0 },
        { id: "high", tokens: ["start", "reward.path"], reward: 5 },
      ],
    };

    const model = await backend.train(dataset, { order: 1, rewardWeighting: true });
    const predictions = backend.predict(model, ["start"]);

    expect(predictions[0]?.token).toBe("reward.path");
    expect(model.metadata.rewardWeighted).toBe(true);
  });

  it("treats reward weighting as opt-out", async () => {
    const backend = new MarkovMovementBackend();
    const dataset: MovementDataset = {
      version: 1,
      samples: [
        { id: "low", tokens: ["start", "a"], reward: 0 },
        { id: "high", tokens: ["start", "b"], reward: 5 },
      ],
    };

    const model = await backend.train(dataset, { order: 1, rewardWeighting: false });
    const predictions = backend.predict(model, ["start"]);

    // Equal weight => deterministic tie-break by token ascending.
    expect(predictions.map((p) => p.token)).toEqual(["a", "b"]);
    expect(predictions[0]?.probability).toBeCloseTo(0.5);
    expect(model.metadata.rewardWeighted).toBe(false);
  });

  it("produces a JSON-serializable model that round-trips", async () => {
    const backend = new MarkovMovementBackend();
    const dataset: MovementDataset = {
      version: 1,
      samples: [{ id: "m", tokens: ["a", "b", "c"] }],
    };

    const model = await backend.train(dataset);
    const roundTripped = JSON.parse(JSON.stringify(model));

    expect(backend.generate(roundTripped)).toEqual(["a", "b", "c"]);
  });

  it("returns no predictions for an out-of-vocabulary context with no backoff match", async () => {
    const backend = new MarkovMovementBackend();
    const model = await backend.train({ version: 1, samples: [] });
    expect(backend.predict(model, ["anything"])).toEqual([]);
    expect(backend.generate(model)).toEqual([]);
  });

  it("halts generation at maxSteps even with cyclic transitions", async () => {
    const backend = new MarkovMovementBackend();
    // A -> B -> A -> B ... never terminates on its own at order 1.
    const dataset: MovementDataset = { version: 1, samples: [{ id: "cycle", tokens: ["a", "b", "a", "b"] }] };
    const model = await backend.train(dataset, { order: 1 });
    const generated = backend.generate(model, { maxSteps: 5 });
    expect(generated.length).toBeLessThanOrEqual(5);
  });
});

describe("movement dataset builders", () => {
  it("builds a dataset from trajectory actions, sorted by timestamp", () => {
    const trajectory = buildTrajectorySpan({
      id: "t1",
      sessionId: "s1",
      actions: [action("mouse.click", "b", 200), action("mouse.move", "a", 100)],
      outcome: { status: "success", summary: "done", reward: 2 },
    });

    const dataset = buildMovementDatasetFromTrajectories([trajectory]);
    expect(dataset.samples).toHaveLength(1);
    expect(dataset.samples[0]?.tokens).toEqual(["mouse.move::a", "mouse.click::b"]);
    expect(dataset.samples[0]?.reward).toBe(2);
  });

  it("honours approvedOnly and a custom tokenizer", () => {
    const approved = buildTrajectorySpan({ id: "ok", sessionId: "s", actions: [action("t", "x", 1)] });
    approved.review = { status: "approved", reviewedAt: "now", reviewedBy: "me" };
    const pending = buildTrajectorySpan({ id: "no", sessionId: "s", actions: [action("t", "y", 1)] });

    const dataset = buildMovementDatasetFromTrajectories([approved, pending], {
      approvedOnly: true,
      tokenizer: (a) => a.tool,
    });

    expect(dataset.samples).toHaveLength(1);
    expect(dataset.samples[0]?.tokens).toEqual(["t"]);
  });

  it("builds a dataset from replay manifests (action events only)", () => {
    const trajectory = buildTrajectorySpan({
      id: "t",
      sessionId: "s",
      observations: [{ kind: "observation", source: "screen", summary: "ignored", ts: 1 }],
      actions: [action("mouse.click", "here", 2)],
    });
    const manifest = buildReplayManifest({ sessionId: "s", transcript: [], trajectories: [trajectory] });

    const dataset = buildMovementDatasetFromReplays([manifest]);
    expect(dataset.samples[0]?.tokens).toEqual(["mouse.click::here"]);
  });
});

describe("evaluateMovementModel", () => {
  it("scores perfect replay and next-token accuracy on the training set", async () => {
    const backend = new MarkovMovementBackend();
    const dataset: MovementDataset = {
      version: 1,
      samples: [
        { id: "a", tokens: ["x", "y", "z"] },
        { id: "b", tokens: ["x", "y", "z"] },
      ],
    };

    const model = await backend.train(dataset, { order: 2 });
    const evaluation = evaluateMovementModel(backend, model, dataset);

    expect(evaluation.sampleCount).toBe(2);
    expect(evaluation.exactReplayRate).toBe(1);
    expect(evaluation.nextTokenAccuracy).toBe(1);
    expect(evaluation.predictionCount).toBe(4);
  });

  it("reports imperfect fidelity on held-out related movements", async () => {
    const backend = new MarkovMovementBackend();
    const train: MovementDataset = { version: 1, samples: [{ id: "seen", tokens: ["a", "b", "c"] }] };
    const heldOut: MovementDataset = { version: 1, samples: [{ id: "novel", tokens: ["a", "b", "d"] }] };

    const model = await backend.train(train, { order: 2 });
    const evaluation = evaluateMovementModel(backend, model, heldOut);

    // "a->b" is learned (predicts "b" correctly) but "a,b->d" was never seen.
    expect(evaluation.nextTokenAccuracy).toBeGreaterThan(0);
    expect(evaluation.nextTokenAccuracy).toBeLessThan(1);
    expect(evaluation.exactReplayRate).toBe(0);
  });
});
