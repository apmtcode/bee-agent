import { describe, expect, it } from "vitest";
import { buildReplayManifest } from "../capture/replay.js";
import { buildTrajectorySpan } from "../capture/trajectory.js";
import {
  MarkovMovementBackend,
  MOVEMENT_END_TOKEN,
  actionToToken,
  buildMovementDataset,
  type MovementDataset,
} from "./backend.js";

function actionSpan(id: string, tools: string[]) {
  return buildTrajectorySpan({
    id,
    sessionId: `session-${id}`,
    actions: tools.map((tool, index) => ({
      kind: "action" as const,
      tool,
      summary: `${tool} step ${index}`,
      ts: index + 1,
    })),
  });
}

describe("movement model backend", () => {
  it("derives a per-trajectory movement dataset from replay manifests", () => {
    const manifest = buildReplayManifest({
      sessionId: "s1",
      transcript: [],
      trajectories: [actionSpan("t1", ["move", "click"]), actionSpan("t2", ["move", "type"])],
    });
    const dataset = buildMovementDataset([manifest]);
    expect(dataset.sequences).toHaveLength(2);
    expect(dataset.sequences[0]?.trajectoryId).toBe("t1");
    expect(dataset.sequences[0]?.tokens).toEqual([actionToToken("move", "move step 0"), actionToToken("click", "click step 1")]);
  });

  it("slugs summaries so superficially different phrasings collapse to one token", () => {
    expect(actionToToken("click", "Click OK button")).toBe(actionToToken("click", "click   ok!! button"));
  });

  it("trains a deterministic model and replays a recorded sequence verbatim", async () => {
    const backend = new MarkovMovementBackend();
    const dataset: MovementDataset = {
      sequences: [{ trajectoryId: "t1", tokens: ["a", "b", "c", "d"] }],
    };
    const model = await backend.train(dataset, { order: 2 });
    expect(model.backend).toBe("markov-backoff");
    expect(model.trainedSequences).toBe(1);

    // Greedy generation from the start sentinel reproduces the recorded movement.
    const replayed = backend.generate(model, [], { maxLength: 10 });
    expect(replayed).toEqual(["a", "b", "c", "d"]);
    // Two independent trainings produce identical models (determinism).
    const again = await backend.train(dataset, { order: 2 });
    expect(again).toEqual(model);
  });

  it("predicts the most frequent continuation with deterministic tie-breaking", async () => {
    const backend = new MarkovMovementBackend();
    const model = await backend.train(
      {
        sequences: [
          { trajectoryId: "t1", tokens: ["open", "scroll", "click"] },
          { trajectoryId: "t2", tokens: ["open", "scroll", "click"] },
          { trajectoryId: "t3", tokens: ["open", "scroll", "type"] },
        ],
      },
      { order: 2 },
    );
    const prediction = backend.predictNext(model, ["open", "scroll"]);
    expect(prediction?.token).toBe("click");
    expect(prediction?.matchedOrder).toBe(2);
    expect(prediction?.probability).toBeCloseTo(2 / 3);
  });

  it("generalizes to an unseen-but-related prefix via stupid-backoff", async () => {
    const backend = new MarkovMovementBackend();
    // The model has never seen the bigram ["launch","scroll"], but it has seen
    // what reliably follows "scroll". Backoff must still produce that movement.
    const model = await backend.train(
      {
        sequences: [
          { trajectoryId: "t1", tokens: ["open", "scroll", "click"] },
          { trajectoryId: "t2", tokens: ["focus", "scroll", "click"] },
        ],
      },
      { order: 2 },
    );
    const exact = backend.predictNext(model, ["open", "scroll"]);
    expect(exact?.matchedOrder).toBe(2);

    const generalized = backend.predictNext(model, ["launch", "scroll"]);
    expect(generalized?.token).toBe("click");
    // Fell back to the 1-token context — i.e. it generalized rather than matched.
    expect(generalized?.matchedOrder).toBe(1);
  });

  it("falls back to global frequency when no context matches", async () => {
    const backend = new MarkovMovementBackend();
    const model = await backend.train({ sequences: [{ trajectoryId: "t1", tokens: ["a", "a", "a", "b"] }] }, { order: 2 });
    const prediction = backend.predictNext(model, ["totally-unseen-token"]);
    expect(prediction?.matchedOrder).toBe(0);
    expect(prediction?.token).toBe("a"); // most frequent next-token overall
  });

  it("stops generation at the end sentinel and never emits sentinels", async () => {
    const backend = new MarkovMovementBackend();
    const model = await backend.train({ sequences: [{ trajectoryId: "t1", tokens: ["x", "y"] }] }, { order: 2 });
    const generated = backend.generate(model, [], { maxLength: 20 });
    expect(generated).toEqual(["x", "y"]);
    expect(generated).not.toContain(MOVEMENT_END_TOKEN);
  });
});
