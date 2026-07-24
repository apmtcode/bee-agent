import { describe, expect, it } from "vitest";
import {
  DeterministicMarkovBackend,
  buildMovementDataset,
  tokenizeReplayEvents,
  type MovementDataset,
  type ReplayLikeManifest,
} from "./model-backend.js";

function seq(...tokens: string[]) {
  return { tokens };
}

describe("tokenizeReplayEvents", () => {
  it("maps action and observation events to stable tokens and drops transcript", () => {
    const tokens = tokenizeReplayEvents([
      { kind: "transcript", role: "user" },
      { kind: "observation", source: "window" },
      { kind: "action", tool: "click" },
      { kind: "transcript", role: "assistant" },
      { kind: "action", tool: "type" },
    ]);
    expect(tokens).toEqual(["obs:window", "act:click", "act:type"]);
  });
});

describe("buildMovementDataset", () => {
  it("builds samples from replay manifests and drops empty sequences", () => {
    const replays: ReplayLikeManifest[] = [
      {
        sessionId: "s1",
        trajectoryIds: ["t1"],
        events: [
          { kind: "observation", source: "window" },
          { kind: "action", tool: "click" },
        ],
      },
      // transcript-only replay tokenizes to nothing → dropped
      { sessionId: "s2", events: [{ kind: "transcript", role: "user" }] },
    ];
    const dataset = buildMovementDataset(replays);
    expect(dataset.samples).toHaveLength(1);
    expect(dataset.samples[0]).toMatchObject({
      sourceSessionId: "s1",
      trajectoryIds: ["t1"],
      tokens: ["obs:window", "act:click"],
    });
  });
});

describe("DeterministicMarkovBackend", () => {
  const dataset: MovementDataset = {
    samples: [
      seq("obs:app", "act:focus", "act:click", "act:type", "act:submit"),
      seq("obs:app", "act:focus", "act:click", "act:type", "act:submit"),
    ],
  };

  it("repeats a recorded movement sequence verbatim via full-order match", async () => {
    const model = await new DeterministicMarkovBackend(2).train(dataset);
    const continuation = model.generate(["obs:app", "act:focus"], { maxSteps: 3 });
    expect(continuation).toEqual(["act:click", "act:type", "act:submit"]);

    const prediction = model.predictNext(["obs:app", "act:focus"]);
    expect(prediction?.token).toBe("act:click");
    expect(prediction?.contextOrder).toBe(2);
    expect(prediction?.probability).toBe(1);
  });

  it("generalizes to an unseen-but-related prefix by backing off to shorter context", async () => {
    // "act:focus" appears in training followed by "act:click"; here the leading
    // observation differs (never seen), so the order-2 context misses and the
    // model backs off to the order-1 context to still predict the next move.
    const model = await new DeterministicMarkovBackend(2).train(dataset);
    const prediction = model.predictNext(["obs:UNSEEN", "act:focus"]);
    expect(prediction?.token).toBe("act:click");
    expect(prediction?.contextOrder).toBe(1);
  });

  it("is deterministic across retrains and tie-breaks reproducibly", async () => {
    const tieDataset: MovementDataset = {
      samples: [seq("act:a", "act:x"), seq("act:a", "act:y")],
    };
    const modelA = await new DeterministicMarkovBackend(1).train(tieDataset);
    const modelB = await new DeterministicMarkovBackend(1).train(tieDataset);
    // Equal counts → lexicographically smallest token wins, every time.
    expect(modelA.predictNext(["act:a"])?.token).toBe("act:x");
    expect(modelB.predictNext(["act:a"])?.token).toBe("act:x");
    expect(modelA.toJSON()).toEqual(modelB.toJSON());
  });

  it("returns undefined when there is nothing to predict from", async () => {
    const model = await new DeterministicMarkovBackend(2).train({ samples: [] });
    expect(model.predictNext(["anything"])).toBeUndefined();
    expect(model.generate(["anything"], { maxSteps: 5 })).toEqual([]);
  });

  it("serializes to a stable, sorted JSON snapshot", async () => {
    const model = await new DeterministicMarkovBackend(1).train({
      samples: [seq("act:a", "act:b")],
    });
    const json = model.toJSON();
    expect(json.backendId).toBe("deterministic-markov");
    expect(json.order).toBe(1);
    expect(json.sampleCount).toBe(1);
    expect(json.vocabulary).toEqual(["act:a", "act:b"]);
    expect(json.transitions["act:a"]).toEqual({ "act:b": 1 });
    expect(json.transitions[""]).toEqual({ "act:b": 1 });
  });

  it("rejects invalid orders", () => {
    expect(() => new DeterministicMarkovBackend(0)).toThrow(/positive integer/);
  });
});
