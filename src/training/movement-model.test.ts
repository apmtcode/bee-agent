import { describe, expect, it } from "vitest";
import {
  DEFAULT_MOVEMENT_MODEL_ORDER,
  MOVEMENT_SEQUENCE_END,
  NGramMovementModelBackend,
  evaluateNextTokenAccuracy,
  type MovementSequence,
} from "./movement-model.js";

const flow: MovementSequence = {
  id: "flow-1",
  tokens: [
    "obs:os:focus-mail",
    "act:device:tapped-compose",
    "act:device:typed-subject",
    "act:device:typed-body",
    "act:device:tapped-send",
    "obs:os:sent-confirmation",
  ],
};

describe("NGramMovementModelBackend", () => {
  it("reproduces a recorded movement sequence exactly (objective 2c)", async () => {
    const model = await new NGramMovementModelBackend().train({ sequences: [flow] });
    const generated = model.generate([flow.tokens[0]!], 20);
    expect(generated).toEqual(flow.tokens.slice(1));
  });

  it("generalizes to a novel-but-related context via back-off (objective 2d)", async () => {
    // Two flows that share the suffix "... C" but differ in prefix.
    const flow1: MovementSequence = { id: "a", tokens: ["A", "B", "C", "D"] };
    const flow2: MovementSequence = { id: "b", tokens: ["X", "Y", "C", "E"] };
    const model = await new NGramMovementModelBackend().train({ sequences: [flow1, flow2], order: 3 });

    // The full context "Z Y C" was never recorded, but its suffix "Y C" was:
    // the model must back off and still predict E, using fewer context tokens.
    const prediction = model.predictNext(["Z", "Y", "C"]);
    expect(prediction?.token).toBe("E");
    expect(prediction?.matchedOrder).toBe(2);
  });

  it("predicts the terminal token so generation stops", async () => {
    const model = await new NGramMovementModelBackend().train({ sequences: [flow] });
    const prediction = model.predictNext(flow.tokens);
    expect(prediction?.token).toBe(MOVEMENT_SEQUENCE_END);
  });

  it("survives a serialize -> load round-trip with identical predictions", async () => {
    const backend = new NGramMovementModelBackend();
    const model = await backend.train({ sequences: [flow], order: 2 });
    const restored = backend.load(model.serialize());

    const context = flow.tokens.slice(0, 3);
    expect(restored.predictNext(context)).toEqual(model.predictNext(context));
    expect(restored.stats).toEqual(model.stats);
  });

  it("defaults to the documented n-gram order", async () => {
    const model = await new NGramMovementModelBackend().train({ sequences: [flow] });
    expect(model.stats.order).toBe(DEFAULT_MOVEMENT_MODEL_ORDER);
    expect(model.backendId).toBe("ngram-backoff");
  });

  it("returns undefined when untrained", async () => {
    const model = await new NGramMovementModelBackend().train({ sequences: [] });
    expect(model.predictNext(["anything"])).toBeUndefined();
    expect(model.generate([], 5)).toEqual([]);
  });

  it("scores perfect next-token accuracy on a recorded sequence", async () => {
    const model = await new NGramMovementModelBackend().train({ sequences: [flow], order: 3 });
    const result = evaluateNextTokenAccuracy(model, [flow]);
    expect(result.predictions).toBe(flow.tokens.length + 1); // + terminal token
    expect(result.accuracy).toBe(1);
  });

  it("ranks candidates by probability with deterministic tie-breaks", async () => {
    const model = await new NGramMovementModelBackend().train({
      sequences: [
        { id: "a", tokens: ["S", "D"] },
        { id: "b", tokens: ["S", "E"] },
      ],
      order: 1,
    });
    const prediction = model.predictNext(["S"]);
    expect(prediction?.candidates.map((candidate) => candidate.token)).toEqual(["D", "E"]);
    expect(prediction?.token).toBe("D"); // 50/50 tie -> lexicographically smallest
    expect(prediction?.probability).toBeCloseTo(0.5);
  });
});
