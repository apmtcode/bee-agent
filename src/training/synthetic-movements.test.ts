import { describe, expect, it } from "vitest";
import { generateSyntheticTrajectories } from "./synthetic-movements.js";
import { buildMovementDataset } from "./movement-dataset.js";
import { NgramMovementBackend } from "./ngram-backend.js";

describe("generateSyntheticTrajectories", () => {
  it("is deterministic for a given seed", () => {
    const a = generateSyntheticTrajectories({ seed: 42, count: 4 });
    const b = generateSyntheticTrajectories({ seed: 42, count: 4 });
    expect(a.map((t) => t.actions.map((x) => x.summary))).toEqual(b.map((t) => t.actions.map((x) => x.summary)));
  });

  it("varies with the seed", () => {
    const a = generateSyntheticTrajectories({ seed: 1, count: 4 });
    const b = generateSyntheticTrajectories({ seed: 2, count: 4 });
    expect(JSON.stringify(a)).not.toEqual(JSON.stringify(b));
  });

  it("respects the requested count and length bounds", () => {
    const trajectories = generateSyntheticTrajectories({ seed: 3, count: 5, lengthRange: [2, 4] });
    expect(trajectories).toHaveLength(5);
    for (const trajectory of trajectories) {
      expect(trajectory.actions.length).toBeGreaterThanOrEqual(2);
      expect(trajectory.actions.length).toBeLessThanOrEqual(4);
    }
  });

  it("produces trajectories whose movements a model can learn", async () => {
    // Generalization smoke test: train on a batch, then confirm the model can
    // predict a next movement for held-out-but-related trajectories.
    const train = buildMovementDataset(generateSyntheticTrajectories({ seed: 100, count: 8 }));
    const model = await new NgramMovementBackend().train(train, { order: 2 });

    const holdout = buildMovementDataset(generateSyntheticTrajectories({ seed: 100, count: 12 })).sequences.slice(8);
    let predicted = 0;
    let total = 0;
    for (const sequence of holdout) {
      const symbols = sequence.tokens.map((token) => token.symbol);
      for (let i = 1; i < symbols.length; i += 1) {
        total += 1;
        if (model.predict(symbols.slice(0, i))) {
          predicted += 1;
        }
      }
    }
    expect(total).toBeGreaterThan(0);
    // Every in-grammar transition should be predictable via backoff.
    expect(predicted).toBe(total);
  });
});
