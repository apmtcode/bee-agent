import { describe, expect, it } from "vitest";
import { NgramMovementBackend } from "./ngram-backend.js";
import { buildMovementDataset, type MovementSequence } from "./movement-event.js";
import { createSeededRng } from "./model-backend.js";

function seq(id: string, kinds: Array<[MovementSequence["events"][number]["kind"], string]>): MovementSequence {
  return {
    id,
    events: kinds.map(([kind, target], index) => ({ kind, target, ts: index * 100 })),
  };
}

const backend = new NgramMovementBackend();

describe("NgramMovementBackend", () => {
  it("learns and reproduces a memorized movement sequence via greedy decoding", async () => {
    const sequence = seq("save", [
      ["focus", "editor"],
      ["click", "body"],
      ["key-type", "body"],
      ["shortcut", "cmd+s"],
    ]);
    const model = await backend.train(buildMovementDataset([sequence, sequence, sequence]), { order: 2 });
    const generated = model.generate({ maxLength: 10 });
    expect(generated).toEqual(["focus:editor", "click:body", "key-type:body", "shortcut:cmd+s"]);
    expect(model.metadata.backend).toBe("ngram");
    expect(model.metadata.order).toBe(2);
  });

  it("assigns highest probability to the observed continuation", async () => {
    const model = await backend.train(
      buildMovementDataset([
        seq("a", [["focus", "editor"], ["click", "body"], ["shortcut", "cmd+s"]]),
        seq("b", [["focus", "editor"], ["click", "body"], ["shortcut", "cmd+s"]]),
      ]),
      { order: 2 },
    );
    const distribution = model.predictNext(["focus:editor", "click:body"]);
    expect(distribution[0]?.token).toBe("shortcut:cmd+s");
    const total = distribution.reduce((sum, entry) => sum + entry.probability, 0);
    expect(total).toBeGreaterThan(0.99);
    expect(total).toBeLessThan(1.01);
  });

  it("generalizes to an unseen context via backoff instead of returning nothing", async () => {
    // Train two intents that share the "click:body -> shortcut" transition.
    const model = await backend.train(
      buildMovementDataset([
        seq("a", [["focus", "editor"], ["click", "body"], ["shortcut", "cmd+s"]]),
        seq("b", [["focus", "browser"], ["click", "body"], ["shortcut", "cmd+s"]]),
      ]),
      { order: 2 },
    );
    // This exact 2-token context was never seen, but the 1-token suffix was.
    const distribution = model.predictNext(["focus:mail", "click:body"]);
    expect(distribution.length).toBeGreaterThan(0);
    expect(distribution[0]?.token).toBe("shortcut:cmd+s");
  });

  it("is deterministic under a fixed seed and varies structure by sampling", async () => {
    const model = await backend.train(
      buildMovementDataset([
        seq("a", [["focus", "editor"], ["click", "body"], ["shortcut", "cmd+s"]]),
        seq("b", [["focus", "editor"], ["scroll", "list"], ["click", "item"]]),
      ]),
      { order: 1, smoothing: 0.1 },
    );
    const first = model.generate({ maxLength: 8, rng: createSeededRng(42) });
    const second = model.generate({ maxLength: 8, rng: createSeededRng(42) });
    expect(first).toEqual(second);
  });
});
