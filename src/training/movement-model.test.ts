import { describe, expect, it } from "vitest";
import {
  MarkovMovementBackend,
  evaluateMovementModel,
  type MovementDataset,
  type MovementExample,
} from "./movement-model.js";
import {
  buildMovementDataset,
  generateSyntheticTrajectories,
  splitTrajectories,
  tokensToExamples,
} from "./movement-dataset.js";

function dataset(examples: MovementExample[]): MovementDataset {
  const vocabulary = new Set<string>();
  for (const example of examples) {
    vocabulary.add(example.next);
    example.context.forEach((token) => vocabulary.add(token));
  }
  return { version: 1, examples, vocabulary: [...vocabulary].sort() };
}

describe("MarkovMovementBackend", () => {
  it("replays an exactly-recorded movement with full confidence", async () => {
    const backend = new MarkovMovementBackend({ maxOrder: 3 });
    const examples = tokensToExamples(["click:a", "type:b", "click:send"], 3);
    const model = await backend.train(dataset(examples));

    expect(backend.predict(model, ["click:a"]).token).toBe("type:b");
    const send = backend.predict(model, ["click:a", "type:b"]);
    expect(send.token).toBe("click:send");
    expect(send.confidence).toBe(1);
    expect(send.order).toBe(2);
  });

  it("backs off to a shorter suffix for an unseen full context (generalizes)", async () => {
    const backend = new MarkovMovementBackend({ maxOrder: 3 });
    // "type:b" is always followed by "click:send" regardless of what preceded it.
    const model = await backend.train(
      dataset([
        ...tokensToExamples(["click:a", "type:b", "click:send"], 3),
        ...tokensToExamples(["click:x", "type:b", "click:send"], 3),
      ]),
    );

    // Never-seen prefix "click:z" -> "type:b"; full context misses but the
    // order-1 suffix "type:b" still predicts "click:send".
    const prediction = backend.predict(model, ["click:z", "type:b"]);
    expect(prediction.token).toBe("click:send");
    expect(prediction.order).toBe(1);
  });

  it("chooses the most frequent continuation and ranks alternatives", async () => {
    const backend = new MarkovMovementBackend({ maxOrder: 1 });
    const model = await backend.train(
      dataset([
        { context: ["menu"], next: "save" },
        { context: ["menu"], next: "save" },
        { context: ["menu"], next: "quit" },
      ]),
    );
    const prediction = backend.predict(model, ["menu"]);
    expect(prediction.token).toBe("save");
    expect(prediction.confidence).toBeCloseTo(2 / 3);
    expect(prediction.alternatives).toEqual([{ token: "quit", confidence: 1 / 3 }]);
  });

  it("breaks ties deterministically by token order", async () => {
    const backend = new MarkovMovementBackend({ maxOrder: 1 });
    const model = await backend.train(
      dataset([
        { context: ["root"], next: "beta" },
        { context: ["root"], next: "alpha" },
      ]),
    );
    expect(backend.predict(model, ["root"]).token).toBe("alpha");
  });

  it("returns a defined zero-confidence miss for an empty model", async () => {
    const backend = new MarkovMovementBackend();
    const model = await backend.train(dataset([]));
    const prediction = backend.predict(model, ["anything"]);
    expect(prediction).toEqual({ token: "", confidence: 0, order: -1, alternatives: [] });
  });

  it("produces a JSON-serialisable artifact that survives a round-trip", async () => {
    const backend = new MarkovMovementBackend({ maxOrder: 2 });
    const model = await backend.train(dataset(tokensToExamples(["a", "b", "c"], 2)));
    const reloaded = JSON.parse(JSON.stringify(model));
    expect(backend.predict(reloaded, ["a"]).token).toBe("b");
  });
});

describe("evaluateMovementModel on synthetic trajectories", () => {
  it("generalizes to held-out related movements well above chance", async () => {
    const trajectories = generateSyntheticTrajectories({ count: 120, seed: 7 });
    const { train, heldOut } = splitTrajectories(trajectories, 0.25);
    expect(train.length).toBeGreaterThan(0);
    expect(heldOut.length).toBeGreaterThan(0);

    const backend = new MarkovMovementBackend({ maxOrder: 3 });
    const model = await backend.train(buildMovementDataset(train, { order: 3 }));
    const heldOutExamples = buildMovementDataset(heldOut, { order: 3 }).examples;

    const result = evaluateMovementModel(backend, model, heldOutExamples);
    expect(result.total).toBe(heldOutExamples.length);
    // The default grammar has clear structure; back-off prediction should be
    // strong on held-out (never-trained-on) walks — i.e. genuine generalization.
    expect(result.accuracy).toBeGreaterThan(0.7);
    expect(result.meanConfidence).toBeGreaterThan(0.5);
  });

  it("is fully reproducible for a fixed seed", () => {
    // `createdAt` comes from the wall clock (buildTrajectorySpan), so compare the
    // deterministic movement content — the actions — not the whole span.
    const content = (trajectories: ReturnType<typeof generateSyntheticTrajectories>) =>
      JSON.stringify(trajectories.map((trajectory) => trajectory.actions));
    const a = generateSyntheticTrajectories({ count: 10, seed: 42 });
    const b = generateSyntheticTrajectories({ count: 10, seed: 42 });
    expect(content(a)).toBe(content(b));
    const c = generateSyntheticTrajectories({ count: 10, seed: 43 });
    expect(content(a)).not.toBe(content(c));
  });
});
