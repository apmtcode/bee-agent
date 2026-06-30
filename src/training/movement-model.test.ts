import { describe, expect, it } from "vitest";
import { buildTrajectorySpan } from "../capture/trajectory.js";
import {
  MarkovMovementBackend,
  buildMovementDatasetFromTrajectories,
  defaultSyntheticSkills,
  evaluateMovementModel,
  generateSyntheticMovementExamples,
  isMovementBoundary,
  restoreMovementModel,
  toolMovementTokenizer,
  type MovementExample,
  type MovementToken,
} from "./movement-model.js";

const openApp: MovementToken[] = [
  { tool: "key.press", summary: "cmd+space" },
  { tool: "key.type", summary: "type-app-name" },
  { tool: "key.press", summary: "enter" },
  { tool: "window.focus", summary: "focus-app" },
];

describe("MarkovMovementBackend", () => {
  it("repeats a recorded movement exactly via generate()", async () => {
    const model = await new MarkovMovementBackend().train({
      examples: [{ actions: openApp }],
    });
    const replayed = model.generate([], openApp.length + 2);
    expect(replayed.map((t) => `${t.tool}:${t.summary}`)).toEqual(
      openApp.map((t) => `${t.tool}:${t.summary}`),
    );
  });

  it("predicts the most likely next token first and is deterministic", async () => {
    const backend = new MarkovMovementBackend();
    const dataset = { examples: [{ actions: openApp }, { actions: openApp }] };
    const a = await backend.train(dataset);
    const b = await backend.train(dataset);
    const predA = a.predictNext([openApp[0]!]);
    const predB = b.predictNext([openApp[0]!]);
    expect(predA[0]!.token.summary).toBe("type-app-name");
    expect(predA).toEqual(predB);
    // probabilities form a normalized distribution
    const total = predA.reduce((sum, p) => sum + p.probability, 0);
    expect(total).toBeCloseTo(1, 10);
  });

  it("knows when to stop (predicts the end boundary after the last action)", async () => {
    const model = await new MarkovMovementBackend().train({
      examples: [{ actions: openApp }],
    });
    const pred = model.predictNext(openApp);
    expect(pred[0]!.isEnd).toBe(true);
    expect(isMovementBoundary(pred[0]!.token)).toBe(true);
  });

  it("backs off to a shorter context for unseen prefixes (generalization)", async () => {
    // Two skills share the trailing bigram (.. -> key.press enter); after an
    // unfamiliar leading action the order-2 context is unseen, so the model
    // must back off to the unigram/bigram it does know.
    const model = await new MarkovMovementBackend().train({
      examples: [
        {
          actions: [
            { tool: "key.press", summary: "cmd+s" },
            { tool: "key.press", summary: "enter" },
          ],
        },
      ],
    });
    const pred = model.predictNext([{ tool: "never.seen", summary: "x" }]);
    // unknown context backs off to the unigram distribution, which still ranks
    // the known tokens rather than returning nothing
    expect(pred.length).toBeGreaterThan(0);
  });

  it("generalizes across labels with the tool-only tokenizer", async () => {
    const model = await new MarkovMovementBackend().train(
      {
        examples: [
          {
            actions: [
              { tool: "mouse.move", summary: "to-a" },
              { tool: "mouse.down", summary: "grab-a" },
            ],
          },
        ],
      },
      { tokenizer: toolMovementTokenizer },
    );
    // A new-but-related move (different summary, same channel) is recognized.
    const pred = model.predictNext([{ tool: "mouse.move", summary: "to-DIFFERENT" }]);
    expect(pred[0]!.token.tool).toBe("mouse.down");
  });

  it("round-trips through serialize()/restoreMovementModel()", async () => {
    const model = await new MarkovMovementBackend().train({
      examples: [{ actions: openApp }],
    });
    const restored = restoreMovementModel(model.serialize());
    expect(restored.order).toBe(model.order);
    expect(restored.generate([], 10)).toEqual(model.generate([], 10));
  });
});

describe("dataset helpers", () => {
  it("builds an ordered dataset from trajectory spans", () => {
    const span = buildTrajectorySpan({
      id: "s1",
      sessionId: "sess",
      actions: [
        { kind: "action", tool: "key.press", summary: "enter", ts: 20 },
        { kind: "action", tool: "key.press", summary: "cmd+space", ts: 10 },
      ],
    });
    const dataset = buildMovementDatasetFromTrajectories([span]);
    expect(dataset.examples).toHaveLength(1);
    expect(dataset.examples[0]!.actions.map((a) => a.summary)).toEqual([
      "cmd+space",
      "enter",
    ]);
  });

  it("drops empty trajectories", () => {
    const span = buildTrajectorySpan({ id: "s2", sessionId: "sess", actions: [] });
    expect(buildMovementDatasetFromTrajectories([span]).examples).toHaveLength(0);
  });
});

describe("generateSyntheticMovementExamples", () => {
  it("is reproducible from its seed", () => {
    const a = generateSyntheticMovementExamples({ seed: 42, count: 8, noiseRate: 0.3 });
    const b = generateSyntheticMovementExamples({ seed: 42, count: 8, noiseRate: 0.3 });
    expect(a).toEqual(b);
    expect(a).toHaveLength(8);
  });

  it("varies with the seed", () => {
    const a = generateSyntheticMovementExamples({ seed: 1, count: 8 });
    const b = generateSyntheticMovementExamples({ seed: 2, count: 8 });
    expect(a).not.toEqual(b);
  });
});

describe("evaluateMovementModel (generalization harness)", () => {
  it("scores high on held-out variations of trained skills", async () => {
    const skills = defaultSyntheticSkills();
    const train = generateSyntheticMovementExamples({ seed: 7, count: 120 });
    const heldOut: MovementExample[] = generateSyntheticMovementExamples({
      seed: 9991,
      count: 30,
    });
    // Tool-only tokenizer => model learns the channel grammar of each skill and
    // generalizes to held-out demonstrations with fresh coordinates/labels.
    const model = await new MarkovMovementBackend().train(
      { examples: train },
      { order: 2, tokenizer: toolMovementTokenizer },
    );
    expect(skills.length).toBeGreaterThan(0);
    const result = evaluateMovementModel(model, heldOut, {
      tokenizer: toolMovementTokenizer,
    });
    expect(result.exampleCount).toBe(30);
    // Every interior decision is a deterministic channel transition; only the
    // very first action of each demo is genuinely ambiguous across the three
    // skills, so top-1 is high and top-3 is essentially perfect.
    expect(result.topAccuracy).toBeGreaterThan(0.7);
    expect(result.topKAccuracy).toBeGreaterThan(0.98);
    expect(result.topKAccuracy).toBeGreaterThanOrEqual(result.topAccuracy);
  });

  it("returns zeroed metrics for empty held-out sets", async () => {
    const model = await new MarkovMovementBackend().train({
      examples: [{ actions: openApp }],
    });
    const result = evaluateMovementModel(model, []);
    expect(result.decisionCount).toBe(0);
    expect(result.topAccuracy).toBe(0);
  });
});
