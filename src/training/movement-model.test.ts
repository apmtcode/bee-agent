import { describe, expect, it } from "vitest";
import type { TrajectorySpan } from "../capture/trajectory.js";
import {
  MarkovMovementBackend,
  MOVEMENT_END,
  buildMovementDataset,
  createSeededRng,
  evaluateGeneralization,
  generateMovements,
  repeatMovements,
  splitMovementDataset,
  synthesizeMovementDataset,
  tokenizeAction,
  tokenizeTrajectory,
  type MovementTemplate,
} from "./movement-model.js";

function span(id: string, actions: Array<{ tool: string; summary: string; ts: number; metadata?: Record<string, unknown> }>): TrajectorySpan {
  return {
    id,
    sessionId: `session-${id}`,
    createdAt: "2026-07-14T00:00:00.000Z",
    captureTier: "full",
    observations: [],
    actions: actions.map((action) => ({ kind: "action", ...action })),
  };
}

describe("movement tokenizer", () => {
  it("prefers structured gesture metadata over the summary", () => {
    expect(
      tokenizeAction({ kind: "action", tool: "device", summary: "tapped Submit", ts: 1, metadata: { gesture: "tap", target: "Submit Button" } }),
    ).toBe("tap:submit-button");
    expect(
      tokenizeAction({ kind: "action", tool: "device", summary: "swiped down", ts: 1, metadata: { gesture: "swipe", direction: "down" } }),
    ).toBe("swipe:down");
  });

  it("falls back to tool + summary slug when no gesture metadata", () => {
    expect(tokenizeAction({ kind: "action", tool: "browser", summary: "Click #login", ts: 1 })).toBe("browser:click-login");
  });

  it("orders trajectory actions by timestamp", () => {
    const sequence = tokenizeTrajectory(
      span("t1", [
        { tool: "device", summary: "b", ts: 20, metadata: { gesture: "swipe", direction: "up" } },
        { tool: "device", summary: "a", ts: 10, metadata: { gesture: "tap", target: "menu" } },
      ]),
    );
    expect(sequence.tokens).toEqual(["tap:menu", "swipe:up"]);
  });
});

describe("MarkovMovementBackend", () => {
  const motif = ["tap:menu", "swipe:down", "tap:item", "type:query", "tap:submit"];

  it("repeats a single recorded movement sequence exactly (objective 2c)", async () => {
    const backend = new MarkovMovementBackend({ order: 2 });
    const artifact = await backend.train({ version: 1, sequences: [{ id: "s0", tokens: motif }] });
    const model = backend.load(artifact);
    expect(repeatMovements(model)).toEqual(motif);
  });

  it("generalizes: same seed is reproducible, and output stays in-vocabulary (objective 2d)", async () => {
    const backend = new MarkovMovementBackend({ order: 1 });
    const dataset = synthesizeMovementDataset({
      templates: [
        {
          name: "search",
          motif,
          variants: { 1: ["swipe:up", "scroll:down"], 3: ["type:name", "type:email"] },
        },
      ],
      perTemplate: 12,
      seed: 7,
    });
    const artifact = await backend.train(dataset);
    const model = backend.load(artifact);

    const first = generateMovements(model, { rng: createSeededRng(99), maxLength: 20 });
    const second = generateMovements(model, { rng: createSeededRng(99), maxLength: 20 });
    expect(first).toEqual(second); // deterministic given a seed
    expect(first.length).toBeGreaterThan(0);
    const vocab = new Set(artifact.vocabulary);
    for (const token of first) {
      expect(vocab.has(token)).toBe(true);
    }
  });

  it("assigns higher likelihood to trained motifs than to scrambled ones", async () => {
    const backend = new MarkovMovementBackend({ order: 2 });
    const artifact = await backend.train({
      version: 1,
      sequences: Array.from({ length: 6 }, (_, i) => ({ id: `s${i}`, tokens: motif })),
    });
    const model = backend.load(artifact);
    const scrambled = ["type:query", "tap:menu", "tap:submit", "swipe:down"];
    expect(model.logLikelihood(motif)).toBeGreaterThan(model.logLikelihood(scrambled));
    expect(model.perplexity(motif)).toBeLessThan(model.perplexity(scrambled));
  });

  it("keeps unseen movements reachable via smoothing", async () => {
    const backend = new MarkovMovementBackend({ order: 1, smoothing: 0.1 });
    const artifact = await backend.train({ version: 1, sequences: [{ id: "s0", tokens: motif }] });
    const model = backend.load(artifact);
    const prediction = model.predict(["tap:menu"]);
    expect(prediction.probability).toBeGreaterThan(0);
    // every vocab token retains non-zero probability
    const dist = [prediction, ...prediction.alternatives.map((a) => ({ ...a, alternatives: [] }))];
    for (const entry of dist) {
      expect(entry.probability).toBeGreaterThan(0);
    }
  });

  it("rejects non-positive smoothing", () => {
    expect(() => new MarkovMovementBackend({ smoothing: 0 })).toThrow(/smoothing/);
  });
});

describe("synthetic generator + dataset build", () => {
  it("is deterministic for a fixed seed and varied across steps", () => {
    const templates: MovementTemplate[] = [
      { name: "compose", motif: ["tap:new", "type:body", "tap:send"], variants: { 1: ["type:body", "type:draft"] } },
    ];
    const a = synthesizeMovementDataset({ templates, perTemplate: 5, seed: 42 });
    const b = synthesizeMovementDataset({ templates, perTemplate: 5, seed: 42 });
    expect(a).toEqual(b);
    expect(a.sequences).toHaveLength(5);
    // at least the motif's fixed steps are always present
    for (const sequence of a.sequences) {
      expect(sequence.tokens[0]).toBe("tap:new");
      expect(sequence.tokens[2]).toBe("tap:send");
    }
  });

  it("builds a dataset from recorded trajectories, dropping empty ones", () => {
    const dataset = buildMovementDataset([
      span("a", [{ tool: "device", summary: "x", ts: 1, metadata: { gesture: "tap", target: "ok" } }]),
      span("b", []),
    ]);
    expect(dataset.sequences).toHaveLength(1);
    expect(dataset.sequences[0]!.tokens).toEqual(["tap:ok"]);
  });
});

describe("evaluateGeneralization", () => {
  it("reports high top-1 accuracy on held-out sequences drawn from the same distribution", async () => {
    const templates: MovementTemplate[] = [
      { name: "search", motif: ["tap:menu", "swipe:down", "tap:item", "tap:submit"] },
      { name: "compose", motif: ["tap:new", "type:body", "tap:send"] },
    ];
    const dataset = synthesizeMovementDataset({ templates, perTemplate: 8, seed: 3, variantRate: 0 });
    const { train, holdout } = splitMovementDataset(dataset, 4);
    expect(holdout.length).toBeGreaterThan(0);

    const backend = new MarkovMovementBackend({ order: 2 });
    const model = backend.load(await backend.train({ version: 1, sequences: train }));

    const result = evaluateGeneralization(model, holdout);
    expect(result.sequenceCount).toBe(holdout.length);
    // With zero variance the only errors are the genuinely-ambiguous first
    // token (two templates start differently), so accuracy is high but < 1.
    expect(result.top1Accuracy).toBeGreaterThan(0.8);
    expect(result.meanPerplexity).toBeLessThan(3);
  });

  it("handles an empty held-out set without dividing by zero", () => {
    const backend = new MarkovMovementBackend();
    // build a tiny model so predict() has a vocabulary
    return backend.train({ version: 1, sequences: [{ id: "s", tokens: ["tap:a", "tap:b"] }] }).then((artifact) => {
      const model = backend.load(artifact);
      const result = evaluateGeneralization(model, []);
      expect(result.top1Accuracy).toBe(0);
      expect(result.meanPerplexity).toBe(0);
    });
  });
});

describe("repeat/generate termination", () => {
  it("terminates at END and respects maxLength", async () => {
    const backend = new MarkovMovementBackend({ order: 1 });
    const model = backend.load(await backend.train({ version: 1, sequences: [{ id: "s", tokens: ["a", "b", "c"] }] }));
    expect(repeatMovements(model)).toEqual(["a", "b", "c"]);
    const capped = generateMovements(model, { rng: createSeededRng(1), maxLength: 2 });
    expect(capped.length).toBeLessThanOrEqual(2);
    expect(capped).not.toContain(MOVEMENT_END);
  });
});
