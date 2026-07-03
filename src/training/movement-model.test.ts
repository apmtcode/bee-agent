import { describe, expect, it } from "vitest";
import {
  MarkovMovementBackend,
  MOVEMENT_END_TOKEN,
  buildMovementDataset,
  createMovementModelBackend,
  createSeededRng,
  defaultActionTokenizer,
  deserializeMovementModel,
  evaluateMovementModel,
  generateSyntheticMovementSequences,
  listMovementBackends,
  movementSourcesFromReplays,
  movementSourcesFromTrajectories,
  registerMovementBackend,
  type MovementModelBackend,
  type TrainedMovementModel,
} from "./movement-model.js";

const openAppFlow = {
  stages: [
    [
      { tool: "device", summary: "tapped notes", metadata: { gesture: "tap", target: "notes" } },
      { tool: "device", summary: "tapped mail", metadata: { gesture: "tap", target: "mail" } },
    ],
    [{ tool: "device", summary: "focused editor", metadata: { gesture: "tap", target: "editor" } }],
    [
      { tool: "device", summary: "typed hello", metadata: { gesture: "type", target: "editor" } },
      { tool: "device", summary: "typed draft", metadata: { gesture: "type", target: "editor" } },
    ],
    [{ tool: "device", summary: "swiped up", metadata: { gesture: "swipe", direction: "up" } }],
  ],
};

describe("defaultActionTokenizer", () => {
  it("builds structured tokens from gesture metadata", () => {
    expect(
      defaultActionTokenizer({ tool: "device", summary: "tapped notes", metadata: { gesture: "tap", target: "Notes App" } }),
    ).toBe("device:tap:notes-app");
  });

  it("includes direction when present", () => {
    expect(
      defaultActionTokenizer({ tool: "device", summary: "swiped up", metadata: { gesture: "swipe", direction: "up" } }),
    ).toBe("device:swipe:up");
  });

  it("falls back to tool:summary when no gesture metadata", () => {
    expect(defaultActionTokenizer({ tool: "Editor", summary: "Save File" })).toBe("editor:save-file");
  });
});

describe("buildMovementDataset", () => {
  it("tokenizes sources, appends an end token, and collects a sorted vocabulary", () => {
    const dataset = buildMovementDataset({
      sources: [
        {
          id: "t1",
          events: [
            { tool: "device", summary: "tapped notes", metadata: { gesture: "tap", target: "notes" } },
            { tool: "device", summary: "swiped up", metadata: { gesture: "swipe", direction: "up" } },
          ],
        },
      ],
    });

    expect(dataset.sequences).toHaveLength(1);
    expect(dataset.sequences[0].tokens).toEqual(["device:tap:notes", "device:swipe:up", MOVEMENT_END_TOKEN]);
    expect(dataset.vocabulary).toEqual([...dataset.vocabulary].sort());
    expect(dataset.vocabulary).toContain(MOVEMENT_END_TOKEN);
  });

  it("drops empty sources", () => {
    const dataset = buildMovementDataset({ sources: [{ id: "empty", events: [] }] });
    expect(dataset.sequences).toHaveLength(0);
  });

  it("adapts trajectory spans and replay manifests into sources", () => {
    const trajSources = movementSourcesFromTrajectories([
      { id: "traj", actions: [{ tool: "device", summary: "tapped notes", metadata: { gesture: "tap" } }] },
    ]);
    expect(trajSources[0].events[0].tool).toBe("device");

    const replaySources = movementSourcesFromReplays([
      {
        sessionId: "s1",
        events: [
          { kind: "observation", summary: "looked" },
          { kind: "action", tool: "device", summary: "tapped", metadata: { gesture: "tap" } },
        ],
      },
    ]);
    expect(replaySources[0].events).toHaveLength(1);
    expect(replaySources[0].events[0].summary).toBe("tapped");
  });
});

describe("MarkovMovementBackend — repeat recorded movements", () => {
  it("reproduces a recorded sequence via deterministic argmax generation", () => {
    const dataset = buildMovementDataset({
      sources: [
        {
          id: "recorded",
          events: [
            { tool: "device", summary: "tapped notes", metadata: { gesture: "tap", target: "notes" } },
            { tool: "device", summary: "focused editor", metadata: { gesture: "tap", target: "editor" } },
            { tool: "device", summary: "typed hello", metadata: { gesture: "type", target: "editor" } },
            { tool: "device", summary: "swiped up", metadata: { gesture: "swipe", direction: "up" } },
          ],
        },
      ],
    });
    const model = new MarkovMovementBackend().train(dataset, { order: 2 });
    const recorded = dataset.sequences[0].tokens.slice(0, -1); // drop end token

    const replayed = model.generate([recorded[0]], { maxLength: 10 });
    expect([recorded[0], ...replayed]).toEqual(recorded);
  });

  it("stops generation at the end token", () => {
    const dataset = buildMovementDataset({
      sources: [{ id: "r", events: [{ tool: "a", summary: "one" }, { tool: "a", summary: "two" }] }],
    });
    const model = new MarkovMovementBackend().train(dataset, { order: 2 });
    const generated = model.generate(["a:one"], { maxLength: 10 });
    expect(generated).not.toContain(MOVEMENT_END_TOKEN);
    expect(generated).toEqual(["a:two"]);
  });
});

describe("MarkovMovementBackend — generalize to new-but-related movements", () => {
  it("predicts a plausible continuation for an unseen context via backoff", () => {
    // Train on several related flows: after focusing the editor, the user always types.
    const sources = generateSyntheticMovementSequences({ flow: openAppFlow, count: 40, seed: 7 });
    const dataset = buildMovementDataset({ sources });
    const model = new MarkovMovementBackend().train(dataset, { order: 3 });

    // A novel prefix (tapping a never-seen "calendar" app) that nonetheless ends
    // in a familiar token: the full context was never observed, but backoff to
    // the known suffix "focused editor" still predicts the learned "then type"
    // transition — that is generalization to a new-but-related movement.
    const predictions = model.predictNext(["device:tap:calendar", "device:tap:editor"]);
    expect(predictions.length).toBeGreaterThan(0);
    expect(predictions[0].token).toBe("device:type:editor");
  });

  it("scores held-out related sequences better than unrelated ones", () => {
    const train = generateSyntheticMovementSequences({ flow: openAppFlow, count: 60, seed: 1, idPrefix: "train" });
    const dataset = buildMovementDataset({ sources: train });
    const model = new MarkovMovementBackend().train(dataset, { order: 3, smoothing: 0.01 });

    const heldOut = buildMovementDataset({
      sources: generateSyntheticMovementSequences({ flow: openAppFlow, count: 20, seed: 999, idPrefix: "eval" }),
    }).sequences;
    const unrelated = buildMovementDataset({
      sources: [{ id: "u", events: [{ tool: "shell", summary: "ran ls" }, { tool: "shell", summary: "ran git" }] }],
    }).sequences;

    const relatedEval = evaluateMovementModel(model, heldOut);
    const unrelatedEval = evaluateMovementModel(model, unrelated);

    expect(relatedEval.tokenCount).toBeGreaterThan(0);
    expect(relatedEval.topOneAccuracy).toBeGreaterThan(0.5);
    expect(relatedEval.perplexity).toBeLessThan(unrelatedEval.perplexity);
  });
});

describe("perplexity and likelihood", () => {
  it("assigns finite, comparable perplexity and empty-sequence baseline", () => {
    const dataset = buildMovementDataset({
      sources: [{ id: "r", events: [{ tool: "a", summary: "one" }, { tool: "a", summary: "two" }] }],
    });
    const model = new MarkovMovementBackend().train(dataset, { order: 2, smoothing: 0.5 });
    expect(model.perplexity([])).toBe(1);
    expect(Number.isFinite(model.perplexity(["a:one", "a:two"]))).toBe(true);
    expect(model.sequenceLogLikelihood(["a:one"])).toBeLessThanOrEqual(0);
  });
});

describe("serialization round-trip", () => {
  it("preserves predictions after serialize/deserialize", () => {
    const dataset = buildMovementDataset({
      sources: generateSyntheticMovementSequences({ flow: openAppFlow, count: 30, seed: 3 }),
    });
    const model = new MarkovMovementBackend().train(dataset, { order: 2, smoothing: 0.01 });
    const restored = deserializeMovementModel(model.serialize());

    const context = ["device:tap:notes"];
    expect(restored.predictNext(context)).toEqual(model.predictNext(context));
    expect(restored.order).toBe(model.order);
    expect(restored.vocabulary).toEqual(model.vocabulary);
  });
});

describe("backend registry", () => {
  it("resolves the default markov backend and lists registered backends", () => {
    expect(listMovementBackends()).toContain("markov");
    expect(createMovementModelBackend().name).toBe("markov");
    expect(createMovementModelBackend("markov")).toBeInstanceOf(MarkovMovementBackend);
  });

  it("throws for an unknown backend", () => {
    expect(() => createMovementModelBackend("nonexistent")).toThrow(/Unknown movement-model backend/);
  });

  it("allows registering a custom (real on-device seam) backend", () => {
    const custom: MovementModelBackend = {
      name: "custom-test-backend",
      train(): TrainedMovementModel {
        return new MarkovMovementBackend().train(
          buildMovementDataset({ sources: [{ id: "x", events: [{ tool: "a", summary: "b" }] }] }),
        );
      },
    };
    registerMovementBackend(custom);
    expect(listMovementBackends()).toContain("custom-test-backend");
    expect(createMovementModelBackend("custom-test-backend").name).toBe("custom-test-backend");
  });
});

describe("sample generation determinism", () => {
  it("produces reproducible output for a fixed seed", () => {
    const dataset = buildMovementDataset({
      sources: generateSyntheticMovementSequences({ flow: openAppFlow, count: 20, seed: 5 }),
    });
    const model = new MarkovMovementBackend().train(dataset, { order: 2, smoothing: 0.1 });
    const first = model.generate(["device:tap:notes"], { maxLength: 6, strategy: "sample", rng: createSeededRng(42) });
    const second = model.generate(["device:tap:notes"], { maxLength: 6, strategy: "sample", rng: createSeededRng(42) });
    expect(first).toEqual(second);
  });
});
