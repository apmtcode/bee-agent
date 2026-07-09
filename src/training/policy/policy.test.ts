import { describe, expect, it } from "vitest";
import { buildMovementDataset, tokenizeMovementAction, type MovementDataset } from "./movement-dataset.js";
import { NgramMovementBackend } from "./ngram-backend.js";
import { MOVEMENT_END_TOKEN } from "./backend.js";
import { evaluateMovementPolicy } from "./eval.js";
import { generateSyntheticReplays, type SyntheticMovementTemplate } from "./synthetic.js";

const tok = (tool: string, summary: string) => tokenizeMovementAction(tool, summary);

function datasetFromTemplates(
  templates: SyntheticMovementTemplate[],
  repeatsPerTemplate = 1,
): MovementDataset {
  return buildMovementDataset(generateSyntheticReplays({ templates, repeatsPerTemplate }));
}

const EDIT_TEMPLATE: SyntheticMovementTemplate = {
  name: "edit",
  observation: "code editor focused",
  actions: [
    { tool: "browser", summary: "open file" },
    { tool: "keyboard", summary: "type text" },
    { tool: "keyboard", summary: "save" },
  ],
};

describe("buildMovementDataset", () => {
  it("derives per-trajectory sequences, vocabulary, and observations from replays", () => {
    const dataset = datasetFromTemplates([EDIT_TEMPLATE]);

    expect(dataset.sequences).toHaveLength(1);
    const sequence = dataset.sequences[0]!;
    expect(sequence.steps.map((step) => step.token)).toEqual([
      tok("browser", "open file"),
      tok("keyboard", "type text"),
      tok("keyboard", "save"),
    ]);
    // Every action is annotated with the most recent preceding observation.
    expect(sequence.steps.every((step) => step.observation === "code editor focused")).toBe(true);
    expect(dataset.stepCount).toBe(3);
    expect(dataset.vocabulary).toEqual([...dataset.vocabulary].sort());
    expect(dataset.vocabulary).toHaveLength(3);
  });

  it("normalizes whitespace and case when tokenizing", () => {
    expect(tokenizeMovementAction("  Browser ", "Open   FILE")).toBe(tok("browser", "open file"));
  });

  it("drops trajectories that have no actions", () => {
    const manifests = generateSyntheticReplays({
      templates: [{ name: "empty", observation: "idle", actions: [] }],
    });
    expect(buildMovementDataset(manifests).sequences).toHaveLength(0);
  });
});

describe("NgramMovementBackend", () => {
  it("reproduces a recorded sequence exactly via rollout", () => {
    const dataset = datasetFromTemplates([EDIT_TEMPLATE]);
    const policy = new NgramMovementBackend().train(dataset);

    const first = tok("browser", "open file");
    const generated = policy.rollout({ prefix: [first] });
    expect(generated).toEqual([tok("keyboard", "type text"), tok("keyboard", "save")]);
    expect(policy.metadata.backendId).toBe("ngram-backoff");
    expect(policy.metadata.stepCount).toBe(3);
  });

  it("predicts the next token with full confidence for a deterministic context", () => {
    const dataset = datasetFromTemplates([EDIT_TEMPLATE]);
    const policy = new NgramMovementBackend().train(dataset);

    const prediction = policy.predict({ prefix: [tok("browser", "open file")] });
    expect(prediction.token).toBe(tok("keyboard", "type text"));
    expect(prediction.confidence).toBe(1);
    expect(prediction.source).toBe("context");
    expect(prediction.backoffOrder).toBe(1);
  });

  it("learns an end-of-sequence marker and stops rollout there", () => {
    const dataset = datasetFromTemplates([EDIT_TEMPLATE]);
    const policy = new NgramMovementBackend().train(dataset);

    const afterLast = policy.predict({ prefix: dataset.sequences[0]!.steps.map((step) => step.token) });
    expect(afterLast.token).toBe(MOVEMENT_END_TOKEN);
    // A rollout seeded with the whole sequence generates nothing further.
    expect(policy.rollout({ prefix: dataset.sequences[0]!.steps.map((step) => step.token) })).toEqual([]);
  });

  it("falls back to the unigram distribution when no context matches", () => {
    const dataset = datasetFromTemplates([EDIT_TEMPLATE]);
    const policy = new NgramMovementBackend().train(dataset);

    const prediction = policy.predict({ prefix: [tok("nonexistent", "never seen")] });
    expect(prediction.source).toBe("unigram");
    expect(prediction.backoffOrder).toBe(-1);
    expect(prediction.token).toBeDefined();
  });

  it("returns an empty prediction when trained on no data", () => {
    const policy = new NgramMovementBackend().train({ version: 1, vocabulary: [], sequences: [], stepCount: 0 });
    expect(policy.predict({ prefix: [] })).toEqual({
      token: undefined,
      confidence: 0,
      candidates: [],
      backoffOrder: -1,
      source: "empty",
    });
  });

  it("generalizes to a novel prefix by backing off to a known suffix context", () => {
    // Two trained flows share the suffix "search -> result -> detail".
    const dataset = datasetFromTemplates([
      {
        name: "from-home",
        observation: "home screen",
        actions: [
          { tool: "ui", summary: "home" },
          { tool: "ui", summary: "search" },
          { tool: "ui", summary: "result" },
          { tool: "ui", summary: "detail" },
        ],
      },
      {
        name: "from-menu",
        observation: "menu open",
        actions: [
          { tool: "ui", summary: "menu" },
          { tool: "ui", summary: "search" },
          { tool: "ui", summary: "result" },
          { tool: "ui", summary: "detail" },
        ],
      },
    ]);
    const policy = new NgramMovementBackend().train(dataset);

    // Novel entry point "settings" never seen, but the "search -> result" context is known.
    const prediction = policy.predict({
      prefix: [tok("ui", "settings"), tok("ui", "search"), tok("ui", "result")],
    });
    expect(prediction.token).toBe(tok("ui", "detail"));
    expect(prediction.source).toBe("context");
    // Backed off from the unseen length-3 context to the known length-2 suffix.
    expect(prediction.backoffOrder).toBe(2);
    expect(prediction.confidence).toBe(1);
  });

  it("orders candidates deterministically by count then lexically", () => {
    // "a" is followed twice by "z" and once by "b": z outranks b by count.
    const dataset: MovementDataset = {
      version: 1,
      vocabulary: [tok("t", "a"), tok("t", "b"), tok("t", "z")].sort(),
      sequences: [
        { trajectoryId: "s1", sessionId: "x", steps: step(["a", "z"]) },
        { trajectoryId: "s2", sessionId: "x", steps: step(["a", "z"]) },
        { trajectoryId: "s3", sessionId: "x", steps: step(["a", "b"]) },
      ],
      stepCount: 6,
    };
    const policy = new NgramMovementBackend().train(dataset);
    const prediction = policy.predict({ prefix: [tok("t", "a")] });
    expect(prediction.candidates.map((c) => c.token)).toEqual([tok("t", "z"), tok("t", "b")]);
    expect(prediction.token).toBe(tok("t", "z"));
    expect(prediction.confidence).toBeCloseTo(2 / 3, 10);
  });
});

describe("evaluateMovementPolicy", () => {
  it("scores a perfect reproduction on the training sequence", () => {
    const dataset = datasetFromTemplates([EDIT_TEMPLATE]);
    const policy = new NgramMovementBackend().train(dataset);

    const result = evaluateMovementPolicy(policy, dataset.sequences);
    expect(result.sequenceCount).toBe(1);
    expect(result.nextTokenAccuracy).toBe(1);
    expect(result.rolloutExactMatch).toBe(1);
    expect(result.rolloutStepFidelity).toBe(1);
    expect(result.averageConfidence).toBe(1);
  });

  it("measures partial generalization on a held-out related sequence", () => {
    const trainDataset = datasetFromTemplates([
      {
        name: "flow",
        observation: "app open",
        actions: [
          { tool: "ui", summary: "search" },
          { tool: "ui", summary: "result" },
          { tool: "ui", summary: "detail" },
        ],
      },
    ]);
    const policy = new NgramMovementBackend().train(trainDataset);

    const heldOut = datasetFromTemplates([
      {
        name: "held",
        observation: "app open",
        actions: [
          { tool: "ui", summary: "settings" }, // novel first step
          { tool: "ui", summary: "search" },
          { tool: "ui", summary: "result" },
          { tool: "ui", summary: "detail" },
        ],
      },
    ]).sequences;

    const result = evaluateMovementPolicy(policy, heldOut);
    // The two positions with a known suffix context are recovered → strictly partial.
    expect(result.nextTokenAccuracy).toBeGreaterThan(0);
    expect(result.nextTokenAccuracy).toBeLessThan(1);
  });
});

function step(summaries: string[]) {
  return summaries.map((summary, index) => ({
    token: tok("t", summary),
    tool: "t",
    summary,
    ts: index,
  }));
}
