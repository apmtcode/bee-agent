import { describe, expect, it } from "vitest";
import type { ReviewedExportManifest } from "./export-manifest.js";
import {
  MarkovMovementBackend,
  datasetFromReviewedExport,
  deserializeMovementModel,
  evaluateGeneralization,
  movementTokenForAction,
  movementTokenForObservation,
  synthesizeMovementSequences,
  type MovementSequence,
  type MovementStep,
} from "./movement-model.js";

function step(token: string, ts: number, label = token): MovementStep {
  return { kind: token.startsWith("action:") ? "action" : "observation", token, label, ts };
}

function sequence(id: string, tokens: string[], gap = 100): MovementSequence {
  return { id, steps: tokens.map((token, index) => step(token, index * gap)) };
}

describe("movement tokenization", () => {
  it("canonicalizes tools and sources into stable tokens", () => {
    expect(movementTokenForAction("Click Button")).toBe("action:click-button");
    expect(movementTokenForObservation("Screen/UI")).toBe("observation:screen-ui");
    expect(movementTokenForAction("  ")).toBe("action:unknown");
  });
});

describe("datasetFromReviewedExport", () => {
  it("keeps only observation/action events in timeline order and drops empty replays", () => {
    const manifest = {
      replays: [
        {
          sessionId: "s1",
          trajectoryIds: ["t1"],
          eventCount: 3,
          events: [
            { kind: "transcript", ts: 0, messageId: "m1", role: "user", content: "hi" },
            { kind: "observation", ts: 1, trajectoryId: "t1", source: "screen", summary: "saw window" },
            { kind: "action", ts: 2, trajectoryId: "t1", tool: "click", summary: "clicked ok" },
          ],
        },
        { sessionId: "s2", trajectoryIds: ["t2"], eventCount: 1, events: [
          { kind: "transcript", ts: 0, messageId: "m2", role: "user", content: "only talk" },
        ] },
      ],
    } as unknown as ReviewedExportManifest;

    const dataset = datasetFromReviewedExport(manifest);
    expect(dataset.sequences).toHaveLength(1);
    expect(dataset.sequences[0]!.steps.map((s) => s.token)).toEqual([
      "observation:screen",
      "action:click",
    ]);
  });
});

describe("MarkovMovementBackend", () => {
  it("repeats a recorded movement: deterministic single-path rollout", async () => {
    const dataset = { sequences: [sequence("a", ["observation:screen", "action:focus", "action:click", "action:submit"])] };
    const model = await new MarkovMovementBackend().train(dataset, { order: 2 });

    const prediction = model.predictNext([step("observation:screen", 0), step("action:focus", 100)]);
    expect(prediction.top).toBe("action:click");
    expect(prediction.candidates[0]!.probability).toBe(1);

    const rollout = model.generate([step("observation:screen", 0)], { maxSteps: 3 });
    expect(rollout.map((s) => s.token)).toEqual(["action:focus", "action:click", "action:submit"]);
    // Timing model advances ts by the learned per-token gap (100ms).
    expect(rollout[0]!.ts).toBe(100);
  });

  it("generalizes: composes learned transitions into an unseen-but-related path", async () => {
    // Neither training sequence contains focus->click->submit, but transitions do.
    const dataset = {
      sequences: [
        sequence("a", ["action:open", "action:focus", "action:click"]),
        sequence("b", ["action:focus", "action:click", "action:submit"]),
      ],
    };
    const model = await new MarkovMovementBackend().train(dataset, { order: 1 });
    const rollout = model.generate([step("action:open", 0)], { maxSteps: 4 });
    expect(rollout.map((s) => s.token)).toEqual([
      "action:focus",
      "action:click",
      "action:submit",
    ]);
  });

  it("backs off to a shorter context when the full n-gram is unseen", async () => {
    const dataset = { sequences: [sequence("a", ["action:a", "action:b", "action:c"])] };
    const model = await new MarkovMovementBackend().train(dataset, { order: 2 });
    // Context [x, b] was never seen at order 2; backoff to [b] finds c.
    const prediction = model.predictNext([step("action:x", 0), step("action:b", 100)]);
    expect(prediction.top).toBe("action:c");
    expect(prediction.contextOrderUsed).toBe(1);
  });

  it("returns no candidates for a fully unknown context", async () => {
    const model = await new MarkovMovementBackend().train({
      sequences: [sequence("a", ["action:a", "action:b"])],
    });
    const prediction = model.predictNext([step("action:zzz", 0)]);
    expect(prediction.candidates).toEqual([]);
    expect(prediction.contextOrderUsed).toBe(-1);
    expect(model.generate([step("action:zzz", 0)])).toEqual([]);
  });

  it("ranks branching candidates by probability", async () => {
    const dataset = {
      sequences: [
        sequence("a", ["action:menu", "action:save"]),
        sequence("b", ["action:menu", "action:save"]),
        sequence("c", ["action:menu", "action:quit"]),
      ],
    };
    const model = await new MarkovMovementBackend().train(dataset, { order: 1 });
    const prediction = model.predictNext([step("action:menu", 0)]);
    expect(prediction.candidates).toEqual([
      { token: "action:save", probability: 2 / 3 },
      { token: "action:quit", probability: 1 / 3 },
    ]);
  });

  it("is deterministic across identical training runs and seeds", async () => {
    const dataset = {
      sequences: [
        sequence("a", ["action:menu", "action:save", "action:close"]),
        sequence("b", ["action:menu", "action:quit"]),
      ],
    };
    const backend = new MarkovMovementBackend();
    const first = (await backend.train(dataset)).generate([step("action:menu", 0)], { seed: 42, maxSteps: 3 });
    const second = (await backend.train(dataset)).generate([step("action:menu", 0)], { seed: 42, maxSteps: 3 });
    expect(first).toEqual(second);
  });
});

describe("model persistence", () => {
  it("serializes and reloads to an identical model for inference", async () => {
    const dataset = {
      sequences: [sequence("a", ["observation:screen", "action:focus", "action:click", "action:submit"])],
    };
    const model = await new MarkovMovementBackend().train(dataset, { order: 2 });
    const restored = deserializeMovementModel(model.serialize());

    const context = [step("observation:screen", 0), step("action:focus", 100)];
    expect(restored.predictNext(context)).toEqual(model.predictNext(context));
    expect(restored.generate([step("observation:screen", 0)], { maxSteps: 3 })).toEqual(
      model.generate([step("observation:screen", 0)], { maxSteps: 3 }),
    );
    expect(restored.vocabulary).toEqual(model.vocabulary);
  });
});

describe("synthesizeMovementSequences + generalization eval", () => {
  it("produces a deterministic dataset for the same seed", () => {
    const grammar = [
      ["observation:screen", "action:focus", "action:click"],
      ["observation:screen", "action:focus", "action:type"],
    ];
    const a = synthesizeMovementSequences({ count: 5, grammar, seed: 3 });
    const b = synthesizeMovementSequences({ count: 5, grammar, seed: 3 });
    expect(a).toEqual(b);
    expect(a).toHaveLength(5);
  });

  it("scores generalization on held-out but related synthetic movements", async () => {
    const grammar = [
      ["observation:screen", "action:focus", "action:click", "action:submit"],
      ["observation:screen", "action:focus", "action:type", "action:submit"],
    ];
    const dataset = { sequences: synthesizeMovementSequences({ count: 12, grammar, seed: 9 }) };
    const result = await evaluateGeneralization(new MarkovMovementBackend(), dataset, { k: 2 });

    expect(result.sequencesEvaluated).toBe(12);
    expect(result.transitionsEvaluated).toBeGreaterThan(0);
    // Shared grammar => the model should predict most held-out transitions well.
    expect(result.top1Accuracy).toBeGreaterThan(0.5);
    expect(result.topKAccuracy).toBeGreaterThanOrEqual(result.top1Accuracy);
  });
});
