import { describe, expect, it } from "vitest";
import type { ExportedReplayManifest } from "./export-manifest.js";
import {
  MarkovMovementBackend,
  createMovementModelBackend,
  defaultMovementTokenizer,
  evaluateMovementModel,
  extractMovementDataset,
  type MovementSequence,
} from "./movement-model.js";

/** Synthetic replay generator: a deterministic "open → navigate → act → confirm" flow. */
function syntheticReplay(id: string, steps: Array<{ tool: string; summary: string }>): ExportedReplayManifest {
  return {
    sessionId: `sess-${id}`,
    trajectoryIds: [id],
    eventCount: steps.length,
    events: steps.map((step, index) => ({
      kind: "action" as const,
      ts: index + 1,
      trajectoryId: id,
      tool: step.tool,
      summary: step.summary,
    })),
  };
}

const FLOW: Array<{ tool: string; summary: string }> = [
  { tool: "browser", summary: "open dashboard" },
  { tool: "browser", summary: "navigate to deploy" },
  { tool: "browser", summary: "click deploy" },
  { tool: "browser", summary: "confirm deploy" },
];

describe("extractMovementDataset", () => {
  it("folds replay events into recurring movement tokens in timeline order", () => {
    const dataset = extractMovementDataset([syntheticReplay("traj-1", FLOW)], { kinds: ["action"] });
    expect(dataset.sequences).toHaveLength(1);
    expect(dataset.sequences[0]).toEqual({
      id: "traj-1",
      tokens: [
        "act:browser:open-dashboard",
        "act:browser:navigate-to-deploy",
        "act:browser:click-deploy",
        "act:browser:confirm-deploy",
      ],
    });
  });

  it("sorts out-of-order events by timestamp and respects minLength", () => {
    const replay: ExportedReplayManifest = {
      sessionId: "s",
      trajectoryIds: ["t"],
      eventCount: 2,
      events: [
        { kind: "action", ts: 5, trajectoryId: "t", tool: "x", summary: "second" },
        { kind: "action", ts: 1, trajectoryId: "t", tool: "x", summary: "first" },
      ],
    };
    const dataset = extractMovementDataset([replay], { kinds: ["action"] });
    expect(dataset.sequences[0]?.tokens).toEqual(["act:x:first", "act:x:second"]);

    const filtered = extractMovementDataset([replay], { kinds: ["action"], minLength: 5 });
    expect(filtered.sequences).toHaveLength(0);
  });

  it("includes observations when requested and supports a custom tokenizer", () => {
    const replay: ExportedReplayManifest = {
      sessionId: "s",
      trajectoryIds: ["t"],
      eventCount: 2,
      events: [
        { kind: "observation", ts: 1, trajectoryId: "t", source: "device", summary: "screen ready" },
        { kind: "action", ts: 2, trajectoryId: "t", tool: "device", summary: "tap go" },
      ],
    };
    const dataset = extractMovementDataset([replay], {
      tokenize: (input) => (input.kind === "action" ? `A/${input.summary}` : `O/${input.summary}`),
    });
    expect(dataset.sequences[0]?.tokens).toEqual(["O/screen ready", "A/tap go"]);
  });
});

describe("defaultMovementTokenizer", () => {
  it("collapses semantically identical movements to the same token", () => {
    const a = defaultMovementTokenizer({ kind: "action", tool: "Browser", summary: "Click Deploy!" });
    const b = defaultMovementTokenizer({ kind: "action", tool: "browser", summary: "click   deploy" });
    expect(a).toBe(b);
    expect(a).toBe("act:browser:click-deploy");
  });
});

describe("MarkovMovementBackend", () => {
  it("repeats a recorded movement sequence (objective c)", () => {
    const dataset = extractMovementDataset([syntheticReplay("traj-1", FLOW)], { kinds: ["action"] });
    const backend = new MarkovMovementBackend();
    const artifact = backend.train(dataset, { order: 2 });
    const model = backend.restore(artifact);

    const generated = model.generate({ maxSteps: 10 });
    expect(generated).toEqual(dataset.sequences[0]?.tokens);
  });

  it("is deterministic: identical artifacts produce identical generations", () => {
    const dataset = extractMovementDataset(
      [syntheticReplay("a", FLOW), syntheticReplay("b", FLOW)],
      { kinds: ["action"] },
    );
    const backend = new MarkovMovementBackend();
    const first = backend.restore(backend.train(dataset)).generate();
    const second = backend.restore(backend.train(dataset)).generate();
    expect(first).toEqual(second);
  });

  it("generalizes: blends transitions across related trajectories into a novel sequence (objective d)", () => {
    // Two related flows that share a prefix but diverge, then reconverge.
    const flowA = [
      { tool: "browser", summary: "open dashboard" },
      { tool: "browser", summary: "navigate to deploy" },
      { tool: "browser", summary: "click deploy" },
      { tool: "browser", summary: "confirm action" },
    ];
    const flowB = [
      { tool: "browser", summary: "open dashboard" },
      { tool: "browser", summary: "navigate to rollback" },
      { tool: "browser", summary: "click rollback" },
      { tool: "browser", summary: "confirm action" },
    ];
    const dataset = extractMovementDataset(
      [syntheticReplay("a", flowA), syntheticReplay("b", flowB)],
      { kinds: ["action"] },
    );
    const backend = new MarkovMovementBackend();
    const model = backend.restore(backend.train(dataset, { order: 1 }));

    // Seed with the rollback branch; the order-1 model should follow rollback
    // transitions then reconverge on the shared "confirm action" ending —
    // a coherent path it can produce for the related context.
    const generated = model.generate({ seed: ["act:browser:navigate-to-rollback"], maxSteps: 10 });
    expect(generated).toContain("act:browser:click-rollback");
    expect(generated.at(-1)).toBe("act:browser:confirm-action");
  });

  it("predictNext returns a normalized, ranked distribution", () => {
    const dataset = extractMovementDataset([syntheticReplay("traj-1", FLOW)], { kinds: ["action"] });
    const model = new MarkovMovementBackend().restore(new MarkovMovementBackend().train(dataset, { order: 1 }));

    const predictions = model.predictNext(["act:browser:open-dashboard"]);
    expect(predictions[0]?.token).toBe("act:browser:navigate-to-deploy");
    const mass = predictions.reduce((sum, p) => sum + p.probability, 0);
    expect(mass).toBeCloseTo(1, 6);
    // Ranking is sorted descending by probability.
    for (let i = 1; i < predictions.length; i += 1) {
      expect(predictions[i - 1]!.probability).toBeGreaterThanOrEqual(predictions[i]!.probability);
    }
  });

  it("rejects restoring a foreign artifact", () => {
    expect(() => new MarkovMovementBackend().restore({ backend: "mlx", version: 1 })).toThrow(/cannot restore/);
  });
});

describe("evaluateMovementModel", () => {
  it("scores perfect next-token accuracy on the training flow", () => {
    const dataset = extractMovementDataset([syntheticReplay("traj-1", FLOW)], { kinds: ["action"] });
    const model = new MarkovMovementBackend().restore(new MarkovMovementBackend().train(dataset, { order: 2 }));
    const result = evaluateMovementModel(model, dataset.sequences);
    expect(result.nextTokenAccuracy).toBe(1);
    expect(result.correctPredictions).toBe(result.totalPredictions);
  });

  it("measures generalization on a held-out related sequence", () => {
    const train = extractMovementDataset(
      [
        syntheticReplay("a", FLOW),
        syntheticReplay("b", [
          { tool: "browser", summary: "open dashboard" },
          { tool: "browser", summary: "navigate to deploy" },
          { tool: "browser", summary: "click deploy" },
          { tool: "browser", summary: "verify deploy" },
        ]),
      ],
      { kinds: ["action"] },
    );
    const model = new MarkovMovementBackend().restore(new MarkovMovementBackend().train(train, { order: 2 }));

    const heldOut: MovementSequence[] = [
      {
        id: "held",
        tokens: [
          "act:browser:open-dashboard",
          "act:browser:navigate-to-deploy",
          "act:browser:click-deploy",
        ],
      },
    ];
    const result = evaluateMovementModel(model, heldOut);
    // The shared prefix is fully learned, so the model predicts the held-out
    // continuation with high accuracy.
    expect(result.nextTokenAccuracy).toBeGreaterThan(0.5);
    expect(result.perSequence[0]?.id).toBe("held");
  });
});

describe("createMovementModelBackend", () => {
  it("returns the markov backend by default", () => {
    expect(createMovementModelBackend().id).toBe("markov");
    expect(createMovementModelBackend("markov")).toBeInstanceOf(MarkovMovementBackend);
  });

  it("documents the on-device mlx seam as cloud-unavailable", () => {
    expect(() => createMovementModelBackend("mlx")).toThrow(/on-device/);
  });
});
