import { describe, expect, it } from "vitest";
import { buildTrajectorySpan, type TrajectoryAction } from "../capture/trajectory.js";
import {
  DeterministicMovementBackend,
  buildMovementDataset,
  evaluateMovementModel,
  slugMovementField,
  tokenizeMovement,
} from "./movement-model.js";
import { generateSyntheticMovementTrajectories } from "./movement-synthetic.js";

function action(tool: string, ts: number, metadata?: Record<string, unknown>, summary = "did a thing"): TrajectoryAction {
  return { kind: "action", tool, summary, ts, ...(metadata ? { metadata } : {}) };
}

describe("tokenizeMovement", () => {
  it("builds a bounded token from structured gesture metadata", () => {
    expect(tokenizeMovement(action("device", 1, { gesture: "tap", target: "Button A" }))).toBe("device/tap/button-a");
    expect(
      tokenizeMovement(action("device", 1, { gesture: "swipe", direction: "left", target: "canvas" })),
    ).toBe("device/swipe/left/canvas");
  });

  it("falls back to a summary slug when no structured metadata is present", () => {
    expect(tokenizeMovement(action("shell", 1, undefined, "Run Build"))).toBe("shell/run-build");
  });

  it("slugs delimiter-heavy fields safely", () => {
    expect(slugMovementField("  Save & Exit!!  ")).toBe("save-exit");
    expect(slugMovementField("")).toBe("unknown");
  });
});

describe("buildMovementDataset", () => {
  it("orders actions by timestamp and derives a sorted vocabulary", () => {
    const span = buildTrajectorySpan({
      id: "t1",
      sessionId: "s1",
      actions: [
        action("device", 30, { gesture: "type", target: "field" }),
        action("device", 10, { gesture: "tap", target: "field" }),
        action("device", 20, { gesture: "tap", target: "submit" }),
      ],
    });
    const dataset = buildMovementDataset([span]);
    expect(dataset.sequences).toHaveLength(1);
    expect(dataset.sequences[0]!.tokens).toEqual([
      "device/tap/field",
      "device/tap/submit",
      "device/type/field",
    ]);
    expect(dataset.vocabulary).toEqual([...dataset.vocabulary].sort());
  });

  it("skips trajectories with no actions", () => {
    const empty = buildTrajectorySpan({ id: "e", sessionId: "s", actions: [] });
    expect(buildMovementDataset([empty]).sequences).toHaveLength(0);
  });
});

describe("DeterministicMovementBackend", () => {
  it("exactly replays a memorized movement chain", () => {
    const template = [
      { tool: "device", gesture: "tap", target: "menu" },
      { tool: "device", gesture: "tap", target: "new" },
      { tool: "device", gesture: "type", target: "title" },
      { tool: "device", gesture: "tap", target: "save" },
    ] as const;
    const span = buildTrajectorySpan({
      id: "t1",
      sessionId: "s1",
      actions: template.map((step, i) => action(step.tool, i, { gesture: step.gesture, target: step.target })),
    });
    const dataset = buildMovementDataset([span]);
    const model = new DeterministicMovementBackend().train(dataset, { order: 3 });

    const generated = model.generate([dataset.sequences[0]!.tokens[0]!], 3);
    expect([dataset.sequences[0]!.tokens[0]!, ...generated]).toEqual(dataset.sequences[0]!.tokens);
  });

  it("is deterministic: identical dataset trains an identical model", () => {
    const trajectories = generateSyntheticMovementTrajectories({
      seed: 7,
      count: 12,
      template: [
        { tool: "device", gesture: "tap", target: "menu" },
        { tool: "device", gesture: "tap", target: "compose" },
        { tool: "device", gesture: "type", target: "body" },
      ],
    });
    const dataset = buildMovementDataset(trajectories);
    const backend = new DeterministicMovementBackend();
    const a = backend.train(dataset).generate(["device/tap/menu"], 4);
    const b = backend.train(dataset).generate(["device/tap/menu"], 4);
    expect(a).toEqual(b);
  });

  it("backs off to shorter context for an unseen prefix and still predicts", () => {
    const span = buildTrajectorySpan({
      id: "t1",
      sessionId: "s1",
      actions: [
        action("device", 1, { gesture: "tap", target: "a" }),
        action("device", 2, { gesture: "tap", target: "b" }),
        action("device", 3, { gesture: "tap", target: "c" }),
      ],
    });
    const model = new DeterministicMovementBackend().train(buildMovementDataset([span]), { order: 2 });
    // Context never seen at full order → must back off but still return a token.
    const prediction = model.predictNext(["device/tap/z", "device/tap/a"]);
    expect(prediction.token).toBe("device/tap/b");
    expect(prediction.order).toBeLessThan(2);
    expect(prediction.confidence).toBeGreaterThan(0);
  });

  it("returns an empty prediction when nothing was learned", () => {
    const model = new DeterministicMovementBackend().train({ version: 1, sequences: [], vocabulary: [] });
    expect(model.predictNext(["anything"])).toEqual({ token: undefined, confidence: 0, order: -1 });
    expect(model.generate(["x"], 5)).toEqual([]);
  });
});

describe("generalization eval harness", () => {
  it("generalizes to held-out but related trajectories above chance", () => {
    const template = [
      { tool: "device", gesture: "tap", target: "menu" },
      { tool: "device", gesture: "tap", target: "compose" },
      { tool: "device", gesture: "type", target: "subject" },
      { tool: "device", gesture: "type", target: "body" },
      { tool: "device", gesture: "tap", target: "send" },
    ];
    const train = generateSyntheticMovementTrajectories({ seed: 1, count: 40, template, noise: 0.15 });
    const heldOut = generateSyntheticMovementTrajectories({ seed: 999, count: 10, template, noise: 0.15 });

    const model = new DeterministicMovementBackend().train(buildMovementDataset(train), { order: 3 });
    const result = evaluateMovementModel(model, buildMovementDataset(heldOut).sequences);

    expect(result.totalPredictions).toBeGreaterThan(0);
    // A 15-token-ish vocabulary → chance << 0.2; a model that generalizes the
    // shared skill should be well above that on held-out variations.
    expect(result.accuracy).toBeGreaterThan(0.6);
    expect(result.averageConfidence).toBeGreaterThan(0);
  });

  it("reports perfect accuracy replaying its own training data", () => {
    const template = [
      { tool: "device", gesture: "tap", target: "menu" },
      { tool: "device", gesture: "tap", target: "open" },
      { tool: "device", gesture: "tap", target: "save" },
    ];
    const trajectories = generateSyntheticMovementTrajectories({ seed: 3, count: 5, template, noise: 0 });
    const dataset = buildMovementDataset(trajectories);
    const model = new DeterministicMovementBackend().train(dataset, { order: 3 });
    const result = evaluateMovementModel(model, dataset.sequences);
    expect(result.accuracy).toBe(1);
  });
});

describe("generateSyntheticMovementTrajectories", () => {
  it("is seeded and reproducible", () => {
    const template = [
      { tool: "device", gesture: "tap", target: "a" },
      { tool: "device", gesture: "type", target: "b" },
    ];
    const first = generateSyntheticMovementTrajectories({ seed: 42, count: 6, template });
    const second = generateSyntheticMovementTrajectories({ seed: 42, count: 6, template });
    expect(buildMovementDataset(first)).toEqual(buildMovementDataset(second));
  });

  it("produces related-but-varied sequences (noise introduces new movements)", () => {
    const template = [
      { tool: "device", gesture: "tap", target: "a" },
      { tool: "device", gesture: "tap", target: "b" },
    ];
    const noisy = generateSyntheticMovementTrajectories({ seed: 5, count: 30, template, noise: 0.9 });
    const dataset = buildMovementDataset(noisy);
    // With high noise, at least one trajectory should differ from the template.
    const canonical = ["device/tap/a", "device/tap/b"];
    const anyVaried = dataset.sequences.some(
      (sequence) => JSON.stringify(sequence.tokens) !== JSON.stringify(canonical),
    );
    expect(anyVaried).toBe(true);
  });
});
