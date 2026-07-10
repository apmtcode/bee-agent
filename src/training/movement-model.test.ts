import { describe, expect, it } from "vitest";
import type { TrajectorySpan } from "../capture/trajectory.js";
import type { ReplayTimelineEvent } from "../capture/replay.js";
import {
  MarkovMovementBackend,
  MOVEMENT_WILDCARD,
  buildMovementDataset,
  evaluateMovementModel,
  movementSequenceFromReplayEvents,
  movementSequenceFromTrajectory,
  movementStepFromAction,
  movementStepShape,
  movementStepToken,
  synthesizeMovementSequences,
  type MovementSequence,
  type MovementTemplate,
} from "./movement-model.js";

function step(tool: string, gesture: string, target: string, direction?: string) {
  return { tool, gesture, target, ...(direction ? { direction } : {}) };
}

describe("movement tokenization", () => {
  it("normalizes fields and abstracts the target in the shape token", () => {
    const s = step("Device", "Tap", "Submit Button");
    expect(movementStepToken(s)).toBe("device|tap|submit button|*");
    expect(movementStepShape(s)).toBe(`device|tap|${MOVEMENT_WILDCARD}|*`);
  });

  it("keeps direction in both specific and shape tokens", () => {
    const s = step("os", "scroll", "list", "down");
    expect(movementStepToken(s)).toBe("os|scroll|list|down");
    expect(movementStepShape(s)).toBe("os|scroll|*|down");
  });

  it("derives a step from a captured trajectory action's metadata", () => {
    const derived = movementStepFromAction({
      kind: "action",
      tool: "device",
      summary: "swiped down",
      ts: 1,
      metadata: { gesture: "swipe", target: "feed", direction: "down" },
    });
    expect(derived).toEqual({ tool: "device", gesture: "swipe", target: "feed", direction: "down" });
  });
});

describe("dataset derivation", () => {
  it("orders trajectory actions by timestamp", () => {
    const trajectory: TrajectorySpan = {
      id: "t1",
      sessionId: "s1",
      createdAt: "2026-01-01T00:00:00.000Z",
      captureTier: "operator",
      observations: [],
      actions: [
        { kind: "action", tool: "device", summary: "b", ts: 20, metadata: { gesture: "tap", target: "b" } },
        { kind: "action", tool: "device", summary: "a", ts: 10, metadata: { gesture: "tap", target: "a" } },
      ],
    };
    const sequence = movementSequenceFromTrajectory(trajectory);
    expect(sequence.steps.map((s) => s.target)).toEqual(["a", "b"]);
  });

  it("keeps only action events from a replay timeline", () => {
    const events: ReplayTimelineEvent[] = [
      { kind: "transcript", ts: 1, messageId: "m", role: "user", content: "hi" },
      { kind: "observation", ts: 2, trajectoryId: "t", source: "device", summary: "screen" },
      { kind: "action", ts: 3, trajectoryId: "t", tool: "device", summary: "tapped ok" },
    ];
    const sequence = movementSequenceFromReplayEvents("t", events);
    expect(sequence.steps).toHaveLength(1);
    expect(sequence.steps[0].target).toBe("tapped ok");
  });

  it("drops empty trajectories from the dataset", () => {
    const empty: TrajectorySpan = {
      id: "e",
      sessionId: "s",
      createdAt: "2026-01-01T00:00:00.000Z",
      captureTier: "operator",
      observations: [],
      actions: [],
    };
    expect(buildMovementDataset([empty]).sequences).toHaveLength(0);
  });
});

describe("MarkovMovementBackend training + replay", () => {
  const sequences: MovementSequence[] = [
    {
      sourceId: "login",
      steps: [
        step("device", "tap", "username"),
        step("device", "type", "user@example.com"),
        step("device", "tap", "password"),
        step("device", "type", "secret"),
        step("device", "tap", "sign in"),
      ],
    },
  ];

  it("replays a memorized sequence exactly via the specific path", () => {
    const model = new MarkovMovementBackend().train({ version: 1, sequences });
    const prediction = model.predictNext([step("device", "tap", "username")]);
    expect(prediction?.level).toBe("specific");
    expect(prediction?.step.target).toBe("user@example.com");
    expect(prediction?.confidence).toBe(1);
  });

  it("generate() reproduces the full recorded continuation from a seed", () => {
    const model = new MarkovMovementBackend().train({ version: 1, sequences });
    const generated = model.generate([step("device", "tap", "username")], 4);
    expect(generated.map((s) => s.target)).toEqual(["user@example.com", "password", "secret", "sign in"]);
  });

  it("is deterministic: identical datasets train identical predictions", () => {
    const a = new MarkovMovementBackend().train({ version: 1, sequences });
    const b = new MarkovMovementBackend().train({ version: 1, sequences });
    const ctx = [step("device", "tap", "password")];
    expect(a.predictNext(ctx)).toEqual(b.predictNext(ctx));
  });
});

describe("generalization to novel-but-related movements", () => {
  // Train on a "tap field -> type -> tap submit" shape across several forms.
  const trainSequences: MovementSequence[] = [
    {
      sourceId: "form-a",
      steps: [step("device", "tap", "name-a"), step("device", "type", "value-a"), step("device", "tap", "submit-a")],
    },
    {
      sourceId: "form-b",
      steps: [step("device", "tap", "name-b"), step("device", "type", "value-b"), step("device", "tap", "submit-b")],
    },
  ];

  it("predicts the right shape for an unseen target via shape backoff", () => {
    const model = new MarkovMovementBackend().train({ version: 1, sequences: trainSequences });
    // "name-c" was never seen — exact context misses, shape context matches.
    const prediction = model.predictNext([step("device", "tap", "name-c")]);
    expect(prediction).toBeDefined();
    expect(prediction?.level).toBe("shape");
    expect(prediction?.step.gesture).toBe("type");
    expect(movementStepShape(prediction!.step)).toBe(`device|type|${MOVEMENT_WILDCARD}|*`);
  });

  it("falls back to the global prior when even the shape is unknown", () => {
    const model = new MarkovMovementBackend().train({ version: 1, sequences: trainSequences });
    const prediction = model.predictNext([step("browser", "scroll", "unknown", "up")]);
    expect(prediction?.level).toBe("prior");
  });

  it("returns undefined when the model has no data at all", () => {
    const model = new MarkovMovementBackend().train({ version: 1, sequences: [] });
    expect(model.predictNext([step("device", "tap", "x")])).toBeUndefined();
  });
});

describe("synthetic generator + eval harness", () => {
  const templates: MovementTemplate[] = [
    {
      id: "search",
      steps: [
        step("browser", "click", "search-box"),
        step("browser", "type", "{slot}"),
        step("browser", "click", "search-button"),
      ],
    },
  ];

  it("expands templates deterministically by substituting targets", () => {
    const seqs = synthesizeMovementSequences({
      templates,
      targets: ["cats", "dogs"],
      variationsPerTemplate: 2,
    });
    expect(seqs).toHaveLength(2);
    expect(seqs[0].steps[1].target).toBe("cats");
    expect(seqs[1].steps[1].target).toBe("dogs");
    // determinism
    const again = synthesizeMovementSequences({ templates, targets: ["cats", "dogs"], variationsPerTemplate: 2 });
    expect(again).toEqual(seqs);
  });

  it("generalizes across a train/held-out split of related sequences", () => {
    const train = synthesizeMovementSequences({
      templates,
      targets: ["alpha", "beta", "gamma"],
      variationsPerTemplate: 3,
    });
    const heldOut = synthesizeMovementSequences({
      templates,
      targets: ["delta", "epsilon"],
      variationsPerTemplate: 2,
    });
    const model = new MarkovMovementBackend().train({ version: 1, sequences: train });
    const metrics = evaluateMovementModel(model, heldOut);

    // Held-out targets are unseen. The static first click matches exactly, but
    // the typed-target transition can only be recovered via shape backoff —
    // perfect shape accuracy, full coverage, and every generalized prediction
    // (non-specific level) is shape-correct.
    expect(metrics.predictions).toBe(4); // 2 sequences x 2 transitions each
    expect(metrics.coverage).toBe(1);
    expect(metrics.shapeAccuracy).toBe(1);
    expect(metrics.generalizationRate).toBe(1);
    expect(metrics.byLevel.shape).toBeGreaterThan(0);
  });

  it("reports high exact accuracy when evaluated on seen sequences", () => {
    const train = synthesizeMovementSequences({ templates, targets: ["x"], variationsPerTemplate: 1 });
    const model = new MarkovMovementBackend().train({ version: 1, sequences: train });
    const metrics = evaluateMovementModel(model, train);
    expect(metrics.exactAccuracy).toBe(1);
    expect(metrics.byLevel.specific).toBe(metrics.predictions);
  });

  it("reports zero predictions for sequences too short to score", () => {
    const model = new MarkovMovementBackend().train({ version: 1, sequences: [] });
    const metrics = evaluateMovementModel(model, [{ sourceId: "s", steps: [step("device", "tap", "x")] }]);
    expect(metrics.predictions).toBe(0);
    expect(metrics.exactAccuracy).toBe(0);
  });
});
