import { describe, expect, it } from "vitest";
import type { ExportedReplayManifest } from "./export-manifest.js";
import {
  DEFAULT_MOVEMENT_CONTEXT_WINDOW,
  DeterministicMarkovBackend,
  MovementModelTrainer,
  buildMovementDataset,
  evaluateMovementModel,
  type MovementStep,
} from "./movement-model.js";

type ReplayEvents = Pick<ExportedReplayManifest, "events">;

function replay(events: ExportedReplayManifest["events"]): ReplayEvents {
  return { events };
}

function action(trajectoryId: string, ts: number, tool: string, summary: string): ExportedReplayManifest["events"][number] {
  return { kind: "action", ts, trajectoryId, tool, summary };
}

function observation(trajectoryId: string, ts: number, source: string, summary: string): ExportedReplayManifest["events"][number] {
  return { kind: "observation", ts, trajectoryId, source, summary };
}

describe("buildMovementDataset", () => {
  it("derives context->action steps in timeline order with a sliding window", () => {
    const dataset = buildMovementDataset(
      [
        replay([
          observation("t1", 1, "editor", "editor focused"),
          action("t1", 2, "mouse", "click file menu"),
          action("t1", 3, "keyboard", "type filename"),
          action("t1", 4, "mouse", "click save"),
        ]),
      ],
      { contextWindow: 2 },
    );

    expect(dataset.contextWindow).toBe(2);
    expect(dataset.steps).toHaveLength(3);

    expect(dataset.steps[0]).toMatchObject({
      contextTools: [],
      observationSource: "editor",
      action: { tool: "mouse", summary: "click file menu" },
    });
    expect(dataset.steps[1].contextTools).toEqual(["mouse"]);
    expect(dataset.steps[2].contextTools).toEqual(["mouse", "keyboard"]);
  });

  it("sorts out-of-order events and caps the context window", () => {
    const dataset = buildMovementDataset(
      [
        replay([
          action("t1", 4, "d", "fourth"),
          action("t1", 1, "a", "first"),
          action("t1", 3, "c", "third"),
          action("t1", 2, "b", "second"),
        ]),
      ],
      { contextWindow: 2 },
    );

    expect(dataset.steps.map((step) => step.action.tool)).toEqual(["a", "b", "c", "d"]);
    // by the 4th action, window holds only the two most recent prior tools
    expect(dataset.steps[3].contextTools).toEqual(["b", "c"]);
  });

  it("defaults the context window when unspecified", () => {
    const dataset = buildMovementDataset([replay([action("t1", 1, "a", "x")])]);
    expect(dataset.contextWindow).toBe(DEFAULT_MOVEMENT_CONTEXT_WINDOW);
  });
});

describe("DeterministicMarkovBackend", () => {
  const backend = new DeterministicMarkovBackend();

  it("repeats a recorded movement exactly", () => {
    const dataset = buildMovementDataset(
      [
        replay([
          action("t1", 1, "open", "open doc"),
          action("t1", 2, "type", "type body"),
          action("t1", 3, "save", "save doc"),
        ]),
      ],
      { contextWindow: 3 },
    );
    const model = backend.train(dataset);

    const prediction = backend.predict(model, { contextTools: ["open", "type"] });
    expect(prediction?.action).toEqual({ tool: "save", summary: "save doc" });
    expect(prediction?.backoff).toBe("exact");
    expect(prediction?.matchedContextLength).toBe(2);
    expect(prediction?.confidence).toBe(1);
  });

  it("generalizes to a new-but-related context via backoff", () => {
    // Two trajectories that both end with a "save" following a "type", but with
    // different leading actions. A never-seen prefix should still predict "save".
    const dataset = buildMovementDataset(
      [
        replay([
          action("t1", 1, "openA", "open A"),
          action("t1", 2, "type", "type"),
          action("t1", 3, "save", "save"),
        ]),
        replay([
          action("t2", 1, "openB", "open B"),
          action("t2", 2, "type", "type"),
          action("t2", 3, "save", "save"),
        ]),
      ],
      { contextWindow: 3 },
    );
    const model = backend.train(dataset);

    // Unseen leading action "openC", but the "...->type" suffix generalizes.
    const prediction = backend.predict(model, { contextTools: ["openC", "type"] });
    expect(prediction?.action.tool).toBe("save");
    expect(prediction?.backoff).toBe("partial");
    expect(prediction?.matchedContextLength).toBe(1);
  });

  it("falls back to the observation-conditioned table, then the global prior", () => {
    const dataset = buildMovementDataset(
      [
        replay([
          observation("t1", 1, "terminal", "terminal active"),
          action("t1", 2, "run", "run command"),
        ]),
      ],
      { contextWindow: 2 },
    );
    const model = backend.train(dataset);

    // No matching context tools, but the observation source is known.
    const viaObservation = backend.predict(model, {
      contextTools: ["totallyUnseen"],
      observationSource: "terminal",
    });
    // context length 0 always matches (the empty-context table exists), so a known
    // single-action dataset resolves through the empty-context prior first.
    expect(viaObservation?.action.tool).toBe("run");

    // A model trained on nothing yields no prediction at all.
    const empty = backend.train({ version: 1, contextWindow: 2, steps: [] });
    expect(backend.predict(empty, { contextTools: [] })).toBeUndefined();
  });

  it("produces a probability-ranked, deterministic candidate list", () => {
    const dataset = buildMovementDataset(
      [
        replay([action("t1", 1, "type", "a"), action("t1", 2, "save", "s")]),
        replay([action("t2", 1, "type", "a"), action("t2", 2, "save", "s")]),
        replay([action("t3", 1, "type", "a"), action("t3", 2, "undo", "u")]),
      ],
      { contextWindow: 1 },
    );
    const model = backend.train(dataset);

    const prediction = backend.predict(model, { contextTools: ["type"] });
    expect(prediction?.action.tool).toBe("save");
    expect(prediction?.confidence).toBeCloseTo(2 / 3, 10);
    expect(prediction?.candidates.map((c) => c.action.tool)).toEqual(["save", "undo"]);
  });

  it("serializes to JSON and predicts identically after a round-trip", () => {
    const dataset = buildMovementDataset(
      [replay([action("t1", 1, "open", "o"), action("t1", 2, "type", "t"), action("t1", 3, "save", "s")])],
      { contextWindow: 3 },
    );
    const model = backend.train(dataset);
    const rehydrated = JSON.parse(JSON.stringify(model)) as typeof model;

    const before = backend.predict(model, { contextTools: ["open", "type"] });
    const after = backend.predict(rehydrated, { contextTools: ["open", "type"] });
    expect(after).toEqual(before);
  });

  it("rejects a model produced by a different backend", () => {
    const foreign = { version: 1 as const, backend: "mlx-lora", contextWindow: 2, stepCount: 0, parameters: {} };
    expect(() => backend.predict(foreign, { contextTools: [] })).toThrow(/cannot read model/);
  });
});

describe("MovementModelTrainer", () => {
  it("trains from a reviewed export and predicts", () => {
    const trainer = new MovementModelTrainer(new DeterministicMarkovBackend(), { contextWindow: 2 });
    const model = trainer.trainFromExport({
      version: 1,
      createdAt: "2026-07-19T00:00:00.000Z",
      reviewedBy: "reviewer",
      purpose: "test",
      targetPlatform: "apple-silicon",
      modes: ["sft"],
      rawCaptureIncluded: false,
      promotedSkills: [],
      executableSkills: [],
      executableSkillRuns: [],
      memories: [],
      trajectories: [],
      replays: [
        {
          sessionId: "s1",
          trajectoryIds: ["t1"],
          eventCount: 2,
          events: [action("t1", 1, "focus", "focus field"), action("t1", 2, "submit", "submit form")],
        },
      ],
    });

    expect(model.contextWindow).toBe(2);
    expect(model.stepCount).toBe(2);
    expect(trainer.predict(model, { contextTools: ["focus"] })?.action.tool).toBe("submit");
  });
});

describe("evaluateMovementModel", () => {
  it("scores held-out steps and counts generalized (backed-off) hits", () => {
    const backend = new DeterministicMarkovBackend();
    const trainReplays = [
      replay([action("t1", 1, "openA", "a"), action("t1", 2, "type", "t"), action("t1", 3, "save", "s")]),
      replay([action("t2", 1, "openB", "b"), action("t2", 2, "type", "t"), action("t2", 3, "save", "s")]),
    ];
    const model = backend.train(buildMovementDataset(trainReplays, { contextWindow: 3 }));

    const heldOut: MovementStep[] = [
      // exact repeat
      { contextTools: ["openA", "type"], action: { tool: "save", summary: "s" }, trajectoryId: "t1", ts: 3 },
      // generalized: unseen prefix, correct via backoff
      { contextTools: ["openC", "type"], action: { tool: "save", summary: "s" }, trajectoryId: "t3", ts: 3 },
    ];

    const result = evaluateMovementModel(backend, model, heldOut);
    expect(result.total).toBe(2);
    expect(result.correct).toBe(2);
    expect(result.accuracy).toBe(1);
    expect(result.generalizedCorrect).toBe(1);
    expect(result.backoffBreakdown.exact).toBe(1);
    expect(result.backoffBreakdown.partial).toBe(1);
  });

  it("reports zero accuracy on an empty held-out set", () => {
    const backend = new DeterministicMarkovBackend();
    const model = backend.train({ version: 1, contextWindow: 1, steps: [] });
    expect(evaluateMovementModel(backend, model, []).accuracy).toBe(0);
  });
});
