import { describe, expect, it } from "vitest";
import type { ExportedReplayManifest, ReviewedExportManifest } from "./export-manifest.js";
import {
  DeterministicMarkovBackend,
  datasetFromExport,
  datasetFromReplays,
  evaluateNextActionAccuracy,
  generateSyntheticReplays,
  loadMovementModel,
  rolloutMovements,
  trainMovementModelFromExport,
  type MovementContext,
  type MovementEvent,
} from "./movement-backend.js";

function action(channel: string, ts: number, summary = `perform ${channel}`): MovementEvent {
  return { kind: "action", channel, summary, ts };
}
function observation(channel: string, ts: number, summary = `observe ${channel}`): MovementEvent {
  return { kind: "observation", channel, summary, ts };
}
function ctx(...recent: MovementEvent[]): MovementContext {
  return { recent };
}

describe("datasetFromReplays", () => {
  it("keeps only observation/action events and sorts them by timestamp", () => {
    const replays: ExportedReplayManifest[] = [
      {
        sessionId: "s1",
        trajectoryIds: ["t1"],
        eventCount: 3,
        events: [
          { kind: "action", ts: 3, trajectoryId: "t1", tool: "click", summary: "click ok" },
          { kind: "transcript", ts: 1, messageId: "m1", role: "user", content: "hi" },
          { kind: "observation", ts: 2, trajectoryId: "t1", source: "screen", summary: "dialog" },
        ],
      },
    ];
    const dataset = datasetFromReplays(replays);
    expect(dataset.sequences).toHaveLength(1);
    const events = dataset.sequences[0]!.events;
    expect(events.map((e) => e.kind)).toEqual(["observation", "action"]);
    expect(events.map((e) => e.ts)).toEqual([2, 3]);
    expect(events[1]).toMatchObject({ kind: "action", channel: "click", summary: "click ok" });
  });
});

describe("DeterministicMarkovBackend training + prediction", () => {
  it("repeats a recorded movement exactly from a seen context", async () => {
    // observe→click→observe→type, twice.
    const dataset = datasetFromReplays(
      generateSyntheticReplays({
        seed: 7,
        sessions: 4,
        stepsPerSession: 3,
        programs: [{ name: "form", steps: [{ observation: "field", action: "focus" }, { observation: "text", action: "type" }] }],
      }),
    );
    const model = await new DeterministicMarkovBackend().train(dataset);

    // After focus + a "text" observation, the recorded next action is "type".
    const prediction = model.predict(ctx(observation("field", 0), action("focus", 1), observation("text", 2)));
    expect(prediction.tool).toBe("type");
    expect(prediction.source).toBe("exact-context");
    expect(prediction.generalized).toBe(false);
    expect(prediction.confidence).toBeGreaterThan(0.5);
    expect(prediction.summary).toBe("perform type");
  });

  it("is deterministic: identical dataset yields byte-identical weights", async () => {
    const replays = generateSyntheticReplays({
      seed: 42,
      sessions: 5,
      stepsPerSession: 4,
      programs: [{ name: "nav", steps: [{ observation: "menu", action: "open" }, { observation: "item", action: "select" }] }],
    });
    const a = await new DeterministicMarkovBackend().train(datasetFromReplays(replays));
    const b = await new DeterministicMarkovBackend().train(datasetFromReplays(replays));
    expect(a.serialize()).toEqual(b.serialize());
  });

  it("returns an empty prediction when nothing was learned", async () => {
    const model = await new DeterministicMarkovBackend().train({ sequences: [] });
    const prediction = model.predict(ctx(observation("anything", 0)));
    expect(prediction.source).toBe("empty");
    expect(prediction.tool).toBe("");
  });
});

describe("generalization to new-but-related movements", () => {
  it("backs off to the action pattern when the observation is novel", async () => {
    // Train: after action "focus", the next action is always "type".
    const dataset = datasetFromReplays(
      generateSyntheticReplays({
        seed: 1,
        sessions: 6,
        stepsPerSession: 2,
        programs: [{ name: "form", steps: [{ observation: "known-field", action: "focus" }, { observation: "known-text", action: "type" }] }],
      }),
    );
    const model = await new DeterministicMarkovBackend().train(dataset);

    // Novel observation source never seen in training, but the "focus" action is known.
    const prediction = model.predict(ctx(action("focus", 1), observation("brand-new-field", 2)));
    expect(prediction.tool).toBe("type");
    expect(prediction.generalized).toBe(true);
    expect(prediction.source).toBe("backoff-action");
  });

  it("scores high next-action accuracy on held-out related sequences", async () => {
    const programs = [
      { name: "editor", steps: [{ observation: "cursor", action: "move" }, { observation: "char", action: "keypress" }] },
      { name: "browser", steps: [{ observation: "link", action: "hover" }, { observation: "page", action: "click" }] },
    ];
    const train = datasetFromReplays(generateSyntheticReplays({ seed: 11, sessions: 12, stepsPerSession: 5, programs }));
    const heldOut = datasetFromReplays(generateSyntheticReplays({ seed: 99, sessions: 6, stepsPerSession: 5, programs }));
    const model = await new DeterministicMarkovBackend().train(train);

    const evaluation = evaluateNextActionAccuracy(model, heldOut);
    expect(evaluation.total).toBeGreaterThan(0);
    // Learnable structure → the model should reproduce the vast majority of actions.
    expect(evaluation.accuracy).toBeGreaterThan(0.8);
  });
});

describe("rolloutMovements", () => {
  it("autoregressively regenerates the recorded action loop", async () => {
    const dataset = datasetFromReplays(
      generateSyntheticReplays({
        seed: 3,
        sessions: 5,
        stepsPerSession: 6,
        programs: [{ name: "loop", steps: [{ observation: "a", action: "stepA" }, { observation: "b", action: "stepB" }] }],
      }),
    );
    const model = await new DeterministicMarkovBackend().train(dataset);
    const rollout = rolloutMovements(model, ctx(observation("a", 0), action("stepA", 1), observation("b", 2)), 4);
    expect(rollout.length).toBeGreaterThan(0);
    // Every produced step is a concrete action tool the model knows.
    for (const prediction of rollout) {
      expect(["stepA", "stepB"]).toContain(prediction.tool);
    }
  });

  it("stops early when the model has nothing to predict", async () => {
    const model = await new DeterministicMarkovBackend().train({ sequences: [] });
    const rollout = rolloutMovements(model, ctx(observation("x", 0)), 5);
    expect(rollout).toHaveLength(0);
  });
});

describe("serialize / loadMovementModel round-trip", () => {
  it("reloads a persisted model and reproduces predictions", async () => {
    const dataset = datasetFromReplays(
      generateSyntheticReplays({
        seed: 5,
        sessions: 4,
        stepsPerSession: 3,
        programs: [{ name: "p", steps: [{ observation: "o1", action: "a1" }, { observation: "o2", action: "a2" }] }],
      }),
    );
    const trained = await new DeterministicMarkovBackend().train(dataset);
    const reloaded = loadMovementModel(trained.serialize());

    const context = ctx(observation("o1", 0), action("a1", 1), observation("o2", 2));
    expect(reloaded.predict(context)).toEqual(trained.predict(context));
    expect(reloaded.serialize()).toEqual(trained.serialize());
  });
});

describe("trainMovementModelFromExport", () => {
  it("trains directly from a reviewed export manifest", async () => {
    const replays = generateSyntheticReplays({
      seed: 8,
      sessions: 3,
      stepsPerSession: 2,
      programs: [{ name: "e", steps: [{ observation: "obs", action: "act" }] }],
    });
    const manifest: ReviewedExportManifest = {
      version: 1,
      createdAt: "2026-01-01T00:00:00.000Z",
      reviewedBy: "tester",
      purpose: "unit-test",
      targetPlatform: "apple-silicon",
      modes: ["sft"],
      rawCaptureIncluded: false,
      promotedSkills: [],
      executableSkills: [],
      executableSkillRuns: [],
      memories: [],
      trajectories: [],
      replays,
    };
    const { dataset, model } = await trainMovementModelFromExport(manifest, new DeterministicMarkovBackend());
    expect(datasetFromExport(manifest).sequences).toHaveLength(dataset.sequences.length);
    expect(model.serialize().eventCount).toBeGreaterThan(0);
    expect(model.backend).toBe("deterministic-markov");
  });
});
