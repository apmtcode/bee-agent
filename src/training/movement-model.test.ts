import { describe, expect, it } from "vitest";
import type { ReplayManifest, ReplayTimelineEvent } from "../capture/replay.js";
import {
  MarkovMovementModelBackend,
  buildMovementDataset,
  evaluateMovementPolicy,
  type MovementModelBackend,
  type MovementSequence,
} from "./movement-model.js";

/** Synthetic event-stream generator: turn action summaries into a replay manifest. */
function syntheticReplay(trajectoryId: string, actions: Array<{ tool: string; summary: string }>): ReplayManifest {
  const events: ReplayTimelineEvent[] = actions.map((action, index) => ({
    kind: "action",
    ts: index * 10,
    trajectoryId,
    tool: action.tool,
    summary: action.summary,
  }));
  // Interleave a noise observation to prove non-action events are ignored.
  events.push({ kind: "observation", ts: 5, trajectoryId, source: "device", summary: "screen active" });
  return {
    version: 1,
    sessionId: `session-${trajectoryId}`,
    trajectoryIds: [trajectoryId],
    eventCount: events.length,
    events,
  };
}

describe("buildMovementDataset", () => {
  it("extracts only action events, grouped by trajectory and ordered by ts", () => {
    const dataset = buildMovementDataset([
      syntheticReplay("t1", [
        { tool: "device", summary: "tapped search" },
        { tool: "device", summary: "typed query" },
        { tool: "device", summary: "tapped submit" },
      ]),
    ]);

    expect(dataset.sequences).toHaveLength(1);
    const steps = dataset.sequences[0].steps;
    expect(steps).toHaveLength(3);
    expect(steps.map((s) => s.action)).toEqual(["tapped", "typed", "tapped"]);
    expect(steps[0]).toMatchObject({ tool: "device", action: "tapped", target: "search" });
    // Observation event was dropped.
    expect(steps.every((s) => s.tool === "device")).toBe(true);
  });
});

describe("MarkovMovementModelBackend — repeat", () => {
  it("reproduces a recorded sequence exactly via rollout", () => {
    // Distinct action verbs give each n-gram context a unique continuation, so
    // an order-2 model reproduces the whole run from a single seed step.
    const recorded = syntheticReplay("t1", [
      { tool: "device", summary: "tapped search" },
      { tool: "device", summary: "typed query" },
      { tool: "device", summary: "scrolled down" },
      { tool: "device", summary: "submitted form" },
    ]);
    const dataset = buildMovementDataset([recorded]);
    const backend: MovementModelBackend = new MarkovMovementModelBackend();
    const policy = backend.train(dataset, { order: 2 });

    const full = dataset.sequences[0].steps;
    const seed = full.slice(0, 1);
    const rolled = policy.rollout(seed, { maxSteps: full.length - seed.length });
    const reconstructed = [...seed, ...rolled].map((s) => `${s.tool} ${s.action}`);
    expect(reconstructed).toEqual(full.map((s) => `${s.tool} ${s.action}`));
  });

  it("predicts the recorded next step as exact with full confidence on a unique path", () => {
    const dataset = buildMovementDataset([
      syntheticReplay("t1", [
        { tool: "device", summary: "tapped a" },
        { tool: "device", summary: "tapped b" },
        { tool: "device", summary: "tapped c" },
      ]),
    ]);
    const policy = new MarkovMovementModelBackend().train(dataset, { order: 2 });
    const prediction = policy.predict(dataset.sequences[0].steps.slice(0, 2));
    expect(prediction?.source).toBe("exact");
    expect(prediction?.step.action).toBe("tapped");
    expect(prediction?.confidence).toBe(1);
  });
});

describe("MarkovMovementModelBackend — generalize", () => {
  it("handles a novel-but-related context by backing off to shared structure", () => {
    // Training teaches the local rule "typed -> submitted". A held-out run
    // reaches "typed" from a brand-new predecessor ("scrolled"), so the full
    // order-2 verb window is unseen and the model must back off to the order-1
    // rule to still predict the right next movement.
    const dataset = buildMovementDataset([
      syntheticReplay("t1", [
        { tool: "device", summary: "tapped menu" },
        { tool: "device", summary: "typed value" },
        { tool: "device", summary: "submitted form" },
      ]),
      syntheticReplay("t2", [
        { tool: "device", summary: "scrolled down" },
        { tool: "device", summary: "tapped icon" },
        { tool: "device", summary: "opened panel" },
      ]),
    ]);
    const policy = new MarkovMovementModelBackend().train(dataset, { order: 2 });

    const novelHistory = [
      { ts: 0, tool: "device", action: "scrolled", target: "page" },
      { ts: 1, tool: "device", action: "typed", target: "query" },
    ];
    const prediction = policy.predict(novelHistory);
    expect(prediction).toBeDefined();
    expect(prediction?.step.action).toBe("submitted"); // learned typed -> submitted
    expect(prediction?.source).toBe("backoff");
    expect(prediction?.matchedOrder).toBe(1);
  });

  it("falls back to the global prior when no context matches", () => {
    const dataset = buildMovementDataset([
      syntheticReplay("t1", [
        { tool: "device", summary: "tapped a" },
        { tool: "device", summary: "tapped a" },
        { tool: "device", summary: "tapped a" },
      ]),
    ]);
    const policy = new MarkovMovementModelBackend().train(dataset, { order: 2 });
    const prediction = policy.predict([{ ts: 0, tool: "keyboard", action: "pressed", target: "unseen" }]);
    expect(prediction?.source).toBe("prior");
    expect(prediction?.step.action).toBe("tapped");
  });
});

describe("evaluateMovementPolicy", () => {
  it("reports perfect fidelity when held-out equals training", () => {
    const replays = [
      syntheticReplay("t1", [
        { tool: "device", summary: "tapped a" },
        { tool: "device", summary: "tapped b" },
        { tool: "device", summary: "tapped c" },
      ]),
    ];
    const dataset = buildMovementDataset(replays);
    const policy = new MarkovMovementModelBackend().train(dataset, { order: 2 });
    const report = evaluateMovementPolicy(policy, dataset.sequences);
    expect(report.accuracy).toBe(1);
    expect(report.correct).toBe(report.predicted);
  });

  it("measures generalization on held-out but related sequences", () => {
    const training = buildMovementDataset([
      syntheticReplay("t1", [
        { tool: "device", summary: "tapped open" },
        { tool: "device", summary: "tapped field" },
        { tool: "device", summary: "typed value" },
        { tool: "device", summary: "tapped submit" },
      ]),
      syntheticReplay("t2", [
        { tool: "device", summary: "tapped launch" },
        { tool: "device", summary: "tapped field" },
        { tool: "device", summary: "typed value" },
        { tool: "device", summary: "tapped submit" },
      ]),
    ]);
    const policy = new MarkovMovementModelBackend().train(training, { order: 2 });

    const heldOut: MovementSequence[] = [
      {
        id: "held-1",
        steps: [
          { ts: 0, tool: "device", action: "tapped", target: "start" },
          { ts: 1, tool: "device", action: "tapped", target: "field" },
          { ts: 2, tool: "device", action: "typed", target: "value" },
          { ts: 3, tool: "device", action: "tapped", target: "submit" },
        ],
      },
    ];
    const report = evaluateMovementPolicy(policy, heldOut);
    expect(report.predicted).toBe(3);
    // The related tail is recovered; at least some hits come from generalization.
    expect(report.correct).toBeGreaterThanOrEqual(2);
    expect(report.generalizationRate).toBeGreaterThan(0);
  });
});

describe("snapshot round-trip", () => {
  it("reloads an equivalent policy from its snapshot", () => {
    const dataset = buildMovementDataset([
      syntheticReplay("t1", [
        { tool: "device", summary: "tapped a" },
        { tool: "device", summary: "tapped b" },
        { tool: "device", summary: "tapped c" },
      ]),
    ]);
    const trained = new MarkovMovementModelBackend().train(dataset, { order: 2 });
    const snapshot = trained.toSnapshot();
    const reloaded = MarkovMovementModelBackend.fromSnapshot(snapshot);

    const history = dataset.sequences[0].steps.slice(0, 2);
    expect(reloaded.predict(history)).toEqual(trained.predict(history));
    expect(reloaded.toSnapshot()).toEqual(snapshot);
  });
});
