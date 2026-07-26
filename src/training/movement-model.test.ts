import { describe, expect, it } from "vitest";
import { buildTrajectorySpan, type TrajectoryAction, type TrajectorySpan } from "../capture/trajectory.js";
import type { ReplayTimelineEvent } from "../capture/replay.js";
import {
  buildMovementDatasetFromReplays,
  buildMovementDatasetFromTrajectories,
  deriveMovementToken,
  evaluateMovementModel,
  MarkovMovementBackend,
  MovementBackendRegistry,
  MOVEMENT_EOS,
  type MovementModelBackend,
  type MovementSequence,
} from "./movement-model.js";

function gesture(kind: string, target: string, ts: number, tool = "device"): TrajectoryAction {
  return {
    kind: "action",
    tool,
    summary: `${kind} ${target}`,
    ts,
    metadata: { gesture: kind, target },
  };
}

function mailTrajectory(id: string, actions: TrajectoryAction[]): TrajectorySpan {
  return buildTrajectorySpan({
    id,
    sessionId: "session-1",
    observations: [
      {
        kind: "observation",
        source: "device",
        summary: "Mail is active",
        ts: 0,
        metadata: { appName: "Mail" },
      },
    ],
    actions,
  });
}

describe("deriveMovementToken", () => {
  it("collapses gesture actions to tool:gesture:target", () => {
    expect(deriveMovementToken({ tool: "device", summary: "tapped compose", metadata: { gesture: "tap", target: "Compose Button" } })).toBe(
      "device:tap:compose-button",
    );
  });

  it("prefers direction over target for directional gestures", () => {
    expect(deriveMovementToken({ tool: "device", summary: "swiped", metadata: { gesture: "swipe", direction: "left", target: "list" } })).toBe(
      "device:swipe:left",
    );
  });

  it("falls back to tool:verb for free-form actions", () => {
    expect(deriveMovementToken({ tool: "bash", summary: "run npm test" })).toBe("bash:run");
  });
});

describe("dataset builders", () => {
  it("builds sequences and a sorted vocabulary from trajectories", () => {
    const dataset = buildMovementDatasetFromTrajectories([
      mailTrajectory("t1", [gesture("tap", "compose", 2), gesture("type", "body", 3), gesture("tap", "send", 4)]),
    ]);
    expect(dataset.sequences).toHaveLength(1);
    expect(dataset.sequences[0].context).toBe("mail");
    expect(dataset.sequences[0].steps.map((step) => step.token)).toEqual([
      "device:tap:compose",
      "device:type:body",
      "device:tap:send",
    ]);
    // Vocabulary is sorted and includes the EOS marker.
    expect(dataset.vocabulary).toContain(MOVEMENT_EOS);
    expect([...dataset.vocabulary]).toEqual([...dataset.vocabulary].sort());
  });

  it("orders actions by timestamp and drops empty trajectories", () => {
    const dataset = buildMovementDatasetFromTrajectories([
      mailTrajectory("t1", [gesture("tap", "send", 9), gesture("tap", "compose", 1)]),
      mailTrajectory("empty", []),
    ]);
    expect(dataset.sequences).toHaveLength(1);
    expect(dataset.sequences[0].steps.map((step) => step.token)).toEqual(["device:tap:compose", "device:tap:send"]);
  });

  it("builds sequences from replay timelines grouped by trajectory", () => {
    const events: ReplayTimelineEvent[] = [
      { kind: "observation", ts: 0, trajectoryId: "t1", source: "browser", summary: "page" },
      { kind: "action", ts: 1, trajectoryId: "t1", tool: "browser", summary: "click login" },
      { kind: "action", ts: 2, trajectoryId: "t1", tool: "browser", summary: "type username" },
    ];
    const dataset = buildMovementDatasetFromReplays([{ events }]);
    expect(dataset.sequences).toHaveLength(1);
    expect(dataset.sequences[0].context).toBe("browser");
    expect(dataset.sequences[0].steps.map((step) => step.token)).toEqual(["browser:click", "browser:type"]);
  });
});

describe("MarkovMovementBackend — repeat recorded movements", () => {
  it("reproduces a recorded movement sequence exactly", () => {
    const dataset = buildMovementDatasetFromTrajectories([
      mailTrajectory("t1", [gesture("tap", "compose", 1), gesture("type", "body", 2), gesture("tap", "send", 3)]),
    ]);
    const model = new MarkovMovementBackend().train(dataset);
    const generated = model.generate({ context: "mail" });
    expect(generated.tokens).toEqual(["device:tap:compose", "device:type:body", "device:tap:send"]);
    expect(generated.reachedEnd).toBe(true);
  });

  it("continues from a seed prefix", () => {
    const dataset = buildMovementDatasetFromTrajectories([
      mailTrajectory("t1", [gesture("tap", "compose", 1), gesture("type", "body", 2), gesture("tap", "send", 3)]),
    ]);
    const model = new MarkovMovementBackend().train(dataset);
    const generated = model.generate({ context: "mail", seed: ["device:tap:compose"] });
    expect(generated.tokens).toEqual(["device:type:body", "device:tap:send"]);
  });
});

describe("MarkovMovementBackend — generalize via back-off", () => {
  it("predicts the learned continuation after an unseen prefix", () => {
    // Every training sequence ends "...type:body -> tap:send", behind varied prefixes.
    const dataset = buildMovementDatasetFromTrajectories([
      mailTrajectory("t1", [gesture("tap", "compose", 1), gesture("type", "body", 2), gesture("tap", "send", 3)]),
      mailTrajectory("t2", [gesture("tap", "reply", 1), gesture("type", "body", 2), gesture("tap", "send", 3)]),
      mailTrajectory("t3", [gesture("tap", "forward", 1), gesture("type", "body", 2), gesture("tap", "send", 3)]),
    ]);
    const model = new MarkovMovementBackend().train(dataset, { order: 3 });

    // A brand-new prefix the model never saw at full order, but which ends in a
    // token whose lower-order continuation it has learned.
    const prediction = model.predictNext({
      context: "mail",
      history: ["device:tap:draft", "device:type:body"],
    });
    expect(prediction?.token).toBe("device:tap:send");
    expect(prediction?.backoff).toBe(true);
    expect(prediction?.order).toBeLessThan(2);
  });

  it("scores memorization high and still generalizes to held-out sequences", () => {
    const training = buildMovementDatasetFromTrajectories([
      mailTrajectory("t1", [gesture("tap", "compose", 1), gesture("type", "body", 2), gesture("tap", "send", 3)]),
      mailTrajectory("t2", [gesture("tap", "reply", 1), gesture("type", "body", 2), gesture("tap", "send", 3)]),
    ]);
    const model = new MarkovMovementBackend().train(training, { order: 3 });

    // High memorization: only the ambiguous first token of a divergent-start
    // sequence can miss; every conditioned step is recovered.
    const onTraining = evaluateMovementModel(model, training.sequences);
    expect(onTraining.accuracy).toBeGreaterThanOrEqual(0.8);

    const heldOut: MovementSequence[] = buildMovementDatasetFromTrajectories([
      mailTrajectory("t3", [gesture("tap", "forward", 1), gesture("type", "body", 2), gesture("tap", "send", 3)]),
    ]).sequences;
    const generalization = evaluateMovementModel(model, heldOut);
    // The shared "type:body -> tap:send -> eos" tail is recovered by back-off.
    expect(generalization.accuracy).toBeGreaterThanOrEqual(0.5);
    expect(generalization.backoffRate).toBeGreaterThan(0);
  });
});

describe("MarkovMovementBackend — serialization", () => {
  it("round-trips through serialize/restore with identical predictions", () => {
    const dataset = buildMovementDatasetFromTrajectories([
      mailTrajectory("t1", [gesture("tap", "compose", 1), gesture("type", "body", 2), gesture("tap", "send", 3)]),
    ]);
    const backend = new MarkovMovementBackend();
    const model = backend.train(dataset);
    const restored = backend.restore(model.serialize());
    expect(restored.order).toBe(model.order);
    expect(restored.generate({ context: "mail" }).tokens).toEqual(model.generate({ context: "mail" }).tokens);
  });
});

describe("MovementBackendRegistry", () => {
  it("registers the markov backend by default and resolves it", () => {
    const registry = new MovementBackendRegistry();
    expect(registry.list()).toContain("markov");
    expect(registry.require("markov").id).toBe("markov");
  });

  it("supports pluggable backends and reports unknown ids", () => {
    const delegate = new MarkovMovementBackend();
    const custom: MovementModelBackend = {
      id: "custom",
      train: (dataset, options) => delegate.train(dataset, options),
      restore: (serialized) => delegate.restore(serialized),
    };
    const registry = new MovementBackendRegistry([custom]);
    expect(registry.get("custom")?.id).toBe("custom");
    expect(() => registry.require("missing")).toThrow(/Unknown movement model backend/);
  });
});
