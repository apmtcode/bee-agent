import { describe, expect, it } from "vitest";
import type { ReplayManifest, ReplayTimelineEvent } from "../capture/replay.js";
import {
  DeterministicNearestNeighborBackend,
  buildMovementExamplesFromReplay,
  buildMovementExamplesFromReplays,
  evaluateMovementModel,
  featurize,
  type MovementExample,
  type MovementModelBackend,
} from "./movement-model.js";

/** Synthetic replay builder — stands in for real OS-captured movement streams. */
function synthReplay(
  sessionId: string,
  steps: Array<{ obs: { source: string; summary: string }; act: { tool: string; summary: string } }>,
): ReplayManifest {
  const events: ReplayTimelineEvent[] = [];
  let ts = 1;
  const trajectoryId = `${sessionId}-traj`;
  for (const step of steps) {
    events.push({ kind: "observation", ts: ts++, trajectoryId, source: step.obs.source, summary: step.obs.summary });
    events.push({ kind: "action", ts: ts++, trajectoryId, tool: step.act.tool, summary: step.act.summary });
  }
  return {
    version: 1,
    sessionId,
    trajectoryIds: [trajectoryId],
    eventCount: events.length,
    events,
  };
}

const LOGIN_FLOW = synthReplay("login", [
  { obs: { source: "screen", summary: "login window focused with username field" }, act: { tool: "click", summary: "click username field" } },
  { obs: { source: "screen", summary: "username field active cursor blinking" }, act: { tool: "type", summary: "type user name" } },
  { obs: { source: "screen", summary: "password field visible below username" }, act: { tool: "click", summary: "click password field" } },
  { obs: { source: "screen", summary: "submit button enabled ready" }, act: { tool: "click", summary: "click submit button" } },
]);

describe("buildMovementExamplesFromReplay", () => {
  it("emits one example per action with the preceding window as context", () => {
    const examples = buildMovementExamplesFromReplay(LOGIN_FLOW);
    expect(examples).toHaveLength(4);
    // First action sees the first observation and no prior actions.
    expect(examples[0]?.context.observations.map((o) => o.summary)).toContain("login window focused with username field");
    expect(examples[0]?.context.recentActions).toHaveLength(0);
    expect(examples[0]?.target).toEqual({ tool: "click", summary: "click username field" });
    // Later actions carry prior actions as context.
    expect(examples[3]?.context.recentActions.map((a) => a.tool)).toEqual(["click", "type", "click"]);
  });

  it("respects the context window size", () => {
    const [example] = buildMovementExamplesFromReplay(LOGIN_FLOW, { contextWindow: 1 }).slice(-1);
    expect(example?.context.observations).toHaveLength(1);
    expect(example?.context.recentActions).toHaveLength(1);
  });

  it("includes transcript events as observations", () => {
    const manifest: ReplayManifest = {
      version: 1,
      sessionId: "chat",
      trajectoryIds: ["chat-traj"],
      eventCount: 2,
      events: [
        { kind: "transcript", ts: 1, messageId: "m1", role: "user", content: "please open settings" },
        { kind: "action", ts: 2, trajectoryId: "chat-traj", tool: "open", summary: "open settings panel" },
      ],
    };
    const [example] = buildMovementExamplesFromReplay(manifest);
    expect(example?.context.observations[0]).toEqual({ source: "transcript:user", summary: "please open settings" });
  });
});

describe("DeterministicNearestNeighborBackend", () => {
  it("(objective 2c) repeats recorded movements exactly", () => {
    const model = new DeterministicNearestNeighborBackend().train(buildMovementExamplesFromReplay(LOGIN_FLOW));
    const examples = buildMovementExamplesFromReplay(LOGIN_FLOW);
    for (const example of examples) {
      const prediction = model.predict(example.context);
      expect(prediction?.tool).toBe(example.target.tool);
      expect(prediction?.summary).toBe(example.target.summary);
      expect(prediction?.source).toBe("recall");
      expect(prediction?.confidence).toBe(1);
    }
  });

  it("(objective 2d) generalizes to related but unseen movements", () => {
    const model = new DeterministicNearestNeighborBackend().train(buildMovementExamplesFromReplay(LOGIN_FLOW));
    // A paraphrased, never-seen context that is closest to the "submit button" step.
    const prediction = model.predict({
      observations: [{ source: "screen", summary: "the submit button is now enabled and ready to press" }],
      recentActions: [{ tool: "click", summary: "click password field" }],
    });
    expect(prediction).toBeDefined();
    expect(prediction?.tool).toBe("click");
    expect(prediction?.summary).toBe("click submit button");
    expect(prediction?.source).toBe("generalize");
    expect(prediction?.confidence).toBeGreaterThan(0);
    expect(prediction?.confidence).toBeLessThan(1);
  });

  it("abstains when nothing in the dataset is related", () => {
    const model = new DeterministicNearestNeighborBackend().train(buildMovementExamplesFromReplay(LOGIN_FLOW));
    const prediction = model.predict({
      observations: [{ source: "audio", summary: "xylophone quartz zephyr" }],
      recentActions: [],
    });
    expect(prediction).toBeUndefined();
  });

  it("is deterministic across repeated training and prediction", () => {
    const examples = buildMovementExamplesFromReplay(LOGIN_FLOW);
    const backend = new DeterministicNearestNeighborBackend();
    const context = examples[2]!.context;
    const a = backend.train(examples).predict(context);
    const b = backend.train(examples).predict(context);
    expect(a).toEqual(b);
  });

  it("exposes vocabulary and example count metadata", () => {
    const model = new DeterministicNearestNeighborBackend().train(buildMovementExamplesFromReplay(LOGIN_FLOW));
    const meta = model.metadata();
    expect(meta.backend).toBe("deterministic-nearest-neighbor");
    expect(meta.exampleCount).toBe(4);
    expect(meta.vocabulary).toEqual(["click", "type"]);
  });
});

describe("evaluateMovementModel", () => {
  it("scores perfect recall on the training set", () => {
    const examples = buildMovementExamplesFromReplay(LOGIN_FLOW);
    const model = new DeterministicNearestNeighborBackend().train(examples);
    const result = evaluateMovementModel(model, examples);
    expect(result.total).toBe(4);
    expect(result.correct).toBe(4);
    expect(result.accuracy).toBe(1);
    expect(result.recalled).toBe(4);
    expect(result.generalized).toBe(0);
  });

  it("measures generalization on a held-out related trajectory", () => {
    const train = buildMovementExamplesFromReplays([LOGIN_FLOW]);
    const model = new DeterministicNearestNeighborBackend().train(train);
    // Held-out flow: same task, paraphrased observations.
    const heldOut: MovementExample[] = [
      {
        context: { observations: [{ source: "screen", summary: "login dialog focused with the username input" }], recentActions: [] },
        target: { tool: "click", summary: "click username field" },
      },
      {
        context: { observations: [{ source: "screen", summary: "submit button now enabled and ready" }], recentActions: [{ tool: "click", summary: "click password field" }] },
        target: { tool: "click", summary: "click submit button" },
      },
    ];
    const result = evaluateMovementModel(model, heldOut);
    expect(result.total).toBe(2);
    expect(result.generalized).toBe(2);
    expect(result.accuracy).toBe(1);
  });

  it("returns zeroed metrics for an empty eval set", () => {
    const model = new DeterministicNearestNeighborBackend().train([]);
    expect(evaluateMovementModel(model, [])).toEqual({
      total: 0,
      correct: 0,
      accuracy: 0,
      recalled: 0,
      generalized: 0,
      abstained: 0,
    });
  });
});

describe("pluggability", () => {
  it("accepts any backend implementing the MovementModelBackend seam", () => {
    // A trivial alternative backend proves callers depend only on the interface.
    const constantBackend: MovementModelBackend = {
      name: "constant",
      train(examples) {
        const target = examples[0]?.target ?? { tool: "noop", summary: "noop" };
        return {
          backend: "constant",
          exampleCount: examples.length,
          predict: () => ({ tool: target.tool, summary: target.summary, confidence: 0.5, source: "generalize" }),
          metadata: () => ({ backend: "constant", exampleCount: examples.length, vocabulary: [target.tool] }),
        };
      },
    };
    const model = constantBackend.train(buildMovementExamplesFromReplay(LOGIN_FLOW));
    expect(model.predict({ observations: [], recentActions: [] })?.tool).toBe("click");
    expect(model.backend).toBe("constant");
  });
});

describe("featurize", () => {
  it("namespaces observation and action tokens so they don't collide", () => {
    const tokens = featurize({
      observations: [{ source: "screen", summary: "open panel" }],
      recentActions: [{ tool: "open", summary: "open panel" }],
    });
    expect(tokens.has("obs:screen:open")).toBe(true);
    expect(tokens.has("act:open")).toBe(true);
    expect(tokens.has("act:tool:open")).toBe(true);
  });
});
