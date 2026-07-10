import { describe, expect, it } from "vitest";
import { buildTrajectorySpan, type TrajectorySpan } from "../capture/trajectory.js";
import {
  buildMovementDataset,
  evaluateMovementModel,
  movementActionKey,
  NgramMovementBackend,
  type LocalMovementModelBackend,
  type MovementDataset,
} from "./movement-model.js";

function span(id: string, events: Array<["obs" | "act", string, string, number]>): TrajectorySpan {
  const observations = events
    .filter(([kind]) => kind === "obs")
    .map(([, source, summary, ts]) => ({ kind: "observation" as const, source, summary, ts }));
  const actions = events
    .filter(([kind]) => kind === "act")
    .map(([, tool, summary, ts]) => ({ kind: "action" as const, tool, summary, ts }));
  return buildTrajectorySpan({ id, sessionId: "s1", observations, actions });
}

describe("buildMovementDataset", () => {
  it("derives one example per action with prior events as context, ordered by ts", () => {
    const trajectory = span("t1", [
      ["obs", "screen", "login form visible", 1],
      ["act", "mouse.click", "click username field", 2],
      ["act", "keyboard.type", "type user@example.com", 3],
    ]);

    const dataset = buildMovementDataset([trajectory]);

    expect(dataset.examples).toHaveLength(2);
    expect(dataset.examples[0]).toMatchObject({
      context: ["obs:screen"],
      action: { tool: "mouse.click", summary: "click username field" },
    });
    expect(dataset.examples[1]).toMatchObject({
      context: ["obs:screen", "act:mouse.click"],
      action: { tool: "keyboard.type", summary: "type user@example.com" },
    });
  });

  it("truncates context to the configured window", () => {
    const trajectory = span("t1", [
      ["obs", "a", "a", 1],
      ["obs", "b", "b", 2],
      ["obs", "c", "c", 3],
      ["act", "do", "x", 4],
    ]);
    const dataset = buildMovementDataset([trajectory], { contextWindow: 2 });
    expect(dataset.examples[0]?.context).toEqual(["obs:b", "obs:c"]);
  });

  it("weights examples by trajectory reward", () => {
    const trajectory: TrajectorySpan = {
      ...span("t1", [["act", "do", "x", 1]]),
      outcome: { status: "success", summary: "ok", reward: 1 },
    };
    const dataset = buildMovementDataset([trajectory]);
    expect(dataset.examples[0]?.weight).toBe(2); // 1 + reward
  });
});

describe("NgramMovementBackend — repeat recorded movements", () => {
  it("reproduces the recorded next action for a seen context (exact match)", async () => {
    const dataset = buildMovementDataset([
      span("t1", [
        ["obs", "screen", "editor open", 1],
        ["act", "mouse.click", "click run button", 2],
      ]),
    ]);
    const backend: LocalMovementModelBackend = new NgramMovementBackend();
    const model = await backend.train(dataset, { contextWindow: 4 });

    const prediction = model.predict(["obs:screen"]);
    expect(prediction.action).toEqual({ tool: "mouse.click", summary: "click run button" });
    expect(prediction.source).toBe("exact");
    expect(prediction.confidence).toBe(1);
  });

  it("picks the most frequent action when a context maps to several", async () => {
    const dataset: MovementDataset = {
      version: 1,
      examples: [
        { context: ["obs:menu"], action: { tool: "click", summary: "file" } },
        { context: ["obs:menu"], action: { tool: "click", summary: "file" } },
        { context: ["obs:menu"], action: { tool: "click", summary: "edit" } },
      ],
    };
    const model = await new NgramMovementBackend().train(dataset, { contextWindow: 2 });
    const prediction = model.predict(["obs:menu"]);
    expect(prediction.action?.summary).toBe("file");
    expect(prediction.confidence).toBeCloseTo(2 / 3);
  });

  it("backs off to a shorter suffix when the full context is unseen", async () => {
    const dataset: MovementDataset = {
      version: 1,
      examples: [{ context: ["obs:a", "act:step"], action: { tool: "commit", summary: "save" } }],
    };
    const model = await new NgramMovementBackend().train(dataset, { contextWindow: 4 });
    const prediction = model.predict(["obs:unseen", "act:step"]);
    expect(prediction.action?.tool).toBe("commit");
    expect(prediction.source).toBe("backoff");
    expect(prediction.matchedContextLength).toBe(1);
  });
});

describe("NgramMovementBackend — generalize to related movements", () => {
  it("uses token-overlap similarity for a new-but-related context", async () => {
    const dataset: MovementDataset = {
      version: 1,
      examples: [
        {
          context: ["obs:browser", "act:scroll", "act:focus"],
          action: { tool: "keyboard.type", summary: "enter search" },
        },
      ],
    };
    const model = await new NgramMovementBackend().train(dataset, { contextWindow: 5 });

    // Novel ordering/extra token, but shares most tokens with the recording.
    const prediction = model.predict(["act:focus", "obs:browser", "act:scroll"]);
    expect(prediction.action?.tool).toBe("keyboard.type");
    expect(prediction.source).toBe("similar");
    expect(prediction.confidence).toBeGreaterThanOrEqual(0.34);
  });

  it("falls back to the global prior when nothing is related", async () => {
    const dataset: MovementDataset = {
      version: 1,
      examples: [
        { context: ["obs:x"], action: { tool: "a", summary: "1" } },
        { context: ["obs:y"], action: { tool: "a", summary: "1" } },
        { context: ["obs:z"], action: { tool: "b", summary: "2" } },
      ],
    };
    const model = await new NgramMovementBackend().train(dataset, {
      contextWindow: 2,
      similarityThreshold: 0.9,
    });
    const prediction = model.predict(["totally:unrelated"]);
    expect(prediction.source).toBe("prior");
    expect(prediction.action?.tool).toBe("a"); // most frequent overall
  });

  it("returns a 'none' prediction for an empty dataset", async () => {
    const model = await new NgramMovementBackend().train({ version: 1, examples: [] }, { contextWindow: 2 });
    const prediction = model.predict(["obs:x"]);
    expect(prediction.source).toBe("none");
    expect(prediction.action).toBeUndefined();
    expect(prediction.confidence).toBe(0);
  });
});

describe("determinism & pluggability", () => {
  it("produces identical predictions across repeated training runs", async () => {
    const dataset: MovementDataset = {
      version: 1,
      examples: [
        { context: ["obs:a"], action: { tool: "x", summary: "1" } },
        { context: ["obs:a"], action: { tool: "y", summary: "2" } },
      ],
    };
    const backend = new NgramMovementBackend();
    const a = (await backend.train(dataset, { contextWindow: 2 })).predict(["obs:a"]);
    const b = (await backend.train(dataset, { contextWindow: 2 })).predict(["obs:a"]);
    expect(a).toEqual(b);
  });

  it("breaks weight ties deterministically by action key", async () => {
    const dataset: MovementDataset = {
      version: 1,
      examples: [
        { context: ["obs:a"], action: { tool: "zzz", summary: "z" } },
        { context: ["obs:a"], action: { tool: "aaa", summary: "a" } },
      ],
    };
    const model = await new NgramMovementBackend().train(dataset, { contextWindow: 2 });
    // Equal weight → lexicographically smallest action key wins ("aaa a").
    expect(model.predict(["obs:a"]).action?.tool).toBe("aaa");
  });

  it("exposes an inspectable summary via describe()", async () => {
    const dataset = buildMovementDataset([
      span("t1", [
        ["obs", "screen", "s", 1],
        ["act", "click", "c", 2],
      ]),
    ]);
    const summary = (await new NgramMovementBackend().train(dataset, { contextWindow: 3 })).describe();
    expect(summary.backendId).toBe("ngram-mock-v1");
    expect(summary.exampleCount).toBe(1);
    expect(summary.distinctActions).toBe(1);
  });
});

describe("evaluateMovementModel", () => {
  it("scores perfect replay fidelity on the training set", async () => {
    const dataset = buildMovementDataset([
      span("t1", [
        ["obs", "screen", "open", 1],
        ["act", "click", "run", 2],
        ["act", "type", "hello", 3],
      ]),
    ]);
    const model = await new NgramMovementBackend().train(dataset, { contextWindow: 4 });
    const evaluation = evaluateMovementModel(model, dataset);
    expect(evaluation.total).toBe(2);
    expect(evaluation.accuracy).toBe(1);
    expect(evaluation.bySource.exact + evaluation.bySource.backoff).toBe(2);
  });

  it("measures generalization on a held-out but related trajectory", async () => {
    // Train on one login flow, evaluate on a structurally similar one.
    const train = buildMovementDataset([
      span("train", [
        ["obs", "login", "form", 1],
        ["act", "mouse.click", "focus username", 2],
        ["act", "keyboard.type", "enter username", 3],
        ["act", "mouse.click", "focus password", 4],
        ["act", "keyboard.type", "enter password", 5],
      ]),
    ]);
    const heldOut = buildMovementDataset([
      span("test", [
        ["obs", "login", "form", 1],
        ["act", "mouse.click", "focus username", 2],
        ["act", "keyboard.type", "enter username", 3],
      ]),
    ]);
    const model = await new NgramMovementBackend().train(train, { contextWindow: 6 });
    const evaluation = evaluateMovementModel(model, heldOut);
    // Same-prefix contexts should reproduce the recorded movements exactly.
    expect(evaluation.toolAccuracy).toBe(1);
    expect(evaluation.accuracy).toBe(1);
  });
});

describe("movementActionKey", () => {
  it("is stable and combines tool + summary", () => {
    expect(movementActionKey({ tool: "click", summary: "run" })).toBe("click run");
  });
});
