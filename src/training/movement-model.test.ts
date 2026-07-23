import { describe, expect, it } from "vitest";
import {
  MarkovMovementBackend,
  MarkovMovementModel,
  buildMovementDataset,
  evaluateMovementModel,
  tokenizeMovementEvent,
  type MovementTrainingDataset,
  type TokenizableReplayManifest,
} from "./movement-model.js";

function actionEvent(trajectoryId: string, ts: number, tool: string, summary?: string) {
  return { kind: "action" as const, ts, trajectoryId, tool, summary };
}

function replay(trajectoryId: string, tools: string[]): TokenizableReplayManifest {
  return {
    trajectoryIds: [trajectoryId],
    events: tools.map((tool, index) => actionEvent(trajectoryId, index + 1, tool)),
  };
}

describe("tokenizeMovementEvent", () => {
  it("tokenizes action events to their tool by default", () => {
    expect(tokenizeMovementEvent(actionEvent("t", 1, "mouse.move"))).toBe("mouse.move");
  });

  it("ignores non-action events unless included", () => {
    const observation = { kind: "observation" as const, ts: 1, trajectoryId: "t", source: "screen", summary: "x" };
    expect(tokenizeMovementEvent(observation)).toBeUndefined();
    expect(tokenizeMovementEvent(observation, { includeKinds: ["observation"] })).toBe("obs:screen");
  });

  it("appends a summary slug when requested", () => {
    const token = tokenizeMovementEvent(actionEvent("t", 1, "key.press", "Cmd + Shift + P"), {
      includeSummary: true,
    });
    expect(token).toBe("key.press#cmd-shift-p");
  });
});

describe("buildMovementDataset", () => {
  it("groups events by trajectory, orders by ts, and derives a sorted vocabulary", () => {
    const dataset = buildMovementDataset([
      { trajectoryIds: ["b"], events: [actionEvent("b", 2, "b.click"), actionEvent("b", 1, "b.move")] },
      { trajectoryIds: ["a"], events: [actionEvent("a", 1, "a.move"), actionEvent("a", 2, "a.click")] },
    ]);
    // Trajectories are emitted in sorted id order for determinism.
    expect(dataset.sequences).toEqual([
      ["a.move", "a.click"],
      ["b.move", "b.click"],
    ]);
    expect(dataset.vocabulary).toEqual(["a.click", "a.move", "b.click", "b.move"]);
  });

  it("is deterministic: identical input yields identical output", () => {
    const manifests = [replay("t1", ["m.move", "m.click"]), replay("t2", ["k.press"])];
    expect(buildMovementDataset(manifests)).toEqual(buildMovementDataset(manifests));
  });
});

describe("MarkovMovementBackend", () => {
  const backend = new MarkovMovementBackend();

  it("repeats a recorded movement exactly from a known context", async () => {
    const dataset = buildMovementDataset([
      replay("t1", ["mouse.move", "mouse.down", "mouse.move", "mouse.up"]),
    ]);
    const model = await backend.train(dataset, { order: 2 });

    // After [mouse.move, mouse.down] the only recorded next move is mouse.move.
    const prediction = model.predictNext(["mouse.move", "mouse.down"]);
    expect(prediction?.token).toBe("mouse.move");
    expect(prediction?.backedOff).toBe(false);
    expect(prediction?.probability).toBe(1);
  });

  it("generates a full recorded trajectory deterministically", async () => {
    const dataset = buildMovementDataset([
      replay("t1", ["app.focus", "menu.open", "menu.select", "dialog.confirm"]),
    ]);
    const model = await backend.train(dataset, { order: 2 });

    const rollout = model.generate(["app.focus"], { maxLength: 5, stopTokens: ["dialog.confirm"] });
    expect(rollout).toEqual(["menu.open", "menu.select", "dialog.confirm"]);
  });

  it("generalizes to a new-but-related movement via back-off", async () => {
    // Two related gestures: both drag then release, but reached via different openings.
    const dataset = buildMovementDataset([
      replay("t1", ["win.focus", "drag.start", "drag.move", "drag.end"]),
      replay("t2", ["tab.focus", "drag.start", "drag.move", "drag.end"]),
    ]);
    const model = await backend.train(dataset, { order: 2 });

    // Context [pane.focus, drag.start] was NEVER recorded (novel opening), but
    // the model backs off to the shared [drag.start] statistics and still
    // predicts the related continuation.
    const prediction = model.predictNext(["pane.focus", "drag.start"]);
    expect(prediction?.token).toBe("drag.move");
    expect(prediction?.backedOff).toBe(true);
  });

  it("breaks ties deterministically by frequency then lexical order", async () => {
    const dataset: MovementTrainingDataset = {
      version: 1,
      // From context [x]: 'b.after' appears twice, 'a.after' once → higher freq wins.
      sequences: [
        ["x", "b.after"],
        ["x", "b.after"],
        ["x", "a.after"],
      ],
      vocabulary: ["a.after", "b.after", "x"],
    };
    const model = await backend.train(dataset, { order: 1 });
    expect(model.predictNext(["x"])?.token).toBe("b.after");
  });

  it("round-trips through JSON serialization", async () => {
    const dataset = buildMovementDataset([replay("t1", ["a", "b", "c", "a", "b"])]);
    const model = (await backend.train(dataset, { order: 2 })) as MarkovMovementModel;
    const restored = MarkovMovementModel.fromJSON(model.toJSON());

    expect(restored.toJSON()).toEqual(model.toJSON());
    expect(restored.predictNext(["a", "b"])?.token).toBe(model.predictNext(["a", "b"])?.token);
  });

  it("returns undefined when the model has no data at all", async () => {
    const model = await backend.train({ version: 1, sequences: [], vocabulary: [] }, { order: 2 });
    expect(model.predictNext(["anything"])).toBeUndefined();
    expect(model.generate(["seed"])).toEqual([]);
  });
});

describe("evaluateMovementModel", () => {
  it("scores perfect recall on the training sequence", async () => {
    const backend = new MarkovMovementBackend();
    const training = buildMovementDataset([replay("t1", ["a", "b", "c", "d"])]);
    const model = (await backend.train(training, { order: 2 })) as MarkovMovementModel;

    const evaluation = evaluateMovementModel(model, training.sequences);
    expect(evaluation.predictionCount).toBe(4);
    expect(evaluation.top1Accuracy).toBe(1);
    // Near-1 perplexity: every step is certain except the first-token prior.
    expect(evaluation.perplexity).toBeGreaterThanOrEqual(1);
    expect(evaluation.perplexity).toBeLessThan(1.5);
  });

  it("measures partial generalization on a held-out related sequence", async () => {
    const backend = new MarkovMovementBackend();
    const training = buildMovementDataset([
      replay("t1", ["home", "search", "open", "read"]),
      replay("t2", ["home", "search", "open", "read"]),
    ]);
    const model = (await backend.train(training, { order: 2 })) as MarkovMovementModel;

    // Held-out sequence shares primitives but starts from a novel token.
    const heldOut = [["landing", "search", "open", "read"]];
    const evaluation = evaluateMovementModel(model, heldOut);
    expect(evaluation.predictionCount).toBe(4);
    // "search"->"open"->"read" are recovered via back-off despite the novel start.
    expect(evaluation.top1Accuracy).toBeGreaterThanOrEqual(0.5);
    expect(evaluation.backoffRate).toBeGreaterThan(0);
  });
});
