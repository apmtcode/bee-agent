import { describe, expect, it } from "vitest";
import type { ExportedReplayManifest } from "./export-manifest.js";
import {
  MarkovMovementBackend,
  buildMovementDataset,
  evaluateNextTokenAccuracy,
  loadMovementModel,
  tokenizeMovementEvent,
  tokenizeMovementEvents,
  type MovementEvent,
} from "./model-backend.js";

function actionEvent(tool: string, ts: number): MovementEvent {
  return { kind: "action", ts, trajectoryId: "t1", tool, summary: `${tool} ${ts}` };
}

function observationEvent(source: string, ts: number): MovementEvent {
  return { kind: "observation", ts, trajectoryId: "t1", source, summary: `${source} ${ts}` };
}

/** A repeatable "open editor → type → save" style movement, tokenized. */
function editorReplay(sessionId: string, offset = 0): { sessionId: string; events: MovementEvent[] } {
  return {
    sessionId,
    events: [
      observationEvent("screen", offset + 1),
      actionEvent("Focus Window", offset + 2),
      actionEvent("KeyPress", offset + 3),
      actionEvent("KeyPress", offset + 4),
      actionEvent("Save", offset + 5),
    ],
  };
}

describe("tokenizeMovementEvent", () => {
  it("derives compact, normalized tokens per event kind", () => {
    expect(tokenizeMovementEvent(actionEvent("Focus Window", 1))).toBe("action:focus-window");
    expect(tokenizeMovementEvent(observationEvent("Screen", 1))).toBe("observation:screen");
    expect(
      tokenizeMovementEvent({ kind: "transcript", ts: 1, messageId: "m1", role: "assistant", content: "hi" }),
    ).toBe("transcript:assistant");
  });

  it("sorts events chronologically before tokenizing", () => {
    const tokens = tokenizeMovementEvents([actionEvent("Save", 5), actionEvent("Open", 1)]);
    expect(tokens).toEqual(["action:open", "action:save"]);
  });
});

describe("buildMovementDataset", () => {
  it("builds a sorted vocabulary and drops empty replays", () => {
    const dataset = buildMovementDataset([editorReplay("s1"), { sessionId: "empty", events: [] }]);
    expect(dataset.examples).toHaveLength(1);
    expect(dataset.examples[0]!.sessionId).toBe("s1");
    expect(dataset.vocabulary).toEqual([...dataset.vocabulary].sort());
    expect(dataset.vocabulary).toContain("action:save");
  });
});

describe("MarkovMovementBackend", () => {
  it("reproduces a recorded movement from its start (repeat)", async () => {
    const dataset = buildMovementDataset([editorReplay("s1")]);
    const model = await new MarkovMovementBackend().train(dataset, { order: 2 });
    const produced = model.generate();
    expect(produced).toEqual(dataset.examples[0]!.tokens);
  });

  it("is deterministic: same dataset + config yields identical output", async () => {
    const dataset = buildMovementDataset([editorReplay("s1"), editorReplay("s2", 100)]);
    const backend = new MarkovMovementBackend();
    const a = await backend.train(dataset, { order: 2 });
    const b = await backend.train(dataset, { order: 2 });
    expect(a.generate()).toEqual(b.generate());
    expect(a.serialize()).toEqual(b.serialize());
  });

  it("predicts the learned next token and scores it as a probability", async () => {
    const dataset = buildMovementDataset([editorReplay("s1"), editorReplay("s2", 50)]);
    const model = await new MarkovMovementBackend().train(dataset, { order: 2 });
    expect(model.predictNext(["observation:screen"])).toBe("action:focus-window");
    const score = model.scoreNext(["observation:screen"], "action:focus-window");
    expect(score).toBeGreaterThan(0);
    expect(score).toBeLessThanOrEqual(1);
  });

  it("generalizes: rolls a novel-but-related seed forward via learned transitions", async () => {
    // Train on two flows sharing a common tail (KeyPress → Save).
    const training = buildMovementDataset([
      {
        sessionId: "a",
        events: [actionEvent("Focus Window", 1), actionEvent("KeyPress", 2), actionEvent("Save", 3)],
      },
      {
        sessionId: "b",
        events: [actionEvent("Open Menu", 1), actionEvent("KeyPress", 2), actionEvent("Save", 3)],
      },
    ]);
    const model = await new MarkovMovementBackend().train(training, { order: 1 });
    // A seed never observed as a full sequence still completes with the learned tail.
    const produced = model.generate({ seed: ["action:keypress"] });
    expect(produced).toEqual(["action:keypress", "action:save"]);
  });

  it("scores generalization with a held-out next-token accuracy eval", async () => {
    const model = await new MarkovMovementBackend().train(
      buildMovementDataset([editorReplay("s1"), editorReplay("s2", 20), editorReplay("s3", 40)]),
      { order: 2 },
    );
    const heldOut = tokenizeMovementEvents(editorReplay("held", 999).events);
    const report = evaluateNextTokenAccuracy(model, [heldOut]);
    expect(report.total).toBe(heldOut.length);
    // The held-out sequence is structurally identical, so the model nails it.
    expect(report.accuracy).toBe(1);
  });

  it("returns an empty accuracy report for no sequences", async () => {
    const model = await new MarkovMovementBackend().train(buildMovementDataset([editorReplay("s1")]));
    expect(evaluateNextTokenAccuracy(model, [])).toEqual({ total: 0, correct: 0, accuracy: 0 });
  });

  it("round-trips through serialize/loadMovementModel", async () => {
    const dataset = buildMovementDataset([editorReplay("s1"), editorReplay("s2", 30)]);
    const model = await new MarkovMovementBackend().train(dataset, { order: 3 });
    const restored = loadMovementModel(JSON.parse(JSON.stringify(model.serialize())));
    expect(restored.order).toBe(3);
    expect(restored.backend).toBe("markov-movement");
    expect(restored.generate()).toEqual(model.generate());
    expect(restored.predictNext(["observation:screen"])).toBe(model.predictNext(["observation:screen"]));
  });

  it("consumes exported replay manifests directly", async () => {
    const manifest: Pick<ExportedReplayManifest, "sessionId" | "events"> = {
      sessionId: "exported",
      events: [
        { kind: "observation", ts: 1, trajectoryId: "t", source: "screen", summary: "s" },
        { kind: "action", ts: 2, trajectoryId: "t", tool: "Click", summary: "c" },
      ],
    };
    const dataset = buildMovementDataset([manifest]);
    const model = await new MarkovMovementBackend().train(dataset);
    expect(model.generate()).toEqual(["observation:screen", "action:click"]);
  });
});
