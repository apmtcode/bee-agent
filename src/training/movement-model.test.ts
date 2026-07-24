import { describe, expect, it } from "vitest";
import type { ReplayTimelineEvent } from "../capture/replay.js";
import { buildTrajectorySpan } from "../capture/trajectory.js";
import {
  MarkovMovementBackend,
  MOVEMENT_END_TOKEN,
  evaluateMovementReplay,
  loadMovementModel,
  movementDatasetFromReplays,
  movementDatasetFromTrajectories,
  parseMovementToken,
  tokenizeAction,
  trainMovementModel,
  type MovementDataset,
  type MovementSequence,
} from "./movement-model.js";

function seq(...actions: Array<[string, string]>): MovementSequence {
  return actions.map(([tool, summary]) => tokenizeAction(tool, summary));
}

function actionEvent(ts: number, tool: string, summary: string): ReplayTimelineEvent {
  return { kind: "action", ts, trajectoryId: "traj", tool, summary };
}

describe("movement token codec", () => {
  it("round-trips tool + summary", () => {
    const token = tokenizeAction("device", "tapped submit");
    expect(parseMovementToken(token)).toEqual({ tool: "device", summary: "tapped submit" });
  });

  it("treats the terminal sentinel as unparseable", () => {
    expect(parseMovementToken(MOVEMENT_END_TOKEN)).toBeUndefined();
  });
});

describe("dataset builders", () => {
  it("extracts one action sequence per replay, ordered by ts", () => {
    const dataset = movementDatasetFromReplays([
      {
        events: [
          actionEvent(30, "device", "submit"),
          { kind: "observation", ts: 5, trajectoryId: "traj", source: "device", summary: "screen" },
          actionEvent(10, "device", "open"),
          { kind: "transcript", ts: 1, messageId: "m1", role: "user", content: "go" },
          actionEvent(20, "device", "type"),
        ],
      },
    ]);
    expect(dataset.sequences).toEqual([seq(["device", "open"], ["device", "type"], ["device", "submit"])]);
  });

  it("drops empty sequences (no actions)", () => {
    const dataset = movementDatasetFromReplays([
      { events: [{ kind: "observation", ts: 1, trajectoryId: "t", source: "device", summary: "s" }] },
    ]);
    expect(dataset.sequences).toEqual([]);
  });

  it("builds sequences from trajectory spans", () => {
    const trajectory = buildTrajectorySpan({
      id: "t1",
      sessionId: "s1",
      actions: [
        { kind: "action", tool: "device", summary: "open", ts: 2 },
        { kind: "action", tool: "device", summary: "close", ts: 9 },
      ],
    });
    const dataset = movementDatasetFromTrajectories([trajectory]);
    expect(dataset.sequences).toEqual([seq(["device", "open"], ["device", "close"])]);
  });
});

describe("MarkovMovementBackend — repeat recorded movements", () => {
  it("reproduces a recorded movement exactly from its first action", () => {
    const recorded = seq(["device", "open app"], ["device", "type query"], ["device", "tap submit"]);
    const model = trainMovementModel({ sequences: [recorded] }, { order: 2 });
    expect(model.generate([recorded[0]])).toEqual(recorded);
  });

  it("reproduces every recorded movement in a multi-sequence dataset", () => {
    const a = seq(["device", "open"], ["device", "search"], ["device", "select"]);
    const b = seq(["browser", "focus"], ["browser", "click"], ["browser", "scroll"]);
    const model = trainMovementModel({ sequences: [a, b] }, { order: 2 });
    const fidelity = evaluateMovementReplay(model, [a, b]);
    expect(fidelity.accuracy).toBe(1);
    expect(fidelity.failures).toEqual([]);
  });

  it("terminates generation via the learned end sentinel", () => {
    const recorded = seq(["device", "one"], ["device", "two"]);
    const model = trainMovementModel({ sequences: [recorded] }, { order: 2 });
    const generated = model.generate([recorded[0]], { maxSteps: 50 });
    expect(generated).toEqual(recorded);
    expect(generated).not.toContain(MOVEMENT_END_TOKEN);
  });
});

describe("MarkovMovementBackend — generalize to related movements", () => {
  it("replays the exact continuation at full order when the prefix is known", () => {
    // Two related flows that share a middle step but differ at the end.
    const flowA = seq(["ui", "open"], ["ui", "type"], ["ui", "submit"]);
    const flowB = seq(["ui", "focus"], ["ui", "type"], ["ui", "cancel"]);
    const model = trainMovementModel({ sequences: [flowA, flowB] }, { order: 2 });

    const prediction = model.predictNext(seq(["ui", "open"], ["ui", "type"]));
    expect(prediction).toBeDefined();
    // The order-2 context (open,type)->submit is known, so it replays literally here.
    expect(prediction?.token).toBe(tokenizeAction("ui", "submit"));
    expect(prediction?.backoffOrder).toBe(2);
  });

  it("falls back to a shorter context when the full-order prefix is novel", () => {
    const flowA = seq(["ui", "open"], ["ui", "type"], ["ui", "submit"]);
    const flowB = seq(["ui", "focus"], ["ui", "type"], ["ui", "cancel"]);
    const model = trainMovementModel({ sequences: [flowA, flowB] }, { order: 2 });

    // A brand-new leading action never seen before "type": order-2 context
    // (scroll,type) is unknown, so the model generalizes via the order-1 "type" context.
    const prediction = model.predictNext(seq(["ui", "scroll"], ["ui", "type"]));
    expect(prediction).toBeDefined();
    expect(prediction?.backoffOrder).toBe(1);
    const options = new Set(prediction?.alternatives.map((alt) => alt.token));
    expect(options.has(tokenizeAction("ui", "submit"))).toBe(true);
    expect(options.has(tokenizeAction("ui", "cancel"))).toBe(true);
  });

  it("returns undefined only when the model is untrained", () => {
    const empty = trainMovementModel({ sequences: [] });
    expect(empty.predictNext(seq(["ui", "open"]))).toBeUndefined();
  });
});

describe("snapshot persistence — the on-device transfer seam", () => {
  it("serializes and reloads to an identical model", () => {
    const dataset: MovementDataset = {
      sequences: [seq(["device", "open"], ["device", "type"], ["device", "submit"])],
    };
    const model = trainMovementModel(dataset, { order: 2 });
    const snapshot = model.serialize();

    const roundTripped = JSON.parse(JSON.stringify(snapshot));
    const reloaded = loadMovementModel(roundTripped);

    const context = seq(["device", "open"], ["device", "type"]);
    expect(reloaded.predictNext(context)).toEqual(model.predictNext(context));
    expect(reloaded.generate([dataset.sequences[0][0]])).toEqual(model.generate([dataset.sequences[0][0]]));
    expect(reloaded.backendId).toBe(new MarkovMovementBackend().id);
  });

  it("throws for an unknown backend id", () => {
    const snapshot = trainMovementModel({ sequences: [seq(["a", "b"])] }).serialize();
    expect(() => loadMovementModel({ ...snapshot, backendId: "nonexistent" })).toThrow(/no movement backend/);
  });

  it("is deterministic — same dataset yields byte-identical snapshots", () => {
    const dataset: MovementDataset = {
      sequences: [seq(["a", "1"], ["a", "2"]), seq(["b", "1"], ["b", "2"])],
    };
    const first = JSON.stringify(trainMovementModel(dataset, { order: 3 }).serialize());
    const second = JSON.stringify(trainMovementModel(dataset, { order: 3 }).serialize());
    expect(first).toBe(second);
  });
});

describe("evaluateMovementReplay — generalization harness", () => {
  it("measures replay fidelity on held-out related sequences", () => {
    // Train on repeated structure; hold out a sequence built from the same vocabulary.
    const train = [
      seq(["ui", "open"], ["ui", "type"], ["ui", "submit"]),
      seq(["ui", "open"], ["ui", "type"], ["ui", "submit"]),
    ];
    const model = trainMovementModel({ sequences: train }, { order: 2 });
    const heldOut = [seq(["ui", "open"], ["ui", "type"], ["ui", "submit"])];
    const fidelity = evaluateMovementReplay(model, heldOut);
    expect(fidelity.total).toBe(1);
    expect(fidelity.reproduced).toBe(1);
    expect(fidelity.accuracy).toBe(1);
  });

  it("reports failures when a held-out movement diverges from what was learned", () => {
    const model = trainMovementModel(
      { sequences: [seq(["ui", "open"], ["ui", "submit"])] },
      { order: 2 },
    );
    // Held-out flow shares the seed but diverges — the model cannot reproduce it.
    const heldOut = [seq(["ui", "open"], ["ui", "type"], ["ui", "cancel"])];
    const fidelity = evaluateMovementReplay(model, heldOut);
    expect(fidelity.reproduced).toBe(0);
    expect(fidelity.accuracy).toBe(0);
    expect(fidelity.failures).toHaveLength(1);
  });
});
