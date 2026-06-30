import { describe, expect, it } from "vitest";
import { buildTrajectorySpan, type TrajectoryAction, type TrajectorySpan } from "../capture/trajectory.js";
import {
  NgramMovementBackend,
  extractMovementDataset,
  replayMovements,
  type MovementEvent,
} from "./movement-model.js";

/** Synthetic event-stream generator: a repeatable "open → type → submit" flow,
 * with the concrete target varying per repetition so the grammar is shared but
 * the targets differ — exactly the shape generalization must handle. */
function syntheticSpan(id: string, formTarget: string, baseTs: number): TrajectorySpan {
  const actions: TrajectoryAction[] = [
    action("device", "tapped Compose", baseTs, { gesture: "tap", target: "Compose" }),
    action("device", `typed into ${formTarget}`, baseTs + 1200, { gesture: "type", target: formTarget }),
    action("device", "tapped Send", baseTs + 2500, { gesture: "tap", target: "Send" }),
  ];
  return buildTrajectorySpan({
    id,
    sessionId: `session-${id}`,
    captureTier: "app",
    actions,
    outcome: { status: "success", summary: "sent" },
  });
}

function action(
  tool: string,
  summary: string,
  ts: number,
  metadata: Record<string, unknown>,
): TrajectoryAction {
  return { kind: "action", tool, summary, ts, metadata };
}

function tokensOf(events: MovementEvent[]): string[] {
  return events.map((event) => `${event.gesture}:${event.target ?? ""}`);
}

describe("extractMovementDataset", () => {
  it("derives ordered movement sequences with inter-event dt from trajectory actions", () => {
    const dataset = extractMovementDataset([syntheticSpan("a", "Subject", 1000)]);
    expect(dataset.sequences).toHaveLength(1);
    const events = dataset.sequences[0]!.events;
    expect(events.map((e) => e.gesture)).toEqual(["tap", "type", "tap"]);
    expect(events[0]!.dt).toBe(0);
    expect(events[1]!.dt).toBe(1200);
    expect(events[2]!.dt).toBe(1300);
  });

  it("skips trajectories with no actions and sorts actions by timestamp", () => {
    const span = buildTrajectorySpan({
      id: "unsorted",
      sessionId: "s",
      actions: [
        action("device", "second", 200, { gesture: "tap", target: "B" }),
        action("device", "first", 100, { gesture: "tap", target: "A" }),
      ],
    });
    const empty = buildTrajectorySpan({ id: "empty", sessionId: "s" });
    const dataset = extractMovementDataset([span, empty]);
    expect(dataset.sequences).toHaveLength(1);
    expect(dataset.sequences[0]!.events.map((e) => e.target)).toEqual(["A", "B"]);
  });
});

describe("NgramMovementBackend", () => {
  it("repeats the dominant recorded movement sequence via autoregressive replay", async () => {
    const dataset = extractMovementDataset([
      syntheticSpan("a", "Subject", 1000),
      syntheticSpan("b", "Subject", 9000),
    ]);
    const model = await new NgramMovementBackend().train(dataset);
    const replayed = replayMovements(model, { steps: 3 });
    expect(tokensOf(replayed)).toEqual(["tap:Compose", "type:Subject", "tap:Send"]);
  });

  it("predicts the learned continuation from a seed (exact match)", async () => {
    const dataset = extractMovementDataset([syntheticSpan("a", "Subject", 1000), syntheticSpan("b", "Subject", 9000)]);
    const model = await new NgramMovementBackend().train(dataset);
    const seed = dataset.sequences[0]!.events.slice(0, 1); // just "tap Compose"
    const prediction = model.predictNext({ history: seed });
    expect(prediction?.event.gesture).toBe("type");
    expect(prediction?.event.target).toBe("Subject");
    expect(prediction?.rationale).toBe("exact");
    expect(prediction?.confidence).toBeGreaterThan(0);
  });

  it("generalizes a learned gesture onto a novel-but-related target", async () => {
    const dataset = extractMovementDataset([syntheticSpan("a", "Subject", 1000), syntheticSpan("b", "Subject", 9000)]);
    const model = await new NgramMovementBackend().train(dataset);
    const seed = dataset.sequences[0]!.events.slice(0, 1);
    // The new scene has no "Subject" field, but offers a related "Subject line".
    const prediction = model.predictNext({ history: seed, availableTargets: ["Subject line", "Attachment"] });
    expect(prediction?.event.gesture).toBe("type");
    expect(prediction?.event.target).toBe("Subject line");
    expect(prediction?.rationale).toBe("generalized");
    expect(prediction?.event.summary).toContain("Subject line");
  });

  it("falls back to the unconditional prior when no context matches", async () => {
    const dataset = extractMovementDataset([syntheticSpan("a", "Subject", 1000)]);
    const model = await new NgramMovementBackend().train(dataset);
    const prediction = model.predictNext({
      history: [{ tool: "device", gesture: "shortcut", summary: "unseen", dt: 0 }],
    });
    expect(prediction).toBeDefined();
    expect(["prior", "backoff"]).toContain(prediction?.rationale);
  });

  it("round-trips through serialize/load with identical predictions", async () => {
    const backend = new NgramMovementBackend();
    const dataset = extractMovementDataset([syntheticSpan("a", "Subject", 1000), syntheticSpan("b", "Subject", 9000)]);
    const trained = await backend.train(dataset);
    const serialized = JSON.parse(JSON.stringify(trained.serialize()));
    const loaded = backend.load(serialized);

    const seed = dataset.sequences[0]!.events.slice(0, 1);
    expect(loaded.predictNext({ history: seed })?.event).toEqual(trained.predictNext({ history: seed })?.event);
    expect(tokensOf(replayMovements(loaded, { steps: 3 }))).toEqual(tokensOf(replayMovements(trained, { steps: 3 })));
  });

  it("rejects loading a foreign serialized model", () => {
    expect(() => new NgramMovementBackend().load({ backendId: "other" })).toThrow(/ngram-movement/u);
  });

  it("returns undefined when trained on an empty dataset", async () => {
    const model = await new NgramMovementBackend().train({ version: 1, sequences: [] });
    expect(model.predictNext({ history: [] })).toBeUndefined();
    expect(replayMovements(model, { steps: 5 })).toEqual([]);
  });
});
