import { describe, expect, it } from "vitest";
import { buildTrajectorySpan, type TrajectoryAction, type TrajectorySpan } from "../capture/trajectory.js";
import {
  MarkovMovementBackend,
  buildMovementDataset,
  buildSamplesFromSequences,
  evaluateMovementModel,
  rolloutMovements,
  tokenizeAction,
  trajectoryToMovementTokens,
} from "./movement-model.js";

type Gesture = { kind: string; target?: string; direction?: string };

function gestureAction(ts: number, gesture: Gesture): TrajectoryAction {
  return {
    kind: "action",
    tool: "device",
    summary: `gesture ${gesture.kind}`,
    ts,
    metadata: {
      gesture: gesture.kind,
      ...(gesture.target ? { target: gesture.target } : {}),
      ...(gesture.direction ? { direction: gesture.direction } : {}),
    },
  };
}

/** Deterministic synthetic trajectory from a scripted gesture flow. */
function syntheticTrajectory(id: string, flow: Gesture[]): TrajectorySpan {
  return buildTrajectorySpan({
    id,
    sessionId: `session-${id}`,
    captureTier: "full",
    actions: flow.map((gesture, index) => gestureAction(1000 + index * 10, gesture)),
  });
}

describe("tokenizeAction", () => {
  it("produces a deterministic, structured token from gesture metadata", () => {
    const token = tokenizeAction(gestureAction(1, { kind: "swipe", direction: "up", target: "Feed List" }));
    expect(token).toBe("deviceswipe/up@feed-list");
    // Identical input ⇒ identical token (no randomness / clock).
    expect(tokenizeAction(gestureAction(999, { kind: "swipe", direction: "up", target: "Feed List" }))).toBe(token);
  });

  it("falls back to the summary slug when no gesture metadata is present", () => {
    const token = tokenizeAction({ kind: "action", tool: "editor", summary: "Save File", ts: 5 });
    expect(token).toBe("editorsave-file");
  });
});

describe("buildMovementDataset", () => {
  it("orders actions by timestamp and emits sliding-window samples", () => {
    const trajectory = syntheticTrajectory("t1", [
      { kind: "tap", target: "Search" },
      { kind: "type", target: "Search" },
      { kind: "tap", target: "Result" },
    ]);
    const dataset = buildMovementDataset([trajectory], { contextWindow: 2 });
    expect(dataset.sequences).toHaveLength(1);
    expect(dataset.sequences[0]).toEqual(trajectoryToMovementTokens(trajectory));
    // One sample per position (including the empty-context first move).
    expect(dataset.samples).toHaveLength(3);
    expect(dataset.samples[0]).toEqual({ context: [], next: dataset.sequences[0]![0] });
    expect(dataset.samples[2]!.context).toHaveLength(2);
  });
});

describe("MarkovMovementBackend", () => {
  const backend = new MarkovMovementBackend(3);

  it("repeats a recorded movement flow via rollout", () => {
    const flow: Gesture[] = [
      { kind: "tap", target: "Menu" },
      { kind: "tap", target: "Settings" },
      { kind: "scroll", direction: "down" },
      { kind: "tap", target: "Privacy" },
    ];
    const dataset = buildMovementDataset([syntheticTrajectory("rec", flow)]);
    const snapshot = backend.train(dataset.samples);
    const tokens = trajectoryToMovementTokens(syntheticTrajectory("rec", flow));

    // Seeded with the first movement, the model reproduces the rest exactly.
    const rolled = rolloutMovements(backend, snapshot, [tokens[0]!], tokens.length - 1);
    expect(rolled).toEqual(tokens.slice(1));

    // High confidence and a real context match for a fully-recorded continuation.
    const prediction = backend.predict(snapshot, tokens.slice(0, 2));
    expect(prediction.token).toBe(tokens[2]);
    expect(prediction.order).toBeGreaterThanOrEqual(1);
    expect(prediction.confidence).toBeGreaterThan(0.9);
    expect(prediction.backoffSteps).toBe(0);
  });

  it("generalizes to a new-but-related flow by backing off to shared sub-movements", () => {
    // Two related flows share the sub-sequence tap@menu → tap@settings → scroll/down.
    const flows: Gesture[][] = [
      [
        { kind: "tap", target: "Menu" },
        { kind: "tap", target: "Settings" },
        { kind: "scroll", direction: "down" },
        { kind: "tap", target: "Privacy" },
      ],
      [
        { kind: "tap", target: "Menu" },
        { kind: "tap", target: "Settings" },
        { kind: "scroll", direction: "down" },
        { kind: "tap", target: "Privacy" },
      ],
    ];
    const dataset = buildMovementDataset(flows.map((flow, i) => syntheticTrajectory(`f${i}`, flow)));
    const snapshot = backend.train(dataset.samples);

    // A context the model never saw *in full* (prefixed by an unfamiliar move) still
    // predicts the learned continuation by backing off to the shared suffix.
    const novelContext = ["devicetap@home", "devicetap@menu", "devicetap@settings"];
    const prediction = backend.predict(snapshot, novelContext);
    expect(prediction.token).toBe("devicescroll/down");
    expect(prediction.order).toBeGreaterThanOrEqual(1);
    expect(prediction.order).toBeLessThan(novelContext.length);
    expect(prediction.backoffSteps).toBeGreaterThan(0);
  });

  it("returns a null prediction for an empty model", () => {
    const snapshot = backend.train([]);
    expect(backend.predict(snapshot, ["deviceTap@x"]).token).toBeNull();
  });

  it("produces byte-identical snapshots for identical datasets (deterministic)", () => {
    const dataset = buildMovementDataset([
      syntheticTrajectory("d", [
        { kind: "tap", target: "A" },
        { kind: "tap", target: "B" },
      ]),
    ]);
    const a = backend.train(dataset.samples);
    const b = backend.train(dataset.samples);
    expect(JSON.stringify(a)).toBe(JSON.stringify(b));
  });
});

describe("evaluateMovementModel", () => {
  it("scores high accuracy on held-out trajectories drawn from the trained distribution", () => {
    const backend = new MarkovMovementBackend(3);
    const flow: Gesture[] = [
      { kind: "tap", target: "Inbox" },
      { kind: "tap", target: "Message" },
      { kind: "swipe", direction: "left" },
      { kind: "tap", target: "Archive" },
    ];
    // Train on several copies, hold out a related instance (same flow, new ids).
    const train = buildMovementDataset(
      [0, 1, 2].map((i) => syntheticTrajectory(`train-${i}`, flow)),
    );
    const snapshot = backend.train(train.samples);
    const heldOut = [trajectoryToMovementTokens(syntheticTrajectory("eval", flow))];

    const result = evaluateMovementModel(backend, snapshot, heldOut);
    expect(result.total).toBe(3);
    expect(result.accuracy).toBe(1);
    expect(result.informedFraction).toBeGreaterThan(0);
    expect(result.meanConfidence).toBeGreaterThan(0);
  });

  it("reports zero metrics with no held-out data", () => {
    const backend = new MarkovMovementBackend(2);
    const snapshot = backend.train(buildSamplesFromSequences([["a", "b"]], 2));
    const result = evaluateMovementModel(backend, snapshot, []);
    expect(result).toMatchObject({ total: 0, accuracy: 0, informedFraction: 0, meanConfidence: 0 });
  });
});
