import { describe, expect, it } from "vitest";
import { buildTrajectorySpan, type TrajectoryAction, type TrajectorySpan } from "../capture/trajectory.js";
import {
  buildMovementDataset,
  MOVEMENT_END_TOKEN,
  MOVEMENT_START_TOKEN,
  NgramMovementBackend,
  tokenizeAction,
  type MovementSequence,
} from "./movement-model.js";

function action(tool: string, summary: string, ts: number, metadata?: Record<string, unknown>): TrajectoryAction {
  return { kind: "action", tool, summary, ts, ...(metadata ? { metadata } : {}) };
}

function seq(id: string, tokens: string[]): MovementSequence {
  return { id, tokens };
}

describe("tokenizeAction", () => {
  it("prefers structured gesture metadata over free-text summary", () => {
    const token = tokenizeAction(action("device", "tapped the Send button", 1, { gesture: "tap", target: "Send Button" }));
    expect(token).toBe("device/tap:send-button");
  });

  it("uses gesture direction when no target is present", () => {
    const token = tokenizeAction(action("device", "scrolled down", 1, { gesture: "scroll", direction: "down" }));
    expect(token).toBe("device/scroll:down");
  });

  it("falls back to tool + summary slug when no gesture metadata", () => {
    expect(tokenizeAction(action("editor", "Open File", 1))).toBe("editor:open-file");
  });

  it("collapses semantically equal gestures to the same token", () => {
    const a = tokenizeAction(action("device", "tapped Send", 1, { gesture: "tap", target: "Send" }));
    const b = tokenizeAction(action("device", "clicked the send control", 2, { gesture: "tap", target: "send" }));
    expect(a).toBe(b);
  });
});

describe("buildMovementDataset", () => {
  it("orders actions by timestamp and drops trajectories with no actions", () => {
    const withActions: TrajectorySpan = buildTrajectorySpan({
      id: "t1",
      sessionId: "s1",
      actions: [action("device", "b", 20), action("device", "a", 10)],
    });
    const empty: TrajectorySpan = buildTrajectorySpan({ id: "t2", sessionId: "s1" });

    const dataset = buildMovementDataset([withActions, empty]);
    expect(dataset.sequences).toHaveLength(1);
    expect(dataset.sequences[0]!.tokens).toEqual(["device:a", "device:b"]);
  });

  it("prefers reviewed/redacted actions when present", () => {
    const trajectory: TrajectorySpan = {
      ...buildTrajectorySpan({ id: "t3", sessionId: "s1", actions: [action("device", "secret", 5)] }),
      review: {
        status: "approved",
        reviewedAt: "2026-01-01T00:00:00Z",
        reviewedBy: "user",
        redactedActions: [{ ts: 5, tool: "device", summary: "public action" }],
      },
    };
    const dataset = buildMovementDataset([trajectory]);
    expect(dataset.sequences[0]!.tokens).toEqual(["device:public-action"]);
  });
});

describe("NgramMovementBackend", () => {
  const backend = new NgramMovementBackend();

  it("repeats a recorded movement sequence exactly", () => {
    const model = backend.train({ sequences: [seq("s1", ["a", "b", "c"])] }, { order: 2 });
    expect(model.generate([])).toEqual(["a", "b", "c"]);
  });

  it("predicts the next recorded token from a known context", () => {
    const model = backend.train({ sequences: [seq("s1", ["open", "file", "save"])] }, { order: 2 });
    const prediction = model.predictNext(["open", "file"]);
    expect(prediction?.token).toBe("save");
    expect(prediction?.exact).toBe(true);
  });

  it("generalizes to a novel-but-related context via back-off", () => {
    // Trained: the bigram (file -> save) is strongly established after `login`.
    const model = backend.train(
      {
        sequences: [seq("s1", ["login", "file", "save"]), seq("s2", ["login", "file", "save"])],
      },
      { order: 2 },
    );

    // Held-out context the model never saw at full order-2 ("logout file"),
    // but the order-1 context ("file") was seen — back-off should still predict.
    const prediction = model.predictNext([MOVEMENT_START_TOKEN, "logout", "file"]);
    expect(prediction?.token).toBe("save");
    expect(prediction?.exact).toBe(false);
    expect(prediction?.contextOrder).toBe(1);
  });

  it("scores next-token accuracy on held-out related trajectories", () => {
    const model = backend.train(
      {
        sequences: [
          seq("s1", ["login", "file", "save", "logout"]),
          seq("s2", ["login", "file", "save", "logout"]),
        ],
      },
      { order: 2 },
    );

    const report = model.evaluate([seq("held", ["login", "file", "save", "logout"])]);
    expect(report.sequenceCount).toBe(1);
    expect(report.predictionCount).toBeGreaterThan(0);
    expect(report.nextTokenAccuracy).toBe(1);
  });

  it("reports novel-context rate for an unrelated held-out sequence", () => {
    const model = backend.train({ sequences: [seq("s1", ["a", "b", "c"])] }, { order: 2 });
    const report = model.evaluate([seq("held", ["x", "y", "z"])]);
    // Nothing about x/y/z was learned, so predictions fall back to order-0.
    expect(report.novelContextRate).toBeGreaterThan(0);
  });

  it("is deterministic: identical datasets produce identical snapshots", () => {
    const dataset = { sequences: [seq("s1", ["a", "b"]), seq("s2", ["a", "c"])] };
    const first = backend.train(dataset, { order: 2 }).toSnapshot();
    const second = backend.train(dataset, { order: 2 }).toSnapshot();
    expect(first).toEqual(second);
  });

  it("breaks prediction ties lexically for reproducibility", () => {
    // From context [a], both b and c were seen exactly once → lexical tie-break picks b.
    const model = backend.train({ sequences: [seq("s1", ["a", "c"]), seq("s2", ["a", "b"])] }, { order: 1 });
    expect(model.predictNext(["a"])?.token).toBe("b");
  });

  it("round-trips through a serializable snapshot", () => {
    const trained = backend.train({ sequences: [seq("s1", ["open", "file", "save"])] }, { order: 2 });
    const snapshot = trained.toSnapshot();
    const restored = backend.load(JSON.parse(JSON.stringify(snapshot)));

    expect(restored.order).toBe(trained.order);
    expect(restored.vocabulary).toEqual(trained.vocabulary);
    expect(restored.generate([])).toEqual(trained.generate([]));
    expect(restored.predictNext(["open", "file"])?.token).toBe("save");
  });

  it("stops generation at the learned end sentinel", () => {
    const model = backend.train({ sequences: [seq("s1", ["a", "b"])] }, { order: 2 });
    const generated = model.generate([]);
    expect(generated).not.toContain(MOVEMENT_END_TOKEN);
    expect(generated).not.toContain(MOVEMENT_START_TOKEN);
    expect(generated).toEqual(["a", "b"]);
  });

  it("learns a full capture→dataset→train→replay round-trip from trajectories", () => {
    const trajectory = buildTrajectorySpan({
      id: "t1",
      sessionId: "s1",
      actions: [
        action("device", "open app", 1, { gesture: "tap", target: "AppIcon" }),
        action("device", "type query", 2, { gesture: "type", target: "SearchBox" }),
        action("device", "submit", 3, { gesture: "tap", target: "Go" }),
      ],
    });
    const dataset = buildMovementDataset([trajectory]);
    const model = backend.train(dataset, { order: 2 });
    expect(model.generate([])).toEqual(dataset.sequences[0]!.tokens);
  });
});
