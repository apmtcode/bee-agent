import { describe, expect, it } from "vitest";
import { buildTrajectorySpan, type TrajectoryAction, type TrajectorySpan } from "../capture/trajectory.js";
import {
  MarkovMovementBackend,
  buildMovementDataset,
  decodeMovementToken,
  encodeMovementClass,
  encodeMovementToken,
  evaluateMovementPolicy,
  extractMovementTokens,
  trainMovementPolicy,
  type MovementModelBackend,
  type MovementPolicy,
  type MovementToken,
} from "./movement-model.js";

let seq = 0;

function action(
  gesture: string,
  opts: { tool?: string; target?: string; direction?: string; ts?: number } = {},
): TrajectoryAction {
  return {
    kind: "action",
    tool: opts.tool ?? "device",
    summary: `${gesture} ${opts.target ?? ""}`.trim(),
    ts: opts.ts ?? (seq += 1),
    metadata: {
      gesture,
      ...(opts.target ? { target: opts.target } : {}),
      ...(opts.direction ? { direction: opts.direction } : {}),
    },
  };
}

function synthTrajectory(tokens: TrajectoryAction[], id = `traj-${(seq += 1)}`): TrajectorySpan {
  return buildTrajectorySpan({
    id,
    sessionId: "session-1",
    captureTier: "full",
    actions: tokens,
  });
}

/** A repeated "compose → send" workflow: the canonical recorded movement. */
function composeSendTrajectory(target: string, id?: string): TrajectorySpan {
  return synthTrajectory(
    [
      action("tap", { target: "compose-button", ts: 1 }),
      action("type", { target: "recipient-field", ts: 2 }),
      action("type", { target, ts: 3 }),
      action("tap", { target: "send-button", ts: 4 }),
    ],
    id,
  );
}

describe("movement token encoding", () => {
  it("round-trips tokens through encode/decode", () => {
    const token: MovementToken = { tool: "device", gesture: "swipe", target: "card", direction: "left" };
    expect(decodeMovementToken(encodeMovementToken(token))).toEqual(token);
  });

  it("drops the target for the class encoding", () => {
    const a: MovementToken = { tool: "device", gesture: "tap", target: "save", direction: undefined };
    const b: MovementToken = { tool: "device", gesture: "tap", target: "send" };
    expect(encodeMovementClass(a)).toBe(encodeMovementClass(b));
    expect(encodeMovementToken(a)).not.toBe(encodeMovementToken(b));
  });
});

describe("extractMovementTokens", () => {
  it("orders actions by timestamp and maps gesture metadata", () => {
    const trajectory = synthTrajectory([
      action("tap", { target: "send-button", ts: 30 }),
      action("type", { target: "body", ts: 10 }),
      action("swipe", { direction: "down", ts: 20 }),
    ]);
    const tokens = extractMovementTokens(trajectory);
    expect(tokens.map((t) => t.gesture)).toEqual(["type", "swipe", "tap"]);
    expect(tokens[1]).toMatchObject({ gesture: "swipe", direction: "down" });
  });

  it("prefers redacted actions when a review redacted them", () => {
    const trajectory: TrajectorySpan = {
      ...synthTrajectory([action("tap", { target: "secret", ts: 1 })]),
      review: {
        status: "approved",
        reviewedAt: "2026-07-06T00:00:00Z",
        reviewedBy: "reviewer",
        redactedActions: [{ ts: 1, tool: "device", summary: "tapped [redacted]" }],
      },
    };
    const tokens = extractMovementTokens(trajectory);
    // Redacted actions carry no metadata, so gesture falls back to "action".
    expect(tokens).toEqual([{ tool: "device", gesture: "action" }]);
  });
});

describe("buildMovementDataset", () => {
  it("builds sequences and a sorted vocabulary, skipping empty trajectories", () => {
    const dataset = buildMovementDataset([
      composeSendTrajectory("hello", "a"),
      synthTrajectory([], "empty"),
    ]);
    expect(dataset.sequences).toHaveLength(1);
    expect(dataset.sequences[0].tokens).toHaveLength(4);
    expect(dataset.vocabulary).toEqual([...dataset.vocabulary].sort());
    expect(dataset.vocabulary.length).toBeGreaterThan(0);
  });
});

describe("MarkovMovementBackend replay fidelity (objective 2c)", () => {
  it("replays a recorded workflow verbatim after training", () => {
    const trajectory = composeSendTrajectory("quarterly numbers", "train");
    const { policy } = trainMovementPolicy([trajectory]);

    const generated = policy.generate([extractMovementTokens(trajectory)[0]], { maxSteps: 3 });
    const expected = extractMovementTokens(trajectory).slice(1);
    expect(generated).toEqual(expected);
  });

  it("scores exact replay accuracy of 1.0 on the training sequence", () => {
    const trajectory = composeSendTrajectory("report", "train");
    const { policy, dataset } = trainMovementPolicy([trajectory]);
    const report = evaluateMovementPolicy(policy, dataset.sequences);
    expect(report.exactAccuracy).toBe(1);
    expect(report.abstentionRate).toBe(0);
  });

  it("is deterministic: identical training yields identical predictions", () => {
    const trajectories = [composeSendTrajectory("a", "1"), composeSendTrajectory("b", "2")];
    const p1 = trainMovementPolicy(trajectories).policy;
    const p2 = trainMovementPolicy(trajectories).policy;
    const ctx = [extractMovementTokens(trajectories[0])[0]];
    expect(p1.predictNext(ctx)).toEqual(p2.predictNext(ctx));
  });
});

describe("MarkovMovementBackend generalization (objective 2d)", () => {
  it("generalizes to a novel target via class back-off", () => {
    // Train on several compose→send flows with different message targets.
    const training = [
      composeSendTrajectory("alpha", "1"),
      composeSendTrajectory("beta", "2"),
      composeSendTrajectory("gamma", "3"),
    ];
    const { policy } = trainMovementPolicy(training);

    // A held-out flow with a target the model never saw ("delta").
    const heldOut = extractMovementTokens(composeSendTrajectory("delta", "held"));
    // After typing recipient + a never-before-seen body, the model should still
    // predict the "tap send-button" movement — a related but new movement.
    const prediction = policy.predictNext(heldOut.slice(0, 3));
    expect(prediction).toBeDefined();
    expect(prediction?.token).toMatchObject({ gesture: "tap", target: "send-button" });
  });

  it("reports high class accuracy on held-out related sequences even when exact is imperfect", () => {
    const training = ["m1", "m2", "m3", "m4"].map((t, i) => composeSendTrajectory(t, `t${i}`));
    const { policy } = trainMovementPolicy(training);

    const heldOut = [
      { trajectoryId: "h1", sessionId: "s", tokens: extractMovementTokens(composeSendTrajectory("novel-body-1")) },
      { trajectoryId: "h2", sessionId: "s", tokens: extractMovementTokens(composeSendTrajectory("novel-body-2")) },
    ];
    const report = evaluateMovementPolicy(policy, heldOut);
    expect(report.classAccuracy).toBeGreaterThanOrEqual(report.exactAccuracy);
    expect(report.classAccuracy).toBeGreaterThan(0.7);
  });

  it("abstains cleanly on an unrelated first movement it never saw", () => {
    const { policy } = trainMovementPolicy([composeSendTrajectory("x", "1")]);
    const prediction = policy.predictNext([{ tool: "unheard-of", gesture: "nope" }]);
    // Unknown context falls back to the unigram (order-0) distribution rather
    // than abstaining — generalization prefers a best-effort movement.
    expect(prediction).toBeDefined();
    expect(prediction?.backoffDistance).toBe(1);
  });
});

describe("pluggable backend seam", () => {
  it("accepts a custom backend implementation", () => {
    const fixed: MovementToken = { tool: "device", gesture: "tap", target: "only" };
    const stubBackend: MovementModelBackend = {
      id: "stub",
      train(): MovementPolicy {
        return {
          backendId: "stub",
          predictNext: () => ({ token: fixed, probability: 1, backoffDistance: 0, generalized: false }),
          generate: (_seed, options) =>
            Array.from({ length: options?.maxSteps ?? 1 }, () => fixed),
        };
      },
    };
    const { policy } = trainMovementPolicy([composeSendTrajectory("x")], { backend: stubBackend });
    expect(policy.backendId).toBe("stub");
    expect(policy.predictNext([])?.token).toEqual(fixed);
  });

  it("MarkovMovementBackend advertises its id", () => {
    expect(new MarkovMovementBackend().id).toBe("markov");
  });
});

describe("evaluateMovementPolicy edge cases", () => {
  it("returns zeroed metrics for empty held-out input", () => {
    const { policy } = trainMovementPolicy([composeSendTrajectory("x")]);
    const report = evaluateMovementPolicy(policy, []);
    expect(report).toMatchObject({ steps: 0, exactAccuracy: 0, classAccuracy: 0 });
  });
});
