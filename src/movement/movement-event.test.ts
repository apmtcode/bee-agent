import { describe, expect, it } from "vitest";
import { buildTrajectorySpan, type TrajectoryAction } from "../capture/trajectory.js";
import {
  buildMovementToken,
  sequenceTokens,
  slugifyMovementLabel,
  tokenizeTrajectorySpan,
} from "./movement-event.js";

function action(partial: Partial<TrajectoryAction> & { ts: number }): TrajectoryAction {
  return {
    kind: "action",
    tool: partial.tool ?? "device",
    summary: partial.summary ?? "did thing",
    ts: partial.ts,
    ...(partial.metadata ? { metadata: partial.metadata } : {}),
  };
}

describe("slugifyMovementLabel", () => {
  it("collapses casing and punctuation to a stable slug", () => {
    expect(slugifyMovementLabel("Submit Order!")).toBe("submit-order");
    expect(slugifyMovementLabel("  search field  ")).toBe("search-field");
  });
});

describe("buildMovementToken", () => {
  it("joins verb and descriptor", () => {
    expect(buildMovementToken("tap", "submit")).toBe("tap:submit");
  });
  it("omits descriptor when absent", () => {
    expect(buildMovementToken("scroll")).toBe("scroll");
  });
  it("normalizes verb and descriptor identically regardless of casing", () => {
    expect(buildMovementToken("Tap", "Submit Order")).toBe("tap:submit-order");
  });
});

describe("tokenizeTrajectorySpan", () => {
  it("derives tokens from gesture metadata and sorts by timestamp", () => {
    const span = buildTrajectorySpan({
      id: "t1",
      sessionId: "s1",
      actions: [
        action({ ts: 200, tool: "device", metadata: { gesture: "tap", target: "submit" } }),
        action({ ts: 100, tool: "device", metadata: { gesture: "type", target: "email" } }),
      ],
    });
    const sequence = tokenizeTrajectorySpan(span);
    expect(sequenceTokens(sequence)).toEqual(["type:email", "tap:submit"]);
    expect(sequence.events[0].action).toBe("type");
    expect(sequence.events[0].descriptor).toBe("email");
  });

  it("falls back to tool name and direction when gesture metadata is missing", () => {
    const span = buildTrajectorySpan({
      id: "t2",
      sessionId: "s2",
      actions: [action({ ts: 1, tool: "Scroll", metadata: { direction: "down" } })],
    });
    const sequence = tokenizeTrajectorySpan(span);
    expect(sequenceTokens(sequence)).toEqual(["scroll:down"]);
  });

  it("carries outcome status through", () => {
    const span = buildTrajectorySpan({
      id: "t3",
      sessionId: "s3",
      actions: [action({ ts: 1 })],
      outcome: { status: "success", summary: "done" },
    });
    expect(tokenizeTrajectorySpan(span).outcome).toBe("success");
  });
});
