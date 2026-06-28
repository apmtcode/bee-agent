import { describe, expect, it } from "vitest";
import { buildMovementDataset, tokenizeMovementText } from "./movement-dataset.js";
import { buildTrajectorySpan } from "../capture/trajectory.js";
import type { TrajectorySpan } from "../capture/trajectory.js";

function approvedSpan(span: TrajectorySpan): TrajectorySpan {
  span.review = { status: "approved", reviewedAt: "2026-01-01T00:00:00.000Z", reviewedBy: "tester" };
  return span;
}

describe("buildMovementDataset", () => {
  it("pairs each action with its most recent preceding observation", () => {
    const span = approvedSpan(
      buildTrajectorySpan({
        id: "t1",
        sessionId: "s1",
        captureTier: "full",
        observations: [
          { kind: "observation", source: "device", summary: "Mail on Inbox", ts: 100, metadata: { appName: "Mail", screenTitle: "Inbox" } },
          { kind: "observation", source: "device", summary: "Mail on Compose", ts: 300, metadata: { appName: "Mail", screenTitle: "Compose" } },
        ],
        actions: [
          { kind: "action", tool: "device", summary: "tapped Compose", ts: 200, metadata: { gesture: "tap", target: "Compose" } },
          { kind: "action", tool: "device", summary: "tapped Send", ts: 400, metadata: { gesture: "tap", target: "Send" } },
        ],
      }),
    );

    const dataset = buildMovementDataset([span]);
    expect(dataset.examples).toHaveLength(2);
    expect(dataset.examples[0].context.screen).toBe("Inbox");
    expect(dataset.examples[0].action.target).toBe("Compose");
    expect(dataset.examples[1].context.screen).toBe("Compose");
    expect(dataset.examples[1].action.target).toBe("Send");
    expect(dataset.apps).toEqual(["Mail"]);
    expect(dataset.targets).toEqual(["Compose", "Send"]);
    expect(dataset.gestures).toEqual(["tap"]);
  });

  it("excludes unreviewed trajectories by default and includes them when asked", () => {
    const span = buildTrajectorySpan({
      id: "t2",
      sessionId: "s2",
      observations: [{ kind: "observation", source: "device", summary: "Files", ts: 1 }],
      actions: [{ kind: "action", tool: "device", summary: "tapped New", ts: 2, metadata: { gesture: "tap", target: "New" } }],
    });

    expect(buildMovementDataset([span]).examples).toHaveLength(0);
    expect(buildMovementDataset([span], { requireApproved: false }).examples).toHaveLength(1);
  });
});

describe("tokenizeMovementText", () => {
  it("lowercases, splits on non-alphanumerics, and de-duplicates", () => {
    expect(tokenizeMovementText("Mail on Compose", "Send-button")).toEqual([
      "mail",
      "on",
      "compose",
      "send",
      "button",
    ]);
    expect(tokenizeMovementText(undefined, "")).toEqual([]);
  });
});
