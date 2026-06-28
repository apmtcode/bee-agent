import { describe, expect, it } from "vitest";
import { MockMovementModelBackend } from "./movement-backend.js";
import { buildMovementDataset } from "./movement-dataset.js";
import { buildTrajectorySpan } from "../capture/trajectory.js";
import type { TrajectorySpan } from "../capture/trajectory.js";

function approved(span: TrajectorySpan): TrajectorySpan {
  span.review = { status: "approved", reviewedAt: "2026-01-01T00:00:00.000Z", reviewedBy: "tester" };
  return span;
}

const span = approved(
  buildTrajectorySpan({
    id: "t1",
    sessionId: "s1",
    captureTier: "full",
    observations: [
      { kind: "observation", source: "device", summary: "Mail on Compose", ts: 100, metadata: { appName: "Mail", screenTitle: "Compose" } },
      { kind: "observation", source: "device", summary: "Files on Browser", ts: 300, metadata: { appName: "Files", screenTitle: "Browser" } },
    ],
    actions: [
      { kind: "action", tool: "device", summary: "tapped Send", ts: 150, metadata: { gesture: "tap", target: "Send" } },
      { kind: "action", tool: "device", summary: "tapped New Folder", ts: 350, metadata: { gesture: "tap", target: "New Folder" } },
    ],
  }),
);

describe("MockMovementModelBackend", () => {
  it("recalls the recorded movement for an (almost) exact context", async () => {
    const dataset = buildMovementDataset([span]);
    const model = await new MockMovementModelBackend().train(dataset, { trainedAt: "2026-01-01T00:00:00.000Z" });

    const prediction = model.predict({ app: "Mail", screen: "Compose", summary: "Mail on Compose" });
    expect(prediction).toBeDefined();
    expect(prediction?.action.target).toBe("Send");
    expect(prediction?.generalized).toBe(false);
    expect(prediction?.matchedTrajectoryId).toBe("t1");
    expect(prediction?.confidence).toBeGreaterThan(0.6);
  });

  it("exposes deterministic, reproducible training stats", async () => {
    const dataset = buildMovementDataset([span]);
    const backend = new MockMovementModelBackend();
    const a = await backend.train(dataset, { trainedAt: "2026-01-01T00:00:00.000Z" });
    const b = await backend.train(dataset, { trainedAt: "2026-01-01T00:00:00.000Z" });
    expect(a.stats).toEqual(b.stats);
    expect(a.stats.exampleCount).toBe(2);
    expect(a.stats.appCount).toBe(2);
    expect(a.predict({ app: "Mail", screen: "Compose", summary: "Mail on Compose" })).toEqual(
      b.predict({ app: "Mail", screen: "Compose", summary: "Mail on Compose" }),
    );
  });

  it("generalizes a known gesture to a related context and retargets to a named slot", async () => {
    const dataset = buildMovementDataset([span]);
    const model = await new MockMovementModelBackend().train(dataset, { trainedAt: "2026-01-01T00:00:00.000Z" });

    // Never-seen context within the known Mail app that names the "New Folder"
    // target (learned in the Files app). The model should keep the tap gesture
    // and retarget to the slot mentioned by the query.
    const prediction = model.predict({ app: "Mail", screen: "Sidebar", summary: "Mail sidebar with New Folder option" });
    expect(prediction).toBeDefined();
    expect(prediction?.generalized).toBe(true);
    expect(prediction?.action.gesture).toBe("tap");
    expect(prediction?.action.target).toBe("New Folder");
    expect(prediction?.confidence).toBeLessThan(0.6);
  });

  it("returns undefined when trained on an empty dataset", async () => {
    const model = await new MockMovementModelBackend().train(buildMovementDataset([]), { trainedAt: "2026-01-01T00:00:00.000Z" });
    expect(model.predict({ summary: "anything" })).toBeUndefined();
  });
});
