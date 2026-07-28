import { describe, expect, it } from "vitest";
import { buildTrajectorySpan } from "../capture/trajectory.js";
import {
  DeterministicMovementBackend,
  buildMovementDataset,
  evaluateMovementGeneralization,
  extractMovementToken,
  movementTokenKey,
  trainMovementModel,
  type MovementDataset,
  type MovementModelBackend,
  type TrainedMovementModel,
} from "./movement-model.js";
import {
  COMPOSE_AND_SEND_TEMPLATE,
  SEARCH_AND_OPEN_TEMPLATE,
  generateSyntheticTrajectory,
  generateSyntheticTrajectoryFamily,
} from "./synthetic-trajectories.js";

describe("movement dataset extraction", () => {
  it("extracts normalized movement tokens from action metadata", () => {
    const token = extractMovementToken(
      {
        kind: "action",
        tool: "device",
        summary: "tapped Search Field",
        ts: 10,
        metadata: { gesture: "tap", target: "Search-Field", direction: "up" },
      },
      "notes",
    );
    expect(token).toEqual({ app: "notes", tool: "device", gesture: "tap", target: "search-field", direction: "up" });
  });

  it("orders actions by timestamp and derives app from observation metadata", () => {
    const trajectory = buildTrajectorySpan({
      id: "t1",
      sessionId: "s1",
      observations: [
        { kind: "observation", source: "device", summary: "Mail active", ts: 1, metadata: { appName: "Mail" } },
      ],
      actions: [
        { kind: "action", tool: "device", summary: "b", ts: 20, metadata: { gesture: "tap", target: "b" } },
        { kind: "action", tool: "device", summary: "a", ts: 10, metadata: { gesture: "tap", target: "a" } },
      ],
    });
    const dataset = buildMovementDataset([trajectory]);
    expect(dataset.sequences).toHaveLength(1);
    expect(dataset.sequences[0].app).toBe("mail");
    expect(dataset.sequences[0].tokens.map((token) => token.target)).toEqual(["a", "b"]);
  });

  it("drops trajectories with no actions", () => {
    const trajectory = buildTrajectorySpan({ id: "empty", sessionId: "s1", actions: [] });
    expect(buildMovementDataset([trajectory]).sequences).toHaveLength(0);
  });
});

describe("recall: repeat recorded movements (objective 2c)", () => {
  it("greedily reproduces a trained trajectory with full fidelity", async () => {
    const trajectory = generateSyntheticTrajectory({
      id: "rec-1",
      sessionId: "s1",
      app: "browser",
      template: SEARCH_AND_OPEN_TEMPLATE,
    });
    const dataset = buildMovementDataset([trajectory]);
    const model = await trainMovementModel(dataset);

    const report = evaluateMovementGeneralization(model, dataset);
    expect(report.rolloutFidelity).toBe(1);
    expect(report.nextStepAccuracy).toBe(1);
    expect(report.bySource.exact.total).toBeGreaterThan(0);
  });

  it("predicts the recorded next movement from an exact context", async () => {
    const trajectory = generateSyntheticTrajectory({
      id: "rec-2",
      sessionId: "s1",
      app: "browser",
      template: SEARCH_AND_OPEN_TEMPLATE,
    });
    const model = await trainMovementModel(buildMovementDataset([trajectory]));

    const firstTap = extractMovementToken(trajectory.actions[0], "browser");
    const prediction = model.predict({ app: "browser", history: [firstTap] });
    expect(prediction.source).toBe("exact");
    expect(prediction.token && movementTokenKey(prediction.token)).toBe(
      movementTokenKey(extractMovementToken(trajectory.actions[1], "browser")),
    );
    expect(prediction.confidence).toBeGreaterThan(0);
  });
});

describe("generalization: new-but-related movements (objective 2d)", () => {
  it("transfers a learned motion to an unseen app", async () => {
    // Train on the same task performed in three apps; hold out a fourth app.
    const trainApps = ["browser", "notes", "finder"];
    const trainTrajectories = generateSyntheticTrajectoryFamily({
      template: SEARCH_AND_OPEN_TEMPLATE,
      apps: trainApps,
    });
    const heldOut = generateSyntheticTrajectoryFamily({
      template: SEARCH_AND_OPEN_TEMPLATE,
      apps: ["music"],
      baseId: "held",
    });

    const model = await trainMovementModel(buildMovementDataset(trainTrajectories));
    const report = evaluateMovementGeneralization(model, buildMovementDataset(heldOut));

    // The held-out app was never trained, so predictions come from transfer...
    expect(report.bySource.exact.total).toBe(0);
    // ...yet the model still reproduces the motion structure with high fidelity.
    expect(report.generalizedNextStepAccuracy).toBeGreaterThanOrEqual(0.75);
    expect(report.rolloutFidelity).toBeGreaterThanOrEqual(0.75);
  });

  it("does not hallucinate a matching motion for an unrelated task", async () => {
    const trainTrajectories = generateSyntheticTrajectoryFamily({
      template: SEARCH_AND_OPEN_TEMPLATE,
      apps: ["browser", "notes"],
    });
    const unrelated = generateSyntheticTrajectory({
      id: "unrelated",
      sessionId: "s1",
      app: "mail",
      template: COMPOSE_AND_SEND_TEMPLATE,
    });
    const model = await trainMovementModel(buildMovementDataset(trainTrajectories));
    const report = evaluateMovementGeneralization(model, buildMovementDataset([unrelated]));

    // A structurally different task should not be reproduced well.
    expect(report.nextStepAccuracy).toBeLessThan(0.5);
  });
});

describe("pluggable backend seam", () => {
  it("accepts a custom backend implementation", async () => {
    const fixed = extractMovementToken(
      { kind: "action", tool: "device", summary: "x", ts: 1, metadata: { gesture: "tap", target: "x" } },
      "app",
    );
    const backend: MovementModelBackend = {
      id: "always-x@test",
      async train(dataset: MovementDataset): Promise<TrainedMovementModel> {
        return {
          backendId: this.id,
          trainedSequenceCount: dataset.sequences.length,
          trainedTokenCount: dataset.sequences.reduce((sum, sequence) => sum + sequence.tokens.length, 0),
          predict() {
            return { token: fixed, confidence: 1, source: "exact", alternatives: [{ token: fixed, weight: 1 }] };
          },
        };
      },
    };
    const dataset = buildMovementDataset([
      generateSyntheticTrajectory({ id: "a", sessionId: "s", app: "app", template: SEARCH_AND_OPEN_TEMPLATE }),
    ]);
    const model = await trainMovementModel(dataset, backend);
    expect(model.backendId).toBe("always-x@test");
    expect(model.predict({ app: "app", history: [] }).token).toEqual(fixed);
  });

  it("reports empty predictions when trained on nothing", async () => {
    const model = await new DeterministicMovementBackend().train({ sequences: [] });
    const prediction = model.predict({ app: "app", history: [] });
    expect(prediction.source).toBe("empty");
    expect(prediction.token).toBeUndefined();
  });
});
