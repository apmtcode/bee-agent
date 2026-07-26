import { describe, expect, it } from "vitest";
import { buildMovementDataset, deriveAppContext, tokenizeAction } from "./dataset.js";
import { generateWorkflowTrajectories, type WorkflowSpec } from "./synthetic.js";
import type { TrajectoryAction, TrajectoryObservation } from "../../capture/trajectory.js";

const WORKFLOW: WorkflowSpec = {
  appName: "notes",
  steps: [
    { tool: "device", gesture: "tap", target: "search" },
    { tool: "device", gesture: "type", target: "query" },
    { tool: "device", gesture: "tap", target: "result" },
  ],
};

function action(overrides: Partial<TrajectoryAction>): TrajectoryAction {
  return { kind: "action", tool: "device", summary: "did thing", ts: 0, ...overrides };
}

function observation(overrides: Partial<TrajectoryObservation>): TrajectoryObservation {
  return { kind: "observation", source: "device", summary: "", ts: 0, ...overrides };
}

describe("tokenizeAction", () => {
  it("prefers structured gesture/target metadata", () => {
    expect(tokenizeAction(action({ metadata: { gesture: "tap", target: "Submit Button" } }))).toBe(
      "device:tap:submit-button",
    );
  });

  it("uses direction when no target is present", () => {
    expect(tokenizeAction(action({ metadata: { gesture: "swipe", direction: "left" } }))).toBe(
      "device:swipe:left",
    );
  });

  it("falls back to the summary when no structured metadata exists", () => {
    expect(tokenizeAction(action({ tool: "shell", summary: "run tests" }))).toBe("shell:run-tests");
  });
});

describe("deriveAppContext", () => {
  it("prefers an app name from observation metadata", () => {
    expect(deriveAppContext([observation({ metadata: { appName: "Xcode" } })])).toBe("xcode");
  });

  it("falls back to the source, then to unknown", () => {
    expect(deriveAppContext([observation({ source: "browser" })])).toBe("browser");
    expect(deriveAppContext([])).toBe("unknown");
  });
});

describe("buildMovementDataset", () => {
  it("produces sliding-window samples ordered by timestamp", () => {
    const [trajectory] = generateWorkflowTrajectories(WORKFLOW, 1);
    const dataset = buildMovementDataset([trajectory!], { order: 2 });

    expect(dataset.samples).toHaveLength(3);
    expect(dataset.samples[0]).toMatchObject({ index: 0, context: [], action: "device:tap:search" });
    expect(dataset.samples[1]).toMatchObject({
      index: 1,
      context: ["device:tap:search"],
      action: "device:type:query",
    });
    expect(dataset.samples[2]).toMatchObject({
      index: 2,
      context: ["device:tap:search", "device:type:query"],
      action: "device:tap:result",
    });
    expect(dataset.vocabulary).toEqual([
      "device:tap:result",
      "device:tap:search",
      "device:type:query",
    ]);
    expect(dataset.contexts).toEqual(["notes"]);
  });

  it("caps context length at the configured order", () => {
    const [trajectory] = generateWorkflowTrajectories(WORKFLOW, 1);
    const dataset = buildMovementDataset([trajectory!], { order: 1 });
    expect(dataset.samples[2]!.context).toEqual(["device:type:query"]);
  });

  it("skips unapproved trajectories when requireApproved is set", () => {
    const [approved] = generateWorkflowTrajectories(WORKFLOW, 1);
    const unreviewed = { ...approved!, id: "pending", review: undefined };
    const dataset = buildMovementDataset([approved!, unreviewed], { requireApproved: true });
    expect(new Set(dataset.samples.map((sample) => sample.trajectoryId))).toEqual(new Set([approved!.id]));
  });
});
