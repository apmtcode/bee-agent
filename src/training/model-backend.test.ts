import { describe, expect, it } from "vitest";
import type { ReplayTimelineEvent } from "../capture/replay.js";
import { buildTrajectorySpan } from "../capture/trajectory.js";
import {
  FrequencyMovementBackend,
  buildMovementDataset,
  buildMovementDatasetFromTrajectories,
  evaluateMovementModel,
  type MovementDataset,
} from "./model-backend.js";

/**
 * Deterministic synthetic event-stream generator. Emits an interleaved
 * observation/action timeline for a scripted workflow so the capture -> dataset
 * -> train -> infer loop can be validated without any real OS input.
 */
function syntheticStream(
  steps: Array<{ source: string; tool: string; summary: string }>,
  startTs = 1_000,
): ReplayTimelineEvent[] {
  const events: ReplayTimelineEvent[] = [];
  let ts = startTs;
  steps.forEach((step, index) => {
    events.push({
      kind: "observation",
      ts: ts++,
      trajectoryId: `traj-${index}`,
      source: step.source,
      summary: `looking at ${step.source}`,
    });
    events.push({
      kind: "action",
      ts: ts++,
      trajectoryId: `traj-${index}`,
      tool: step.tool,
      summary: step.summary,
    });
  });
  return events;
}

const WORKFLOW = [
  { source: "browser:inbox", tool: "click", summary: "open first email" },
  { source: "browser:email", tool: "scroll", summary: "scroll to attachment" },
  { source: "browser:email", tool: "click", summary: "download attachment" },
  { source: "finder:downloads", tool: "drag", summary: "move file to project" },
];

describe("buildMovementDataset", () => {
  it("derives context -> action examples from an interleaved event stream", () => {
    const dataset = buildMovementDataset(syntheticStream(WORKFLOW));
    expect(dataset.examples).toHaveLength(4);
    // First action has no previous tool but sees the inbox observation.
    expect(dataset.examples[0]).toEqual({
      context: { observationSource: "browser:inbox", previousTool: undefined },
      action: { tool: "click", summary: "open first email" },
    });
    // Second action's context carries the prior tool forward.
    expect(dataset.examples[1].context).toEqual({
      observationSource: "browser:email",
      previousTool: "click",
    });
  });

  it("ignores transcript events and sorts by timestamp", () => {
    const events: ReplayTimelineEvent[] = [
      { kind: "action", ts: 20, trajectoryId: "t", tool: "type", summary: "b" },
      { kind: "transcript", ts: 5, messageId: "m", role: "user", content: "hi" },
      { kind: "observation", ts: 10, trajectoryId: "t", source: "app", summary: "s" },
    ];
    const dataset = buildMovementDataset(events);
    expect(dataset.examples).toHaveLength(1);
    expect(dataset.examples[0].context.observationSource).toBe("app");
  });
});

describe("FrequencyMovementBackend replay fidelity", () => {
  it("reproduces every recorded movement exactly (train == eval)", async () => {
    const dataset = buildMovementDataset(syntheticStream(WORKFLOW));
    const model = await new FrequencyMovementBackend().train(dataset);
    const result = evaluateMovementModel(model, dataset);
    expect(result.accuracy).toBe(1);
    expect(model.exampleCount).toBe(4);
    expect(model.backend).toBe("frequency-backoff");
    // Every recorded step resolves through the most specific bucket.
    for (const example of dataset.examples) {
      const prediction = model.predict(example.context);
      expect(prediction.tool).toBe(example.action.tool);
      expect(prediction.summary).toBe(example.action.summary);
    }
  });

  it("is deterministic across repeated training runs", async () => {
    const dataset = buildMovementDataset(syntheticStream(WORKFLOW));
    const a = await new FrequencyMovementBackend().train(dataset);
    const b = await new FrequencyMovementBackend().train(dataset);
    const context = { observationSource: "browser:email", previousTool: "click" };
    expect(a.predict(context)).toEqual(b.predict(context));
  });
});

describe("FrequencyMovementBackend generalisation", () => {
  it("backs off to observation-only signal for a novel previous tool", async () => {
    const dataset = buildMovementDataset(syntheticStream(WORKFLOW));
    const model = await new FrequencyMovementBackend().train(dataset);
    // Same window as a recorded step, but arrived via a tool never paired with it.
    const prediction = model.predict({
      observationSource: "browser:email",
      previousTool: "hotkey",
    });
    expect(prediction.basis).toBe("observation-only");
    expect(["scroll", "click"]).toContain(prediction.tool);
    expect(prediction.confidence).toBeGreaterThan(0);
  });

  it("backs off to previous-tool signal for a novel observation source", async () => {
    // A workflow where 'click' is reliably followed by 'confirm'.
    const events = syntheticStream([
      { source: "dialog:save", tool: "click", summary: "press save" },
      { source: "dialog:confirm", tool: "confirm", summary: "confirm overwrite" },
      { source: "dialog:save2", tool: "click", summary: "press save again" },
      { source: "dialog:confirm2", tool: "confirm", summary: "confirm overwrite again" },
    ]);
    const model = await new FrequencyMovementBackend().train(buildMovementDataset(events));
    const prediction = model.predict({
      observationSource: "dialog:brand-new",
      previousTool: "click",
    });
    expect(prediction.basis).toBe("previous-tool-only");
    expect(prediction.tool).toBe("confirm");
  });

  it("falls back to the global prior when nothing in context is known", async () => {
    const dataset = buildMovementDataset(syntheticStream(WORKFLOW));
    const model = await new FrequencyMovementBackend().train(dataset);
    const prediction = model.predict({
      observationSource: "unseen",
      previousTool: "unseen",
    });
    expect(prediction.basis).toBe("global-prior");
    // 'click' appears most often in the workflow, so it wins the global prior.
    expect(prediction.tool).toBe("click");
  });

  it("returns an 'unknown' prediction for an empty dataset", async () => {
    const model = await new FrequencyMovementBackend().train({ examples: [] });
    const prediction = model.predict({ observationSource: "x", previousTool: "y" });
    expect(prediction).toEqual({ tool: "", summary: "", confidence: 0, basis: "unknown" });
  });
});

describe("evaluateMovementModel on held-out related trajectories", () => {
  it("generalises above chance to an unseen-but-related workflow", async () => {
    // Train on two runs of a workflow; hold out a third related run whose
    // window/tool transitions overlap but are not identical.
    const trainStream = [
      ...syntheticStream(WORKFLOW, 1_000),
      ...syntheticStream(WORKFLOW, 2_000),
    ];
    const model = await new FrequencyMovementBackend().train(buildMovementDataset(trainStream));

    const heldOut: MovementDataset = buildMovementDataset(
      syntheticStream(
        [
          { source: "browser:inbox", tool: "click", summary: "open a different email" },
          { source: "browser:email", tool: "scroll", summary: "scroll further" },
          { source: "finder:downloads", tool: "drag", summary: "move a different file" },
        ],
        9_000,
      ),
    );

    const result = evaluateMovementModel(model, heldOut);
    expect(result.total).toBe(3);
    expect(result.accuracy).toBeGreaterThanOrEqual(0.6);
    const graded = result.byBasis["full-context"].total + result.byBasis["observation-only"].total;
    expect(graded).toBeGreaterThan(0);
  });
});

describe("buildMovementDatasetFromTrajectories", () => {
  it("builds equivalent examples from trajectory spans", async () => {
    const trajectory = buildTrajectorySpan({
      id: "t1",
      sessionId: "s1",
      observations: [
        { kind: "observation", source: "editor", summary: "cursor in file", ts: 1 },
      ],
      actions: [
        { kind: "action", tool: "type", summary: "write line", ts: 2 },
        { kind: "action", tool: "save", summary: "cmd+s", ts: 3 },
      ],
    });
    const dataset = buildMovementDatasetFromTrajectories([trajectory]);
    expect(dataset.examples).toHaveLength(2);
    expect(dataset.examples[0].context).toEqual({
      observationSource: "editor",
      previousTool: undefined,
    });
    expect(dataset.examples[1].context).toEqual({
      observationSource: "editor",
      previousTool: "type",
    });

    const model = await new FrequencyMovementBackend().train(dataset);
    expect(model.predict({ observationSource: "editor", previousTool: "type" }).tool).toBe("save");
  });
});
