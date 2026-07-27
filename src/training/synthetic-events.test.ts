import { describe, expect, it } from "vitest";
import type { ReplayManifest } from "../capture/replay.js";
import { movementToken } from "./movement-model.js";
import {
  DEFAULT_SYNTHETIC_WORKFLOWS,
  generateSyntheticDataset,
  movementSequencesFromReplay,
  movementStepFromDeviceGesture,
} from "./synthetic-events.js";

describe("generateSyntheticDataset", () => {
  it("is reproducible for a given seed", () => {
    const a = generateSyntheticDataset({ seed: 42 });
    const b = generateSyntheticDataset({ seed: 42 });
    expect(a).toEqual(b);
  });

  it("varies with the seed", () => {
    const a = generateSyntheticDataset({ seed: 1, variation: 0.5 });
    const b = generateSyntheticDataset({ seed: 2, variation: 0.5 });
    expect(a).not.toEqual(b);
  });

  it("produces the requested split sizes across all workflows", () => {
    const { train, heldOut } = generateSyntheticDataset({
      seed: 3,
      trainPerWorkflow: 5,
      heldOutPerWorkflow: 2,
    });
    expect(train).toHaveLength(DEFAULT_SYNTHETIC_WORKFLOWS.length * 5);
    expect(heldOut).toHaveLength(DEFAULT_SYNTHETIC_WORKFLOWS.length * 2);
  });

  it("held-out sequences differ from training sequences (related, not identical)", () => {
    const { train, heldOut } = generateSyntheticDataset({ seed: 9, variation: 0.4 });
    const trainKeys = new Set(train.map((s) => `${s.context}|${s.steps.map(movementToken).join(">")}`));
    const heldOutKeys = heldOut.map((s) => `${s.context}|${s.steps.map(movementToken).join(">")}`);
    // At least some held-out sequences are not verbatim copies of training ones.
    expect(heldOutKeys.some((key) => !trainKeys.has(key))).toBe(true);
  });

  it("every generated sequence contains its workflow's canonical steps in order", () => {
    const { train } = generateSyntheticDataset({ seed: 11, trainPerWorkflow: 3, variation: 0.3 });
    for (const workflow of DEFAULT_SYNTHETIC_WORKFLOWS) {
      const canonical = workflow.template.map(movementToken);
      const forWorkflow = train.filter((s) => s.context === workflow.context);
      for (const sequence of forWorkflow) {
        const tokens = sequence.steps.map(movementToken);
        // Canonical steps appear as an ordered subsequence (perturbations only insert).
        let idx = 0;
        for (const token of tokens) {
          if (token === canonical[idx]) {
            idx += 1;
          }
        }
        expect(idx).toBe(canonical.length);
      }
    }
  });

  it("clamps variation to zero for degenerate inputs (no perturbation)", () => {
    const { train } = generateSyntheticDataset({ seed: 5, trainPerWorkflow: 2, variation: 0 });
    for (const workflow of DEFAULT_SYNTHETIC_WORKFLOWS) {
      const canonical = workflow.template.map(movementToken);
      for (const sequence of train.filter((s) => s.context === workflow.context)) {
        expect(sequence.steps.map(movementToken)).toEqual(canonical);
      }
    }
  });
});

describe("movementSequencesFromReplay", () => {
  it("extracts one sequence per trajectory from action events only", () => {
    const manifest: ReplayManifest = {
      version: 1,
      sessionId: "s1",
      trajectoryIds: ["t1", "t2"],
      eventCount: 5,
      events: [
        { kind: "transcript", ts: 1, messageId: "m", role: "user", content: "go" },
        { kind: "observation", ts: 2, trajectoryId: "t1", source: "os", summary: "editor active" },
        { kind: "action", ts: 3, trajectoryId: "t1", tool: "device", summary: "tapped button:save" },
        { kind: "action", ts: 4, trajectoryId: "t1", tool: "editor", summary: "typed hello" },
        { kind: "action", ts: 5, trajectoryId: "t2", tool: "device", summary: "swiped left" },
      ],
    };
    const sequences = movementSequencesFromReplay(manifest);
    expect(sequences).toHaveLength(2);
    const t1 = sequences.find((s) => s.id.endsWith("t1"));
    expect(t1?.steps).toHaveLength(2);
    expect(t1?.context).toBe("s1");
    // device tool -> gesture actor, normalized verb.
    expect(t1?.steps[0]).toMatchObject({ actor: "gesture", action: "tap", target: "device" });
    expect(t1?.steps[1]).toMatchObject({ actor: "tool", action: "type", target: "editor" });
  });

  it("returns no sequences when there are no action events", () => {
    const manifest: ReplayManifest = {
      version: 1,
      sessionId: "s2",
      trajectoryIds: [],
      eventCount: 1,
      events: [{ kind: "observation", ts: 1, trajectoryId: "t1", source: "os", summary: "idle" }],
    };
    expect(movementSequencesFromReplay(manifest)).toEqual([]);
  });
});

describe("movementStepFromDeviceGesture", () => {
  it("maps gesture kinds to movement actions", () => {
    expect(movementStepFromDeviceGesture({ kind: "tap", target: "button:ok" })).toMatchObject({
      actor: "gesture",
      action: "click",
      target: "button:ok",
    });
    expect(movementStepFromDeviceGesture({ kind: "swipe", direction: "left" })).toMatchObject({
      action: "swipe",
      direction: "left",
    });
    expect(movementStepFromDeviceGesture({ kind: "shortcut", valueSummary: "cmd+s" })).toMatchObject({
      action: "shortcut",
      value: "cmd+s",
    });
  });
});
