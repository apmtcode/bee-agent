import { describe, expect, it } from "vitest";
import {
  DEFAULT_SYNTHETIC_FLOWS,
  extractMovementSequences,
  generateSyntheticMovementSequences,
  sequencesFromTrajectories,
} from "./movement-dataset.js";
import { MarkovMovementBackend, MovementModelInference, evaluateMovementModel } from "./movement-model.js";
import type { ReplayManifest } from "../capture/replay.js";
import { buildTrajectorySpan } from "../capture/trajectory.js";

describe("extractMovementSequences", () => {
  it("keeps only action events, grouped per trajectory in timestamp order", () => {
    const replay: Pick<ReplayManifest, "events"> = {
      events: [
        { kind: "action", ts: 30, trajectoryId: "t1", tool: "mouse.click", summary: "c" },
        { kind: "transcript", ts: 5, messageId: "m", role: "user", content: "hi" },
        { kind: "action", ts: 10, trajectoryId: "t1", tool: "window.focus", summary: "f" },
        { kind: "observation", ts: 15, trajectoryId: "t1", source: "os", summary: "o" },
        { kind: "action", ts: 20, trajectoryId: "t2", tool: "key.type", summary: "k" },
      ],
    };
    const sequences = extractMovementSequences([replay]);
    const t1 = sequences.find((s) => s.trajectoryId === "t1");
    expect(t1?.steps.map((s) => s.tool)).toEqual(["window.focus", "mouse.click"]);
    const t2 = sequences.find((s) => s.trajectoryId === "t2");
    expect(t2?.steps.map((s) => s.tool)).toEqual(["key.type"]);
  });
});

describe("sequencesFromTrajectories", () => {
  it("orders actions by timestamp", () => {
    const span = buildTrajectorySpan({
      id: "t1",
      sessionId: "s1",
      actions: [
        { kind: "action", tool: "b", summary: "b", ts: 20 },
        { kind: "action", tool: "a", summary: "a", ts: 10 },
      ],
    });
    const [sequence] = sequencesFromTrajectories([span]);
    expect(sequence?.steps.map((s) => s.tool)).toEqual(["a", "b"]);
  });
});

describe("generateSyntheticMovementSequences", () => {
  it("is deterministic for a fixed seed", () => {
    const a = generateSyntheticMovementSequences({ seed: 42, count: 8 });
    const b = generateSyntheticMovementSequences({ seed: 42, count: 8 });
    expect(a).toEqual(b);
    expect(a).toHaveLength(8);
  });

  it("differs across seeds", () => {
    const a = generateSyntheticMovementSequences({ seed: 1, count: 8 });
    const b = generateSyntheticMovementSequences({ seed: 2, count: 8 });
    expect(a).not.toEqual(b);
  });

  it("only emits tokens from the flow vocabulary", () => {
    const vocab = new Set(DEFAULT_SYNTHETIC_FLOWS.flatMap((flow) => flow.tools));
    const sequences = generateSyntheticMovementSequences({ seed: 7, count: 10 });
    for (const sequence of sequences) {
      for (const step of sequence.steps) {
        expect(vocab.has(step.tool)).toBe(true);
      }
    }
  });
});

describe("end-to-end: synthetic dataset → train → generalize", () => {
  it("trains on synthetic movements and generalizes to a held-out split", async () => {
    const all = generateSyntheticMovementSequences({ seed: 123, count: 40 });
    const train = all.slice(0, 30);
    const heldOut = all.slice(30);

    const backend = new MarkovMovementBackend();
    const artifact = await backend.train({ jobId: "synthetic", mode: "sft", order: 2, sequences: train });
    const inference = new MovementModelInference(artifact);

    const report = evaluateMovementModel(inference, heldOut);
    // Flows share vocabulary and structure, so a backed-off model should
    // predict a healthy majority of held-out movements correctly.
    expect(report.steps).toBeGreaterThan(0);
    expect(report.accuracy).toBeGreaterThan(0.6);
  });
});
