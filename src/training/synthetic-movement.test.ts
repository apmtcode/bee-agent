import { describe, expect, it } from "vitest";
import {
  DEFAULT_SYNTHETIC_FLOWS,
  generateSyntheticMovementDataset,
  splitMovementDataset,
  syntheticSequenceToTrajectory,
} from "./synthetic-movement.js";
import { buildMovementDatasetFromTrajectories } from "./movement-model.js";

describe("generateSyntheticMovementDataset", () => {
  it("is reproducible for a fixed seed", () => {
    const a = generateSyntheticMovementDataset({ sequenceCount: 25, seed: 5 });
    const b = generateSyntheticMovementDataset({ sequenceCount: 25, seed: 5 });
    expect(a).toEqual(b);
  });

  it("produces different datasets for different seeds", () => {
    const a = generateSyntheticMovementDataset({ sequenceCount: 25, seed: 1 });
    const b = generateSyntheticMovementDataset({ sequenceCount: 25, seed: 2 });
    expect(a).not.toEqual(b);
  });

  it("emits the requested number of well-formed sequences", () => {
    const dataset = generateSyntheticMovementDataset({ sequenceCount: 10, seed: 3 });
    expect(dataset.sequences).toHaveLength(10);
    for (const sequence of dataset.sequences) {
      expect(sequence.steps.length).toBe(DEFAULT_SYNTHETIC_FLOWS[0]!.stages.length);
      for (const step of sequence.steps) {
        expect(step.channel).toBeTruthy();
        expect(step.verb).toBeTruthy();
      }
    }
  });
});

describe("splitMovementDataset", () => {
  it("partitions without overlap and covers the whole dataset", () => {
    const dataset = generateSyntheticMovementDataset({ sequenceCount: 40, seed: 9 });
    const { train, heldOut } = splitMovementDataset(dataset, 0.25);
    expect(train.sequences.length + heldOut.sequences.length).toBe(40);
    const heldOutIds = new Set(heldOut.sequences.map((s) => s.id));
    for (const sequence of train.sequences) {
      expect(heldOutIds.has(sequence.id)).toBe(false);
    }
    expect(heldOut.sequences.length).toBeGreaterThan(0);
  });

  it("keeps everything in train when the hold-out fraction is zero", () => {
    const dataset = generateSyntheticMovementDataset({ sequenceCount: 10, seed: 1 });
    const { train, heldOut } = splitMovementDataset(dataset, 0);
    expect(train.sequences).toHaveLength(10);
    expect(heldOut.sequences).toHaveLength(0);
  });
});

describe("syntheticSequenceToTrajectory", () => {
  it("renders a trajectory that tokenizes back to the source steps", () => {
    const dataset = generateSyntheticMovementDataset({ sequenceCount: 1, seed: 11 });
    const sequence = dataset.sequences[0]!;
    const trajectory = syntheticSequenceToTrajectory(sequence, { baseTs: 1000 });
    expect(trajectory.actions).toHaveLength(sequence.steps.length);

    const rebuilt = buildMovementDatasetFromTrajectories([trajectory]);
    // Qualifiers become bucketed targets; verbs and channels must survive the round-trip.
    const rebuiltSteps = rebuilt.sequences[0]!.steps;
    expect(rebuiltSteps.map((s) => s.channel)).toEqual(sequence.steps.map((s) => s.channel));
    expect(rebuiltSteps.map((s) => s.verb)).toEqual(sequence.steps.map((s) => s.verb));
  });
});
