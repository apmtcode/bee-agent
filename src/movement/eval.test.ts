import { describe, expect, it } from "vitest";
import { MarkovMovementBackend } from "./backend.js";
import { createCoordinateQuantizer, type MovementTrajectory } from "./events.js";
import { generateMovementDataset, type SyntheticTaskSpec } from "./synthetic.js";
import {
  evaluateNextStepAccuracy,
  evaluateReplayFidelity,
  longestCommonSubsequence,
} from "./eval.js";

const quantizer = createCoordinateQuantizer({ width: 1280, height: 800, cols: 16, rows: 10 });

describe("longestCommonSubsequence", () => {
  it("computes overlap length and handles empty inputs", () => {
    expect(longestCommonSubsequence(["a", "b", "c"], ["a", "x", "c"])).toBe(2);
    expect(longestCommonSubsequence([], ["a"])).toBe(0);
    expect(longestCommonSubsequence(["a", "b"], ["a", "b"])).toBe(2);
  });
});

describe("movement round-trip: repeat recorded movements (objective 2c)", () => {
  it("reproduces a single recorded trajectory with perfect fidelity", async () => {
    const dataset = generateMovementDataset({
      seed: 11,
      quantizer,
      specs: [{ task: "open-and-type", target: { x: 300, y: 200, windowId: "editor" }, typeLength: 3 }],
    });
    const model = await new MarkovMovementBackend().train(dataset, { maxOrder: 4 });

    const nextStep = evaluateNextStepAccuracy(model, quantizer, dataset.trajectories);
    const replay = evaluateReplayFidelity(model, quantizer, dataset.trajectories);

    expect(nextStep.accuracy).toBe(1);
    expect(replay.meanFidelity).toBe(1);
    expect(replay.perTrajectory[0]!.exact).toBe(true);
  });
});

describe("movement round-trip: generalize to related movements (objective 2d)", () => {
  const trainSpecs: SyntheticTaskSpec[] = [
    { task: "open-and-type", target: { x: 300, y: 200, windowId: "editor" }, typeLength: 2 },
    { task: "open-and-type", target: { x: 300, y: 200, windowId: "editor" }, typeLength: 4 },
    { task: "menu-select", target: { x: 100, y: 500, windowId: "menu" } },
  ];

  it("recomposes learned sub-movements to replay a novel-but-related trajectory", async () => {
    const train = generateMovementDataset({ seed: 21, quantizer, specs: trainSpecs });
    const model = await new MarkovMovementBackend().train(train, { maxOrder: 4 });

    // Held-out: same task/window/target region, an unseen type length (3).
    const heldOut = generateMovementDataset({
      seed: 77,
      quantizer,
      specs: [{ task: "open-and-type", target: { x: 300, y: 200, windowId: "editor" }, typeLength: 3 }],
    }).trajectories;

    const fidelity = evaluateReplayFidelity(model, quantizer, heldOut);
    const nextStep = evaluateNextStepAccuracy(model, quantizer, heldOut);

    // Generalizes by recombining familiar sub-movements, not memorized verbatim.
    expect(fidelity.meanFidelity).toBeGreaterThanOrEqual(0.6);
    expect(nextStep.accuracy).toBeGreaterThanOrEqual(0.6);

    // Correct intent must replay better than a mislabeled (cross-task) intent.
    const mislabeled: MovementTrajectory[] = heldOut.map((trajectory) => ({
      ...trajectory,
      label: "menu-select",
    }));
    const mislabeledFidelity = evaluateReplayFidelity(model, quantizer, mislabeled);
    expect(fidelity.meanFidelity).toBeGreaterThan(mislabeledFidelity.meanFidelity);
  });
});
