import { describe, expect, it } from "vitest";
import {
  AppleSiliconTrainingBackend,
  MockLocalTrainingBackend,
  predictNextMovement,
  replayMovementSequence,
  trainMockMovementModel,
} from "./backends.js";
import { createLocalTrainingExecution, createLocalTrainingJobManifest } from "./job-manifest.js";
import type { ReviewedExportManifest } from "./export-manifest.js";

const exportManifest: ReviewedExportManifest = {
  version: 1,
  createdAt: "2026-01-01T00:00:00.000Z",
  reviewedBy: "operator",
  purpose: "local fine-tuning",
  targetPlatform: "apple-silicon",
  modes: ["sft", "rl"],
  rawCaptureIncluded: false,
  executableSkills: [],
  executableSkillRuns: [],
  promotedSkills: [],
  memories: [],
  trajectories: [],
  replays: [],
};

function makeJob(mode: "sft" | "rl", id = `job-${mode}`) {
  const job = createLocalTrainingJobManifest({ id, exportManifest, mode });
  const execution = createLocalTrainingExecution({ jobId: job.id, mode: job.mode });
  return { job, execution };
}

describe("AppleSiliconTrainingBackend", () => {
  it("describes the mlx SFT runtime", () => {
    const { job, execution } = makeJob("sft");
    const contribution = new AppleSiliconTrainingBackend().describe(job, execution);
    expect(contribution.runtime).toBe("mlx");
    expect(contribution.targetPlatform).toBe("apple-silicon");
    expect(contribution.outputFileName).toBe("model.gguf");
    expect(contribution.command[0]).toBe("python3");
    expect(contribution.command).toContain("mlx_lm.lora");
    expect(contribution.environment.OPENCLAW_TRAINING_RUNTIME).toBe("mlx");
  });

  it("describes the axolotl RL runtime", () => {
    const { job, execution } = makeJob("rl");
    const contribution = new AppleSiliconTrainingBackend().describe(job, execution);
    expect(contribution.runtime).toBe("axolotl");
    expect(contribution.outputFileName).toBe("policy.gguf");
    expect(contribution.command).toContain("axolotl.cli.train");
    expect(contribution.environment.OPENCLAW_TRAINING_RUNTIME).toBe("axolotl");
  });
});

describe("MockLocalTrainingBackend", () => {
  it("produces a portable, node-runnable command with no python dependency", () => {
    const { job, execution } = makeJob("sft");
    const contribution = new MockLocalTrainingBackend().describe(job, execution);
    expect(contribution.runtime).toBe("mock");
    expect(contribution.targetPlatform).toBe("portable");
    expect(contribution.outputFileName).toBe("model.json");
    expect(contribution.command[0]).toBe("node");
    expect(contribution.command).not.toContain("python3");
    expect(contribution.command).toContain("--dataset");
    expect(contribution.command).toContain(execution.datasetDir);
    expect(contribution.environment.BEE_TRAINING_RUNTIME).toBe("mock");
  });

  it("names the RL artifact policy.json", () => {
    const { job, execution } = makeJob("rl");
    const contribution = new MockLocalTrainingBackend().describe(job, execution);
    expect(contribution.outputFileName).toBe("policy.json");
  });
});

describe("mock movement model", () => {
  it("trains a deterministic first-order markov model", () => {
    const sequences = [
      ["window:focus", "mousedown:left", "mouseup:left", "key:Enter"],
      ["window:focus", "mousedown:left", "mouseup:left", "key:Enter"],
      ["window:focus", "key:Escape"],
    ];
    const a = trainMockMovementModel(sequences);
    const b = trainMockMovementModel(sequences);
    expect(a).toEqual(b); // determinism

    expect(a.sequenceCount).toBe(3);
    expect(a.starts["window:focus"]).toBe(3);
    expect(a.transitions["window:focus"]).toEqual({ "key:Escape": 1, "mousedown:left": 2 });
    expect(a.vocabulary).toContain("key:Enter");
  });

  it("predicts the highest-probability next movement and returns undefined for unseen states", () => {
    const model = trainMockMovementModel([
      ["a", "b", "c"],
      ["a", "b", "c"],
      ["a", "x"],
    ]);
    expect(predictNextMovement(model, ["a"])).toBe("b"); // 2 vs 1
    expect(predictNextMovement(model, ["b"])).toBe("c");
    expect(predictNextMovement(model, ["unseen"])).toBeUndefined();
    // Empty context falls back to the most common start token.
    expect(predictNextMovement(model, [])).toBe("a");
  });

  it("replays a recorded movement sequence from the model", () => {
    const model = trainMockMovementModel([
      ["open", "type", "save", "close"],
      ["open", "type", "save", "close"],
    ]);
    expect(replayMovementSequence(model)).toEqual(["open", "type", "save", "close"]);
  });

  it("generalizes to a related but unrecorded movement path", () => {
    // Two disjoint recordings that share the token "menu". A first-order model
    // can compose a path never seen end-to-end: settings -> menu -> logout.
    const model = trainMockMovementModel([
      ["settings", "menu"],
      ["menu", "logout"],
    ]);
    // Starting from "settings", the model should route through the shared
    // "menu" state into the transition learned from the other recording.
    expect(replayMovementSequence(model, { start: "settings" })).toEqual([
      "settings",
      "menu",
      "logout",
    ]);
  });

  it("terminates on self-looping transitions instead of running unbounded", () => {
    const model = trainMockMovementModel([["scroll", "scroll", "scroll", "scroll", "scroll"]]);
    const replay = replayMovementSequence(model, { start: "scroll", maxLength: 100 });
    expect(replay.length).toBeLessThan(100);
    expect(replay.every((token) => token === "scroll")).toBe(true);
  });
});
