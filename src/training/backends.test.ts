import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import {
  SimulatedTrainingBackend,
  canonicalTrainingInput,
  createDefaultTrainingBackends,
  createSimulatedTrainingBackends,
  fingerprintTrainingInput,
} from "./backends.js";
import { createLocalTrainingExecution, createLocalTrainingJobManifest } from "./job-manifest.js";
import { LocalAppleSiliconTrainingRunner } from "./runner.js";
import type { ReviewedExportManifest } from "./export-manifest.js";

const tempDirs: string[] = [];

async function makeTempDir(): Promise<string> {
  const dir = await fs.mkdtemp(path.join(os.tmpdir(), "training-backends-"));
  tempDirs.push(dir);
  return dir;
}

afterEach(async () => {
  await Promise.all(tempDirs.splice(0).map((dir) => fs.rm(dir, { recursive: true, force: true })));
});

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

function makeJob(mode: "sft" | "rl") {
  const job = createLocalTrainingJobManifest({ id: `job-${mode}`, exportManifest, mode });
  const execution = createLocalTrainingExecution({ jobId: job.id, mode: job.mode });
  return { job, execution };
}

describe("training backends", () => {
  it("default registry preserves the mlx/axolotl runtimes", () => {
    const backends = createDefaultTrainingBackends();
    expect(backends.sft.runtime).toBe("mlx");
    expect(backends.rl.runtime).toBe("axolotl");

    const { job, execution } = makeJob("sft");
    const plan = backends.sft.planExecution({ job, execution });
    expect(plan.outputFileName).toBe("model.gguf");
    expect(plan.command).toContain("mlx_lm.lora");
  });

  it("injecting simulated backends changes the runner plan without external toolchain", () => {
    const runner = new LocalAppleSiliconTrainingRunner("/root", createSimulatedTrainingBackends());
    const { job, execution } = makeJob("sft");
    const plan = runner.buildPlan(job, execution);

    expect(plan.runtime).toBe("simulated");
    expect(plan.targetPlatform).toBe("simulated");
    expect(plan.outputPath).toBe(`${execution.artifactDir}/model.sim.json`);
    expect(plan.command[0]).toBe("node");
    expect(plan.command).not.toContain("python3");
    expect(plan.environment.OPENCLAW_TRAINING_SIMULATED).toBe("true");
    expect(plan.environment.OPENCLAW_TRAINING_RUNTIME).toBe("simulated");
  });

  it("fingerprint is deterministic for identical inputs and differs across modes", () => {
    const sft = makeJob("sft");
    const rl = makeJob("rl");
    const sftFp = fingerprintTrainingInput(canonicalTrainingInput(sft.job));
    const rlFp = fingerprintTrainingInput(canonicalTrainingInput(rl.job));

    expect(sftFp).toMatch(/^[0-9a-f]{8}$/);
    expect(fingerprintTrainingInput(canonicalTrainingInput(sft.job))).toBe(sftFp);
    expect(rlFp).not.toBe(sftFp);
  });

  it("simulate() runs the pipeline in-process, writing a deterministic artifact + completed state", async () => {
    const rootDir = await makeTempDir();
    const backend = new SimulatedTrainingBackend();
    const { job, execution } = makeJob("sft");

    const first = await backend.simulate(rootDir, { job, execution });
    const artifact = JSON.parse(await fs.readFile(first.artifactPath, "utf8"));
    const state = JSON.parse(await fs.readFile(first.statePath, "utf8"));

    expect(first.runtime).toBe("simulated");
    expect(artifact).toMatchObject({
      jobId: job.id,
      mode: "sft",
      runtime: "simulated",
      fingerprint: first.fingerprint,
    });
    expect(state).toMatchObject({
      jobId: job.id,
      status: "completed",
      exitCode: 0,
      outputModelRef: `${execution.artifactDir}/model.sim.json`,
    });

    // Re-running with identical inputs is reproducible.
    const second = await backend.simulate(rootDir, { job, execution });
    expect(second.fingerprint).toBe(first.fingerprint);
  });
});
