import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { OperatorCronService } from "./cron-service.js";
import { StandaloneOperatorRuntime } from "../orchestrator/operator-runtime.js";

const tempDirs: string[] = [];

async function makeTempDir(): Promise<string> {
  const dir = await fs.mkdtemp(path.join(os.tmpdir(), "cron-service-"));
  tempDirs.push(dir);
  return dir;
}

afterEach(async () => {
  await Promise.all(tempDirs.splice(0).map((dir) => fs.rm(dir, { recursive: true, force: true })));
});

describe("OperatorCronService", () => {
  it("creates runtime-backed cron sessions and runs", async () => {
    const rootDir = await makeTempDir();
    const runtime = new StandaloneOperatorRuntime({ rootDir });
    const service = new OperatorCronService(rootDir, { runtime });
    const recurring = await service.createJob({ cron: "* * * * *", prompt: "do recurring" });
    const oneShot = await service.createJob({ cron: "* * * * *", prompt: "do once", recurring: false });

    await expect(service.listJobs()).resolves.toHaveLength(2);
    const runs = await service.tick(new Date(Date.now() + 120_000));
    expect(runs).toHaveLength(2);
    expect(runs[0]).toMatchObject({ status: "completed", outcome: "success", sessionId: expect.any(String), operatorRunId: expect.any(String) });
    await expect(service.listRuns()).resolves.toHaveLength(2);

    const jobsAfterTick = await service.listJobs();
    const recurringAfterTick = jobsAfterTick.find((job) => job.id === recurring.id);
    expect(recurringAfterTick).toMatchObject({ lastStatus: "success", lastRunAt: expect.any(String), nextRunAt: expect.any(String) });
    expect(jobsAfterTick.map((job) => job.id)).not.toContain(oneShot.id);

    const sessions = await runtime.listSessions();
    expect(sessions).toHaveLength(2);
    expect(sessions.every((session) => session.status === "completed")).toBe(true);

    const runtimeRuns = await runtime.listRuns();
    expect(runtimeRuns).toHaveLength(2);
    expect(runtimeRuns.every((run) => run.status === "completed")).toBe(true);

    const transcript = await runtime.getTranscript(runs[0]?.sessionId ?? "missing");
    expect(transcript.some((record) => "message" in record && record.message.content === "do recurring")).toBe(true);

    await expect(service.deleteJob(recurring.id)).resolves.toBe(true);
    await expect(service.listJobs()).resolves.toEqual([]);
  });

  it("persists failed cron outcomes and recovers interrupted runs", async () => {
    const rootDir = await makeTempDir();
    const runtime = new StandaloneOperatorRuntime({ rootDir });
    const service = new OperatorCronService(rootDir, {
      runtime,
      dispatchCronJob: async () => {
        throw new Error("cron dispatch failed");
      },
    });
    const job = await service.createJob({ cron: "* * * * *", prompt: "break cron" });

    const [failedRun] = await service.tick(new Date(Date.now() + 120_000));
    expect(failedRun).toMatchObject({ status: "failed", outcome: "error", error: "cron dispatch failed" });
    const failedJob = await service.store.getJob(job.id);
    expect(failedJob).toMatchObject({ lastStatus: "error", lastError: "cron dispatch failed" });

    const recoveryRoot = await makeTempDir();
    const recoveryRuntime = new StandaloneOperatorRuntime({ rootDir: recoveryRoot });
    const recoveryService = new OperatorCronService(recoveryRoot, { runtime: recoveryRuntime });
    const recoveryJob = await recoveryService.createJob({ cron: "* * * * *", prompt: "recover me" });
    await recoveryService.store.updateJob(recoveryJob.id, {
      nextRunAt: new Date(Date.now() + 600_000).toISOString(),
    });
    const session = await recoveryRuntime.startSession({ title: "stale", agentId: "operator-cron" });
    const run = await recoveryRuntime.startRun({ sessionId: session.id, title: "stale cron" });
    const staleRun = await recoveryService.store.addRun({
      jobId: recoveryJob.id,
      status: "running",
      sessionId: session.id,
      operatorRunId: run.id,
      startedAt: new Date(Date.now() - 60_000).toISOString(),
    });

    await recoveryService.tick(new Date(Date.now() + 120_000));
    const recovered = await recoveryService.store.getRun(staleRun.id);
    expect(recovered).toMatchObject({ status: "failed", outcome: "interrupted", error: "Interrupted before completion." });
    await expect(recoveryRuntime.getRun(run.id)).resolves.toMatchObject({ status: "failed" });
    await expect(recoveryRuntime.getSession(session.id)).resolves.toMatchObject({ status: "failed" });
    await expect(recoveryService.store.getJob(recoveryJob.id)).resolves.toMatchObject({ lastStatus: "interrupted" });
  });
});
