import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { OperatorCronService } from "./cron-service.js";
import { OperatorDeliveryService } from "./delivery.js";
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
    const recurring = await service.createJob({
      cron: "* * * * *",
      prompt: "do recurring",
      delivery: { onSuccess: [{ kind: "local" }] },
    });
    const oneShot = await service.createJob({ cron: "* * * * *", prompt: "do once", recurring: false });

    await expect(service.listJobs()).resolves.toHaveLength(2);
    const runs = await service.tick(new Date(Date.now() + 120_000));
    expect(runs).toHaveLength(2);
    expect(runs[0]).toMatchObject({ status: "completed", outcome: "success", sessionId: expect.any(String), operatorRunId: expect.any(String) });
    await expect(service.listRuns()).resolves.toHaveLength(2);
    expect(runs[0]?.deliveryResults).toMatchObject([{ status: "sent", target: { kind: "local" } }]);

    const jobsAfterTick = await service.listJobs();
    const recurringAfterTick = jobsAfterTick.find((job) => job.id === recurring.id);
    expect(recurringAfterTick).toMatchObject({
      lastStatus: "success",
      lastRunAt: expect.any(String),
      nextRunAt: expect.any(String),
      lastDeliveryStatus: "sent",
      lastDeliveryAt: expect.any(String),
    });
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

  it("persists explicit cron model selection overrides across jobs, runs, sessions, and operator runs", async () => {
    const rootDir = await makeTempDir();
    const runtime = new StandaloneOperatorRuntime({ rootDir });
    const service = new OperatorCronService(rootDir, {
      runtime,
      modelSelection: {
        primary: "claude-haiku-4-5-20251001",
        fallbacks: ["claude-sonnet-4-6"],
        source: "default",
      },
    });
    const job = await service.createJob({
      cron: "* * * * *",
      prompt: "route this cron",
      modelSelection: {
        primary: "claude-opus-4-7",
        fallbacks: ["claude-sonnet-4-6", "claude-haiku-4-5-20251001"],
      },
    });

    const [run] = await service.tick(new Date(Date.now() + 120_000));
    expect(run).toMatchObject({
      status: "completed",
      outcome: "success",
      modelSelection: {
        primary: "claude-opus-4-7",
        fallbacks: ["claude-sonnet-4-6", "claude-haiku-4-5-20251001"],
        source: "job",
      },
    });

    const session = await runtime.getSession(run?.sessionId ?? "missing");
    expect(session).toMatchObject({
      metadata: {
        modelSelection: {
          primary: "claude-opus-4-7",
          fallbacks: ["claude-sonnet-4-6", "claude-haiku-4-5-20251001"],
          source: "job",
        },
      },
    });

    const operatorRun = await runtime.getRun(run?.operatorRunId ?? "missing");
    expect(operatorRun).toMatchObject({
      metadata: {
        trigger: "cron",
        modelSelection: {
          primary: "claude-opus-4-7",
          fallbacks: ["claude-sonnet-4-6", "claude-haiku-4-5-20251001"],
          source: "job",
        },
      },
    });

    const storedJob = await service.store.getJob(job.id);
    expect(storedJob).toMatchObject({
      modelSelection: {
        primary: "claude-opus-4-7",
        fallbacks: ["claude-sonnet-4-6", "claude-haiku-4-5-20251001"],
      },
    });
  });

  it("inherits base cron model selection when the job has no override", async () => {
    const rootDir = await makeTempDir();
    const runtime = new StandaloneOperatorRuntime({ rootDir });
    const service = new OperatorCronService(rootDir, {
      runtime,
      modelSelection: {
        primary: "claude-opus-4-7",
        fallbacks: ["claude-sonnet-4-6", "claude-haiku-4-5-20251001"],
        source: "default",
      },
    });
    await service.createJob({
      cron: "* * * * *",
      prompt: "inherit this cron",
    });

    const [run] = await service.tick(new Date(Date.now() + 120_000));
    expect(run).toMatchObject({
      modelSelection: {
        primary: "claude-opus-4-7",
        fallbacks: ["claude-sonnet-4-6", "claude-haiku-4-5-20251001"],
        source: "default",
      },
    });
    await expect(runtime.getSession(run?.sessionId ?? "missing")).resolves.toMatchObject({
      metadata: {
        modelSelection: {
          primary: "claude-opus-4-7",
          fallbacks: ["claude-sonnet-4-6", "claude-haiku-4-5-20251001"],
          source: "default",
        },
      },
    });
    await expect(runtime.getRun(run?.operatorRunId ?? "missing")).resolves.toMatchObject({
      metadata: {
        modelSelection: {
          primary: "claude-opus-4-7",
          fallbacks: ["claude-sonnet-4-6", "claude-haiku-4-5-20251001"],
          source: "default",
        },
      },
    });
  });

  it("preserves fallback-only cron overrides and explicit empty fallback overrides", async () => {
    const rootDir = await makeTempDir();
    const runtime = new StandaloneOperatorRuntime({ rootDir });
    const service = new OperatorCronService(rootDir, {
      runtime,
      modelSelection: {
        primary: "claude-opus-4-7",
        fallbacks: ["claude-sonnet-4-6", "claude-haiku-4-5-20251001"],
        source: "default",
      },
    });
    const overrideJob = await service.createJob({
      cron: "* * * * *",
      prompt: "override fallbacks",
      modelSelection: {
        fallbacks: ["claude-haiku-4-5-20251001"],
      },
    });
    const clearJob = await service.createJob({
      cron: "* * * * *",
      prompt: "clear fallbacks",
      modelSelection: {
        fallbacks: [],
      },
    });

    const runs = await service.tick(new Date(Date.now() + 120_000));
    const overrideRun = runs.find((run) => run.jobId === overrideJob.id);
    const clearRun = runs.find((run) => run.jobId === clearJob.id);

    expect(overrideRun).toMatchObject({
      modelSelection: {
        primary: "claude-opus-4-7",
        fallbacks: ["claude-haiku-4-5-20251001"],
        source: "job",
      },
    });
    await expect(runtime.getSession(overrideRun?.sessionId ?? "missing")).resolves.toMatchObject({
      metadata: {
        modelSelection: {
          primary: "claude-opus-4-7",
          fallbacks: ["claude-haiku-4-5-20251001"],
          source: "job",
        },
      },
    });
    await expect(runtime.getRun(overrideRun?.operatorRunId ?? "missing")).resolves.toMatchObject({
      metadata: {
        modelSelection: {
          primary: "claude-opus-4-7",
          fallbacks: ["claude-haiku-4-5-20251001"],
          source: "job",
        },
      },
    });

    expect(clearRun).toMatchObject({
      modelSelection: {
        primary: "claude-opus-4-7",
        fallbacks: [],
        source: "job",
      },
    });
    await expect(runtime.getSession(clearRun?.sessionId ?? "missing")).resolves.toMatchObject({
      metadata: {
        modelSelection: {
          primary: "claude-opus-4-7",
          fallbacks: [],
          source: "job",
        },
      },
    });
    await expect(runtime.getRun(clearRun?.operatorRunId ?? "missing")).resolves.toMatchObject({
      metadata: {
        modelSelection: {
          primary: "claude-opus-4-7",
          fallbacks: [],
          source: "job",
        },
      },
    });

    await expect(service.store.getJob(clearJob.id)).resolves.toMatchObject({
      modelSelection: {
        fallbacks: [],
      },
    });
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
    const job = await service.createJob({
      cron: "* * * * *",
      prompt: "break cron",
      delivery: { onFailure: [{ kind: "local" }] },
    });

    const [failedRun] = await service.tick(new Date(Date.now() + 120_000));
    expect(failedRun).toMatchObject({
      status: "failed",
      outcome: "error",
      error: "cron dispatch failed",
      deliveryResults: [{ status: "sent", target: { kind: "local" } }],
    });
    const failedJob = await service.store.getJob(job.id);
    expect(failedJob).toMatchObject({
      lastStatus: "error",
      lastError: "cron dispatch failed",
      lastDeliveryStatus: "sent",
    });

    const recoveryRoot = await makeTempDir();
    const recoveryRuntime = new StandaloneOperatorRuntime({ rootDir: recoveryRoot });
    const recoveryDelivery = new OperatorDeliveryService(recoveryRoot, {
      fetchImpl: async () => new Response(null, { status: 204 }),
    });
    const recoveryService = new OperatorCronService(recoveryRoot, { runtime: recoveryRuntime, delivery: recoveryDelivery });
    const recoveryJob = await recoveryService.createJob({
      cron: "* * * * *",
      prompt: "recover me",
      delivery: { onFailure: [{ kind: "webhook", url: "https://example.test/cron" }] },
    });
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
    expect(recovered).toMatchObject({
      status: "failed",
      outcome: "interrupted",
      error: "Interrupted before completion.",
      deliveryResults: [{ status: "sent", target: { kind: "webhook", url: "https://example.test/cron" } }],
    });
    await expect(recoveryRuntime.getRun(run.id)).resolves.toMatchObject({ status: "failed" });
    await expect(recoveryRuntime.getSession(session.id)).resolves.toMatchObject({ status: "failed" });
    await expect(recoveryService.store.getJob(recoveryJob.id)).resolves.toMatchObject({ lastStatus: "interrupted", lastDeliveryStatus: "sent" });
  });
});
