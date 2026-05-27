import path from "node:path";
import { FileCronStore, type CronJob, type CronJobLastStatus, type CronRun } from "./cron-store.js";
import { JsonlTranscriptStore } from "../harness/transcript-store.js";
import type { StandaloneOperatorOptions, StandaloneOperatorRuntime } from "../orchestrator/operator-runtime.js";

function nowIso(): string {
  return new Date().toISOString();
}

function computeNextRunAt(from = new Date()): string {
  return new Date(from.getTime() + 60_000).toISOString();
}

function buildCronSessionTitle(prompt: string): string {
  const trimmed = prompt.trim();
  return `Cron: ${trimmed.length > 48 ? `${trimmed.slice(0, 45)}...` : trimmed}`;
}

function buildCronRunTitle(prompt: string): string {
  const trimmed = prompt.trim();
  return `Cron run: ${trimmed.length > 44 ? `${trimmed.slice(0, 41)}...` : trimmed}`;
}

function buildCronSummary(text: string): string {
  const trimmed = text.trim();
  return trimmed.length > 160 ? `${trimmed.slice(0, 157)}...` : trimmed;
}

function normalizeError(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

async function defaultCronDispatcher(job: CronJob): Promise<{ assistantText: string; summary?: string }> {
  const assistantText = `Cron executed prompt: ${job.prompt}`;
  return { assistantText, summary: buildCronSummary(assistantText) };
}

export type OperatorCronDispatcher = (job: CronJob, context: { run: CronRun; runtime: StandaloneOperatorRuntime }) => Promise<{
  assistantText: string;
  summary?: string;
}>;

export class OperatorCronService {
  readonly store: FileCronStore;
  private readonly runtime?: StandaloneOperatorRuntime;
  private readonly dispatchCronJob: OperatorCronDispatcher;

  constructor(
    rootDirOrOptions: string | StandaloneOperatorOptions,
    options: {
      runtime?: StandaloneOperatorRuntime;
      dispatchCronJob?: OperatorCronDispatcher;
    } = {},
  ) {
    const rootDir = typeof rootDirOrOptions === "string" ? rootDirOrOptions : rootDirOrOptions.rootDir;
    this.store = new FileCronStore(path.join(rootDir, "cron.json"));
    this.runtime = options.runtime;
    this.dispatchCronJob = options.dispatchCronJob ?? defaultCronDispatcher;
  }

  async listJobs(): Promise<CronJob[]> {
    return await this.store.listJobs();
  }

  async createJob(params: {
    cron: string;
    prompt: string;
    recurring?: boolean;
    durable?: boolean;
  }): Promise<CronJob> {
    return await this.store.createJob({
      ...params,
      nextRunAt: computeNextRunAt(),
    });
  }

  async deleteJob(jobId: string): Promise<boolean> {
    return await this.store.deleteJob(jobId);
  }

  async tick(now = new Date()): Promise<CronRun[]> {
    await this.recoverInterruptedRuns();
    const jobs = await this.store.listJobs();
    const readyJobs = jobs.filter((job) => !job.nextRunAt || Date.parse(job.nextRunAt) <= now.getTime());
    const runs: CronRun[] = [];
    for (const job of readyJobs) {
      runs.push(await this.dispatchJob(job, now));
    }
    return runs;
  }

  async listRuns(jobId?: string): Promise<CronRun[]> {
    return await this.store.listRuns(jobId);
  }

  private async recoverInterruptedRuns(): Promise<void> {
    if (!this.runtime) {
      return;
    }
    const runs = await this.store.listRuns();
    const staleRuns = runs.filter((run) => run.status === "scheduled" || run.status === "running");
    for (const run of staleRuns) {
      if (run.operatorRunId) {
        const linkedRun = await this.runtime.getRun(run.operatorRunId);
        if (linkedRun && linkedRun.status !== "completed" && linkedRun.status !== "failed") {
          await this.runtime.updateRunStatus(run.operatorRunId, "failed", {
            cronRunId: run.id,
            cronRecovery: "interrupted",
          });
        }
      }
      if (run.sessionId) {
        const session = await this.runtime.getSession(run.sessionId);
        if (session && session.status !== "completed" && session.status !== "failed") {
          await this.runtime.failSession(run.sessionId);
        }
      }
      await this.store.updateRun(run.id, {
        status: "failed",
        outcome: "interrupted",
        error: "Interrupted before completion.",
        finishedAt: nowIso(),
      });
      await this.updateJobOutcome(run.jobId, {
        lastStatus: "interrupted",
        lastError: "Interrupted before completion.",
      });
    }
  }

  private async dispatchJob(job: CronJob, now: Date): Promise<CronRun> {
    const scheduledAt = now.toISOString();
    const scheduled = await this.store.addRun({
      jobId: job.id,
      status: "scheduled",
      scheduledFor: scheduledAt,
    });
    await this.store.updateJob(job.id, {
      lastRunAt: scheduledAt,
      ...(job.recurring ? { nextRunAt: computeNextRunAt(now) } : {}),
    });

    if (!this.runtime) {
      const completed = await this.store.updateRun(scheduled.id, {
        status: "completed",
        startedAt: scheduledAt,
        finishedAt: nowIso(),
        outcome: "success",
        summary: buildCronSummary(`Cron executed prompt: ${job.prompt}`),
      });
      await this.updateJobOutcome(job.id, { lastStatus: "success", lastError: null });
      if (!job.recurring) {
        await this.store.deleteJob(job.id);
      }
      if (!completed) {
        throw new Error(`Unable to finalize cron run: ${scheduled.id}`);
      }
      return completed;
    }

    const session = await this.runtime.startSession({
      title: buildCronSessionTitle(job.prompt),
      agentId: "operator-cron",
    });
    const operatorRun = await this.runtime.startRun({
      sessionId: session.id,
      title: buildCronRunTitle(job.prompt),
      metadata: {
        trigger: "cron",
        cronJobId: job.id,
        cronRunId: scheduled.id,
      },
    });
    const running = await this.store.updateRun(scheduled.id, {
      status: "running",
      sessionId: session.id,
      operatorRunId: operatorRun.id,
      startedAt: nowIso(),
    });
    if (!running) {
      throw new Error(`Unable to start cron run: ${scheduled.id}`);
    }

    try {
      const dispatched = await this.dispatchCronJob(job, { run: running, runtime: this.runtime });
      await this.runtime.recordTurn({
        sessionId: session.id,
        userText: job.prompt,
        assistantText: dispatched.assistantText,
        trajectorySummary: dispatched.summary ?? dispatched.assistantText,
      });
      await this.runtime.updateRunStatus(operatorRun.id, "completed", {
        cronJobId: job.id,
        cronRunId: scheduled.id,
        trigger: "cron",
      });
      await this.runtime.completeSession(session.id);
      const completed = await this.store.updateRun(scheduled.id, {
        status: "completed",
        sessionId: session.id,
        operatorRunId: operatorRun.id,
        startedAt: running.startedAt,
        finishedAt: nowIso(),
        outcome: "success",
        error: null,
        summary: buildCronSummary(dispatched.summary ?? dispatched.assistantText),
      });
      await this.updateJobOutcome(job.id, { lastStatus: "success", lastError: null });
      if (!job.recurring) {
        await this.store.deleteJob(job.id);
      }
      if (!completed) {
        throw new Error(`Unable to finalize cron run: ${scheduled.id}`);
      }
      return completed;
    } catch (error) {
      const message = normalizeError(error);
      await this.runtime.updateRunStatus(operatorRun.id, "failed", {
        cronJobId: job.id,
        cronRunId: scheduled.id,
        trigger: "cron",
        error: message,
      });
      await this.runtime.failSession(session.id);
      const transcriptSession = await this.runtime.getSession(session.id);
      if (transcriptSession?.transcriptFile) {
        const transcript = new JsonlTranscriptStore(transcriptSession.transcriptFile);
        await transcript.appendMessage({
          role: "assistant",
          content: `Cron execution failed: ${message}`,
          metadata: { trigger: "cron", cronJobId: job.id, cronRunId: scheduled.id, status: "failed" },
        });
      }
      const failed = await this.store.updateRun(scheduled.id, {
        status: "failed",
        sessionId: session.id,
        operatorRunId: operatorRun.id,
        startedAt: running.startedAt,
        finishedAt: nowIso(),
        outcome: "error",
        error: message,
        summary: null,
      });
      await this.updateJobOutcome(job.id, { lastStatus: "error", lastError: message });
      if (!job.recurring) {
        await this.store.deleteJob(job.id);
      }
      if (!failed) {
        throw new Error(`Unable to finalize failed cron run: ${scheduled.id}`);
      }
      return failed;
    }
  }

  private async updateJobOutcome(
    jobId: string,
    params: { lastStatus: CronJobLastStatus; lastError: string | null },
  ): Promise<void> {
    await this.store.updateJob(jobId, {
      lastStatus: params.lastStatus,
      lastError: params.lastError,
    });
  }
}
