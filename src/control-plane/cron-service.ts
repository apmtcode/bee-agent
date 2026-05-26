import path from "node:path";
import { FileCronStore, type CronJob, type CronRun } from "./cron-store.js";
import type { StandaloneOperatorOptions } from "../orchestrator/operator-runtime.js";

function nowIso(): string {
  return new Date().toISOString();
}

function computeNextRunAt(from = new Date()): string {
  return new Date(from.getTime() + 60_000).toISOString();
}

export class OperatorCronService {
  readonly store: FileCronStore;

  constructor(rootDirOrOptions: string | StandaloneOperatorOptions) {
    const rootDir = typeof rootDirOrOptions === "string" ? rootDirOrOptions : rootDirOrOptions.rootDir;
    this.store = new FileCronStore(path.join(rootDir, "cron.json"));
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
    const jobs = await this.store.listJobs();
    const readyJobs = jobs.filter((job) => !job.nextRunAt || Date.parse(job.nextRunAt) <= now.getTime());
    const runs: CronRun[] = [];
    for (const job of readyJobs) {
      const scheduled = await this.store.addRun(job.id);
      const running = await this.store.updateRun(scheduled.id, {
        status: "running",
        startedAt: nowIso(),
      });
      const completed = await this.store.updateRun(scheduled.id, {
        status: "completed",
        startedAt: running?.startedAt,
        finishedAt: nowIso(),
      });
      await this.store.updateJob(job.id, {
        lastRunAt: nowIso(),
        ...(job.recurring ? { nextRunAt: computeNextRunAt(now) } : {}),
      });
      if (!job.recurring) {
        await this.store.deleteJob(job.id);
      }
      if (completed) {
        runs.push(completed);
      }
    }
    return runs;
  }

  async listRuns(jobId?: string): Promise<CronRun[]> {
    return await this.store.listRuns(jobId);
  }
}
