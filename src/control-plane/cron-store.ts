import { randomUUID } from "node:crypto";
import { readJsonFile, writeJsonAtomic } from "../shared/fs.js";

export type CronJob = {
  id: string;
  cron: string;
  prompt: string;
  recurring: boolean;
  durable: boolean;
  createdAt: string;
  lastRunAt?: string;
  nextRunAt?: string;
};

export type CronRun = {
  id: string;
  jobId: string;
  status: "scheduled" | "running" | "completed";
  createdAt: string;
  startedAt?: string;
  finishedAt?: string;
};

export type CronStoreShape = {
  version: 1;
  jobs: CronJob[];
  runs: CronRun[];
};

const EMPTY_STORE: CronStoreShape = {
  version: 1,
  jobs: [],
  runs: [],
};

function nowIso(): string {
  return new Date().toISOString();
}

export class FileCronStore {
  constructor(private readonly filePath: string) {}

  async load(): Promise<CronStoreShape> {
    return await readJsonFile(this.filePath, EMPTY_STORE);
  }

  async save(store: CronStoreShape): Promise<void> {
    await writeJsonAtomic(this.filePath, store);
  }

  async listJobs(): Promise<CronJob[]> {
    const store = await this.load();
    return [...store.jobs].sort((a, b) => a.createdAt.localeCompare(b.createdAt));
  }

  async getJob(jobId: string): Promise<CronJob | undefined> {
    const store = await this.load();
    return store.jobs.find((job) => job.id === jobId);
  }

  async createJob(params: {
    cron: string;
    prompt: string;
    recurring?: boolean;
    durable?: boolean;
    nextRunAt?: string;
  }): Promise<CronJob> {
    const store = await this.load();
    const job: CronJob = {
      id: randomUUID(),
      cron: params.cron,
      prompt: params.prompt,
      recurring: params.recurring ?? true,
      durable: params.durable ?? false,
      createdAt: nowIso(),
      ...(params.nextRunAt ? { nextRunAt: params.nextRunAt } : {}),
    };
    store.jobs.push(job);
    await this.save(store);
    return job;
  }

  async updateJob(jobId: string, params: { lastRunAt?: string; nextRunAt?: string }): Promise<CronJob | undefined> {
    const store = await this.load();
    const index = store.jobs.findIndex((job) => job.id === jobId);
    if (index < 0) {
      return undefined;
    }
    const updated: CronJob = {
      ...store.jobs[index],
      ...(params.lastRunAt ? { lastRunAt: params.lastRunAt } : {}),
      ...(params.nextRunAt ? { nextRunAt: params.nextRunAt } : {}),
    };
    store.jobs[index] = updated;
    await this.save(store);
    return updated;
  }

  async deleteJob(jobId: string): Promise<boolean> {
    const store = await this.load();
    const nextJobs = store.jobs.filter((job) => job.id !== jobId);
    if (nextJobs.length === store.jobs.length) {
      return false;
    }
    store.jobs = nextJobs;
    await this.save(store);
    return true;
  }

  async addRun(jobId: string): Promise<CronRun> {
    const store = await this.load();
    const run: CronRun = {
      id: randomUUID(),
      jobId,
      status: "scheduled",
      createdAt: nowIso(),
    };
    store.runs.push(run);
    await this.save(store);
    return run;
  }

  async updateRun(
    runId: string,
    params: { status: CronRun["status"]; startedAt?: string; finishedAt?: string },
  ): Promise<CronRun | undefined> {
    const store = await this.load();
    const index = store.runs.findIndex((run) => run.id === runId);
    if (index < 0) {
      return undefined;
    }
    const updated: CronRun = {
      ...store.runs[index],
      status: params.status,
      ...(params.startedAt ? { startedAt: params.startedAt } : {}),
      ...(params.finishedAt ? { finishedAt: params.finishedAt } : {}),
    };
    store.runs[index] = updated;
    await this.save(store);
    return updated;
  }

  async listRuns(jobId?: string): Promise<CronRun[]> {
    const store = await this.load();
    const runs = jobId ? store.runs.filter((run) => run.jobId === jobId) : store.runs;
    return [...runs].sort((a, b) => a.createdAt.localeCompare(b.createdAt));
  }
}
