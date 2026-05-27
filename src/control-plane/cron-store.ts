import { randomUUID } from "node:crypto";
import { readJsonFile, writeJsonAtomic } from "../shared/fs.js";

export type CronJobLastStatus = "success" | "error" | "interrupted";
export type CronRunStatus = "scheduled" | "running" | "completed" | "failed";
export type CronRunOutcome = "success" | "error" | "interrupted";

export type CronJob = {
  id: string;
  cron: string;
  prompt: string;
  recurring: boolean;
  durable: boolean;
  createdAt: string;
  lastRunAt?: string;
  nextRunAt?: string;
  lastStatus?: CronJobLastStatus;
  lastError?: string;
};

export type CronRun = {
  id: string;
  jobId: string;
  status: CronRunStatus;
  createdAt: string;
  scheduledFor?: string;
  sessionId?: string;
  operatorRunId?: string;
  startedAt?: string;
  finishedAt?: string;
  outcome?: CronRunOutcome;
  error?: string;
  summary?: string;
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

function hasOwn<T extends object, K extends keyof T>(value: T, key: K): boolean {
  return Object.prototype.hasOwnProperty.call(value, key);
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

  async updateJob(
    jobId: string,
    params: {
      lastRunAt?: string;
      nextRunAt?: string;
      lastStatus?: CronJobLastStatus;
      lastError?: string | null;
    },
  ): Promise<CronJob | undefined> {
    const store = await this.load();
    const index = store.jobs.findIndex((job) => job.id === jobId);
    if (index < 0) {
      return undefined;
    }
    const updated: CronJob = {
      ...store.jobs[index],
      ...(params.lastRunAt ? { lastRunAt: params.lastRunAt } : {}),
      ...(params.nextRunAt ? { nextRunAt: params.nextRunAt } : {}),
      ...(params.lastStatus ? { lastStatus: params.lastStatus } : {}),
    };
    if (hasOwn(params, "lastError")) {
      if (typeof params.lastError === "string") {
        updated.lastError = params.lastError;
      } else {
        delete updated.lastError;
      }
    }
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

  async addRun(params: {
    jobId: string;
    status?: CronRunStatus;
    scheduledFor?: string;
    sessionId?: string;
    operatorRunId?: string;
    startedAt?: string;
    finishedAt?: string;
    outcome?: CronRunOutcome;
    error?: string;
    summary?: string;
  }): Promise<CronRun> {
    const store = await this.load();
    const run: CronRun = {
      id: randomUUID(),
      jobId: params.jobId,
      status: params.status ?? "scheduled",
      createdAt: nowIso(),
      ...(params.scheduledFor ? { scheduledFor: params.scheduledFor } : {}),
      ...(params.sessionId ? { sessionId: params.sessionId } : {}),
      ...(params.operatorRunId ? { operatorRunId: params.operatorRunId } : {}),
      ...(params.startedAt ? { startedAt: params.startedAt } : {}),
      ...(params.finishedAt ? { finishedAt: params.finishedAt } : {}),
      ...(params.outcome ? { outcome: params.outcome } : {}),
      ...(params.error ? { error: params.error } : {}),
      ...(params.summary ? { summary: params.summary } : {}),
    };
    store.runs.push(run);
    await this.save(store);
    return run;
  }

  async getRun(runId: string): Promise<CronRun | undefined> {
    const store = await this.load();
    return store.runs.find((run) => run.id === runId);
  }

  async updateRun(
    runId: string,
    params: {
      status?: CronRunStatus;
      scheduledFor?: string;
      sessionId?: string;
      operatorRunId?: string;
      startedAt?: string;
      finishedAt?: string;
      outcome?: CronRunOutcome;
      error?: string | null;
      summary?: string | null;
    },
  ): Promise<CronRun | undefined> {
    const store = await this.load();
    const index = store.runs.findIndex((run) => run.id === runId);
    if (index < 0) {
      return undefined;
    }
    const updated: CronRun = {
      ...store.runs[index],
      ...(params.status ? { status: params.status } : {}),
      ...(params.scheduledFor ? { scheduledFor: params.scheduledFor } : {}),
      ...(params.sessionId ? { sessionId: params.sessionId } : {}),
      ...(params.operatorRunId ? { operatorRunId: params.operatorRunId } : {}),
      ...(params.startedAt ? { startedAt: params.startedAt } : {}),
      ...(params.finishedAt ? { finishedAt: params.finishedAt } : {}),
      ...(params.outcome ? { outcome: params.outcome } : {}),
    };
    if (hasOwn(params, "error")) {
      if (typeof params.error === "string") {
        updated.error = params.error;
      } else {
        delete updated.error;
      }
    }
    if (hasOwn(params, "summary")) {
      if (typeof params.summary === "string") {
        updated.summary = params.summary;
      } else {
        delete updated.summary;
      }
    }
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
