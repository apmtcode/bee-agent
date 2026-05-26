import path from "node:path";
import { randomUUID } from "node:crypto";
import { readJsonFile, writeJsonAtomic } from "../shared/fs.js";
import {
  LocalTrainingExecutionService,
  type TrainingExecutionState,
} from "./execution-service.js";
import type { ReviewedExportManifest, TrainingMode } from "./export-manifest.js";
import {
  createLocalTrainingExecution,
  createLocalTrainingJobManifest,
  type LocalTrainingExecution,
  type LocalTrainingExecutionStepStatus,
  type LocalTrainingJobManifest,
  type LocalTrainingJobStatus,
} from "./job-manifest.js";
import {
  LocalAppleSiliconTrainingRunner,
  type TrainingJobPlan,
} from "./runner.js";

export type TrainingJobStoreShape = {
  version: 1;
  jobs: LocalTrainingJobManifest[];
};

export type CreateTrainingJobParams = {
  exportManifest: ReviewedExportManifest;
  mode: TrainingMode;
};

export type CompleteTrainingJobParams = {
  outputModelRef: string;
};

export type FailTrainingJobParams = {
  reason: string;
};

export type LaunchTrainingJobParams = {
  pid: number;
};

const EMPTY_STORE: TrainingJobStoreShape = {
  version: 1,
  jobs: [],
};

export class FileTrainingJobStore {
  private readonly runner: LocalAppleSiliconTrainingRunner;
  readonly executionService: LocalTrainingExecutionService;

  constructor(private readonly filePath: string) {
    const rootDir = path.dirname(filePath);
    this.runner = new LocalAppleSiliconTrainingRunner(rootDir);
    this.executionService = new LocalTrainingExecutionService(rootDir);
  }

  async load(): Promise<TrainingJobStoreShape> {
    return await readJsonFile(this.filePath, EMPTY_STORE);
  }

  async save(store: TrainingJobStoreShape): Promise<void> {
    await writeJsonAtomic(this.filePath, store);
  }

  async list(): Promise<LocalTrainingJobManifest[]> {
    const store = await this.load();
    return [...store.jobs].sort((a, b) => a.createdAt.localeCompare(b.createdAt));
  }

  async get(jobId: string): Promise<LocalTrainingJobManifest | undefined> {
    const store = await this.load();
    return store.jobs.find((job) => job.id === jobId);
  }

  async create(params: CreateTrainingJobParams): Promise<LocalTrainingJobManifest> {
    const store = await this.load();
    const job = createLocalTrainingJobManifest({
      id: randomUUID(),
      exportManifest: params.exportManifest,
      mode: params.mode,
    });
    store.jobs.push(job);
    await this.save(store);
    return job;
  }

  async updateStatus(
    jobId: string,
    status: LocalTrainingJobStatus,
  ): Promise<LocalTrainingJobManifest | undefined> {
    return await this.update(jobId, (job) => ({ ...job, status }));
  }

  async prepareExecution(jobId: string): Promise<LocalTrainingJobManifest | undefined> {
    const job = await this.update(jobId, (currentJob) => {
      const execution = currentJob.execution ?? createLocalTrainingExecution({
        jobId: currentJob.id,
        mode: currentJob.mode,
      });
      const plan = this.runner.buildPlan(currentJob, execution);
      return {
        ...currentJob,
        status: currentJob.status === "planned" ? "ready" : currentJob.status,
        execution: {
          ...execution,
          command: [...plan.command],
          environment: { ...plan.environment },
        },
      };
    });
    if (!job?.execution) {
      return job;
    }
    const plan = await this.runner.writeArtifacts(job, job.execution);
    return {
      ...job,
      execution: {
        ...job.execution,
        command: [...plan.command],
        environment: { ...plan.environment },
      },
    };
  }

  async startExecution(jobId: string): Promise<LocalTrainingJobManifest | undefined> {
    const prepared = await this.prepareExecution(jobId);
    if (!prepared?.execution) {
      return prepared;
    }
    const launched = await this.executionService.launch(prepared);
    return await this.markExecutionStarted(jobId, { pid: launched.pid });
  }

  async markExecutionStarted(
    jobId: string,
    params: LaunchTrainingJobParams,
  ): Promise<LocalTrainingJobManifest | undefined> {
    return await this.update(jobId, (job) => {
      const execution = ensureExecution(job);
      const startedAt = execution.startedAt ?? new Date().toISOString();
      return {
        ...job,
        status: "running",
        execution: {
          ...execution,
          startedAt,
          processId: params.pid,
          lastUpdatedAt: startedAt,
          completedAt: undefined,
          failedAt: undefined,
          failureReason: undefined,
          steps: updateExecutionSteps(execution.steps, {
            "prepare-dataset": "completed",
            "launch-training": "running",
          }),
        },
      };
    });
  }

  async completeExecution(
    jobId: string,
    params: CompleteTrainingJobParams,
  ): Promise<LocalTrainingJobManifest | undefined> {
    return await this.update(jobId, (job) => {
      const execution = ensureExecution(job);
      const completedAt = new Date().toISOString();
      return {
        ...job,
        status: "completed",
        execution: {
          ...execution,
          completedAt,
          failedAt: undefined,
          failureReason: undefined,
          lastUpdatedAt: completedAt,
          outputModelRef: params.outputModelRef,
          steps: updateExecutionSteps(execution.steps, {
            "prepare-dataset": "completed",
            "launch-training": "completed",
            "collect-artifacts": "completed",
            "evaluate-replay": "completed",
          }),
        },
      };
    });
  }

  async failExecution(
    jobId: string,
    params: FailTrainingJobParams,
  ): Promise<LocalTrainingJobManifest | undefined> {
    return await this.update(jobId, (job) => {
      const execution = ensureExecution(job);
      const failedAt = new Date().toISOString();
      return {
        ...job,
        status: "failed",
        execution: {
          ...execution,
          completedAt: undefined,
          failedAt,
          lastUpdatedAt: failedAt,
          failureReason: params.reason,
          steps: updateExecutionSteps(execution.steps, {
            "prepare-dataset": execution.startedAt ? "completed" : "running",
            "launch-training": "failed",
          }),
        },
      };
    });
  }

  async getExecutionPlan(jobId: string): Promise<TrainingJobPlan | undefined> {
    const job = await this.get(jobId);
    return job ? await this.runner.readPlan(job) : undefined;
  }

  async getLaunchScript(jobId: string): Promise<string | undefined> {
    const job = await this.get(jobId);
    return job ? await this.runner.readLaunchScript(job) : undefined;
  }

  async syncExecution(jobId: string): Promise<LocalTrainingJobManifest | undefined> {
    const job = await this.get(jobId);
    if (!job) {
      return undefined;
    }
    const state = await this.executionService.readState(job);
    if (!state || state.status === "running") {
      return job;
    }
    if (state.status === "completed") {
      const plan = await this.runner.readPlan(job);
      return await this.completeExecution(jobId, {
        outputModelRef: plan?.outputPath ?? job.execution?.outputModelRef ?? "completed-model",
      });
    }
    return await this.failExecution(jobId, {
      reason: state.error ?? `training process exited with code ${state.exitCode ?? "unknown"}`,
    });
  }

  async getExecutionState(jobId: string): Promise<TrainingExecutionState | undefined> {
    const job = await this.syncExecution(jobId);
    return job ? await this.executionService.readState(job) : undefined;
  }

  async getExecutionLog(jobId: string, lineLimit?: number): Promise<string | undefined> {
    const job = await this.get(jobId);
    return job ? await this.executionService.readLog(job, { lineLimit }) : undefined;
  }

  private async update(
    jobId: string,
    apply: (job: LocalTrainingJobManifest) => LocalTrainingJobManifest,
  ): Promise<LocalTrainingJobManifest | undefined> {
    const store = await this.load();
    const index = store.jobs.findIndex((job) => job.id === jobId);
    if (index < 0) {
      return undefined;
    }
    const updated = apply(store.jobs[index]);
    store.jobs[index] = updated;
    await this.save(store);
    return updated;
  }
}

function ensureExecution(job: LocalTrainingJobManifest): LocalTrainingExecution {
  return job.execution ?? createLocalTrainingExecution({ jobId: job.id, mode: job.mode });
}

function updateExecutionSteps(
  steps: LocalTrainingExecution["steps"],
  updates: Partial<Record<string, LocalTrainingExecutionStepStatus>>,
): LocalTrainingExecution["steps"] {
  return steps.map((step) => ({
    ...step,
    ...(updates[step.id] ? { status: updates[step.id] } : {}),
  }));
}
