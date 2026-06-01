import path from "node:path";
import type { OperatorResolvedModelSelection } from "../harness/types.js";
import {
  OperatorDeliveryService,
  summarizeDeliveryResults,
  type CronTerminalDeliveryPayload,
  type DeliveryTarget,
} from "./delivery.js";
import { FileCronStore, type CronJob, type CronJobLastStatus, type CronRun } from "./cron-store.js";
import { JsonlTranscriptStore } from "../harness/transcript-store.js";
import type { StandaloneOperatorOptions, StandaloneOperatorRuntime } from "../orchestrator/operator-runtime.js";

function hasOwn<T extends object, K extends PropertyKey>(value: T, key: K): boolean {
  return Object.prototype.hasOwnProperty.call(value, key);
}

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

function buildResolvedModelSelection(params: {
  primary?: string;
  fallbacks?: string[];
  source: OperatorResolvedModelSelection["source"];
  preserveEmptyFallbacks?: boolean;
}): OperatorResolvedModelSelection | undefined {
  const primary = typeof params.primary === "string" && params.primary.trim().length > 0 ? params.primary : undefined;
  const fallbacks = Array.isArray(params.fallbacks) ? [...params.fallbacks] : undefined;
  if (!primary && !params.preserveEmptyFallbacks && !fallbacks) {
    return undefined;
  }
  return {
    ...(primary ? { primary } : {}),
    ...(params.preserveEmptyFallbacks ? { fallbacks: fallbacks ?? [] } : fallbacks ? { fallbacks } : {}),
    source: params.source,
  };
}

function cloneResolvedModelSelection(
  selection: OperatorResolvedModelSelection | undefined,
): OperatorResolvedModelSelection | undefined {
  if (!selection) {
    return undefined;
  }
  return buildResolvedModelSelection({
    primary: typeof selection.primary === "string" ? selection.primary : undefined,
    fallbacks: Array.isArray(selection.fallbacks)
      ? selection.fallbacks.filter((item): item is string => typeof item === "string")
      : undefined,
    source: selection.source,
    preserveEmptyFallbacks: Array.isArray(selection.fallbacks),
  });
}

export type OperatorCronDispatcher = (job: CronJob, context: { run: CronRun; runtime: StandaloneOperatorRuntime }) => Promise<{
  assistantText: string;
  summary?: string;
}>;

export class OperatorCronService {
  readonly store: FileCronStore;
  private readonly runtime?: StandaloneOperatorRuntime;
  private readonly dispatchCronJob: OperatorCronDispatcher;
  private readonly delivery: OperatorDeliveryService;
  private readonly baseModelSelection?: OperatorResolvedModelSelection;

  constructor(
    rootDirOrOptions: string | StandaloneOperatorOptions,
    options: {
      runtime?: StandaloneOperatorRuntime;
      dispatchCronJob?: OperatorCronDispatcher;
      delivery?: OperatorDeliveryService;
      modelSelection?: OperatorResolvedModelSelection;
    } = {},
  ) {
    const rootDir = typeof rootDirOrOptions === "string" ? rootDirOrOptions : rootDirOrOptions.rootDir;
    this.store = new FileCronStore(path.join(rootDir, "cron.json"));
    this.runtime = options.runtime;
    this.dispatchCronJob = options.dispatchCronJob ?? defaultCronDispatcher;
    this.delivery = options.delivery ?? new OperatorDeliveryService(rootDir);
    this.baseModelSelection = cloneResolvedModelSelection(options.modelSelection);
  }

  async listJobs(): Promise<CronJob[]> {
    return await this.store.listJobs();
  }

  async createJob(params: {
    cron: string;
    prompt: string;
    recurring?: boolean;
    durable?: boolean;
    delivery?: CronJob["delivery"];
    modelSelection?: CronJob["modelSelection"];
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

  private resolveCronModelSelection(job: CronJob): OperatorResolvedModelSelection | undefined {
    const baseSelection = this.baseModelSelection;
    if (!job.modelSelection) {
      return cloneResolvedModelSelection(baseSelection);
    }
    const hasPrimary = hasOwn(job.modelSelection, "primary");
    const hasFallbacks = hasOwn(job.modelSelection, "fallbacks");
    if (!hasPrimary && !hasFallbacks) {
      return cloneResolvedModelSelection(baseSelection);
    }
    const primary = hasPrimary
      ? typeof job.modelSelection.primary === "string"
        ? job.modelSelection.primary
        : undefined
      : baseSelection?.primary;
    const fallbacks = hasFallbacks
      ? Array.isArray(job.modelSelection.fallbacks)
        ? job.modelSelection.fallbacks.filter((item): item is string => typeof item === "string")
        : []
      : baseSelection?.fallbacks;
    return buildResolvedModelSelection({
      primary,
      fallbacks,
      source: "job",
      preserveEmptyFallbacks: hasFallbacks,
    });
  }

  private buildDeliveryPayload(job: CronJob, run: CronRun): CronTerminalDeliveryPayload {
    return {
      kind: "cron-terminal-state",
      cron: {
        jobId: job.id,
        runId: run.id,
        cron: job.cron,
        prompt: job.prompt,
        recurring: job.recurring,
        status: run.status,
        ...(run.outcome ? { outcome: run.outcome } : {}),
        ...(run.summary ? { summary: run.summary } : {}),
        ...(run.error ? { error: run.error } : {}),
        ...(run.scheduledFor ? { scheduledFor: run.scheduledFor } : {}),
        ...(run.startedAt ? { startedAt: run.startedAt } : {}),
        ...(run.finishedAt ? { finishedAt: run.finishedAt } : {}),
        ...(run.sessionId ? { sessionId: run.sessionId } : {}),
        ...(run.operatorRunId ? { operatorRunId: run.operatorRunId } : {}),
      },
    };
  }

  private getDeliveryTargets(job: CronJob, run: CronRun): DeliveryTarget[] {
    if (run.outcome === "success") {
      return job.delivery?.onSuccess ?? [];
    }
    return job.delivery?.onFailure ?? [];
  }

  private async persistDeliveryResults(job: CronJob, run: CronRun): Promise<CronRun> {
    const targets = this.getDeliveryTargets(job, run);
    if (targets.length === 0) {
      return run;
    }
    const deliveryResults = await this.delivery.deliver(targets, this.buildDeliveryPayload(job, run));
    const updatedRun = await this.store.updateRun(run.id, { deliveryResults });
    const summary = summarizeDeliveryResults(deliveryResults);
    await this.store.updateJob(job.id, {
      lastDeliveryAt: deliveryResults[deliveryResults.length - 1]?.attemptedAt,
      lastDeliveryStatus: summary.lastDeliveryStatus,
      lastDeliveryError: summary.lastDeliveryError ?? null,
    });
    return updatedRun ?? { ...run, deliveryResults };
  }

  private async recoverInterruptedRuns(): Promise<void> {
    if (!this.runtime) {
      return;
    }
    const runs = await this.store.listRuns();
    const staleRuns = runs.filter((run) => run.status === "scheduled" || run.status === "running");
    for (const run of staleRuns) {
      const job = await this.store.getJob(run.jobId);
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
      const interrupted = await this.store.updateRun(run.id, {
        status: "failed",
        outcome: "interrupted",
        error: "Interrupted before completion.",
        finishedAt: nowIso(),
      });
      await this.updateJobOutcome(run.jobId, {
        lastStatus: "interrupted",
        lastError: "Interrupted before completion.",
      });
      if (job && interrupted) {
        await this.persistDeliveryResults(job, interrupted);
      }
    }
  }

  private async dispatchJob(job: CronJob, now: Date): Promise<CronRun> {
    const scheduledAt = now.toISOString();
    const modelSelection = this.resolveCronModelSelection(job);
    const scheduled = await this.store.addRun({
      jobId: job.id,
      status: "scheduled",
      scheduledFor: scheduledAt,
      ...(modelSelection ? { modelSelection } : {}),
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
        ...(modelSelection ? { modelSelection } : {}),
      });
      await this.updateJobOutcome(job.id, { lastStatus: "success", lastError: null });
      if (!completed) {
        throw new Error(`Unable to finalize cron run: ${scheduled.id}`);
      }
      const delivered = await this.persistDeliveryResults(job, completed);
      if (!job.recurring) {
        await this.store.deleteJob(job.id);
      }
      return delivered;
    }

    const session = await this.runtime.startSession({
      title: buildCronSessionTitle(job.prompt),
      agentId: "operator-cron",
      ...(modelSelection ? { modelSelection } : {}),
    });
    const operatorRun = await this.runtime.startRun({
      sessionId: session.id,
      title: buildCronRunTitle(job.prompt),
      metadata: {
        trigger: "cron",
        cronJobId: job.id,
        cronRunId: scheduled.id,
        ...(modelSelection ? { modelSelection } : {}),
      },
    });
    const running = await this.store.updateRun(scheduled.id, {
      status: "running",
      sessionId: session.id,
      operatorRunId: operatorRun.id,
      startedAt: nowIso(),
      ...(modelSelection ? { modelSelection } : {}),
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
        ...(modelSelection ? { modelSelection } : {}),
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
        ...(modelSelection ? { modelSelection } : {}),
      });
      await this.updateJobOutcome(job.id, { lastStatus: "success", lastError: null });
      if (!completed) {
        throw new Error(`Unable to finalize cron run: ${scheduled.id}`);
      }
      const delivered = await this.persistDeliveryResults(job, completed);
      if (!job.recurring) {
        await this.store.deleteJob(job.id);
      }
      return delivered;
    } catch (error) {
      const message = normalizeError(error);
      await this.runtime.updateRunStatus(operatorRun.id, "failed", {
        cronJobId: job.id,
        cronRunId: scheduled.id,
        trigger: "cron",
        error: message,
        ...(modelSelection ? { modelSelection } : {}),
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
        ...(modelSelection ? { modelSelection } : {}),
      });
      await this.updateJobOutcome(job.id, { lastStatus: "error", lastError: message });
      if (!failed) {
        throw new Error(`Unable to finalize failed cron run: ${scheduled.id}`);
      }
      const delivered = await this.persistDeliveryResults(job, failed);
      if (!job.recurring) {
        await this.store.deleteJob(job.id);
      }
      return delivered;
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
