import path from "node:path";
import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { randomUUID } from "node:crypto";
import type { OperatorCliExecutionConfig } from "../cli/config.js";
import {
  evaluateOperatorExecutionAction,
  runOperatorCommandHooks,
  type OperatorHookResult,
} from "../cli/execution-policy.js";
import { FileApprovalStore } from "../control-plane/approval-store.js";
import {
  InMemoryApprovalManager,
  type ApprovalRequest,
  type ApprovalResolution,
} from "../control-plane/approvals.js";
import { CaptureIngestionService } from "../capture/ingestion.js";
import { FileCaptureConsentStore } from "../capture/consent-store.js";
import { ReplayRuntimeService } from "../capture/replay-service.js";
import { FileTrajectoryStore, type ReviewTrajectoryParams as StoreReviewTrajectoryParams } from "../capture/trajectory-store.js";
import {
  buildTrajectorySpan,
  type CaptureConsentGrant,
  type CaptureTier,
  type TrajectoryReviewStatus,
  type TrajectorySpan,
} from "../capture/trajectory.js";
import type { ReplayManifest } from "../capture/replay.js";
import { JsonlTranscriptStore, type TranscriptReadOptions, type TranscriptRecord } from "../harness/transcript-store.js";
import {
  FileBackgroundTaskStore,
  type BackgroundTaskExecutionState,
  type BackgroundTaskKind,
  type BackgroundTaskRecord,
  type BackgroundTaskRecoveryReason,
  type BackgroundTaskRecoveryResult,
  type ReadBackgroundTaskOutputOptions,
  type StartBackgroundTaskParams,
} from "../harness/background-tasks.js";
import { FileSessionStore } from "../harness/session-store.js";
import { FileMemoryStore } from "../memory/memory-store.js";
import { OperatorEventBus, type OperatorEvent } from "../kernel/event-bus.js";
import { FileRunStore, type OperatorRunRecord } from "./run-store.js";
import { FileSubagentRegistry, type SubagentRunRecord } from "./subagent-registry.js";
import {
  FileTrainingJobStore,
  type CompleteTrainingJobParams,
  type FailTrainingJobParams,
} from "../training/job-store.js";
import {
  FilePluginManifestRegistry,
  type RegisterPluginParams,
  type RegisteredPluginRecord,
} from "../plugins/manifest-registry.js";
import { OperatorPluginRuntimeManager } from "../plugins/runtime.js";
import type {
  ExecutableSkill,
  ExecutableSkillRun,
  ExecutableSkillStep,
  ExecutableSkillStepResult,
  PromotedSkill,
  RecallSummary,
  SkillCandidate,
  SkillCandidateStatus,
} from "../memory/types.js";
import type { SessionRecord } from "../harness/types.js";
import type { ReviewedExportManifest, TrainingMode } from "../training/export-manifest.js";
import type { LocalTrainingJobManifest, LocalTrainingJobStatus } from "../training/job-manifest.js";
import type { TrainingExecutionState } from "../training/execution-service.js";
import type { OperatorPluginCapability, OperatorPluginRuntime } from "../plugins/manifest.js";
import type { BrowserCaptureInput } from "../capture/browser-adapter.js";
import type { DeviceCaptureInput } from "../capture/device-adapter.js";
import type { OsObservationInput } from "../capture/os-observer.js";
import type { CaptureRecordResult } from "../capture/recorder.js";
import { LocalTrainingExporter, type CreateReviewedExportParams } from "../training/exporter.js";

export type StandaloneOperatorOptions = {
  rootDir: string;
  replayLimit?: number;
  backgroundTaskSpawnProcess?: Parameters<typeof FileBackgroundTaskStore>[1];
  backgroundTaskIsProcessRunning?: Parameters<typeof FileBackgroundTaskStore>[2];
};

export type StartSessionParams = {
  title?: string;
  cwd?: string;
  agentId?: string;
};

export type CommandExecutionPolicyResult = {
  allowed: boolean;
  pendingApprovalId?: string;
  message: string;
  preHookResults: OperatorHookResult[];
  postHookResults: OperatorHookResult[];
};

export type ApprovalPromptParams = {
  sessionId: string;
  title: string;
  summary: string;
  scope?: string;
  metadata?: Record<string, unknown>;
  timeoutMs?: number;
};

export type RecordTurnParams = {
  sessionId: string;
  userText: string;
  assistantText: string;
  trajectorySummary?: string;
};

export type StartRunParams = {
  sessionId: string;
  title: string;
  parentRunId?: string;
  metadata?: Record<string, unknown>;
};

export type RegisterSubagentParams = {
  sessionId: string;
  parentRunId: string;
  childSessionId: string;
  title: string;
  metadata?: Record<string, unknown>;
};

export type CreateTrainingJobParams = {
  exportManifest: ReviewedExportManifest;
  mode: TrainingMode;
};

const execFileAsync = promisify(execFile);

export type ListPluginsParams = {
  capability?: OperatorPluginCapability;
};

export type CreateCaptureConsentParams = {
  tier: CaptureTier;
  purpose: string;
  expiresAt?: string;
};

export type RecoverBackgroundTasksParams = {
  sessionId?: string;
  reasons?: BackgroundTaskRecoveryReason[];
};

export type ReviewTrajectoryParams = {
  trajectoryId: string;
  status: TrajectoryReviewStatus;
  reviewedBy: string;
  reviewNote?: string;
  redactedObservations?: StoreReviewTrajectoryParams["redactedObservations"];
  redactedActions?: StoreReviewTrajectoryParams["redactedActions"];
  redactedTranscript?: StoreReviewTrajectoryParams["redactedTranscript"];
};

export type SessionLifecycleEventType =
  | "session.started"
  | "session.resumed"
  | "session.idle"
  | "session.completed"
  | "session.failed";

export type OperatorRunProgressPhase =
  | "planning"
  | "status"
  | "recall"
  | "session-summary"
  | "approvals"
  | "approval-resolution"
  | "executing-skill"
  | "starting-background-task"
  | "syncing-background-task"
  | "cancelling-background-task"
  | "training"
  | "config"
  | "prompt"
  | "awaiting-approval"
  | "completed"
  | "failed";

export type OperatorRunProgressEvent = {
  runId?: string;
  sessionId: string;
  phase: OperatorRunProgressPhase;
  message: string;
  relatedApprovalId?: string;
  relatedTaskId?: string;
  relatedSkillId?: string;
};

export type OperatorRuntimeEventType =
  | SessionLifecycleEventType
  | "approval.requested"
  | "approval.resolved"
  | "trajectory.recorded"
  | "trajectory.reviewed"
  | "training.export.created"
  | "skill.reviewed"
  | "skill.promoted"
  | "skill.run.completed"
  | "run.started"
  | "run.updated"
  | "run.progress"
  | "subagent.registered"
  | "subagent.updated"
  | "training-job.created"
  | "training-job.updated"
  | "training-job.prepared"
  | "training-job.started"
  | "training-job.synced"
  | "training-job.completed"
  | "training-job.failed"
  | "background-task.started"
  | "background-task.updated"
  | "background-task.completed"
  | "background-task.failed"
  | "background-task.cancelled"
  | "background-task.recovered"
  | "plugin.registered"
  | "plugin.updated"
  | "plugin.activated"
  | "capture.consent-created"
  | "capture.consent-revoked"
  | "capture.browser-recorded"
  | "capture.device-recorded"
  | "capture.os-recorded";

export class StandaloneOperatorRuntime {
  get rootDir(): string {
    return this.options.rootDir;
  }

  readonly events: OperatorEventBus<OperatorEvent>;
  readonly approvals: InMemoryApprovalManager;
  readonly sessions: FileSessionStore;
  readonly memory: FileMemoryStore;
  readonly trajectories: FileTrajectoryStore;
  readonly captureConsents: FileCaptureConsentStore;
  readonly capture: CaptureIngestionService;
  readonly replays: ReplayRuntimeService;
  readonly runs: FileRunStore;
  readonly subagents: FileSubagentRegistry;
  readonly backgroundTasks: FileBackgroundTaskStore;
  readonly trainingJobs: FileTrainingJobStore;
  readonly plugins: FilePluginManifestRegistry;
  readonly pluginRuntime: OperatorPluginRuntimeManager;
  readonly trainingExporter: LocalTrainingExporter;
  private readonly transcriptsDir: string;
  private approvalsReady: Promise<void> | null = null;

  constructor(private readonly options: StandaloneOperatorOptions) {
    this.events = new OperatorEventBus({ replayLimit: options.replayLimit ?? 200 });
    this.approvals = new InMemoryApprovalManager(new FileApprovalStore(path.join(options.rootDir, "approvals.json")));
    this.sessions = new FileSessionStore(path.join(options.rootDir, "sessions.json"));
    this.memory = new FileMemoryStore(path.join(options.rootDir, "memory.json"));
    this.trajectories = new FileTrajectoryStore(path.join(options.rootDir, "trajectories.json"));
    this.captureConsents = new FileCaptureConsentStore(path.join(options.rootDir, "capture-consent.json"));
    this.capture = new CaptureIngestionService({
      consentStore: this.captureConsents,
      trajectoryStore: this.trajectories,
    });
    this.replays = new ReplayRuntimeService(this.sessions, this.trajectories);
    this.runs = new FileRunStore(path.join(options.rootDir, "runs.json"));
    this.subagents = new FileSubagentRegistry(options.rootDir);
    this.backgroundTasks = new FileBackgroundTaskStore(
      path.join(options.rootDir, "background-tasks.json"),
      options.backgroundTaskSpawnProcess,
      options.backgroundTaskIsProcessRunning,
    );
    this.trainingJobs = new FileTrainingJobStore(path.join(options.rootDir, "training-jobs.json"));
    this.plugins = new FilePluginManifestRegistry(path.join(options.rootDir, "plugins.json"), options.rootDir);
    this.pluginRuntime = new OperatorPluginRuntimeManager(this.plugins);
    this.trainingExporter = new LocalTrainingExporter({
      memory: this.memory,
      trajectories: this.trajectories,
      resolveSession: async (sessionId) => await this.sessions.get(sessionId),
    });
    this.transcriptsDir = path.join(options.rootDir, "transcripts");
  }

  async startSession(params: StartSessionParams = {}): Promise<SessionRecord> {
    const sessionId = randomUUID();
    const transcriptFile = path.join(this.transcriptsDir, `${sessionId}.jsonl`);
    const transcript = new JsonlTranscriptStore(transcriptFile);
    await transcript.ensureHeader({ sessionId, ...(params.cwd ? { cwd: params.cwd } : {}) });
    const session = await this.sessions.upsert({
      id: sessionId,
      transcriptFile,
      metadata: {
        ...(params.title ? { title: params.title } : {}),
        ...(params.cwd ? { cwd: params.cwd } : {}),
        ...(params.agentId ? { agentId: params.agentId } : {}),
      },
    });
    this.events.publish({ type: "session.started", payload: session, ts: Date.now() });
    return session;
  }

  async getSession(sessionId: string): Promise<SessionRecord | undefined> {
    return await this.sessions.get(sessionId);
  }

  async listSessions(): Promise<SessionRecord[]> {
    return await this.sessions.list();
  }

  async resumeSession(sessionId: string): Promise<SessionRecord | undefined> {
    const session = await this.sessions.update(sessionId, { status: "active" });
    if (session) {
      this.events.publish({ type: "session.resumed", payload: session, ts: Date.now() });
    }
    return session;
  }

  async markSessionIdle(sessionId: string): Promise<SessionRecord | undefined> {
    const session = await this.sessions.update(sessionId, { status: "idle" });
    if (session) {
      this.events.publish({ type: "session.idle", payload: session, ts: Date.now() });
    }
    return session;
  }

  async completeSession(sessionId: string): Promise<SessionRecord | undefined> {
    const session = await this.sessions.update(sessionId, { status: "completed" });
    if (session) {
      this.events.publish({ type: "session.completed", payload: session, ts: Date.now() });
    }
    return session;
  }

  async failSession(sessionId: string): Promise<SessionRecord | undefined> {
    const session = await this.sessions.update(sessionId, { status: "failed" });
    if (session) {
      this.events.publish({ type: "session.failed", payload: session, ts: Date.now() });
    }
    return session;
  }

  async promptApproval(params: ApprovalPromptParams): Promise<ApprovalRequest> {
    await this.ensureApprovalsLoaded();
    const request = await this.approvals.create({
      title: params.title,
      summary: params.summary,
      sessionId: params.sessionId,
      scope: params.scope,
      metadata: params.metadata,
      timeoutMs: params.timeoutMs,
    });
    this.events.publish({ type: "approval.requested", payload: request, ts: Date.now() });
    return request;
  }

  async listApprovals(): Promise<ApprovalRequest[]> {
    await this.ensureApprovalsLoaded();
    return this.approvals.list();
  }

  async resolveApproval(
    approvalId: string,
    decision: "approved" | "denied" | "expired",
    resolvedBy?: string,
  ): Promise<ApprovalResolution | null> {
    await this.ensureApprovalsLoaded();
    const resolution = await this.approvals.resolve(approvalId, decision, resolvedBy);
    if (resolution) {
      this.events.publish({ type: "approval.resolved", payload: resolution, ts: Date.now() });
    }
    return resolution;
  }

  async startRun(params: StartRunParams): Promise<OperatorRunRecord> {
    const run = await this.runs.create({
      sessionId: params.sessionId,
      title: params.title,
      status: "running",
      parentRunId: params.parentRunId,
      metadata: params.metadata,
    });
    this.events.publish({ type: "run.started", payload: run, ts: Date.now() });
    return run;
  }

  async getRun(runId: string): Promise<OperatorRunRecord | undefined> {
    return await this.runs.get(runId);
  }

  async updateRunStatus(
    runId: string,
    status: "pending" | "running" | "paused" | "completed" | "failed",
    metadata?: Record<string, unknown>,
  ): Promise<OperatorRunRecord | undefined> {
    const run = await this.runs.update(runId, { status, metadata });
    if (run) {
      this.events.publish({ type: "run.updated", payload: run, ts: Date.now() });
    }
    return run;
  }

  publishRunProgress(event: OperatorRunProgressEvent): void {
    this.events.publish({ type: "run.progress", payload: event, ts: Date.now() });
  }

  async listRuns(sessionId?: string): Promise<OperatorRunRecord[]> {
    return sessionId ? await this.runs.listBySession(sessionId) : await this.runs.list();
  }

  async getActiveRun(sessionId?: string): Promise<OperatorRunRecord | undefined> {
    const runs = await this.listRuns(sessionId);
    return [...runs].reverse().find(
      (run) => !run.parentRunId && (run.status === "running" || run.status === "paused" || run.status === "pending"),
    );
  }

  async registerSubagent(params: RegisterSubagentParams): Promise<SubagentRunRecord> {
    const subagent = await this.subagents.register(params);
    this.events.publish({ type: "subagent.registered", payload: subagent, ts: Date.now() });
    return subagent;
  }

  async getSubagent(runId: string): Promise<SubagentRunRecord | undefined> {
    return await this.subagents.get(runId);
  }

  async updateSubagentStatus(
    runId: string,
    status: "pending" | "running" | "paused" | "completed" | "failed",
  ): Promise<SubagentRunRecord | undefined> {
    const subagent = await this.subagents.updateStatus(runId, status);
    if (subagent) {
      this.events.publish({ type: "subagent.updated", payload: subagent, ts: Date.now() });
    }
    return subagent;
  }

  async listSubagents(parentRunId?: string): Promise<SubagentRunRecord[]> {
    return parentRunId ? await this.subagents.listByParent(parentRunId) : await this.subagents.list();
  }

  async listActiveSubagents(): Promise<SubagentRunRecord[]> {
    return await this.subagents.listActive();
  }

  async startBackgroundTask(params: StartBackgroundTaskParams): Promise<BackgroundTaskRecord> {
    const task = await this.backgroundTasks.start({
      ...params,
      ...(params.cwd ? {} : { cwd: this.rootDir }),
    });
    this.events.publish({ type: "background-task.started", payload: task, ts: Date.now() });
    if (task.status === "completed") {
      this.events.publish({ type: "background-task.completed", payload: task, ts: Date.now() });
    } else if (task.status === "failed") {
      this.events.publish({ type: "background-task.failed", payload: task, ts: Date.now() });
    }
    return task;
  }

  async authorizeCommandExecution(params: {
    sessionId?: string;
    cwd: string;
    command: string;
    title?: string;
    kind: "background.start" | "skill.command";
    source: "cli" | "runtime" | "rpc";
    executionConfig: OperatorCliExecutionConfig;
    approvedFingerprints?: ReadonlySet<string>;
  }): Promise<CommandExecutionPolicyResult> {
    const action = {
      kind: params.kind,
      sessionId: params.sessionId,
      cwd: params.cwd,
      command: params.command,
      ...(params.title ? { title: params.title } : {}),
      source: params.source,
    };
    const decision = evaluateOperatorExecutionAction({
      action,
      config: params.executionConfig,
      approvedFingerprints: params.approvedFingerprints,
    });
    if (decision.outcome === "deny") {
      return { allowed: false, message: decision.reason, preHookResults: [], postHookResults: [] };
    }
    if (decision.outcome === "request-approval") {
      const approval = await this.promptApproval({
        sessionId: params.sessionId ?? "operator-cli",
        title: "Approve local command execution",
        summary: `${params.kind}: ${params.command}`,
        scope: params.kind,
        metadata: {
          command: params.command,
          cwd: params.cwd,
          permissionMode: decision.permissionMode,
          dangerousReasons: decision.dangerousReasons,
          fingerprint: decision.fingerprint,
        },
      });
      return {
        allowed: false,
        pendingApprovalId: approval.id,
        message: `Approval required: ${approval.id}. Use /approve ${approval.id} or /deny ${approval.id} and rerun the command.`,
        preHookResults: [],
        postHookResults: [],
      };
    }
    try {
      const preHookResults = await runOperatorCommandHooks({
        config: params.executionConfig,
        event: "PreCommand",
        action,
      });
      return { allowed: true, message: decision.reason, preHookResults, postHookResults: [] };
    } catch (error) {
      return {
        allowed: false,
        message: `Pre-command hook blocked execution: ${error instanceof Error ? error.message : String(error)}`,
        preHookResults: [],
        postHookResults: [],
      };
    }
  }

  async runPostCommandHooks(params: {
    sessionId?: string;
    cwd: string;
    command: string;
    title?: string;
    kind: "background.start" | "skill.command";
    source: "cli" | "runtime" | "rpc";
    executionConfig: OperatorCliExecutionConfig;
  }): Promise<OperatorHookResult[]> {
    try {
      return await runOperatorCommandHooks({
        config: params.executionConfig,
        event: "PostCommand",
        action: {
          kind: params.kind,
          sessionId: params.sessionId,
          cwd: params.cwd,
          command: params.command,
          ...(params.title ? { title: params.title } : {}),
          source: params.source,
        },
      });
    } catch (error) {
      return [{
        event: "PostCommand",
        command: "<post-hook>",
        stdout: "",
        stderr: `post-command hook failed: ${error instanceof Error ? error.message : String(error)}`,
      }];
    }
  }

  async getBackgroundTask(taskId: string): Promise<BackgroundTaskRecord | undefined> {
    return await this.backgroundTasks.get(taskId);
  }

  async listBackgroundTasks(sessionId?: string): Promise<BackgroundTaskRecord[]> {
    return sessionId ? await this.backgroundTasks.listBySession(sessionId) : await this.backgroundTasks.list();
  }

  async getActiveBackgroundTask(sessionId?: string): Promise<BackgroundTaskRecord | undefined> {
    const tasks = await this.listBackgroundTasks(sessionId);
    return [...tasks].reverse().find((task) => task.status === "running" || task.status === "planned");
  }

  async syncBackgroundTask(taskId: string): Promise<BackgroundTaskRecord | undefined> {
    const task = await this.backgroundTasks.sync(taskId);
    if (task && (task.status === "completed" || task.status === "failed" || task.status === "cancelled")) {
      this.events.publish({ type: "background-task.updated", payload: task, ts: Date.now() });
      if (task.status === "completed") {
        this.events.publish({ type: "background-task.completed", payload: task, ts: Date.now() });
      } else if (task.status === "cancelled") {
        this.events.publish({ type: "background-task.cancelled", payload: task, ts: Date.now() });
      } else {
        this.events.publish({ type: "background-task.failed", payload: task, ts: Date.now() });
      }
    }
    return task;
  }

  async cancelBackgroundTask(taskId: string): Promise<BackgroundTaskRecord | undefined> {
    const task = await this.backgroundTasks.cancel(taskId);
    if (task) {
      this.events.publish({ type: "background-task.cancelled", payload: task, ts: Date.now() });
    }
    return task;
  }

  async getBackgroundTaskExecutionState(taskId: string): Promise<BackgroundTaskExecutionState | undefined> {
    return await this.backgroundTasks.getExecutionState(taskId);
  }

  async recoverBackgroundTask(taskId: string): Promise<BackgroundTaskRecoveryResult | undefined> {
    const recovered = await this.backgroundTasks.recover(taskId);
    if (recovered?.changed) {
      this.publishBackgroundTaskRecovery(recovered);
    }
    return recovered;
  }

  async recoverBackgroundTasks(params: RecoverBackgroundTasksParams = {}): Promise<BackgroundTaskRecoveryResult[]> {
    const recovered = params.sessionId
      ? await this.backgroundTasks.recoverBySession(params.sessionId)
      : await this.backgroundTasks.recoverAll();
    const filtered = params.reasons?.length
      ? recovered.filter((item) => params.reasons?.includes(item.reason))
      : recovered;
    for (const item of filtered) {
      if (item.changed) {
        this.publishBackgroundTaskRecovery(item);
      }
    }
    return filtered;
  }

  async getBackgroundTaskOutput(
    taskId: string,
    options: number | ReadBackgroundTaskOutputOptions = {},
  ): Promise<{ taskId: string; output: string } | undefined> {
    const readOptions = typeof options === "number" ? { lineLimit: options } : options;
    const output = await this.backgroundTasks.getOutput(taskId, readOptions);
    return typeof output === "string" ? { taskId, output } : undefined;
  }

  private publishBackgroundTaskRecovery(recovered: BackgroundTaskRecoveryResult): void {
    this.events.publish({ type: "background-task.recovered", payload: recovered, ts: Date.now() });
    this.events.publish({ type: "background-task.updated", payload: recovered.task, ts: Date.now() });
    if (recovered.task.status === "completed") {
      this.events.publish({ type: "background-task.completed", payload: recovered.task, ts: Date.now() });
    } else if (recovered.task.status === "cancelled") {
      this.events.publish({ type: "background-task.cancelled", payload: recovered.task, ts: Date.now() });
    } else if (recovered.task.status === "failed") {
      this.events.publish({ type: "background-task.failed", payload: recovered.task, ts: Date.now() });
    }
  }

  async createTrainingJob(params: CreateTrainingJobParams): Promise<LocalTrainingJobManifest> {
    const job = await this.trainingJobs.create(params);
    this.events.publish({ type: "training-job.created", payload: job, ts: Date.now() });
    return job;
  }

  async getTrainingJob(jobId: string): Promise<LocalTrainingJobManifest | undefined> {
    return await this.trainingJobs.get(jobId);
  }

  async listTrainingJobs(): Promise<LocalTrainingJobManifest[]> {
    return await this.trainingJobs.list();
  }

  async updateTrainingJobStatus(
    jobId: string,
    status: LocalTrainingJobStatus,
  ): Promise<LocalTrainingJobManifest | undefined> {
    const job = await this.trainingJobs.updateStatus(jobId, status);
    if (job) {
      this.events.publish({ type: "training-job.updated", payload: job, ts: Date.now() });
    }
    return job;
  }

  async prepareTrainingJob(jobId: string): Promise<LocalTrainingJobManifest | undefined> {
    const job = await this.trainingJobs.prepareExecution(jobId);
    if (job) {
      this.events.publish({ type: "training-job.prepared", payload: job, ts: Date.now() });
    }
    return job;
  }

  async startTrainingJob(jobId: string): Promise<LocalTrainingJobManifest | undefined> {
    const job = await this.trainingJobs.startExecution(jobId);
    if (job) {
      this.events.publish({ type: "training-job.started", payload: job, ts: Date.now() });
    }
    return job;
  }

  async completeTrainingJob(
    jobId: string,
    params: CompleteTrainingJobParams,
  ): Promise<LocalTrainingJobManifest | undefined> {
    const job = await this.trainingJobs.completeExecution(jobId, params);
    if (job) {
      this.events.publish({ type: "training-job.completed", payload: job, ts: Date.now() });
    }
    return job;
  }

  async failTrainingJob(
    jobId: string,
    params: FailTrainingJobParams,
  ): Promise<LocalTrainingJobManifest | undefined> {
    const job = await this.trainingJobs.failExecution(jobId, params);
    if (job) {
      this.events.publish({ type: "training-job.failed", payload: job, ts: Date.now() });
    }
    return job;
  }

  async syncTrainingJob(jobId: string): Promise<LocalTrainingJobManifest | undefined> {
    const job = await this.trainingJobs.syncExecution(jobId);
    if (job && (job.status === "completed" || job.status === "failed")) {
      this.events.publish({ type: "training-job.synced", payload: job, ts: Date.now() });
    }
    return job;
  }

  async getTrainingJobPlan(jobId: string) {
    return await this.trainingJobs.getExecutionPlan(jobId);
  }

  async getTrainingJobLaunchScript(jobId: string): Promise<string | undefined> {
    return await this.trainingJobs.getLaunchScript(jobId);
  }

  async getTrainingJobExecutionState(jobId: string): Promise<TrainingExecutionState | undefined> {
    return await this.trainingJobs.getExecutionState(jobId);
  }

  async getTrainingJobLog(jobId: string, lineLimit?: number): Promise<{ jobId: string; log: string } | undefined> {
    const log = await this.trainingJobs.getExecutionLog(jobId, lineLimit);
    return typeof log === "string" ? { jobId, log } : undefined;
  }

  async registerPlugin(params: RegisterPluginParams): Promise<RegisteredPluginRecord> {
    const plugin = await this.plugins.register(params);
    this.events.publish({ type: "plugin.registered", payload: plugin, ts: Date.now() });
    return plugin;
  }

  async getPlugin(pluginId: string): Promise<RegisteredPluginRecord | undefined> {
    return await this.plugins.get(pluginId);
  }

  async listPlugins(params: ListPluginsParams = {}): Promise<RegisteredPluginRecord[]> {
    return params.capability
      ? await this.plugins.listByCapability(params.capability)
      : await this.plugins.list();
  }

  async updatePluginEnabled(pluginId: string, enabled: boolean): Promise<RegisteredPluginRecord | undefined> {
    const plugin = await this.plugins.setEnabled(pluginId, enabled);
    if (plugin) {
      this.events.publish({ type: "plugin.updated", payload: plugin, ts: Date.now() });
    }
    return plugin;
  }

  async activatePlugin(pluginId: string): Promise<OperatorPluginRuntime> {
    const runtime = await this.pluginRuntime.activate(pluginId);
    this.events.publish({ type: "plugin.activated", payload: { pluginId }, ts: Date.now() });
    return runtime;
  }

  async createCaptureConsent(params: CreateCaptureConsentParams): Promise<CaptureConsentGrant> {
    const grant = await this.capture.createConsent(params);
    this.events.publish({ type: "capture.consent-created", payload: grant, ts: Date.now() });
    return grant;
  }

  async listActiveCaptureConsents(now: Date = new Date()): Promise<CaptureConsentGrant[]> {
    return await this.capture.listActiveConsents(now);
  }

  async revokeCaptureConsent(grantId: string): Promise<CaptureConsentGrant | undefined> {
    const grant = await this.capture.revokeConsent(grantId);
    if (grant) {
      this.events.publish({ type: "capture.consent-revoked", payload: grant, ts: Date.now() });
    }
    return grant;
  }

  async recordBrowserCapture(input: BrowserCaptureInput): Promise<CaptureRecordResult> {
    const result = await this.capture.recordBrowser(input);
    if (result.recorded) {
      this.events.publish({ type: "capture.browser-recorded", payload: result.trajectory, ts: Date.now() });
    }
    return result;
  }

  async recordDeviceCapture(input: DeviceCaptureInput): Promise<CaptureRecordResult> {
    const result = await this.capture.recordDevice(input);
    if (result.recorded) {
      this.events.publish({ type: "capture.device-recorded", payload: result.trajectory, ts: Date.now() });
    }
    return result;
  }

  async recordOsObservation(input: OsObservationInput): Promise<CaptureRecordResult> {
    const result = await this.capture.observeOs(input);
    if (result.recorded) {
      this.events.publish({ type: "capture.os-recorded", payload: result.trajectory, ts: Date.now() });
    }
    return result;
  }

  async recordTurn(params: RecordTurnParams): Promise<{ trajectory: TrajectorySpan; skillCandidate?: SkillCandidate }> {
    const session = await this.sessions.get(params.sessionId);
    if (!session?.transcriptFile) {
      throw new Error(`Unknown session: ${params.sessionId}`);
    }
    const transcript = new JsonlTranscriptStore(session.transcriptFile);
    await transcript.appendMessage({ role: "user", content: params.userText });
    await transcript.appendMessage({ role: "assistant", content: params.assistantText });

    const trajectory = buildTrajectorySpan({
      id: randomUUID(),
      sessionId: params.sessionId,
      observations: [
        {
          kind: "observation",
          source: "operator",
          summary: params.userText,
          ts: Date.now(),
        },
      ],
      actions: [
        {
          kind: "action",
          tool: "assistant-response",
          summary: params.assistantText,
          ts: Date.now(),
        },
      ],
      outcome: {
        status: "success",
        summary: params.trajectorySummary ?? params.assistantText,
        reward: 1,
      },
    });
    await this.trajectories.add(trajectory);

    const summaryText = params.trajectorySummary ?? params.assistantText;
    await this.memory.addItem({
      type: "session-summary",
      summary: summaryText,
      sourceSessionId: params.sessionId,
      sourceTrajectoryId: trajectory.id,
      tags: ["session", "trajectory"],
    });

    let skillCandidate: SkillCandidate | undefined;
    if (summaryText.trim().length >= 24) {
      skillCandidate = await this.memory.addSkillCandidate({
        title: `Workflow from ${session.metadata.title ?? params.sessionId}`,
        summary: summaryText,
        sourceTrajectoryIds: [trajectory.id],
        confidence: 0.5,
      });
    }

    this.events.publish({
      type: "trajectory.recorded",
      payload: { sessionId: params.sessionId, trajectoryId: trajectory.id },
      ts: Date.now(),
    });

    return { trajectory, ...(skillCandidate ? { skillCandidate } : {}) };
  }

  async reviewTrajectory(params: ReviewTrajectoryParams): Promise<TrajectorySpan | undefined> {
    const trajectory = await this.trajectories.review(params.trajectoryId, {
      status: params.status,
      reviewedBy: params.reviewedBy,
      reviewNote: params.reviewNote,
      redactedObservations: params.redactedObservations,
      redactedActions: params.redactedActions,
      redactedTranscript: params.redactedTranscript,
    });
    if (trajectory) {
      this.events.publish({ type: "trajectory.reviewed", payload: trajectory, ts: Date.now() });
    }
    return trajectory;
  }

  async listReviewedTrajectories(status: TrajectoryReviewStatus = "approved"): Promise<TrajectorySpan[]> {
    return await this.trajectories.listReviewed(status);
  }

  async createReviewedExport(params: CreateReviewedExportParams): Promise<ReviewedExportManifest> {
    const manifest = await this.trainingExporter.createReviewedExport(params);
    this.events.publish({
      type: "training.export.created",
      payload: {
        outputFile: params.outputFile,
        reviewedBy: params.reviewedBy,
        trajectoryCount: manifest.trajectories.length,
        replayCount: manifest.replays.length,
      },
      ts: Date.now(),
    });
    return manifest;
  }

  async recall(query: string): Promise<RecallSummary> {
    const memories = await this.memory.searchDetailed(query);
    const skills = await this.memory.searchPromotedSkills(query);
    const trajectories = await this.trajectories.searchReviewed(query);
    const trajectoryHits = (
      await Promise.all(
        trajectories.map(async (trajectory) => {
          const fullTrajectory = await this.trajectories.get(trajectory.trajectoryId);
          if (!fullTrajectory) {
            return null;
          }
          return {
            ...trajectory,
            preview: this.replays.buildTrajectoryPreview(fullTrajectory),
          };
        }),
      )
    ).filter((trajectory): trajectory is (typeof trajectories)[number] & { preview: ReturnType<ReplayRuntimeService["buildTrajectoryPreview"]> } => trajectory !== null);
    const hits = [...memories, ...skills, ...trajectoryHits].sort((a, b) => b.score - a.score);
    return {
      query,
      hits,
      memories,
      skills,
      trajectories: trajectoryHits,
    };
  }

  async listSkillCandidates(status?: SkillCandidateStatus): Promise<SkillCandidate[]> {
    return await this.memory.listSkillCandidates(status);
  }

  async reviewSkillCandidate(
    candidateId: string,
    status: SkillCandidateStatus,
    reviewNote?: string,
  ): Promise<SkillCandidate | null> {
    const candidate = await this.memory.reviewSkillCandidate(candidateId, status, reviewNote);
    if (candidate) {
      this.events.publish({ type: "skill.reviewed", payload: candidate, ts: Date.now() });
    }
    return candidate;
  }

  async listPromotedSkills(): Promise<PromotedSkill[]> {
    return await this.memory.listPromotedSkills();
  }

  async listExecutableSkills(): Promise<ExecutableSkill[]> {
    return await this.memory.listExecutableSkills();
  }

  async listExecutableSkillRuns(skillId?: string): Promise<ExecutableSkillRun[]> {
    return await this.memory.listExecutableSkillRuns(skillId);
  }

  async createExecutableSkillFromPromoted(params: {
    promotedSkillId: string;
    steps: ExecutableSkillStep[];
  }): Promise<ExecutableSkill | null> {
    return await this.memory.createExecutableSkill(params);
  }

  async runExecutableSkill(params: {
    skillId: string;
    sessionId?: string;
    parentRunId?: string;
    executionConfig?: OperatorCliExecutionConfig;
    approvedCommandFingerprints?: ReadonlySet<string>;
  }): Promise<ExecutableSkillRun | null> {
    const skill = await this.memory.getExecutableSkill(params.skillId);
    if (!skill) {
      return null;
    }
    const stepResults: ExecutableSkillStepResult[] = [];
    const startedAt = new Date().toISOString();
    for (const step of skill.steps) {
      const stepStartedAt = new Date().toISOString();
      let subagentRunId: string | undefined;
      if (params.sessionId && params.parentRunId && step.agentRole && step.agentRole !== "main") {
        const childSession = await this.startSession({ title: `${skill.title} / ${step.title}`, agentId: step.agentRole });
        const subagent = await this.registerSubagent({
          sessionId: params.sessionId,
          parentRunId: params.parentRunId,
          childSessionId: childSession.id,
          title: step.title,
          metadata: { skillId: skill.id, stepId: step.id },
        });
        await this.updateSubagentStatus(subagent.runId, "completed");
        subagentRunId = subagent.runId;
      }
      const handled = await this.executeSkillStepWithPlugin(skill, step);
      const result = handled ?? (await this.executeBuiltInSkillStep(skill, step, params));
      stepResults.push({
        stepId: step.id,
        title: step.title,
        kind: step.kind,
        status: "completed",
        output: result.output,
        startedAt: stepStartedAt,
        completedAt: new Date().toISOString(),
        ...(step.agentRole ? { agentRole: step.agentRole } : {}),
        ...(subagentRunId ? { subagentRunId } : {}),
        ...(result.commentOutputs?.length ? { commentOutputs: result.commentOutputs } : {}),
      });
    }
    const preview = await this.buildSkillReplayPreview(skill.sourceTrajectoryIds);
    const run: ExecutableSkillRun = {
      id: randomUUID(),
      skillId: skill.id,
      ...(params.sessionId ? { sessionId: params.sessionId } : {}),
      ...(params.parentRunId ? { parentRunId: params.parentRunId } : {}),
      startedAt,
      completedAt: new Date().toISOString(),
      status: "completed",
      sourceTrajectoryIds: [...skill.sourceTrajectoryIds],
      replayPreview: preview,
      stepResults,
    };
    await this.memory.addExecutableSkillRun(run);
    this.events.publish({ type: "skill.run.completed", payload: run, ts: Date.now() });
    return run;
  }

  async promoteSkill(candidateId: string): Promise<PromotedSkill | null> {
    const skill = await this.memory.promoteSkill(candidateId);
    if (skill) {
      this.events.publish({ type: "skill.promoted", payload: skill, ts: Date.now() });
    }
    return skill;
  }

  async getTranscript(
    sessionId: string,
    options?: TranscriptReadOptions,
  ): Promise<TranscriptRecord[]> {
    const session = await this.sessions.get(sessionId);
    if (!session?.transcriptFile) {
      return [];
    }
    return await new JsonlTranscriptStore(session.transcriptFile).read(options);
  }

  private async ensureApprovalsLoaded(): Promise<void> {
    this.approvalsReady ??= this.approvals.loadPersisted();
    await this.approvalsReady;
  }

  async getLatestAssistantText(sessionId: string): Promise<string | undefined> {
    const session = await this.sessions.get(sessionId);
    if (!session?.transcriptFile) {
      return undefined;
    }
    return await new JsonlTranscriptStore(session.transcriptFile).readLatestAssistantText();
  }

  async listTrajectories(sessionId: string): Promise<TrajectorySpan[]> {
    return await this.trajectories.listBySession(sessionId);
  }

  async getReplay(sessionId: string): Promise<ReplayManifest | undefined> {
    return await this.replays.getSessionReplay(sessionId);
  }

  async listReplays(): Promise<ReplayManifest[]> {
    return await this.replays.listSessionReplays();
  }

  private async buildSkillReplayPreview(sourceTrajectoryIds: string[]) {
    const previews = await Promise.all(
      sourceTrajectoryIds.map(async (trajectoryId) => {
        const trajectory = await this.trajectories.get(trajectoryId);
        return trajectory ? this.replays.buildTrajectoryPreview(trajectory, 3) : [];
      }),
    );
    return previews.flat().slice(0, 6);
  }

  private async executeSkillStepWithPlugin(skill: ExecutableSkill, step: ExecutableSkillStep) {
    const handler = await this.pluginRuntime.getSkillStepHandler(step.kind);
    if (!handler) {
      return undefined;
    }
    return await handler({ skillId: skill.id, step });
  }

  private async executeBuiltInSkillStep(
    skill: ExecutableSkill,
    step: ExecutableSkillStep,
    params: {
      skillId: string;
      sessionId?: string;
      parentRunId?: string;
      executionConfig?: OperatorCliExecutionConfig;
      approvedCommandFingerprints?: ReadonlySet<string>;
    },
  ): Promise<{ output: string; commentOutputs?: string[] }> {
    if (step.kind === "summary") {
      return { output: step.content ?? skill.summary };
    }
    if (step.kind === "x-post") {
      const content = (step.content ?? skill.summary).trim();
      const maxCharacters = step.maxCharacters ?? 280;
      if (content.length <= maxCharacters) {
        return { output: content };
      }
      const head = content.slice(0, maxCharacters).trimEnd();
      const tail = content.slice(maxCharacters).trim();
      return {
        output: head,
        ...(step.overflowToComment && tail ? { commentOutputs: splitIntoCommentOutputs(tail, maxCharacters) } : {}),
      };
    }
    if (step.kind === "command") {
      if (!step.command) {
        return { output: "" };
      }
      if (params.executionConfig) {
        const authorization = await this.authorizeCommandExecution({
          sessionId: params.sessionId,
          cwd: this.rootDir,
          command: step.command,
          title: step.title,
          kind: "skill.command",
          source: "runtime",
          executionConfig: params.executionConfig,
          approvedFingerprints: params.approvedCommandFingerprints,
        });
        if (!authorization.allowed) {
          return { output: authorization.message };
        }
      }
      const { stdout, stderr } = await execFileAsync("bash", ["-lc", step.command], { cwd: this.rootDir, maxBuffer: 1024 * 1024 });
      const postHookResults = params.executionConfig
        ? await this.runPostCommandHooks({
            sessionId: params.sessionId,
            cwd: this.rootDir,
            command: step.command,
            title: step.title,
            kind: "skill.command",
            source: "runtime",
            executionConfig: params.executionConfig,
          })
        : [];
      return {
        output: [[stdout, stderr].filter(Boolean).join("").trim(), ...postHookResults.map((result) => formatHookResult(result))]
          .filter(Boolean)
          .join("\n")
          .trim(),
      };
    }
    return { output: "" };
  }
}

function formatHookResult(result: OperatorHookResult): string {
  const parts = [result.stdout, result.stderr].filter(Boolean).join(" ").trim();
  return parts ? `[${result.event}] ${parts}` : `[${result.event}] ${result.command}`;
}

function splitIntoCommentOutputs(content: string, maxCharacters: number): string[] {
  const comments: string[] = [];
  let remaining = content.trim();
  while (remaining.length > 0) {
    comments.push(remaining.slice(0, maxCharacters).trimEnd());
    remaining = remaining.slice(maxCharacters).trim();
  }
  return comments;
}
