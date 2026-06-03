import path from "node:path";
import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { randomUUID } from "node:crypto";
import type { OperatorCliExecutionConfig } from "../cli/config.js";
import {
  buildOperatorExecutionFingerprint,
  evaluateOperatorExecutionAction,
  runOperatorCommandHooks,
  runOperatorHooks,
  runOperatorPreToolUseHooks,
  type OperatorHookResult,
} from "../cli/execution-policy.js";
import { FileApprovalStore } from "../control-plane/approval-store.js";
import {
  InMemoryApprovalManager,
  type ApprovalRequest,
  type ApprovalResolution,
} from "../control-plane/approvals.js";
import {
  OperatorDeliveryService,
  type DeliveryAttemptResult,
  type DeliveryTarget,
} from "../control-plane/delivery.js";
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
import {
  cloneResolvedModelSelection,
  resolvePatchedModelSelection,
} from "./model-selection.js";
import { FileTaskStore, type OperatorTaskRecord, type OperatorTaskStatus } from "./task-store.js";
import {
  FilePlanStore,
  type OperatorPlanApprovalRecord,
  type OperatorPlanRecord,
  type OperatorPlanStatus,
  type OperatorPlanVerificationRecord,
  type OperatorPlanVerificationStatus,
} from "./plan-store.js";
import { FileMessageStore, type OperatorMessageRecord } from "./message-store.js";
import { FileSubagentRegistry, type SubagentRunRecord } from "./subagent-registry.js";
import {
  FileTrainingJobStore,
  type CompleteTrainingJobParams,
  type FailTrainingJobParams,
} from "../training/job-store.js";
import { FileOperatorCliTeamStore } from "../cli/team-store.js";
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
import type { OperatorResolvedModelSelection, SessionRecord } from "../harness/types.js";
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
  executionConfig?: OperatorCliExecutionConfig;
};

export type StartSessionParams = {
  title?: string;
  cwd?: string;
  agentId?: string;
  remoteId?: string;
  remoteSource?: string;
  parentSessionId?: string;
  modelSelection?: OperatorResolvedModelSelection;
};

export type CommandExecutionPolicyResult = {
  allowed: boolean;
  pendingApprovalId?: string;
  message: string;
  command: string;
  title?: string;
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

export type CreateTaskParams = {
  sessionId: string;
  subject: string;
  description: string;
  activeForm?: string;
  owner?: string;
  status?: OperatorTaskStatus;
  metadata?: Record<string, unknown>;
  blocks?: string[];
  blockedBy?: string[];
};

export type UpdateTaskParams = {
  subject?: string;
  description?: string;
  activeForm?: string | null;
  owner?: string | null;
  status?: OperatorTaskStatus;
  metadata?: Record<string, unknown>;
  blocks?: string[];
  blockedBy?: string[];
};

export type UpsertPlanParams = {
  sessionId: string;
  content: string;
  status?: OperatorPlanStatus;
};

export type RequestPlanApprovalParams = {
  sessionId: string;
  approverSessionId: string;
  content?: string;
};

export type RespondPlanApprovalParams = {
  requestId: string;
  sessionId: string;
  approve: boolean;
  feedback?: string;
};

export type UpdatePlanVerificationParams = {
  sessionId: string;
  status: OperatorPlanVerificationStatus;
  reminderAt?: string;
  notes?: string;
};

export type SendMessageParams = {
  fromSessionId: string;
  toSessionId: string;
  summary?: string;
  message: string;
  metadata?: Record<string, unknown>;
};

export type SendNotificationParams = {
  sessionId?: string;
  status: string;
  message: string;
  targets?: DeliveryTarget[];
};

export type OperatorNotificationRecord = {
  id: string;
  sessionId?: string;
  status: string;
  message: string;
  createdAt: string;
  deliveryResults: DeliveryAttemptResult[];
};

export type RegisterSubagentParams = {
  sessionId: string;
  parentRunId: string;
  childSessionId: string;
  title: string;
  metadata?: Record<string, unknown>;
};

export type SpawnSubagentParams = {
  sessionId: string;
  parentRunId: string;
  title: string;
  agentId?: string;
  cwd?: string;
  remoteId?: string;
  remoteSource?: string;
  modelPrimary?: string;
  modelFallbacks?: string[];
  metadata?: Record<string, unknown>;
};

export type SpawnSubagentResult = {
  subagent: SubagentRunRecord;
  childSession: SessionRecord;
  childRun: OperatorRunRecord;
};

export type TeammateStatus = SubagentRunRecord["status"];

export type StartTeammateParams = {
  teamName: string;
  sessionId: string;
  parentRunId: string;
  teammateName: string;
  title: string;
  agentId?: string;
  cwd?: string;
  remoteId?: string;
  remoteSource?: string;
  modelPrimary?: string;
  modelFallbacks?: string[];
  metadata?: Record<string, unknown>;
};

export type StartTeammateResult = {
  teamName: string;
  teammateName: string;
  taskSessionId: string;
  spawned: SpawnSubagentResult;
};

export type SendTeammateMessageParams = {
  teamName: string;
  fromSessionId: string;
  teammateSessionId?: string;
  teammateName?: string;
  summary?: string;
  message: string;
  metadata?: Record<string, unknown>;
};

export type SendTeammateMessageResult = {
  teamName: string;
  teammateName: string;
  teammateSessionId: string;
  message: OperatorMessageRecord;
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
  | "task.created"
  | "task.updated"
  | "plan.updated"
  | "plan.approval.requested"
  | "plan.approval.resolved"
  | "plan.verification.updated"
  | "message.sent"
  | "notification.sent"
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
  readonly tasks: FileTaskStore;
  readonly plans: FilePlanStore;
  readonly messages: FileMessageStore;
  readonly teams: FileOperatorCliTeamStore;
  readonly delivery: OperatorDeliveryService;
  readonly subagents: FileSubagentRegistry;
  readonly backgroundTasks: FileBackgroundTaskStore;
  readonly trainingJobs: FileTrainingJobStore;
  readonly plugins: FilePluginManifestRegistry;
  readonly pluginRuntime: OperatorPluginRuntimeManager;
  readonly trainingExporter: LocalTrainingExporter;
  private readonly transcriptsDir: string;
  private approvalsReady: Promise<void> | null = null;
  private executionConfig?: OperatorCliExecutionConfig;

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
    this.tasks = new FileTaskStore(path.join(options.rootDir, "tasks.json"));
    this.plans = new FilePlanStore(path.join(options.rootDir, "plans.json"));
    this.messages = new FileMessageStore(path.join(options.rootDir, "messages.json"));
    this.teams = new FileOperatorCliTeamStore(options.rootDir);
    this.delivery = new OperatorDeliveryService(options.rootDir);
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
    this.executionConfig = options.executionConfig;
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
        ...(params.remoteId ? { remoteId: params.remoteId } : {}),
        ...(params.remoteSource ? { remoteSource: params.remoteSource } : {}),
        ...(params.parentSessionId ? { parentSessionId: params.parentSessionId } : {}),
        ...(params.modelSelection ? { modelSelection: params.modelSelection } : {}),
      },
    });
    this.events.publish({ type: "session.started", payload: session, ts: Date.now() });
    await this.runLifecycleHooks({
      event: "SessionStart",
      sessionId: session.id,
      cwd: session.metadata.cwd ?? this.rootDir,
      input: {
        session,
      },
    });
    return session;
  }

  async getSession(sessionId: string): Promise<SessionRecord | undefined> {
    return await this.sessions.get(sessionId);
  }

  async listSessions(): Promise<SessionRecord[]> {
    return await this.sessions.list();
  }

  async findSessionByRemoteId(remoteId: string): Promise<SessionRecord | undefined> {
    const sessions = await this.sessions.list();
    return sessions.find(
      (session) =>
        session.metadata.remoteId === remoteId &&
        (session.status === "active" || session.status === "idle"),
    );
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
      await this.runLifecycleHooks({
        event: "SessionEnd",
        sessionId: session.id,
        cwd: session.metadata.cwd ?? this.rootDir,
        input: {
          status: session.status,
          session,
        },
      });
    }
    return session;
  }

  async failSession(sessionId: string): Promise<SessionRecord | undefined> {
    const session = await this.sessions.update(sessionId, { status: "failed" });
    if (session) {
      this.events.publish({ type: "session.failed", payload: session, ts: Date.now() });
      await this.runLifecycleHooks({
        event: "SessionEnd",
        sessionId: session.id,
        cwd: session.metadata.cwd ?? this.rootDir,
        input: {
          status: session.status,
          session,
        },
      });
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
    await this.runLifecycleHooks({
      event: "ApprovalRequested",
      sessionId: request.sessionId,
      approvalId: request.id,
      cwd: await this.resolveSessionHookCwd(request.sessionId),
      input: {
        request,
      },
    });
    return request;
  }

  async listApprovals(sessionId?: string): Promise<ApprovalRequest[]> {
    await this.ensureApprovalsLoaded();
    const approvals = this.approvals.list();
    return sessionId ? approvals.filter((approval) => approval.sessionId === sessionId) : approvals;
  }

  async resolveApproval(
    approvalId: string,
    decision: "approved" | "denied" | "expired",
    resolvedBy?: string,
  ): Promise<ApprovalResolution | null> {
    await this.ensureApprovalsLoaded();
    const request = this.approvals.list().find((item) => item.id === approvalId);
    const resolution = await this.approvals.resolve(approvalId, decision, resolvedBy);
    if (resolution) {
      this.events.publish({
        type: "approval.resolved",
        payload: {
          ...resolution,
          ...(request?.sessionId ? { sessionId: request.sessionId } : {}),
        },
        ts: Date.now(),
      });
      await this.runLifecycleHooks({
        event: "ApprovalResolved",
        sessionId: request?.sessionId,
        approvalId,
        cwd: await this.resolveSessionHookCwd(request?.sessionId),
        input: {
          request,
          resolution,
        },
      });
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

  async createTask(params: CreateTaskParams): Promise<OperatorTaskRecord> {
    const task = await this.tasks.create(params);
    this.events.publish({ type: "task.created", payload: task, ts: Date.now() });
    return task;
  }

  async getTask(taskId: string): Promise<OperatorTaskRecord | undefined> {
    return await this.tasks.get(taskId);
  }

  async updateTask(taskId: string, params: UpdateTaskParams): Promise<OperatorTaskRecord | undefined> {
    const task = await this.tasks.update(taskId, params);
    if (task) {
      this.events.publish({ type: "task.updated", payload: task, ts: Date.now() });
    }
    return task;
  }

  async listTasks(sessionId?: string): Promise<OperatorTaskRecord[]> {
    return sessionId ? await this.tasks.listBySession(sessionId) : await this.tasks.list();
  }

  async getPlan(sessionId: string): Promise<OperatorPlanRecord | undefined> {
    return await this.plans.getBySession(sessionId);
  }

  async upsertPlan(params: UpsertPlanParams): Promise<OperatorPlanRecord> {
    const existing = await this.plans.getBySession(params.sessionId);
    const plan = await this.plans.upsert({
      sessionId: params.sessionId,
      content: params.content,
      status: params.status ?? existing?.status,
      approval: existing?.approval,
      verification: existing?.verification,
    });
    this.events.publish({ type: "plan.updated", payload: plan, ts: Date.now() });
    return plan;
  }

  async requestPlanApproval(params: RequestPlanApprovalParams): Promise<{
    plan: OperatorPlanRecord;
    message: OperatorMessageRecord;
  }> {
    const existing = await this.plans.getBySession(params.sessionId);
    const content = params.content ?? existing?.content;
    if (!content) {
      throw new Error(`missing plan content for session ${params.sessionId}`);
    }
    const approval: OperatorPlanApprovalRecord = {
      requestId: randomUUID(),
      requestedBySessionId: params.sessionId,
      approverSessionId: params.approverSessionId,
      requestedAt: new Date().toISOString(),
    };
    const plan = await this.plans.upsert({
      sessionId: params.sessionId,
      content,
      status: "awaiting_approval",
      approval,
      verification: existing?.verification,
    });
    const message = await this.sendMessage({
      fromSessionId: params.sessionId,
      toSessionId: params.approverSessionId,
      summary: `Plan approval request ${approval.requestId}`,
      message: content,
      metadata: {
        type: "plan_approval_request",
        requestId: approval.requestId,
        planId: plan.id,
        fromSessionId: params.sessionId,
      },
    });
    this.events.publish({
      type: "plan.approval.requested",
      payload: {
        sessionId: plan.sessionId,
        fromSessionId: params.sessionId,
        toSessionId: params.approverSessionId,
        plan,
        message,
      },
      ts: Date.now(),
    });
    return { plan, message };
  }

  async respondPlanApproval(params: RespondPlanApprovalParams): Promise<{
    plan: OperatorPlanRecord;
    message: OperatorMessageRecord;
  }> {
    const plan = await this.plans.getByApprovalRequestId(params.requestId);
    if (!plan?.approval) {
      throw new Error(`unknown plan approval request: ${params.requestId}`);
    }
    if (plan.approval.approverSessionId !== params.sessionId) {
      throw new Error(`plan approval request ${params.requestId} is not assigned to session ${params.sessionId}`);
    }
    const approval: OperatorPlanApprovalRecord = {
      ...plan.approval,
      resolvedAt: new Date().toISOString(),
      approved: params.approve,
      ...(params.feedback ? { feedback: params.feedback } : {}),
    };
    const updatedPlan = await this.plans.update(plan.id, {
      status: params.approve ? "approved" : "rejected",
      approval,
    });
    if (!updatedPlan) {
      throw new Error(`unknown plan: ${plan.id}`);
    }
    const message = await this.sendMessage({
      fromSessionId: params.sessionId,
      toSessionId: plan.sessionId,
      summary: `Plan approval response ${params.requestId}`,
      message: params.feedback ?? (params.approve ? "approved" : "rejected"),
      metadata: {
        type: "plan_approval_response",
        requestId: params.requestId,
        approve: params.approve,
        ...(params.feedback ? { feedback: params.feedback } : {}),
      },
    });
    this.events.publish({
      type: "plan.approval.resolved",
      payload: {
        sessionId: updatedPlan.sessionId,
        fromSessionId: params.sessionId,
        toSessionId: updatedPlan.sessionId,
        plan: updatedPlan,
        message,
      },
      ts: Date.now(),
    });
    return { plan: updatedPlan, message };
  }

  async updatePlanVerification(params: UpdatePlanVerificationParams): Promise<OperatorPlanRecord> {
    const plan = await this.plans.getBySession(params.sessionId);
    if (!plan) {
      throw new Error(`unknown plan session: ${params.sessionId}`);
    }
    const verification: OperatorPlanVerificationRecord = {
      status: params.status,
      ...(params.reminderAt ? { reminderAt: params.reminderAt } : {}),
      ...(params.notes ? { notes: params.notes } : {}),
    };
    const updated = await this.plans.update(plan.id, {
      verification,
    });
    if (!updated) {
      throw new Error(`unknown plan: ${plan.id}`);
    }
    this.events.publish({ type: "plan.verification.updated", payload: updated, ts: Date.now() });
    return updated;
  }

  async sendMessage(params: SendMessageParams): Promise<OperatorMessageRecord> {
    const message = await this.messages.send(params);
    this.events.publish({ type: "message.sent", payload: message, ts: Date.now() });
    return message;
  }

  async sendNotification(params: SendNotificationParams): Promise<OperatorNotificationRecord> {
    const createdAt = new Date().toISOString();
    const id = randomUUID();
    const notification: OperatorNotificationRecord = {
      id,
      ...(params.sessionId ? { sessionId: params.sessionId } : {}),
      status: params.status,
      message: params.message,
      createdAt,
      deliveryResults: await this.delivery.deliver(params.targets ?? [{ kind: "local" }], {
        kind: "push-notification",
        notification: {
          id,
          ...(params.sessionId ? { sessionId: params.sessionId } : {}),
          status: params.status,
          message: params.message,
          createdAt,
        },
      }),
    };
    this.events.publish({ type: "notification.sent", payload: notification, ts: Date.now() });
    return notification;
  }

  async getMessage(messageId: string): Promise<OperatorMessageRecord | undefined> {
    return await this.messages.get(messageId);
  }

  async listMessages(sessionId?: string): Promise<OperatorMessageRecord[]> {
    return sessionId ? await this.messages.listBySession(sessionId) : await this.messages.list();
  }

  async listInbox(sessionId: string): Promise<OperatorMessageRecord[]> {
    return await this.messages.listInbox(sessionId);
  }

  async listOutbox(sessionId: string): Promise<OperatorMessageRecord[]> {
    return await this.messages.listOutbox(sessionId);
  }

  async startTeammate(params: StartTeammateParams): Promise<StartTeammateResult> {
    const team = await this.teams.get(params.teamName);
    if (!team) {
      throw new Error(`unknown team: ${params.teamName}`);
    }
    if (!team.taskSessionId) {
      throw new Error(`team ${params.teamName} has no task session`);
    }
    const teammate = await this.teams.getTeammate(params.teamName, params.teammateName);
    if (teammate) {
      throw new Error(`duplicate teammate: ${params.teamName}/${params.teammateName}`);
    }
    const spawned = await this.spawnSubagent({
      sessionId: params.sessionId,
      parentRunId: params.parentRunId,
      title: params.title,
      ...(params.agentId ? { agentId: params.agentId } : {}),
      ...(params.cwd ? { cwd: params.cwd } : {}),
      ...(params.remoteId ? { remoteId: params.remoteId } : {}),
      ...(params.remoteSource ? { remoteSource: params.remoteSource } : {}),
      ...(params.modelPrimary ? { modelPrimary: params.modelPrimary } : {}),
      ...(params.modelFallbacks ? { modelFallbacks: params.modelFallbacks } : {}),
      metadata: {
        ...(params.metadata ?? {}),
        teamName: params.teamName,
        teammateName: params.teammateName,
        taskSessionId: team.taskSessionId,
      },
    });
    await this.teams.addTeammate({
      teamName: params.teamName,
      teammateName: params.teammateName,
      sessionId: spawned.childSession.id,
      subagentRunId: spawned.subagent.runId,
      childRunId: spawned.childRun.id,
      title: params.title,
      ...(params.agentId ? { agentId: params.agentId } : {}),
      status: spawned.subagent.status,
    });
    return {
      teamName: params.teamName,
      teammateName: params.teammateName,
      taskSessionId: team.taskSessionId,
      spawned,
    };
  }

  async sendTeammateMessage(params: SendTeammateMessageParams): Promise<SendTeammateMessageResult> {
    const teammate = params.teammateName
      ? await this.teams.getTeammate(params.teamName, params.teammateName)
      : await this.resolveTeammateBySession(params.teamName, params.teammateSessionId);
    if (!teammate) {
      throw new Error(`unknown teammate in team ${params.teamName}`);
    }
    const message = await this.sendMessage({
      fromSessionId: params.fromSessionId,
      toSessionId: teammate.sessionId,
      ...(params.summary ? { summary: params.summary } : {}),
      message: params.message,
      ...(params.metadata ? { metadata: params.metadata } : {}),
    });
    return {
      teamName: params.teamName,
      teammateName: teammate.name,
      teammateSessionId: teammate.sessionId,
      message,
    };
  }

  async listTeammates(teamName: string) {
    return await this.teams.listTeammates(teamName);
  }

  async getTeammate(teamName: string, teammateName: string) {
    return await this.teams.getTeammate(teamName, teammateName);
  }

  async updateTeammateStatus(
    teamName: string,
    teammateName: string,
    status: TeammateStatus,
  ) {
    const teammate = await this.teams.getTeammate(teamName, teammateName);
    if (!teammate) {
      return undefined;
    }
    const subagent = await this.updateSubagentStatus(teammate.subagentRunId, status);
    if (!subagent) {
      return undefined;
    }
    await this.updateRunStatus(teammate.childRunId, status);
    return await this.teams.updateTeammate(teamName, teammateName, {
      status,
      subagentRunId: subagent.runId,
    });
  }

  async pauseActiveRun(sessionId: string, reason?: string): Promise<OperatorRunRecord | undefined> {
    const run = await this.getActiveRun(sessionId);
    if (!run) {
      return undefined;
    }
    return await this.updateRunStatus(run.id, "paused", {
      remoteControlAction: "pause",
      remoteControlSource: "remote-control",
      ...(reason ? { remoteControlReason: reason } : {}),
    });
  }

  async registerSubagent(params: RegisterSubagentParams): Promise<SubagentRunRecord> {
    const subagent = await this.subagents.register(params);
    this.events.publish({ type: "subagent.registered", payload: subagent, ts: Date.now() });
    return subagent;
  }

  async spawnSubagent(params: SpawnSubagentParams): Promise<SpawnSubagentResult> {
    const modelSelection = await this.resolveSubagentModelSelection(params);
    const metadata = mergeModelSelectionMetadata(params.metadata, modelSelection);
    const childSession = await this.startSession({
      title: params.title,
      ...(params.cwd ? { cwd: params.cwd } : {}),
      ...(params.agentId ? { agentId: params.agentId } : {}),
      ...(params.remoteId ? { remoteId: params.remoteId } : {}),
      ...(params.remoteSource ? { remoteSource: params.remoteSource } : {}),
      parentSessionId: params.sessionId,
      ...(modelSelection ? { modelSelection } : {}),
    });
    const subagent = await this.registerSubagent({
      sessionId: params.sessionId,
      parentRunId: params.parentRunId,
      childSessionId: childSession.id,
      title: params.title,
      metadata,
    });
    const childRun = await this.startRun({
      sessionId: childSession.id,
      parentRunId: subagent.runId,
      title: params.title,
      metadata,
    });
    const runningSubagent = await this.updateSubagentStatus(subagent.runId, "running");
    if (!runningSubagent) {
      throw new Error(`unknown subagent: ${subagent.runId}`);
    }
    return { subagent: runningSubagent, childSession, childRun };
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
      return {
        allowed: false,
        message: decision.reason,
        command: action.command,
        ...(action.title ? { title: action.title } : {}),
        preHookResults: [],
        postHookResults: [],
      };
    }

    let effectiveAction = action;
    let preHookResults: OperatorHookResult[] = [];

    try {
      const preToolUse = await runOperatorPreToolUseHooks({
        config: params.executionConfig,
        action,
      });
      effectiveAction = preToolUse.action;
      preHookResults = [...preToolUse.results];
      if (preToolUse.outcome === "deny") {
        return {
          allowed: false,
          message: preToolUse.reason ?? "PreToolUse hook denied execution.",
          command: effectiveAction.command,
          ...(effectiveAction.title ? { title: effectiveAction.title } : {}),
          preHookResults,
          postHookResults: [],
        };
      }

      const requiresApproval = decision.outcome === "request-approval" || preToolUse.outcome === "request-approval";
      if (requiresApproval) {
        const fingerprint = buildOperatorExecutionFingerprint(effectiveAction);
        const approval = await this.promptApproval({
          sessionId: params.sessionId ?? "operator-cli",
          title: "Approve local command execution",
          summary: `${params.kind}: ${effectiveAction.command}`,
          scope: params.kind,
          metadata: {
            command: effectiveAction.command,
            ...(effectiveAction.title ? { title: effectiveAction.title } : {}),
            cwd: effectiveAction.cwd,
            permissionMode: decision.permissionMode,
            dangerousReasons: decision.dangerousReasons,
            fingerprint,
            reason: preToolUse.reason ?? decision.reason,
          },
        });
        return {
          allowed: false,
          pendingApprovalId: approval.id,
          message: `Approval required: ${approval.id}. Use /approve ${approval.id} or /deny ${approval.id} and rerun the command.`,
          command: effectiveAction.command,
          ...(effectiveAction.title ? { title: effectiveAction.title } : {}),
          preHookResults,
          postHookResults: [],
        };
      }

      const preCommandResults = await runOperatorCommandHooks({
        config: params.executionConfig,
        event: "PreCommand",
        action: effectiveAction,
      });
      preHookResults = [...preHookResults, ...preCommandResults];
      return {
        allowed: true,
        message: preToolUse.additionalContext ?? preToolUse.reason ?? decision.reason,
        command: effectiveAction.command,
        ...(effectiveAction.title ? { title: effectiveAction.title } : {}),
        preHookResults,
        postHookResults: [],
      };
    } catch (error) {
      return {
        allowed: false,
        message: `Pre-command hook blocked execution: ${error instanceof Error ? error.message : String(error)}`,
        command: effectiveAction.command,
        ...(effectiveAction.title ? { title: effectiveAction.title } : {}),
        preHookResults,
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

  setExecutionConfig(config?: OperatorCliExecutionConfig): void {
    this.executionConfig = config;
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

  private async runLifecycleHooks(params: {
    event: "SessionStart" | "SessionEnd" | "ApprovalRequested" | "ApprovalResolved";
    cwd: string;
    sessionId?: string;
    runId?: string;
    approvalId?: string;
    input: Record<string, unknown>;
  }): Promise<OperatorHookResult[]> {
    if (!this.executionConfig) {
      return [];
    }
    try {
      return await runOperatorHooks({
        config: this.executionConfig,
        context: {
          event: params.event,
          cwd: params.cwd,
          permissionMode: this.executionConfig.permissionMode,
          ...(params.sessionId ? { sessionId: params.sessionId } : {}),
          ...(params.runId ? { runId: params.runId } : {}),
          ...(params.approvalId ? { approvalId: params.approvalId } : {}),
          input: params.input,
        },
      });
    } catch {
      return [];
    }
  }

  private async resolveSessionHookCwd(sessionId?: string): Promise<string> {
    if (!sessionId) {
      return this.rootDir;
    }
    const session = await this.sessions.get(sessionId);
    return session?.metadata.cwd ?? this.rootDir;
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

  private async resolveTeammateBySession(teamName: string, teammateSessionId?: string) {
    if (!teammateSessionId) {
      return undefined;
    }
    const teammates = await this.teams.listTeammates(teamName);
    return teammates.find((teammate) => teammate.sessionId === teammateSessionId);
  }

  private async resolveSubagentModelSelection(
    params: SpawnSubagentParams,
  ): Promise<OperatorResolvedModelSelection | undefined> {
    const parentRun = await this.getRun(params.parentRunId);
    const parentSession = await this.getSession(params.sessionId);
    const parentSelection = getModelSelectionFromMetadata(parentRun?.metadata) ?? parentSession?.metadata.modelSelection;
    const hasModelPrimary = hasOwn(params, "modelPrimary");
    const hasModelFallbacks = hasOwn(params, "modelFallbacks");
    if (hasModelPrimary || hasModelFallbacks) {
      return resolvePatchedModelSelection({
        baseSelection: parentSelection,
        source: "override",
        primary: params.modelPrimary,
        hasPrimary: hasModelPrimary,
        fallbacks: params.modelFallbacks,
        hasFallbacks: hasModelFallbacks,
      });
    }
    const selection = cloneResolvedModelSelection(parentSelection);
    return selection
      ? {
          ...selection,
          source: "inherited",
        }
      : undefined;
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
      let authorizedCommand = step.command;
      let authorizedTitle = step.title;
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
        authorizedCommand = authorization.command;
        authorizedTitle = authorization.title;
      }
      const { stdout, stderr } = await execFileAsync("bash", ["-lc", authorizedCommand], { cwd: this.rootDir, maxBuffer: 1024 * 1024 });
      const postHookResults = params.executionConfig
        ? await this.runPostCommandHooks({
            sessionId: params.sessionId,
            cwd: this.rootDir,
            command: authorizedCommand,
            ...(authorizedTitle ? { title: authorizedTitle } : {}),
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

function hasOwn<T extends object, K extends PropertyKey>(value: T, key: K): boolean {
  return Object.prototype.hasOwnProperty.call(value, key);
}

function getModelSelectionFromMetadata(metadata?: Record<string, unknown>): OperatorResolvedModelSelection | undefined {
  const value = metadata?.modelSelection;
  if (!value || typeof value !== "object") {
    return undefined;
  }
  const candidate = value as Record<string, unknown>;
  if (
    candidate.source !== "job"
    && candidate.source !== "inherited"
    && candidate.source !== "override"
    && candidate.source !== "default"
  ) {
    return undefined;
  }
  const primary = typeof candidate.primary === "string" ? candidate.primary : undefined;
  const fallbacks = Array.isArray(candidate.fallbacks)
    ? candidate.fallbacks.filter((item): item is string => typeof item === "string")
    : undefined;
  return {
    ...(primary ? { primary } : {}),
    ...(fallbacks ? { fallbacks } : {}),
    source: candidate.source,
  };
}

function mergeModelSelectionMetadata(
  metadata: Record<string, unknown> | undefined,
  modelSelection: OperatorResolvedModelSelection | undefined,
): Record<string, unknown> | undefined {
  if (!modelSelection) {
    return metadata;
  }
  return {
    ...(metadata ?? {}),
    modelSelection,
  };
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
