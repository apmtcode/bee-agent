import type { ApprovalDecision, ApprovalRequest } from "./approvals.js";
import { OperatorCronService } from "./cron-service.js";
import type { CronDeliveryConfig, DeliveryTarget } from "./delivery.js";
import {
  FilePairingStore,
  PairingTicketError,
  type PairingTicket,
  type PairingTicketStatus,
} from "./pairing-store.js";
import { buildRuntimeEventFilter } from "./subscriptions.js";
import type { BackgroundTaskKind, BackgroundTaskRecoveryReason } from "../harness/background-tasks.js";
import type { TranscriptReadOptions } from "../harness/transcript-store.js";
import type { SessionRecord } from "../harness/types.js";
import type { OperatorEvent } from "../kernel/event-bus.js";
import type { SkillCandidateStatus } from "../memory/types.js";
import type { StandaloneOperatorRuntime } from "../orchestrator/operator-runtime.js";
import type { ReviewedExportManifest } from "../training/export-manifest.js";
import type { TrainingMode } from "../training/export-manifest.js";
import type { LocalTrainingJobStatus } from "../training/job-manifest.js";
import type { OperatorPluginCapability, OperatorPluginManifest } from "../plugins/manifest.js";
import type { BrowserCaptureInput } from "../capture/browser-adapter.js";
import type { DeviceCaptureInput } from "../capture/device-adapter.js";
import type { OsObservationInput } from "../capture/os-observer.js";
import type {
  CaptureTier,
  ReviewedTranscriptMessage,
  ReviewedTrajectoryAction,
  ReviewedTrajectoryObservation,
  TrajectoryReviewStatus,
} from "../capture/trajectory.js";

export type ControlPlaneRequest = {
  method: string;
  params?: Record<string, unknown>;
};

export type ControlPlaneSuccess<T = unknown> = {
  ok: true;
  result: T;
};

export type ControlPlaneFailure = {
  ok: false;
  error: {
    code: "INVALID_REQUEST" | "NOT_FOUND" | "INTERNAL_ERROR";
    message: string;
  };
};

export type ControlPlaneResponse<T = unknown> = ControlPlaneSuccess<T> | ControlPlaneFailure;

export type ControlPlaneServerOptions = {
  runtime: StandaloneOperatorRuntime;
  cron?: OperatorCronService;
  pairing?: FilePairingStore;
};

export type PairingTicketCreateResult = PairingTicket;

export type PairingTicketResolveResult = PairingTicket;

export type SessionBootstrapResult = {
  session: SessionRecord;
  created: boolean;
  resumed: boolean;
  approvals: ApprovalRequest[];
  events: OperatorEvent[];
};

export class OperatorControlPlaneServer {
  private readonly cron: OperatorCronService;
  private readonly pairing: FilePairingStore;

  get runtime(): StandaloneOperatorRuntime {
    return this.options.runtime;
  }

  constructor(private readonly options: ControlPlaneServerOptions) {
    this.cron = options.cron ?? new OperatorCronService(options.runtime.rootDir, { runtime: options.runtime });
    this.pairing = options.pairing ?? new FilePairingStore(buildPairingStoreFilePath(options.runtime.rootDir));
  }

  async handle(request: ControlPlaneRequest): Promise<ControlPlaneResponse> {
    try {
      switch (request.method) {
        case "sessions.list":
          return ok(await this.options.runtime.listSessions());
        case "sessions.get": {
          const sessionId = getString(request.params, "sessionId");
          const session = await this.options.runtime.getSession(sessionId);
          return session ? ok(session) : notFound(`unknown session: ${sessionId}`);
        }
        case "sessions.bootstrap": {
          const requestedSessionId = getOptionalString(request.params, "sessionId");
          const pairingCode = getOptionalString(request.params, "pairingCode");
          let remoteId = getOptionalString(request.params, "remoteId");
          let remoteSource = getOptionalString(request.params, "remoteSource");
          const title = getOptionalString(request.params, "title");
          const cwd = getOptionalString(request.params, "cwd");
          const agentId = getOptionalString(request.params, "agentId");
          const resume = getOptionalBoolean(request.params, "resume") ?? true;
          let pairingTicket: PairingTicket | undefined;
          if (!requestedSessionId && pairingCode) {
            pairingTicket = await this.pairing.getTicket(pairingCode);
            if (!pairingTicket) {
              return notFound(`unknown pairing ticket: ${pairingCode}`);
            }
            if (pairingTicket.status !== "approved") {
              return invalidRequest(`pairing ticket is ${pairingTicket.status}: ${pairingCode}`);
            }
            remoteId = pairingTicket.remoteId;
            remoteSource = pairingTicket.remoteSource;
          }
          const session = requestedSessionId
            ? await this.options.runtime.getSession(requestedSessionId)
            : remoteId
              ? await this.options.runtime.findSessionByRemoteId(remoteId)
              : undefined;
          const created = !session;
          const resolvedSession = session ?? await this.options.runtime.startSession({ title, cwd, agentId, remoteId, remoteSource });
          if (!resolvedSession) {
            return notFound(`unknown session: ${requestedSessionId}`);
          }
          if (pairingTicket) {
            await this.pairing.redeemTicket(pairingTicket.code, { sessionId: resolvedSession.id });
          }
          const resumed = Boolean(!created && resume && resolvedSession.status === "idle" && (await this.options.runtime.resumeSession(resolvedSession.id)));
          const effectiveSession = resumed ? await this.options.runtime.getSession(resolvedSession.id) ?? resolvedSession : resolvedSession;
          return ok(await buildSessionBootstrapResult(this.options.runtime, effectiveSession, {
            created,
            resumed,
            runId: getOptionalString(request.params, "runId"),
            family: getOptionalRuntimeEventFamily(request.params, "family"),
          }));
        }
        case "sessions.resume": {
          const sessionId = getString(request.params, "sessionId");
          const session = await this.options.runtime.resumeSession(sessionId);
          return session ? ok(session) : notFound(`unknown session: ${sessionId}`);
        }
        case "sessions.idle": {
          const sessionId = getString(request.params, "sessionId");
          const session = await this.options.runtime.markSessionIdle(sessionId);
          return session ? ok(session) : notFound(`unknown session: ${sessionId}`);
        }
        case "sessions.complete": {
          const sessionId = getString(request.params, "sessionId");
          const session = await this.options.runtime.completeSession(sessionId);
          return session ? ok(session) : notFound(`unknown session: ${sessionId}`);
        }
        case "sessions.fail": {
          const sessionId = getString(request.params, "sessionId");
          const session = await this.options.runtime.failSession(sessionId);
          return session ? ok(session) : notFound(`unknown session: ${sessionId}`);
        }
        case "pairing.create": {
          return ok(
            await this.pairing.createTicket({
              remoteSource: getOptionalString(request.params, "remoteSource") ?? "gateway",
              remoteId: getOptionalString(request.params, "remoteId"),
              expiresAt: getOptionalString(request.params, "expiresAt") ?? new Date(Date.now() + 10 * 60_000).toISOString(),
            }),
          );
        }
        case "pairing.list":
          return ok(await this.pairing.listTickets(getOptionalPairingTicketStatus(request.params, "status")));
        case "pairing.resolve": {
          const identifier = getOptionalString(request.params, "ticketId") ?? getOptionalString(request.params, "code");
          if (!identifier) {
            return invalidRequest("pairing.resolve requires ticketId or code");
          }
          return ok(
            await this.pairing.resolveTicket(identifier, {
              decision: getPairingResolutionDecision(request.params, "decision"),
              resolvedBy: getOptionalString(request.params, "resolvedBy"),
            }),
          );
        }
        case "transcript.get": {
          const sessionId = getString(request.params, "sessionId");
          const transcript = await this.options.runtime.getTranscript(sessionId, getTranscriptReadOptions(request.params));
          return ok(transcript);
        }
        case "approvals.list":
          return ok(await this.options.runtime.listApprovals(getOptionalString(request.params, "sessionId")));
        case "approvals.resolve": {
          const approvalId = getString(request.params, "approvalId");
          const decision = getApprovalDecision(request.params, "decision");
          const resolvedBy = getOptionalString(request.params, "resolvedBy");
          const resolution = await this.options.runtime.resolveApproval(approvalId, decision, resolvedBy);
          return resolution ? ok(resolution) : notFound(`unknown approval: ${approvalId}`);
        }
        case "trajectories.list": {
          const sessionId = getString(request.params, "sessionId");
          return ok(await this.options.runtime.listTrajectories(sessionId));
        }
        case "trajectories.review": {
          const trajectoryId = getString(request.params, "trajectoryId");
          const trajectory = await this.options.runtime.reviewTrajectory({
            trajectoryId,
            status: getTrajectoryReviewStatus(request.params, "status"),
            reviewedBy: getString(request.params, "reviewedBy"),
            reviewNote: getOptionalString(request.params, "reviewNote"),
            redactedObservations: getOptionalReviewedObservations(request.params, "redactedObservations"),
            redactedActions: getOptionalReviewedActions(request.params, "redactedActions"),
            redactedTranscript: getOptionalReviewedTranscript(request.params, "redactedTranscript"),
          });
          return trajectory ? ok(trajectory) : notFound(`unknown trajectory: ${trajectoryId}`);
        }
        case "trajectories.reviewed":
          return ok(await this.options.runtime.listReviewedTrajectories(getOptionalTrajectoryReviewStatus(request.params, "status") ?? "approved"));
        case "replays.get": {
          const sessionId = getString(request.params, "sessionId");
          const replay = await this.options.runtime.getReplay(sessionId);
          return replay ? ok(replay) : notFound(`unknown replay session: ${sessionId}`);
        }
        case "replays.list":
          return ok(await this.options.runtime.listReplays());
        case "runs.start":
          return ok(
            await this.options.runtime.startRun({
              sessionId: getString(request.params, "sessionId"),
              title: getString(request.params, "title"),
              parentRunId: getOptionalString(request.params, "parentRunId"),
              metadata: getOptionalRecord(request.params, "metadata"),
            }),
          );
        case "runs.get": {
          const runId = getString(request.params, "runId");
          const run = await this.options.runtime.getRun(runId);
          return run ? ok(run) : notFound(`unknown run: ${runId}`);
        }
        case "runs.list":
          return ok(await this.options.runtime.listRuns(getOptionalString(request.params, "sessionId")));
        case "runs.active": {
          const sessionId = getOptionalString(request.params, "sessionId");
          const run = await this.options.runtime.getActiveRun(sessionId);
          return run ? ok(run) : notFound(`no active run${sessionId ? ` for session: ${sessionId}` : ""}`);
        }
        case "runs.events": {
          const filter = buildRuntimeEventFilter({
            sessionId: getOptionalString(request.params, "sessionId"),
            runId: getOptionalString(request.params, "runId"),
            family: getOptionalRuntimeEventFamily(request.params, "family"),
          });
          return ok(this.options.runtime.events.snapshot(filter));
        }
        case "runs.update": {
          const runId = getString(request.params, "runId");
          const run = await this.options.runtime.updateRunStatus(
            runId,
            getRunStatus(request.params, "status"),
            getOptionalRecord(request.params, "metadata"),
          );
          return run ? ok(run) : notFound(`unknown run: ${runId}`);
        }
        case "subagents.register":
          return ok(
            await this.options.runtime.registerSubagent({
              sessionId: getString(request.params, "sessionId"),
              parentRunId: getString(request.params, "parentRunId"),
              childSessionId: getString(request.params, "childSessionId"),
              title: getString(request.params, "title"),
              metadata: getOptionalRecord(request.params, "metadata"),
            }),
          );
        case "subagents.get": {
          const runId = getString(request.params, "runId");
          const subagent = await this.options.runtime.getSubagent(runId);
          return subagent ? ok(subagent) : notFound(`unknown subagent: ${runId}`);
        }
        case "subagents.list":
          return ok(await this.options.runtime.listSubagents(getOptionalString(request.params, "parentRunId")));
        case "subagents.active":
          return ok(await this.options.runtime.listActiveSubagents());
        case "subagents.update": {
          const runId = getString(request.params, "runId");
          const subagent = await this.options.runtime.updateSubagentStatus(
            runId,
            getRunStatus(request.params, "status"),
          );
          return subagent ? ok(subagent) : notFound(`unknown subagent: ${runId}`);
        }
        case "memory.recall": {
          const query = getString(request.params, "query");
          return ok(await this.options.runtime.recall(query));
        }
        case "skills.candidates.list":
          return ok(await this.options.runtime.listSkillCandidates(getOptionalSkillCandidateStatus(request.params, "status")));
        case "skills.candidates.review": {
          const candidateId = getString(request.params, "candidateId");
          const candidate = await this.options.runtime.reviewSkillCandidate(
            candidateId,
            getSkillCandidateStatus(request.params, "status"),
            getOptionalString(request.params, "reviewNote"),
          );
          return candidate ? ok(candidate) : notFound(`unknown skill candidate: ${candidateId}`);
        }
        case "skills.promote": {
          const candidateId = getString(request.params, "candidateId");
          const skill = await this.options.runtime.promoteSkill(candidateId);
          return skill ? ok(skill) : notFound(`unknown skill candidate: ${candidateId}`);
        }
        case "skills.list":
          return ok(await this.options.runtime.listPromotedSkills());
        case "skills.executable.create": {
          const executableSkill = await this.options.runtime.createExecutableSkillFromPromoted({
            promotedSkillId: getString(request.params, "promotedSkillId"),
            steps: getExecutableSkillSteps(request.params, "steps"),
          });
          return executableSkill ? ok(executableSkill) : notFound(`unknown promoted skill: ${getString(request.params, "promotedSkillId")}`);
        }
        case "skills.executable.list":
          return ok(await this.options.runtime.listExecutableSkills());
        case "skills.executable.run": {
          const skillRun = await this.options.runtime.runExecutableSkill({
            skillId: getString(request.params, "skillId"),
            sessionId: getOptionalString(request.params, "sessionId"),
            parentRunId: getOptionalString(request.params, "parentRunId"),
          });
          return skillRun ? ok(skillRun) : notFound(`unknown executable skill: ${getString(request.params, "skillId")}`);
        }
        case "skills.executable.runs":
          return ok(await this.options.runtime.listExecutableSkillRuns(getOptionalString(request.params, "skillId")));
        case "background.tasks.start":
          return ok(
            await this.options.runtime.startBackgroundTask({
              sessionId: getOptionalString(request.params, "sessionId"),
              title: getString(request.params, "title"),
              command: getString(request.params, "command"),
              cwd: getOptionalString(request.params, "cwd"),
              kind: getOptionalBackgroundTaskKind(request.params, "kind"),
              metadata: getOptionalRecord(request.params, "metadata"),
            }),
          );
        case "background.tasks.get": {
          const taskId = getString(request.params, "taskId");
          const task = await this.options.runtime.getBackgroundTask(taskId);
          return task ? ok(task) : notFound(`unknown background task: ${taskId}`);
        }
        case "background.tasks.list":
          return ok(await this.options.runtime.listBackgroundTasks(getOptionalString(request.params, "sessionId")));
        case "background.tasks.active": {
          const sessionId = getOptionalString(request.params, "sessionId");
          const task = await this.options.runtime.getActiveBackgroundTask(sessionId);
          return task ? ok(task) : notFound(`no active background task${sessionId ? ` for session: ${sessionId}` : ""}`);
        }
        case "background.tasks.sync": {
          const taskId = getString(request.params, "taskId");
          const task = await this.options.runtime.syncBackgroundTask(taskId);
          return task ? ok(task) : notFound(`unknown background task: ${taskId}`);
        }
        case "background.tasks.cancel": {
          const taskId = getString(request.params, "taskId");
          const task = await this.options.runtime.cancelBackgroundTask(taskId);
          return task ? ok(task) : notFound(`unknown background task: ${taskId}`);
        }
        case "background.tasks.state": {
          const taskId = getString(request.params, "taskId");
          const state = await this.options.runtime.getBackgroundTaskExecutionState(taskId);
          return state ? ok(state) : notFound(`unknown background task: ${taskId}`);
        }
        case "background.tasks.recover": {
          const taskId = getString(request.params, "taskId");
          const recovered = await this.options.runtime.recoverBackgroundTask(taskId);
          return recovered ? ok(recovered) : notFound(`unknown background task: ${taskId}`);
        }
        case "background.tasks.recoverAll":
          return ok(
            await this.options.runtime.recoverBackgroundTasks({
              sessionId: getOptionalString(request.params, "sessionId"),
              reasons: getOptionalBackgroundTaskRecoveryReasons(request.params, "reasons"),
            }),
          );
        case "background.tasks.output": {
          const taskId = getString(request.params, "taskId");
          const output = await this.options.runtime.getBackgroundTaskOutput(taskId, getOptionalNumber(request.params, "lineLimit"));
          return output ? ok(output) : notFound(`unknown background task: ${taskId}`);
        }
        case "training.exports.create":
          return ok(
            await this.options.runtime.createReviewedExport({
              reviewedBy: getString(request.params, "reviewedBy"),
              purpose: getString(request.params, "purpose"),
              outputFile: getString(request.params, "outputFile"),
              modes: getOptionalTrainingModes(request.params, "modes"),
            }),
          );
        case "training.jobs.create":
          return ok(
            await this.options.runtime.createTrainingJob({
              exportManifest: getReviewedExportManifest(request.params, "exportManifest"),
              mode: getTrainingMode(request.params, "mode"),
            }),
          );
        case "training.jobs.get": {
          const jobId = getString(request.params, "jobId");
          const job = await this.options.runtime.getTrainingJob(jobId);
          return job ? ok(job) : notFound(`unknown training job: ${jobId}`);
        }
        case "training.jobs.list":
          return ok(await this.options.runtime.listTrainingJobs());
        case "training.jobs.update": {
          const jobId = getString(request.params, "jobId");
          const job = await this.options.runtime.updateTrainingJobStatus(
            jobId,
            getTrainingJobStatus(request.params, "status"),
          );
          return job ? ok(job) : notFound(`unknown training job: ${jobId}`);
        }
        case "training.jobs.prepare": {
          const jobId = getString(request.params, "jobId");
          const job = await this.options.runtime.prepareTrainingJob(jobId);
          return job ? ok(job) : notFound(`unknown training job: ${jobId}`);
        }
        case "training.jobs.start": {
          const jobId = getString(request.params, "jobId");
          const job = await this.options.runtime.startTrainingJob(jobId);
          return job ? ok(job) : notFound(`unknown training job: ${jobId}`);
        }
        case "training.jobs.complete": {
          const jobId = getString(request.params, "jobId");
          const job = await this.options.runtime.completeTrainingJob(jobId, {
            outputModelRef: getString(request.params, "outputModelRef"),
          });
          return job ? ok(job) : notFound(`unknown training job: ${jobId}`);
        }
        case "training.jobs.fail": {
          const jobId = getString(request.params, "jobId");
          const job = await this.options.runtime.failTrainingJob(jobId, {
            reason: getString(request.params, "reason"),
          });
          return job ? ok(job) : notFound(`unknown training job: ${jobId}`);
        }
        case "training.jobs.sync": {
          const jobId = getString(request.params, "jobId");
          const job = await this.options.runtime.syncTrainingJob(jobId);
          return job ? ok(job) : notFound(`unknown training job: ${jobId}`);
        }
        case "training.jobs.plan": {
          const jobId = getString(request.params, "jobId");
          const plan = await this.options.runtime.getTrainingJobPlan(jobId);
          return plan ? ok(plan) : notFound(`unknown training job: ${jobId}`);
        }
        case "training.jobs.script": {
          const jobId = getString(request.params, "jobId");
          const script = await this.options.runtime.getTrainingJobLaunchScript(jobId);
          return script ? ok({ jobId, script }) : notFound(`unknown training job: ${jobId}`);
        }
        case "training.jobs.state": {
          const jobId = getString(request.params, "jobId");
          const state = await this.options.runtime.getTrainingJobExecutionState(jobId);
          return state ? ok(state) : notFound(`unknown training job: ${jobId}`);
        }
        case "training.jobs.log": {
          const jobId = getString(request.params, "jobId");
          const log = await this.options.runtime.getTrainingJobLog(jobId, getOptionalNumber(request.params, "lineLimit"));
          return log ? ok(log) : notFound(`unknown training job: ${jobId}`);
        }
        case "plugins.register":
          return ok(
            await this.options.runtime.registerPlugin({
              manifest: getPluginManifest(request.params, "manifest"),
              enabled: getOptionalBoolean(request.params, "enabled"),
            }),
          );
        case "plugins.get": {
          const pluginId = getString(request.params, "pluginId");
          const plugin = await this.options.runtime.getPlugin(pluginId);
          return plugin ? ok(plugin) : notFound(`unknown plugin: ${pluginId}`);
        }
        case "plugins.list":
          return ok(
            await this.options.runtime.listPlugins({
              capability: getOptionalPluginCapability(request.params, "capability"),
            }),
          );
        case "plugins.update": {
          const pluginId = getString(request.params, "pluginId");
          const plugin = await this.options.runtime.updatePluginEnabled(
            pluginId,
            getBoolean(request.params, "enabled"),
          );
          return plugin ? ok(plugin) : notFound(`unknown plugin: ${pluginId}`);
        }
        case "plugins.activate": {
          const pluginId = getString(request.params, "pluginId");
          await this.options.runtime.activatePlugin(pluginId);
          return ok({ pluginId, activated: true });
        }
        case "capture.consents.create":
          return ok(
            await this.options.runtime.createCaptureConsent({
              tier: getCaptureTier(request.params, "tier"),
              purpose: getString(request.params, "purpose"),
              expiresAt: getOptionalString(request.params, "expiresAt"),
            }),
          );
        case "capture.consents.list":
          return ok(await this.options.runtime.listActiveCaptureConsents());
        case "capture.consents.revoke": {
          const grantId = getString(request.params, "grantId");
          const grant = await this.options.runtime.revokeCaptureConsent(grantId);
          return grant ? ok(grant) : notFound(`unknown capture consent: ${grantId}`);
        }
        case "capture.browser.record":
          return ok(await this.options.runtime.recordBrowserCapture(getBrowserCaptureInput(request.params)));
        case "capture.device.record":
          return ok(await this.options.runtime.recordDeviceCapture(getDeviceCaptureInput(request.params)));
        case "capture.os.observe":
          return ok(await this.options.runtime.recordOsObservation(getOsObservationInput(request.params)));
        case "cron.list":
          return ok(await this.cron.listJobs());
        case "cron.create":
          return ok(
            await this.cron.createJob({
              cron: getString(request.params, "cron"),
              prompt: getString(request.params, "prompt"),
              recurring: getOptionalBoolean(request.params, "recurring") ?? true,
              durable: getOptionalBoolean(request.params, "durable") ?? false,
              delivery: getOptionalCronDeliveryConfig(request.params, "delivery"),
            }),
          );
        case "cron.delete": {
          const jobId = getString(request.params, "jobId");
          const deleted = await this.cron.deleteJob(jobId);
          return deleted ? ok({ deleted: true }) : notFound(`unknown cron job: ${jobId}`);
        }
        case "cron.tick": {
          const nowMs = getOptionalNumber(request.params, "nowMs");
          return ok(await this.cron.tick(typeof nowMs === "number" ? new Date(nowMs) : undefined));
        }
        case "cron.runs":
          return ok(await this.cron.listRuns(getOptionalString(request.params, "jobId")));
        default:
          return invalidRequest(`unknown method: ${request.method}`);
      }
    } catch (error) {
      if (error instanceof PairingTicketError) {
        return error.code === "NOT_FOUND"
          ? notFound(error.message)
          : invalidRequest(error.message);
      }
      return {
        ok: false,
        error: {
          code: "INTERNAL_ERROR",
          message: error instanceof Error ? error.message : String(error),
        },
      };
    }
  }
}

function buildPairingStoreFilePath(rootDir: string): string {
  return `${rootDir}/pairing-tickets.json`;
}

async function buildSessionBootstrapResult(
  runtime: StandaloneOperatorRuntime,
  session: SessionRecord,
  options: { created: boolean; resumed: boolean; runId?: string; family?: "run" | "approval" | "background-task" | "skill" },
): Promise<SessionBootstrapResult> {
  return {
    session,
    created: options.created,
    resumed: options.resumed,
    approvals: await runtime.listApprovals(session.id),
    events: runtime.events.snapshot(buildRuntimeEventFilter({
      sessionId: session.id,
      ...(options.runId ? { runId: options.runId } : {}),
      ...(options.family ? { family: options.family } : {}),
    })),
  };
}

function ok<T>(result: T): ControlPlaneSuccess<T> {
  return { ok: true, result };
}

function invalidRequest(message: string): ControlPlaneFailure {
  return { ok: false, error: { code: "INVALID_REQUEST", message } };
}

function notFound(message: string): ControlPlaneFailure {
  return { ok: false, error: { code: "NOT_FOUND", message } };
}

function getString(params: Record<string, unknown> | undefined, key: string): string {
  const value = params?.[key];
  if (typeof value !== "string" || !value.trim()) {
    throw new Error(`Invalid string parameter: ${key}`);
  }
  return value;
}

function getOptionalString(params: Record<string, unknown> | undefined, key: string): string | undefined {
  const value = params?.[key];
  if (value == null) {
    return undefined;
  }
  if (typeof value !== "string" || !value.trim()) {
    throw new Error(`Invalid string parameter: ${key}`);
  }
  return value;
}

function getApprovalDecision(
  params: Record<string, unknown> | undefined,
  key: string,
): ApprovalDecision {
  const value = params?.[key];
  if (value === "approved" || value === "denied" || value === "expired") {
    return value;
  }
  throw new Error(`Invalid approval decision: ${key}`);
}

function getRunStatus(
  params: Record<string, unknown> | undefined,
  key: string,
): "pending" | "running" | "paused" | "completed" | "failed" {
  const value = params?.[key];
  if (value === "pending" || value === "running" || value === "paused" || value === "completed" || value === "failed") {
    return value;
  }
  throw new Error(`Invalid run status: ${key}`);
}

function getSkillCandidateStatus(
  params: Record<string, unknown> | undefined,
  key: string,
): SkillCandidateStatus {
  const value = params?.[key];
  if (value === "candidate" || value === "reviewed" || value === "rejected") {
    return value;
  }
  throw new Error(`Invalid skill candidate status: ${key}`);
}

function getOptionalSkillCandidateStatus(
  params: Record<string, unknown> | undefined,
  key: string,
): SkillCandidateStatus | undefined {
  const value = params?.[key];
  if (value == null) {
    return undefined;
  }
  if (value === "candidate" || value === "reviewed" || value === "rejected") {
    return value;
  }
  throw new Error(`Invalid skill candidate status: ${key}`);
}

function getOptionalRuntimeEventFamily(
  params: Record<string, unknown> | undefined,
  key: string,
): "run" | "approval" | "background-task" | "skill" | undefined {
  const value = params?.[key];
  if (value == null) {
    return undefined;
  }
  if (value === "run" || value === "approval" || value === "background-task" || value === "skill") {
    return value;
  }
  throw new Error(`Invalid runtime event family: ${key}`);
}

function getPairingResolutionDecision(
  params: Record<string, unknown> | undefined,
  key: string,
): "approved" | "rejected" {
  const value = params?.[key];
  if (value === "approved" || value === "rejected") {
    return value;
  }
  throw new Error(`Invalid pairing resolution decision: ${key}`);
}

function getOptionalPairingTicketStatus(
  params: Record<string, unknown> | undefined,
  key: string,
): PairingTicketStatus | undefined {
  const value = params?.[key];
  if (value == null) {
    return undefined;
  }
  if (value === "pending" || value === "approved" || value === "rejected" || value === "redeemed" || value === "expired") {
    return value;
  }
  throw new Error(`Invalid pairing ticket status: ${key}`);
}

function getOptionalBoolean(
  params: Record<string, unknown> | undefined,
  key: string,
): boolean | undefined {
  const value = params?.[key];
  if (value == null) {
    return undefined;
  }
  if (typeof value !== "boolean") {
    throw new Error(`Invalid boolean parameter: ${key}`);
  }
  return value;
}

function getBoolean(
  params: Record<string, unknown> | undefined,
  key: string,
): boolean {
  const value = params?.[key];
  if (typeof value !== "boolean") {
    throw new Error(`Invalid boolean parameter: ${key}`);
  }
  return value;
}

function getOptionalNumber(
  params: Record<string, unknown> | undefined,
  key: string,
): number | undefined {
  const value = params?.[key];
  if (value == null) {
    return undefined;
  }
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new Error(`Invalid number parameter: ${key}`);
  }
  return value;
}

function getBackgroundTaskKind(
  params: Record<string, unknown> | undefined,
  key: string,
): BackgroundTaskKind {
  const value = params?.[key];
  if (value === "task" || value === "monitor") {
    return value;
  }
  throw new Error(`Invalid background task kind: ${key}`);
}

function getOptionalBackgroundTaskKind(
  params: Record<string, unknown> | undefined,
  key: string,
): BackgroundTaskKind | undefined {
  const value = params?.[key];
  if (value == null) {
    return undefined;
  }
  if (value === "task" || value === "monitor") {
    return value;
  }
  throw new Error(`Invalid background task kind: ${key}`);
}

function getOptionalBackgroundTaskRecoveryReasons(
  params: Record<string, unknown> | undefined,
  key: string,
): BackgroundTaskRecoveryReason[] | undefined {
  const value = params?.[key];
  if (value == null) {
    return undefined;
  }
  if (!Array.isArray(value)) {
    throw new Error(`Invalid background task recovery reasons: ${key}`);
  }
  return value.map((item) => {
    if (
      item === "unchanged" ||
      item === "state-running" ||
      item === "state-completed" ||
      item === "state-failed" ||
      item === "state-cancelled" ||
      item === "process-running" ||
      item === "missing-process" ||
      item === "missing-state"
    ) {
      return item;
    }
    throw new Error(`Invalid background task recovery reason: ${key}`);
  });
}

function getCaptureTier(
  params: Record<string, unknown> | undefined,
  key: string,
): CaptureTier {
  const value = params?.[key];
  if (value === "off" || value === "operator" || value === "app" || value === "full") {
    return value;
  }
  throw new Error(`Invalid capture tier: ${key}`);
}

function getTrajectoryReviewStatus(
  params: Record<string, unknown> | undefined,
  key: string,
): TrajectoryReviewStatus {
  const value = params?.[key];
  if (value === "pending" || value === "approved" || value === "rejected") {
    return value;
  }
  throw new Error(`Invalid trajectory review status: ${key}`);
}

function getOptionalTrajectoryReviewStatus(
  params: Record<string, unknown> | undefined,
  key: string,
): TrajectoryReviewStatus | undefined {
  const value = params?.[key];
  if (value == null) {
    return undefined;
  }
  if (value === "pending" || value === "approved" || value === "rejected") {
    return value;
  }
  throw new Error(`Invalid trajectory review status: ${key}`);
}

function getTrainingMode(
  params: Record<string, unknown> | undefined,
  key: string,
): TrainingMode {
  const value = params?.[key];
  if (value === "sft" || value === "rl") {
    return value;
  }
  throw new Error(`Invalid training mode: ${key}`);
}

function getOptionalTrainingModes(
  params: Record<string, unknown> | undefined,
  key: string,
): TrainingMode[] | undefined {
  const value = params?.[key];
  if (value == null) {
    return undefined;
  }
  if (!Array.isArray(value)) {
    throw new Error(`Invalid training modes: ${key}`);
  }
  return value.map((item) => {
    if (item === "sft" || item === "rl") {
      return item;
    }
    throw new Error(`Invalid training mode: ${key}`);
  });
}

function getTrainingJobStatus(
  params: Record<string, unknown> | undefined,
  key: string,
): LocalTrainingJobStatus {
  const value = params?.[key];
  if (value === "planned" || value === "ready" || value === "running" || value === "completed" || value === "failed") {
    return value;
  }
  throw new Error(`Invalid training job status: ${key}`);
}

function getReviewedExportManifest(
  params: Record<string, unknown> | undefined,
  key: string,
): ReviewedExportManifest {
  const value = params?.[key];
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new Error(`Invalid reviewed export manifest: ${key}`);
  }
  return value as ReviewedExportManifest;
}

function getPluginManifest(
  params: Record<string, unknown> | undefined,
  key: string,
): OperatorPluginManifest {
  const value = params?.[key];
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new Error(`Invalid plugin manifest: ${key}`);
  }
  return value as OperatorPluginManifest;
}

function getOptionalPluginCapability(
  params: Record<string, unknown> | undefined,
  key: string,
): OperatorPluginCapability | undefined {
  const value = params?.[key];
  if (value == null) {
    return undefined;
  }
  if (
    value === "tool" ||
    value === "memory-enricher" ||
    value === "capture-source" ||
    value === "skill-provider" ||
    value === "training-runtime"
  ) {
    return value;
  }
  throw new Error(`Invalid plugin capability: ${key}`);
}

function getOptionalCronDeliveryConfig(
  params: Record<string, unknown> | undefined,
  key: string,
): CronDeliveryConfig | undefined {
  const value = params?.[key];
  if (value == null) {
    return undefined;
  }
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new Error(`Invalid cron delivery config: ${key}`);
  }
  const config = value as Record<string, unknown>;
  return {
    ...(config.onSuccess ? { onSuccess: getDeliveryTargets(config.onSuccess, `${key}.onSuccess`) } : {}),
    ...(config.onFailure ? { onFailure: getDeliveryTargets(config.onFailure, `${key}.onFailure`) } : {}),
  };
}

function getDeliveryTargets(value: unknown, key: string): DeliveryTarget[] {
  if (!Array.isArray(value)) {
    throw new Error(`Invalid delivery targets: ${key}`);
  }
  return value.map((item, index) => getDeliveryTarget(item, `${key}[${index}]`));
}

function getDeliveryTarget(value: unknown, key: string): DeliveryTarget {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new Error(`Invalid delivery target: ${key}`);
  }
  const target = value as Record<string, unknown>;
  if (target.kind === "local") {
    return { kind: "local" };
  }
  if (target.kind === "webhook") {
    const url = typeof target.url === "string" && target.url.trim() ? target.url : undefined;
    if (!url) {
      throw new Error(`Invalid delivery target url: ${key}`);
    }
    return {
      kind: "webhook",
      url,
      ...(target.headers ? { headers: getOptionalStringRecord(target.headers, `${key}.headers`) } : {}),
    };
  }
  throw new Error(`Invalid delivery target kind: ${key}`);
}

function getOptionalStringRecord(value: unknown, key: string): Record<string, string> | undefined {
  if (value == null) {
    return undefined;
  }
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new Error(`Invalid string record: ${key}`);
  }
  return Object.fromEntries(
    Object.entries(value as Record<string, unknown>).map(([entryKey, entryValue]) => {
      if (typeof entryValue !== "string") {
        throw new Error(`Invalid string record value: ${key}.${entryKey}`);
      }
      return [entryKey, entryValue];
    }),
  );
}

function getOptionalRecord(
  params: Record<string, unknown> | undefined,
  key: string,
): Record<string, unknown> | undefined {
  const value = params?.[key];
  if (value == null) {
    return undefined;
  }
  if (typeof value !== "object" || Array.isArray(value)) {
    throw new Error(`Invalid object parameter: ${key}`);
  }
  return value as Record<string, unknown>;
}

function getExecutableSkillSteps(
  params: Record<string, unknown> | undefined,
  key: string,
): Array<{
  id: string;
  title: string;
  kind: "summary" | "x-post" | "command";
  content?: string;
  command?: string;
  agentRole?: string;
  maxCharacters?: number;
  overflowToComment?: boolean;
}> {
  const value = params?.[key];
  if (!Array.isArray(value)) {
    throw new Error(`Invalid executable skill steps: ${key}`);
  }
  return value.map((item, index) => {
    if (!item || typeof item !== "object" || Array.isArray(item)) {
      throw new Error(`Invalid executable skill step: ${key}[${index}]`);
    }
    const step = item as Record<string, unknown>;
    const kind = step.kind;
    if (kind !== "summary" && kind !== "x-post" && kind !== "command") {
      throw new Error(`Invalid executable skill step kind: ${key}[${index}]`);
    }
    return {
      id: typeof step.id === "string" ? step.id : `step-${index + 1}`,
      title: typeof step.title === "string" ? step.title : `Step ${index + 1}`,
      kind,
      content: typeof step.content === "string" ? step.content : undefined,
      command: typeof step.command === "string" ? step.command : undefined,
      agentRole: typeof step.agentRole === "string" ? step.agentRole : undefined,
      maxCharacters: typeof step.maxCharacters === "number" ? step.maxCharacters : undefined,
      overflowToComment: typeof step.overflowToComment === "boolean" ? step.overflowToComment : undefined,
    };
  });
}

function getTranscriptReadOptions(params: Record<string, unknown> | undefined): TranscriptReadOptions {
  const limit = params?.limit;
  const reverse = params?.reverse;
  const includeHeader = params?.includeHeader;
  return {
    ...(typeof limit === "number" ? { limit } : {}),
    ...(typeof reverse === "boolean" ? { reverse } : {}),
    ...(typeof includeHeader === "boolean" ? { includeHeader } : {}),
  };
}

function getBrowserCaptureInput(params: Record<string, unknown> | undefined): BrowserCaptureInput {
  return {
    sessionId: getString(params, "sessionId"),
    pageUrl: getString(params, "pageUrl"),
    pageTitle: getOptionalString(params, "pageTitle"),
    domSummary: getOptionalString(params, "domSummary"),
    activeElementLabel: getOptionalString(params, "activeElementLabel"),
    screenshotRef: getOptionalString(params, "screenshotRef"),
    appId: getOptionalString(params, "appId"),
    captureTier: getOptionalCaptureTier(params, "captureTier"),
    visibleIndicator: getBoolean(params, "visibleIndicator"),
    ts: getRequiredNumber(params, "ts"),
    action: getOptionalBrowserAction(params?.action),
  };
}

function getOptionalBrowserAction(value: unknown): BrowserCaptureInput["action"] | undefined {
  if (value == null) {
    return undefined;
  }
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new Error("Invalid browser action");
  }
  const action = value as Record<string, unknown>;
  const kind = action.kind;
  if (kind !== "navigate" && kind !== "click" && kind !== "type" && kind !== "submit") {
    throw new Error("Invalid browser action kind");
  }
  return {
    kind,
    target: typeof action.target === "string" ? action.target : undefined,
    valueSummary: typeof action.valueSummary === "string" ? action.valueSummary : undefined,
    ts: getRequiredNumber(action, "ts"),
  };
}

function getDeviceCaptureInput(params: Record<string, unknown> | undefined): DeviceCaptureInput {
  return {
    sessionId: getString(params, "sessionId"),
    deviceId: getString(params, "deviceId"),
    platform: getDevicePlatform(params, "platform"),
    appId: getString(params, "appId"),
    appName: getString(params, "appName"),
    screenTitle: getOptionalString(params, "screenTitle"),
    selectionSummary: getOptionalString(params, "selectionSummary"),
    captureTier: getOptionalCaptureTier(params, "captureTier"),
    visibleIndicator: getBoolean(params, "visibleIndicator"),
    ts: getRequiredNumber(params, "ts"),
    gesture: getOptionalDeviceGesture(params?.gesture),
  };
}

function getOptionalDeviceGesture(value: unknown): DeviceCaptureInput["gesture"] | undefined {
  if (value == null) {
    return undefined;
  }
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new Error("Invalid device gesture");
  }
  const gesture = value as Record<string, unknown>;
  const kind = gesture.kind;
  if (kind !== "tap" && kind !== "swipe" && kind !== "scroll" && kind !== "type" && kind !== "shortcut") {
    throw new Error("Invalid device gesture kind");
  }
  const direction = gesture.direction;
  if (direction != null && direction !== "up" && direction !== "down" && direction !== "left" && direction !== "right") {
    throw new Error("Invalid device gesture direction");
  }
  return {
    kind,
    target: typeof gesture.target === "string" ? gesture.target : undefined,
    direction: typeof direction === "string" ? direction : undefined,
    valueSummary: typeof gesture.valueSummary === "string" ? gesture.valueSummary : undefined,
    ts: getRequiredNumber(gesture, "ts"),
  };
}

function getDevicePlatform(
  params: Record<string, unknown> | undefined,
  key: string,
): DeviceCaptureInput["platform"] {
  const value = params?.[key];
  if (value === "ios" || value === "android" || value === "macos" || value === "windows" || value === "linux") {
    return value;
  }
  throw new Error(`Invalid device platform: ${key}`);
}

function getOsObservationInput(params: Record<string, unknown> | undefined): OsObservationInput {
  return {
    sessionId: getString(params, "sessionId"),
    appId: getString(params, "appId"),
    visibleIndicator: getBoolean(params, "visibleIndicator"),
    captureTier: getOptionalCaptureTier(params, "captureTier"),
    ts: getRequiredNumber(params, "ts"),
    event: getOsEvent(params, "event"),
    windowTitle: getOptionalString(params, "windowTitle"),
    filePath: getOptionalString(params, "filePath"),
    commandSummary: getOptionalString(params, "commandSummary"),
  };
}

function getOsEvent(
  params: Record<string, unknown> | undefined,
  key: string,
): OsObservationInput["event"] {
  const value = params?.[key];
  if (value === "focus-changed" || value === "window-opened" || value === "file-opened" || value === "command-ran") {
    return value;
  }
  throw new Error(`Invalid OS event: ${key}`);
}

function getOptionalCaptureTier(
  params: Record<string, unknown> | undefined,
  key: string,
): CaptureTier | undefined {
  const value = params?.[key];
  if (value == null) {
    return undefined;
  }
  if (value === "off" || value === "operator" || value === "app" || value === "full") {
    return value;
  }
  throw new Error(`Invalid capture tier: ${key}`);
}

function getOptionalReviewedObservations(
  params: Record<string, unknown> | undefined,
  key: string,
): ReviewedTrajectoryObservation[] | undefined {
  const value = params?.[key];
  if (value == null) {
    return undefined;
  }
  if (!Array.isArray(value)) {
    throw new Error(`Invalid reviewed observations: ${key}`);
  }
  return value.map((item) => {
    if (!item || typeof item !== "object" || Array.isArray(item)) {
      throw new Error(`Invalid reviewed observations: ${key}`);
    }
    const observation = item as Record<string, unknown>;
    return {
      ts: getRequiredNumber(observation, "ts"),
      source: getString(observation, "source"),
      summary: getString(observation, "summary"),
    };
  });
}

function getOptionalReviewedActions(
  params: Record<string, unknown> | undefined,
  key: string,
): ReviewedTrajectoryAction[] | undefined {
  const value = params?.[key];
  if (value == null) {
    return undefined;
  }
  if (!Array.isArray(value)) {
    throw new Error(`Invalid reviewed actions: ${key}`);
  }
  return value.map((item) => {
    if (!item || typeof item !== "object" || Array.isArray(item)) {
      throw new Error(`Invalid reviewed actions: ${key}`);
    }
    const action = item as Record<string, unknown>;
    return {
      ts: getRequiredNumber(action, "ts"),
      tool: getString(action, "tool"),
      summary: getString(action, "summary"),
    };
  });
}

function getOptionalReviewedTranscript(
  params: Record<string, unknown> | undefined,
  key: string,
): ReviewedTranscriptMessage[] | undefined {
  const value = params?.[key];
  if (value == null) {
    return undefined;
  }
  if (!Array.isArray(value)) {
    throw new Error(`Invalid reviewed transcript: ${key}`);
  }
  return value.map((item) => {
    if (!item || typeof item !== "object" || Array.isArray(item)) {
      throw new Error(`Invalid reviewed transcript: ${key}`);
    }
    const message = item as Record<string, unknown>;
    return {
      ts: getRequiredNumber(message, "ts"),
      role: getTranscriptRole(message, "role"),
      content: getString(message, "content"),
    };
  });
}

function getTranscriptRole(
  params: Record<string, unknown> | undefined,
  key: string,
): ReviewedTranscriptMessage["role"] {
  const value = params?.[key];
  if (value === "system" || value === "user" || value === "assistant" || value === "tool") {
    return value;
  }
  throw new Error(`Invalid transcript role: ${key}`);
}

function getRequiredNumber(params: Record<string, unknown> | undefined, key: string): number {
  const value = params?.[key];
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new Error(`Invalid number parameter: ${key}`);
  }
  return value;
}
