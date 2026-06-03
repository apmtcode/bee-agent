import readline from "node:readline/promises";
import process from "node:process";
import { subscribeRuntimeEvents } from "../control-plane/subscriptions.js";
import { OperatorControlPlaneServer } from "../control-plane/server.js";
import { StandaloneOperatorRuntime } from "../orchestrator/operator-runtime.js";
import type { TranscriptRecord } from "../harness/transcript-store.js";
import {
  OperatorCliConfigLoader,
  resolveOperatorCliExecutionConfig,
  type OperatorCliRuntimeConfig,
} from "./config.js";
import { runOperatorCliConversationTurn } from "./conversation.js";
import { discoverPromptContext, type OperatorCliPromptContext } from "./prompt.js";

export type OperatorCliSlashCommand =
  | { kind: "help" }
  | { kind: "status" }
  | { kind: "sessions" }
  | { kind: "resume"; sessionId: string }
  | { kind: "idle" }
  | { kind: "complete" }
  | { kind: "transcript"; limit?: number }
  | { kind: "approvals" }
  | { kind: "approve"; approvalId: string }
  | { kind: "deny"; approvalId: string }
  | { kind: "pairing-list"; status?: "pending" | "approved" | "rejected" | "redeemed" | "expired" }
  | { kind: "pairing-create"; remoteSource?: string }
  | { kind: "pairing-approve"; identifier: string }
  | { kind: "pairing-reject"; identifier: string }
  | { kind: "remote-list"; remoteSource?: string }
  | { kind: "remote-status"; identifier: string }
  | { kind: "remote-pause"; identifier: string; reason?: string }
  | { kind: "remote-resume"; identifier: string }
  | { kind: "remote-repair"; identifier: string; reasons?: Array<"missing-process" | "missing-state"> }
  | { kind: "platform-list" }
  | { kind: "platform-status"; platform: string }
  | { kind: "platform-pause"; platform: string; reason?: string }
  | { kind: "platform-resume"; platform: string }
  | { kind: "recall"; query: string }
  | { kind: "tasks" }
  | { kind: "task-create"; subject: string }
  | { kind: "task-update"; taskId: string; status: "pending" | "in_progress" | "completed" }
  | { kind: "messages" }
  | { kind: "inbox" }
  | { kind: "outbox" }
  | { kind: "send"; toSessionId: string; message: string }
  | { kind: "notify"; status: string; message: string }
  | { kind: "skills" }
  | { kind: "run-skill"; skillId: string }
  | { kind: "watch-run"; runId?: string }
  | { kind: "watch-task"; taskId?: string }
  | { kind: "watch-active" }
  | { kind: "background-list" }
  | { kind: "background-start"; title: string; command: string }
  | { kind: "background-view"; taskId: string; lines?: number }
  | { kind: "background-sync"; taskId: string }
  | { kind: "background-cancel"; taskId: string }
  | { kind: "background" }
  | { kind: "training" }
  | { kind: "cron-list" }
  | { kind: "cron-create"; cron: string; prompt: string; delivery?: { onSuccess?: string[]; onFailure?: string[] } }
  | { kind: "cron-runs"; jobId?: string }
  | { kind: "cron-tick" }
  | { kind: "cron-delete"; jobId: string }
  | { kind: "config" }
  | { kind: "prompt" }
  | { kind: "invalid"; message: string }
  | { kind: "unknown"; name: string };

export type OperatorCliAppOptions = {
  rootDir: string;
  cwd?: string;
  currentDate?: string;
  stdin?: NodeJS.ReadableStream;
  stdout?: NodeJS.WritableStream;
  stderr?: NodeJS.WritableStream;
};

export class OperatorCliApp {
  readonly runtime: StandaloneOperatorRuntime;
  readonly server: OperatorControlPlaneServer;
  readonly cwd: string;
  readonly currentDate: string;
  private readonly stdin: NodeJS.ReadableStream;
  private readonly stdout: NodeJS.WritableStream;
  private readonly stderr: NodeJS.WritableStream;
  private activeSessionId?: string;
  private readonly approvedCommandFingerprints = new Set<string>();

  constructor(private readonly options: OperatorCliAppOptions) {
    this.runtime = new StandaloneOperatorRuntime({ rootDir: options.rootDir });
    this.server = new OperatorControlPlaneServer({ runtime: this.runtime });
    this.cwd = options.cwd ?? process.cwd();
    this.currentDate = options.currentDate ?? new Date().toISOString().slice(0, 10);
    this.stdin = options.stdin ?? process.stdin;
    this.stdout = options.stdout ?? process.stdout;
    this.stderr = options.stderr ?? process.stderr;
  }

  async loadRuntimeConfig(): Promise<OperatorCliRuntimeConfig> {
    const config = await new OperatorCliConfigLoader(this.cwd).load();
    this.runtime.setExecutionConfig(resolveOperatorCliExecutionConfig(config));
    return config;
  }

  async loadPromptContext(config?: OperatorCliRuntimeConfig): Promise<OperatorCliPromptContext> {
    return await discoverPromptContext({ cwd: this.cwd, currentDate: this.currentDate, config });
  }

  async runRepl(): Promise<void> {
    const config = await this.loadRuntimeConfig();
    const promptContext = await this.loadPromptContext(config);
    const session = await this.runtime.startSession({ title: "Operator CLI", cwd: this.cwd, agentId: "operator-cli" });
    this.activeSessionId = session.id;
    this.stdout.write("Standalone Operator CLI\n");
    this.stdout.write(`Session: ${session.id}\n`);
    this.stdout.write(`Loaded config files: ${config.loadedEntries.length}\n`);
    this.stdout.write(`Loaded instruction files: ${promptContext.project.instructionFiles.length}\n`);
    this.stdout.write("Type /help for commands. Ctrl+D exits.\n");

    const reader = readline.createInterface({ input: this.stdin, output: this.stdout });
    try {
      while (true) {
        const line = await reader.question("operator> ");
        if (!line.trim()) {
          continue;
        }
        this.stdout.write(`${await this.handleInput(line, undefined, config, promptContext)}\n`);
      }
    } catch {
      this.stdout.write("\n");
    } finally {
      reader.close();
    }
  }

  async handleInput(
    input: string,
    sessionId?: string,
    config?: OperatorCliRuntimeConfig,
    promptContext?: OperatorCliPromptContext,
  ): Promise<string> {
    const command = parseSlashCommand(input);
    if (!command) {
      const activeSessionId = this.requireActiveSessionId(sessionId);
      const effectiveConfig = config ?? (await this.loadRuntimeConfig());
      let progressOutput = "";
      const streamPromise = this.collectRunProgress(activeSessionId, async () => {
        const result = await runOperatorCliConversationTurn({
          runtime: this.runtime,
          sessionId: activeSessionId,
          cwd: this.cwd,
          currentDate: this.currentDate,
          input,
          config: effectiveConfig,
          approvedCommandFingerprints: this.approvedCommandFingerprints,
        });
        await this.runtime.recordTurn({
          sessionId: activeSessionId,
          userText: input,
          assistantText: result.assistantText,
          trajectorySummary: result.trajectorySummary,
        });
        return result;
      });
      const { result, progressLines } = await streamPromise;
      progressOutput = progressLines.join("\n");
      return progressOutput ? `${progressOutput}\n${result.assistantText}` : result.assistantText;
    }
    return await this.dispatchSlashCommand(command, sessionId, config, promptContext);
  }

  async dispatchSlashCommand(
    command: OperatorCliSlashCommand,
    sessionId?: string,
    config?: OperatorCliRuntimeConfig,
    promptContext?: OperatorCliPromptContext,
  ): Promise<string> {
    if (command.kind === "invalid") {
      return command.message;
    }

    if (sessionId && !this.activeSessionId) {
      this.activeSessionId = sessionId;
    }

    switch (command.kind) {
      case "help":
        return [
          "Available commands:",
          "  /help",
          "  /status",
          "  /sessions",
          "  /resume <sessionId>",
          "  /idle",
          "  /complete",
          "  /transcript [limit]",
          "  /approvals",
          "  /approve <approvalId>",
          "  /deny <approvalId>",
          "  /pairing [status]",
          "  /pairing create [source]",
          "  /pairing approve <code|id>",
          "  /pairing reject <code|id>",
          "  /remote list [source]",
          "  /remote status <remoteId|sessionId>",
          "  /remote pause <remoteId|sessionId> [reason]",
          "  /remote resume <remoteId|sessionId>",
          "  /remote repair <remoteId|sessionId> [missing-process|missing-state]",
          "  /platform list",
          "  /platform status <platform>",
          "  /platform pause <platform> [reason]",
          "  /platform resume <platform>",
          "  /recall <query>",
          "  /tasks",
          "  /task-create <subject>",
          "  /task-update <taskId> <pending|in_progress|completed>",
          "  /messages",
          "  /inbox",
          "  /outbox",
          "  /send <sessionId> <message>",
          "  /notify <status> <message>",
          "  /skills",
          "  /run-skill <id>",
          "  /watch run [runId]",
          "  /watch task [taskId]",
          "  /watch active",
          "  /background",
          "  /background start <title> -- <command>",
          "  /background view <taskId> [lines]",
          "  /background sync <taskId>",
          "  /background cancel <taskId>",
          "  /training",
          "  /cron",
          "  /cron create <cronExpr> <prompt>",
          "  /cron runs [jobId]",
          "  /cron tick",
          "  /cron delete <jobId>",
          "  /config",
          "  /prompt",
        ].join("\n");
      case "status": {
        const activeSessionId = this.requireActiveSessionId(sessionId);
        const session = await this.runtime.getSession(activeSessionId);
        const latestAssistant = await this.runtime.getLatestAssistantText(activeSessionId);
        const effectiveConfig = config ?? (await this.loadRuntimeConfig());
        const effectivePromptContext = promptContext ?? (await this.loadPromptContext(effectiveConfig));
        return [
          `session=${session?.id ?? activeSessionId}`,
          `status=${session?.status ?? "unknown"}`,
          `cwd=${this.cwd}`,
          `configFiles=${effectiveConfig.loadedEntries.length}`,
          `instructionFiles=${effectivePromptContext.project.instructionFiles.length}`,
          `latestAssistant=${latestAssistant ?? "<none>"}`,
        ].join(" ");
      }
      case "sessions": {
        const activeSessionId = this.activeSessionId;
        const sessions = await this.runtime.listSessions();
        if (sessions.length === 0) {
          return "No sessions.";
        }
        return sessions
          .map((item) => {
            const prefix = item.id === activeSessionId ? "* " : "  ";
            const title = item.metadata.title ? ` title=${item.metadata.title}` : "";
            const cwdValue = item.metadata.cwd ? ` cwd=${item.metadata.cwd}` : "";
            return `${prefix}${item.id} ${item.status}${title}${cwdValue}`;
          })
          .join("\n");
      }
      case "resume": {
        const resumed = await this.runtime.resumeSession(command.sessionId);
        if (!resumed) {
          return `Unknown session: ${command.sessionId}`;
        }
        this.activeSessionId = resumed.id;
        return `Resumed session ${resumed.id}.`;
      }
      case "idle": {
        const activeSessionId = this.requireActiveSessionId(sessionId);
        const idle = await this.runtime.markSessionIdle(activeSessionId);
        return idle ? `Marked session ${idle.id} idle.` : `Unknown session: ${activeSessionId}`;
      }
      case "complete": {
        const activeSessionId = this.requireActiveSessionId(sessionId);
        const completed = await this.runtime.completeSession(activeSessionId);
        return completed ? `Completed session ${completed.id}.` : `Unknown session: ${activeSessionId}`;
      }
      case "transcript": {
        const activeSessionId = this.requireActiveSessionId(sessionId);
        const transcript = await this.runtime.getTranscript(activeSessionId);
        const messages = transcript.filter(isTranscriptMessageRecord);
        const limit = command.limit ?? 20;
        const visible = messages.slice(-limit);
        if (visible.length === 0) {
          return `No transcript messages for session ${activeSessionId}.`;
        }
        return visible
          .map((record) => `${record.message.role}: ${record.message.content}`)
          .join("\n");
      }
      case "approvals": {
        const activeSessionId = this.requireActiveSessionId(sessionId);
        const approvals = await this.runtime.listApprovals();
        if (approvals.length === 0) {
          return "No pending approvals.";
        }
        return approvals
          .map((approval) => {
            const activeMarker = approval.sessionId === activeSessionId ? "* " : "  ";
            const scope = approval.scope ? ` scope=${approval.scope}` : "";
            const sessionLabel = approval.sessionId ? ` session=${approval.sessionId}` : "";
            return `${activeMarker}${approval.id} ${approval.title} (${approval.summary})${sessionLabel}${scope}`;
          })
          .join("\n");
      }
      case "approve": {
        const approvals = await this.runtime.listApprovals();
        const approval = approvals.find((item) => item.id === command.approvalId);
        const resolution = await this.runtime.resolveApproval(command.approvalId, "approved", "operator-cli");
        if (!resolution) {
          return `Unknown approval: ${command.approvalId}`;
        }
        const fingerprint = typeof approval?.metadata?.fingerprint === "string" ? approval.metadata.fingerprint : undefined;
        if (fingerprint) {
          this.approvedCommandFingerprints.add(fingerprint);
        }
        return `Approved ${command.approvalId}.`;
      }
      case "deny": {
        const resolution = await this.runtime.resolveApproval(command.approvalId, "denied", "operator-cli");
        return resolution ? `Denied ${command.approvalId}.` : `Unknown approval: ${command.approvalId}`;
      }
      case "pairing-list": {
        const response = await this.server.handle({
          method: "pairing.list",
          ...(command.status ? { params: { status: command.status } } : {}),
        });
        if (!response.ok) {
          return response.error.message;
        }
        return response.result.length === 0
          ? "No pairing tickets."
          : response.result
              .map((ticket) => {
                const resolved = ticket.resolvedAt ? ` resolved=${ticket.resolvedAt}` : "";
                const resolvedBy = ticket.resolvedBy ? ` by=${ticket.resolvedBy}` : "";
                const redeemed = ticket.redeemedSessionId ? ` session=${ticket.redeemedSessionId}` : "";
                return `${ticket.code} ${ticket.status} source=${ticket.remoteSource} id=${ticket.id} expires=${ticket.expiresAt}${resolved}${resolvedBy}${redeemed}`;
              })
              .join("\n");
      }
      case "pairing-create": {
        const response = await this.server.handle({
          method: "pairing.create",
          params: command.remoteSource ? { remoteSource: command.remoteSource } : {},
        });
        if (!response.ok) {
          return response.error.message;
        }
        return `Created pairing ticket ${response.result.code} source=${response.result.remoteSource} expires=${response.result.expiresAt} id=${response.result.id}`;
      }
      case "pairing-approve": {
        const response = await this.server.handle({
          method: "pairing.resolve",
          params: { code: command.identifier, decision: "approved", resolvedBy: "operator-cli" },
        });
        return response.ok ? `Approved pairing ticket ${response.result.code}.` : response.error.message;
      }
      case "pairing-reject": {
        const response = await this.server.handle({
          method: "pairing.resolve",
          params: { code: command.identifier, decision: "rejected", resolvedBy: "operator-cli" },
        });
        return response.ok ? `Rejected pairing ticket ${response.result.code}.` : response.error.message;
      }
      case "remote-list": {
        const response = await this.server.handle({
          method: "sessions.remoteInventory",
          params: command.remoteSource ? { remoteSource: command.remoteSource } : {},
        });
        if (!response.ok) {
          return response.error.message;
        }
        return formatRemoteInventory(response.result.items);
      }
      case "remote-status": {
        const response = await this.server.handle({
          method: "sessions.remoteStatus",
          params: { identifier: command.identifier },
        });
        if (!response.ok) {
          return response.error.message;
        }
        return formatRemoteStatus(response.result);
      }
      case "remote-pause": {
        const response = await this.server.handle({
          method: "sessions.remoteControl",
          params: {
            identifier: command.identifier,
            action: "pause",
            ...(command.reason ? { reason: command.reason } : {}),
          },
        });
        if (!response.ok) {
          return response.error.message;
        }
        return `Paused remote ${response.result.remoteId} control=${response.result.control.state}${response.result.control.reason ? ` reason=${response.result.control.reason}` : ""}`;
      }
      case "remote-resume": {
        const response = await this.server.handle({
          method: "sessions.remoteControl",
          params: {
            identifier: command.identifier,
            action: "resume",
          },
        });
        if (!response.ok) {
          return response.error.message;
        }
        return `Resumed remote ${response.result.remoteId} control=${response.result.control.state}${response.result.control.reason ? ` reason=${response.result.control.reason}` : ""}`;
      }
      case "remote-repair": {
        const response = await this.server.handle({
          method: "sessions.remoteRepair",
          params: {
            identifier: command.identifier,
            action: "recover-background-tasks",
            ...(command.reasons?.length ? { reasons: command.reasons } : {}),
          },
        });
        if (!response.ok) {
          return response.error.message;
        }
        const repairedTasks = response.result.repaired.results.length === 0
          ? "<none>"
          : response.result.repaired.results.map((item) => `${item.task.id}:${item.reason}:${item.task.status}${item.changed ? ":changed" : ""}`).join(",");
        return `Repaired remote ${response.result.remoteId} changed=${response.result.repaired.changed} control=${response.result.control.state}${response.result.control.reason ? ` reason=${response.result.control.reason}` : ""} tasks=${repairedTasks}`;
      }
      case "platform-list": {
        const response = await this.server.handle({ method: "sessions.platformInventory" });
        if (!response.ok) {
          return response.error.message;
        }
        return formatPlatformInventory(response.result.items);
      }
      case "platform-status": {
        const response = await this.server.handle({
          method: "sessions.platformStatus",
          params: { platform: command.platform },
        });
        if (!response.ok) {
          return response.error.message;
        }
        return formatPlatformStatus(response.result);
      }
      case "platform-pause": {
        const response = await this.server.handle({
          method: "sessions.platformControl",
          params: {
            platform: command.platform,
            action: "pause",
            ...(command.reason ? { reason: command.reason } : {}),
          },
        });
        if (!response.ok) {
          return response.error.message;
        }
        return formatPlatformControlResult(response.result, "Paused");
      }
      case "platform-resume": {
        const response = await this.server.handle({
          method: "sessions.platformControl",
          params: {
            platform: command.platform,
            action: "resume",
          },
        });
        if (!response.ok) {
          return response.error.message;
        }
        return formatPlatformControlResult(response.result, "Resumed");
      }
      case "recall": {
        const recall = await this.runtime.recall(command.query);
        if (recall.hits.length === 0) {
          return `No recall hits for \"${command.query}\".`;
        }
        return recall.hits
          .slice(0, 5)
          .map((hit) => {
            if (hit.kind === "memory") {
              return `memory ${hit.item.id}: ${hit.item.summary}`;
            }
            if (hit.kind === "skill") {
              return `skill ${hit.skill.id}: ${hit.skill.title}`;
            }
            return `trajectory ${hit.trajectoryId}: ${hit.summary}`;
          })
          .join("\n");
      }
      case "tasks": {
        const activeSessionId = this.requireActiveSessionId(sessionId);
        const tasks = await this.runtime.listTasks(activeSessionId);
        return formatOperatorTasks(tasks);
      }
      case "task-create": {
        const activeSessionId = this.requireActiveSessionId(sessionId);
        const task = await this.runtime.createTask({
          sessionId: activeSessionId,
          subject: command.subject,
          description: command.subject,
        });
        return `Created task ${task.id} ${task.status} ${task.subject}`;
      }
      case "task-update": {
        const task = await this.runtime.updateTask(command.taskId, { status: command.status });
        return task ? `Updated task ${task.id} ${task.status} ${task.subject}` : `Unknown task: ${command.taskId}`;
      }
      case "messages": {
        const activeSessionId = this.requireActiveSessionId(sessionId);
        const messages = await this.runtime.listMessages(activeSessionId);
        return formatOperatorMessages(messages);
      }
      case "inbox": {
        const activeSessionId = this.requireActiveSessionId(sessionId);
        const messages = await this.runtime.listInbox(activeSessionId);
        return formatOperatorMessages(messages);
      }
      case "outbox": {
        const activeSessionId = this.requireActiveSessionId(sessionId);
        const messages = await this.runtime.listOutbox(activeSessionId);
        return formatOperatorMessages(messages);
      }
      case "send": {
        const activeSessionId = this.requireActiveSessionId(sessionId);
        const message = await this.runtime.sendMessage({
          fromSessionId: activeSessionId,
          toSessionId: command.toSessionId,
          message: command.message,
        });
        return `Sent message ${message.id} to=${message.toSessionId} summary=${message.summary}`;
      }
      case "notify": {
        const activeSessionId = this.requireActiveSessionId(sessionId);
        const notification = await this.runtime.sendNotification({
          sessionId: activeSessionId,
          status: command.status,
          message: command.message,
        });
        const delivered = notification.deliveryResults.length === 0
          ? "suppressed"
          : notification.deliveryResults.map((result) => result.status).join(",");
        return `Sent notification ${notification.id} status=${notification.status} delivery=${delivered} message=${notification.message}`;
      }
      case "skills": {
        const promoted = await this.runtime.listPromotedSkills();
        const executable = await this.runtime.listExecutableSkills();
        if (promoted.length === 0 && executable.length === 0) {
          return "No promoted or executable skills.";
        }
        return [
          ...promoted.map((skill) => `promoted ${skill.id}: ${skill.title}`),
          ...executable.map((skill) => `executable ${skill.id}: ${skill.title}`),
        ].join("\n");
      }
      case "run-skill": {
        const activeSessionId = this.requireActiveSessionId(sessionId);
        const effectiveConfig = config ?? (await this.loadRuntimeConfig());
        const run = await this.runtime.runExecutableSkill({
          skillId: command.skillId,
          sessionId: activeSessionId,
          executionConfig: resolveOperatorCliExecutionConfig(effectiveConfig),
          approvedCommandFingerprints: this.approvedCommandFingerprints,
        });
        if (!run) {
          return `Unknown executable skill: ${command.skillId}`;
        }
        return [`run ${run.id} completed`, ...run.stepResults.map((step) => `- ${step.stepId}: ${step.output}`)].join("\n");
      }
      case "watch-run": {
        const activeSessionId = this.requireActiveSessionId(sessionId);
        return await this.watchRun(activeSessionId, command.runId);
      }
      case "watch-task": {
        const activeSessionId = this.requireActiveSessionId(sessionId);
        return await this.watchTask(activeSessionId, command.taskId);
      }
      case "watch-active": {
        const activeSessionId = this.requireActiveSessionId(sessionId);
        return await this.watchActive(activeSessionId);
      }
      case "background":
      case "background-list": {
        const activeSessionId = this.requireActiveSessionId(sessionId);
        const tasks = await this.runtime.listBackgroundTasks(activeSessionId);
        return tasks.length === 0
          ? "No background tasks."
          : tasks.map((task) => `${task.id} ${task.status} ${task.title}`).join("\n");
      }
      case "background-start": {
        const activeSessionId = this.requireActiveSessionId(sessionId);
        const effectiveConfig = config ?? (await this.loadRuntimeConfig());
        const executionConfig = resolveOperatorCliExecutionConfig(effectiveConfig);
        const authorization = await this.runtime.authorizeCommandExecution({
          sessionId: activeSessionId,
          cwd: this.cwd,
          command: command.command,
          title: command.title,
          kind: "background.start",
          source: "cli",
          executionConfig,
          approvedFingerprints: this.approvedCommandFingerprints,
        });
        if (!authorization.allowed) {
          return authorization.message;
        }
        const task = await this.runtime.startBackgroundTask({
          sessionId: activeSessionId,
          title: authorization.title ?? command.title,
          command: authorization.command,
          cwd: this.cwd,
        });
        const postHookResults = await this.runtime.runPostCommandHooks({
          sessionId: activeSessionId,
          cwd: this.cwd,
          command: authorization.command,
          ...(authorization.title ? { title: authorization.title } : {}),
          kind: "background.start",
          source: "cli",
          executionConfig,
        });
        return [
          `Started background task ${task.id} (${task.status}) ${task.title}`,
          ...postHookResults.map((result) => formatHookResult(result)),
        ]
          .filter(Boolean)
          .join("\n");
      }
      case "background-view": {
        const task = await this.runtime.syncBackgroundTask(command.taskId);
        if (!task) {
          return `Unknown background task: ${command.taskId}`;
        }
        const output = await this.runtime.getBackgroundTaskOutput(command.taskId, command.lines ?? 50);
        return [
          `${task.id} ${task.status} ${task.title}`,
          output?.output?.trim() ? output.output : "<no output>",
        ].join("\n");
      }
      case "background-sync": {
        const task = await this.runtime.syncBackgroundTask(command.taskId);
        return task ? `${task.id} ${task.status} ${task.title}` : `Unknown background task: ${command.taskId}`;
      }
      case "background-cancel": {
        const task = await this.runtime.cancelBackgroundTask(command.taskId);
        return task ? `Cancelled background task ${task.id}.` : `Unknown background task: ${command.taskId}`;
      }
      case "training": {
        const jobs = await this.runtime.listTrainingJobs();
        return jobs.length === 0
          ? "No training jobs."
          : jobs.map((job) => `${job.id} ${job.status} ${job.mode}`).join("\n");
      }
      case "cron-list": {
        const response = await this.server.handle({ method: "cron.list" });
        if (!response.ok) {
          return response.error.message;
        }
        return response.result.length === 0
          ? "No cron jobs."
          : response.result
              .map((job) => {
                const deliveryTargets = [
                  ...(job.delivery?.onSuccess?.map((target) => `success:${target.kind === "local" ? "local" : target.url}`) ?? []),
                  ...(job.delivery?.onFailure?.map((target) => `failure:${target.kind === "local" ? "local" : target.url}`) ?? []),
                ];
                return `${job.id} ${job.cron} recurring=${job.recurring} next=${job.nextRunAt ?? "<none>"} last=${job.lastRunAt ?? "<none>"} status=${job.lastStatus ?? "<none>"} prompt=${job.prompt}${job.lastError ? ` error=${job.lastError}` : ""}${job.lastDeliveryStatus ? ` delivery=${job.lastDeliveryStatus}` : ""}${job.lastDeliveryError ? ` deliveryError=${job.lastDeliveryError}` : ""}${deliveryTargets.length > 0 ? ` targets=${deliveryTargets.join(",")}` : ""}`;
              })
              .join("\n");
      }
      case "cron-create": {
        const response = await this.server.handle({
          method: "cron.create",
          params: {
            cron: command.cron,
            prompt: command.prompt,
            ...(command.delivery
              ? {
                  delivery: {
                    ...(command.delivery.onSuccess?.length
                      ? { onSuccess: command.delivery.onSuccess.map((target) => (target === "local" ? { kind: "local" } : { kind: "webhook", url: target })) }
                      : {}),
                    ...(command.delivery.onFailure?.length
                      ? { onFailure: command.delivery.onFailure.map((target) => (target === "local" ? { kind: "local" } : { kind: "webhook", url: target })) }
                      : {}),
                  },
                }
              : {}),
          },
        });
        if (!response.ok) {
          return response.error.message;
        }
        return `Created cron job ${response.result.id} next=${response.result.nextRunAt ?? "<none>"}`;
      }
      case "cron-runs": {
        const response = await this.server.handle({
          method: "cron.runs",
          params: command.jobId ? { jobId: command.jobId } : {},
        });
        if (!response.ok) {
          return response.error.message;
        }
        return response.result.length === 0
          ? "No cron runs."
          : response.result
              .map((run) => {
                const delivery = run.deliveryResults?.map((result) => `${result.target.kind === "local" ? "local" : result.target.url}:${result.status}${result.error ? `:${result.error}` : ""}`).join(",");
                return `${run.id} job=${run.jobId} ${run.status}${run.outcome ? ` outcome=${run.outcome}` : ""}${run.sessionId ? ` session=${run.sessionId}` : ""}${run.operatorRunId ? ` run=${run.operatorRunId}` : ""}${run.summary ? ` summary=${run.summary}` : ""}${run.error ? ` error=${run.error}` : ""}${delivery ? ` delivery=${delivery}` : ""}`;
              })
              .join("\n");
      }
      case "cron-tick": {
        const response = await this.server.handle({ method: "cron.tick" });
        if (!response.ok) {
          return response.error.message;
        }
        return response.result.length === 0
          ? "No due cron jobs."
          : response.result
              .map((run) => {
                const delivery = run.deliveryResults?.map((result) => `${result.target.kind === "local" ? "local" : result.target.url}:${result.status}${result.error ? `:${result.error}` : ""}`).join(",");
                return `${run.id} job=${run.jobId} ${run.status}${run.outcome ? ` outcome=${run.outcome}` : ""}${run.sessionId ? ` session=${run.sessionId}` : ""}${run.operatorRunId ? ` run=${run.operatorRunId}` : ""}${run.summary ? ` summary=${run.summary}` : ""}${run.error ? ` error=${run.error}` : ""}${delivery ? ` delivery=${delivery}` : ""}`;
              })
              .join("\n");
      }
      case "cron-delete": {
        const response = await this.server.handle({ method: "cron.delete", params: { jobId: command.jobId } });
        return response.ok ? `Deleted cron job ${command.jobId}.` : response.error.message;
      }
      case "config": {
        const effectiveConfig = config ?? (await this.loadRuntimeConfig());
        const executionConfig = resolveOperatorCliExecutionConfig(effectiveConfig);
        return [
          ...(effectiveConfig.loadedEntries.length === 0
            ? ["Loaded config files: <none>"]
            : ["Loaded config files:", ...effectiveConfig.loadedEntries.map((entry) => `- ${entry.source}: ${entry.path}`)]),
          `permissionMode=${executionConfig.permissionMode}`,
          ...(executionConfig.invalidPermissionMode ? [`invalidPermissionMode=${executionConfig.invalidPermissionMode}`] : []),
          `hooks.PreToolUse=${executionConfig.hooks.PreToolUse.length}`,
          `hooks.PreCommand=${executionConfig.hooks.PreCommand.length}`,
          `hooks.PostCommand=${executionConfig.hooks.PostCommand.length}`,
          `hooks.SessionStart=${executionConfig.hooks.SessionStart.length}`,
          `hooks.SessionEnd=${executionConfig.hooks.SessionEnd.length}`,
          `hooks.ApprovalRequested=${executionConfig.hooks.ApprovalRequested.length}`,
          `hooks.ApprovalResolved=${executionConfig.hooks.ApprovalResolved.length}`,
          `unsupportedHooks=${executionConfig.unsupportedHookKeys.length === 0 ? "<none>" : executionConfig.unsupportedHookKeys.join(",")}`,
          "Merged config:",
          JSON.stringify(effectiveConfig.merged, null, 2),
        ].join("\n");
      }
      case "prompt": {
        const effectiveConfig = config ?? (await this.loadRuntimeConfig());
        const effectivePromptContext = promptContext ?? (await this.loadPromptContext(effectiveConfig));
        return [
          `cwd=${effectivePromptContext.project.cwd}`,
          `date=${effectivePromptContext.project.currentDate}`,
          `gitStatus=${effectivePromptContext.project.gitStatus ?? "<none>"}`,
          ...(effectivePromptContext.project.instructionFiles.length === 0
            ? ["instructionFiles=<none>"]
            : [
                "instructionFiles:",
                ...effectivePromptContext.project.instructionFiles.map((file) => `- ${file.path}`),
              ]),
        ].join("\n");
      }
      case "unknown":
        return `Unknown slash command: /${command.name}`;
    }
  }

  writeError(message: string): void {
    this.stderr.write(`${message}\n`);
  }

  private async collectRunProgress<T>(
    sessionId: string,
    action: () => Promise<T>,
  ): Promise<{ result: T; progressLines: string[] }> {
    const events = subscribeRuntimeEvents(this.runtime, { sessionId, family: "run" })[Symbol.asyncIterator]();
    const progressLines: string[] = [];
    let cancelled = false;
    const collector = (async () => {
      while (!cancelled) {
        const next = await events.next();
        if (next.done) {
          break;
        }
        const line = formatProgressEvent(next.value);
        if (line) {
          progressLines.push(line);
        }
      }
    })();
    try {
      const result = await action();
      return { result, progressLines };
    } finally {
      cancelled = true;
      await events.return?.();
      await collector;
    }
  }

  private async watchActive(sessionId: string): Promise<string> {
    const run = await this.runtime.getActiveRun(sessionId);
    if (run) {
      return await this.watchRun(sessionId, run.id);
    }
    const task = await this.runtime.getActiveBackgroundTask(sessionId);
    if (task) {
      return await this.watchTask(sessionId, task.id);
    }
    return `No active run or background task for session ${sessionId}.`;
  }

  private async watchRun(sessionId: string, requestedRunId?: string): Promise<string> {
    const run = requestedRunId ? await this.runtime.getRun(requestedRunId) : await this.runtime.getActiveRun(sessionId);
    if (!run) {
      return requestedRunId ? `Unknown run: ${requestedRunId}` : `No active run for session ${sessionId}.`;
    }
    const snapshot = this.runtime.events.snapshot((event) => {
      if ((event.type !== "run.progress" && event.type !== "run.updated" && event.type !== "run.started") || !event.payload || typeof event.payload !== "object") {
        return false;
      }
      if (event.type === "run.progress") {
        return (event.payload as { runId?: string }).runId === run.id;
      }
      return (event.payload as { id?: string }).id === run.id;
    });
    const lines = snapshot
      .map((event) => formatWatchEvent(event))
      .filter((line): line is string => Boolean(line));
    if (lines.length === 0) {
      lines.push(`[run ${run.id}] ${run.status} ${run.title}`);
    }
    return lines.join("\n");
  }

  private async watchTask(sessionId: string, requestedTaskId?: string): Promise<string> {
    const task = requestedTaskId ? await this.runtime.getBackgroundTask(requestedTaskId) : await this.runtime.getActiveBackgroundTask(sessionId);
    if (!task) {
      return requestedTaskId ? `Unknown background task: ${requestedTaskId}` : `No active background task for session ${sessionId}.`;
    }
    const synced = await this.runtime.syncBackgroundTask(task.id) ?? task;
    const output = await this.runtime.getBackgroundTaskOutput(task.id, { lineLimit: 200, offset: 0 });
    return [
      `[task ${synced.id}] ${synced.status} ${synced.title}`,
      output?.output?.trim() ? output.output : "<no output>",
    ].join("\n");
  }

  private requireActiveSessionId(sessionId?: string): string {
    if (sessionId) {
      this.activeSessionId = sessionId;
      return sessionId;
    }
    if (this.activeSessionId) {
      return this.activeSessionId;
    }
    throw new Error("No active session.");
  }
}

function isTranscriptMessageRecord(record: TranscriptRecord): record is Extract<TranscriptRecord, { message: { role: string; content: string } }> {
  return "message" in record;
}

function formatHookResult(result: { event: string; command: string; stdout: string; stderr: string }): string {
  const parts = [result.stdout, result.stderr].filter(Boolean).join(" ").trim();
  return parts ? `[${result.event}] ${parts}` : `[${result.event}] ${result.command}`;
}

function formatProgressEvent(event: { type: string; payload?: unknown }): string | undefined {
  if (event.type !== "run.progress" || !event.payload || typeof event.payload !== "object") {
    return undefined;
  }
  const payload = event.payload as {
    runId?: string;
    message?: string;
    relatedApprovalId?: string;
    relatedTaskId?: string;
    relatedSkillId?: string;
  };
  if (!payload.message) {
    return undefined;
  }
  const prefix = payload.runId ? `[run ${payload.runId}]` : "[run]";
  const related = [
    payload.relatedApprovalId ? `approval=${payload.relatedApprovalId}` : undefined,
    payload.relatedTaskId ? `task=${payload.relatedTaskId}` : undefined,
    payload.relatedSkillId ? `skill=${payload.relatedSkillId}` : undefined,
  ].filter(Boolean).join(" ");
  return related ? `${prefix} ${payload.message} ${related}` : `${prefix} ${payload.message}`;
}

function formatWatchEvent(event: { type: string; payload?: unknown }): string | undefined {
  if (!event.payload || typeof event.payload !== "object") {
    return undefined;
  }
  if (event.type === "run.progress") {
    return formatProgressEvent(event);
  }
  if (event.type === "run.started" || event.type === "run.updated") {
    const payload = event.payload as { id?: string; status?: string; title?: string };
    if (!payload.id) {
      return undefined;
    }
    return `[run ${payload.id}] ${payload.status ?? "unknown"}${payload.title ? ` ${payload.title}` : ""}`;
  }
  if (event.type.startsWith("background-task.")) {
    const payload = event.payload as { id?: string; status?: string; title?: string };
    if (!payload.id) {
      return undefined;
    }
    return `[task ${payload.id}] ${payload.status ?? "unknown"}${payload.title ? ` ${payload.title}` : ""}`;
  }
  return undefined;
}

function formatOperatorTasks(tasks: Array<{
  id: string;
  status: string;
  subject: string;
  owner?: string;
  blockedBy: string[];
  blocks: string[];
}>): string {
  if (tasks.length === 0) {
    return "No tasks.";
  }
  return tasks
    .map((task) => {
      const details = [
        task.owner ? `owner=${task.owner}` : undefined,
        task.blockedBy.length > 0 ? `blockedBy=${task.blockedBy.join(",")}` : undefined,
        task.blocks.length > 0 ? `blocks=${task.blocks.join(",")}` : undefined,
      ].filter(Boolean).join(" ");
      return details ? `${task.id} ${task.status} ${task.subject} ${details}` : `${task.id} ${task.status} ${task.subject}`;
    })
    .join("\n");
}

function formatOperatorMessages(messages: Array<{
  id: string;
  fromSessionId: string;
  toSessionId: string;
  summary: string;
}>): string {
  if (messages.length === 0) {
    return "No messages.";
  }
  return messages
    .map((message) => `${message.id} from=${message.fromSessionId} to=${message.toSessionId} ${message.summary}`)
    .join("\n");
}

function formatRemoteInventory(items: Array<{
  remoteId: string;
  remoteSource?: string;
  session?: { id: string; status: string; agentId?: string; updatedAt: string };
  pairing?: { code: string; status: string; expiresAt: string };
  control?: { state: "active" | "paused" | "quarantined" | "degraded"; reason?: string };
  diagnostics?: { cause: string; recoverable: boolean; taskId?: string; recommendedAction?: string };
  activeRun?: { id: string; title: string; status: string; updatedAt: string };
  activeBackgroundTask?: { id: string; title: string; kind: string; status: string; updatedAt: string };
  gateway?: { state: "healthy" | "stale" | "closed"; updatedAt: number; lastClientActivityAt?: number; lastServerEventAt?: number };
}>): string {
  if (items.length === 0) {
    return "No remote inventory.";
  }
  return items.map((item) => {
    const parts = [
      `remote=${item.remoteId}`,
      `source=${item.remoteSource ?? "<unknown>"}`,
      `control=${item.control?.state ?? "<none>"}${item.control?.reason ? `:${item.control.reason}` : ""}`,
      item.pairing ? `pairing=${item.pairing.status}:${item.pairing.code}` : "pairing=<none>",
      item.session ? `session=${item.session.id}:${item.session.status}` : "session=<none>",
      item.activeRun ? `run=${item.activeRun.status}:${item.activeRun.title}` : "run=<none>",
      item.activeBackgroundTask ? `task=${item.activeBackgroundTask.status}:${item.activeBackgroundTask.title}` : "task=<none>",
      item.gateway ? `gateway=${item.gateway.state}` : "gateway=<none>",
    ];
    if (item.diagnostics?.cause) {
      parts.push(`diagnostics=${item.diagnostics.cause}`);
    }
    if (item.diagnostics?.recommendedAction) {
      parts.push(`action=${item.diagnostics.recommendedAction}`);
    }
    return parts.join(" ");
  }).join("\n");
}

function formatRemoteStatus(status: {
  remoteId: string;
  remoteSource?: string;
  session: { id: string; status: string; metadata: { agentId?: string } };
  pairing?: { code: string; status: string; expiresAt: string };
  approvals: Array<{ id: string; title: string }>;
  control: { state: "active" | "paused" | "quarantined" | "degraded"; reason?: string };
  diagnostics?: { cause: string; recoverable: boolean; taskId?: string; recommendedAction?: string };
  activeRun?: { id: string; title: string; status: string; updatedAt: string };
  activeBackgroundTask?: { id: string; title: string; kind: string; status: string; updatedAt: string };
  recentEvents: Array<{ type: string }>;
}): string {
  return [
    `remote=${status.remoteId} source=${status.remoteSource ?? "<unknown>"}`,
    `session=${status.session.id} status=${status.session.status} agent=${status.session.metadata.agentId ?? "<none>"}`,
    `control=${status.control.state}${status.control.reason ? ` reason=${status.control.reason}` : ""}`,
    status.diagnostics
      ? `diagnostics cause=${status.diagnostics.cause} recoverable=${status.diagnostics.recoverable}${status.diagnostics.taskId ? ` task=${status.diagnostics.taskId}` : ""}${status.diagnostics.recommendedAction ? ` action=${status.diagnostics.recommendedAction}` : ""}`
      : "diagnostics=<none>",
    status.pairing
      ? `pairing=${status.pairing.code} ${status.pairing.status} expires=${status.pairing.expiresAt}`
      : "pairing=<none>",
    `approvals=${status.approvals.length === 0 ? "<none>" : status.approvals.map((approval) => `${approval.id}:${approval.title}`).join(",")}`,
    status.activeRun
      ? `run=${status.activeRun.id} ${status.activeRun.status} ${status.activeRun.title} updated=${status.activeRun.updatedAt}`
      : "run=<none>",
    status.activeBackgroundTask
      ? `task=${status.activeBackgroundTask.id} ${status.activeBackgroundTask.status} ${status.activeBackgroundTask.kind} ${status.activeBackgroundTask.title} updated=${status.activeBackgroundTask.updatedAt}`
      : "task=<none>",
    `events=${status.recentEvents.length === 0 ? "<none>" : status.recentEvents.map((event) => event.type).join(",")}`,
  ].join("\n");
}

function formatPlatformInventory(items: Array<{
  platform: string;
  remoteCount: number;
  control: { state: "active" | "paused" | "quarantined" | "degraded" | "mixed"; reason?: string };
}>): string {
  if (items.length === 0) {
    return "No platform inventory.";
  }
  return items
    .map((item) => `platform=${item.platform} remotes=${item.remoteCount} control=${item.control.state}${item.control.reason ? `:${item.control.reason}` : ""}`)
    .join("\n");
}

function formatPlatformStatus(status: {
  platform: string;
  remoteCount: number;
  control: { state: "active" | "paused" | "quarantined" | "degraded" | "mixed"; reason?: string };
  remotes: Array<{
    remoteId: string;
    remoteSource?: string;
    control?: { state: "active" | "paused" | "quarantined" | "degraded"; reason?: string };
    diagnostics?: { cause: string; recoverable: boolean; taskId?: string; recommendedAction?: string };
    session?: { id: string; status: string; agentId?: string; updatedAt: string };
    activeRun?: { id: string; title: string; status: string; updatedAt: string };
    activeBackgroundTask?: { id: string; title: string; kind: string; status: string; updatedAt: string };
  }>;
}): string {
  return [
    `platform=${status.platform} remotes=${status.remoteCount}`,
    `control=${status.control.state}${status.control.reason ? ` reason=${status.control.reason}` : ""}`,
    `members=${status.remotes.length === 0 ? "<none>" : status.remotes.map((remote) => {
      const parts = [
        `${remote.remoteId}`,
        `control=${remote.control?.state ?? "<none>"}${remote.control?.reason ? `:${remote.control.reason}` : ""}`,
        remote.session ? `session=${remote.session.id}:${remote.session.status}` : "session=<none>",
        remote.activeRun ? `run=${remote.activeRun.status}:${remote.activeRun.title}` : "run=<none>",
        remote.activeBackgroundTask ? `task=${remote.activeBackgroundTask.status}:${remote.activeBackgroundTask.title}` : "task=<none>",
      ];
      if (remote.diagnostics?.cause) {
        parts.push(`diagnostics=${remote.diagnostics.cause}`);
      }
      if (remote.diagnostics?.recommendedAction) {
        parts.push(`action=${remote.diagnostics.recommendedAction}`);
      }
      return parts.join(" ");
    }).join(" | ")}`,
  ].join("\n");
}

function formatPlatformControlResult(
  result: {
    platform: string;
    remoteCount: number;
    control: { state: "active" | "paused" | "quarantined" | "degraded" | "mixed"; reason?: string };
    remotes: Array<{ remoteId: string }>;
    results: Array<{ remoteId: string; ok: true; control: { state: "active" | "paused" | "quarantined" | "degraded"; reason?: string } } | { remoteId: string; ok: false; error: string }>;
  },
  verb: "Paused" | "Resumed",
): string {
  const summary = `${verb} platform ${result.platform} remotes=${result.remoteCount} control=${result.control.state}${result.control.reason ? ` reason=${result.control.reason}` : ""}`;
  const details = result.results.length === 0
    ? "results=<none>"
    : `results=${result.results.map((entry) => entry.ok ? `${entry.remoteId}:${entry.control.state}${entry.control.reason ? `:${entry.control.reason}` : ""}` : `${entry.remoteId}:error:${entry.error}`).join(",")}`;
  return [summary, details].join("\n");
}

export function parseSlashCommand(input: string): OperatorCliSlashCommand | undefined {
  const trimmed = input.trim();
  if (!trimmed.startsWith("/")) {
    return undefined;
  }
  const [name, ...rest] = trimmed.slice(1).split(/\s+/);
  const tail = rest.join(" ").trim();
  switch (name) {
    case "help":
      return { kind: "help" };
    case "status":
      return { kind: "status" };
    case "sessions":
      return { kind: "sessions" };
    case "resume":
      return tail ? { kind: "resume", sessionId: tail } : { kind: "invalid", message: "Usage: /resume <sessionId>" };
    case "idle":
      return { kind: "idle" };
    case "complete":
      return { kind: "complete" };
    case "transcript":
      return parseTranscriptCommand(tail);
    case "approvals":
      return { kind: "approvals" };
    case "approve":
      return tail ? { kind: "approve", approvalId: tail } : { kind: "invalid", message: "Usage: /approve <approvalId>" };
    case "deny":
      return tail ? { kind: "deny", approvalId: tail } : { kind: "invalid", message: "Usage: /deny <approvalId>" };
    case "pairing":
      return parsePairingCommand(tail);
    case "remote":
      return parseRemoteCommand(tail);
    case "platform":
      return parsePlatformCommand(tail);
    case "recall":
      return { kind: "recall", query: tail };
    case "tasks":
      return tail ? { kind: "invalid", message: "Usage: /tasks" } : { kind: "tasks" };
    case "task-create":
      return tail ? { kind: "task-create", subject: tail } : { kind: "invalid", message: "Usage: /task-create <subject>" };
    case "task-update":
      return parseTaskUpdateCommand(tail);
    case "messages":
      return tail ? { kind: "invalid", message: "Usage: /messages" } : { kind: "messages" };
    case "inbox":
      return tail ? { kind: "invalid", message: "Usage: /inbox" } : { kind: "inbox" };
    case "outbox":
      return tail ? { kind: "invalid", message: "Usage: /outbox" } : { kind: "outbox" };
    case "send":
      return parseSendCommand(tail);
    case "notify":
      return parseNotifyCommand(tail);
    case "skills":
      return { kind: "skills" };
    case "run-skill":
      return { kind: "run-skill", skillId: tail };
    case "watch":
    case "follow":
      return parseWatchCommand(tail);
    case "background":
      return parseBackgroundCommand(tail);
    case "training":
      return { kind: "training" };
    case "cron":
      return parseCronCommand(tail);
    case "config":
      return { kind: "config" };
    case "prompt":
      return { kind: "prompt" };
    default:
      return { kind: "unknown", name };
  }
}

function parseTranscriptCommand(tail: string): OperatorCliSlashCommand {
  if (!tail) {
    return { kind: "transcript" };
  }
  const limit = parseOptionalPositiveInteger(tail);
  return typeof limit === "number"
    ? { kind: "transcript", limit }
    : { kind: "invalid", message: "Usage: /transcript [limit]" };
}

function parseNotifyCommand(tail: string): OperatorCliSlashCommand {
  if (!tail) {
    return { kind: "invalid", message: "Usage: /notify <status> <message>" };
  }
  const [status, ...messageParts] = tail.split(/\s+/);
  const message = messageParts.join(" ").trim();
  return status && message
    ? { kind: "notify", status, message }
    : { kind: "invalid", message: "Usage: /notify <status> <message>" };
}

function parseWatchCommand(tail: string): OperatorCliSlashCommand {
  if (!tail || tail === "active") {
    return { kind: "watch-active" };
  }
  if (tail === "run") {
    return { kind: "watch-run" };
  }
  if (tail === "task") {
    return { kind: "watch-task" };
  }
  if (tail.startsWith("run ")) {
    const runId = tail.slice("run ".length).trim();
    return runId ? { kind: "watch-run", runId } : { kind: "invalid", message: "Usage: /watch run [runId]" };
  }
  if (tail.startsWith("task ")) {
    const taskId = tail.slice("task ".length).trim();
    return taskId ? { kind: "watch-task", taskId } : { kind: "invalid", message: "Usage: /watch task [taskId]" };
  }
  return { kind: "invalid", message: "Usage: /watch [active|run <runId>|task <taskId>]" };
}

function parsePairingCommand(tail: string): OperatorCliSlashCommand {
  if (!tail) {
    return { kind: "pairing-list" };
  }
  if (tail === "create") {
    return { kind: "pairing-create" };
  }
  if (tail.startsWith("create ")) {
    const remoteSource = tail.slice("create ".length).trim();
    return remoteSource ? { kind: "pairing-create", remoteSource } : { kind: "invalid", message: "Usage: /pairing create [source]" };
  }
  if (tail === "approve") {
    return { kind: "invalid", message: "Usage: /pairing approve <code|id>" };
  }
  if (tail === "reject") {
    return { kind: "invalid", message: "Usage: /pairing reject <code|id>" };
  }
  if (tail.startsWith("approve ")) {
    const identifier = tail.slice("approve ".length).trim();
    return identifier ? { kind: "pairing-approve", identifier } : { kind: "invalid", message: "Usage: /pairing approve <code|id>" };
  }
  if (tail.startsWith("reject ")) {
    const identifier = tail.slice("reject ".length).trim();
    return identifier ? { kind: "pairing-reject", identifier } : { kind: "invalid", message: "Usage: /pairing reject <code|id>" };
  }
  if (tail === "pending" || tail === "approved" || tail === "rejected" || tail === "redeemed" || tail === "expired") {
    return { kind: "pairing-list", status: tail };
  }
  return { kind: "invalid", message: "Usage: /pairing [create|approve|reject|pending|approved|rejected|redeemed|expired]" };
}

function parseRemoteCommand(tail: string): OperatorCliSlashCommand {
  const usage = "Usage: /remote <list [source]|status|pause|resume|repair> <remoteId|sessionId> [reason|missing-process|missing-state]";
  if (!tail) {
    return { kind: "invalid", message: usage };
  }
  if (tail === "list") {
    return { kind: "remote-list" };
  }
  if (tail.startsWith("list ")) {
    const remoteSource = tail.slice("list ".length).trim();
    return remoteSource ? { kind: "remote-list", remoteSource } : { kind: "invalid", message: usage };
  }
  if (tail === "status" || tail === "pause" || tail === "resume" || tail === "repair") {
    return { kind: "invalid", message: usage };
  }
  if (tail.startsWith("status ")) {
    const identifier = tail.slice("status ".length).trim();
    return identifier ? { kind: "remote-status", identifier } : { kind: "invalid", message: usage };
  }
  if (tail.startsWith("pause ")) {
    const rest = tail.slice("pause ".length).trim();
    if (!rest) {
      return { kind: "invalid", message: usage };
    }
    const [identifier, ...reasonParts] = rest.split(/\s+/);
    return {
      kind: "remote-pause",
      identifier,
      ...(reasonParts.length > 0 ? { reason: reasonParts.join(" ") } : {}),
    };
  }
  if (tail.startsWith("resume ")) {
    const identifier = tail.slice("resume ".length).trim();
    return identifier ? { kind: "remote-resume", identifier } : { kind: "invalid", message: usage };
  }
  if (tail.startsWith("repair ")) {
    const rest = tail.slice("repair ".length).trim();
    if (!rest) {
      return { kind: "invalid", message: usage };
    }
    const [identifier, ...reasonParts] = rest.split(/\s+/);
    if (reasonParts.some((reason) => reason !== "missing-process" && reason !== "missing-state")) {
      return { kind: "invalid", message: usage };
    }
    return {
      kind: "remote-repair",
      identifier,
      ...(reasonParts.length > 0 ? { reasons: [...new Set(reasonParts)] as Array<"missing-process" | "missing-state"> } : {}),
    };
  }
  return { kind: "invalid", message: usage };
}

function parsePlatformCommand(tail: string): OperatorCliSlashCommand {
  const usage = "Usage: /platform <list|status|pause|resume> <platform> [reason]";
  if (!tail || tail === "list") {
    return tail ? { kind: "platform-list" } : { kind: "invalid", message: usage };
  }
  if (tail === "status" || tail === "pause" || tail === "resume") {
    return { kind: "invalid", message: usage };
  }
  if (tail.startsWith("status ")) {
    const platform = tail.slice("status ".length).trim();
    return platform ? { kind: "platform-status", platform } : { kind: "invalid", message: usage };
  }
  if (tail.startsWith("pause ")) {
    const rest = tail.slice("pause ".length).trim();
    if (!rest) {
      return { kind: "invalid", message: usage };
    }
    const [platform, ...reasonParts] = rest.split(/\s+/);
    return {
      kind: "platform-pause",
      platform,
      ...(reasonParts.length > 0 ? { reason: reasonParts.join(" ") } : {}),
    };
  }
  if (tail.startsWith("resume ")) {
    const platform = tail.slice("resume ".length).trim();
    return platform ? { kind: "platform-resume", platform } : { kind: "invalid", message: usage };
  }
  return { kind: "invalid", message: usage };
}

function parseBackgroundCommand(tail: string): OperatorCliSlashCommand {
  if (!tail) {
    return { kind: "background-list" };
  }
  if (tail === "list") {
    return { kind: "background-list" };
  }
  if (tail === "view") {
    return { kind: "invalid", message: "Usage: /background view <taskId> [lines]" };
  }
  if (tail === "sync") {
    return { kind: "invalid", message: "Usage: /background sync <taskId>" };
  }
  if (tail === "cancel") {
    return { kind: "invalid", message: "Usage: /background cancel <taskId>" };
  }
  if (tail.startsWith("start ")) {
    const rest = tail.slice("start ".length);
    const delimiter = " -- ";
    const delimiterIndex = rest.indexOf(delimiter);
    if (delimiterIndex < 0) {
      return { kind: "invalid", message: "Usage: /background start <title> -- <command>" };
    }
    const title = rest.slice(0, delimiterIndex).trim();
    const command = rest.slice(delimiterIndex + delimiter.length).trim();
    if (!title || !command) {
      return { kind: "invalid", message: "Usage: /background start <title> -- <command>" };
    }
    return { kind: "background-start", title, command };
  }
  if (tail.startsWith("view ")) {
    const [taskId, lines] = tail.slice("view ".length).trim().split(/\s+/, 2);
    if (!taskId) {
      return { kind: "invalid", message: "Usage: /background view <taskId> [lines]" };
    }
    if (!lines) {
      return { kind: "background-view", taskId };
    }
    const parsedLines = parseOptionalPositiveInteger(lines);
    return typeof parsedLines === "number"
      ? { kind: "background-view", taskId, lines: parsedLines }
      : { kind: "invalid", message: "Usage: /background view <taskId> [lines]" };
  }
  if (tail.startsWith("sync ")) {
    const taskId = tail.slice("sync ".length).trim();
    return taskId ? { kind: "background-sync", taskId } : { kind: "invalid", message: "Usage: /background sync <taskId>" };
  }
  if (tail.startsWith("cancel ")) {
    const taskId = tail.slice("cancel ".length).trim();
    return taskId ? { kind: "background-cancel", taskId } : { kind: "invalid", message: "Usage: /background cancel <taskId>" };
  }
  return { kind: "invalid", message: "Usage: /background [start|view|sync|cancel] ..." };
}

function parseTaskUpdateCommand(tail: string): OperatorCliSlashCommand {
  const [taskId, status, ...rest] = tail.split(/\s+/).filter(Boolean);
  if (!taskId || !status || rest.length > 0) {
    return { kind: "invalid", message: "Usage: /task-update <taskId> <pending|in_progress|completed>" };
  }
  if (status !== "pending" && status !== "in_progress" && status !== "completed") {
    return { kind: "invalid", message: "Usage: /task-update <taskId> <pending|in_progress|completed>" };
  }
  return { kind: "task-update", taskId, status };
}

function parseSendCommand(tail: string): OperatorCliSlashCommand {
  const [toSessionId, ...messageParts] = tail.split(/\s+/).filter(Boolean);
  const message = messageParts.join(" ").trim();
  if (!toSessionId || !message) {
    return { kind: "invalid", message: "Usage: /send <sessionId> <message>" };
  }
  return { kind: "send", toSessionId, message };
}

function parseCronCommand(tail: string): OperatorCliSlashCommand {
  if (!tail) {
    return { kind: "cron-list" };
  }
  if (tail === "list") {
    return { kind: "cron-list" };
  }
  if (tail === "tick") {
    return { kind: "cron-tick" };
  }
  if (tail.startsWith("runs")) {
    const jobId = tail.slice("runs".length).trim();
    return jobId ? { kind: "cron-runs", jobId } : { kind: "cron-runs" };
  }
  if (tail.startsWith("delete ")) {
    const jobId = tail.slice("delete ".length).trim();
    return jobId ? { kind: "cron-delete", jobId } : { kind: "invalid", message: "Usage: /cron delete <jobId>" };
  }
  if (tail.startsWith("create ")) {
    const rest = tail.slice("create ".length).trim();
    const tokens = rest.split(/\s+/);
    if (tokens.length < 6) {
      return { kind: "invalid", message: "Usage: /cron create <cronExpr> <prompt> [--on-success <target>] [--on-failure <target>]" };
    }
    const cron = tokens.slice(0, 5).join(" ");
    const remainder = tokens.slice(5);
    const promptTokens: string[] = [];
    const onSuccess: string[] = [];
    const onFailure: string[] = [];
    let index = 0;
    while (index < remainder.length) {
      const token = remainder[index];
      if (token === "--on-success" || token === "--on-failure") {
        const target = remainder[index + 1];
        if (!target) {
          return { kind: "invalid", message: "Usage: /cron create <cronExpr> <prompt> [--on-success <target>] [--on-failure <target>]" };
        }
        if (token === "--on-success") {
          onSuccess.push(target);
        } else {
          onFailure.push(target);
        }
        index += 2;
        continue;
      }
      promptTokens.push(token);
      index += 1;
    }
    const prompt = promptTokens.join(" ").trim();
    if (!prompt) {
      return { kind: "invalid", message: "Usage: /cron create <cronExpr> <prompt> [--on-success <target>] [--on-failure <target>]" };
    }
    return {
      kind: "cron-create",
      cron,
      prompt,
      ...((onSuccess.length > 0 || onFailure.length > 0)
        ? {
            delivery: {
              ...(onSuccess.length > 0 ? { onSuccess } : {}),
              ...(onFailure.length > 0 ? { onFailure } : {}),
            },
          }
        : {}),
    };
  }
  return { kind: "invalid", message: "Usage: /cron [create|runs|tick|delete] ..." };
}

function parseOptionalPositiveInteger(value: string): number | undefined {
  if (!/^\d+$/.test(value)) {
    return undefined;
  }
  const parsed = Number.parseInt(value, 10);
  return parsed > 0 ? parsed : undefined;
}
