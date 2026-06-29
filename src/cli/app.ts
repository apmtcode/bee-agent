import { execFile } from "node:child_process";
import fs from "node:fs/promises";
import path from "node:path";
import process from "node:process";
import { clearLine, clearScreenDown, cursorTo } from "node:readline";
import readline from "node:readline/promises";
import type { Interface as ReadlineInterface } from "node:readline/promises";
import { promisify } from "node:util";
import { subscribeRuntimeEvents } from "../control-plane/subscriptions.js";
import { OperatorControlPlaneServer } from "../control-plane/server.js";
import type { DeliveryTarget } from "../control-plane/delivery.js";
import { StandaloneOperatorRuntime } from "../orchestrator/operator-runtime.js";
import type { SpawnBackgroundProcess } from "../harness/background-tasks.js";
import type { TranscriptRecord } from "../harness/transcript-store.js";
import {
  OperatorCliConfigLoader,
  resolveOperatorCliExecutionConfig,
  resolveOperatorCliStatusLineConfig,
  setProjectPermissionModeConfig,
  setProjectStatusLineConfig,
  type OperatorCliPermissionMode,
  type OperatorCliRuntimeConfig,
} from "./config.js";
import { runOperatorCliConversationTurn } from "./conversation.js";
import { discoverPromptContext, type OperatorCliPromptContext } from "./prompt.js";
import {
  buildOperatorCliStatusLineSnapshot,
  formatOperatorCliPrompt,
  OperatorCliStatusLineController,
} from "./status-line.js";
import { FileOperatorCliTeamStore } from "./team-store.js";

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
  | { kind: "tasks"; team?: string }
  | { kind: "task-create"; subject: string; team?: string; owner?: string; blockedBy?: string[] }
  | { kind: "task-update"; taskId: string; status?: "pending" | "in_progress" | "completed"; owner?: string | null; blockedBy?: string[] }
  | { kind: "messages" }
  | { kind: "inbox" }
  | { kind: "outbox" }
  | { kind: "send"; toSessionId: string; message: string }
  | { kind: "notify"; status: string; message: string }
  | { kind: "plan-show" }
  | { kind: "plan-set"; content: string }
  | { kind: "plan-request-approval"; approverSessionId: string }
  | { kind: "plan-respond"; requestId: string; approve: boolean; feedback?: string }
  | { kind: "verify"; status: "pending" | "requested" | "completed" | "skipped"; notes?: string }
  | { kind: "plan" }
  | { kind: "plan-exit"; mode?: Exclude<OperatorCliPermissionMode, "plan"> }
  | { kind: "statusline"; command?: string }
  | { kind: "worktree"; name?: string }
  | { kind: "worktree-exit"; action: "keep" | "remove" }
  | { kind: "teams" }
  | { kind: "team-create"; name: string; description?: string }
  | { kind: "team-delete"; name: string }
  | { kind: "teammates"; team: string }
  | { kind: "teammate-start"; team: string; name: string; title?: string; agentId?: string }
  | { kind: "teammate-message"; team: string; teammate: string; message: string }
  | { kind: "teammate-update"; team: string; teammate: string; status: "pending" | "running" | "paused" | "completed" | "failed" }
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
  | { kind: "task-stop"; taskId: string }
  | { kind: "background" }
  | { kind: "monitor-list" }
  | { kind: "monitor-start"; title: string; command: string }
  | { kind: "monitor-view"; taskId: string; lines?: number }
  | { kind: "monitor-sync"; taskId: string }
  | { kind: "monitor-stop"; taskId: string }
  | { kind: "monitor" }
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
  /**
   * Root of the user-level config (the `.claude` directory). Defaults to
   * `CLAUDE_CONFIG_HOME` / `$HOME/.claude` in production. Tests should set this
   * to an isolated path so the developer's real global config never leaks in.
   */
  configHome?: string;
  /**
   * Overrides how background-task launch scripts are spawned. Production uses a
   * real detached process; tests that drive task state directly can inject
   * {@link inertBackgroundSpawn} so a live launch script doesn't race their
   * writes (and skew aggregated remote-control health).
   */
  backgroundTaskSpawnProcess?: SpawnBackgroundProcess;
  stdin?: NodeJS.ReadableStream;
  stdout?: NodeJS.WritableStream;
  stderr?: NodeJS.WritableStream;
};

type OperatorCliWorktreeSession = {
  originalCwd: string;
  path: string;
  branch?: string;
};

const execFileAsync = promisify(execFile);

export class OperatorCliApp {
  readonly runtime: StandaloneOperatorRuntime;
  readonly server: OperatorControlPlaneServer;
  private cwd: string;
  readonly currentDate: string;
  private readonly stdin: NodeJS.ReadableStream;
  private readonly stdout: NodeJS.WritableStream;
  private readonly stderr: NodeJS.WritableStream;
  private activeSessionId?: string;
  private readonly approvedCommandFingerprints = new Set<string>();
  private worktreeSession?: OperatorCliWorktreeSession;
  private statusLineController?: OperatorCliStatusLineController;
  private activeReader?: ReadlineInterface;
  readonly teams: FileOperatorCliTeamStore;

  constructor(private readonly options: OperatorCliAppOptions) {
    this.runtime = new StandaloneOperatorRuntime({
      rootDir: options.rootDir,
      ...(options.backgroundTaskSpawnProcess
        ? { backgroundTaskSpawnProcess: options.backgroundTaskSpawnProcess }
        : {}),
    });
    this.server = new OperatorControlPlaneServer({ runtime: this.runtime });
    this.teams = new FileOperatorCliTeamStore(options.rootDir);
    this.cwd = options.cwd ?? process.cwd();
    this.currentDate = options.currentDate ?? new Date().toISOString().slice(0, 10);
    this.stdin = options.stdin ?? process.stdin;
    this.stdout = options.stdout ?? process.stdout;
    this.stderr = options.stderr ?? process.stderr;
  }

  async loadRuntimeConfig(): Promise<OperatorCliRuntimeConfig> {
    const config = await new OperatorCliConfigLoader(this.cwd, this.options.configHome).load();
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

    this.writeCliOutput("Standalone Operator CLI");
    this.writeCliOutput(`Session: ${session.id}`);
    this.writeCliOutput(`Loaded config files: ${config.loadedEntries.length}`);
    this.writeCliOutput(`Loaded instruction files: ${promptContext.project.instructionFiles.length}`);
    this.writeCliOutput("Type /help for commands. Ctrl+D exits.");

    const reader = readline.createInterface({
      input: this.stdin,
      output: this.stdout,
      prompt: formatOperatorCliPrompt(undefined),
    });
    this.activeReader = reader;

    await this.configureStatusLine(config, session.id);
    this.renderPrompt();
    try {
      while (true) {
        const line = await reader.question("");
        if (!line.trim()) {
          this.renderPrompt();
          continue;
        }
        const output = await this.handleInput(line);
        this.writeCliOutput(output);
        await this.configureStatusLine(await this.loadRuntimeConfig());
        await this.refreshStatusLine();
        this.renderPrompt();
      }
    } catch {
      this.clearInteractivePrompt();
      this.stdout.write("\n");
    } finally {
      this.activeReader = undefined;
      await this.statusLineController?.stop();
      this.statusLineController = undefined;
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
          "  /tasks [team <name>]",
          "  /task-create <subject> [--team <name>] [--owner <owner>] [--blocked-by <taskId,...>]",
          "  /task-update <taskId> [<pending|in_progress|completed>] [--owner <owner|off>] [--blocked-by <taskId,...>]",
          "  /messages",
          "  /inbox",
          "  /outbox",
          "  /send <sessionId> <message>",
          "  /notify <status> <message>",
          "  /plan-show",
          "  /plan-set <content>",
          "  /plan-request-approval <sessionId>",
          "  /plan-respond <requestId> <approve|reject> [feedback]",
          "  /verify <pending|requested|completed|skipped> [notes]",
          "  /plan",
          "  /plan-exit [default|acceptEdits|bypassPermissions]",
          "  /statusline [command|off]",
          "  /worktree [name]",
          "  /worktree-exit [keep|remove]",
          "  /teams",
          "  /team-create <name> [description]",
          "  /team-delete <name>",
          "  /teammates <team>",
          "  /teammate-start <team> <name> [title] [--agent <agentId>]",
          "  /teammate-message <team> <name> <message>",
          "  /teammate-update <team> <name> <pending|running|paused|completed|failed>",
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
          "  /task-stop <taskId>",
          "  /monitor",
          "  /monitor start <title> -- <command>",
          "  /monitor view <taskId> [lines]",
          "  /monitor sync <taskId>",
          "  /monitor stop <taskId>",
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
        const team = command.team ? await this.teams.get(command.team) : undefined;
        if (command.team && !team) {
          return `Unknown team: ${command.team}`;
        }
        const tasks = await this.runtime.listTasks(team?.taskSessionId ?? activeSessionId);
        return formatOperatorTasks(tasks);
      }
      case "task-create": {
        const activeSessionId = this.requireActiveSessionId(sessionId);
        const team = command.team ? await this.teams.get(command.team) : undefined;
        if (command.team && !team) {
          return `Unknown team: ${command.team}`;
        }
        const task = await this.runtime.createTask({
          sessionId: team?.taskSessionId ?? activeSessionId,
          subject: command.subject,
          description: command.subject,
          ...(command.owner ? { owner: command.owner } : {}),
          ...(command.blockedBy?.length ? { blockedBy: command.blockedBy } : {}),
        });
        return `Created task ${task.id} ${task.status} ${task.subject}`;
      }
      case "task-update": {
        const task = await this.runtime.updateTask(command.taskId, {
          ...(command.status ? { status: command.status } : {}),
          ...(command.owner !== undefined ? { owner: command.owner } : {}),
          ...(command.blockedBy ? { blockedBy: command.blockedBy } : {}),
        });
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
      case "plan-show": {
        const activeSessionId = this.requireActiveSessionId(sessionId);
        const plan = await this.runtime.getPlan(activeSessionId);
        return plan ? formatOperatorPlan(plan) : `No plan for session ${activeSessionId}.`;
      }
      case "plan-set": {
        const activeSessionId = this.requireActiveSessionId(sessionId);
        const plan = await this.runtime.upsertPlan({
          sessionId: activeSessionId,
          content: command.content,
        });
        return `Updated plan ${plan.id} status=${plan.status}`;
      }
      case "plan-request-approval": {
        const activeSessionId = this.requireActiveSessionId(sessionId);
        const requested = await this.runtime.requestPlanApproval({
          sessionId: activeSessionId,
          approverSessionId: command.approverSessionId,
        });
        return `Requested plan approval ${requested.plan.approval?.requestId} plan=${requested.plan.id} approver=${command.approverSessionId}`;
      }
      case "plan-respond": {
        const activeSessionId = this.requireActiveSessionId(sessionId);
        const responded = await this.runtime.respondPlanApproval({
          requestId: command.requestId,
          sessionId: activeSessionId,
          approve: command.approve,
          ...(command.feedback ? { feedback: command.feedback } : {}),
        });
        return `Responded to plan approval ${command.requestId} status=${responded.plan.status}`;
      }
      case "verify": {
        const activeSessionId = this.requireActiveSessionId(sessionId);
        const plan = await this.runtime.updatePlanVerification({
          sessionId: activeSessionId,
          status: command.status,
          ...(command.notes ? { notes: command.notes } : {}),
        });
        return `Updated plan verification ${plan.id} status=${command.status}${command.notes ? ` notes=${command.notes}` : ""}`;
      }
      case "plan": {
        const effectiveConfig = config ?? (await this.loadRuntimeConfig());
        const executionConfig = resolveOperatorCliExecutionConfig(effectiveConfig);
        if (executionConfig.permissionMode === "plan") {
          const previous = readPreviousPermissionMode(effectiveConfig.merged);
          return `Plan mode already enabled${previous ? ` previous=${previous}` : ""}`;
        }
        const previousPermissionMode = executionConfig.permissionMode === "default"
          ? undefined
          : executionConfig.permissionMode;
        await setProjectPermissionModeConfig(this.cwd, "plan", previousPermissionMode);
        return `Enabled plan mode${previousPermissionMode ? ` previous=${previousPermissionMode}` : ""}`;
      }
      case "plan-exit": {
        const effectiveConfig = config ?? (await this.loadRuntimeConfig());
        const executionConfig = resolveOperatorCliExecutionConfig(effectiveConfig);
        const restoredMode = command.mode ?? readPreviousPermissionMode(effectiveConfig.merged) ?? "default";
        if (executionConfig.permissionMode !== "plan" && !command.mode) {
          return `Plan mode is not enabled. permissionMode=${executionConfig.permissionMode}`;
        }
        await setProjectPermissionModeConfig(this.cwd, restoredMode, undefined);
        return `Disabled plan mode permissionMode=${restoredMode}`;
      }
      case "statusline": {
        if (!command.command) {
          const effectiveConfig = config ?? (await this.loadRuntimeConfig());
          const statusLine = resolveOperatorCliStatusLineConfig(effectiveConfig);
          return statusLine
            ? `statusLine command=${statusLine.command}`
            : "statusLine disabled";
        }
        if (command.command === "off") {
          await setProjectStatusLineConfig(this.cwd, undefined);
          await this.configureStatusLine(undefined, this.activeSessionId);
          return `Disabled status line in ${this.cwd}/.claude/settings.local.json`;
        }
        const nextStatusLine = { type: "command" as const, command: command.command };
        await setProjectStatusLineConfig(this.cwd, nextStatusLine);
        await this.configureStatusLine({ merged: { statusLine: nextStatusLine }, loadedEntries: [] }, this.activeSessionId);
        return `Configured status line command=${command.command}`;
      }
      case "worktree": {
        return await this.enterWorktree(command.name);
      }
      case "worktree-exit": {
        return await this.exitWorktree(command.action);
      }
      case "teams": {
        const teams = await this.teams.list();
        return teams.length === 0
          ? "No teams."
          : teams.map((team) => `${team.name}${team.description ? ` ${team.description}` : ""}${team.taskSessionId ? ` taskSession=${team.taskSessionId}` : ""}`).join("\n");
      }
      case "team-create": {
        const existing = await this.teams.get(command.name);
        const taskSession = existing?.taskSessionId
          ? await this.runtime.getSession(existing.taskSessionId)
          : undefined;
        const createdTaskSession = taskSession ?? await this.runtime.startSession({
          title: `Team ${command.name}`,
          cwd: this.cwd,
          agentId: `team:${command.name}`,
        });
        const team = await this.teams.create({
          name: command.name,
          ...(command.description ? { description: command.description } : {}),
          taskSessionId: createdTaskSession.id,
        });
        return `Created team ${team.name}${team.description ? ` description=${team.description}` : ""} taskSession=${team.taskSessionId ?? createdTaskSession.id}`;
      }
      case "team-delete": {
        const team = await this.teams.delete(command.name);
        return team ? `Deleted team ${team.name}.` : `Unknown team: ${command.name}`;
      }
      case "teammates": {
        const team = await this.teams.get(command.team);
        if (!team) {
          return `Unknown team: ${command.team}`;
        }
        const teammates = await this.runtime.listTeammates(command.team);
        return formatTeammates(command.team, teammates, team.taskSessionId);
      }
      case "teammate-start": {
        const activeSessionId = this.requireActiveSessionId(sessionId);
        const team = await this.teams.get(command.team);
        if (!team) {
          return `Unknown team: ${command.team}`;
        }
        const parentRun = await this.runtime.getActiveRun(activeSessionId);
        if (!parentRun) {
          return `No active run for session ${activeSessionId}.`;
        }
        const started = await this.runtime.startTeammate({
          teamName: command.team,
          sessionId: activeSessionId,
          parentRunId: parentRun.id,
          teammateName: command.name,
          title: command.title ?? `Teammate ${command.name}`,
          ...(command.agentId ? { agentId: command.agentId } : {}),
          cwd: this.cwd,
        });
        return `Started teammate ${started.teammateName} team=${started.teamName} session=${started.spawned.childSession.id} run=${started.spawned.subagent.runId} taskSession=${started.taskSessionId}`;
      }
      case "teammate-message": {
        const activeSessionId = this.requireActiveSessionId(sessionId);
        const result = await this.runtime.sendTeammateMessage({
          teamName: command.team,
          fromSessionId: activeSessionId,
          teammateName: command.teammate,
          message: command.message,
        });
        return `Sent teammate message ${result.message.id} team=${result.teamName} teammate=${result.teammateName} session=${result.teammateSessionId}`;
      }
      case "teammate-update": {
        const updated = await this.runtime.updateTeammateStatus(command.team, command.teammate, command.status);
        return updated
          ? `Updated teammate ${updated.name} team=${command.team} status=${updated.status}`
          : `Unknown teammate: ${command.team}/${command.teammate}`;
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
      case "monitor":
      case "monitor-list": {
        const activeSessionId = this.requireActiveSessionId(sessionId);
        const response = await this.server.handle({ method: "monitors.list", params: { sessionId: activeSessionId } });
        if (!response.ok) {
          return response.error.message;
        }
        return response.result.length === 0
          ? "No monitors."
          : response.result.map((task) => `${task.id} ${task.status} ${task.title}`).join("\n");
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
      case "monitor-start": {
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
        const response = await this.server.handle({
          method: "monitors.start",
          params: {
            sessionId: activeSessionId,
            title: authorization.title ?? command.title,
            command: authorization.command,
            cwd: this.cwd,
          },
        });
        if (!response.ok) {
          return response.error.message;
        }
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
          `Started monitor ${response.result.id} (${response.result.status}) ${response.result.title}`,
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
      case "monitor-view": {
        const response = await this.server.handle({ method: "monitors.sync", params: { taskId: command.taskId } });
        if (!response.ok) {
          return response.error.message;
        }
        const outputResponse = await this.server.handle({ method: "monitors.output", params: { taskId: command.taskId, lineLimit: command.lines ?? 50 } });
        if (!outputResponse.ok) {
          return outputResponse.error.message;
        }
        return [
          `${response.result.id} ${response.result.status} ${response.result.title}`,
          outputResponse.result.output.trim() ? outputResponse.result.output : "<no output>",
        ].join("\n");
      }
      case "background-sync": {
        const task = await this.runtime.syncBackgroundTask(command.taskId);
        return task ? `${task.id} ${task.status} ${task.title}` : `Unknown background task: ${command.taskId}`;
      }
      case "monitor-sync": {
        const response = await this.server.handle({ method: "monitors.sync", params: { taskId: command.taskId } });
        if (!response.ok) {
          return response.error.message;
        }
        return `${response.result.id} ${response.result.status} ${response.result.title}`;
      }
      case "background-cancel": {
        const task = await this.runtime.cancelBackgroundTask(command.taskId);
        return task ? `Cancelled background task ${task.id}.` : `Unknown background task: ${command.taskId}`;
      }
      case "monitor-stop": {
        const response = await this.server.handle({ method: "monitors.stop", params: { taskId: command.taskId } });
        if (!response.ok) {
          return response.error.message;
        }
        return `Stopped monitor ${response.result.id}.`;
      }
      case "task-stop": {
        const response = await this.server.handle({ method: "tasks.stop", params: { taskId: command.taskId } });
        if (!response.ok) {
          return response.error.message;
        }
        return `Stopped task ${response.result.id}.`;
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
                  ...(job.delivery?.onSuccess?.map((target) => `success:${formatDeliveryTargetLabel(target)}`) ?? []),
                  ...(job.delivery?.onFailure?.map((target) => `failure:${formatDeliveryTargetLabel(target)}`) ?? []),
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
                const delivery = run.deliveryResults?.map((result) => `${formatDeliveryTargetLabel(result.target)}:${result.status}${result.error ? `:${result.error}` : ""}`).join(",");
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
                const delivery = run.deliveryResults?.map((result) => `${formatDeliveryTargetLabel(result.target)}:${result.status}${result.error ? `:${result.error}` : ""}`).join(",");
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
    this.clearInteractivePrompt();
    this.stderr.write(`${message}\n`);
    this.renderPrompt();
  }

  private async configureStatusLine(config: OperatorCliRuntimeConfig | undefined, sessionId?: string): Promise<void> {
    const activeSessionId = sessionId ?? this.activeSessionId;
    const statusLineConfig = resolveOperatorCliStatusLineConfig(config);
    if (!activeSessionId) {
      await this.statusLineController?.updateConfig(statusLineConfig);
      return;
    }
    const buildSnapshot = async () => buildOperatorCliStatusLineSnapshot({
      runtime: this.runtime,
      sessionId: activeSessionId,
      cwd: this.cwd,
      currentDate: this.currentDate,
      config,
    });
    if (!this.statusLineController) {
      this.statusLineController = new OperatorCliStatusLineController({
        config: statusLineConfig,
        stdout: this.stdout as NodeJS.WritableStream & { isTTY?: boolean; columns?: number },
        buildSnapshot,
        onRenderedLineChange: () => {
          this.renderPrompt();
        },
      });
      await this.statusLineController.start();
      return;
    }
    await this.statusLineController.updateConfig(statusLineConfig);
  }

  private async refreshStatusLine(): Promise<void> {
    await this.statusLineController?.refresh();
  }

  private writeCliOutput(message: string): void {
    this.clearInteractivePrompt();
    this.stdout.write(`${message}\n`);
  }

  private clearInteractivePrompt(): void {
    if (!this.activeReader || !this.isInteractiveTTY()) {
      return;
    }
    this.activeReader.pause();
    cursorTo(this.stdout, 0);
    clearLine(this.stdout, 0);
    clearScreenDown(this.stdout);
  }

  private renderPrompt(): void {
    if (!this.activeReader) {
      return;
    }
    const prompt = formatOperatorCliPrompt(this.statusLineController?.currentLine);
    this.activeReader.setPrompt(prompt);
    if (!this.isInteractiveTTY()) {
      return;
    }
    this.activeReader.prompt(true);
  }

  private isInteractiveTTY(): boolean {
    return Boolean((this.stdout as NodeJS.WritableStream & { isTTY?: boolean }).isTTY);
  }

  private async enterWorktree(name?: string): Promise<string> {
    if (this.worktreeSession) {
      return `Already in worktree ${this.worktreeSession.path}`;
    }
    const repoRoot = await this.resolveGitRoot(this.cwd);
    if (!repoRoot) {
      return `Not a git repository: ${this.cwd}`;
    }
    const safeName = sanitizeWorktreeName(name) ?? `operator-${Date.now().toString(36)}`;
    const worktreesDir = path.join(repoRoot, ".operator", "worktrees");
    const worktreePath = path.join(worktreesDir, safeName);
    await fs.mkdir(worktreesDir, { recursive: true });
    const branch = `operator/${safeName}`;
    await execFileAsync("git", ["worktree", "add", "-b", branch, worktreePath], { cwd: repoRoot });
    this.worktreeSession = { originalCwd: this.cwd, path: worktreePath, branch };
    this.cwd = worktreePath;
    return `Entered worktree ${worktreePath} branch=${branch}`;
  }

  private async exitWorktree(action: "keep" | "remove"): Promise<string> {
    if (!this.worktreeSession) {
      return "No active worktree.";
    }
    const session = this.worktreeSession;
    this.cwd = session.originalCwd;
    this.worktreeSession = undefined;
    if (action === "keep") {
      return `Exited worktree ${session.path} and kept branch=${session.branch ?? "<none>"}`;
    }
    await execFileAsync("git", ["worktree", "remove", "--force", session.path], { cwd: session.originalCwd });
    if (session.branch) {
      await execFileAsync("git", ["branch", "-D", session.branch], { cwd: session.originalCwd });
    }
    return `Exited and removed worktree ${session.path}`;
  }

  private async resolveGitRoot(cwd: string): Promise<string | undefined> {
    try {
      const { stdout } = await execFileAsync("git", ["rev-parse", "--show-toplevel"], { cwd });
      const root = stdout.trim();
      return root || undefined;
    } catch {
      return undefined;
    }
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

function formatOperatorPlan(plan: {
  id: string;
  status: string;
  content: string;
  approval?: {
    requestId: string;
    approverSessionId: string;
    approved?: boolean;
    feedback?: string;
  };
  verification?: {
    status: string;
    reminderAt?: string;
    notes?: string;
  };
}): string {
  const parts = [
    `plan=${plan.id}`,
    `status=${plan.status}`,
    plan.approval ? `approval=${plan.approval.requestId}:${plan.approval.approverSessionId}${plan.approval.approved === undefined ? "" : `:${plan.approval.approved ? "approved" : "rejected"}`}` : undefined,
    plan.verification ? `verification=${plan.verification.status}` : undefined,
    plan.verification?.reminderAt ? `reminder=${plan.verification.reminderAt}` : undefined,
    plan.verification?.notes ? `notes=${plan.verification.notes}` : undefined,
    `content=${plan.content.replace(/\s+/g, " ").trim()}`,
  ].filter(Boolean).join(" ");
  return parts;
}

function formatTeammates(
  team: string,
  teammates: Array<{
    name: string;
    sessionId: string;
    subagentRunId: string;
    childRunId: string;
    title: string;
    agentId?: string;
    status: string;
  }>,
  taskSessionId?: string,
): string {
  if (teammates.length === 0) {
    return `No teammates for ${team}${taskSessionId ? ` taskSession=${taskSessionId}` : ""}.`;
  }
  return teammates
    .map((teammate) => {
      const details = [
        `team=${team}`,
        `status=${teammate.status}`,
        `session=${teammate.sessionId}`,
        `subagent=${teammate.subagentRunId}`,
        `run=${teammate.childRunId}`,
        taskSessionId ? `taskSession=${taskSessionId}` : undefined,
        teammate.agentId ? `agent=${teammate.agentId}` : undefined,
      ].filter(Boolean).join(" ");
      return `${teammate.name} ${teammate.title} ${details}`;
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

function readPreviousPermissionMode(config: Record<string, unknown>): Exclude<OperatorCliPermissionMode, "plan"> | undefined {
  const operator = config.operator;
  if (!operator || typeof operator !== "object" || Array.isArray(operator)) {
    return undefined;
  }
  const previousPermissionMode = (operator as { previousPermissionMode?: unknown }).previousPermissionMode;
  return previousPermissionMode === "default" || previousPermissionMode === "acceptEdits" || previousPermissionMode === "bypassPermissions"
    ? previousPermissionMode
    : undefined;
}

function sanitizeWorktreeName(value?: string): string | undefined {
  if (!value) {
    return undefined;
  }
  const trimmed = value.trim();
  if (!trimmed) {
    return undefined;
  }
  const sanitized = trimmed.replace(/[^A-Za-z0-9._-]+/g, "-").replace(/^-+|-+$/g, "");
  return sanitized || undefined;
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
      return parseTasksCommand(tail);
    case "task-create":
      return parseTaskCreateCommand(tail);
    case "task-update":
      return parseTaskUpdateCommand(tail);
    case "task-stop":
      return tail ? { kind: "task-stop", taskId: tail } : { kind: "invalid", message: "Usage: /task-stop <taskId>" };
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
    case "plan-show":
      return tail ? { kind: "invalid", message: "Usage: /plan-show" } : { kind: "plan-show" };
    case "plan-set":
      return tail ? { kind: "plan-set", content: tail } : { kind: "invalid", message: "Usage: /plan-set <content>" };
    case "plan-request-approval":
      return tail ? { kind: "plan-request-approval", approverSessionId: tail } : { kind: "invalid", message: "Usage: /plan-request-approval <sessionId>" };
    case "plan-respond":
      return parsePlanRespondCommand(tail);
    case "verify":
      return parseVerifyCommand(tail);
    case "plan":
      return tail ? { kind: "invalid", message: "Usage: /plan" } : { kind: "plan" };
    case "plan-exit":
      return parsePlanExitCommand(tail);
    case "statusline":
      return parseStatusLineCommand(tail);
    case "worktree":
      return parseWorktreeCommand(tail);
    case "worktree-exit":
      return parseWorktreeExitCommand(tail);
    case "teams":
      return tail ? { kind: "invalid", message: "Usage: /teams" } : { kind: "teams" };
    case "team-create":
      return parseTeamCreateCommand(tail);
    case "team-delete":
      return tail ? { kind: "team-delete", name: tail } : { kind: "invalid", message: "Usage: /team-delete <name>" };
    case "teammates":
      return parseTeammatesCommand(tail);
    case "teammate-start":
      return parseTeammateStartCommand(tail);
    case "teammate-message":
      return parseTeammateMessageCommand(tail);
    case "teammate-update":
      return parseTeammateUpdateCommand(tail);
    case "skills":
      return { kind: "skills" };
    case "run-skill":
      return { kind: "run-skill", skillId: tail };
    case "watch":
    case "follow":
      return parseWatchCommand(tail);
    case "background":
      return parseBackgroundCommand(tail);
    case "monitor":
      return parseMonitorCommand(tail);
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

function parsePlanExitCommand(tail: string): OperatorCliSlashCommand {
  if (!tail) {
    return { kind: "plan-exit" };
  }
  if (tail === "default" || tail === "acceptEdits" || tail === "bypassPermissions") {
    return { kind: "plan-exit", mode: tail };
  }
  return { kind: "invalid", message: "Usage: /plan-exit [default|acceptEdits|bypassPermissions]" };
}

function parsePlanRespondCommand(tail: string): OperatorCliSlashCommand {
  const [requestId, decision, ...feedbackParts] = tail.split(/\s+/).filter(Boolean);
  if (!requestId || !decision || (decision !== "approve" && decision !== "reject")) {
    return { kind: "invalid", message: "Usage: /plan-respond <requestId> <approve|reject> [feedback]" };
  }
  const feedback = feedbackParts.join(" ").trim();
  return {
    kind: "plan-respond",
    requestId,
    approve: decision === "approve",
    ...(feedback ? { feedback } : {}),
  };
}

function parseVerifyCommand(tail: string): OperatorCliSlashCommand {
  const [status, ...noteParts] = tail.split(/\s+/).filter(Boolean);
  if (status !== "pending" && status !== "requested" && status !== "completed" && status !== "skipped") {
    return { kind: "invalid", message: "Usage: /verify <pending|requested|completed|skipped> [notes]" };
  }
  const notes = noteParts.join(" ").trim();
  return {
    kind: "verify",
    status,
    ...(notes ? { notes } : {}),
  };
}

function parseStatusLineCommand(tail: string): OperatorCliSlashCommand {
  if (!tail) {
    return { kind: "statusline" };
  }
  return { kind: "statusline", command: tail };
}

function parseWorktreeCommand(tail: string): OperatorCliSlashCommand {
  return tail ? { kind: "worktree", name: tail } : { kind: "worktree" };
}

function parseWorktreeExitCommand(tail: string): OperatorCliSlashCommand {
  if (!tail) {
    return { kind: "worktree-exit", action: "keep" };
  }
  if (tail === "keep" || tail === "remove") {
    return { kind: "worktree-exit", action: tail };
  }
  return { kind: "invalid", message: "Usage: /worktree-exit [keep|remove]" };
}

function parseTeamCreateCommand(tail: string): OperatorCliSlashCommand {
  const [name, ...descriptionParts] = tail.split(/\s+/).filter(Boolean);
  if (!name) {
    return { kind: "invalid", message: "Usage: /team-create <name> [description]" };
  }
  const description = descriptionParts.join(" ").trim();
  return description
    ? { kind: "team-create", name, description }
    : { kind: "team-create", name };
}

function parseTeammatesCommand(tail: string): OperatorCliSlashCommand {
  return tail
    ? { kind: "teammates", team: tail }
    : { kind: "invalid", message: "Usage: /teammates <team>" };
}

function parseTeammateStartCommand(tail: string): OperatorCliSlashCommand {
  const tokens = tail.split(/\s+/).filter(Boolean);
  const [team, name, ...rest] = tokens;
  if (!team || !name) {
    return { kind: "invalid", message: "Usage: /teammate-start <team> <name> [title] [--agent <agentId>]" };
  }
  let agentId: string | undefined;
  const titleParts: string[] = [];
  for (let index = 0; index < rest.length; index += 1) {
    const token = rest[index];
    if (token === "--agent") {
      const value = rest[index + 1];
      if (!value) {
        return { kind: "invalid", message: "Usage: /teammate-start <team> <name> [title] [--agent <agentId>]" };
      }
      agentId = value;
      index += 1;
      continue;
    }
    if (token.startsWith("--")) {
      return { kind: "invalid", message: "Usage: /teammate-start <team> <name> [title] [--agent <agentId>]" };
    }
    titleParts.push(token);
  }
  const title = titleParts.join(" ").trim();
  return {
    kind: "teammate-start",
    team,
    name,
    ...(title ? { title } : {}),
    ...(agentId ? { agentId } : {}),
  };
}

function parseTeammateMessageCommand(tail: string): OperatorCliSlashCommand {
  const [team, teammate, ...messageParts] = tail.split(/\s+/).filter(Boolean);
  const message = messageParts.join(" ").trim();
  if (!team || !teammate || !message) {
    return { kind: "invalid", message: "Usage: /teammate-message <team> <name> <message>" };
  }
  return { kind: "teammate-message", team, teammate, message };
}

function parseTeammateUpdateCommand(tail: string): OperatorCliSlashCommand {
  const [team, teammate, status] = tail.split(/\s+/).filter(Boolean);
  if (!team || !teammate || !status) {
    return { kind: "invalid", message: "Usage: /teammate-update <team> <name> <pending|running|paused|completed|failed>" };
  }
  if (status !== "pending" && status !== "running" && status !== "paused" && status !== "completed" && status !== "failed") {
    return { kind: "invalid", message: "Usage: /teammate-update <team> <name> <pending|running|paused|completed|failed>" };
  }
  return { kind: "teammate-update", team, teammate, status };
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

function parseTasksCommand(tail: string): OperatorCliSlashCommand {
  if (!tail) {
    return { kind: "tasks" };
  }
  const parts = tail.split(/\s+/).filter(Boolean);
  if (parts.length === 2 && parts[0] === "team") {
    return { kind: "tasks", team: parts[1] };
  }
  return { kind: "invalid", message: "Usage: /tasks [team <name>]" };
}

function parseTaskCreateCommand(tail: string): OperatorCliSlashCommand {
  const parsed = parseFlagSegments(tail);
  if (!parsed.subject) {
    return { kind: "invalid", message: "Usage: /task-create <subject> [--team <name>] [--owner <owner>] [--blocked-by <taskId,...>]" };
  }
  if (parsed.invalid) {
    return { kind: "invalid", message: "Usage: /task-create <subject> [--team <name>] [--owner <owner>] [--blocked-by <taskId,...>]" };
  }
  return {
    kind: "task-create",
    subject: parsed.subject,
    ...(parsed.team ? { team: parsed.team } : {}),
    ...(parsed.owner ? { owner: parsed.owner } : {}),
    ...(parsed.blockedBy ? { blockedBy: parsed.blockedBy } : {}),
  };
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

function parseMonitorCommand(tail: string): OperatorCliSlashCommand {
  if (!tail) {
    return { kind: "monitor-list" };
  }
  if (tail === "list") {
    return { kind: "monitor-list" };
  }
  if (tail === "view") {
    return { kind: "invalid", message: "Usage: /monitor view <taskId> [lines]" };
  }
  if (tail === "sync") {
    return { kind: "invalid", message: "Usage: /monitor sync <taskId>" };
  }
  if (tail === "stop") {
    return { kind: "invalid", message: "Usage: /monitor stop <taskId>" };
  }
  if (tail.startsWith("start ")) {
    const rest = tail.slice("start ".length);
    const delimiter = " -- ";
    const delimiterIndex = rest.indexOf(delimiter);
    if (delimiterIndex < 0) {
      return { kind: "invalid", message: "Usage: /monitor start <title> -- <command>" };
    }
    const title = rest.slice(0, delimiterIndex).trim();
    const command = rest.slice(delimiterIndex + delimiter.length).trim();
    if (!title || !command) {
      return { kind: "invalid", message: "Usage: /monitor start <title> -- <command>" };
    }
    return { kind: "monitor-start", title, command };
  }
  if (tail.startsWith("view ")) {
    const [taskId, lines] = tail.slice("view ".length).trim().split(/\s+/, 2);
    if (!taskId) {
      return { kind: "invalid", message: "Usage: /monitor view <taskId> [lines]" };
    }
    if (!lines) {
      return { kind: "monitor-view", taskId };
    }
    const parsedLines = parseOptionalPositiveInteger(lines);
    return typeof parsedLines === "number"
      ? { kind: "monitor-view", taskId, lines: parsedLines }
      : { kind: "invalid", message: "Usage: /monitor view <taskId> [lines]" };
  }
  if (tail.startsWith("sync ")) {
    const taskId = tail.slice("sync ".length).trim();
    return taskId ? { kind: "monitor-sync", taskId } : { kind: "invalid", message: "Usage: /monitor sync <taskId>" };
  }
  if (tail.startsWith("stop ")) {
    const taskId = tail.slice("stop ".length).trim();
    return taskId ? { kind: "monitor-stop", taskId } : { kind: "invalid", message: "Usage: /monitor stop <taskId>" };
  }
  return { kind: "invalid", message: "Usage: /monitor [start|view|sync|stop] ..." };
}

function parseTaskUpdateCommand(tail: string): OperatorCliSlashCommand {
  const tokens = tail.split(/\s+/).filter(Boolean);
  const [taskId, maybeStatus, ...rest] = tokens;
  if (!taskId) {
    return { kind: "invalid", message: "Usage: /task-update <taskId> [<pending|in_progress|completed>] [--owner <owner|off>] [--blocked-by <taskId,...>]" };
  }
  const hasStatus = maybeStatus === "pending" || maybeStatus === "in_progress" || maybeStatus === "completed";
  if (maybeStatus && !hasStatus && !maybeStatus.startsWith("--")) {
    return { kind: "invalid", message: "Usage: /task-update <taskId> [<pending|in_progress|completed>] [--owner <owner|off>] [--blocked-by <taskId,...>]" };
  }
  const flags = parseFlagSegments([...(hasStatus ? rest : tokens.slice(1))].join(" "));
  if (flags.invalid || (!hasStatus && flags.owner === undefined && flags.blockedBy === undefined)) {
    return { kind: "invalid", message: "Usage: /task-update <taskId> [<pending|in_progress|completed>] [--owner <owner|off>] [--blocked-by <taskId,...>]" };
  }
  return {
    kind: "task-update",
    taskId,
    ...(hasStatus ? { status: maybeStatus } : {}),
    ...(flags.owner !== undefined ? { owner: flags.owner } : {}),
    ...(flags.blockedBy ? { blockedBy: flags.blockedBy } : {}),
  };
}

function parseFlagSegments(tail: string): {
  subject?: string;
  team?: string;
  owner?: string | null;
  blockedBy?: string[];
  invalid?: boolean;
} {
  const tokens = tail.split(/\s+/).filter(Boolean);
  const subjectParts: string[] = [];
  let team: string | undefined;
  let owner: string | null | undefined;
  let blockedBy: string[] | undefined;
  for (let index = 0; index < tokens.length; index += 1) {
    const token = tokens[index];
    if (token === "--team") {
      const value = tokens[index + 1];
      if (!value) {
        return { invalid: true };
      }
      team = value;
      index += 1;
      continue;
    }
    if (token === "--owner") {
      const value = tokens[index + 1];
      if (!value) {
        return { invalid: true };
      }
      owner = value === "off" ? null : value;
      index += 1;
      continue;
    }
    if (token === "--blocked-by") {
      const value = tokens[index + 1];
      if (!value) {
        return { invalid: true };
      }
      blockedBy = value.split(",").map((item) => item.trim()).filter(Boolean);
      if (blockedBy.length === 0) {
        return { invalid: true };
      }
      index += 1;
      continue;
    }
    if (token.startsWith("--")) {
      return { invalid: true };
    }
    subjectParts.push(token);
  }
  return {
    ...(subjectParts.length > 0 ? { subject: subjectParts.join(" ") } : {}),
    ...(team ? { team } : {}),
    ...(owner !== undefined ? { owner } : {}),
    ...(blockedBy ? { blockedBy } : {}),
  };
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

/** Render a delivery target as a short CLI label, covering every target kind. */
function formatDeliveryTargetLabel(target: DeliveryTarget): string {
  switch (target.kind) {
    case "local":
      return "local";
    case "webhook":
      return target.url;
    case "browser-push":
      return `browser-push:${target.endpoint}`;
  }
}
