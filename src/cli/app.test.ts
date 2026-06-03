import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { OperatorCliApp, parseSlashCommand } from "./app.js";

const tempDirs: string[] = [];

async function makeTempDir(): Promise<string> {
  const dir = await fs.mkdtemp(path.join(os.tmpdir(), "operator-cli-app-"));
  tempDirs.push(dir);
  return dir;
}

afterEach(async () => {
  await Promise.all(tempDirs.splice(0).map((dir) => fs.rm(dir, { recursive: true, force: true, maxRetries: 5, retryDelay: 50 })));
});

describe("parseSlashCommand", () => {
  it("parses supported slash commands", () => {
    expect(parseSlashCommand("/status")).toEqual({ kind: "status" });
    expect(parseSlashCommand("/sessions")).toEqual({ kind: "sessions" });
    expect(parseSlashCommand("/resume session-1")).toEqual({ kind: "resume", sessionId: "session-1" });
    expect(parseSlashCommand("/transcript 5")).toEqual({ kind: "transcript", limit: 5 });
    expect(parseSlashCommand("/approve approval-1")).toEqual({ kind: "approve", approvalId: "approval-1" });
    expect(parseSlashCommand("/deny approval-1")).toEqual({ kind: "deny", approvalId: "approval-1" });
    expect(parseSlashCommand("/pairing")).toEqual({ kind: "pairing-list" });
    expect(parseSlashCommand("/pairing pending")).toEqual({ kind: "pairing-list", status: "pending" });
    expect(parseSlashCommand("/pairing create gateway")).toEqual({ kind: "pairing-create", remoteSource: "gateway" });
    expect(parseSlashCommand("/pairing approve code-1")).toEqual({ kind: "pairing-approve", identifier: "code-1" });
    expect(parseSlashCommand("/pairing reject ticket-1")).toEqual({ kind: "pairing-reject", identifier: "ticket-1" });
    expect(parseSlashCommand("/remote list")).toEqual({ kind: "remote-list" });
    expect(parseSlashCommand("/remote list gateway")).toEqual({ kind: "remote-list", remoteSource: "gateway" });
    expect(parseSlashCommand("/remote status device-1")).toEqual({ kind: "remote-status", identifier: "device-1" });
    expect(parseSlashCommand("/remote pause device-1 gateway unhealthy")).toEqual({
      kind: "remote-pause",
      identifier: "device-1",
      reason: "gateway unhealthy",
    });
    expect(parseSlashCommand("/remote resume device-1")).toEqual({
      kind: "remote-resume",
      identifier: "device-1",
    });
    expect(parseSlashCommand("/remote repair device-1 missing-process missing-state")).toEqual({
      kind: "remote-repair",
      identifier: "device-1",
      reasons: ["missing-process", "missing-state"],
    });
    expect(parseSlashCommand("/platform list")).toEqual({ kind: "platform-list" });
    expect(parseSlashCommand("/platform status gateway")).toEqual({ kind: "platform-status", platform: "gateway" });
    expect(parseSlashCommand("/platform pause gateway gateway unhealthy")).toEqual({
      kind: "platform-pause",
      platform: "gateway",
      reason: "gateway unhealthy",
    });
    expect(parseSlashCommand("/platform resume gateway")).toEqual({ kind: "platform-resume", platform: "gateway" });
    expect(parseSlashCommand("/recall deploy bug")).toEqual({ kind: "recall", query: "deploy bug" });
    expect(parseSlashCommand("/tasks")).toEqual({ kind: "tasks" });
    expect(parseSlashCommand("/task-create Inspect logs")).toEqual({ kind: "task-create", subject: "Inspect logs" });
    expect(parseSlashCommand("/task-update task-1 in_progress")).toEqual({ kind: "task-update", taskId: "task-1", status: "in_progress" });
    expect(parseSlashCommand("/messages")).toEqual({ kind: "messages" });
    expect(parseSlashCommand("/inbox")).toEqual({ kind: "inbox" });
    expect(parseSlashCommand("/outbox")).toEqual({ kind: "outbox" });
    expect(parseSlashCommand("/send session-2 Please collect logs")).toEqual({ kind: "send", toSessionId: "session-2", message: "Please collect logs" });
    expect(parseSlashCommand("/background")).toEqual({ kind: "background-list" });
    expect(parseSlashCommand("/background start smoke -- printf ok")).toEqual({
      kind: "background-start",
      title: "smoke",
      command: "printf ok",
    });
    expect(parseSlashCommand("/background view task-1 25")).toEqual({ kind: "background-view", taskId: "task-1", lines: 25 });
    expect(parseSlashCommand("/background sync task-1")).toEqual({ kind: "background-sync", taskId: "task-1" });
    expect(parseSlashCommand("/background cancel task-1")).toEqual({ kind: "background-cancel", taskId: "task-1" });
    expect(parseSlashCommand("/watch active")).toEqual({ kind: "watch-active" });
    expect(parseSlashCommand("/watch run run-1")).toEqual({ kind: "watch-run", runId: "run-1" });
    expect(parseSlashCommand("/watch task task-1")).toEqual({ kind: "watch-task", taskId: "task-1" });
    expect(parseSlashCommand("/follow run run-2")).toEqual({ kind: "watch-run", runId: "run-2" });
    expect(parseSlashCommand("/cron")).toEqual({ kind: "cron-list" });
    expect(parseSlashCommand("/cron create */5 * * * * remind me")).toEqual({
      kind: "cron-create",
      cron: "*/5 * * * *",
      prompt: "remind me",
    });
    expect(parseSlashCommand("/cron create */5 * * * * remind me --on-success local --on-failure https://example.test/hook")).toEqual({
      kind: "cron-create",
      cron: "*/5 * * * *",
      prompt: "remind me",
      delivery: {
        onSuccess: ["local"],
        onFailure: ["https://example.test/hook"],
      },
    });
    expect(parseSlashCommand("/cron runs job-1")).toEqual({ kind: "cron-runs", jobId: "job-1" });
    expect(parseSlashCommand("/cron tick")).toEqual({ kind: "cron-tick" });
    expect(parseSlashCommand("/cron delete job-1")).toEqual({ kind: "cron-delete", jobId: "job-1" });
    expect(parseSlashCommand("/config")).toEqual({ kind: "config" });
    expect(parseSlashCommand("/prompt")).toEqual({ kind: "prompt" });
    expect(parseSlashCommand("hello")).toBeUndefined();
  });

  it("rejects malformed slash commands with usage help", () => {
    expect(parseSlashCommand("/resume")).toEqual({ kind: "invalid", message: "Usage: /resume <sessionId>" });
    expect(parseSlashCommand("/background start smoke")).toEqual({
      kind: "invalid",
      message: "Usage: /background start <title> -- <command>",
    });
    expect(parseSlashCommand("/background view")).toEqual({
      kind: "invalid",
      message: "Usage: /background view <taskId> [lines]",
    });
    expect(parseSlashCommand("/cron create * * *")).toEqual({
      kind: "invalid",
      message: "Usage: /cron create <cronExpr> <prompt> [--on-success <target>] [--on-failure <target>]",
    });
    expect(parseSlashCommand("/pairing approve")).toEqual({
      kind: "invalid",
      message: "Usage: /pairing approve <code|id>",
    });
    expect(parseSlashCommand("/pairing mystery")).toEqual({
      kind: "invalid",
      message: "Usage: /pairing [create|approve|reject|pending|approved|rejected|redeemed|expired]",
    });
    expect(parseSlashCommand("/remote")).toEqual({
      kind: "invalid",
      message: "Usage: /remote <list [source]|status|pause|resume|repair> <remoteId|sessionId> [reason|missing-process|missing-state]",
    });
    expect(parseSlashCommand("/remote pause")).toEqual({
      kind: "invalid",
      message: "Usage: /remote <list [source]|status|pause|resume|repair> <remoteId|sessionId> [reason|missing-process|missing-state]",
    });
    expect(parseSlashCommand("/remote repair device-1 bad-reason")).toEqual({
      kind: "invalid",
      message: "Usage: /remote <list [source]|status|pause|resume|repair> <remoteId|sessionId> [reason|missing-process|missing-state]",
    });
    expect(parseSlashCommand("/platform")).toEqual({
      kind: "invalid",
      message: "Usage: /platform <list|status|pause|resume> <platform> [reason]",
    });
    expect(parseSlashCommand("/platform pause")).toEqual({
      kind: "invalid",
      message: "Usage: /platform <list|status|pause|resume> <platform> [reason]",
    });
    expect(parseSlashCommand("/task-create")).toEqual({
      kind: "invalid",
      message: "Usage: /task-create <subject>",
    });
    expect(parseSlashCommand("/task-update")).toEqual({
      kind: "invalid",
      message: "Usage: /task-update <taskId> <pending|in_progress|completed>",
    });
    expect(parseSlashCommand("/task-update task-1 blocked")).toEqual({
      kind: "invalid",
      message: "Usage: /task-update <taskId> <pending|in_progress|completed>",
    });
    expect(parseSlashCommand("/messages later")).toEqual({
      kind: "invalid",
      message: "Usage: /messages",
    });
    expect(parseSlashCommand("/inbox later")).toEqual({
      kind: "invalid",
      message: "Usage: /inbox",
    });
    expect(parseSlashCommand("/outbox later")).toEqual({
      kind: "invalid",
      message: "Usage: /outbox",
    });
    expect(parseSlashCommand("/send session-2")).toEqual({
      kind: "invalid",
      message: "Usage: /send <sessionId> <message>",
    });
    expect(parseSlashCommand("/watch nope")).toEqual({
      kind: "invalid",
      message: "Usage: /watch [active|run <runId>|task <taskId>]",
    });
  });
});

describe("OperatorCliApp", () => {
  it("dispatches status and runtime-backed commands", async () => {
    const rootDir = await makeTempDir();
    const app = new OperatorCliApp({ rootDir, cwd: rootDir, currentDate: "2026-05-25" });
    const session = await app.runtime.startSession({ title: "CLI test", cwd: rootDir, agentId: "operator-cli" });

    await app.runtime.recordTurn({
      sessionId: session.id,
      userText: "check deploy",
      assistantText: "deploy checked",
      trajectorySummary: "Deploy checked for recall",
    });

    const status = await app.dispatchSlashCommand({ kind: "status" }, session.id);
    expect(status).toContain(`session=${session.id}`);
    expect(status).toContain("instructionFiles=");

    const recall = await app.dispatchSlashCommand({ kind: "recall", query: "deploy" }, session.id);
    expect(recall).toContain("Deploy checked for recall");
  });

  it("creates, lists, and updates operator tasks", async () => {
    const rootDir = await makeTempDir();
    const app = new OperatorCliApp({ rootDir, cwd: rootDir, currentDate: "2026-05-25" });
    const session = await app.runtime.startSession({ title: "task cli", cwd: rootDir, agentId: "operator-cli" });

    const createdOutput = await app.dispatchSlashCommand({ kind: "task-create", subject: "Inspect logs" }, session.id);
    expect(createdOutput).toContain("Created task");
    expect(createdOutput).toContain("pending Inspect logs");

    const tasks = await app.runtime.listTasks(session.id);
    expect(tasks).toHaveLength(1);
    const [task] = tasks;
    if (!task) {
      throw new Error("expected operator task");
    }

    const listOutput = await app.dispatchSlashCommand({ kind: "tasks" }, session.id);
    expect(listOutput).toContain(`${task.id} pending Inspect logs`);

    const updatedOutput = await app.dispatchSlashCommand({ kind: "task-update", taskId: task.id, status: "completed" }, session.id);
    expect(updatedOutput).toContain(`Updated task ${task.id} completed Inspect logs`);

    const updatedTask = await app.runtime.getTask(task.id);
    expect(updatedTask).toMatchObject({ id: task.id, status: "completed", subject: "Inspect logs" });
  });

  it("sends and lists operator messages", async () => {
    const rootDir = await makeTempDir();
    const app = new OperatorCliApp({ rootDir, cwd: rootDir, currentDate: "2026-05-25" });
    const sender = await app.runtime.startSession({ title: "sender cli", cwd: rootDir, agentId: "operator-cli" });
    const recipient = await app.runtime.startSession({ title: "recipient cli", cwd: rootDir, agentId: "worker" });

    const sentOutput = await app.dispatchSlashCommand({ kind: "send", toSessionId: recipient.id, message: "Please collect logs." }, sender.id);
    expect(sentOutput).toContain("Sent message");
    expect(sentOutput).toContain(`to=${recipient.id}`);
    expect(sentOutput).toContain("summary=Please collect logs.");

    const messages = await app.runtime.listMessages(sender.id);
    expect(messages).toHaveLength(1);
    const [message] = messages;
    if (!message) {
      throw new Error("expected operator message");
    }

    const listOutput = await app.dispatchSlashCommand({ kind: "messages" }, sender.id);
    expect(listOutput).toContain(`${message.id} from=${sender.id} to=${recipient.id} Please collect logs.`);

    const inboxOutput = await app.dispatchSlashCommand({ kind: "inbox" }, recipient.id);
    expect(inboxOutput).toContain(`${message.id} from=${sender.id} to=${recipient.id} Please collect logs.`);

    const outboxOutput = await app.dispatchSlashCommand({ kind: "outbox" }, sender.id);
    expect(outboxOutput).toContain(`${message.id} from=${sender.id} to=${recipient.id} Please collect logs.`);
  });

  it("streams in-flight freeform run updates and reuses transcript context across resume", async () => {
    const rootDir = await makeTempDir();
    const app = new OperatorCliApp({ rootDir, cwd: rootDir, currentDate: "2026-05-25" });
    const session = await app.runtime.startSession({ title: "freeform", cwd: rootDir, agentId: "operator-cli" });

    const firstTurn = await app.handleInput("hello", session.id);
    expect(firstTurn).toContain("[run ");
    expect(firstTurn).toContain("Handling freeform greeting request.");
    expect(firstTurn).toContain("Freeform turn completed.");
    expect(firstTurn).toContain(`Hello. I'm ready in ${rootDir}.`);
    expect(firstTurn).not.toContain("Freeform chat is not implemented yet");

    const transcriptAfterFirstTurn = await app.dispatchSlashCommand({ kind: "transcript", limit: 10 }, session.id);
    expect(transcriptAfterFirstTurn).toContain("user: hello");
    expect(transcriptAfterFirstTurn).toContain("assistant: Hello. I'm ready in");

    const otherSession = await app.runtime.startSession({ title: "other", cwd: rootDir, agentId: "operator-cli" });
    const resumeOutput = await app.dispatchSlashCommand({ kind: "resume", sessionId: session.id }, otherSession.id);
    expect(resumeOutput).toContain(session.id);

    const resumedTurn = await app.handleInput("what did i just say");
    expect(resumedTurn).toContain("[run ");
    expect(resumedTurn).toContain("Looking up the last user message.");
    expect(resumedTurn).toContain("Freeform turn completed.");
    expect(resumedTurn).toContain("hello");

    const sessionSummary = await app.handleInput("summarize this session");
    expect(sessionSummary).toContain(`Session summary for ${rootDir}:`);
    expect(sessionSummary).toContain("- user: hello");
    expect(sessionSummary).toContain("- assistant: Hello. I'm ready in");

    const runs = await app.runtime.listRuns(session.id);
    expect(runs.some((run) => run.metadata.kind === "cli.freeform" && run.status === "completed")).toBe(true);
  });

  it("gates dangerous freeform background commands and reruns after approval", async () => {
    const rootDir = await makeTempDir();
    await fs.mkdir(path.join(rootDir, ".claude"), { recursive: true });
    await fs.writeFile(path.join(rootDir, ".claude", "settings.local.json"), '{"permissionMode":"default"}');
    const app = new OperatorCliApp({ rootDir, cwd: rootDir, currentDate: "2026-05-25" });
    const session = await app.runtime.startSession({ title: "freeform policy", cwd: rootDir, agentId: "operator-cli" });

    const firstAttempt = await app.handleInput("start background wipe -- rm -rf build", session.id);
    expect(firstAttempt).toContain("Awaiting approval");
    expect(firstAttempt).toContain("Approval required:");
    expect(await app.runtime.listBackgroundTasks(session.id)).toEqual([]);

    const approvals = await app.runtime.listApprovals();
    expect(approvals).toHaveLength(1);
    const [approval] = approvals;
    if (!approval) {
      throw new Error("expected approval");
    }

    const approved = await app.handleInput(`approve ${approval.id}`, session.id);
    expect(approved).toContain(`Approved ${approval.id}`);

    const secondAttempt = await app.handleInput("start background wipe -- rm -rf build", session.id);
    expect(secondAttempt).toContain("Started background task");
    expect(secondAttempt).toContain("task=");
    expect(await app.runtime.listBackgroundTasks(session.id)).toHaveLength(1);
  });

  it("runs executable skills from slash commands and can watch runs", async () => {
    const rootDir = await makeTempDir();
    const app = new OperatorCliApp({ rootDir, cwd: rootDir, currentDate: "2026-05-25" });
    const session = await app.runtime.startSession({ title: "CLI skill", cwd: rootDir, agentId: "operator-cli" });
    const recorded = await app.runtime.recordTurn({
      sessionId: session.id,
      userText: "summarize deploy",
      assistantText: "deploy summarized",
      trajectorySummary: "Deploy summary workflow for CLI skill",
    });
    if (!recorded.skillCandidate) {
      throw new Error("expected skill candidate");
    }
    await app.runtime.reviewSkillCandidate(recorded.skillCandidate.id, "reviewed", "reusable");
    const promoted = await app.runtime.promoteSkill(recorded.skillCandidate.id);
    if (!promoted) {
      throw new Error("expected promoted skill");
    }
    const executable = await app.runtime.createExecutableSkillFromPromoted({
      promotedSkillId: promoted.id,
      steps: [{ id: "summary", title: "Summarize", kind: "summary", content: "deploy summary" }],
    });
    if (!executable) {
      throw new Error("expected executable skill");
    }

    const output = await app.dispatchSlashCommand({ kind: "run-skill", skillId: executable.id }, session.id);
    expect(output).toContain("completed");
    expect(output).toContain("deploy summary");

    const run = await app.runtime.startRun({ sessionId: session.id, title: "Watchable run" });
    app.runtime.publishRunProgress({
      runId: run.id,
      sessionId: session.id,
      phase: "planning",
      message: "Watching this run.",
    });
    const watchOutput = await app.dispatchSlashCommand({ kind: "watch-run", runId: run.id }, session.id);
    expect(watchOutput).toContain(`[run ${run.id}]`);
    expect(watchOutput).toContain("Watching this run.");
  });

  it("supports session lifecycle, transcript, approvals, pairing, config, and prompt commands", async () => {
    const rootDir = await makeTempDir();
    const app = new OperatorCliApp({ rootDir, cwd: rootDir, currentDate: "2026-05-25" });
    const firstSession = await app.runtime.startSession({ title: "first", cwd: rootDir, agentId: "operator-cli" });
    const secondSession = await app.runtime.startSession({ title: "second", cwd: rootDir, agentId: "operator-cli" });

    await app.runtime.recordTurn({
      sessionId: secondSession.id,
      userText: "hello",
      assistantText: "world",
      trajectorySummary: "Second session transcript",
    });
    const approval = await app.runtime.promptApproval({
      sessionId: secondSession.id,
      title: "Run command",
      summary: "Allow command execution",
      scope: "shell",
    });

    const sessionsOutput = await app.dispatchSlashCommand({ kind: "sessions" }, firstSession.id);
    expect(sessionsOutput).toContain(firstSession.id);
    expect(sessionsOutput).toContain(secondSession.id);

    const resumeOutput = await app.dispatchSlashCommand({ kind: "resume", sessionId: secondSession.id }, firstSession.id);
    expect(resumeOutput).toContain(secondSession.id);

    const transcriptOutput = await app.dispatchSlashCommand({ kind: "transcript", limit: 2 });
    expect(transcriptOutput).toContain("user: hello");
    expect(transcriptOutput).toContain("assistant: world");

    const approvalsOutput = await app.dispatchSlashCommand({ kind: "approvals" });
    expect(approvalsOutput).toContain(approval.id);
    expect(approvalsOutput).toContain("Allow command execution");

    const pairingCreateOutput = await app.dispatchSlashCommand({ kind: "pairing-create", remoteSource: "gateway" });
    expect(pairingCreateOutput).toContain("Created pairing ticket");
    expect(pairingCreateOutput).toContain("source=gateway");

    const pairingListOutput = await app.dispatchSlashCommand({ kind: "pairing-list" });
    expect(pairingListOutput).toContain("pending");
    expect(pairingListOutput).toContain("source=gateway");
    const pairingCodeMatch = pairingListOutput.match(/[A-Z0-9]{4}-[A-Z0-9]{4}/);
    if (!pairingCodeMatch) {
      throw new Error("expected pairing code");
    }

    const pairingApproveOutput = await app.dispatchSlashCommand({ kind: "pairing-approve", identifier: pairingCodeMatch[0] });
    expect(pairingApproveOutput).toContain(`Approved pairing ticket ${pairingCodeMatch[0]}`);
    expect(await app.dispatchSlashCommand({ kind: "pairing-list", status: "approved" })).toContain(pairingCodeMatch[0]);

    const pairedBootstrap = await app.server.handle({
      method: "sessions.bootstrap",
      params: { pairingCode: pairingCodeMatch[0], agentId: "gateway" },
    });
    if (!pairedBootstrap.ok) {
      throw new Error("expected paired bootstrap");
    }
    const secondGatewayBootstrap = await app.server.handle({
      method: "sessions.bootstrap",
      params: { title: "Second gateway remote", remoteId: "device-cli-2", remoteSource: "gateway", agentId: "gateway" },
    });
    if (!secondGatewayBootstrap.ok) {
      throw new Error("expected second gateway bootstrap");
    }
    await app.runtime.promptApproval({
      sessionId: pairedBootstrap.result.session.id,
      title: "Remote approval",
      summary: "Allow remote action",
    });
    const remoteRun = await app.runtime.startRun({ sessionId: pairedBootstrap.result.session.id, title: "Remote run" });
    const remoteTask = await app.runtime.startBackgroundTask({
      sessionId: pairedBootstrap.result.session.id,
      title: "Remote task",
      command: "sleep 5",
      kind: "task",
    });
    const degradedBootstrap = await app.server.handle({
      method: "sessions.bootstrap",
      params: { title: "Degraded remote", remoteId: "device-cli-drift-1", remoteSource: "gateway", agentId: "gateway" },
    });
    if (!degradedBootstrap.ok) {
      throw new Error("expected degraded bootstrap");
    }
    const degradedTask = await app.runtime.startBackgroundTask({
      sessionId: degradedBootstrap.result.session.id,
      title: "Degraded remote task",
      command: "printf drift",
      kind: "task",
    });
    const remoteListOutput = await app.dispatchSlashCommand({ kind: "remote-list" });
    expect(remoteListOutput).toContain(`remote=${pairedBootstrap.result.session.metadata.remoteId}`);
    expect(remoteListOutput).toContain("source=gateway");
    expect(remoteListOutput).toContain(`pairing=redeemed:${pairingCodeMatch[0]}`);
    expect(remoteListOutput).toContain(`session=${pairedBootstrap.result.session.id}:active`);
    expect(remoteListOutput).toContain("run=running:Remote run");
    expect(remoteListOutput).toContain("task=running:Remote task");
    expect(remoteListOutput).toContain("gateway=<none>");

    const remoteListFilteredOutput = await app.dispatchSlashCommand({ kind: "remote-list", remoteSource: "gateway" });
    expect(remoteListFilteredOutput).toContain(`remote=${pairedBootstrap.result.session.metadata.remoteId}`);

    const platformListOutput = await app.dispatchSlashCommand({ kind: "platform-list" });
    expect(platformListOutput).toContain("platform=gateway");
    expect(platformListOutput).toContain("remotes=3");
    expect(platformListOutput).toContain("control=active");

    const platformStatusOutput = await app.dispatchSlashCommand({ kind: "platform-status", platform: "gateway" });
    expect(platformStatusOutput).toContain("platform=gateway remotes=3");
    expect(platformStatusOutput).toContain(`control=active`);
    expect(platformStatusOutput).toContain(`${pairedBootstrap.result.session.metadata.remoteId} control=active`);
    expect(platformStatusOutput).toContain(`device-cli-2 control=active`);
    expect(platformStatusOutput).toContain(`device-cli-drift-1 control=active`);

    const remoteStatusOutput = await app.dispatchSlashCommand({ kind: "remote-status", identifier: pairedBootstrap.result.session.metadata.remoteId ?? pairedBootstrap.result.session.id });
    expect(remoteStatusOutput).toContain(`remote=${pairedBootstrap.result.session.metadata.remoteId}`);
    expect(remoteStatusOutput).toContain(`session=${pairedBootstrap.result.session.id}`);
    expect(remoteStatusOutput).toContain("control=active");
    expect(remoteStatusOutput).toContain(`pairing=${pairingCodeMatch[0]} redeemed`);
    expect(remoteStatusOutput).toContain("approvals=");
    expect(remoteStatusOutput).toContain("Remote approval");
    expect(remoteStatusOutput).toContain("run=");
    expect(remoteStatusOutput).toContain("Remote run");
    expect(remoteStatusOutput).toContain("task=");
    expect(remoteStatusOutput).toContain("Remote task");

    await app.server.updateGatewaySessionHealth({
      connectionId: "cli-gateway-1",
      sessionId: pairedBootstrap.result.session.id,
      remoteId: pairedBootstrap.result.session.metadata.remoteId,
      state: "stale",
      updatedAt: Date.now(),
      lastClientActivityAt: Date.now(),
    });
    const quarantinedRemoteStatusOutput = await app.dispatchSlashCommand({ kind: "remote-status", identifier: pairedBootstrap.result.session.metadata.remoteId ?? pairedBootstrap.result.session.id });
    expect(quarantinedRemoteStatusOutput).toContain("control=quarantined reason=gateway heartbeat stale");
    expect(quarantinedRemoteStatusOutput).toContain(`action=Run /remote resume ${pairedBootstrap.result.session.metadata.remoteId}`);

    await app.server.updateGatewaySessionHealth({
      connectionId: "cli-gateway-1",
      sessionId: pairedBootstrap.result.session.id,
      remoteId: pairedBootstrap.result.session.metadata.remoteId,
      state: "healthy",
      updatedAt: Date.now(),
      lastClientActivityAt: Date.now(),
    });
    const quarantinedRemoteListOutput = await app.dispatchSlashCommand({ kind: "remote-list" });
    expect(quarantinedRemoteListOutput).toContain(`remote=${pairedBootstrap.result.session.metadata.remoteId}`);
    expect(quarantinedRemoteListOutput).toContain("control=quarantined:gateway heartbeat stale");
    expect(quarantinedRemoteListOutput).toContain(`action=Run /remote resume ${pairedBootstrap.result.session.metadata.remoteId}`);

    const remoteResumeOutput = await app.dispatchSlashCommand({
      kind: "remote-resume",
      identifier: pairedBootstrap.result.session.metadata.remoteId ?? pairedBootstrap.result.session.id,
    });
    expect(remoteResumeOutput).toContain(`Resumed remote ${pairedBootstrap.result.session.metadata.remoteId}`);
    expect(remoteResumeOutput).toContain("control=active");

    const platformPauseOutput = await app.dispatchSlashCommand({
      kind: "platform-pause",
      platform: "gateway",
      reason: "platform unhealthy",
    });
    expect(platformPauseOutput).toContain("Paused platform gateway");
    expect(platformPauseOutput).toContain("control=paused reason=platform unhealthy");
    expect(platformPauseOutput).toContain(`${pairedBootstrap.result.session.metadata.remoteId}:paused:platform unhealthy`);
    expect(platformPauseOutput).toContain("device-cli-2:error:no active run for remote or session: device-cli-2");
    expect(platformPauseOutput).toContain("device-cli-drift-1:error:no active run for remote or session: device-cli-drift-1");

    const platformListAfterPause = await app.dispatchSlashCommand({ kind: "platform-list" });
    expect(platformListAfterPause).toContain("platform=gateway remotes=3 control=paused:platform unhealthy");

    const platformStatusAfterPause = await app.dispatchSlashCommand({ kind: "platform-status", platform: "gateway" });
    expect(platformStatusAfterPause).toContain("control=paused reason=platform unhealthy");
    expect(platformStatusAfterPause).toContain(`${pairedBootstrap.result.session.metadata.remoteId} control=paused:platform unhealthy`);

    const remotePauseOutput = await app.dispatchSlashCommand({
      kind: "remote-pause",
      identifier: pairedBootstrap.result.session.metadata.remoteId ?? pairedBootstrap.result.session.id,
      reason: "gateway unhealthy",
    });
    expect(remotePauseOutput).toContain(`Paused remote ${pairedBootstrap.result.session.metadata.remoteId}`);
    expect(remotePauseOutput).toContain("control=paused");
    expect(remotePauseOutput).toContain("reason=gateway unhealthy");

    const pausedRemoteStatusOutput = await app.dispatchSlashCommand({ kind: "remote-status", identifier: pairedBootstrap.result.session.metadata.remoteId ?? pairedBootstrap.result.session.id });
    expect(pausedRemoteStatusOutput).toContain("control=paused reason=gateway unhealthy");
    expect(pausedRemoteStatusOutput).toContain(`run=${remoteRun.id} paused Remote run`);

    await app.runtime.backgroundTasks.executionService.writeState(degradedTask, {
      version: 1,
      taskId: degradedTask.id,
      kind: "task",
      status: "running",
      pid: 999999,
      startedAt: degradedTask.execution.startedAt ?? degradedTask.updatedAt,
      updatedAt: degradedTask.updatedAt,
      outputFile: degradedTask.execution.outputFile,
      cwd: degradedTask.cwd,
      command: degradedTask.command,
    });
    await app.runtime.syncBackgroundTask(degradedTask.id);

    const degradedRemoteListOutput = await app.dispatchSlashCommand({
      kind: "remote-list",
      remoteSource: "gateway",
    });
    expect(degradedRemoteListOutput).toContain(`remote=${degradedBootstrap.result.session.metadata.remoteId}`);
    expect(degradedRemoteListOutput).toContain("control=degraded:background task failed");
    expect(degradedRemoteListOutput).toContain("diagnostics=background task failed");

    const degradedRemoteStatusOutput = await app.dispatchSlashCommand({
      kind: "remote-status",
      identifier: degradedBootstrap.result.session.metadata.remoteId ?? degradedBootstrap.result.session.id,
    });
    expect(degradedRemoteStatusOutput).toContain("control=degraded reason=background task failed");
    expect(degradedRemoteStatusOutput).toContain(`diagnostics cause=background task failed recoverable=false task=${degradedTask.id}`);

    const remoteRepairOutput = await app.dispatchSlashCommand({
      kind: "remote-repair",
      identifier: degradedBootstrap.result.session.metadata.remoteId ?? degradedBootstrap.result.session.id,
      reasons: ["missing-process"],
    });
    expect(remoteRepairOutput).toContain(`Repaired remote ${degradedBootstrap.result.session.metadata.remoteId}`);
    expect(remoteRepairOutput).toContain("changed=false");
    expect(remoteRepairOutput).toContain("control=active");
    expect(remoteRepairOutput).toContain("tasks=<none>");

    const mixedPlatformStatusOutput = await app.dispatchSlashCommand({ kind: "platform-status", platform: "gateway" });
    expect(mixedPlatformStatusOutput).toContain("control=paused reason=platform unhealthy");
    expect(mixedPlatformStatusOutput).toContain("diagnostics=background task failed");

    const platformResumeOutput = await app.dispatchSlashCommand({ kind: "platform-resume", platform: "gateway" });
    expect(platformResumeOutput).toContain("Resumed platform gateway");
    expect(platformResumeOutput).toContain("control=mixed");
    expect(platformResumeOutput).toContain(`${pairedBootstrap.result.session.metadata.remoteId}:paused:gateway unhealthy`);
    expect(platformResumeOutput).toContain("device-cli-2:active");
    expect(platformResumeOutput).toContain("device-cli-drift-1:degraded:background task failed");

    const approveOutput = await app.dispatchSlashCommand({ kind: "approve", approvalId: approval.id });
    expect(approveOutput).toContain(`Approved ${approval.id}`);
    const approvalsAfterLocalApprove = await app.dispatchSlashCommand({ kind: "approvals" });
    expect(approvalsAfterLocalApprove).toContain("Remote approval");
    expect(approvalsAfterLocalApprove).not.toContain(approval.id);

    const idleOutput = await app.dispatchSlashCommand({ kind: "idle" });
    expect(idleOutput).toContain(`Marked session ${secondSession.id} idle.`);

    const statusOutput = await app.dispatchSlashCommand({ kind: "status" });
    expect(statusOutput).toContain(`session=${secondSession.id}`);
    expect(statusOutput).toContain("status=idle");

    const configOutput = await app.dispatchSlashCommand({ kind: "config" });
    expect(configOutput).toContain("Loaded config files");
    expect(configOutput).toContain("permissionMode=default");
    expect(configOutput).toContain("hooks.PreToolUse=0");
    expect(configOutput).toContain("hooks.PreCommand=0");

    const promptOutput = await app.dispatchSlashCommand({ kind: "prompt" });
    expect(promptOutput).toContain(`cwd=${rootDir}`);
    expect(promptOutput).toContain("instructionFiles=");
  });

  it("supports background task watch and cron commands", async () => {
    const rootDir = await makeTempDir();
    const app = new OperatorCliApp({ rootDir, cwd: rootDir, currentDate: "2026-05-25" });
    const session = await app.runtime.startSession({ title: "CLI ops", cwd: rootDir, agentId: "operator-cli" });

    const startOutput = await app.dispatchSlashCommand(
      { kind: "background-start", title: "smoke", command: "printf ok" },
      session.id,
    );
    expect(startOutput).toContain("Started background task");

    const tasks = await app.runtime.listBackgroundTasks(session.id);
    expect(tasks).toHaveLength(1);
    const [task] = tasks;
    if (!task) {
      throw new Error("expected background task");
    }

    const listOutput = await app.dispatchSlashCommand({ kind: "background-list" });
    expect(listOutput).toContain(task.id);
    expect(listOutput).toContain("smoke");

    const syncOutput = await app.dispatchSlashCommand({ kind: "background-sync", taskId: task.id });
    expect(syncOutput).toContain(task.id);

    const viewOutput = await app.dispatchSlashCommand({ kind: "background-view", taskId: task.id, lines: 20 });
    expect(viewOutput).toContain(task.id);
    expect(viewOutput).toContain("ok");

    const watchTaskOutput = await app.dispatchSlashCommand({ kind: "watch-task", taskId: task.id }, session.id);
    expect(watchTaskOutput).toContain(`[task ${task.id}]`);
    expect(watchTaskOutput).toContain("ok");

    const activeWatchOutput = await app.dispatchSlashCommand({ kind: "watch-active" }, session.id);
    expect(activeWatchOutput).toContain(`[task ${task.id}]`);

    const cronCreate = await app.dispatchSlashCommand(
      {
        kind: "cron-create",
        cron: "* * * * *",
        prompt: "check deploy",
        delivery: { onSuccess: ["local"], onFailure: ["https://example.test/hook"] },
      },
      session.id,
    );
    expect(cronCreate).toContain("Created cron job");

    const cronList = await app.dispatchSlashCommand({ kind: "cron-list" }, session.id);
    expect(cronList).toContain("check deploy");
    expect(cronList).toContain("status=<none>");
    expect(cronList).toContain("targets=success:local,failure:https://example.test/hook");

    const cronJobs = await app.server.handle({ method: "cron.list" });
    if (!cronJobs.ok || cronJobs.result.length === 0) {
      throw new Error("expected cron job");
    }
    const [job] = cronJobs.result;
    if (!job) {
      throw new Error("expected first cron job");
    }

    await app.server.handle({ method: "cron.tick", params: { nowMs: Date.now() + 120_000 } });

    const cronRuns = await app.dispatchSlashCommand({ kind: "cron-runs", jobId: job.id }, session.id);
    expect(cronRuns).toContain(`job=${job.id}`);
    expect(cronRuns).toContain("outcome=success");
    expect(cronRuns).toContain("session=");
    expect(cronRuns).toContain("run=");
    expect(cronRuns).toContain("delivery=local:sent");

    const cronDelete = await app.dispatchSlashCommand({ kind: "cron-delete", jobId: job.id }, session.id);
    expect(cronDelete).toContain(`Deleted cron job ${job.id}`);
  });

  it("gates dangerous background commands and allows rerun after approval", async () => {
    const rootDir = await makeTempDir();
    await fs.mkdir(path.join(rootDir, ".claude"), { recursive: true });
    await fs.writeFile(path.join(rootDir, ".claude", "settings.local.json"), '{"permissionMode":"default"}');
    const app = new OperatorCliApp({ rootDir, cwd: rootDir, currentDate: "2026-05-25" });
    const session = await app.runtime.startSession({ title: "policy", cwd: rootDir, agentId: "operator-cli" });

    const firstAttempt = await app.dispatchSlashCommand(
      { kind: "background-start", title: "wipe", command: "rm -rf build" },
      session.id,
    );
    expect(firstAttempt).toContain("Approval required:");
    expect(await app.runtime.listBackgroundTasks(session.id)).toEqual([]);

    const approvals = await app.runtime.listApprovals();
    expect(approvals).toHaveLength(1);
    const [approval] = approvals;
    if (!approval) {
      throw new Error("expected approval");
    }

    const approved = await app.dispatchSlashCommand({ kind: "approve", approvalId: approval.id });
    expect(approved).toContain(`Approved ${approval.id}`);

    const secondAttempt = await app.dispatchSlashCommand(
      { kind: "background-start", title: "wipe", command: "rm -rf build" },
      session.id,
    );
    expect(secondAttempt).toContain("Started background task");
    expect(await app.runtime.listBackgroundTasks(session.id)).toHaveLength(1);
  });

  it("runs configured hooks and reports policy config", async () => {
    const rootDir = await makeTempDir();
    await fs.mkdir(path.join(rootDir, ".claude"), { recursive: true });
    await fs.writeFile(
      path.join(rootDir, ".claude", "settings.local.json"),
      JSON.stringify({
        permissionMode: "default",
        hooks: {
          PreToolUse: ["printf '{\"updatedInput\":{\"command\":\"printf hooked\",\"title\":\"hooked title\"}}'"],
          PreCommand: ["printf 'pre:%s' \"$OPERATOR_ACTION_KIND\""],
          PostCommand: ["printf 'post:%s' \"$OPERATOR_HOOK_EVENT\""],
          SessionStart: ["printf unsupported"],
        },
      }),
    );
    const app = new OperatorCliApp({ rootDir, cwd: rootDir, currentDate: "2026-05-25" });
    const session = await app.runtime.startSession({ title: "hooks", cwd: rootDir, agentId: "operator-cli" });

    const output = await app.dispatchSlashCommand(
      { kind: "background-start", title: "smoke", command: "printf ok" },
      session.id,
    );
    expect(output).toContain("Started background task");
    expect(output).toContain("hooked title");
    expect(output).toContain("[PostCommand] post:PostCommand");
    const tasks = await app.runtime.listBackgroundTasks(session.id);
    expect(tasks[0]).toMatchObject({ title: "hooked title", command: "printf hooked" });

    const configOutput = await app.dispatchSlashCommand({ kind: "config" }, session.id);
    expect(configOutput).toContain("permissionMode=default");
    expect(configOutput).toContain("hooks.PreToolUse=1");
    expect(configOutput).toContain("hooks.PreCommand=1");
    expect(configOutput).toContain("hooks.PostCommand=1");
    expect(configOutput).toContain("hooks.SessionStart=1");
    expect(configOutput).toContain("unsupportedHooks=<none>");
  });

  it("gates dangerous executable skill command steps", async () => {
    const rootDir = await makeTempDir();
    await fs.mkdir(path.join(rootDir, ".claude"), { recursive: true });
    await fs.writeFile(path.join(rootDir, ".claude", "settings.local.json"), '{"permissionMode":"default"}');
    const app = new OperatorCliApp({ rootDir, cwd: rootDir, currentDate: "2026-05-25" });
    const session = await app.runtime.startSession({ title: "skill policy", cwd: rootDir, agentId: "operator-cli" });
    const recorded = await app.runtime.recordTurn({
      sessionId: session.id,
      userText: "dangerous skill",
      assistantText: "recorded",
      trajectorySummary: "Dangerous skill workflow for CLI",
    });
    if (!recorded.skillCandidate) {
      throw new Error("expected skill candidate");
    }
    await app.runtime.reviewSkillCandidate(recorded.skillCandidate.id, "reviewed", "reusable");
    const promoted = await app.runtime.promoteSkill(recorded.skillCandidate.id);
    if (!promoted) {
      throw new Error("expected promoted skill");
    }
    const executable = await app.runtime.createExecutableSkillFromPromoted({
      promotedSkillId: promoted.id,
      steps: [{ id: "danger", title: "Danger", kind: "command", command: "rm -rf build" }],
    });
    if (!executable) {
      throw new Error("expected executable skill");
    }

    const output = await app.dispatchSlashCommand({ kind: "run-skill", skillId: executable.id }, session.id);
    expect(output).toContain("Approval required:");
    expect((await app.runtime.listApprovals()).length).toBeGreaterThan(0);
  });
});
