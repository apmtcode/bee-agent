import { execFile } from "node:child_process";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { promisify } from "node:util";
import { afterEach, describe, expect, it } from "vitest";
import { OperatorCliApp, parseSlashCommand } from "./app.js";

const tempDirs: string[] = [];
const execFileAsync = promisify(execFile);

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
    expect(parseSlashCommand("/tasks team reviewers")).toEqual({ kind: "tasks", team: "reviewers" });
    expect(parseSlashCommand("/task-create Inspect logs")).toEqual({ kind: "task-create", subject: "Inspect logs" });
    expect(parseSlashCommand("/task-create Inspect logs --team reviewers --owner worker --blocked-by task-1,task-2")).toEqual({
      kind: "task-create",
      subject: "Inspect logs",
      team: "reviewers",
      owner: "worker",
      blockedBy: ["task-1", "task-2"],
    });
    expect(parseSlashCommand("/task-update task-1 in_progress")).toEqual({ kind: "task-update", taskId: "task-1", status: "in_progress" });
    expect(parseSlashCommand("/task-update task-1 --owner worker --blocked-by task-2")).toEqual({
      kind: "task-update",
      taskId: "task-1",
      owner: "worker",
      blockedBy: ["task-2"],
    });
    expect(parseSlashCommand("/task-stop task-1")).toEqual({ kind: "task-stop", taskId: "task-1" });
    expect(parseSlashCommand("/messages")).toEqual({ kind: "messages" });
    expect(parseSlashCommand("/inbox")).toEqual({ kind: "inbox" });
    expect(parseSlashCommand("/outbox")).toEqual({ kind: "outbox" });
    expect(parseSlashCommand("/send session-2 Please collect logs")).toEqual({ kind: "send", toSessionId: "session-2", message: "Please collect logs" });
    expect(parseSlashCommand("/notify success Deploy finished")).toEqual({ kind: "notify", status: "success", message: "Deploy finished" });
    expect(parseSlashCommand("/plan-show")).toEqual({ kind: "plan-show" });
    expect(parseSlashCommand("/plan-set 1. Draft plan")).toEqual({ kind: "plan-set", content: "1. Draft plan" });
    expect(parseSlashCommand("/plan-request-approval session-2")).toEqual({ kind: "plan-request-approval", approverSessionId: "session-2" });
    expect(parseSlashCommand("/plan-respond req-1 approve looks good")).toEqual({ kind: "plan-respond", requestId: "req-1", approve: true, feedback: "looks good" });
    expect(parseSlashCommand("/verify requested rerun tests")).toEqual({ kind: "verify", status: "requested", notes: "rerun tests" });
    expect(parseSlashCommand("/plan")).toEqual({ kind: "plan" });
    expect(parseSlashCommand("/plan-exit")).toEqual({ kind: "plan-exit" });
    expect(parseSlashCommand("/plan-exit acceptEdits")).toEqual({ kind: "plan-exit", mode: "acceptEdits" });
    expect(parseSlashCommand("/statusline")).toEqual({ kind: "statusline" });
    expect(parseSlashCommand("/statusline node .claude/statusline.js")).toEqual({ kind: "statusline", command: "node .claude/statusline.js" });
    expect(parseSlashCommand("/worktree")).toEqual({ kind: "worktree" });
    expect(parseSlashCommand("/worktree feature-x")).toEqual({ kind: "worktree", name: "feature-x" });
    expect(parseSlashCommand("/worktree-exit")).toEqual({ kind: "worktree-exit", action: "keep" });
    expect(parseSlashCommand("/worktree-exit remove")).toEqual({ kind: "worktree-exit", action: "remove" });
    expect(parseSlashCommand("/teams")).toEqual({ kind: "teams" });
    expect(parseSlashCommand("/team-create reviewers Review team")).toEqual({ kind: "team-create", name: "reviewers", description: "Review team" });
    expect(parseSlashCommand("/team-delete reviewers")).toEqual({ kind: "team-delete", name: "reviewers" });
    expect(parseSlashCommand("/teammates reviewers")).toEqual({ kind: "teammates", team: "reviewers" });
    expect(parseSlashCommand("/teammate-start reviewers alice Review PR --agent reviewer")).toEqual({
      kind: "teammate-start",
      team: "reviewers",
      name: "alice",
      title: "Review PR",
      agentId: "reviewer",
    });
    expect(parseSlashCommand("/teammate-message reviewers alice Please inspect logs")).toEqual({
      kind: "teammate-message",
      team: "reviewers",
      teammate: "alice",
      message: "Please inspect logs",
    });
    expect(parseSlashCommand("/teammate-update reviewers alice completed")).toEqual({
      kind: "teammate-update",
      team: "reviewers",
      teammate: "alice",
      status: "completed",
    });
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
    expect(parseSlashCommand("/tasks later")).toEqual({
      kind: "invalid",
      message: "Usage: /tasks [team <name>]",
    });
    expect(parseSlashCommand("/task-create")).toEqual({
      kind: "invalid",
      message: "Usage: /task-create <subject> [--team <name>] [--owner <owner>] [--blocked-by <taskId,...>]",
    });
    expect(parseSlashCommand("/task-create --team reviewers")).toEqual({
      kind: "invalid",
      message: "Usage: /task-create <subject> [--team <name>] [--owner <owner>] [--blocked-by <taskId,...>]",
    });
    expect(parseSlashCommand("/task-update")).toEqual({
      kind: "invalid",
      message: "Usage: /task-update <taskId> [<pending|in_progress|completed>] [--owner <owner|off>] [--blocked-by <taskId,...>]",
    });
    expect(parseSlashCommand("/task-update task-1 blocked")).toEqual({
      kind: "invalid",
      message: "Usage: /task-update <taskId> [<pending|in_progress|completed>] [--owner <owner|off>] [--blocked-by <taskId,...>]",
    });
    expect(parseSlashCommand("/task-stop")).toEqual({
      kind: "invalid",
      message: "Usage: /task-stop <taskId>",
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
    expect(parseSlashCommand("/notify success")).toEqual({
      kind: "invalid",
      message: "Usage: /notify <status> <message>",
    });
    expect(parseSlashCommand("/plan-show later")).toEqual({
      kind: "invalid",
      message: "Usage: /plan-show",
    });
    expect(parseSlashCommand("/plan-set")).toEqual({
      kind: "invalid",
      message: "Usage: /plan-set <content>",
    });
    expect(parseSlashCommand("/plan-request-approval")).toEqual({
      kind: "invalid",
      message: "Usage: /plan-request-approval <sessionId>",
    });
    expect(parseSlashCommand("/plan-respond req-1 maybe")).toEqual({
      kind: "invalid",
      message: "Usage: /plan-respond <requestId> <approve|reject> [feedback]",
    });
    expect(parseSlashCommand("/verify later")).toEqual({
      kind: "invalid",
      message: "Usage: /verify <pending|requested|completed|skipped> [notes]",
    });
    expect(parseSlashCommand("/plan later")).toEqual({
      kind: "invalid",
      message: "Usage: /plan",
    });
    expect(parseSlashCommand("/plan-exit plan")).toEqual({
      kind: "invalid",
      message: "Usage: /plan-exit [default|acceptEdits|bypassPermissions]",
    });
    expect(parseSlashCommand("/worktree-exit nope")).toEqual({
      kind: "invalid",
      message: "Usage: /worktree-exit [keep|remove]",
    });
    expect(parseSlashCommand("/teams later")).toEqual({
      kind: "invalid",
      message: "Usage: /teams",
    });
    expect(parseSlashCommand("/team-create")).toEqual({
      kind: "invalid",
      message: "Usage: /team-create <name> [description]",
    });
    expect(parseSlashCommand("/team-delete")).toEqual({
      kind: "invalid",
      message: "Usage: /team-delete <name>",
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

    const updatedOutput = await app.dispatchSlashCommand({ kind: "task-update", taskId: task.id, status: "completed", owner: "worker", blockedBy: ["task-2"] }, session.id);
    expect(updatedOutput).toContain(`Updated task ${task.id} completed Inspect logs`);

    const updatedTask = await app.runtime.getTask(task.id);
    expect(updatedTask).toMatchObject({ id: task.id, status: "completed", subject: "Inspect logs", owner: "worker", blockedBy: ["task-2"] });
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

  it("sends operator notifications", async () => {
    const rootDir = await makeTempDir();
    const app = new OperatorCliApp({ rootDir, cwd: rootDir, currentDate: "2026-05-25" });
    const session = await app.runtime.startSession({ title: "notify cli", cwd: rootDir, agentId: "operator-cli" });

    const output = await app.dispatchSlashCommand({ kind: "notify", status: "success", message: "Deploy finished." }, session.id);
    expect(output).toContain("Sent notification");
    expect(output).toContain("status=success");
    expect(output).toContain("delivery=sent");
    expect(output).toContain("message=Deploy finished.");

    const deliveryLog = await fs.readFile(path.join(rootDir, "delivery-local.jsonl"), "utf8");
    expect(deliveryLog).toContain("\"kind\":\"push-notification\"");
    expect(deliveryLog).toContain("Deploy finished.");
  });

  it("shows, updates, requests approval for, responds to, and verifies plans", async () => {
    const rootDir = await makeTempDir();
    const app = new OperatorCliApp({ rootDir, cwd: rootDir, currentDate: "2026-05-25" });
    const author = await app.runtime.startSession({ title: "plan author", cwd: rootDir, agentId: "operator-cli" });
    const reviewer = await app.runtime.startSession({ title: "plan reviewer", cwd: rootDir, agentId: "reviewer" });

    await expect(app.dispatchSlashCommand({ kind: "plan-show" }, author.id)).resolves.toBe(`No plan for session ${author.id}.`);

    const setOutput = await app.dispatchSlashCommand({ kind: "plan-set", content: "1. Add plan store\n2. Add plan RPC" }, author.id);
    expect(setOutput).toContain("Updated plan");
    expect(setOutput).toContain("status=draft");

    const plan = await app.runtime.getPlan(author.id);
    expect(plan).toMatchObject({
      sessionId: author.id,
      content: "1. Add plan store\n2. Add plan RPC",
      status: "draft",
    });
    if (!plan) {
      throw new Error("expected plan");
    }

    const showOutput = await app.dispatchSlashCommand({ kind: "plan-show" }, author.id);
    expect(showOutput).toContain(`plan=${plan.id}`);
    expect(showOutput).toContain("status=draft");
    expect(showOutput).toContain("content=1. Add plan store 2. Add plan RPC");

    const requestOutput = await app.dispatchSlashCommand({ kind: "plan-request-approval", approverSessionId: reviewer.id }, author.id);
    expect(requestOutput).toContain("Requested plan approval");
    expect(requestOutput).toContain(`plan=${plan.id}`);
    expect(requestOutput).toContain(`approver=${reviewer.id}`);

    const requestedPlan = await app.runtime.getPlan(author.id);
    expect(requestedPlan).toMatchObject({
      id: plan.id,
      status: "awaiting_approval",
      approval: { approverSessionId: reviewer.id },
    });
    if (!requestedPlan?.approval?.requestId) {
      throw new Error("expected approval request id");
    }

    const respondOutput = await app.dispatchSlashCommand(
      { kind: "plan-respond", requestId: requestedPlan.approval.requestId, approve: true, feedback: "Looks good" },
      reviewer.id,
    );
    expect(respondOutput).toContain(`Responded to plan approval ${requestedPlan.approval.requestId}`);
    expect(respondOutput).toContain("status=approved");

    const verifyOutput = await app.dispatchSlashCommand({ kind: "verify", status: "requested", notes: "Run focused tests" }, author.id);
    expect(verifyOutput).toContain(`Updated plan verification ${plan.id} status=requested`);
    expect(verifyOutput).toContain("notes=Run focused tests");

    const verifiedPlan = await app.runtime.getPlan(author.id);
    expect(verifiedPlan).toMatchObject({
      id: plan.id,
      status: "approved",
      verification: { status: "requested", notes: "Run focused tests" },
    });
  });

  it("enables and exits plan mode with persisted local config", async () => {
    const rootDir = await makeTempDir();
    await fs.mkdir(path.join(rootDir, ".claude"), { recursive: true });
    await fs.writeFile(path.join(rootDir, ".claude", "settings.local.json"), '{"permissionMode":"acceptEdits"}');
    const app = new OperatorCliApp({ rootDir, cwd: rootDir, currentDate: "2026-05-25" });
    const session = await app.runtime.startSession({ title: "plan cli", cwd: rootDir, agentId: "operator-cli" });

    const enabled = await app.dispatchSlashCommand({ kind: "plan" }, session.id);
    expect(enabled).toContain("Enabled plan mode");
    expect(enabled).toContain("previous=acceptEdits");

    await expect(app.dispatchSlashCommand({ kind: "plan" }, session.id)).resolves.toContain("Plan mode already enabled");

    const enabledSettings = JSON.parse(await fs.readFile(path.join(rootDir, ".claude", "settings.local.json"), "utf8")) as Record<string, unknown>;
    expect(enabledSettings).toMatchObject({
      permissionMode: "plan",
      operator: { previousPermissionMode: "acceptEdits" },
    });

    await expect(app.dispatchSlashCommand({ kind: "config" }, session.id)).resolves.toContain("permissionMode=plan");

    const exited = await app.dispatchSlashCommand({ kind: "plan-exit" }, session.id);
    expect(exited).toContain("Disabled plan mode permissionMode=acceptEdits");

    const exitedSettings = JSON.parse(await fs.readFile(path.join(rootDir, ".claude", "settings.local.json"), "utf8")) as Record<string, unknown>;
    expect(exitedSettings).toMatchObject({ permissionMode: "acceptEdits" });
    expect(exitedSettings).not.toHaveProperty("operator");
    await expect(app.dispatchSlashCommand({ kind: "config" }, session.id)).resolves.toContain("permissionMode=acceptEdits");
  });

  it("overrides plan-exit restore mode and reports when plan mode is inactive", async () => {
    const rootDir = await makeTempDir();
    const app = new OperatorCliApp({ rootDir, cwd: rootDir, currentDate: "2026-05-25" });
    const session = await app.runtime.startSession({ title: "plan exit cli", cwd: rootDir, agentId: "operator-cli" });

    await expect(app.dispatchSlashCommand({ kind: "plan-exit" }, session.id)).resolves.toBe("Plan mode is not enabled. permissionMode=default");

    await expect(app.dispatchSlashCommand({ kind: "plan" }, session.id)).resolves.toContain("Enabled plan mode");
    const blocked = await app.dispatchSlashCommand({ kind: "background-start", title: "wipe", command: "printf ok" }, session.id);
    expect(blocked).toContain("Execution is disabled while permissionMode=plan.");

    const exited = await app.dispatchSlashCommand({ kind: "plan-exit", mode: "bypassPermissions" }, session.id);
    expect(exited).toContain("Disabled plan mode permissionMode=bypassPermissions");

    const settings = JSON.parse(await fs.readFile(path.join(rootDir, ".claude", "settings.local.json"), "utf8")) as Record<string, unknown>;
    expect(settings).toMatchObject({ permissionMode: "bypassPermissions" });
  });

  it("shows, configures, and disables the status line command", async () => {
    const rootDir = await makeTempDir();
    const app = new OperatorCliApp({ rootDir, cwd: rootDir, currentDate: "2026-05-25" });
    const session = await app.runtime.startSession({ title: "statusline cli", cwd: rootDir, agentId: "operator-cli" });

    await expect(app.dispatchSlashCommand({ kind: "statusline" }, session.id)).resolves.toBe("statusLine disabled");

    const configured = await app.dispatchSlashCommand({ kind: "statusline", command: "node .claude/statusline.js" }, session.id);
    expect(configured).toContain("Configured status line command=node .claude/statusline.js");

    await expect(app.dispatchSlashCommand({ kind: "statusline" }, session.id)).resolves.toBe("statusLine command=node .claude/statusline.js");

    const settingsText = await fs.readFile(path.join(rootDir, ".claude", "settings.local.json"), "utf8");
    expect(settingsText).toContain("statusLine");
    expect(settingsText).toContain("node .claude/statusline.js");

    const disabled = await app.dispatchSlashCommand({ kind: "statusline", command: "off" }, session.id);
    expect(disabled).toContain("Disabled status line");

    const settings = JSON.parse(await fs.readFile(path.join(rootDir, ".claude", "settings.local.json"), "utf8")) as Record<string, unknown>;
    expect(settings).not.toHaveProperty("statusLine");
  });

  it("enters keeps and removes git worktrees", async () => {
    const rootDir = await makeTempDir();
    await execFileAsync("git", ["init", "-b", "main"], { cwd: rootDir });
    await fs.writeFile(path.join(rootDir, "tracked.txt"), "hello\n");
    await execFileAsync("git", ["add", "tracked.txt"], { cwd: rootDir });
    await execFileAsync("git", ["-c", "user.name=Operator Test", "-c", "user.email=operator@example.test", "commit", "-m", "init"], { cwd: rootDir });

    const app = new OperatorCliApp({ rootDir, cwd: rootDir, currentDate: "2026-05-25" });
    const session = await app.runtime.startSession({ title: "worktree cli", cwd: rootDir, agentId: "operator-cli" });

    const nonRepoRoot = await makeTempDir();
    const notRepoApp = new OperatorCliApp({ rootDir: nonRepoRoot, cwd: nonRepoRoot, currentDate: "2026-05-25" });
    await expect(notRepoApp.dispatchSlashCommand({ kind: "worktree" }, session.id)).resolves.toContain("Not a git repository");

    const entered = await app.dispatchSlashCommand({ kind: "worktree", name: "feature-x" }, session.id);
    expect(entered).toContain("Entered worktree");
    expect(entered).toContain("branch=operator/feature-x");

    const status = await app.dispatchSlashCommand({ kind: "status" }, session.id);
    expect(status).toContain("cwd=");
    expect(status).toContain(`${path.sep}.operator${path.sep}worktrees${path.sep}feature-x`);

    const kept = await app.dispatchSlashCommand({ kind: "worktree-exit", action: "keep" }, session.id);
    expect(kept).toContain("kept branch=operator/feature-x");
    await expect(fs.stat(path.join(rootDir, ".operator", "worktrees", "feature-x"))).resolves.toBeDefined();

    const reentered = await app.dispatchSlashCommand({ kind: "worktree", name: "feature-y" }, session.id);
    expect(reentered).toContain("branch=operator/feature-y");

    const removed = await app.dispatchSlashCommand({ kind: "worktree-exit", action: "remove" }, session.id);
    expect(removed).toContain("Exited and removed worktree");
    await expect(fs.stat(path.join(rootDir, ".operator", "worktrees", "feature-y"))).rejects.toMatchObject({ code: "ENOENT" });
  });

  it("creates team task sessions and coordinates team-scoped tasks", async () => {
    const rootDir = await makeTempDir();
    const app = new OperatorCliApp({ rootDir, cwd: rootDir, currentDate: "2026-05-25" });
    const session = await app.runtime.startSession({ title: "team task cli", cwd: rootDir, agentId: "operator-cli" });

    await expect(app.dispatchSlashCommand({ kind: "teams" })).resolves.toBe("No teams.");

    const created = await app.dispatchSlashCommand({ kind: "team-create", name: "reviewers", description: "Review team" }, session.id);
    expect(created).toContain("Created team reviewers");
    expect(created).toContain("description=Review team");
    expect(created).toContain("taskSession=");

    const listed = await app.dispatchSlashCommand({ kind: "teams" }, session.id);
    expect(listed).toContain("reviewers Review team taskSession=");

    const teamTask = await app.dispatchSlashCommand({ kind: "task-create", subject: "Inspect PR", team: "reviewers", owner: "alice", blockedBy: ["task-0"] }, session.id);
    expect(teamTask).toContain("Created task");

    const listedTasks = await app.dispatchSlashCommand({ kind: "tasks", team: "reviewers" }, session.id);
    expect(listedTasks).toContain("Inspect PR owner=alice blockedBy=task-0");

    const team = await app.teams.get("reviewers");
    if (!team?.taskSessionId) {
      throw new Error("expected team task session");
    }
    const tasks = await app.runtime.listTasks(team.taskSessionId);
    expect(tasks).toHaveLength(1);

    const deleted = await app.dispatchSlashCommand({ kind: "team-delete", name: "reviewers" }, session.id);
    expect(deleted).toContain("Deleted team reviewers.");
    await expect(app.dispatchSlashCommand({ kind: "teams" }, session.id)).resolves.toBe("No teams.");
  });

  it("starts lists messages and updates teammates", async () => {
    const rootDir = await makeTempDir();
    const app = new OperatorCliApp({ rootDir, cwd: rootDir, currentDate: "2026-05-25" });
    const session = await app.runtime.startSession({ title: "teammate cli", cwd: rootDir, agentId: "operator-cli" });

    await expect(app.dispatchSlashCommand({ kind: "team-create", name: "reviewers", description: "Review team" }, session.id)).resolves.toContain("Created team reviewers");
    await app.runtime.startRun({ sessionId: session.id, title: "Review coordinator" });

    const startedOutput = await app.dispatchSlashCommand(
      { kind: "teammate-start", team: "reviewers", name: "alice", title: "Review PR", agentId: "reviewer" },
      session.id,
    );
    expect(startedOutput).toContain("Started teammate alice");
    expect(startedOutput).toContain("team=reviewers");
    expect(startedOutput).toContain("run=");
    expect(startedOutput).toContain("taskSession=");

    const teammate = await app.runtime.getTeammate("reviewers", "alice");
    expect(teammate).toMatchObject({
      name: "alice",
      title: "Review PR",
      agentId: "reviewer",
      status: "running",
    });
    if (!teammate) {
      throw new Error("expected teammate");
    }

    const listed = await app.dispatchSlashCommand({ kind: "teammates", team: "reviewers" }, session.id);
    expect(listed).toContain("alice Review PR");
    expect(listed).toContain("team=reviewers");
    expect(listed).toContain("status=running");
    expect(listed).toContain(`session=${teammate.sessionId}`);
    expect(listed).toContain(`subagent=${teammate.subagentRunId}`);
    expect(listed).toContain(`run=${teammate.childRunId}`);
    expect(listed).toContain("agent=reviewer");

    const messageOutput = await app.dispatchSlashCommand(
      { kind: "teammate-message", team: "reviewers", teammate: "alice", message: "Please inspect logs" },
      session.id,
    );
    expect(messageOutput).toContain("Sent teammate message");
    expect(messageOutput).toContain("team=reviewers");
    expect(messageOutput).toContain("teammate=alice");
    expect(messageOutput).toContain(`session=${teammate.sessionId}`);

    const inbox = await app.runtime.listInbox(teammate.sessionId);
    expect(inbox).toHaveLength(1);
    expect(inbox[0]).toMatchObject({
      fromSessionId: session.id,
      toSessionId: teammate.sessionId,
      summary: "Please inspect logs",
    });

    const updatedOutput = await app.dispatchSlashCommand(
      { kind: "teammate-update", team: "reviewers", teammate: "alice", status: "completed" },
      session.id,
    );
    expect(updatedOutput).toContain("Updated teammate alice team=reviewers status=completed");

    const updatedTeammate = await app.runtime.getTeammate("reviewers", "alice");
    expect(updatedTeammate).toMatchObject({ name: "alice", status: "completed" });
    const childRun = await app.runtime.getRun(teammate.childRunId);
    expect(childRun).toMatchObject({ id: teammate.childRunId, status: "completed" });
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

    const stopOutput = await app.dispatchSlashCommand({ kind: "task-stop", taskId: task.id }, session.id);
    expect(stopOutput).toContain(`Stopped task ${task.id}.`);

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
