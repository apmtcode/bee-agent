import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { StandaloneOperatorRuntime } from "./operator-runtime.js";
import { resolveOperatorCliExecutionConfig } from "../cli/config.js";
import { runOperatorHooks } from "../cli/execution-policy.js";
import type { ReviewedExportManifest } from "../training/export-manifest.js";

const tempDirs: string[] = [];

async function makeTempDir(): Promise<string> {
  const dir = await fs.mkdtemp(path.join(os.tmpdir(), "operator-runtime-"));
  tempDirs.push(dir);
  return dir;
}

function buildHookCaptureCommand(outputFile: string): string {
  const script = [
    "import fs from 'node:fs';",
    "const payload = {",
    "  event: process.env.OPERATOR_HOOK_EVENT,",
    "  sessionId: process.env.OPERATOR_SESSION_ID,",
    "  approvalId: process.env.OPERATOR_APPROVAL_ID,",
    "  cwd: process.env.OPERATOR_CWD,",
    "  input: JSON.parse(process.env.OPERATOR_HOOK_INPUT || '{}'),",
    "};",
    `fs.appendFileSync(${JSON.stringify(outputFile)}, JSON.stringify(payload) + '\\n');`,
  ].join(" ");
  return `node -e ${JSON.stringify(script)}`;
}

afterEach(async () => {
  await Promise.all(tempDirs.splice(0).map((dir) => fs.rm(dir, { recursive: true, force: true })));
});

const exportManifest: ReviewedExportManifest = {
  version: 1,
  createdAt: "2026-01-01T00:00:00.000Z",
  reviewedBy: "operator",
  purpose: "local fine-tuning",
  targetPlatform: "apple-silicon",
  modes: ["sft", "rl"],
  rawCaptureIncluded: false,
  executableSkills: [],
  executableSkillRuns: [],
  promotedSkills: [
    {
      id: "skill-1",
      title: "Deploy workflow",
      summary: "Deploy workflow repaired and ready to reuse",
      sourceCandidateId: "candidate-1",
      sourceTrajectoryIds: ["traj-1"],
      promotedAt: "2026-01-01T00:10:00.000Z",
      version: 1,
    },
  ],
  memories: [
    {
      id: "memory-1",
      type: "session-summary",
      summary: "Deploy workflow repaired and ready to reuse",
      sourceSessionId: "sess-1",
      sourceTrajectoryId: "traj-1",
      tags: ["deploy"],
    },
  ],
  trajectories: [
    {
      id: "traj-1",
      sessionId: "sess-1",
      createdAt: "2026-01-01T00:00:00.000Z",
      captureTier: "operator",
      observationCount: 1,
      actionCount: 1,
      outcomeStatus: "success",
      reward: 1,
    },
  ],
  replays: [
    {
      sessionId: "sess-1",
      trajectoryIds: ["traj-1"],
      eventCount: 2,
      events: [
        { kind: "observation", ts: 1, trajectoryId: "traj-1", source: "browser", summary: "opened deploy" },
        { kind: "action", ts: 2, trajectoryId: "traj-1", tool: "browser", summary: "clicked deploy" },
      ],
    },
  ],
};

describe("StandaloneOperatorRuntime", () => {
  it("starts sessions, records turns, creates trajectories, and supports recall", async () => {
    const runtime = new StandaloneOperatorRuntime({ rootDir: await makeTempDir() });
    const session = await runtime.startSession({ title: "Deploy fix", cwd: "/tmp/repo", agentId: "main" });

    const approval = await runtime.promptApproval({
      sessionId: session.id,
      title: "Deploy",
      summary: "Push deploy button",
      timeoutMs: 5_000,
    });
    expect(approval.sessionId).toBe(session.id);

    const recorded = await runtime.recordTurn({
      sessionId: session.id,
      userText: "investigate deploy issue",
      assistantText: "fixed deploy pipeline and verified the workflow",
      trajectorySummary: "Deploy workflow repaired and ready to reuse",
    });

    expect(recorded.trajectory.sessionId).toBe(session.id);
    expect(recorded.skillCandidate?.sourceTrajectoryIds).toEqual([recorded.trajectory.id]);
    await expect(runtime.listTrajectories(session.id)).resolves.toHaveLength(1);
    await expect(runtime.getLatestAssistantText(session.id)).resolves.toBe(
      "fixed deploy pipeline and verified the workflow",
    );
    await expect(runtime.listSkillCandidates("candidate")).resolves.toEqual([
      expect.objectContaining({ id: recorded.skillCandidate?.id, status: "candidate" }),
    ]);
    await expect(
      runtime.reviewSkillCandidate(recorded.skillCandidate?.id ?? "missing", "reviewed", "reusable workflow"),
    ).resolves.toMatchObject({ id: recorded.skillCandidate?.id, status: "reviewed", reviewNote: "reusable workflow" });

    const promoted = await runtime.promoteSkill(recorded.skillCandidate?.id ?? "missing");
    expect(promoted?.sourceTrajectoryIds).toEqual([recorded.trajectory.id]);
    await expect(runtime.listPromotedSkills()).resolves.toEqual([
      expect.objectContaining({ id: promoted?.id, sourceCandidateId: recorded.skillCandidate?.id }),
    ]);

    await runtime.reviewTrajectory({
      trajectoryId: recorded.trajectory.id,
      status: "approved",
      reviewedBy: "reviewer",
      reviewNote: "safe deploy subset",
      redactedObservations: [{ ts: 11, source: "operator", summary: "reviewed deploy screen" }],
      redactedActions: [{ ts: 12, tool: "assistant-response", summary: "reviewed deploy action" }],
      redactedTranscript: [{ ts: 13, role: "assistant", content: "reviewed deploy transcript" }],
    });

    const recall = await runtime.recall("deploy");
    expect(recall.memories.some((hit) => hit.item.summary.includes("Deploy workflow repaired"))).toBe(true);
    expect(recall.skills.some((hit) => hit.skill.summary.includes("Deploy workflow repaired"))).toBe(true);
    expect(recall.trajectories).toEqual([
      expect.objectContaining({
        trajectoryId: recorded.trajectory.id,
        reviewStatus: "approved",
        preview: expect.arrayContaining([
          expect.objectContaining({ kind: "observation", summary: "reviewed deploy screen" }),
          expect.objectContaining({ kind: "action", summary: "reviewed deploy action" }),
          expect.objectContaining({ kind: "transcript", summary: "reviewed deploy transcript" }),
        ]),
      }),
    ]);
    expect(recall.hits[0]?.score).toBeGreaterThan(0);
    await expect(runtime.getReplay(session.id)).resolves.toMatchObject({
      sessionId: session.id,
      trajectoryIds: [recorded.trajectory.id],
      eventCount: 4,
    });
    await expect(runtime.listReplays()).resolves.toEqual([
      expect.objectContaining({ sessionId: session.id, trajectoryIds: [recorded.trajectory.id] }),
    ]);
  });

  it("updates session lifecycle state", async () => {
    const runtime = new StandaloneOperatorRuntime({ rootDir: await makeTempDir() });
    const session = await runtime.startSession({ title: "Lifecycle", agentId: "main" });

    await expect(runtime.getSession(session.id)).resolves.toMatchObject({ status: "active" });
    await expect(runtime.markSessionIdle(session.id)).resolves.toMatchObject({ status: "idle" });
    await expect(runtime.resumeSession(session.id)).resolves.toMatchObject({ status: "active" });
    await expect(runtime.completeSession(session.id)).resolves.toMatchObject({ status: "completed" });
    await expect(runtime.failSession(session.id)).resolves.toMatchObject({ status: "failed" });
    await expect(runtime.listSessions()).resolves.toHaveLength(1);
  });

  it("updates session metadata for webhook chat bindings", async () => {
    const runtime = new StandaloneOperatorRuntime({ rootDir: await makeTempDir() });
    const session = await runtime.startSession({ title: "Webhook", agentId: "webhook-chat", remoteId: "local-web:thread-1", remoteSource: "webhook-chat:local-web" });

    await expect(runtime.updateSessionMetadata(session.id, {
      webhookChat: {
        channel: "local-web",
        remoteId: "local-web:thread-1",
        conversationId: "conversation-1",
        threadId: "thread-1",
        senderId: "user-1",
        senderName: "Pat",
        reply: { url: "https://example.test/reply" },
        recentDeliveryIds: ["delivery-1"],
        lastInboundAt: "2026-06-04T00:00:00.000Z",
      },
    })).resolves.toMatchObject({
      metadata: {
        remoteId: "local-web:thread-1",
        remoteSource: "webhook-chat:local-web",
        webhookChat: {
          channel: "local-web",
          conversationId: "conversation-1",
          threadId: "thread-1",
          senderId: "user-1",
          senderName: "Pat",
          reply: { url: "https://example.test/reply" },
          recentDeliveryIds: ["delivery-1"],
        },
      },
    });
  });

  it("runs configured lifecycle hooks for session and approval events", async () => {
    const rootDir = await makeTempDir();
    const sessionStartFile = path.join(rootDir, "session-start.jsonl");
    const sessionEndFile = path.join(rootDir, "session-end.jsonl");
    const approvalRequestedFile = path.join(rootDir, "approval-requested.jsonl");
    const approvalResolvedFile = path.join(rootDir, "approval-resolved.jsonl");
    const runtime = new StandaloneOperatorRuntime({
      rootDir,
      executionConfig: resolveOperatorCliExecutionConfig({
        permissionMode: "acceptEdits",
        hooks: {
          SessionStart: [buildHookCaptureCommand(sessionStartFile)],
          SessionEnd: [buildHookCaptureCommand(sessionEndFile)],
          ApprovalRequested: [buildHookCaptureCommand(approvalRequestedFile)],
          ApprovalResolved: [buildHookCaptureCommand(approvalResolvedFile)],
        },
      }),
    });

    const session = await runtime.startSession({ title: "Lifecycle hooks", cwd: rootDir, agentId: "main" });
    const approval = await runtime.promptApproval({
      sessionId: session.id,
      title: "Approve lifecycle",
      summary: "Verify hook payloads",
      metadata: { source: "test" },
    });
    const resolution = await runtime.resolveApproval(approval.id, "approved", "tester");
    await runtime.completeSession(session.id);

    expect(JSON.parse((await fs.readFile(sessionStartFile, "utf8")).trim())).toMatchObject({
      event: "SessionStart",
      sessionId: session.id,
      cwd: rootDir,
      input: {
        session: {
          id: session.id,
          status: "active",
          metadata: expect.objectContaining({ title: "Lifecycle hooks", cwd: rootDir, agentId: "main" }),
        },
      },
    });
    expect(JSON.parse((await fs.readFile(approvalRequestedFile, "utf8")).trim())).toMatchObject({
      event: "ApprovalRequested",
      sessionId: session.id,
      approvalId: approval.id,
      cwd: rootDir,
      input: {
        request: {
          id: approval.id,
          sessionId: session.id,
          metadata: { source: "test" },
        },
      },
    });
    expect(JSON.parse((await fs.readFile(approvalResolvedFile, "utf8")).trim())).toMatchObject({
      event: "ApprovalResolved",
      sessionId: session.id,
      approvalId: approval.id,
      cwd: rootDir,
      input: {
        request: {
          id: approval.id,
          sessionId: session.id,
        },
        resolution: {
          id: approval.id,
          decision: "approved",
          resolvedBy: "tester",
          resolvedAtMs: resolution?.resolvedAtMs,
        },
      },
    });
    expect(JSON.parse((await fs.readFile(sessionEndFile, "utf8")).trim())).toMatchObject({
      event: "SessionEnd",
      sessionId: session.id,
      cwd: rootDir,
      input: {
        status: "completed",
        session: {
          id: session.id,
          status: "completed",
        },
      },
    });
  });

  it("lets PreToolUse deny command execution", async () => {
    const runtime = new StandaloneOperatorRuntime({
      rootDir: await makeTempDir(),
      executionConfig: resolveOperatorCliExecutionConfig({
        hooks: {
          PreToolUse: ["printf '{\"permissionDecision\":\"deny\",\"permissionDecisionReason\":\"blocked by hook\"}'"],
        },
      }),
    });

    const result = await runtime.authorizeCommandExecution({
      sessionId: "sess-1",
      cwd: runtime.rootDir,
      command: "printf ok",
      title: "demo",
      kind: "background.start",
      source: "cli",
      executionConfig: resolveOperatorCliExecutionConfig({
        hooks: {
          PreToolUse: ["printf '{\"permissionDecision\":\"deny\",\"permissionDecisionReason\":\"blocked by hook\"}'"],
        },
      }),
    });

    expect(result).toMatchObject({
      allowed: false,
      message: "blocked by hook",
      command: "printf ok",
      title: "demo",
    });
  });

  it("reuses approval flow for PreToolUse request-approval", async () => {
    const executionConfig = resolveOperatorCliExecutionConfig({
      hooks: {
        PreToolUse: ["printf '{\"permissionDecision\":\"request-approval\",\"permissionDecisionReason\":\"needs review\"}'"],
      },
    });
    const runtime = new StandaloneOperatorRuntime({
      rootDir: await makeTempDir(),
      executionConfig,
    });

    const result = await runtime.authorizeCommandExecution({
      sessionId: "sess-1",
      cwd: runtime.rootDir,
      command: "printf ok",
      kind: "background.start",
      source: "cli",
      executionConfig,
    });

    expect(result.allowed).toBe(false);
    expect(result.pendingApprovalId).toBeTruthy();
    expect(result.message).toContain("Approval required:");
    expect((await runtime.listApprovals()).length).toBe(1);
  });

  it("propagates rewritten command and title into executable skill commands", async () => {
    const executionConfig = resolveOperatorCliExecutionConfig({
      hooks: {
        PreToolUse: ["printf '{\"updatedInput\":{\"command\":\"printf hooked\",\"title\":\"Hooked verify\"}}'"],
      },
    });
    const runtime = new StandaloneOperatorRuntime({
      rootDir: await makeTempDir(),
      executionConfig,
    });
    const session = await runtime.startSession({ title: "Rewrite", agentId: "main" });
    const run = await runtime.startRun({ sessionId: session.id, title: "Skill run" });
    const recorded = await runtime.recordTurn({
      sessionId: session.id,
      userText: "rewrite skill",
      assistantText: "recorded",
      trajectorySummary: "rewrite skill workflow ready to reuse safely",
    });
    if (!recorded.skillCandidate) {
      throw new Error("expected skill candidate");
    }
    await runtime.reviewSkillCandidate(recorded.skillCandidate.id, "reviewed", "reusable");
    const promoted = await runtime.promoteSkill(recorded.skillCandidate.id);
    if (!promoted) {
      throw new Error("expected promoted skill");
    }
    const executable = await runtime.createExecutableSkillFromPromoted({
      promotedSkillId: promoted.id,
      steps: [{ id: "command", title: "Verify", kind: "command", command: "printf original" }],
    });
    if (!executable) {
      throw new Error("expected executable skill");
    }

    const skillRun = await runtime.runExecutableSkill({
      skillId: executable.id,
      sessionId: session.id,
      parentRunId: run.id,
      executionConfig,
    });

    expect(skillRun?.stepResults[0]).toMatchObject({
      output: "hooked",
    });
  });

  it("restores persisted approvals and trajectories across runtime instances", async () => {
    const rootDir = await makeTempDir();
    const runtime = new StandaloneOperatorRuntime({ rootDir });
    const session = await runtime.startSession({ title: "Persisted", agentId: "main" });

    const request = await runtime.promptApproval({
      sessionId: session.id,
      title: "Approve",
      summary: "Run persisted action",
      timeoutMs: 30_000,
    });
    await runtime.recordTurn({
      sessionId: session.id,
      userText: "do the thing",
      assistantText: "done the thing with persistence",
      trajectorySummary: "Persistent thing completed safely",
    });

    const restored = new StandaloneOperatorRuntime({ rootDir });
    await expect(restored.listTrajectories(session.id)).resolves.toHaveLength(1);
    await expect(restored.listApprovals()).resolves.toEqual([
      expect.objectContaining({ id: request.id, sessionId: session.id }),
    ]);
  });

  it("stores plans, routes approval responses through messages, and emits plan events", async () => {
    const runtime = new StandaloneOperatorRuntime({ rootDir: await makeTempDir(), replayLimit: 50 });
    const author = await runtime.startSession({ title: "Author", agentId: "main" });
    const reviewer = await runtime.startSession({ title: "Reviewer", agentId: "reviewer" });

    const draft = await runtime.upsertPlan({
      sessionId: author.id,
      content: "1. Add plan store\n2. Wire approval flow",
    });
    expect(draft).toMatchObject({
      sessionId: author.id,
      status: "draft",
      content: "1. Add plan store\n2. Wire approval flow",
    });

    const requested = await runtime.requestPlanApproval({
      sessionId: author.id,
      approverSessionId: reviewer.id,
    });
    expect(requested.plan).toMatchObject({
      id: draft.id,
      sessionId: author.id,
      status: "awaiting_approval",
      approval: {
        requestedBySessionId: author.id,
        approverSessionId: reviewer.id,
      },
    });
    expect(requested.message).toMatchObject({
      fromSessionId: author.id,
      toSessionId: reviewer.id,
      metadata: {
        type: "plan_approval_request",
        requestId: requested.plan.approval?.requestId,
        planId: draft.id,
        fromSessionId: author.id,
      },
    });
    await expect(runtime.listInbox(reviewer.id)).resolves.toEqual([
      expect.objectContaining({ id: requested.message.id }),
    ]);

    const responded = await runtime.respondPlanApproval({
      requestId: requested.plan.approval?.requestId ?? "missing",
      sessionId: reviewer.id,
      approve: true,
      feedback: "Ship it.",
    });
    expect(responded.plan).toMatchObject({
      id: draft.id,
      status: "approved",
      approval: {
        requestId: requested.plan.approval?.requestId,
        approverSessionId: reviewer.id,
        approved: true,
        feedback: "Ship it.",
      },
    });
    expect(responded.message).toMatchObject({
      fromSessionId: reviewer.id,
      toSessionId: author.id,
      metadata: {
        type: "plan_approval_response",
        requestId: requested.plan.approval?.requestId,
        approve: true,
        feedback: "Ship it.",
      },
    });

    await expect(
      runtime.updatePlanVerification({
        sessionId: author.id,
        status: "requested",
        reminderAt: "2026-06-03T03:00:00.000Z",
        notes: "Re-run focused vitest before push",
      }),
    ).resolves.toMatchObject({
      id: draft.id,
      verification: {
        status: "requested",
        reminderAt: "2026-06-03T03:00:00.000Z",
        notes: "Re-run focused vitest before push",
      },
    });
    await expect(runtime.getPlan(author.id)).resolves.toMatchObject({
      id: draft.id,
      status: "approved",
      verification: {
        status: "requested",
      },
    });

    expect(runtime.events.snapshot().map((event) => event.type)).toEqual(
      expect.arrayContaining([
        "plan.updated",
        "message.sent",
        "plan.approval.requested",
        "plan.approval.resolved",
        "plan.verification.updated",
      ]),
    );
  });

  it("starts, syncs, recovers, lists, and cancels background tasks", async () => {
    const runtime = new StandaloneOperatorRuntime({
      rootDir: await makeTempDir(),
      backgroundTaskIsProcessRunning: () => false,
      backgroundTaskSpawnProcess: () => ({ pid: 999001, unref() {} }),
    });
    const session = await runtime.startSession({ title: "Tasks", agentId: "main" });

    const task = await runtime.startBackgroundTask({
      sessionId: session.id,
      title: "Collect logs",
      command: "printf 'line-1\nline-2\n'",
      kind: "task",
    });
    expect(task.status).toBe("running");
    await expect(runtime.getBackgroundTask(task.id)).resolves.toMatchObject({
      id: task.id,
      sessionId: session.id,
      kind: "task",
      status: "running",
    });
    await expect(runtime.listBackgroundTasks(session.id)).resolves.toEqual([
      expect.objectContaining({ id: task.id, sessionId: session.id }),
    ]);

    await runtime.backgroundTasks.executionService.writeState(task, {
      version: 1,
      taskId: task.id,
      kind: "task",
      status: "completed",
      pid: task.execution.processId ?? 1234,
      startedAt: "2026-01-01T00:00:00.000Z",
      updatedAt: "2026-01-01T00:10:00.000Z",
      completedAt: "2026-01-01T00:10:00.000Z",
      exitCode: 0,
      outputFile: task.execution.outputFile,
      cwd: task.cwd,
      command: task.command,
    });
    await runtime.backgroundTasks.executionService.writeOutput(task, "line-1\nline-2\n");
    await expect(runtime.getBackgroundTaskExecutionState(task.id)).resolves.toMatchObject({
      taskId: task.id,
      status: "completed",
    });
    await expect(runtime.getBackgroundTaskOutput(task.id, 1)).resolves.toEqual({ taskId: task.id, output: "line-2" });
    await expect(runtime.syncBackgroundTask(task.id)).resolves.toMatchObject({
      id: task.id,
      status: "completed",
      execution: { exitCode: 0 },
    });

    const recoverable = await runtime.startBackgroundTask({
      sessionId: session.id,
      title: "Watch logs",
      command: "tail -f app.log",
      kind: "monitor",
    });
    await runtime.backgroundTasks.executionService.writeState(recoverable, {
      version: 1,
      taskId: recoverable.id,
      kind: "monitor",
      status: "running",
      pid: recoverable.execution.processId ?? 5678,
      startedAt: "2026-01-01T00:00:00.000Z",
      updatedAt: "2026-01-01T00:00:00.000Z",
      outputFile: recoverable.execution.outputFile,
      cwd: recoverable.cwd,
      command: recoverable.command,
    });
    await expect(runtime.recoverBackgroundTask(recoverable.id)).resolves.toMatchObject({
      task: { id: recoverable.id, status: "failed" },
      reason: "missing-process",
      changed: true,
    });
    await expect(
      runtime.recoverBackgroundTasks({ sessionId: session.id, reasons: ["missing-process", "unchanged"] }),
    ).resolves.toEqual(
      expect.arrayContaining([
        expect.objectContaining({ task: expect.objectContaining({ id: recoverable.id }), reason: "unchanged", changed: false }),
      ]),
    );

    const monitor = await runtime.startBackgroundTask({
      sessionId: session.id,
      title: "Watch logs",
      command: "tail -f app.log",
      kind: "monitor",
    });
    await runtime.backgroundTasks.executionService.writeState(monitor, {
      version: 1,
      taskId: monitor.id,
      kind: "monitor",
      status: "running",
      pid: monitor.execution.processId ?? 5678,
      startedAt: "2026-01-01T00:00:00.000Z",
      updatedAt: "2026-01-01T00:00:00.000Z",
      outputFile: monitor.execution.outputFile,
      cwd: monitor.cwd,
      command: monitor.command,
    });
    const originalKill = process.kill;
    process.kill = (() => true) as typeof process.kill;
    try {
      await expect(runtime.cancelBackgroundTask(monitor.id)).resolves.toMatchObject({
        id: monitor.id,
        status: "cancelled",
        execution: { signal: "SIGTERM" },
      });
    } finally {
      process.kill = originalKill;
    }
  });

  it("tracks runs and subagents", async () => {
    const runtime = new StandaloneOperatorRuntime({ rootDir: await makeTempDir() });
    const session = await runtime.startSession({ title: "Runs", agentId: "main" });
    const childSession = await runtime.startSession({ title: "Child", agentId: "worker" });

    const run = await runtime.startRun({ sessionId: session.id, title: "Investigate issue" });
    expect(run.status).toBe("running");

    await expect(runtime.updateRunStatus(run.id, "paused", { reason: "awaiting approval" })).resolves.toMatchObject({
      status: "paused",
      metadata: { reason: "awaiting approval" },
    });
    await expect(runtime.pauseActiveRun(session.id, "gateway unhealthy")).resolves.toMatchObject({
      id: run.id,
      status: "paused",
      metadata: {
        reason: "awaiting approval",
        remoteControlAction: "pause",
        remoteControlSource: "remote-control",
        remoteControlReason: "gateway unhealthy",
      },
    });
    await expect(runtime.listRuns(session.id)).resolves.toEqual([
      expect.objectContaining({ id: run.id, sessionId: session.id }),
    ]);

    const subagent = await runtime.registerSubagent({
      sessionId: session.id,
      parentRunId: run.id,
      childSessionId: childSession.id,
      title: "Collect logs",
    });
    expect(subagent.parentRunId).toBe(run.id);

    await expect(runtime.updateSubagentStatus(subagent.runId, "running")).resolves.toMatchObject({
      status: "running",
    });
    await expect(runtime.listSubagents(run.id)).resolves.toEqual([
      expect.objectContaining({ runId: subagent.runId, childSessionId: childSession.id }),
    ]);
    await expect(runtime.listActiveSubagents()).resolves.toEqual([
      expect.objectContaining({ runId: subagent.runId, status: "running" }),
    ]);

    const parentSessionWithModel = await runtime.startSession({
      title: "Parent with model",
      agentId: "main",
      modelSelection: {
        primary: "claude-opus-4-7",
        fallbacks: ["claude-sonnet-4-6", "claude-haiku-4-5-20251001"],
        source: "default",
      },
    });
    const parentRunWithModel = await runtime.startRun({
      sessionId: parentSessionWithModel.id,
      title: "Parent run with model",
      metadata: {
        modelSelection: {
          primary: "claude-opus-4-7",
          fallbacks: ["claude-sonnet-4-6", "claude-haiku-4-5-20251001"],
          source: "default",
        },
      },
    });

    const spawned = await runtime.spawnSubagent({
      sessionId: parentSessionWithModel.id,
      parentRunId: parentRunWithModel.id,
      title: "Spawned child work",
      agentId: "worker",
      remoteSource: "gateway",
      metadata: { source: "spawn" },
    });
    expect(spawned.subagent).toMatchObject({
      parentRunId: parentRunWithModel.id,
      sessionId: parentSessionWithModel.id,
      title: "Spawned child work",
      status: "running",
    });
    expect(spawned.childSession.metadata).toMatchObject({
      title: "Spawned child work",
      agentId: "worker",
      remoteSource: "gateway",
      parentSessionId: parentSessionWithModel.id,
      modelSelection: {
        primary: "claude-opus-4-7",
        fallbacks: ["claude-sonnet-4-6", "claude-haiku-4-5-20251001"],
        source: "inherited",
      },
    });
    expect(spawned.childRun).toMatchObject({
      sessionId: spawned.childSession.id,
      parentRunId: spawned.subagent.runId,
      title: "Spawned child work",
      status: "running",
      metadata: {
        source: "spawn",
        modelSelection: {
          primary: "claude-opus-4-7",
          fallbacks: ["claude-sonnet-4-6", "claude-haiku-4-5-20251001"],
          source: "inherited",
        },
      },
    });

    const overridden = await runtime.spawnSubagent({
      sessionId: parentSessionWithModel.id,
      parentRunId: parentRunWithModel.id,
      title: "Override child work",
      modelPrimary: "claude-sonnet-4-6",
    });
    expect(overridden.childSession.metadata).toMatchObject({
      parentSessionId: parentSessionWithModel.id,
      modelSelection: {
        primary: "claude-sonnet-4-6",
        fallbacks: ["claude-sonnet-4-6", "claude-haiku-4-5-20251001"],
        source: "override",
      },
    });
    expect(overridden.childRun.metadata).toMatchObject({
      modelSelection: {
        primary: "claude-sonnet-4-6",
        fallbacks: ["claude-sonnet-4-6", "claude-haiku-4-5-20251001"],
        source: "override",
      },
    });

    const clearedFallbacks = await runtime.spawnSubagent({
      sessionId: parentSessionWithModel.id,
      parentRunId: parentRunWithModel.id,
      title: "Clear fallback child work",
      modelFallbacks: [],
    });
    expect(clearedFallbacks.childSession.metadata).toMatchObject({
      modelSelection: {
        primary: "claude-opus-4-7",
        fallbacks: [],
        source: "override",
      },
    });
    expect(clearedFallbacks.childRun.metadata).toMatchObject({
      modelSelection: {
        primary: "claude-opus-4-7",
        fallbacks: [],
        source: "override",
      },
    });

    const parentRunPrimaryOnly = await runtime.startRun({
      sessionId: parentSessionWithModel.id,
      title: "Parent run primary only",
      metadata: {
        modelSelection: {
          primary: "claude-opus-4-7",
          source: "default",
        },
      },
    });
    const strictOverride = await runtime.spawnSubagent({
      sessionId: parentSessionWithModel.id,
      parentRunId: parentRunPrimaryOnly.id,
      title: "Strict override child work",
      modelPrimary: "claude-sonnet-4-6",
    });
    expect(strictOverride.childSession.metadata).toMatchObject({
      modelSelection: {
        primary: "claude-sonnet-4-6",
        fallbacks: [],
        source: "override",
      },
    });
    expect(strictOverride.childRun.metadata).toMatchObject({
      modelSelection: {
        primary: "claude-sonnet-4-6",
        fallbacks: [],
        source: "override",
      },
    });
  });

  it("creates executable skills, runs them, and exports their provenance", async () => {
    const runtime = new StandaloneOperatorRuntime({ rootDir: await makeTempDir() });
    const session = await runtime.startSession({ title: "Export review", agentId: "main" });
    const run = await runtime.startRun({ sessionId: session.id, title: "Skill run" });
    const recorded = await runtime.recordTurn({
      sessionId: session.id,
      userText: "open billing settings",
      assistantText: "updated the billing workflow",
      trajectorySummary: "Billing workflow updated and ready to reuse safely",
    });
    if (!recorded.skillCandidate) {
      throw new Error("expected skill candidate from recorded turn");
    }
    await runtime.reviewSkillCandidate(recorded.skillCandidate.id, "reviewed", "safe to promote");
    const promotedSkill = await runtime.promoteSkill(recorded.skillCandidate.id);
    if (!promotedSkill) {
      throw new Error("expected promoted skill");
    }

    await expect(
      runtime.reviewTrajectory({
        trajectoryId: recorded.trajectory.id,
        status: "approved",
        reviewedBy: "reviewer",
        redactedObservations: [{ ts: 11, source: "operator", summary: "reviewed observation" }],
        redactedActions: [{ ts: 12, tool: "assistant-response", summary: "reviewed action" }],
        redactedTranscript: [{ ts: 13, role: "assistant", content: "reviewed assistant" }],
      }),
    ).resolves.toMatchObject({
      id: recorded.trajectory.id,
      review: {
        status: "approved",
        reviewedBy: "reviewer",
        reviewedAt: expect.any(String),
      },
    });
    await expect(runtime.listReviewedTrajectories()).resolves.toEqual([
      expect.objectContaining({ id: recorded.trajectory.id, review: expect.objectContaining({ status: "approved" }) }),
    ]);

    const executableSkill = await runtime.createExecutableSkillFromPromoted({
      promotedSkillId: promotedSkill.id,
      steps: [
        { id: "summary", title: "Summarize", kind: "summary", content: "billing summary" },
        {
          id: "x-post",
          title: "Post update",
          kind: "x-post",
          content: "This update is intentionally longer than twenty characters.",
          maxCharacters: 20,
          overflowToComment: true,
        },
        { id: "command", title: "Verify", kind: "command", command: "printf 'verified'", agentRole: "worker" },
      ],
    });
    expect(executableSkill).toMatchObject({ promotedSkillId: promotedSkill.id, usage: { executionCount: 0 } });
    if (!executableSkill) {
      throw new Error("expected executable skill");
    }

    await expect(runtime.listExecutableSkills()).resolves.toEqual([
      expect.objectContaining({ id: executableSkill.id, promotedSkillId: promotedSkill.id }),
    ]);
    await expect(runtime.runExecutableSkill({ skillId: executableSkill.id, sessionId: session.id, parentRunId: run.id })).resolves.toMatchObject({
      skillId: executableSkill.id,
      parentRunId: run.id,
      sourceTrajectoryIds: [recorded.trajectory.id],
      stepResults: [
        expect.objectContaining({ stepId: "summary", output: "billing summary" }),
        expect.objectContaining({ stepId: "x-post", output: "This update is inten", commentOutputs: expect.any(Array) }),
        expect.objectContaining({ stepId: "command", output: "verified", agentRole: "worker", subagentRunId: expect.any(String) }),
      ],
    });
    await expect(runtime.listExecutableSkillRuns(executableSkill.id)).resolves.toEqual([
      expect.objectContaining({ skillId: executableSkill.id, sourceTrajectoryIds: [recorded.trajectory.id] }),
    ]);
    await expect(runtime.listPromotedSkills()).resolves.toEqual([
      expect.objectContaining({ id: promotedSkill.id, usage: expect.objectContaining({ executionCount: 1, lastExecutedAt: expect.any(String) }) }),
    ]);

    const exportFile = path.join(tempDirs[tempDirs.length - 1] ?? "", "reviewed-export.json");
    await expect(
      runtime.createReviewedExport({
        reviewedBy: "operator",
        purpose: "local fine-tuning",
        outputFile: exportFile,
        modes: ["sft", "rl"],
      }),
    ).resolves.toMatchObject({
      reviewedBy: "operator",
      executableSkills: [expect.objectContaining({ promotedSkillId: promotedSkill.id, usage: expect.objectContaining({ executionCount: 1 }) })],
      executableSkillRuns: [expect.objectContaining({ skillId: executableSkill.id, sourceTrajectoryIds: [recorded.trajectory.id] })],
      trajectories: [
        expect.objectContaining({
          id: recorded.trajectory.id,
          reviewedBy: "reviewer",
          observationCount: 1,
          actionCount: 1,
        }),
      ],
      replays: [
        expect.objectContaining({
          sessionId: session.id,
          trajectoryIds: [recorded.trajectory.id],
        }),
      ],
    });
    await expect(fs.readFile(exportFile, "utf8")).resolves.toContain("reviewed assistant");
  });

  it("creates and executes reviewed local training jobs", async () => {
    const runtime = new StandaloneOperatorRuntime({ rootDir: await makeTempDir() });

    const job = await runtime.createTrainingJob({ exportManifest, mode: "sft" });
    expect(job.mode).toBe("sft");
    await expect(runtime.getTrainingJob(job.id)).resolves.toMatchObject({ id: job.id, status: "planned" });
    await expect(runtime.listTrainingJobs()).resolves.toEqual([
      expect.objectContaining({ id: job.id, mode: "sft", status: "planned" }),
    ]);
    await expect(runtime.updateTrainingJobStatus(job.id, "ready")).resolves.toMatchObject({ id: job.id, status: "ready" });
    await expect(runtime.prepareTrainingJob(job.id)).resolves.toMatchObject({
      id: job.id,
      status: "ready",
      execution: {
        workingDirectory: `training-jobs/${job.id}`,
        logFile: `training-jobs/${job.id}/logs/sft.log`,
        command: [
          "python3",
          "-m",
          "mlx_lm.lora",
          "--train",
          "--data",
          `training-jobs/${job.id}/dataset`,
          "--adapter-path",
          `training-jobs/${job.id}/artifacts`,
          "--learning-rate",
          "0.00001",
          "--batch-size",
          "1",
          "--iters",
          "3000",
        ],
        environment: { OPENCLAW_TRAINING_RUNTIME: "mlx" },
      },
    });
    await expect(runtime.getTrainingJobPlan(job.id)).resolves.toMatchObject({
      jobId: job.id,
      runtime: "mlx",
      outputPath: `training-jobs/${job.id}/artifacts/model.gguf`,
      statePath: `training-jobs/${job.id}/state.json`,
    });
    await expect(runtime.getTrainingJobLaunchScript(job.id)).resolves.toContain("mlx_lm.lora");
    await expect(runtime.getTrainingJobExecutionState(job.id)).resolves.toBeUndefined();
    await expect(runtime.getTrainingJobLog(job.id)).resolves.toBeUndefined();
    const preparedJob = await runtime.getTrainingJob(job.id);
    if (!preparedJob?.execution) {
      throw new Error("expected prepared execution");
    }
    await runtime.trainingJobs.executionService.writeState(preparedJob, {
      version: 1,
      jobId: job.id,
      status: "completed",
      pid: 1234,
      startedAt: "2026-01-01T00:00:00.000Z",
      updatedAt: "2026-01-01T00:10:00.000Z",
      completedAt: "2026-01-01T00:10:00.000Z",
      exitCode: 0,
      logFile: preparedJob.execution.logFile,
      workingDirectory: preparedJob.execution.workingDirectory,
      command: preparedJob.execution.command ?? [],
    });
    await runtime.trainingJobs.executionService.writeLog(preparedJob, "line-1\nline-2\n");
    await expect(runtime.getTrainingJobExecutionState(job.id)).resolves.toMatchObject({
      jobId: job.id,
      status: "completed",
      pid: 1234,
    });
    await expect(runtime.getTrainingJobLog(job.id, 1)).resolves.toEqual({ jobId: job.id, log: "line-2" });
    await expect(runtime.syncTrainingJob(job.id)).resolves.toMatchObject({
      id: job.id,
      status: "completed",
      execution: { outputModelRef: `training-jobs/${job.id}/artifacts/model.gguf` },
    });
    const restarted = await runtime.createTrainingJob({ exportManifest, mode: "sft" });
    await runtime.prepareTrainingJob(restarted.id);
    await expect(runtime.trainingJobs.markExecutionStarted(restarted.id, { pid: 1234 })).resolves.toMatchObject({
      id: restarted.id,
      status: "running",
      execution: {
        startedAt: expect.any(String),
        processId: 1234,
        steps: [
          { id: "prepare-dataset", status: "completed" },
          { id: "launch-training", status: "running" },
          { id: "collect-artifacts", status: "pending" },
          { id: "evaluate-replay", status: "pending" },
        ],
      },
    });
    await expect(runtime.completeTrainingJob(job.id, { outputModelRef: "artifacts/models/sft.gguf" })).resolves.toMatchObject({
      id: job.id,
      status: "completed",
      execution: {
        completedAt: expect.any(String),
        outputModelRef: "artifacts/models/sft.gguf",
      },
    });

    const restored = new StandaloneOperatorRuntime({ rootDir: tempDirs[tempDirs.length - 1] ?? "" });
    await expect(restored.getTrainingJob(job.id)).resolves.toMatchObject({
      id: job.id,
      status: "completed",
      execution: { outputModelRef: "artifacts/models/sft.gguf" },
    });
  });

  it("records failed training executions", async () => {
    const runtime = new StandaloneOperatorRuntime({ rootDir: await makeTempDir() });
    const job = await runtime.createTrainingJob({ exportManifest, mode: "rl" });

    await expect(runtime.failTrainingJob(job.id, { reason: "reward model unavailable" })).resolves.toMatchObject({
      id: job.id,
      status: "failed",
      execution: {
        failureReason: "reward model unavailable",
        failedAt: expect.any(String),
        logFile: `training-jobs/${job.id}/logs/rl.log`,
      },
    });
  });

  it("publishes a sync event when persisted execution state completes", async () => {
    const runtime = new StandaloneOperatorRuntime({ rootDir: await makeTempDir() });
    const stream = runtime.events.stream((event) => event.type === "training-job.synced")[Symbol.asyncIterator]();

    const job = await runtime.createTrainingJob({ exportManifest, mode: "sft" });
    const prepared = await runtime.prepareTrainingJob(job.id);
    if (!prepared?.execution) {
      throw new Error("expected prepared execution");
    }

    await runtime.trainingJobs.executionService.writeState(prepared, {
      version: 1,
      jobId: job.id,
      status: "completed",
      pid: 1234,
      startedAt: "2026-01-01T00:00:00.000Z",
      updatedAt: "2026-01-01T00:10:00.000Z",
      completedAt: "2026-01-01T00:10:00.000Z",
      exitCode: 0,
      logFile: prepared.execution.logFile,
      workingDirectory: prepared.execution.workingDirectory,
      command: prepared.execution.command ?? [],
    });

    await expect(runtime.syncTrainingJob(job.id)).resolves.toMatchObject({ id: job.id, status: "completed" });
    const next = await stream.next();
    expect(next.done).toBe(false);
    expect(next.value).toMatchObject({ type: "training-job.synced", payload: { id: job.id, status: "completed" } });
    await stream.return?.();
  });

  it("registers, lists, updates, and activates plugins", async () => {
    const rootDir = await makeTempDir();
    const runtime = new StandaloneOperatorRuntime({ rootDir });
    const entrypoint = path.join(rootDir, "runtime-plugin.mjs");

    await fs.writeFile(
      entrypoint,
      [
        "export default {",
        "  async activate() {",
        "    globalThis.__runtimePluginActivations = (globalThis.__runtimePluginActivations ?? 0) + 1;",
        "  },",
        "};",
        "",
      ].join("\n"),
      "utf8",
    );

    const plugin = await runtime.registerPlugin({
      manifest: {
        id: "runtime-plugin",
        version: "1.0.0",
        name: "Runtime Plugin",
        description: "Provides a tool runtime",
        entrypoint,
        capabilities: ["tool"],
      },
    });

    expect(plugin.enabled).toBe(true);
    await expect(runtime.getPlugin("runtime-plugin")).resolves.toMatchObject({
      manifest: { id: "runtime-plugin" },
      enabled: true,
    });
    await expect(runtime.listPlugins({ capability: "tool" })).resolves.toEqual([
      expect.objectContaining({ manifest: expect.objectContaining({ id: "runtime-plugin" }) }),
    ]);
    await expect(runtime.updatePluginEnabled("runtime-plugin", false)).resolves.toMatchObject({ enabled: false });
    await runtime.activatePlugin("runtime-plugin");
    expect((globalThis as Record<string, unknown>).__runtimePluginActivations).toBe(1);
  });

  it("creates capture consent and ingests browser, device, and os events", async () => {
    const runtime = new StandaloneOperatorRuntime({ rootDir: await makeTempDir() });

    const consent = await runtime.createCaptureConsent({ tier: "operator", purpose: "capture workflow" });
    expect(consent.tier).toBe("operator");
    await expect(runtime.listActiveCaptureConsents()).resolves.toEqual([
      expect.objectContaining({ id: consent.id, tier: "operator" }),
    ]);

    await expect(
      runtime.recordBrowserCapture({
        sessionId: "sess-capture",
        pageUrl: "https://example.test",
        pageTitle: "Home",
        visibleIndicator: true,
        ts: 1,
        action: { kind: "navigate", target: "https://example.test", ts: 2 },
      }),
    ).resolves.toMatchObject({ recorded: true, trajectory: { sessionId: "sess-capture" } });

    await expect(
      runtime.recordDeviceCapture({
        sessionId: "sess-capture",
        deviceId: "sim-1",
        platform: "ios",
        appId: "mobile-app",
        appName: "Mobile App",
        visibleIndicator: true,
        ts: 3,
        gesture: { kind: "tap", target: "Pay", ts: 4 },
      }),
    ).resolves.toMatchObject({ recorded: true, trajectory: { sessionId: "sess-capture" } });

    await expect(
      runtime.recordOsObservation({
        sessionId: "sess-capture",
        appId: "terminal",
        visibleIndicator: true,
        ts: 5,
        event: "command-ran",
        commandSummary: "pnpm build",
      }),
    ).resolves.toMatchObject({ recorded: true, trajectory: { sessionId: "sess-capture" } });

    await expect(runtime.listTrajectories("sess-capture")).resolves.toHaveLength(3);
    await expect(runtime.revokeCaptureConsent(consent.id)).resolves.toMatchObject({ id: consent.id, revokedAt: expect.any(String) });
  });
});
