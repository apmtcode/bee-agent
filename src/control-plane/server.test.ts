import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { OperatorControlPlaneServer } from "./server.js";
import { OperatorCronService } from "./cron-service.js";
import { OperatorDeliveryService } from "./delivery.js";
import { buildRuntimeEventFilter, subscribeRuntimeEvents } from "./subscriptions.js";
import { StandaloneOperatorRuntime } from "../orchestrator/operator-runtime.js";
import type { ReviewedExportManifest } from "../training/export-manifest.js";

const tempDirs: string[] = [];

async function makeTempDir(): Promise<string> {
  const dir = await fs.mkdtemp(path.join(os.tmpdir(), "control-plane-"));
  tempDirs.push(dir);
  return dir;
}

afterEach(async () => {
  await Promise.all(tempDirs.splice(0).map((dir) => fs.rm(dir, { recursive: true, force: true, maxRetries: 5, retryDelay: 50 })));
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

describe("OperatorControlPlaneServer", () => {
  it("handles session, transcript, approval, trajectory, memory, and orchestration methods", async () => {
    const runtime = new StandaloneOperatorRuntime({
      rootDir: await makeTempDir(),
      backgroundTaskIsProcessRunning: () => false,
    });
    const server = new OperatorControlPlaneServer({ runtime });
    const session = await runtime.startSession({ title: "RPC", agentId: "main" });
    const childSession = await runtime.startSession({ title: "Worker", agentId: "worker" });
    const approval = await runtime.promptApproval({
      sessionId: session.id,
      title: "Deploy",
      summary: "Approve deploy",
      timeoutMs: 30_000,
    });
    const recorded = await runtime.recordTurn({
      sessionId: session.id,
      userText: "check deploy",
      assistantText: "deploy checked and fixed",
      trajectorySummary: "Deploy repaired for recall",
    });
    const taskCreate = await server.handle({
      method: "tasks.create",
      params: {
        sessionId: session.id,
        subject: "Inspect logs",
        description: "Look at the latest logs.",
        activeForm: "Inspecting logs",
        owner: "operator",
        metadata: { priority: "high" },
        blocks: ["follow-up"],
      },
    });
    expect(taskCreate).toMatchObject({
      ok: true,
      result: {
        sessionId: session.id,
        subject: "Inspect logs",
        description: "Look at the latest logs.",
        activeForm: "Inspecting logs",
        owner: "operator",
        status: "pending",
        metadata: { priority: "high" },
        blocks: ["follow-up"],
        blockedBy: [],
      },
    });
    const task = taskCreate.ok ? taskCreate.result : undefined;
    expect(task).toBeDefined();
    if (!task) {
      throw new Error("expected task to be created");
    }
    await expect(server.handle({ method: "tasks.get", params: { taskId: task.id } })).resolves.toMatchObject({
      ok: true,
      result: { id: task.id, sessionId: session.id, subject: "Inspect logs", status: "pending" },
    });
    await expect(server.handle({ method: "tasks.list", params: { sessionId: session.id } })).resolves.toMatchObject({
      ok: true,
      result: [expect.objectContaining({ id: task.id, sessionId: session.id, subject: "Inspect logs" })],
    });
    await expect(
      server.handle({
        method: "tasks.update",
        params: {
          taskId: task.id,
          status: "in_progress",
          activeForm: null,
          owner: null,
          metadata: { lane: "ops" },
          blockedBy: [approval.id],
        },
      }),
    ).resolves.toMatchObject({
      ok: true,
      result: {
        id: task.id,
        status: "in_progress",
        metadata: { priority: "high", lane: "ops" },
        blocks: ["follow-up"],
        blockedBy: [approval.id],
      },
    });
    await expect(server.handle({ method: "runs.events", params: { sessionId: session.id, family: "task" } })).resolves.toMatchObject({
      ok: true,
      result: expect.arrayContaining([
        expect.objectContaining({ type: "task.created", payload: expect.objectContaining({ id: task.id, sessionId: session.id }) }),
        expect.objectContaining({ type: "task.updated", payload: expect.objectContaining({ id: task.id, status: "in_progress" }) }),
      ]),
    });
    const messageSend = await server.handle({
      method: "messages.send",
      params: {
        fromSessionId: session.id,
        toSessionId: childSession.id,
        summary: "Need deploy logs",
        message: "Please collect the latest deploy logs.",
        metadata: { priority: "high" },
      },
    });
    expect(messageSend).toMatchObject({
      ok: true,
      result: {
        fromSessionId: session.id,
        toSessionId: childSession.id,
        summary: "Need deploy logs",
        message: "Please collect the latest deploy logs.",
        metadata: { priority: "high" },
      },
    });
    const message = messageSend.ok ? messageSend.result : undefined;
    expect(message).toBeDefined();
    if (!message) {
      throw new Error("expected message to be sent");
    }
    await expect(server.handle({ method: "messages.get", params: { messageId: message.id } })).resolves.toMatchObject({
      ok: true,
      result: { id: message.id, fromSessionId: session.id, toSessionId: childSession.id, summary: "Need deploy logs" },
    });
    await expect(server.handle({ method: "messages.list", params: { sessionId: session.id } })).resolves.toMatchObject({
      ok: true,
      result: [expect.objectContaining({ id: message.id, fromSessionId: session.id, toSessionId: childSession.id })],
    });
    await expect(server.handle({ method: "messages.inbox", params: { sessionId: childSession.id } })).resolves.toMatchObject({
      ok: true,
      result: [expect.objectContaining({ id: message.id, toSessionId: childSession.id })],
    });
    await expect(server.handle({ method: "messages.outbox", params: { sessionId: session.id } })).resolves.toMatchObject({
      ok: true,
      result: [expect.objectContaining({ id: message.id, fromSessionId: session.id })],
    });
    await expect(server.handle({ method: "runs.events", params: { sessionId: session.id, family: "message" } })).resolves.toMatchObject({
      ok: true,
      result: [
        expect.objectContaining({ type: "message.sent", payload: expect.objectContaining({ id: message.id, fromSessionId: session.id, toSessionId: childSession.id }) }),
      ],
    });
    const runCreate = await server.handle({
      method: "runs.start",
      params: { sessionId: session.id, title: "Investigate deploy", metadata: { source: "rpc" } },
    });
    expect(runCreate).toMatchObject({
      ok: true,
      result: { sessionId: session.id, title: "Investigate deploy", status: "running", metadata: { source: "rpc" } },
    });
    const run = runCreate.ok ? runCreate.result : undefined;
    expect(run).toBeDefined();
    runtime.publishRunProgress({
      runId: run?.id,
      sessionId: session.id,
      phase: "planning",
      message: "Collecting deploy evidence.",
    });
    const subagentCreate = await server.handle({
      method: "subagents.register",
      params: {
        sessionId: session.id,
        parentRunId: run?.id,
        childSessionId: childSession.id,
        title: "Collect logs",
        metadata: { source: "rpc" },
      },
    });
    expect(subagentCreate).toMatchObject({
      ok: true,
      result: { parentRunId: run?.id, childSessionId: childSession.id, title: "Collect logs", status: "pending" },
    });
    const subagent = subagentCreate.ok ? subagentCreate.result : undefined;
    expect(subagent).toBeDefined();

    const bootstrappedSession = await server.handle({
      method: "sessions.bootstrap",
      params: { sessionId: session.id, family: "approval" },
    });
    expect(bootstrappedSession).toMatchObject({
      ok: true,
      result: {
        session: { id: session.id },
        created: false,
        resumed: false,
        approvals: [expect.objectContaining({ id: approval.id, sessionId: session.id })],
        events: [expect.objectContaining({ type: "approval.requested" })],
      },
    });
    const approvalReplayCursor = Date.now();
    const filteredBootstrap = await server.handle({
      method: "sessions.bootstrap",
      params: { sessionId: session.id, family: "approval", afterTs: approvalReplayCursor },
    });
    expect(filteredBootstrap).toMatchObject({
      ok: true,
      result: {
        session: { id: session.id },
        events: [],
      },
    });

    const remoteBootstrappedSession = await server.handle({
      method: "sessions.bootstrap",
      params: { title: "Remote bootstrap", remoteId: "device-server-1", remoteSource: "gateway", agentId: "gateway" },
    });
    expect(remoteBootstrappedSession).toMatchObject({
      ok: true,
      result: {
        created: true,
        resumed: false,
        session: {
          metadata: { remoteId: "device-server-1", remoteSource: "gateway", agentId: "gateway" },
        },
      },
    });
    const remoteSessionId = remoteBootstrappedSession.ok ? remoteBootstrappedSession.result.session.id : undefined;
    if (!remoteSessionId) {
      throw new Error("expected remote bootstrapped session id");
    }
    await runtime.markSessionIdle(remoteSessionId);
    await expect(
      server.handle({
        method: "sessions.bootstrap",
        params: { remoteId: "device-server-1", remoteSource: "gateway" },
      }),
    ).resolves.toMatchObject({
      ok: true,
      result: {
        created: false,
        resumed: true,
        session: {
          id: remoteSessionId,
          metadata: { remoteId: "device-server-1", remoteSource: "gateway" },
        },
      },
    });
    await expect(
      server.handle({
        method: "sessions.bootstrap",
        params: { sessionId: session.id, remoteId: "device-server-1", remoteSource: "gateway" },
      }),
    ).resolves.toMatchObject({
      ok: true,
      result: {
        created: false,
        session: { id: session.id },
      },
    });
    const pairingCreate = await server.handle({
      method: "pairing.create",
      params: { remoteSource: "gateway" },
    });
    expect(pairingCreate).toMatchObject({
      ok: true,
      result: {
        remoteSource: "gateway",
        status: "pending",
        code: expect.stringMatching(/^[A-Z0-9]{4}-[A-Z0-9]{4}$/),
      },
    });
    if (!pairingCreate.ok) {
      throw new Error("expected pairing ticket");
    }
    await expect(server.handle({ method: "pairing.list", params: { status: "pending" } })).resolves.toMatchObject({
      ok: true,
      result: [expect.objectContaining({ id: pairingCreate.result.id, code: pairingCreate.result.code, status: "pending" })],
    });
    await expect(
      server.handle({
        method: "pairing.resolve",
        params: { code: pairingCreate.result.code, decision: "approved", resolvedBy: "tester" },
      }),
    ).resolves.toMatchObject({
      ok: true,
      result: { id: pairingCreate.result.id, status: "approved", resolvedBy: "tester" },
    });
    const pairedBootstrap = await server.handle({
      method: "sessions.bootstrap",
      params: { pairingCode: pairingCreate.result.code, agentId: "gateway" },
    });
    expect(pairedBootstrap).toMatchObject({
      ok: true,
      result: {
        session: {
          metadata: {
            remoteId: pairingCreate.result.remoteId,
            remoteSource: "gateway",
            agentId: "gateway",
          },
        },
      },
    });
    if (!pairedBootstrap.ok) {
      throw new Error("expected paired bootstrap session");
    }
    await runtime.promptApproval({
      sessionId: pairedBootstrap.result.session.id,
      title: "Remote approval",
      summary: "Approve remote command",
    });
    const remoteRun = await runtime.startRun({ sessionId: pairedBootstrap.result.session.id, title: "Remote run" });
    const remoteTask = await runtime.startBackgroundTask({
      sessionId: pairedBootstrap.result.session.id,
      title: "Remote task",
      command: "sleep 5",
      kind: "task",
    });
    await expect(server.handle({ method: "pairing.list", params: { status: "redeemed" } })).resolves.toMatchObject({
      ok: true,
      result: [expect.objectContaining({ id: pairingCreate.result.id, status: "redeemed" })],
    });
    server.registerGatewaySessionConnection({
      connectionId: "gateway-healthy-1",
      sessionId: pairedBootstrap.result.session.id,
      remoteId: pairingCreate.result.remoteId,
      state: "healthy",
      updatedAt: Date.now(),
      lastClientActivityAt: Date.now(),
    });
    await expect(
      server.handle({
        method: "sessions.remoteStatus",
        params: { identifier: pairingCreate.result.remoteId },
      }),
    ).resolves.toMatchObject({
      ok: true,
      result: {
        remoteId: pairingCreate.result.remoteId,
        remoteSource: "gateway",
        session: { id: expect.any(String), metadata: { remoteId: pairingCreate.result.remoteId, remoteSource: "gateway" } },
        pairing: { code: pairingCreate.result.code, status: "redeemed" },
        approvals: [expect.objectContaining({ title: "Remote approval", sessionId: expect.any(String) })],
        control: { state: "active" },
        activeRun: { id: remoteRun.id, title: "Remote run", status: "running" },
        activeBackgroundTask: { id: remoteTask.id, title: "Remote task", kind: "task", status: "running" },
        recentEvents: expect.arrayContaining([
          expect.objectContaining({ type: "approval.requested" }),
          expect.objectContaining({ type: "run.started" }),
          expect.objectContaining({ type: "background-task.started" }),
        ]),
      },
    });
    server.updateGatewaySessionHealth({
      connectionId: "gateway-healthy-1",
      sessionId: pairedBootstrap.result.session.id,
      remoteId: pairingCreate.result.remoteId,
      state: "stale",
      updatedAt: Date.now(),
      lastClientActivityAt: Date.now(),
    });
    await expect(
      server.handle({
        method: "sessions.remoteStatus",
        params: { identifier: pairingCreate.result.remoteId },
      }),
    ).resolves.toMatchObject({
      ok: true,
      result: {
        remoteId: pairingCreate.result.remoteId,
        control: { state: "quarantined", reason: "gateway heartbeat stale" },
        diagnostics: {
          cause: "gateway heartbeat stale",
          recoverable: true,
          recommendedAction: `Run /remote resume ${pairingCreate.result.remoteId}`,
        },
      },
    });
    server.updateGatewaySessionHealth({
      connectionId: "gateway-healthy-1",
      sessionId: pairedBootstrap.result.session.id,
      remoteId: pairingCreate.result.remoteId,
      state: "healthy",
      updatedAt: Date.now(),
      lastClientActivityAt: Date.now(),
    });
    await expect(
      server.handle({
        method: "sessions.remoteStatus",
        params: { identifier: pairingCreate.result.remoteId },
      }),
    ).resolves.toMatchObject({
      ok: true,
      result: {
        remoteId: pairingCreate.result.remoteId,
        control: { state: "quarantined", reason: "gateway heartbeat stale" },
        diagnostics: {
          cause: "gateway heartbeat stale",
          recoverable: true,
          recommendedAction: `Run /remote resume ${pairingCreate.result.remoteId}`,
        },
      },
    });
    await expect(
      server.handle({
        method: "sessions.remoteControl",
        params: { identifier: pairingCreate.result.remoteId, action: "resume" },
      }),
    ).resolves.toMatchObject({
      ok: true,
      result: {
        remoteId: pairingCreate.result.remoteId,
        control: { state: "active" },
      },
    });
    await expect(
      server.handle({
        method: "sessions.remoteControl",
        params: { identifier: pairingCreate.result.remoteId, action: "pause", reason: "gateway unhealthy" },
      }),
    ).resolves.toMatchObject({
      ok: true,
      result: {
        remoteId: pairingCreate.result.remoteId,
        control: { state: "paused", reason: "gateway unhealthy" },
        activeRun: {
          id: remoteRun.id,
          title: "Remote run",
          status: "paused",
        },
      },
    });
    const secondRemoteBootstrap = await server.handle({
      method: "sessions.bootstrap",
      params: { title: "Second gateway remote", remoteId: "device-server-2", remoteSource: "gateway", agentId: "gateway" },
    });
    expect(secondRemoteBootstrap).toMatchObject({
      ok: true,
      result: {
        session: {
          metadata: { remoteId: "device-server-2", remoteSource: "gateway", agentId: "gateway" },
        },
      },
    });
    const inventoryPairing = await server.handle({
      method: "pairing.create",
      params: { remoteSource: "gateway" },
    });
    if (!inventoryPairing.ok) {
      throw new Error("expected inventory pairing seed");
    }
    await expect(
      server.handle({
        method: "pairing.resolve",
        params: { ticketId: inventoryPairing.result.id, decision: "approved", resolvedBy: "tester" },
      }),
    ).resolves.toMatchObject({
      ok: true,
      result: { id: inventoryPairing.result.id, status: "approved" },
    });
    const remoteInventory = await server.handle({
      method: "sessions.remoteInventory",
    });
    expect(remoteInventory.ok).toBe(true);
    if (!remoteInventory.ok) {
      throw new Error("expected remote inventory");
    }
    expect(remoteInventory.result.items).toEqual(expect.arrayContaining([
      expect.objectContaining({
        remoteId: pairingCreate.result.remoteId,
        remoteSource: "gateway",
        control: { state: "paused", reason: "gateway unhealthy" },
        pairing: expect.objectContaining({ code: pairingCreate.result.code, status: "redeemed" }),
        session: expect.objectContaining({ id: pairedBootstrap.result.session.id, status: pairedBootstrap.result.session.status }),
        activeRun: expect.objectContaining({ id: remoteRun.id, status: "paused", title: "Remote run" }),
        activeBackgroundTask: expect.objectContaining({ id: remoteTask.id, status: "running", title: "Remote task" }),
        gateway: expect.objectContaining({ state: "healthy" }),
      }),
      expect.objectContaining({
        remoteId: inventoryPairing.result.remoteId,
        remoteSource: "gateway",
        pairing: expect.objectContaining({ status: "approved", code: inventoryPairing.result.code }),
      }),
    ]));
    const filteredRemoteInventory = await server.handle({
      method: "sessions.remoteInventory",
      params: { remoteSource: "gateway" },
    });
    expect(filteredRemoteInventory.ok).toBe(true);
    if (!filteredRemoteInventory.ok) {
      throw new Error("expected filtered remote inventory");
    }
    expect(filteredRemoteInventory.result.items).toEqual(expect.arrayContaining([
      expect.objectContaining({ remoteId: pairingCreate.result.remoteId, remoteSource: "gateway" }),
      expect.objectContaining({ remoteId: inventoryPairing.result.remoteId, remoteSource: "gateway" }),
    ]));
    await expect(server.handle({ method: "sessions.platformInventory" })).resolves.toMatchObject({
      ok: true,
      result: {
        items: [
          {
            platform: "gateway",
            remoteCount: 4,
            control: { state: "mixed" },
          },
        ],
      },
    });
    await expect(
      server.handle({ method: "sessions.platformStatus", params: { platform: "gateway" } }),
    ).resolves.toMatchObject({
      ok: true,
      result: {
        platform: "gateway",
        remoteCount: 4,
        control: { state: "mixed" },
        remotes: expect.arrayContaining([
          expect.objectContaining({
            remoteId: pairingCreate.result.remoteId,
            control: { state: "paused", reason: "gateway unhealthy" },
          }),
          expect.objectContaining({
            remoteId: "device-server-2",
            control: { state: "active" },
          }),
          expect.objectContaining({
            remoteId: inventoryPairing.result.remoteId,
          }),
        ]),
      },
    });
    await expect(
      server.handle({ method: "sessions.platformControl", params: { platform: "gateway", action: "pause", reason: "platform unhealthy" } }),
    ).resolves.toMatchObject({
      ok: true,
      result: {
        platform: "gateway",
        remoteCount: 4,
        action: "pause",
        control: { state: "paused", reason: "platform unhealthy" },
        results: expect.arrayContaining([
          {
            remoteId: pairingCreate.result.remoteId,
            ok: true,
            control: { state: "paused", reason: "platform unhealthy" },
          },
          {
            remoteId: "device-server-1",
            ok: false,
            error: "no active run for remote or session: device-server-1",
          },
          {
            remoteId: "device-server-2",
            ok: false,
            error: "no active run for remote or session: device-server-2",
          },
          {
            remoteId: inventoryPairing.result.remoteId,
            ok: false,
            error: `unknown remote or session: ${inventoryPairing.result.remoteId}`,
          },
        ]),
      },
    });
    await expect(server.handle({ method: "sessions.platformInventory" })).resolves.toMatchObject({
      ok: true,
      result: {
        items: [
          {
            platform: "gateway",
            remoteCount: 4,
            control: { state: "paused", reason: "platform unhealthy" },
          },
        ],
      },
    });
    await expect(
      server.handle({ method: "sessions.platformStatus", params: { platform: "gateway" } }),
    ).resolves.toMatchObject({
      ok: true,
      result: {
        platform: "gateway",
        remoteCount: 4,
        control: { state: "paused", reason: "platform unhealthy" },
        remotes: expect.arrayContaining([
          expect.objectContaining({
            remoteId: pairingCreate.result.remoteId,
            control: { state: "paused", reason: "platform unhealthy" },
          }),
          expect.objectContaining({
            remoteId: "device-server-2",
            control: { state: "active" },
          }),
        ]),
      },
    });
    const restartedServer = new OperatorControlPlaneServer({ runtime });
    await expect(
      restartedServer.handle({ method: "sessions.platformStatus", params: { platform: "gateway" } }),
    ).resolves.toMatchObject({
      ok: true,
      result: {
        platform: "gateway",
        remoteCount: 4,
        control: { state: "paused", reason: "platform unhealthy" },
      },
    });
    await expect(
      server.handle({ method: "sessions.platformControl", params: { platform: "gateway", action: "resume" } }),
    ).resolves.toMatchObject({
      ok: true,
      result: {
        platform: "gateway",
        remoteCount: 4,
        action: "resume",
        control: { state: "mixed" },
        results: expect.arrayContaining([
          {
            remoteId: pairingCreate.result.remoteId,
            ok: true,
            control: { state: "paused", reason: "platform unhealthy" },
          },
          {
            remoteId: "device-server-1",
            ok: true,
            control: { state: "active" },
          },
          {
            remoteId: "device-server-2",
            ok: true,
            control: { state: "active" },
          },
          {
            remoteId: inventoryPairing.result.remoteId,
            ok: false,
            error: `unknown remote or session: ${inventoryPairing.result.remoteId}`,
          },
        ]),
      },
    });

    const driftingRootDir = await makeTempDir();
    const driftingRuntime = new StandaloneOperatorRuntime({
      rootDir: driftingRootDir,
      backgroundTaskIsProcessRunning: () => false,
    });
    const driftingServer = new OperatorControlPlaneServer({ runtime: driftingRuntime });
    const driftingBootstrap = await driftingServer.handle({
      method: "sessions.bootstrap",
      params: { title: "Drifting remote", remoteId: "device-drift-1", remoteSource: "gateway", agentId: "gateway" },
    });
    if (!driftingBootstrap.ok) {
      throw new Error("expected drifting bootstrap");
    }
    const driftingTask = await driftingRuntime.startBackgroundTask({
      sessionId: driftingBootstrap.result.session.id,
      title: "Drifting remote task",
      command: "printf 'drift'",
      kind: "task",
    });
    await driftingRuntime.backgroundTasks.executionService.writeState(driftingTask, {
      version: 1,
      taskId: driftingTask.id,
      kind: "task",
      status: "running",
      pid: driftingTask.execution.processId ?? 9999,
      startedAt: driftingTask.execution.startedAt ?? driftingTask.updatedAt,
      updatedAt: driftingTask.updatedAt,
      outputFile: driftingTask.execution.outputFile,
      cwd: driftingTask.cwd,
      command: driftingTask.command,
    });
    await driftingRuntime.syncBackgroundTask(driftingTask.id);
    const degradedRemoteStatus = await driftingServer.handle({
      method: "sessions.remoteStatus",
      params: { identifier: "device-drift-1" },
    });
    expect(degradedRemoteStatus).toMatchObject({
      ok: true,
      result: {
        remoteId: "device-drift-1",
        control: { state: "degraded", reason: "background task failed" },
        diagnostics: {
          cause: "background task failed",
          recoverable: false,
          taskId: driftingTask.id,
        },
      },
    });
    await expect(
      driftingServer.handle({
        method: "sessions.remoteRepair",
        params: { identifier: "device-drift-1", action: "recover-background-tasks", reasons: ["missing-process"] },
      }),
    ).resolves.toMatchObject({
      ok: true,
      result: {
        remoteId: "device-drift-1",
        repaired: {
          action: "recover-background-tasks",
          changed: false,
          results: [],
        },
        control: { state: "active" },
      },
    });

    const breakerRootDir = await makeTempDir();
    const breakerRuntime = new StandaloneOperatorRuntime({
      rootDir: breakerRootDir,
      backgroundTaskIsProcessRunning: () => false,
    });
    const breakerServer = new OperatorControlPlaneServer({ runtime: breakerRuntime });
    const breakerOne = await breakerServer.handle({
      method: "sessions.bootstrap",
      params: { title: "Breaker remote 1", remoteId: "device-breaker-1", remoteSource: "gateway", agentId: "gateway" },
    });
    const breakerTwo = await breakerServer.handle({
      method: "sessions.bootstrap",
      params: { title: "Breaker remote 2", remoteId: "device-breaker-2", remoteSource: "gateway", agentId: "gateway" },
    });
    const breakerThree = await breakerServer.handle({
      method: "sessions.bootstrap",
      params: { title: "Breaker remote 3", remoteId: "device-breaker-3", remoteSource: "gateway", agentId: "gateway" },
    });
    [breakerOne, breakerTwo, breakerThree].forEach((response) => expect(response).toMatchObject({ ok: true }));
    if (!breakerOne || !breakerTwo || !breakerThree) {
      throw new Error("expected breaker bootstraps");
    }
    const breakerRunOne = await breakerRuntime.startRun({ sessionId: breakerOne.result.session.id, title: "Breaker run 1" });
    const breakerRunTwo = await breakerRuntime.startRun({ sessionId: breakerTwo.result.session.id, title: "Breaker run 2" });
    const breakerRunThree = await breakerRuntime.startRun({ sessionId: breakerThree.result.session.id, title: "Breaker run 3" });
    const breakerTaskOne = await breakerRuntime.startBackgroundTask({
      sessionId: breakerOne.result.session.id,
      title: "Breaker task 1",
      command: "sleep 5",
      kind: "task",
    });
    const breakerTaskTwo = await breakerRuntime.startBackgroundTask({
      sessionId: breakerTwo.result.session.id,
      title: "Breaker task 2",
      command: "sleep 5",
      kind: "task",
    });
    const breakerTaskThree = await breakerRuntime.startBackgroundTask({
      sessionId: breakerThree.result.session.id,
      title: "Breaker task 3",
      command: "sleep 5",
      kind: "task",
    });
    await breakerRuntime.backgroundTasks.executionService.writeState(breakerTaskOne, {
      version: 1,
      taskId: breakerTaskOne.id,
      kind: "task",
      status: "running",
      pid: breakerTaskOne.execution.processId ?? 9999,
      startedAt: breakerTaskOne.execution.startedAt ?? breakerTaskOne.updatedAt,
      updatedAt: breakerTaskOne.updatedAt,
      outputFile: breakerTaskOne.execution.outputFile,
      cwd: breakerTaskOne.cwd,
      command: breakerTaskOne.command,
    });
    await expect(breakerServer.handle({ method: "sessions.platformInventory" })).resolves.toMatchObject({
      ok: true,
      result: {
        items: [
          {
            platform: "gateway",
            remoteCount: 3,
            control: { state: "mixed" },
          },
        ],
      },
    });

    await breakerRuntime.backgroundTasks.executionService.writeState(breakerTaskTwo, {
      version: 1,
      taskId: breakerTaskTwo.id,
      kind: "task",
      status: "running",
      pid: breakerTaskTwo.execution.processId ?? 9999,
      startedAt: breakerTaskTwo.execution.startedAt ?? breakerTaskTwo.updatedAt,
      updatedAt: breakerTaskTwo.updatedAt,
      outputFile: breakerTaskTwo.execution.outputFile,
      cwd: breakerTaskTwo.cwd,
      command: breakerTaskTwo.command,
    });
    await expect(breakerServer.handle({ method: "sessions.platformInventory" })).resolves.toMatchObject({
      ok: true,
      result: {
        items: [
          {
            platform: "gateway",
            remoteCount: 3,
            control: {
              state: "degraded",
              source: "automatic",
              failureCount: 2,
              threshold: 2,
              reason: expect.stringContaining("background task missing-process"),
            },
          },
        ],
      },
    });
    await expect(
      breakerServer.handle({ method: "sessions.platformStatus", params: { platform: "gateway" } }),
    ).resolves.toMatchObject({
      ok: true,
      result: {
        platform: "gateway",
        remoteCount: 3,
        control: {
          state: "degraded",
          source: "automatic",
          failureCount: 2,
          threshold: 2,
        },
      },
    });

    await breakerRuntime.backgroundTasks.executionService.writeState(breakerTaskThree, {
      version: 1,
      taskId: breakerTaskThree.id,
      kind: "task",
      status: "running",
      pid: breakerTaskThree.execution.processId ?? 9999,
      startedAt: breakerTaskThree.execution.startedAt ?? breakerTaskThree.updatedAt,
      updatedAt: breakerTaskThree.updatedAt,
      outputFile: breakerTaskThree.execution.outputFile,
      cwd: breakerTaskThree.cwd,
      command: breakerTaskThree.command,
    });
    await expect(breakerServer.handle({ method: "sessions.remoteStatus", params: { identifier: "device-breaker-3" } })).resolves.toMatchObject({
      ok: true,
      result: {
        remoteId: "device-breaker-3",
        control: { state: "degraded", reason: expect.stringContaining("automatic retryable failures 3/3") },
        activeRun: { id: breakerRunThree.id, status: "paused" },
      },
    });
    await expect(breakerServer.handle({ method: "sessions.platformInventory" })).resolves.toMatchObject({
      ok: true,
      result: {
        items: [
          {
            platform: "gateway",
            remoteCount: 3,
            control: {
              state: "paused",
              source: "automatic",
              failureCount: 3,
              threshold: 3,
              reason: expect.stringContaining("background task missing-process"),
            },
          },
        ],
      },
    });
    await expect(breakerRuntime.getRun(breakerRunOne.id)).resolves.toMatchObject({
      id: breakerRunOne.id,
      status: "paused",
      metadata: {
        remoteControlSource: "platform-breaker",
        remoteControlAction: "pause",
      },
    });
    await expect(breakerRuntime.getRun(breakerRunTwo.id)).resolves.toMatchObject({
      id: breakerRunTwo.id,
      status: "paused",
      metadata: {
        remoteControlSource: "platform-breaker",
        remoteControlAction: "pause",
      },
    });
    await expect(breakerRuntime.getRun(breakerRunThree.id)).resolves.toMatchObject({
      id: breakerRunThree.id,
      status: "paused",
      metadata: {
        remoteControlSource: "platform-breaker",
        remoteControlAction: "pause",
      },
    });

    await breakerRuntime.syncBackgroundTask(breakerTaskOne.id);
    await breakerRuntime.syncBackgroundTask(breakerTaskTwo.id);
    await breakerRuntime.syncBackgroundTask(breakerTaskThree.id);
    await expect(breakerServer.handle({ method: "sessions.platformInventory" })).resolves.toMatchObject({
      ok: true,
      result: {
        items: [
          {
            platform: "gateway",
            remoteCount: 3,
            control: { state: "paused" },
          },
        ],
      },
    });
    const restartedBreakerServer = new OperatorControlPlaneServer({ runtime: breakerRuntime });
    await expect(restartedBreakerServer.handle({ method: "sessions.platformInventory" })).resolves.toMatchObject({
      ok: true,
      result: {
        items: [
          {
            platform: "gateway",
            remoteCount: 3,
            control: { state: "paused" },
          },
        ],
      },
    });
    await expect(
      server.handle({
        method: "sessions.bootstrap",
        params: { pairingCode: pairingCreate.result.code },
      }),
    ).resolves.toEqual({
      ok: false,
      error: { code: "INVALID_REQUEST", message: `pairing ticket is redeemed: ${pairingCreate.result.code}` },
    });
    const sessionsList = await server.handle({ method: "sessions.list" });
    expect(sessionsList.ok).toBe(true);
    if (sessionsList.ok) {
      expect(sessionsList.result.some((item) => item.id === session.id)).toBe(true);
      expect(sessionsList.result.some((item) => item.id === childSession.id)).toBe(true);
    }
    const transcriptResponse = await server.handle({
      method: "transcript.get",
      params: { sessionId: session.id, includeHeader: false },
    });
    expect(transcriptResponse.ok).toBe(true);
    if (transcriptResponse.ok) {
      expect(
        transcriptResponse.result.some(
          (record) => "message" in record && record.message.role === "user" && record.message.content === "check deploy",
        ),
      ).toBe(true);
    }
    await expect(server.handle({ method: "approvals.list" })).resolves.toMatchObject({
      ok: true,
      result: expect.arrayContaining([
        expect.objectContaining({ id: approval.id }),
        expect.objectContaining({ title: "Remote approval" }),
      ]),
    });
    await expect(server.handle({ method: "approvals.list", params: { sessionId: session.id } })).resolves.toMatchObject({
      ok: true,
      result: [{ id: approval.id, sessionId: session.id }],
    });
    await expect(server.handle({ method: "trajectories.list", params: { sessionId: session.id } })).resolves.toMatchObject({
      ok: true,
      result: [{ sessionId: session.id }],
    });
    await expect(server.handle({ method: "replays.get", params: { sessionId: session.id } })).resolves.toMatchObject({
      ok: true,
      result: { sessionId: session.id, trajectoryIds: [recorded.trajectory.id], eventCount: 4 },
    });
    await expect(server.handle({ method: "replays.list" })).resolves.toMatchObject({
      ok: true,
      result: [expect.objectContaining({ sessionId: session.id, trajectoryIds: [recorded.trajectory.id] })],
    });
    await expect(server.handle({ method: "memory.recall", params: { query: "deploy" } })).resolves.toMatchObject({
      ok: true,
      result: {
        query: "deploy",
        memories: [expect.objectContaining({ item: expect.objectContaining({ sourceSessionId: session.id }) })],
        hits: expect.any(Array),
      },
    });
    if (!recorded.skillCandidate) {
      throw new Error("expected skill candidate from recorded turn");
    }
    await expect(server.handle({ method: "skills.candidates.list" })).resolves.toMatchObject({
      ok: true,
      result: [{ id: recorded.skillCandidate.id, status: "candidate" }],
    });
    await expect(
      server.handle({
        method: "skills.candidates.review",
        params: { candidateId: recorded.skillCandidate.id, status: "reviewed", reviewNote: "looks reusable" },
      }),
    ).resolves.toMatchObject({
      ok: true,
      result: { id: recorded.skillCandidate.id, status: "reviewed", reviewNote: "looks reusable" },
    });
    await expect(server.handle({ method: "skills.promote", params: { candidateId: recorded.skillCandidate.id } })).resolves.toMatchObject({
      ok: true,
      result: { title: recorded.skillCandidate.title, sourceCandidateId: recorded.skillCandidate.id },
    });
    await expect(server.handle({ method: "skills.list" })).resolves.toMatchObject({
      ok: true,
      result: [{ title: recorded.skillCandidate.title, sourceCandidateId: recorded.skillCandidate.id }],
    });
    const promotedSkills = await runtime.listPromotedSkills();
    const promotedSkill = promotedSkills[0];
    if (!promotedSkill) {
      throw new Error("expected promoted skill");
    }
    await expect(
      server.handle({
        method: "skills.executable.create",
        params: {
          promotedSkillId: promotedSkill.id,
          steps: [
            { id: "summary", title: "Summarize", kind: "summary", content: "deploy summary" },
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
        },
      }),
    ).resolves.toMatchObject({
      ok: true,
      result: { promotedSkillId: promotedSkill.id, usage: { executionCount: 0 } },
    });
    const executableSkills = await runtime.listExecutableSkills();
    const executableSkill = executableSkills[0];
    if (!executableSkill) {
      throw new Error("expected executable skill");
    }
    await expect(server.handle({ method: "skills.executable.list" })).resolves.toMatchObject({
      ok: true,
      result: [{ id: executableSkill.id, promotedSkillId: promotedSkill.id }],
    });
    await expect(
      server.handle({ method: "skills.executable.run", params: { skillId: executableSkill.id, sessionId: session.id, parentRunId: run?.id } }),
    ).resolves.toMatchObject({
      ok: true,
      result: {
        skillId: executableSkill.id,
        parentRunId: run?.id,
        stepResults: [
          expect.objectContaining({ stepId: "summary", output: "deploy summary" }),
          expect.objectContaining({ stepId: "x-post", commentOutputs: expect.any(Array) }),
          expect.objectContaining({ stepId: "command", output: "verified", subagentRunId: expect.any(String) }),
        ],
      },
    });
    await expect(server.handle({ method: "skills.executable.runs", params: { skillId: executableSkill.id } })).resolves.toMatchObject({
      ok: true,
      result: [expect.objectContaining({ skillId: executableSkill.id, sourceTrajectoryIds: [recorded.trajectory.id] })],
    });
    const reviewedExportFile = path.join(runtime.rootDir, "reviewed-export.json");
    await expect(
      server.handle({
        method: "trajectories.review",
        params: {
          trajectoryId: recorded.trajectory.id,
          status: "approved",
          reviewedBy: "reviewer",
          reviewNote: "safe subset",
          redactedObservations: [{ ts: 11, source: "operator", summary: "reviewed observation" }],
          redactedActions: [{ ts: 12, tool: "assistant-response", summary: "reviewed action" }],
          redactedTranscript: [{ ts: 13, role: "assistant", content: "reviewed assistant" }],
        },
      }),
    ).resolves.toMatchObject({
      ok: true,
      result: {
        id: recorded.trajectory.id,
        review: {
          status: "approved",
          reviewedBy: "reviewer",
          reviewNote: "safe subset",
          reviewedAt: expect.any(String),
          redactedObservations: [{ ts: 11, source: "operator", summary: "reviewed observation" }],
          redactedActions: [{ ts: 12, tool: "assistant-response", summary: "reviewed action" }],
          redactedTranscript: [{ ts: 13, role: "assistant", content: "reviewed assistant" }],
        },
      },
    });
    await expect(server.handle({ method: "trajectories.reviewed" })).resolves.toMatchObject({
      ok: true,
      result: [expect.objectContaining({ id: recorded.trajectory.id, review: expect.objectContaining({ status: "approved" }) })],
    });
    await expect(
      server.handle({
        method: "training.exports.create",
        params: {
          reviewedBy: "operator",
          purpose: "local fine-tuning",
          outputFile: reviewedExportFile,
          modes: ["sft", "rl"],
        },
      }),
    ).resolves.toMatchObject({
      ok: true,
      result: {
        reviewedBy: "operator",
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
            events: expect.arrayContaining([
              expect.objectContaining({ kind: "observation", summary: "reviewed observation" }),
              expect.objectContaining({ kind: "action", summary: "reviewed action" }),
              expect.objectContaining({ kind: "transcript", content: "reviewed assistant" }),
            ]),
          }),
        ],
      },
    });
    await expect(fs.readFile(reviewedExportFile, "utf8")).resolves.toContain("reviewed assistant");
    const backgroundTaskStart = await server.handle({
      method: "background.tasks.start",
      params: {
        sessionId: session.id,
        title: "Collect logs",
        command: "printf 'line-1\nline-2\n'",
        kind: "task",
      },
    });
    expect(backgroundTaskStart).toMatchObject({
      ok: true,
      result: { sessionId: session.id, title: "Collect logs", kind: "task", status: "running" },
    });
    const backgroundTask = backgroundTaskStart.ok ? backgroundTaskStart.result : undefined;
    expect(backgroundTask).toBeDefined();
    if (!backgroundTask) {
      throw new Error("expected background task to be created");
    }
    await expect(server.handle({ method: "background.tasks.get", params: { taskId: backgroundTask.id } })).resolves.toMatchObject({
      ok: true,
      result: { id: backgroundTask.id, sessionId: session.id, kind: "task", status: "running" },
    });
    await expect(server.handle({ method: "background.tasks.list", params: { sessionId: session.id } })).resolves.toMatchObject({
      ok: true,
      result: [expect.objectContaining({ id: backgroundTask.id, sessionId: session.id })],
    });
    await expect(server.handle({ method: "background.tasks.active", params: { sessionId: session.id } })).resolves.toMatchObject({
      ok: true,
      result: expect.objectContaining({ id: backgroundTask.id, sessionId: session.id }),
    });
    await expect(server.handle({ method: "background.tasks.state", params: { taskId: backgroundTask.id } })).resolves.toEqual({
      ok: false,
      error: { code: "NOT_FOUND", message: `unknown background task: ${backgroundTask.id}` },
    });
    await expect(server.handle({ method: "background.tasks.recover", params: { taskId: backgroundTask.id } })).resolves.toMatchObject({
      ok: true,
      result: {
        task: { id: backgroundTask.id, status: "failed" },
        reason: "unchanged",
        changed: false,
      },
    });
    await expect(
      server.handle({ method: "background.tasks.recoverAll", params: { sessionId: session.id, reasons: ["missing-process", "unchanged"] } }),
    ).resolves.toEqual({
      ok: true,
      result: expect.arrayContaining([
        expect.objectContaining({ task: expect.objectContaining({ id: backgroundTask.id }), reason: "unchanged", changed: false }),
      ]),
    });
    await expect(server.handle({ method: "background.tasks.output", params: { taskId: backgroundTask.id } })).resolves.toEqual({
      ok: false,
      error: { code: "NOT_FOUND", message: `unknown background task: ${backgroundTask.id}` },
    });
    await runtime.backgroundTasks.executionService.writeState(backgroundTask, {
      version: 1,
      taskId: backgroundTask.id,
      kind: "task",
      status: "completed",
      pid: backgroundTask.execution.processId ?? 1234,
      startedAt: "2026-01-01T00:00:00.000Z",
      updatedAt: "2026-01-01T00:10:00.000Z",
      completedAt: "2026-01-01T00:10:00.000Z",
      exitCode: 0,
      outputFile: backgroundTask.execution.outputFile,
      cwd: backgroundTask.cwd,
      command: backgroundTask.command,
    });
    await runtime.backgroundTasks.executionService.writeOutput(backgroundTask, "line-1\nline-2\n");
    await expect(server.handle({ method: "background.tasks.state", params: { taskId: backgroundTask.id } })).resolves.toMatchObject({
      ok: true,
      result: { taskId: backgroundTask.id, status: "completed" },
    });
    await expect(server.handle({ method: "background.tasks.output", params: { taskId: backgroundTask.id, lineLimit: 1 } })).resolves.toEqual({
      ok: true,
      result: { taskId: backgroundTask.id, output: "line-2" },
    });
    await expect(server.handle({ method: "background.tasks.sync", params: { taskId: backgroundTask.id } })).resolves.toMatchObject({
      ok: true,
      result: { id: backgroundTask.id, status: "completed", execution: { exitCode: 0 } },
    });
    const monitorTaskStart = await server.handle({
      method: "background.tasks.start",
      params: {
        sessionId: session.id,
        title: "Watch logs",
        command: "tail -f app.log",
        kind: "monitor",
      },
    });
    expect(monitorTaskStart).toMatchObject({ ok: true, result: { kind: "monitor", status: "running" } });
    const monitorTask = monitorTaskStart.ok ? monitorTaskStart.result : undefined;
    expect(monitorTask).toBeDefined();
    if (!monitorTask) {
      throw new Error("expected monitor task to be created");
    }
    await runtime.backgroundTasks.executionService.writeState(monitorTask, {
      version: 1,
      taskId: monitorTask.id,
      kind: "monitor",
      status: "running",
      pid: monitorTask.execution.processId ?? 5678,
      startedAt: "2026-01-01T00:00:00.000Z",
      updatedAt: "2026-01-01T00:00:00.000Z",
      outputFile: monitorTask.execution.outputFile,
      cwd: monitorTask.cwd,
      command: monitorTask.command,
    });
    const originalKill = process.kill;
    process.kill = (() => true) as typeof process.kill;
    try {
      await expect(server.handle({ method: "background.tasks.cancel", params: { taskId: monitorTask.id } })).resolves.toMatchObject({
        ok: true,
        result: { id: monitorTask.id, status: "cancelled", execution: { signal: "SIGTERM" } },
      });
    } finally {
      process.kill = originalKill;
    }
    const trainingJobCreate = await server.handle({
      method: "training.jobs.create",
      params: { exportManifest, mode: "sft" },
    });
    expect(trainingJobCreate).toMatchObject({
      ok: true,
      result: { mode: "sft", status: "planned", runtime: { environment: "local-apple-silicon" } },
    });
    const trainingJob = trainingJobCreate.ok ? trainingJobCreate.result : undefined;
    expect(trainingJob).toBeDefined();
    if (!trainingJob) {
      throw new Error("expected training job to be created");
    }
    await expect(server.handle({ method: "training.jobs.get", params: { jobId: trainingJob.id } })).resolves.toMatchObject({
      ok: true,
      result: { id: trainingJob.id, mode: "sft", status: "planned" },
    });
    await expect(server.handle({ method: "training.jobs.list" })).resolves.toMatchObject({
      ok: true,
      result: [{ id: trainingJob.id, mode: "sft", status: "planned" }],
    });
    await expect(server.handle({ method: "training.jobs.update", params: { jobId: trainingJob.id, status: "ready" } })).resolves.toMatchObject({
      ok: true,
      result: { id: trainingJob.id, status: "ready" },
    });
    await expect(server.handle({ method: "training.jobs.prepare", params: { jobId: trainingJob.id } })).resolves.toMatchObject({
      ok: true,
      result: {
        id: trainingJob.id,
        status: "ready",
        execution: {
          workingDirectory: `training-jobs/${trainingJob.id}`,
          logFile: `training-jobs/${trainingJob.id}/logs/sft.log`,
          planFile: `training-jobs/${trainingJob.id}/plan.json`,
          launchScript: `training-jobs/${trainingJob.id}/run-sft.sh`,
          command: [
            "python3",
            "-m",
            "mlx_lm.lora",
            "--train",
            "--data",
            `training-jobs/${trainingJob.id}/dataset`,
            "--adapter-path",
            `training-jobs/${trainingJob.id}/artifacts`,
            "--learning-rate",
            "0.00001",
            "--batch-size",
            "1",
            "--iters",
            "3000",
          ],
          environment: { OPENCLAW_TRAINING_RUNTIME: "mlx" },
        },
      },
    });
    await expect(server.handle({ method: "training.jobs.plan", params: { jobId: trainingJob.id } })).resolves.toMatchObject({
      ok: true,
      result: {
        jobId: trainingJob.id,
        runtime: "mlx",
        outputPath: `training-jobs/${trainingJob.id}/artifacts/model.gguf`,
      },
    });
    await expect(server.handle({ method: "training.jobs.script", params: { jobId: trainingJob.id } })).resolves.toMatchObject({
      ok: true,
      result: {
        jobId: trainingJob.id,
        script: expect.stringContaining("mlx_lm.lora"),
      },
    });
    await expect(server.handle({ method: "training.jobs.state", params: { jobId: trainingJob.id } })).resolves.toEqual({
      ok: false,
      error: { code: "NOT_FOUND", message: `unknown training job: ${trainingJob.id}` },
    });
    await expect(server.handle({ method: "training.jobs.log", params: { jobId: trainingJob.id } })).resolves.toEqual({
      ok: false,
      error: { code: "NOT_FOUND", message: `unknown training job: ${trainingJob.id}` },
    });
    const preparedTrainingJob = await runtime.getTrainingJob(trainingJob.id);
    if (!preparedTrainingJob?.execution) {
      throw new Error("expected prepared training execution");
    }
    await runtime.trainingJobs.executionService.writeState(preparedTrainingJob, {
      version: 1,
      jobId: trainingJob.id,
      status: "completed",
      pid: 1234,
      startedAt: "2026-01-01T00:00:00.000Z",
      updatedAt: "2026-01-01T00:10:00.000Z",
      completedAt: "2026-01-01T00:10:00.000Z",
      exitCode: 0,
      logFile: preparedTrainingJob.execution.logFile,
      workingDirectory: preparedTrainingJob.execution.workingDirectory,
      command: preparedTrainingJob.execution.command ?? [],
    });
    await runtime.trainingJobs.executionService.writeLog(preparedTrainingJob, "line-1\nline-2\n");
    await expect(server.handle({ method: "training.jobs.state", params: { jobId: trainingJob.id } })).resolves.toMatchObject({
      ok: true,
      result: { jobId: trainingJob.id, status: "completed", pid: 1234 },
    });
    await expect(server.handle({ method: "training.jobs.log", params: { jobId: trainingJob.id, lineLimit: 1 } })).resolves.toEqual({
      ok: true,
      result: { jobId: trainingJob.id, log: "line-2" },
    });
    await expect(server.handle({ method: "training.jobs.sync", params: { jobId: trainingJob.id } })).resolves.toMatchObject({
      ok: true,
      result: {
        id: trainingJob.id,
        status: "completed",
        execution: { outputModelRef: `training-jobs/${trainingJob.id}/artifacts/model.gguf` },
      },
    });
    await expect(runtime.trainingJobs.markExecutionStarted(trainingJob.id, { pid: 1234 })).resolves.toMatchObject({
      id: trainingJob.id,
      status: "running",
      execution: { startedAt: expect.any(String), processId: 1234 },
    });
    await expect(server.handle({ method: "training.jobs.sync", params: { jobId: trainingJob.id } })).resolves.toMatchObject({
      ok: true,
      result: {
        id: trainingJob.id,
        status: "completed",
      },
    });
    await expect(server.handle({ method: "training.jobs.start", params: { jobId: trainingJob.id } })).resolves.toMatchObject({
      ok: true,
      result: {
        id: trainingJob.id,
        status: "running",
        execution: {
          startedAt: expect.any(String),
        },
      },
    });
    await expect(
      server.handle({
        method: "training.jobs.complete",
        params: { jobId: trainingJob.id, outputModelRef: "artifacts/models/sft.gguf" },
      }),
    ).resolves.toMatchObject({
      ok: true,
      result: {
        id: trainingJob.id,
        status: "completed",
        execution: { outputModelRef: "artifacts/models/sft.gguf" },
      },
    });
    const failedTrainingJobCreate = await server.handle({
      method: "training.jobs.create",
      params: { exportManifest, mode: "rl" },
    });
    expect(failedTrainingJobCreate).toMatchObject({ ok: true, result: { mode: "rl", status: "planned" } });
    const failedTrainingJob = failedTrainingJobCreate.ok ? failedTrainingJobCreate.result : undefined;
    expect(failedTrainingJob).toBeDefined();
    if (!failedTrainingJob) {
      throw new Error("expected failed training job to be created");
    }
    await expect(
      server.handle({
        method: "training.jobs.fail",
        params: { jobId: failedTrainingJob.id, reason: "reward model unavailable" },
      }),
    ).resolves.toMatchObject({
      ok: true,
      result: {
        id: failedTrainingJob.id,
        status: "failed",
        execution: {
          failureReason: "reward model unavailable",
          logFile: `training-jobs/${failedTrainingJob.id}/logs/rl.log`,
        },
      },
    });
    const pluginEntrypoint = path.join(runtime.rootDir, "server-plugin.mjs");
    await fs.writeFile(
      pluginEntrypoint,
      [
        "export default {",
        "  async activate() {",
        "    globalThis.__serverPluginActivations = (globalThis.__serverPluginActivations ?? 0) + 1;",
        "  },",
        "};",
        "",
      ].join("\n"),
      "utf8",
    );
    await expect(
      server.handle({
        method: "plugins.register",
        params: {
          manifest: {
            id: "server-plugin",
            version: "1.0.0",
            name: "Server Plugin",
            description: "Provides a plugin-exposed tool",
            entrypoint: pluginEntrypoint,
            capabilities: ["tool"],
          },
        },
      }),
    ).resolves.toMatchObject({
      ok: true,
      result: { manifest: { id: "server-plugin" }, enabled: true },
    });
    await expect(server.handle({ method: "plugins.get", params: { pluginId: "server-plugin" } })).resolves.toMatchObject({
      ok: true,
      result: { manifest: { id: "server-plugin" }, enabled: true },
    });
    await expect(server.handle({ method: "plugins.list", params: { capability: "tool" } })).resolves.toMatchObject({
      ok: true,
      result: [{ manifest: { id: "server-plugin" }, enabled: true }],
    });
    await expect(server.handle({ method: "plugins.update", params: { pluginId: "server-plugin", enabled: false } })).resolves.toMatchObject({
      ok: true,
      result: { manifest: { id: "server-plugin" }, enabled: false },
    });
    await expect(server.handle({ method: "plugins.activate", params: { pluginId: "server-plugin" } })).resolves.toMatchObject({
      ok: true,
      result: { pluginId: "server-plugin", activated: true },
    });
    expect((globalThis as Record<string, unknown>).__serverPluginActivations).toBe(1);
    const consentCreate = await server.handle({
      method: "capture.consents.create",
      params: { tier: "operator", purpose: "capture workflow" },
    });
    expect(consentCreate).toMatchObject({ ok: true, result: { tier: "operator", purpose: "capture workflow" } });
    const consent = consentCreate.ok ? consentCreate.result : undefined;
    expect(consent).toBeDefined();
    if (!consent) {
      throw new Error("expected capture consent to be created");
    }
    await expect(server.handle({ method: "capture.consents.list" })).resolves.toMatchObject({
      ok: true,
      result: [{ id: consent.id, tier: "operator" }],
    });
    await expect(
      server.handle({
        method: "capture.browser.record",
        params: {
          sessionId: session.id,
          pageUrl: "https://example.test/deploy",
          pageTitle: "Deploy",
          visibleIndicator: true,
          ts: 1,
          action: { kind: "click", target: "Deploy", ts: 2 },
        },
      }),
    ).resolves.toMatchObject({ ok: true, result: { recorded: true, trajectory: { sessionId: session.id } } });
    await expect(
      server.handle({
        method: "capture.device.record",
        params: {
          sessionId: session.id,
          deviceId: "sim-1",
          platform: "ios",
          appId: "mobile-app",
          appName: "Mobile App",
          visibleIndicator: true,
          ts: 3,
          gesture: { kind: "tap", target: "Pay", ts: 4 },
        },
      }),
    ).resolves.toMatchObject({ ok: true, result: { recorded: true, trajectory: { sessionId: session.id } } });
    await expect(
      server.handle({
        method: "capture.os.observe",
        params: {
          sessionId: session.id,
          appId: "terminal",
          visibleIndicator: true,
          ts: 5,
          event: "command-ran",
          commandSummary: "pnpm build",
        },
      }),
    ).resolves.toMatchObject({ ok: true, result: { recorded: true, trajectory: { sessionId: session.id } } });
    await expect(server.handle({ method: "capture.consents.revoke", params: { grantId: consent.id } })).resolves.toMatchObject({
      ok: true,
      result: { id: consent.id, revokedAt: expect.any(String) },
    });
    await expect(server.handle({ method: "runs.get", params: { runId: run.id } })).resolves.toMatchObject({
      ok: true,
      result: { id: run.id, sessionId: session.id, status: "running", metadata: { source: "rpc" } },
    });
    const runsList = await server.handle({ method: "runs.list", params: { sessionId: session.id } });
    expect(runsList.ok).toBe(true);
    if (runsList.ok && run && subagent) {
      expect(runsList.result.some((item) => item.id === run.id && item.sessionId === session.id && item.status === "running")).toBe(true);
      expect(runsList.result.some((item) => item.id === subagent.runId && item.status === "pending")).toBe(true);
    }
    if (!run || !subagent) {
      throw new Error("expected orchestration records to be created");
    }
    await expect(server.handle({ method: "runs.active", params: { sessionId: session.id } })).resolves.toMatchObject({
      ok: true,
      result: expect.objectContaining({ id: run.id, sessionId: session.id }),
    });
    await expect(server.handle({ method: "runs.events", params: { sessionId: session.id, family: "run" } })).resolves.toMatchObject({
      ok: true,
      result: expect.arrayContaining([
        expect.objectContaining({ type: "run.started" }),
        expect.objectContaining({
          type: "run.progress",
          payload: expect.objectContaining({ sessionId: session.id, runId: run.id, message: "Collecting deploy evidence." }),
        }),
      ]),
    });
    await expect(
      server.handle({
        method: "runs.update",
        params: { runId: run.id, status: "paused", metadata: { reason: "awaiting approval" } },
      }),
    ).resolves.toMatchObject({
      ok: true,
      result: { id: run.id, status: "paused", metadata: { source: "rpc", reason: "awaiting approval" } },
    });
    await expect(server.handle({ method: "subagents.get", params: { runId: subagent.runId } })).resolves.toMatchObject({
      ok: true,
      result: { runId: subagent.runId, childSessionId: childSession.id, status: "pending" },
    });
    const subagentsList = await server.handle({ method: "subagents.list", params: { parentRunId: run.id } });
    expect(subagentsList.ok).toBe(true);
    if (subagentsList.ok) {
      expect(
        subagentsList.result.some(
          (item) => item.runId === subagent.runId && item.childSessionId === childSession.id && item.status === "pending",
        ),
      ).toBe(true);
    }
    await expect(server.handle({ method: "subagents.active" })).resolves.toMatchObject({
      ok: true,
      result: [{ runId: subagent.runId, status: "pending" }],
    });
    await expect(server.handle({ method: "subagents.update", params: { runId: subagent.runId, status: "running" } })).resolves.toMatchObject({
      ok: true,
      result: { runId: subagent.runId, status: "running" },
    });
    await expect(
      server.handle({
        method: "subagents.spawn",
        params: {
          sessionId: session.id,
          parentRunId: run.id,
          title: "Remote child work",
          agentId: "worker",
          remoteSource: "gateway",
          metadata: { source: "remote" },
        },
      }),
    ).resolves.toMatchObject({
      ok: true,
      result: {
        subagent: {
          parentRunId: run.id,
          sessionId: session.id,
          title: "Remote child work",
          status: "running",
        },
        childSession: {
          metadata: {
            title: "Remote child work",
            agentId: "worker",
            remoteSource: "gateway",
          },
        },
        childRun: {
          title: "Remote child work",
          status: "running",
          metadata: { source: "remote" },
        },
      },
    });
  }, 120_000);

  it("streams runtime events through subscriptions", async () => {
    const runtime = new StandaloneOperatorRuntime({ rootDir: await makeTempDir() });
    const events = subscribeRuntimeEvents(runtime, { replay: true });
    const iterator = events[Symbol.asyncIterator]();

    const session = await runtime.startSession({ title: "Events" });
    const first = await iterator.next();

    expect(first.done).toBe(false);
    expect(first.value).toMatchObject({ type: "session.started", payload: { id: session.id } });

    await iterator.return?.();
  });

  it("filters runtime events by session, run, and family", async () => {
    const runtime = new StandaloneOperatorRuntime({ rootDir: await makeTempDir() });
    const sessionA = await runtime.startSession({ title: "Session A" });
    const sessionB = await runtime.startSession({ title: "Session B" });
    const runA = await runtime.startRun({ sessionId: sessionA.id, title: "Run A" });
    const runB = await runtime.startRun({ sessionId: sessionB.id, title: "Run B" });
    const childA = await runtime.startSession({ title: "Child A", agentId: "worker" });
    const childB = await runtime.startSession({ title: "Child B", agentId: "worker" });
    await runtime.registerSubagent({
      sessionId: sessionA.id,
      parentRunId: runA.id,
      childSessionId: childA.id,
      title: "Subagent A",
    });
    await runtime.registerSubagent({
      sessionId: sessionB.id,
      parentRunId: runB.id,
      childSessionId: childB.id,
      title: "Subagent B",
    });

    runtime.publishRunProgress({
      runId: runA.id,
      sessionId: sessionA.id,
      phase: "planning",
      message: "Session A progress.",
    });
    runtime.publishRunProgress({
      runId: runB.id,
      sessionId: sessionB.id,
      phase: "planning",
      message: "Session B progress.",
    });
    const taskA = await runtime.createTask({ sessionId: sessionA.id, subject: "Inspect A", description: "Inspect session A." });
    await runtime.updateTask(taskA.id, { status: "completed" });
    await runtime.createTask({ sessionId: sessionB.id, subject: "Inspect B", description: "Inspect session B." });

    const sessionARunEvents = runtime.events.snapshot(buildRuntimeEventFilter({ sessionId: sessionA.id, family: "run" }));
    expect(sessionARunEvents).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ type: "run.started", payload: expect.objectContaining({ id: runA.id, sessionId: sessionA.id }) }),
        expect.objectContaining({
          type: "run.progress",
          payload: expect.objectContaining({ runId: runA.id, sessionId: sessionA.id, message: "Session A progress." }),
        }),
      ]),
    );
    expect(
      sessionARunEvents.some(
        (event) => event.type === "run.progress"
          && "payload" in event
          && typeof event.payload === "object"
          && event.payload != null
          && "runId" in event.payload
          && event.payload.runId === runB.id,
      ),
    ).toBe(false);

    const runOnlyEvents = runtime.events.snapshot(buildRuntimeEventFilter({ runId: runA.id, family: "run" }));
    expect(runOnlyEvents.every((event) => {
      if (event.type === "run.started" || event.type === "run.updated") {
        return "payload" in event && typeof event.payload === "object" && event.payload != null && "id" in event.payload && event.payload.id === runA.id;
      }
      if (event.type === "run.progress") {
        return "payload" in event && typeof event.payload === "object" && event.payload != null && "runId" in event.payload && event.payload.runId === runA.id;
      }
      return true;
    })).toBe(true);

    const approvalEvents = runtime.events.snapshot(buildRuntimeEventFilter({ family: "approval" }));
    expect(approvalEvents).toEqual([]);

    const subagentEvents = runtime.events.snapshot(buildRuntimeEventFilter({ family: "subagent", sessionId: sessionA.id }));
    expect(subagentEvents).toEqual([
      expect.objectContaining({ type: "subagent.registered", payload: expect.objectContaining({ sessionId: sessionA.id, title: "Subagent A" }) }),
    ]);

    const sessionATaskEvents = runtime.events.snapshot(buildRuntimeEventFilter({ family: "task", sessionId: sessionA.id }));
    expect(sessionATaskEvents).toEqual([
      expect.objectContaining({ type: "task.created", payload: expect.objectContaining({ id: taskA.id, sessionId: sessionA.id, subject: "Inspect A" }) }),
      expect.objectContaining({ type: "task.updated", payload: expect.objectContaining({ id: taskA.id, status: "completed", sessionId: sessionA.id }) }),
    ]);
  });

  it("handles cron methods", async () => {
    const rootDir = await makeTempDir();
    const runtime = new StandaloneOperatorRuntime({ rootDir });
    const delivery = new OperatorDeliveryService(rootDir, {
      fetchImpl: async () => new Response(null, { status: 204 }),
    });
    const cron = new OperatorCronService(rootDir, { runtime, delivery });
    const server = new OperatorControlPlaneServer({ runtime, cron });

    const created = await server.handle({
      method: "cron.create",
      params: {
        cron: "* * * * *",
        prompt: "run cron",
        recurring: false,
        delivery: {
          onSuccess: [{ kind: "local" }],
          onFailure: [{ kind: "webhook", url: "https://example.test/hook" }],
        },
        modelSelection: {
          primary: "claude-opus-4-7",
          fallbacks: ["claude-sonnet-4-6"],
        },
      },
    });
    expect(created).toMatchObject({
      ok: true,
      result: {
        prompt: "run cron",
        recurring: false,
        delivery: {
          onSuccess: [{ kind: "local" }],
          onFailure: [{ kind: "webhook", url: "https://example.test/hook" }],
        },
        modelSelection: {
          primary: "claude-opus-4-7",
          fallbacks: ["claude-sonnet-4-6"],
        },
      },
    });

    const list = await server.handle({ method: "cron.list" });
    expect(list).toMatchObject({
      ok: true,
      result: [{
        prompt: "run cron",
        delivery: { onSuccess: [{ kind: "local" }], onFailure: [{ kind: "webhook", url: "https://example.test/hook" }] },
        modelSelection: {
          primary: "claude-opus-4-7",
          fallbacks: ["claude-sonnet-4-6"],
        },
      }],
    });

    const tick = await server.handle({ method: "cron.tick", params: { nowMs: Date.now() + 120_000 } });
    expect(tick).toMatchObject({
      ok: true,
      result: [
        {
          status: "completed",
          outcome: "success",
          sessionId: expect.any(String),
          operatorRunId: expect.any(String),
          modelSelection: {
            primary: "claude-opus-4-7",
            fallbacks: ["claude-sonnet-4-6"],
            source: "job",
          },
          deliveryResults: [{ status: "sent", target: { kind: "local" } }],
        },
      ],
    });

    const runs = await server.handle({ method: "cron.runs" });
    expect(runs).toMatchObject({
      ok: true,
      result: [
        {
          status: "completed",
          outcome: "success",
          sessionId: expect.any(String),
          operatorRunId: expect.any(String),
          modelSelection: {
            primary: "claude-opus-4-7",
            fallbacks: ["claude-sonnet-4-6"],
            source: "job",
          },
          deliveryResults: [{ status: "sent", target: { kind: "local" } }],
        },
      ],
    });

    const jobs = await server.handle({ method: "cron.list" });
    expect(jobs).toMatchObject({ ok: true, result: [] });

    if (created.ok) {
      await expect(server.handle({ method: "cron.delete", params: { jobId: created.result.id } })).resolves.toEqual({
        ok: false,
        error: { code: "NOT_FOUND", message: `unknown cron job: ${created.result.id}` },
      });
    }
  }, 30_000);

  it("returns not-found and invalid-method responses", async () => {
    const runtime = new StandaloneOperatorRuntime({ rootDir: await makeTempDir() });
    const server = new OperatorControlPlaneServer({ runtime });

    await expect(server.handle({ method: "sessions.get", params: { sessionId: "missing" } })).resolves.toEqual({
      ok: false,
      error: { code: "NOT_FOUND", message: "unknown session: missing" },
    });
    await expect(server.handle({ method: "background.tasks.get", params: { taskId: "missing" } })).resolves.toEqual({
      ok: false,
      error: { code: "NOT_FOUND", message: "unknown background task: missing" },
    });
    await expect(server.handle({ method: "tasks.get", params: { taskId: "missing" } })).resolves.toEqual({
      ok: false,
      error: { code: "NOT_FOUND", message: "unknown task: missing" },
    });
    await expect(server.handle({ method: "tasks.update", params: { taskId: "missing", status: "completed" } })).resolves.toEqual({
      ok: false,
      error: { code: "NOT_FOUND", message: "unknown task: missing" },
    });
    await expect(server.handle({ method: "messages.get", params: { messageId: "missing" } })).resolves.toEqual({
      ok: false,
      error: { code: "NOT_FOUND", message: "unknown message: missing" },
    });
    await expect(server.handle({ method: "background.tasks.sync", params: { taskId: "missing" } })).resolves.toEqual({
      ok: false,
      error: { code: "NOT_FOUND", message: "unknown background task: missing" },
    });
    await expect(server.handle({ method: "background.tasks.cancel", params: { taskId: "missing" } })).resolves.toEqual({
      ok: false,
      error: { code: "NOT_FOUND", message: "unknown background task: missing" },
    });
    await expect(server.handle({ method: "background.tasks.state", params: { taskId: "missing" } })).resolves.toEqual({
      ok: false,
      error: { code: "NOT_FOUND", message: "unknown background task: missing" },
    });
    await expect(server.handle({ method: "background.tasks.recover", params: { taskId: "missing" } })).resolves.toEqual({
      ok: false,
      error: { code: "NOT_FOUND", message: "unknown background task: missing" },
    });
    await expect(server.handle({ method: "background.tasks.recoverAll", params: { reasons: ["missing-process"] } })).resolves.toMatchObject({
      ok: true,
      result: [],
    });
    await expect(server.handle({ method: "background.tasks.output", params: { taskId: "missing" } })).resolves.toEqual({
      ok: false,
      error: { code: "NOT_FOUND", message: "unknown background task: missing" },
    });
    await expect(server.handle({ method: "runs.get", params: { runId: "missing" } })).resolves.toEqual({
      ok: false,
      error: { code: "NOT_FOUND", message: "unknown run: missing" },
    });
    await expect(server.handle({ method: "runs.update", params: { runId: "missing", status: "running" } })).resolves.toEqual({
      ok: false,
      error: { code: "NOT_FOUND", message: "unknown run: missing" },
    });
    await expect(server.handle({ method: "runs.start", params: { sessionId: "missing", title: "No validation yet" } })).resolves.toMatchObject({
      ok: true,
      result: { sessionId: "missing", title: "No validation yet", status: "running" },
    });
    await expect(server.handle({ method: "subagents.get", params: { runId: "missing" } })).resolves.toEqual({
      ok: false,
      error: { code: "NOT_FOUND", message: "unknown subagent: missing" },
    });
    await expect(server.handle({ method: "subagents.update", params: { runId: "missing", status: "running" } })).resolves.toEqual({
      ok: false,
      error: { code: "NOT_FOUND", message: "unknown subagent: missing" },
    });
    await expect(
      server.handle({
        method: "subagents.register",
        params: { sessionId: "missing", parentRunId: "missing", childSessionId: "child", title: "No validation yet" },
      }),
    ).resolves.toMatchObject({
      ok: true,
      result: { parentRunId: "missing", childSessionId: "child", title: "No validation yet", status: "pending" },
    });
    await expect(server.handle({ method: "skills.promote", params: { candidateId: "missing" } })).resolves.toEqual({
      ok: false,
      error: { code: "NOT_FOUND", message: "unknown skill candidate: missing" },
    });
    await expect(server.handle({ method: "replays.get", params: { sessionId: "missing" } })).resolves.toEqual({
      ok: false,
      error: { code: "NOT_FOUND", message: "unknown replay session: missing" },
    });
    await expect(server.handle({ method: "training.jobs.get", params: { jobId: "missing" } })).resolves.toEqual({
      ok: false,
      error: { code: "NOT_FOUND", message: "unknown training job: missing" },
    });
    await expect(server.handle({ method: "training.jobs.update", params: { jobId: "missing", status: "ready" } })).resolves.toEqual({
      ok: false,
      error: { code: "NOT_FOUND", message: "unknown training job: missing" },
    });
    await expect(server.handle({ method: "training.jobs.prepare", params: { jobId: "missing" } })).resolves.toEqual({
      ok: false,
      error: { code: "NOT_FOUND", message: "unknown training job: missing" },
    });
    await expect(server.handle({ method: "training.jobs.start", params: { jobId: "missing" } })).resolves.toEqual({
      ok: false,
      error: { code: "NOT_FOUND", message: "unknown training job: missing" },
    });
    await expect(
      server.handle({ method: "training.jobs.complete", params: { jobId: "missing", outputModelRef: "model.gguf" } }),
    ).resolves.toEqual({
      ok: false,
      error: { code: "NOT_FOUND", message: "unknown training job: missing" },
    });
    await expect(server.handle({ method: "training.jobs.fail", params: { jobId: "missing", reason: "boom" } })).resolves.toEqual({
      ok: false,
      error: { code: "NOT_FOUND", message: "unknown training job: missing" },
    });
    await expect(server.handle({ method: "training.jobs.sync", params: { jobId: "missing" } })).resolves.toEqual({
      ok: false,
      error: { code: "NOT_FOUND", message: "unknown training job: missing" },
    });
    await expect(server.handle({ method: "training.jobs.plan", params: { jobId: "missing" } })).resolves.toEqual({
      ok: false,
      error: { code: "NOT_FOUND", message: "unknown training job: missing" },
    });
    await expect(server.handle({ method: "training.jobs.script", params: { jobId: "missing" } })).resolves.toEqual({
      ok: false,
      error: { code: "NOT_FOUND", message: "unknown training job: missing" },
    });
    await expect(server.handle({ method: "training.jobs.state", params: { jobId: "missing" } })).resolves.toEqual({
      ok: false,
      error: { code: "NOT_FOUND", message: "unknown training job: missing" },
    });
    await expect(server.handle({ method: "training.jobs.log", params: { jobId: "missing" } })).resolves.toEqual({
      ok: false,
      error: { code: "NOT_FOUND", message: "unknown training job: missing" },
    });
    await expect(server.handle({ method: "plugins.get", params: { pluginId: "missing" } })).resolves.toEqual({
      ok: false,
      error: { code: "NOT_FOUND", message: "unknown plugin: missing" },
    });
    await expect(server.handle({ method: "plugins.update", params: { pluginId: "missing", enabled: true } })).resolves.toEqual({
      ok: false,
      error: { code: "NOT_FOUND", message: "unknown plugin: missing" },
    });
    await expect(server.handle({ method: "plugins.activate", params: { pluginId: "missing" } })).resolves.toEqual({
      ok: false,
      error: { code: "INTERNAL_ERROR", message: "Unknown plugin: missing" },
    });
    await expect(server.handle({ method: "capture.consents.revoke", params: { grantId: "missing" } })).resolves.toEqual({
      ok: false,
      error: { code: "NOT_FOUND", message: "unknown capture consent: missing" },
    });
    await expect(server.handle({ method: "sessions.remoteStatus", params: { identifier: "missing" } })).resolves.toEqual({
      ok: false,
      error: { code: "NOT_FOUND", message: "unknown remote or session: missing" },
    });
    await expect(
      server.handle({
        method: "sessions.remoteControl",
        params: { identifier: "missing", action: "pause" },
      }),
    ).resolves.toEqual({
      ok: false,
      error: { code: "NOT_FOUND", message: "unknown remote or session: missing" },
    });
    await expect(
      server.handle({
        method: "sessions.remoteRepair",
        params: { identifier: "missing", action: "recover-background-tasks" },
      }),
    ).resolves.toEqual({
      ok: false,
      error: { code: "NOT_FOUND", message: "unknown remote or session: missing" },
    });
    await expect(
      server.handle({
        method: "subagents.spawn",
        params: { sessionId: "missing", parentRunId: "run-missing", title: "Spawn" },
      }),
    ).resolves.toEqual({
      ok: false,
      error: { code: "NOT_FOUND", message: "unknown session: missing" },
    });
    await expect(server.handle({ method: "nope" })).resolves.toEqual({
      ok: false,
      error: { code: "INVALID_REQUEST", message: "unknown method: nope" },
    });
  });
});
