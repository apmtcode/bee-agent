import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { OperatorControlPlaneServer } from "./server.js";
import { OperatorControlPlaneSessionStream } from "./session-stream.js";
import { StandaloneOperatorRuntime } from "../orchestrator/operator-runtime.js";

const tempDirs: string[] = [];

async function makeTempDir(): Promise<string> {
  const dir = await fs.mkdtemp(path.join(os.tmpdir(), "operator-session-stream-"));
  tempDirs.push(dir);
  return dir;
}

afterEach(async () => {
  await Promise.all(tempDirs.splice(0).map((dir) => fs.rm(dir, { recursive: true, force: true, maxRetries: 5, retryDelay: 50 })));
});

describe("OperatorControlPlaneSessionStream", () => {
  it("bootstraps a session, binds requests, and filters live events", async () => {
    const runtime = new StandaloneOperatorRuntime({
      rootDir: await makeTempDir(),
      backgroundTaskIsProcessRunning: () => false,
    });
    const server = new OperatorControlPlaneServer({ runtime });
    const stream = new OperatorControlPlaneSessionStream(server);

    const bootstrapped = await stream.bootstrap({ title: "Remote session", agentId: "gateway", remoteId: "device-1", remoteSource: "gateway" });
    expect(bootstrapped.created).toBe(true);
    expect(bootstrapped.resumed).toBe(false);
    expect(bootstrapped.session.metadata.agentId).toBe("gateway");
    expect(bootstrapped.session.metadata.remoteId).toBe("device-1");
    expect(bootstrapped.session.metadata.remoteSource).toBe("gateway");
    expect(bootstrapped.approvals).toEqual([]);
    expect(stream.sessionId).toBe(bootstrapped.session.id);

    const runResponse = await stream.request<{ id: string; sessionId: string; title: string }>({
      method: "runs.start",
      params: { title: "Remote run" },
    });
    expect(runResponse.ok).toBe(true);
    if (!runResponse.ok) {
      throw new Error("expected run response");
    }
    expect(runResponse.result.sessionId).toBe(bootstrapped.session.id);

    const eventIterator = stream.events({ family: "run" })[Symbol.asyncIterator]();
    runtime.publishRunProgress({
      runId: runResponse.result.id,
      sessionId: bootstrapped.session.id,
      phase: "planning",
      message: "Remote progress",
    });
    const nextEvent = await eventIterator.next();
    expect(nextEvent.done).toBe(false);
    expect(nextEvent.value?.type).toBe("run.progress");
    expect((nextEvent.value?.payload as { sessionId?: string }).sessionId).toBe(bootstrapped.session.id);
    await eventIterator.return?.();

    const spawned = await stream.request<{
      subagent: { parentRunId: string; sessionId: string; childSessionId: string; status: string };
      childSession: { id: string; metadata: { agentId?: string; remoteSource?: string } };
      childRun: { sessionId: string; parentRunId?: string; status: string };
    }>({
      method: "subagents.spawn",
      params: { parentRunId: runResponse.result.id, title: "Remote child", agentId: "worker", remoteSource: "gateway" },
    });
    expect(spawned.ok).toBe(true);
    if (!spawned.ok) {
      throw new Error("expected spawned subagent response");
    }
    expect(spawned.result.subagent).toMatchObject({
      parentRunId: runResponse.result.id,
      sessionId: bootstrapped.session.id,
      status: "running",
    });
    expect(spawned.result.childSession.metadata).toMatchObject({ agentId: "worker", remoteSource: "gateway" });
    expect(spawned.result.childRun).toMatchObject({
      sessionId: spawned.result.childSession.id,
      parentRunId: spawned.result.subagent.runId,
      status: "running",
    });

    const approvals = await stream.request<Array<{ sessionId?: string; id: string }>>({ method: "approvals.list" });
    expect(approvals.ok).toBe(true);
    if (!approvals.ok) {
      throw new Error("expected approvals response");
    }
    expect(approvals.result).toEqual([]);

    const remoteStatus = await stream.request<{ remoteId: string; control: { state: string } }>({
      method: "sessions.remoteStatus",
      params: {},
    });
    expect(remoteStatus.ok).toBe(true);
    if (!remoteStatus.ok) {
      throw new Error("expected remote status response");
    }
    expect(remoteStatus.result).toMatchObject({ remoteId: "device-1", control: { state: "active" } });

    const paused = await stream.request<{ control: { state: string; reason?: string }; activeRun?: { status: string } }>({
      method: "sessions.remoteControl",
      params: { action: "pause", reason: "gateway unhealthy" },
    });
    expect(paused.ok).toBe(true);
    if (!paused.ok) {
      throw new Error("expected paused remote response");
    }
    expect(paused.result).toMatchObject({
      control: { state: "paused", reason: "gateway unhealthy" },
      activeRun: { status: "paused" },
    });

    const repair = await stream.request<{ repaired: { action: string; changed: boolean; results: Array<{ reason: string }> } }>({
      method: "sessions.remoteRepair",
      params: { action: "recover-background-tasks", reasons: ["missing-process"] },
    });
    expect(repair.ok).toBe(true);
    if (!repair.ok) {
      throw new Error("expected remote repair response");
    }
    expect(repair.result).toMatchObject({
      repaired: {
        action: "recover-background-tasks",
        changed: false,
        results: [],
      },
    });
  });

  it("replays and streams subagent events for a bootstrapped session", async () => {
    const runtime = new StandaloneOperatorRuntime({
      rootDir: await makeTempDir(),
      backgroundTaskIsProcessRunning: () => false,
    });
    const server = new OperatorControlPlaneServer({ runtime });
    const session = await runtime.startSession({ title: "Parent", agentId: "gateway", remoteId: "device-subagent-1", remoteSource: "gateway" });
    const run = await runtime.startRun({ sessionId: session.id, title: "Parent run" });

    const stream = new OperatorControlPlaneSessionStream(server);
    const bootstrapped = await stream.bootstrap({ sessionId: session.id, family: "subagent" });
    expect(bootstrapped.events).toEqual([]);

    const eventIterator = stream.events({ family: "subagent" })[Symbol.asyncIterator]();
    const spawned = await stream.request<{
      subagent: { runId: string; parentRunId: string; sessionId: string; title: string; status: string };
      childSession: { id: string };
      childRun: { id: string };
    }>({ method: "subagents.spawn", params: { parentRunId: run.id, title: "Follow-up child" } });
    expect(spawned.ok).toBe(true);
    if (!spawned.ok) {
      throw new Error("expected spawned subagent");
    }
    const registeredEvent = await eventIterator.next();
    expect(registeredEvent.done).toBe(false);
    expect(registeredEvent.value?.type).toBe("subagent.registered");
    const updatedEvent = await eventIterator.next();
    expect(updatedEvent.done).toBe(false);
    expect(updatedEvent.value?.type).toBe("subagent.updated");
    await eventIterator.return?.();

    const replay = await stream.request<Array<{ type: string }>>({
      method: "runs.events",
      params: { family: "subagent" },
    });
    expect(replay.ok).toBe(true);
    if (!replay.ok) {
      throw new Error("expected replay");
    }
    expect(replay.result).toEqual([
      expect.objectContaining({ type: "subagent.registered" }),
      expect.objectContaining({ type: "subagent.updated" }),
    ]);
  });

  it("reattaches to an idle session and returns only session-scoped bootstrap state", async () => {
    const runtime = new StandaloneOperatorRuntime({
      rootDir: await makeTempDir(),
      backgroundTaskIsProcessRunning: () => false,
    });
    const server = new OperatorControlPlaneServer({ runtime });
    const targetSession = await runtime.startSession({ title: "Target", agentId: "remote" });
    const otherSession = await runtime.startSession({ title: "Other", agentId: "other" });
    await runtime.markSessionIdle(targetSession.id);

    const targetApproval = await runtime.promptApproval({
      sessionId: targetSession.id,
      title: "Target approval",
      summary: "Approve target work",
    });
    await runtime.promptApproval({
      sessionId: otherSession.id,
      title: "Other approval",
      summary: "Approve other work",
    });
    const targetRun = await runtime.startRun({ sessionId: targetSession.id, title: "Target run" });
    runtime.publishRunProgress({
      runId: targetRun.id,
      sessionId: targetSession.id,
      phase: "planning",
      message: "Target event",
    });
    const otherRun = await runtime.startRun({ sessionId: otherSession.id, title: "Other run" });
    runtime.publishRunProgress({
      runId: otherRun.id,
      sessionId: otherSession.id,
      phase: "planning",
      message: "Other event",
    });

    const stream = new OperatorControlPlaneSessionStream(server);
    const bootstrapped = await stream.bootstrap({ sessionId: targetSession.id, family: "approval" });

    expect(bootstrapped.created).toBe(false);
    expect(bootstrapped.resumed).toBe(true);
    expect(bootstrapped.session.id).toBe(targetSession.id);
    expect(bootstrapped.session.status).toBe("active");
    expect(bootstrapped.approvals).toEqual([expect.objectContaining({ id: targetApproval.id, sessionId: targetSession.id })]);
    expect(bootstrapped.events.every((event) => event.type.startsWith("approval."))).toBe(true);

    const boundApprovals = await stream.request<Array<{ id: string; sessionId?: string }>>({ method: "approvals.list" });
    expect(boundApprovals.ok).toBe(true);
    if (!boundApprovals.ok) {
      throw new Error("expected bound approvals response");
    }
    expect(boundApprovals.result).toEqual([expect.objectContaining({ id: targetApproval.id, sessionId: targetSession.id })]);
  });

  it("reattaches by remoteId when sessionId is absent", async () => {
    const runtime = new StandaloneOperatorRuntime({
      rootDir: await makeTempDir(),
      backgroundTaskIsProcessRunning: () => false,
    });
    const server = new OperatorControlPlaneServer({ runtime });
    const firstStream = new OperatorControlPlaneSessionStream(server);
    const initial = await firstStream.bootstrap({
      title: "Remote identity session",
      remoteId: "device-remote-1",
      remoteSource: "gateway",
      agentId: "gateway",
    });
    await runtime.markSessionIdle(initial.session.id);

    const secondStream = new OperatorControlPlaneSessionStream(server);
    const reattached = await secondStream.bootstrap({
      remoteId: "device-remote-1",
      remoteSource: "gateway",
    });

    expect(reattached.created).toBe(false);
    expect(reattached.resumed).toBe(true);
    expect(reattached.session.id).toBe(initial.session.id);
    expect(reattached.session.metadata.remoteId).toBe("device-remote-1");
    expect(secondStream.sessionId).toBe(initial.session.id);
  });

  it("redeems an approved pairing code through bootstrap", async () => {
    const runtime = new StandaloneOperatorRuntime({
      rootDir: await makeTempDir(),
      backgroundTaskIsProcessRunning: () => false,
    });
    const server = new OperatorControlPlaneServer({ runtime });
    const created = await server.handle({ method: "pairing.create", params: { remoteSource: "gateway" } });
    if (!created.ok) {
      throw new Error("expected pairing ticket");
    }
    await server.handle({
      method: "pairing.resolve",
      params: { code: created.result.code, decision: "approved", resolvedBy: "tester" },
    });

    const stream = new OperatorControlPlaneSessionStream(server);
    const bootstrapped = await stream.bootstrap({ pairingCode: created.result.code, agentId: "gateway" });

    expect(bootstrapped.created).toBe(true);
    expect(bootstrapped.session.metadata.remoteId).toBe(created.result.remoteId);
    expect(bootstrapped.session.metadata.remoteSource).toBe("gateway");
  });
});
