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

    const approvals = await stream.request<Array<{ sessionId?: string; id: string }>>({ method: "approvals.list" });
    expect(approvals.ok).toBe(true);
    if (!approvals.ok) {
      throw new Error("expected approvals response");
    }
    expect(approvals.result).toEqual([]);
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
