import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { OperatorControlPlaneServer } from "./server.js";
import { OperatorGatewayTransportConnection, type GatewayClientMessage, type GatewayServerMessage } from "./gateway-transport.js";
import { StandaloneOperatorRuntime } from "../orchestrator/operator-runtime.js";

const tempDirs: string[] = [];

async function makeTempDir(): Promise<string> {
  const dir = await fs.mkdtemp(path.join(os.tmpdir(), "operator-gateway-transport-"));
  tempDirs.push(dir);
  return dir;
}

afterEach(async () => {
  await Promise.all(tempDirs.splice(0).map((dir) => fs.rm(dir, { recursive: true, force: true, maxRetries: 5, retryDelay: 50 })));
});

class InMemoryGatewayTransport {
  readonly sent: GatewayServerMessage[] = [];

  async send(message: GatewayServerMessage): Promise<void> {
    this.sent.push(message);
  }
}

describe("OperatorGatewayTransportConnection", () => {
  it("bootstraps a remote connection, binds requests, and forwards live events", async () => {
    const runtime = new StandaloneOperatorRuntime({
      rootDir: await makeTempDir(),
      backgroundTaskIsProcessRunning: () => false,
    });
    const server = new OperatorControlPlaneServer({ runtime });
    const transport = new InMemoryGatewayTransport();
    const connection = new OperatorGatewayTransportConnection(server, transport);

    await connection.receive({
      type: "bootstrap",
      id: "boot-1",
      params: { title: "Gateway session", agentId: "gateway", remoteId: "device-1", remoteSource: "gateway" },
      eventSubscription: { family: "run" },
    });

    const bootstrapped = transport.sent.find((message) => message.type === "bootstrap.ok");
    expect(bootstrapped).toBeDefined();
    if (!bootstrapped || bootstrapped.type !== "bootstrap.ok") {
      throw new Error("expected bootstrap response");
    }
    expect(bootstrapped.id).toBe("boot-1");
    expect(bootstrapped.result.session.metadata.agentId).toBe("gateway");

    await connection.receive({
      type: "request",
      id: "req-1",
      request: { method: "runs.start", params: { title: "Remote run" } },
    });

    const response = transport.sent.find((message) => message.type === "response" && message.id === "req-1");
    expect(response).toBeDefined();
    if (!response || response.type !== "response" || !response.result.ok) {
      throw new Error("expected run response");
    }
    const sessionId = bootstrapped.result.session.id;
    expect(response.result.result).toMatchObject({ sessionId, title: "Remote run" });

    runtime.publishRunProgress({
      runId: response.result.result.id,
      sessionId,
      phase: "planning",
      message: "Transport event",
    });

    await new Promise((resolve) => setTimeout(resolve, 0));
    const eventMessage = transport.sent.findLast(
      (message) => message.type === "event" && message.event.type === "run.progress",
    );
    expect(eventMessage).toBeDefined();
    if (!eventMessage || eventMessage.type !== "event") {
      throw new Error("expected event frame");
    }
    expect(eventMessage.event.type).toBe("run.progress");
    expect((eventMessage.event.payload as { sessionId?: string }).sessionId).toBe(sessionId);

    await connection.receive({
      type: "request",
      id: "req-subagent-1",
      request: { method: "subagents.spawn", params: { parentRunId: response.result.result.id, title: "Gateway child", agentId: "worker" } },
    });
    const spawnResponse = transport.sent.find((message) => message.type === "response" && message.id === "req-subagent-1");
    expect(spawnResponse).toBeDefined();
    if (!spawnResponse || spawnResponse.type !== "response" || !spawnResponse.result.ok) {
      throw new Error("expected spawn response");
    }
    expect(spawnResponse.result.result).toMatchObject({
      subagent: { parentRunId: response.result.result.id, sessionId, title: "Gateway child", status: "running" },
      childSession: { metadata: { agentId: "worker" } },
    });
  });

  it("rejects requests before bootstrap and filters unrelated events", async () => {
    const runtime = new StandaloneOperatorRuntime({
      rootDir: await makeTempDir(),
      backgroundTaskIsProcessRunning: () => false,
    });
    const server = new OperatorControlPlaneServer({ runtime });
    const transport = new InMemoryGatewayTransport();
    const connection = new OperatorGatewayTransportConnection(server, transport);

    await connection.receive({
      type: "request",
      id: "req-before-bootstrap",
      request: { method: "approvals.list" },
    });
    expect(transport.sent).toContainEqual({
      type: "error",
      id: "req-before-bootstrap",
      message: "Connection is not bootstrapped.",
    });

    const otherSession = await runtime.startSession({ title: "Other", agentId: "other" });
    const targetSession = await runtime.startSession({ title: "Target", agentId: "target" });
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

    await connection.receive({
      type: "bootstrap",
      id: "boot-2",
      params: { sessionId: targetSession.id, family: "approval" },
      eventSubscription: { family: "approval" },
    });

    const bootstrapResult = transport.sent.find((message) => message.type === "bootstrap.ok" && message.id === "boot-2");
    expect(bootstrapResult).toBeDefined();
    if (!bootstrapResult || bootstrapResult.type !== "bootstrap.ok") {
      throw new Error("expected bootstrap result");
    }
    expect(bootstrapResult.result.approvals).toEqual([expect.objectContaining({ id: targetApproval.id, sessionId: targetSession.id })]);
    expect(bootstrapResult.result.events.every((event) => event.type.startsWith("approval."))).toBe(true);

    await runtime.resolveApproval(targetApproval.id, "approved", "tester");
    await new Promise((resolve) => setTimeout(resolve, 0));
    const approvalEvent = transport.sent.findLast(
      (message) => message.type === "event" && message.event.type === "approval.resolved",
    );
    expect(approvalEvent).toBeDefined();
    if (!approvalEvent || approvalEvent.type !== "event") {
      throw new Error("expected approval event");
    }
    expect(approvalEvent.event.type).toBe("approval.resolved");
  });

  it("reattaches by remoteId across gateway connections", async () => {
    const runtime = new StandaloneOperatorRuntime({
      rootDir: await makeTempDir(),
      backgroundTaskIsProcessRunning: () => false,
    });
    const server = new OperatorControlPlaneServer({ runtime });
    const firstTransport = new InMemoryGatewayTransport();
    const firstConnection = new OperatorGatewayTransportConnection(server, firstTransport);

    await firstConnection.receive({
      type: "bootstrap",
      id: "boot-remote-1",
      params: { title: "Gateway remote", remoteId: "device-gateway-1", remoteSource: "gateway", agentId: "gateway" },
    });
    const firstBootstrap = firstTransport.sent.find((message) => message.type === "bootstrap.ok" && message.id === "boot-remote-1");
    if (!firstBootstrap || firstBootstrap.type !== "bootstrap.ok") {
      throw new Error("expected first bootstrap");
    }
    await runtime.markSessionIdle(firstBootstrap.result.session.id);

    const secondTransport = new InMemoryGatewayTransport();
    const secondConnection = new OperatorGatewayTransportConnection(server, secondTransport);
    await secondConnection.receive({
      type: "bootstrap",
      id: "boot-remote-2",
      params: { remoteId: "device-gateway-1", remoteSource: "gateway" },
    });

    const secondBootstrap = secondTransport.sent.find((message) => message.type === "bootstrap.ok" && message.id === "boot-remote-2");
    expect(secondBootstrap).toBeDefined();
    if (!secondBootstrap || secondBootstrap.type !== "bootstrap.ok") {
      throw new Error("expected second bootstrap");
    }
    expect(secondBootstrap.result.created).toBe(false);
    expect(secondBootstrap.result.resumed).toBe(true);
    expect(secondBootstrap.result.session.id).toBe(firstBootstrap.result.session.id);
  });

  it("forwards subagent-family events across the gateway", async () => {
    const runtime = new StandaloneOperatorRuntime({
      rootDir: await makeTempDir(),
      backgroundTaskIsProcessRunning: () => false,
    });
    const server = new OperatorControlPlaneServer({ runtime });
    const transport = new InMemoryGatewayTransport();
    const connection = new OperatorGatewayTransportConnection(server, transport);
    const session = await runtime.startSession({ title: "Gateway subagent", agentId: "gateway" });
    const run = await runtime.startRun({ sessionId: session.id, title: "Gateway parent run" });

    await connection.receive({
      type: "bootstrap",
      id: "boot-subagent-1",
      params: { sessionId: session.id },
      eventSubscription: { family: "subagent" },
    });
    await connection.receive({
      type: "request",
      id: "req-subagent-events",
      request: { method: "subagents.spawn", params: { parentRunId: run.id, title: "Gateway spawned child" } },
    });

    await new Promise((resolve) => setTimeout(resolve, 0));
    const subagentEvents = transport.sent.filter((message) => message.type === "event").map((message) => message.event.type);
    expect(subagentEvents).toContain("subagent.registered");
    expect(subagentEvents).toContain("subagent.updated");
  });

  it("passes pairingCode through gateway bootstrap", async () => {
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
    const transport = new InMemoryGatewayTransport();
    const connection = new OperatorGatewayTransportConnection(server, transport);

    await connection.receive({
      type: "bootstrap",
      id: "boot-pairing-1",
      params: { pairingCode: created.result.code, agentId: "gateway" },
    });

    const bootstrapped = transport.sent.find((message) => message.type === "bootstrap.ok" && message.id === "boot-pairing-1");
    expect(bootstrapped).toBeDefined();
    if (!bootstrapped || bootstrapped.type !== "bootstrap.ok") {
      throw new Error("expected paired bootstrap");
    }
    expect(bootstrapped.result.session.metadata.remoteId).toBe(created.result.remoteId);
    expect(bootstrapped.result.session.metadata.remoteSource).toBe("gateway");
  });
});
