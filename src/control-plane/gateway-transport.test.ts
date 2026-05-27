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
      params: { title: "Gateway session", agentId: "gateway" },
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
});
