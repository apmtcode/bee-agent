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
      childSession: { id: string; metadata?: { modelSelection?: { primary: string; fallbacks: string[]; source: string } } };
      childRun: { id: string; metadata?: { modelSelection?: { primary: string; fallbacks: string[]; source: string } } };
    }>({ method: "subagents.spawn", params: { parentRunId: run.id, title: "Follow-up child", modelPrimary: "claude-sonnet-4-6", modelFallbacks: ["claude-haiku-4-5-20251001"] } });
    expect(spawned.ok).toBe(true);
    if (!spawned.ok) {
      throw new Error("expected spawned subagent");
    }
    expect(spawned.result.childSession.metadata?.modelSelection).toMatchObject({
      primary: "claude-sonnet-4-6",
      fallbacks: ["claude-haiku-4-5-20251001"],
      source: "override",
    });
    expect(spawned.result.childRun.metadata?.modelSelection).toMatchObject({
      primary: "claude-sonnet-4-6",
      fallbacks: ["claude-haiku-4-5-20251001"],
      source: "override",
    });
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

  it("binds task requests and streams task events for a bootstrapped session", async () => {
    const runtime = new StandaloneOperatorRuntime({
      rootDir: await makeTempDir(),
      backgroundTaskIsProcessRunning: () => false,
    });
    const server = new OperatorControlPlaneServer({ runtime });
    const stream = new OperatorControlPlaneSessionStream(server);

    const bootstrapped = await stream.bootstrap({ title: "Task session", family: "task" });
    expect(bootstrapped.created).toBe(true);
    expect(bootstrapped.taskPlan).toEqual({ entries: [] });
    expect(bootstrapped.events).toEqual([]);

    const created = await stream.request<{
      id: string;
      sessionId: string;
      subject: string;
      status: string;
    }>({
      method: "tasks.create",
      params: { subject: "Inspect logs", description: "Look at the latest logs." },
    });
    expect(created.ok).toBe(true);
    if (!created.ok) {
      throw new Error("expected created task");
    }
    expect(created.result).toMatchObject({
      sessionId: bootstrapped.session.id,
      subject: "Inspect logs",
      status: "pending",
    });
    const resumedBootstrap = await stream.bootstrap({ sessionId: bootstrapped.session.id, family: "task" });
    expect(resumedBootstrap.taskPlan).toEqual({
      entries: [
        expect.objectContaining({
          id: created.result.id,
          subject: "Inspect logs",
          description: "Look at the latest logs.",
          status: "pending",
        }),
      ],
    });

    const listed = await stream.request<Array<{ id: string; sessionId: string; subject: string }>>({
      method: "tasks.list",
      params: {},
    });
    expect(listed.ok).toBe(true);
    if (!listed.ok) {
      throw new Error("expected listed tasks");
    }
    expect(listed.result).toEqual([
      expect.objectContaining({
        id: created.result.id,
        sessionId: bootstrapped.session.id,
        subject: "Inspect logs",
      }),
    ]);

    const eventIterator = stream.events({ family: "task" })[Symbol.asyncIterator]();
    const updated = await stream.request<{ id: string; status: string }>({
      method: "tasks.update",
      params: { taskId: created.result.id, status: "completed" },
    });
    expect(updated.ok).toBe(true);
    if (!updated.ok) {
      throw new Error("expected updated task");
    }
    expect(updated.result).toMatchObject({ id: created.result.id, status: "completed" });

    const nextEvent = await eventIterator.next();
    expect(nextEvent.done).toBe(false);
    expect(nextEvent.value?.type).toBe("task.updated");
    expect(nextEvent.value?.payload).toMatchObject({ id: created.result.id, status: "completed", sessionId: bootstrapped.session.id });
    await eventIterator.return?.();

    const replay = await stream.request<Array<{ type: string; payload?: { id?: string; sessionId?: string; status?: string } }>>({
      method: "runs.events",
      params: { family: "task" },
    });
    expect(replay.ok).toBe(true);
    if (!replay.ok) {
      throw new Error("expected task replay");
    }
    expect(replay.result).toEqual([
      expect.objectContaining({ type: "task.created", payload: expect.objectContaining({ id: created.result.id, sessionId: bootstrapped.session.id }) }),
      expect.objectContaining({ type: "task.updated", payload: expect.objectContaining({ id: created.result.id, status: "completed" }) }),
    ]);
  });

  it("binds plan requests and streams plan events for a bootstrapped session", async () => {
    const runtime = new StandaloneOperatorRuntime({
      rootDir: await makeTempDir(),
      backgroundTaskIsProcessRunning: () => false,
    });
    const server = new OperatorControlPlaneServer({ runtime });
    const reviewer = await runtime.startSession({ title: "Reviewer", agentId: "reviewer" });
    const stream = new OperatorControlPlaneSessionStream(server);

    const bootstrapped = await stream.bootstrap({ title: "Plan session", family: "plan" });
    expect(bootstrapped.created).toBe(true);
    expect(bootstrapped.plan).toBeUndefined();
    expect(bootstrapped.events).toEqual([]);

    const created = await stream.request<{
      id: string;
      sessionId: string;
      content: string;
      status: string;
    }>({
      method: "plans.upsert",
      params: { content: "1. Add plan store\n2. Wire RPC" },
    });
    expect(created.ok).toBe(true);
    if (!created.ok) {
      throw new Error("expected created plan");
    }
    expect(created.result).toMatchObject({
      sessionId: bootstrapped.session.id,
      content: "1. Add plan store\n2. Wire RPC",
      status: "draft",
    });

    const listed = await stream.request<{
      id: string;
      content: string;
      status: string;
    }>({
      method: "plans.get",
      params: {},
    });
    expect(listed.ok).toBe(true);
    if (!listed.ok) {
      throw new Error("expected listed plan");
    }
    expect(listed.result).toMatchObject({
      id: created.result.id,
      content: "1. Add plan store\n2. Wire RPC",
      status: "draft",
    });

    const requested = await stream.request<{
      plan: { id: string; sessionId: string; status: string; approval?: { requestId: string; approverSessionId: string } };
      message: { fromSessionId: string; toSessionId: string; summary: string };
    }>({
      method: "plans.requestApproval",
      params: { approverSessionId: reviewer.id },
    });
    expect(requested.ok).toBe(true);
    if (!requested.ok) {
      throw new Error("expected requested plan approval");
    }
    expect(requested.result).toMatchObject({
      plan: {
        id: created.result.id,
        sessionId: bootstrapped.session.id,
        status: "awaiting_approval",
        approval: {
          approverSessionId: reviewer.id,
        },
      },
      message: {
        fromSessionId: bootstrapped.session.id,
        toSessionId: reviewer.id,
      },
    });

    const eventIterator = stream.events({ family: "plan" })[Symbol.asyncIterator]();
    const verified = await stream.request<{ id: string; verification: { status: string; notes?: string } }>({
      method: "plans.verify",
      params: { status: "requested", notes: "Run focused tests" },
    });
    expect(verified.ok).toBe(true);
    if (!verified.ok) {
      throw new Error("expected verified plan");
    }
    expect(verified.result).toMatchObject({
      id: created.result.id,
      verification: { status: "requested", notes: "Run focused tests" },
    });

    const nextEvent = await eventIterator.next();
    expect(nextEvent.done).toBe(false);
    expect(nextEvent.value?.type).toBe("plan.verification.updated");
    expect(nextEvent.value?.payload).toMatchObject({
      id: created.result.id,
      sessionId: bootstrapped.session.id,
      verification: { status: "requested", notes: "Run focused tests" },
    });
    await eventIterator.return?.();

    const resumedBootstrap = await stream.bootstrap({ sessionId: bootstrapped.session.id, family: "plan" });
    expect(resumedBootstrap.plan).toMatchObject({
      id: created.result.id,
      content: "1. Add plan store\n2. Wire RPC",
      status: "awaiting_approval",
      approval: {
        requestId: requested.result.plan.approval?.requestId,
        approverSessionId: reviewer.id,
      },
      verification: { status: "requested", notes: "Run focused tests" },
    });
    expect(resumedBootstrap.events).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ type: "plan.updated" }),
        expect.objectContaining({ type: "plan.approval.requested" }),
        expect.objectContaining({ type: "plan.verification.updated" }),
      ]),
    );
  });

  it("binds message requests and streams message events for a bootstrapped session", async () => {
    const runtime = new StandaloneOperatorRuntime({
      rootDir: await makeTempDir(),
      backgroundTaskIsProcessRunning: () => false,
    });
    const server = new OperatorControlPlaneServer({ runtime });
    const recipient = await runtime.startSession({ title: "Recipient", agentId: "worker" });
    const stream = new OperatorControlPlaneSessionStream(server);

    const bootstrapped = await stream.bootstrap({ title: "Message session", family: "message" });
    expect(bootstrapped.created).toBe(true);
    expect(bootstrapped.events).toEqual([]);

    const sent = await stream.request<{
      id: string;
      fromSessionId: string;
      toSessionId: string;
      summary: string;
    }>({
      method: "messages.send",
      params: { toSessionId: recipient.id, message: "Please collect the latest logs." },
    });
    expect(sent.ok).toBe(true);
    if (!sent.ok) {
      throw new Error("expected sent message");
    }
    expect(sent.result).toMatchObject({
      fromSessionId: bootstrapped.session.id,
      toSessionId: recipient.id,
      summary: "Please collect the latest logs.",
    });

    const listed = await stream.request<Array<{ id: string; fromSessionId: string; toSessionId: string }>>({
      method: "messages.list",
      params: {},
    });
    expect(listed.ok).toBe(true);
    if (!listed.ok) {
      throw new Error("expected listed messages");
    }
    expect(listed.result).toEqual([
      expect.objectContaining({
        id: sent.result.id,
        fromSessionId: bootstrapped.session.id,
        toSessionId: recipient.id,
      }),
    ]);

    const eventIterator = stream.events({ family: "message" })[Symbol.asyncIterator]();
    const second = await stream.request<{ id: string; fromSessionId: string; toSessionId: string; summary: string }>({
      method: "messages.send",
      params: { toSessionId: recipient.id, message: "Collected and attached." },
    });
    expect(second.ok).toBe(true);
    if (!second.ok) {
      throw new Error("expected second sent message");
    }

    const nextEvent = await eventIterator.next();
    expect(nextEvent.done).toBe(false);
    expect(nextEvent.value?.type).toBe("message.sent");
    expect(nextEvent.value?.payload).toMatchObject({
      id: second.result.id,
      fromSessionId: bootstrapped.session.id,
      toSessionId: recipient.id,
      summary: "Collected and attached.",
    });
    await eventIterator.return?.();

    const replay = await stream.request<Array<{ type: string; payload?: { id?: string; fromSessionId?: string; toSessionId?: string } }>>({
      method: "runs.events",
      params: { family: "message" },
    });
    expect(replay.ok).toBe(true);
    if (!replay.ok) {
      throw new Error("expected message replay");
    }
    expect(replay.result).toEqual([
      expect.objectContaining({ type: "message.sent", payload: expect.objectContaining({ id: sent.result.id, fromSessionId: bootstrapped.session.id, toSessionId: recipient.id }) }),
      expect.objectContaining({ type: "message.sent", payload: expect.objectContaining({ id: second.result.id, fromSessionId: bootstrapped.session.id, toSessionId: recipient.id }) }),
    ]);
  });

  it("binds teammate requests for a bootstrapped session", async () => {
    const runtime = new StandaloneOperatorRuntime({
      rootDir: await makeTempDir(),
      backgroundTaskIsProcessRunning: () => false,
    });
    const server = new OperatorControlPlaneServer({ runtime });
    const stream = new OperatorControlPlaneSessionStream(server);

    const bootstrapped = await stream.bootstrap({ title: "Teammate session" });
    const run = await runtime.startRun({ sessionId: bootstrapped.session.id, title: "Coordinator" });
    const teamTaskSession = await runtime.startSession({ title: "Team reviewers", agentId: "team:reviewers" });
    await runtime.teams.create({ name: "reviewers", description: "Review team", taskSessionId: teamTaskSession.id });

    const started = await stream.request<{
      teamName: string;
      teammateName: string;
      taskSessionId: string;
      spawned: {
        subagent: { sessionId: string; runId: string; status: string };
        childSession: { id: string };
        childRun: { id: string; status: string };
      };
    }>({
      method: "teams.teammates.start",
      params: { teamName: "reviewers", parentRunId: run.id, teammateName: "alice", title: "Review PR", agentId: "reviewer" },
    });
    expect(started.ok).toBe(true);
    if (!started.ok) {
      throw new Error("expected started teammate");
    }
    expect(started.result).toMatchObject({
      teamName: "reviewers",
      teammateName: "alice",
      taskSessionId: teamTaskSession.id,
      spawned: {
        subagent: { sessionId: bootstrapped.session.id, status: "running" },
        childRun: { status: "running" },
      },
    });

    const listed = await stream.request<Array<{ name: string; sessionId: string; status: string }>>({
      method: "teams.teammates.list",
      params: { teamName: "reviewers" },
    });
    expect(listed.ok).toBe(true);
    if (!listed.ok) {
      throw new Error("expected listed teammates");
    }
    expect(listed.result).toEqual([
      expect.objectContaining({ name: "alice", sessionId: started.result.spawned.childSession.id, status: "running" }),
    ]);

    const sent = await stream.request<{
      teammateName: string;
      teammateSessionId: string;
      message: { fromSessionId: string; toSessionId: string; summary: string };
    }>({
      method: "teams.teammates.message",
      params: { teamName: "reviewers", teammateName: "alice", message: "Please inspect logs." },
    });
    expect(sent.ok).toBe(true);
    if (!sent.ok) {
      throw new Error("expected sent teammate message");
    }
    expect(sent.result).toMatchObject({
      teammateName: "alice",
      teammateSessionId: started.result.spawned.childSession.id,
      message: {
        fromSessionId: bootstrapped.session.id,
        toSessionId: started.result.spawned.childSession.id,
        summary: "Please inspect logs.",
      },
    });
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
