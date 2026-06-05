import { subscribeRuntimeEvents, type RuntimeEventFamily } from "./subscriptions.js";
import type { ControlPlaneRequest, ControlPlaneResponse, OperatorControlPlaneServer, SessionBootstrapResult } from "./server.js";
import type { OperatorEvent } from "../kernel/event-bus.js";

export type SessionStreamBootstrapParams = {
  sessionId?: string;
  pairingCode?: string;
  title?: string;
  cwd?: string;
  agentId?: string;
  remoteId?: string;
  remoteSource?: string;
  resume?: boolean;
  runId?: string;
  family?: RuntimeEventFamily;
  afterTs?: number;
};

export class OperatorControlPlaneSessionStream {
  private boundSessionId?: string;

  constructor(private readonly server: OperatorControlPlaneServer) {}

  get sessionId(): string | undefined {
    return this.boundSessionId;
  }

  async bootstrap(params: SessionStreamBootstrapParams = {}): Promise<SessionBootstrapResult> {
    const response = await this.server.handle({
      method: "sessions.bootstrap",
      params,
    }) as ControlPlaneResponse<SessionBootstrapResult>;
    if (!response.ok) {
      throw new Error(response.error.message);
    }
    this.boundSessionId = response.result.session.id;
    return response.result;
  }

  async request<T = unknown>(request: ControlPlaneRequest): Promise<ControlPlaneResponse<T>> {
    return await this.server.handle({
      ...request,
      params: this.bindSessionParams(request.method, request.params),
    }) as ControlPlaneResponse<T>;
  }

  events(options: { replay?: boolean; runId?: string; family?: RuntimeEventFamily } = {}): AsyncIterable<OperatorEvent> {
    const sessionId = this.requireSessionId();
    return subscribeRuntimeEvents(this.server.runtime, {
      replay: options.replay,
      sessionId,
      ...(options.runId ? { runId: options.runId } : {}),
      ...(options.family ? { family: options.family } : {}),
    });
  }

  private bindSessionParams(method: string, params?: Record<string, unknown>): Record<string, unknown> | undefined {
    if (!this.boundSessionId) {
      return params;
    }
    if (method === "sessions.remoteStatus" || method === "sessions.remoteControl" || method === "sessions.remoteRepair") {
      return {
        ...(params ?? {}),
        identifier: params?.identifier ?? this.boundSessionId,
      };
    }
    if (method === "messages.send" || method === "teams.teammates.message") {
      return {
        ...(params ?? {}),
        fromSessionId: params?.fromSessionId ?? this.boundSessionId,
      };
    }
    if (method === "plans.requestApproval") {
      return {
        ...(params ?? {}),
        sessionId: params?.sessionId ?? this.boundSessionId,
      };
    }
    if (method === "plans.respondApproval") {
      return {
        ...(params ?? {}),
        sessionId: params?.sessionId ?? this.boundSessionId,
      };
    }
    if (method === "notifications.send" || method === "push.subscriptions.create" || method === "push.subscriptions.list" || method === "push.subscriptions.delete" || method === "push.test") {
      return {
        ...(params ?? {}),
        sessionId: params?.sessionId ?? this.boundSessionId,
      };
    }
    if (!requiresSessionBinding(method)) {
      return params;
    }
    return {
      ...(params ?? {}),
      sessionId: params?.sessionId ?? this.boundSessionId,
    };
  }

  private requireSessionId(): string {
    if (!this.boundSessionId) {
      throw new Error("Session stream is not bootstrapped.");
    }
    return this.boundSessionId;
  }
}

function requiresSessionBinding(method: string): boolean {
  return (
    method === "sessions.get" ||
    method === "sessions.resume" ||
    method === "sessions.idle" ||
    method === "sessions.complete" ||
    method === "sessions.fail" ||
    method === "transcript.get" ||
    method === "approvals.list" ||
    method === "runs.start" ||
    method === "runs.list" ||
    method === "runs.active" ||
    method === "runs.events" ||
    method === "subagents.list" ||
    method === "subagents.spawn" ||
    method === "teams.teammates.start" ||
    method === "background.tasks.start" ||
    method === "background.tasks.list" ||
    method === "background.tasks.active" ||
    method === "background.tasks.recoverAll" ||
    method === "monitors.start" ||
    method === "monitors.list" ||
    method === "monitors.active" ||
    method === "tasks.create" ||
    method === "tasks.list" ||
    method === "plans.get" ||
    method === "plans.upsert" ||
    method === "plans.verify" ||
    method === "messages.send" ||
    method === "notifications.send" ||
    method === "push.subscriptions.create" ||
    method === "push.subscriptions.list" ||
    method === "push.subscriptions.delete" ||
    method === "push.test" ||
    method === "messages.list" ||
    method === "messages.inbox" ||
    method === "messages.outbox" ||
    method === "skills.executable.run"
  );
}
