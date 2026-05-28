import { randomUUID } from "node:crypto";
import type { ControlPlaneRequest, ControlPlaneResponse, SessionBootstrapResult, OperatorControlPlaneServer, GatewaySessionHealth } from "./server.js";
import {
  OperatorControlPlaneSessionStream,
  type SessionStreamBootstrapParams,
} from "./session-stream.js";
import type { OperatorEvent } from "../kernel/event-bus.js";

export type GatewayClientMessage =
  | {
      type: "bootstrap";
      id?: string;
      params?: SessionStreamBootstrapParams;
      eventSubscription?: GatewayEventSubscription;
    }
  | {
      type: "request";
      id: string;
      request: ControlPlaneRequest;
    }
  | {
      type: "pong";
      id?: string;
    };

export type GatewayServerMessage =
  | {
      type: "bootstrap.ok";
      id?: string;
      result: SessionBootstrapResult;
    }
  | {
      type: "response";
      id: string;
      result: ControlPlaneResponse;
    }
  | {
      type: "event";
      event: OperatorEvent;
    }
  | {
      type: "ping";
      id: string;
      ts: number;
    }
  | {
      type: "error";
      id?: string;
      message: string;
    };

export type GatewayEventSubscription = {
  replay?: boolean;
  runId?: string;
  family?: "run" | "approval" | "background-task" | "skill" | "subagent";
};

export type GatewayTransport = {
  send(message: GatewayServerMessage): Promise<void> | void;
};

export type OperatorGatewayTransportConnectionOptions = {
  heartbeatIntervalMs?: number;
  heartbeatTimeoutMs?: number;
  now?: () => number;
};

export class OperatorGatewayTransportConnection {
  private readonly stream: OperatorControlPlaneSessionStream;
  private readonly connectionId = randomUUID();
  private readonly heartbeatIntervalMs: number;
  private readonly heartbeatTimeoutMs: number;
  private readonly now: () => number;
  private eventLoop?: Promise<void>;
  private heartbeatTimer?: ReturnType<typeof setTimeout>;
  private heartbeatTimeout?: ReturnType<typeof setTimeout>;
  private closed = false;
  private bootstrapped = false;
  private lastClientActivityAt?: number;
  private lastServerEventAt?: number;
  private activeSessionId?: string;
  private activeRemoteId?: string;
  private pendingPingId?: string;

  constructor(
    private readonly server: OperatorControlPlaneServer,
    private readonly transport: GatewayTransport,
    options: OperatorGatewayTransportConnectionOptions = {},
  ) {
    this.stream = new OperatorControlPlaneSessionStream(server);
    this.heartbeatIntervalMs = options.heartbeatIntervalMs ?? 30_000;
    this.heartbeatTimeoutMs = options.heartbeatTimeoutMs ?? 10_000;
    this.now = options.now ?? (() => Date.now());
  }

  async receive(message: GatewayClientMessage): Promise<void> {
    if (this.closed) {
      await this.transport.send({ type: "error", id: "id" in message ? message.id : undefined, message: "Connection is closed." });
      return;
    }
    this.touchClientActivity();
    if (message.type === "bootstrap") {
      await this.handleBootstrap(message);
      return;
    }
    if (message.type === "pong") {
      this.handlePong(message);
      return;
    }
    await this.handleRequest(message);
  }

  async close(): Promise<void> {
    this.closed = true;
    this.clearHeartbeatTimers();
    this.publishGatewayHealth("closed");
  }

  private async handleBootstrap(message: Extract<GatewayClientMessage, { type: "bootstrap" }>): Promise<void> {
    try {
      const result = await this.stream.bootstrap(message.params ?? {});
      this.bootstrapped = true;
      this.activeSessionId = result.session.id;
      this.activeRemoteId = result.session.metadata.remoteId;
      this.server.registerGatewaySessionConnection(this.buildGatewayHealth("healthy"));
      await this.transport.send({ type: "bootstrap.ok", id: message.id, result });
      this.startHeartbeatLoop();
      this.startEventLoop(message.eventSubscription);
    } catch (error) {
      await this.transport.send({
        type: "error",
        id: message.id,
        message: error instanceof Error ? error.message : String(error),
      });
    }
  }

  private handlePong(_message: Extract<GatewayClientMessage, { type: "pong" }>): void {
    this.pendingPingId = undefined;
    if (this.heartbeatTimeout) {
      clearTimeout(this.heartbeatTimeout);
      this.heartbeatTimeout = undefined;
    }
    this.publishGatewayHealth("healthy");
    this.scheduleNextPing();
  }

  private async handleRequest(message: Extract<GatewayClientMessage, { type: "request" }>): Promise<void> {
    if (!this.bootstrapped) {
      await this.transport.send({ type: "error", id: message.id, message: "Connection is not bootstrapped." });
      return;
    }
    this.publishGatewayHealth("healthy");
    const result = await this.stream.request(message.request);
    await this.transport.send({ type: "response", id: message.id, result });
  }

  private startEventLoop(subscription?: GatewayEventSubscription): void {
    if (this.eventLoop) {
      return;
    }
    this.eventLoop = (async () => {
      try {
        for await (const event of this.stream.events(subscription ?? {})) {
          if (this.closed) {
            break;
          }
          this.lastServerEventAt = event.ts;
          this.publishGatewayHealth("healthy");
          await this.transport.send({ type: "event", event });
        }
      } catch (error) {
        if (!this.closed) {
          await this.transport.send({
            type: "error",
            message: error instanceof Error ? error.message : String(error),
          });
        }
      }
    })();
  }

  private startHeartbeatLoop(): void {
    if (this.heartbeatIntervalMs <= 0) {
      return;
    }
    this.scheduleNextPing();
  }

  private scheduleNextPing(): void {
    if (this.closed) {
      return;
    }
    if (this.heartbeatTimer) {
      clearTimeout(this.heartbeatTimer);
    }
    this.heartbeatTimer = setTimeout(() => {
      void this.sendPing();
    }, this.heartbeatIntervalMs);
  }

  private async sendPing(): Promise<void> {
    if (this.closed || !this.bootstrapped) {
      return;
    }
    const pingId = randomUUID();
    this.pendingPingId = pingId;
    await this.transport.send({ type: "ping", id: pingId, ts: this.now() });
    if (this.heartbeatTimeout) {
      clearTimeout(this.heartbeatTimeout);
    }
    this.heartbeatTimeout = setTimeout(() => {
      this.markStale();
    }, this.heartbeatTimeoutMs);
  }

  private markStale(): void {
    if (this.closed) {
      return;
    }
    this.closed = true;
    this.clearHeartbeatTimers();
    this.publishGatewayHealth("stale");
  }

  private touchClientActivity(): void {
    this.lastClientActivityAt = this.now();
  }

  private publishGatewayHealth(state: GatewaySessionHealth["state"]): void {
    if (!this.activeSessionId) {
      return;
    }
    this.server.updateGatewaySessionHealth(this.buildGatewayHealth(state));
  }

  private buildGatewayHealth(state: GatewaySessionHealth["state"]): GatewaySessionHealth {
    return {
      connectionId: this.connectionId,
      sessionId: this.activeSessionId ?? this.stream.sessionId ?? "",
      ...(this.activeRemoteId ? { remoteId: this.activeRemoteId } : {}),
      state,
      updatedAt: this.now(),
      ...(typeof this.lastClientActivityAt === "number" ? { lastClientActivityAt: this.lastClientActivityAt } : {}),
      ...(typeof this.lastServerEventAt === "number" ? { lastServerEventAt: this.lastServerEventAt } : {}),
    };
  }

  private clearHeartbeatTimers(): void {
    if (this.heartbeatTimer) {
      clearTimeout(this.heartbeatTimer);
      this.heartbeatTimer = undefined;
    }
    if (this.heartbeatTimeout) {
      clearTimeout(this.heartbeatTimeout);
      this.heartbeatTimeout = undefined;
    }
    this.pendingPingId = undefined;
  }
}
