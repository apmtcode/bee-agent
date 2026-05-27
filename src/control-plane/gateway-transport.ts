import type { ControlPlaneRequest, ControlPlaneResponse, SessionBootstrapResult } from "./server.js";
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

export class OperatorGatewayTransportConnection {
  private readonly stream: OperatorControlPlaneSessionStream;
  private eventLoop?: Promise<void>;
  private closed = false;
  private bootstrapped = false;

  constructor(server: ConstructorParameters<typeof OperatorControlPlaneSessionStream>[0], private readonly transport: GatewayTransport) {
    this.stream = new OperatorControlPlaneSessionStream(server);
  }

  async receive(message: GatewayClientMessage): Promise<void> {
    if (this.closed) {
      await this.transport.send({ type: "error", id: "id" in message ? message.id : undefined, message: "Connection is closed." });
      return;
    }
    if (message.type === "bootstrap") {
      await this.handleBootstrap(message);
      return;
    }
    await this.handleRequest(message);
  }

  async close(): Promise<void> {
    this.closed = true;
  }

  private async handleBootstrap(message: Extract<GatewayClientMessage, { type: "bootstrap" }>): Promise<void> {
    try {
      const result = await this.stream.bootstrap(message.params ?? {});
      this.bootstrapped = true;
      await this.transport.send({ type: "bootstrap.ok", id: message.id, result });
      this.startEventLoop(message.eventSubscription);
    } catch (error) {
      await this.transport.send({
        type: "error",
        id: message.id,
        message: error instanceof Error ? error.message : String(error),
      });
    }
  }

  private async handleRequest(message: Extract<GatewayClientMessage, { type: "request" }>): Promise<void> {
    if (!this.bootstrapped) {
      await this.transport.send({ type: "error", id: message.id, message: "Connection is not bootstrapped." });
      return;
    }
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
}
