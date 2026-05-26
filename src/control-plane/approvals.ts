import { randomUUID } from "node:crypto";
import { FileApprovalStore } from "./approval-store.js";

export type ApprovalDecision = "approved" | "denied" | "expired";

export type ApprovalRequestInput = {
  title: string;
  summary: string;
  scope?: string;
  sessionId?: string;
  metadata?: Record<string, unknown>;
  timeoutMs?: number;
};

export type ApprovalRequest = {
  id: string;
  title: string;
  summary: string;
  scope?: string;
  sessionId?: string;
  metadata?: Record<string, unknown>;
  createdAtMs: number;
  expiresAtMs: number;
};

export type ApprovalResolution = {
  id: string;
  decision: ApprovalDecision;
  resolvedAtMs: number;
  resolvedBy?: string;
};

export const DEFAULT_APPROVAL_TIMEOUT_MS = 30 * 60_000;

export class InMemoryApprovalManager {
  private readonly pending = new Map<string, ApprovalRequest>();
  private readonly waiters = new Map<string, (resolution: ApprovalResolution) => void>();

  constructor(private readonly store?: FileApprovalStore) {}

  async loadPersisted(): Promise<void> {
    if (!this.store) {
      return;
    }
    const snapshot = await this.store.load();
    this.pending.clear();
    for (const request of snapshot.pending) {
      if (request.expiresAtMs <= Date.now()) {
        continue;
      }
      this.pending.set(request.id, request);
    }
    await this.syncStore();
  }

  async create(input: ApprovalRequestInput): Promise<ApprovalRequest> {
    const createdAtMs = Date.now();
    const request: ApprovalRequest = {
      id: randomUUID(),
      title: input.title,
      summary: input.summary,
      ...(input.scope ? { scope: input.scope } : {}),
      ...(input.sessionId ? { sessionId: input.sessionId } : {}),
      ...(input.metadata ? { metadata: input.metadata } : {}),
      createdAtMs,
      expiresAtMs: createdAtMs + Math.max(1, input.timeoutMs ?? DEFAULT_APPROVAL_TIMEOUT_MS),
    };
    this.pending.set(request.id, request);
    await this.syncStore();
    return request;
  }

  list(): ApprovalRequest[] {
    return Array.from(this.pending.values()).sort((a, b) => a.createdAtMs - b.createdAtMs);
  }

  async resolve(id: string, decision: ApprovalDecision, resolvedBy?: string): Promise<ApprovalResolution | null> {
    const request = this.pending.get(id);
    if (!request) {
      return null;
    }
    this.pending.delete(id);
    const resolution: ApprovalResolution = {
      id,
      decision,
      resolvedAtMs: Date.now(),
      ...(resolvedBy ? { resolvedBy } : {}),
    };
    await this.syncStore(resolution);
    const waiter = this.waiters.get(id);
    if (waiter) {
      this.waiters.delete(id);
      waiter(resolution);
    }
    return resolution;
  }

  async waitForDecision(id: string): Promise<ApprovalResolution> {
    const request = this.pending.get(id);
    if (!request) {
      return {
        id,
        decision: "expired",
        resolvedAtMs: Date.now(),
      };
    }
    const remainingMs = Math.max(1, request.expiresAtMs - Date.now());
    return await new Promise<ApprovalResolution>((resolve) => {
      const timer = setTimeout(async () => {
        this.pending.delete(id);
        this.waiters.delete(id);
        const resolution = { id, decision: "expired" as const, resolvedAtMs: Date.now() };
        await this.syncStore(resolution);
        resolve(resolution);
      }, remainingMs);
      this.waiters.set(id, (resolution) => {
        clearTimeout(timer);
        resolve(resolution);
      });
    });
  }

  private async syncStore(resolution?: ApprovalResolution): Promise<void> {
    if (!this.store) {
      return;
    }
    const snapshot = await this.store.load();
    snapshot.pending = this.list();
    if (resolution) {
      snapshot.resolutions.push(resolution);
    }
    await this.store.save(snapshot);
  }
}
