import { randomUUID } from "node:crypto";
import { readJsonFile, writeJsonAtomic } from "../shared/fs.js";

export type OperatorPlanStatus = "draft" | "awaiting_approval" | "approved" | "rejected" | "in_progress" | "completed";

export type OperatorPlanVerificationStatus = "pending" | "requested" | "completed" | "skipped";

export type OperatorPlanApprovalRecord = {
  requestId: string;
  requestedBySessionId: string;
  approverSessionId: string;
  requestedAt: string;
  resolvedAt?: string;
  approved?: boolean;
  feedback?: string;
};

export type OperatorPlanVerificationRecord = {
  status: OperatorPlanVerificationStatus;
  reminderAt?: string;
  notes?: string;
};

export type OperatorPlanRecord = {
  id: string;
  sessionId: string;
  content: string;
  status: OperatorPlanStatus;
  approval?: OperatorPlanApprovalRecord;
  verification?: OperatorPlanVerificationRecord;
  createdAt: string;
  updatedAt: string;
};

export type PlanStoreShape = {
  version: 1;
  plans: OperatorPlanRecord[];
};

const EMPTY_STORE: PlanStoreShape = {
  version: 1,
  plans: [],
};

function nowIso(): string {
  return new Date().toISOString();
}

export class FilePlanStore {
  constructor(private readonly filePath: string) {}

  async load(): Promise<PlanStoreShape> {
    return await readJsonFile(this.filePath, EMPTY_STORE);
  }

  async save(store: PlanStoreShape): Promise<void> {
    await writeJsonAtomic(this.filePath, store);
  }

  async list(): Promise<OperatorPlanRecord[]> {
    const store = await this.load();
    return [...store.plans].sort((a, b) => a.createdAt.localeCompare(b.createdAt));
  }

  async get(planId: string): Promise<OperatorPlanRecord | undefined> {
    const store = await this.load();
    return store.plans.find((plan) => plan.id === planId);
  }

  async getBySession(sessionId: string): Promise<OperatorPlanRecord | undefined> {
    const plans = await this.list();
    return plans.find((plan) => plan.sessionId === sessionId);
  }

  async getByApprovalRequestId(requestId: string): Promise<OperatorPlanRecord | undefined> {
    const plans = await this.list();
    return plans.find((plan) => plan.approval?.requestId === requestId);
  }

  async upsert(params: {
    sessionId: string;
    content: string;
    status?: OperatorPlanStatus;
    approval?: OperatorPlanApprovalRecord | null;
    verification?: OperatorPlanVerificationRecord | null;
  }): Promise<OperatorPlanRecord> {
    const store = await this.load();
    const timestamp = nowIso();
    const existingIndex = store.plans.findIndex((plan) => plan.sessionId === params.sessionId);
    if (existingIndex < 0) {
      const record: OperatorPlanRecord = {
        id: randomUUID(),
        sessionId: params.sessionId,
        content: params.content,
        status: params.status ?? "draft",
        createdAt: timestamp,
        updatedAt: timestamp,
        ...(params.approval ? { approval: params.approval } : {}),
        ...(params.verification ? { verification: params.verification } : {}),
      };
      store.plans.push(record);
      await this.save(store);
      return record;
    }
    const existing = store.plans[existingIndex];
    const updated: OperatorPlanRecord = {
      ...existing,
      content: params.content,
      status: params.status ?? existing.status,
      updatedAt: timestamp,
    };
    if (params.approval !== undefined) {
      if (params.approval === null) {
        delete updated.approval;
      } else {
        updated.approval = params.approval;
      }
    }
    if (params.verification !== undefined) {
      if (params.verification === null) {
        delete updated.verification;
      } else {
        updated.verification = params.verification;
      }
    }
    store.plans[existingIndex] = updated;
    await this.save(store);
    return updated;
  }

  async update(
    planId: string,
    params: {
      content?: string;
      status?: OperatorPlanStatus;
      approval?: OperatorPlanApprovalRecord | null;
      verification?: OperatorPlanVerificationRecord | null;
    },
  ): Promise<OperatorPlanRecord | undefined> {
    const store = await this.load();
    const index = store.plans.findIndex((plan) => plan.id === planId);
    if (index < 0) {
      return undefined;
    }
    const existing = store.plans[index];
    const updated: OperatorPlanRecord = {
      ...existing,
      ...(params.content !== undefined ? { content: params.content } : {}),
      ...(params.status ? { status: params.status } : {}),
      updatedAt: nowIso(),
    };
    if (params.approval !== undefined) {
      if (params.approval === null) {
        delete updated.approval;
      } else {
        updated.approval = params.approval;
      }
    }
    if (params.verification !== undefined) {
      if (params.verification === null) {
        delete updated.verification;
      } else {
        updated.verification = params.verification;
      }
    }
    store.plans[index] = updated;
    await this.save(store);
    return updated;
  }
}
