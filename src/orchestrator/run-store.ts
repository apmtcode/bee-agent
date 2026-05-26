import { randomUUID } from "node:crypto";
import { readJsonFile, writeJsonAtomic } from "../shared/fs.js";

export type OperatorRunStatus = "pending" | "running" | "paused" | "completed" | "failed";

export type OperatorRunRecord = {
  id: string;
  sessionId: string;
  parentRunId?: string;
  title: string;
  status: OperatorRunStatus;
  createdAt: string;
  updatedAt: string;
  metadata?: Record<string, unknown>;
};

export type RunStoreShape = {
  version: 1;
  runs: OperatorRunRecord[];
};

const EMPTY_STORE: RunStoreShape = {
  version: 1,
  runs: [],
};

function nowIso(): string {
  return new Date().toISOString();
}

export class FileRunStore {
  constructor(private readonly filePath: string) {}

  async load(): Promise<RunStoreShape> {
    return await readJsonFile(this.filePath, EMPTY_STORE);
  }

  async save(store: RunStoreShape): Promise<void> {
    await writeJsonAtomic(this.filePath, store);
  }

  async list(): Promise<OperatorRunRecord[]> {
    const store = await this.load();
    return [...store.runs].sort((a, b) => a.createdAt.localeCompare(b.createdAt));
  }

  async get(runId: string): Promise<OperatorRunRecord | undefined> {
    const store = await this.load();
    return store.runs.find((run) => run.id === runId);
  }

  async create(params: {
    sessionId: string;
    title: string;
    status?: OperatorRunStatus;
    parentRunId?: string;
    metadata?: Record<string, unknown>;
  }): Promise<OperatorRunRecord> {
    const store = await this.load();
    const record: OperatorRunRecord = {
      id: randomUUID(),
      sessionId: params.sessionId,
      title: params.title,
      status: params.status ?? "pending",
      createdAt: nowIso(),
      updatedAt: nowIso(),
      ...(params.parentRunId ? { parentRunId: params.parentRunId } : {}),
      ...(params.metadata ? { metadata: params.metadata } : {}),
    };
    store.runs.push(record);
    await this.save(store);
    return record;
  }

  async update(
    runId: string,
    params: { status?: OperatorRunStatus; metadata?: Record<string, unknown> },
  ): Promise<OperatorRunRecord | undefined> {
    const store = await this.load();
    const index = store.runs.findIndex((run) => run.id === runId);
    if (index < 0) {
      return undefined;
    }
    const existing = store.runs[index];
    const updated: OperatorRunRecord = {
      ...existing,
      ...(params.status ? { status: params.status } : {}),
      ...(params.metadata ? { metadata: { ...(existing.metadata ?? {}), ...params.metadata } } : {}),
      updatedAt: nowIso(),
    };
    store.runs[index] = updated;
    await this.save(store);
    return updated;
  }

  async listBySession(sessionId: string): Promise<OperatorRunRecord[]> {
    const runs = await this.list();
    return runs.filter((run) => run.sessionId === sessionId);
  }

  async listChildren(parentRunId: string): Promise<OperatorRunRecord[]> {
    const runs = await this.list();
    return runs.filter((run) => run.parentRunId === parentRunId);
  }
}
