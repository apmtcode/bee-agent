import path from "node:path";
import { readJsonFile, writeJsonAtomic } from "../shared/fs.js";
import { FileRunStore, type OperatorRunRecord } from "./run-store.js";

export type SubagentRunRecord = {
  runId: string;
  parentRunId: string;
  sessionId: string;
  childSessionId: string;
  title: string;
  createdAt: string;
  updatedAt: string;
  status: "pending" | "running" | "paused" | "completed" | "failed";
};

export type SubagentRegistryShape = {
  version: 1;
  subagents: SubagentRunRecord[];
};

const EMPTY_REGISTRY: SubagentRegistryShape = {
  version: 1,
  subagents: [],
};

function nowIso(): string {
  return new Date().toISOString();
}

export class FileSubagentRegistry {
  private readonly store: FileRunStore;
  private readonly filePath: string;

  constructor(rootDir: string) {
    this.store = new FileRunStore(path.join(rootDir, "runs.json"));
    this.filePath = path.join(rootDir, "subagents.json");
  }

  private async loadRegistry(): Promise<SubagentRegistryShape> {
    return await readJsonFile(this.filePath, EMPTY_REGISTRY);
  }

  private async saveRegistry(store: SubagentRegistryShape): Promise<void> {
    await writeJsonAtomic(this.filePath, store);
  }

  async list(): Promise<SubagentRunRecord[]> {
    const store = await this.loadRegistry();
    return [...store.subagents].sort((a, b) => a.createdAt.localeCompare(b.createdAt));
  }

  async get(runId: string): Promise<SubagentRunRecord | undefined> {
    const store = await this.loadRegistry();
    return store.subagents.find((entry) => entry.runId === runId);
  }

  async register(params: {
    sessionId: string;
    parentRunId: string;
    childSessionId: string;
    title: string;
    metadata?: Record<string, unknown>;
  }): Promise<SubagentRunRecord> {
    const run = await this.store.create({
      sessionId: params.sessionId,
      parentRunId: params.parentRunId,
      title: params.title,
      status: "pending",
      metadata: params.metadata,
    });
    const store = await this.loadRegistry();
    const record: SubagentRunRecord = {
      runId: run.id,
      parentRunId: params.parentRunId,
      sessionId: params.sessionId,
      childSessionId: params.childSessionId,
      title: params.title,
      createdAt: run.createdAt,
      updatedAt: run.updatedAt,
      status: "pending",
    };
    store.subagents.push(record);
    await this.saveRegistry(store);
    return record;
  }

  async updateStatus(
    runId: string,
    status: SubagentRunRecord["status"],
  ): Promise<SubagentRunRecord | undefined> {
    const store = await this.loadRegistry();
    const index = store.subagents.findIndex((entry) => entry.runId === runId);
    if (index < 0) {
      return undefined;
    }
    const updated: SubagentRunRecord = {
      ...store.subagents[index],
      status,
      updatedAt: nowIso(),
    };
    store.subagents[index] = updated;
    await this.saveRegistry(store);
    await this.store.update(runId, { status });
    return updated;
  }

  async listByParent(parentRunId: string): Promise<SubagentRunRecord[]> {
    const entries = await this.list();
    return entries.filter((entry) => entry.parentRunId === parentRunId);
  }

  async listActive(): Promise<SubagentRunRecord[]> {
    const entries = await this.list();
    return entries.filter((entry) => entry.status === "pending" || entry.status === "running" || entry.status === "paused");
  }

  async getRun(runId: string): Promise<OperatorRunRecord | undefined> {
    return await this.store.get(runId);
  }

  async listRuns(): Promise<OperatorRunRecord[]> {
    return await this.store.list();
  }
}
