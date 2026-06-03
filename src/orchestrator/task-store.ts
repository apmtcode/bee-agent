import { randomUUID } from "node:crypto";
import { readJsonFile, writeJsonAtomic } from "../shared/fs.js";

export type OperatorTaskStatus = "pending" | "in_progress" | "completed";

export type OperatorTaskRecord = {
  id: string;
  sessionId: string;
  subject: string;
  description: string;
  activeForm?: string;
  owner?: string;
  status: OperatorTaskStatus;
  metadata?: Record<string, unknown>;
  blocks: string[];
  blockedBy: string[];
  createdAt: string;
  updatedAt: string;
};

export type TaskStoreShape = {
  version: 1;
  tasks: OperatorTaskRecord[];
};

const EMPTY_STORE: TaskStoreShape = {
  version: 1,
  tasks: [],
};

function nowIso(): string {
  return new Date().toISOString();
}

export class FileTaskStore {
  constructor(private readonly filePath: string) {}

  async load(): Promise<TaskStoreShape> {
    return await readJsonFile(this.filePath, EMPTY_STORE);
  }

  async save(store: TaskStoreShape): Promise<void> {
    await writeJsonAtomic(this.filePath, store);
  }

  async list(): Promise<OperatorTaskRecord[]> {
    const store = await this.load();
    return [...store.tasks].sort((a, b) => a.createdAt.localeCompare(b.createdAt));
  }

  async get(taskId: string): Promise<OperatorTaskRecord | undefined> {
    const store = await this.load();
    return store.tasks.find((task) => task.id === taskId);
  }

  async create(params: {
    sessionId: string;
    subject: string;
    description: string;
    activeForm?: string;
    owner?: string;
    status?: OperatorTaskStatus;
    metadata?: Record<string, unknown>;
    blocks?: string[];
    blockedBy?: string[];
  }): Promise<OperatorTaskRecord> {
    const store = await this.load();
    const timestamp = nowIso();
    const record: OperatorTaskRecord = {
      id: randomUUID(),
      sessionId: params.sessionId,
      subject: params.subject,
      description: params.description,
      status: params.status ?? "pending",
      blocks: [...(params.blocks ?? [])],
      blockedBy: [...(params.blockedBy ?? [])],
      createdAt: timestamp,
      updatedAt: timestamp,
      ...(params.activeForm ? { activeForm: params.activeForm } : {}),
      ...(params.owner ? { owner: params.owner } : {}),
      ...(params.metadata ? { metadata: params.metadata } : {}),
    };
    store.tasks.push(record);
    await this.save(store);
    return record;
  }

  async update(
    taskId: string,
    params: {
      subject?: string;
      description?: string;
      activeForm?: string | null;
      owner?: string | null;
      status?: OperatorTaskStatus;
      metadata?: Record<string, unknown>;
      blocks?: string[];
      blockedBy?: string[];
    },
  ): Promise<OperatorTaskRecord | undefined> {
    const store = await this.load();
    const index = store.tasks.findIndex((task) => task.id === taskId);
    if (index < 0) {
      return undefined;
    }
    const existing = store.tasks[index];
    const updated: OperatorTaskRecord = {
      ...existing,
      ...(params.subject ? { subject: params.subject } : {}),
      ...(params.description ? { description: params.description } : {}),
      ...(params.status ? { status: params.status } : {}),
      ...(params.metadata ? { metadata: { ...(existing.metadata ?? {}), ...params.metadata } } : {}),
      ...(params.blocks ? { blocks: [...params.blocks] } : {}),
      ...(params.blockedBy ? { blockedBy: [...params.blockedBy] } : {}),
      updatedAt: nowIso(),
    };
    if (params.activeForm !== undefined) {
      if (params.activeForm === null) {
        delete updated.activeForm;
      } else {
        updated.activeForm = params.activeForm;
      }
    }
    if (params.owner !== undefined) {
      if (params.owner === null) {
        delete updated.owner;
      } else {
        updated.owner = params.owner;
      }
    }
    store.tasks[index] = updated;
    await this.save(store);
    return updated;
  }

  async listBySession(sessionId: string): Promise<OperatorTaskRecord[]> {
    const tasks = await this.list();
    return tasks.filter((task) => task.sessionId === sessionId);
  }
}
