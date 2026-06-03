import { randomUUID } from "node:crypto";
import { readJsonFile, writeJsonAtomic } from "../shared/fs.js";

export type OperatorMessageRecord = {
  id: string;
  fromSessionId: string;
  toSessionId: string;
  summary: string;
  message: string;
  metadata?: Record<string, unknown>;
  createdAt: string;
};

export type MessageStoreShape = {
  version: 1;
  messages: OperatorMessageRecord[];
};

const EMPTY_STORE: MessageStoreShape = {
  version: 1,
  messages: [],
};

function nowIso(): string {
  return new Date().toISOString();
}

export class FileMessageStore {
  constructor(private readonly filePath: string) {}

  async load(): Promise<MessageStoreShape> {
    return await readJsonFile(this.filePath, EMPTY_STORE);
  }

  async save(store: MessageStoreShape): Promise<void> {
    await writeJsonAtomic(this.filePath, store);
  }

  async list(): Promise<OperatorMessageRecord[]> {
    const store = await this.load();
    return [...store.messages].sort((a, b) => a.createdAt.localeCompare(b.createdAt));
  }

  async get(messageId: string): Promise<OperatorMessageRecord | undefined> {
    const store = await this.load();
    return store.messages.find((message) => message.id === messageId);
  }

  async send(params: {
    fromSessionId: string;
    toSessionId: string;
    summary?: string;
    message: string;
    metadata?: Record<string, unknown>;
  }): Promise<OperatorMessageRecord> {
    const store = await this.load();
    const message = params.message.trim();
    const summary = buildMessageSummary(params.summary, message);
    const record: OperatorMessageRecord = {
      id: randomUUID(),
      fromSessionId: params.fromSessionId,
      toSessionId: params.toSessionId,
      summary,
      message,
      createdAt: nowIso(),
      ...(params.metadata ? { metadata: params.metadata } : {}),
    };
    store.messages.push(record);
    await this.save(store);
    return record;
  }

  async listBySession(sessionId: string): Promise<OperatorMessageRecord[]> {
    const messages = await this.list();
    return messages.filter((message) => message.fromSessionId === sessionId || message.toSessionId === sessionId);
  }

  async listInbox(sessionId: string): Promise<OperatorMessageRecord[]> {
    const messages = await this.list();
    return messages.filter((message) => message.toSessionId === sessionId);
  }

  async listOutbox(sessionId: string): Promise<OperatorMessageRecord[]> {
    const messages = await this.list();
    return messages.filter((message) => message.fromSessionId === sessionId);
  }
}

function buildMessageSummary(summary: string | undefined, message: string): string {
  if (summary?.trim()) {
    return summary.trim();
  }
  const normalized = message.replace(/\s+/g, " ").trim();
  return normalized.length <= 80 ? normalized : `${normalized.slice(0, 77).trimEnd()}...`;
}
