import { randomUUID } from "node:crypto";
import { readJsonFile, writeJsonAtomic } from "../shared/fs.js";
import type { SessionMetadata, SessionRecord, SessionStatus } from "./types.js";

export type SessionStoreRecord = Record<string, SessionRecord>;

export type SessionStoreShape = {
  version: 1;
  sessions: SessionStoreRecord;
};

const EMPTY_STORE: SessionStoreShape = {
  version: 1,
  sessions: {},
};

function nowIso(): string {
  return new Date().toISOString();
}

export class FileSessionStore {
  constructor(private readonly filePath: string) {}

  async load(): Promise<SessionStoreShape> {
    return await readJsonFile(this.filePath, EMPTY_STORE);
  }

  async save(store: SessionStoreShape): Promise<void> {
    await writeJsonAtomic(this.filePath, store);
  }

  async list(): Promise<SessionRecord[]> {
    const store = await this.load();
    return Object.values(store.sessions).sort((a, b) => b.updatedAt.localeCompare(a.updatedAt));
  }

  async get(sessionId: string): Promise<SessionRecord | undefined> {
    const store = await this.load();
    return store.sessions[sessionId];
  }

  async upsert(params: {
    id?: string;
    status?: SessionStatus;
    transcriptFile?: string;
    metadata?: SessionMetadata;
  }): Promise<SessionRecord> {
    const store = await this.load();
    const id = params.id ?? randomUUID();
    const record = buildSessionRecord(store.sessions[id], id, params);
    store.sessions[id] = record;
    await this.save(store);
    return record;
  }

  async update(
    sessionId: string,
    params: {
      status?: SessionStatus;
      transcriptFile?: string;
      metadata?: SessionMetadata;
    },
  ): Promise<SessionRecord | undefined> {
    const store = await this.load();
    const existing = store.sessions[sessionId];
    if (!existing) {
      return undefined;
    }
    const record = buildSessionRecord(existing, sessionId, params);
    store.sessions[sessionId] = record;
    await this.save(store);
    return record;
  }
}

function buildSessionRecord(
  existing: SessionRecord | undefined,
  id: string,
  params: {
    status?: SessionStatus;
    transcriptFile?: string;
    metadata?: SessionMetadata;
  },
): SessionRecord {
  return {
    id,
    createdAt: existing?.createdAt ?? nowIso(),
    updatedAt: nowIso(),
    status: params.status ?? existing?.status ?? "active",
    transcriptFile: params.transcriptFile ?? existing?.transcriptFile,
    metadata: {
      ...(existing?.metadata ?? {}),
      ...(params.metadata ?? {}),
    },
  };
}
