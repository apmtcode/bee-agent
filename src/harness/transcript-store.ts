import fs from "node:fs";
import { randomUUID } from "node:crypto";
import { ensureParentDir } from "../shared/fs.js";
import type { TranscriptMessageRole } from "./types.js";

export type TranscriptHeader = {
  type: "session";
  version: 1;
  id: string;
  timestamp: string;
  cwd?: string;
};

export type TranscriptMessageRecord = {
  role: TranscriptMessageRole;
  content: string;
  timestamp: number;
  metadata?: Record<string, unknown>;
};

export type TranscriptRecord =
  | TranscriptHeader
  | {
      id: string;
      message: TranscriptMessageRecord;
    };

export type TranscriptAppendMessageParams = {
  role: TranscriptMessageRole;
  content: string;
  messageId?: string;
  timestamp?: number;
  metadata?: Record<string, unknown>;
};

export type TranscriptReadOptions = {
  limit?: number;
  reverse?: boolean;
  includeHeader?: boolean;
};

export class JsonlTranscriptStore {
  constructor(private readonly filePath: string) {}

  async ensureHeader(params: { sessionId: string; cwd?: string }): Promise<void> {
    if (fs.existsSync(this.filePath)) {
      return;
    }
    await ensureParentDir(this.filePath);
    const header: TranscriptHeader = {
      type: "session",
      version: 1,
      id: params.sessionId,
      timestamp: new Date().toISOString(),
      ...(params.cwd ? { cwd: params.cwd } : {}),
    };
    await fs.promises.writeFile(this.filePath, `${JSON.stringify(header)}\n`, {
      encoding: "utf8",
      mode: 0o600,
    });
  }

  async appendMessage(params: TranscriptAppendMessageParams): Promise<string> {
    const messageId = params.messageId ?? randomUUID();
    const record: TranscriptRecord = {
      id: messageId,
      message: {
        role: params.role,
        content: params.content,
        timestamp: params.timestamp ?? Date.now(),
        ...(params.metadata ? { metadata: params.metadata } : {}),
      },
    };
    await fs.promises.appendFile(this.filePath, `${JSON.stringify(record)}\n`, "utf8");
    return messageId;
  }

  async readAll(): Promise<TranscriptRecord[]> {
    return await this.read({ includeHeader: true });
  }

  async read(options: TranscriptReadOptions = {}): Promise<TranscriptRecord[]> {
    const records = await this.loadRecords();
    const includeHeader = options.includeHeader ?? true;
    const filtered = includeHeader ? records : records.filter((record) => "message" in record);
    const ordered = options.reverse ? [...filtered].reverse() : filtered;
    if (!options.limit || options.limit >= ordered.length) {
      return ordered;
    }
    return ordered.slice(0, options.limit);
  }

  async readMessages(options: TranscriptReadOptions = {}): Promise<Array<Extract<TranscriptRecord, { message: TranscriptMessageRecord }>>> {
    const records = await this.read({ ...options, includeHeader: false });
    return records.filter((record): record is Extract<TranscriptRecord, { message: TranscriptMessageRecord }> =>
      "message" in record,
    );
  }

  async readLatestAssistantText(): Promise<string | undefined> {
    const records = await this.readMessages({ reverse: true });
    for (const record of records) {
      if (record.message.role === "assistant") {
        const text = record.message.content.trim();
        if (text) {
          return text;
        }
      }
    }
    return undefined;
  }

  async readTail(limit: number): Promise<TranscriptRecord[]> {
    if (limit <= 0) {
      return [];
    }
    const records = await this.loadRecords();
    return records.slice(-limit);
  }

  private async loadRecords(): Promise<TranscriptRecord[]> {
    try {
      const raw = await fs.promises.readFile(this.filePath, "utf8");
      return raw
        .split("\n")
        .map((line) => line.trim())
        .filter(Boolean)
        .map((line) => JSON.parse(line) as TranscriptRecord);
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code === "ENOENT") {
        return [];
      }
      throw error;
    }
  }
}
