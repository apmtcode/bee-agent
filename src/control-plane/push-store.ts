import { randomUUID } from "node:crypto";
import { readJsonFile, writeJsonAtomic } from "../shared/fs.js";

export type WebPushSubscriptionKeys = {
  p256dh: string;
  auth: string;
};

export type WebPushSubscriptionPayload = {
  endpoint: string;
  keys: WebPushSubscriptionKeys;
};

export type PushSubscriptionRecord = {
  id: string;
  sessionId: string;
  endpoint: string;
  keys: WebPushSubscriptionKeys;
  createdAt: string;
  updatedAt: string;
  label?: string;
  userAgent?: string;
};

export type PushStoreShape = {
  version: 1;
  subscriptions: PushSubscriptionRecord[];
};

const EMPTY_STORE: PushStoreShape = {
  version: 1,
  subscriptions: [],
};

const MAX_ENDPOINT_LENGTH = 2048;
const MAX_KEY_LENGTH = 512;
const MAX_LABEL_LENGTH = 120;
const MAX_USER_AGENT_LENGTH = 512;

export class FilePushStore {
  constructor(private readonly filePath: string) {}

  async load(): Promise<PushStoreShape> {
    return await readJsonFile(this.filePath, EMPTY_STORE);
  }

  async save(store: PushStoreShape): Promise<void> {
    await writeJsonAtomic(this.filePath, store);
  }

  async upsertSubscription(params: {
    sessionId: string;
    subscription: WebPushSubscriptionPayload;
    label?: string;
    userAgent?: string;
  }): Promise<PushSubscriptionRecord> {
    const endpoint = normalizeEndpoint(params.subscription.endpoint);
    const keys = normalizeKeys(params.subscription.keys);
    const label = normalizeOptionalBoundedString(params.label, "label", MAX_LABEL_LENGTH);
    const userAgent = normalizeOptionalBoundedString(params.userAgent, "userAgent", MAX_USER_AGENT_LENGTH);
    const store = await this.load();
    const now = new Date().toISOString();
    const existing = store.subscriptions.find((item) => item.sessionId === params.sessionId && item.endpoint === endpoint);
    const record: PushSubscriptionRecord = existing
      ? {
          ...existing,
          keys,
          updatedAt: now,
          ...(label ? { label } : {}),
          ...(userAgent ? { userAgent } : {}),
        }
      : {
          id: randomUUID(),
          sessionId: params.sessionId,
          endpoint,
          keys,
          createdAt: now,
          updatedAt: now,
          ...(label ? { label } : {}),
          ...(userAgent ? { userAgent } : {}),
        };
    const nextSubscriptions = existing
      ? store.subscriptions.map((item) => item.id === existing.id ? record : item)
      : [...store.subscriptions, record];
    await this.save({ version: 1, subscriptions: nextSubscriptions });
    return record;
  }

  async listSubscriptions(sessionId?: string): Promise<PushSubscriptionRecord[]> {
    const store = await this.load();
    const subscriptions = sessionId
      ? store.subscriptions.filter((item) => item.sessionId === sessionId)
      : store.subscriptions;
    return [...subscriptions].sort((a, b) => a.createdAt.localeCompare(b.createdAt));
  }

  async deleteSubscription(params: { id?: string; endpoint?: string; sessionId?: string }): Promise<boolean> {
    const id = normalizeOptionalBoundedString(params.id, "id", 256);
    const endpoint = params.endpoint == null ? undefined : normalizeEndpoint(params.endpoint);
    const sessionId = normalizeOptionalBoundedString(params.sessionId, "sessionId", 256);
    if (!id && !endpoint) {
      throw new Error("push.subscriptions.delete requires id or endpoint");
    }
    const store = await this.load();
    const filtered = store.subscriptions.filter((item) => {
      if (id && item.id === id) {
        return false;
      }
      if (endpoint && item.endpoint === endpoint && (!sessionId || item.sessionId === sessionId)) {
        return false;
      }
      return true;
    });
    if (filtered.length === store.subscriptions.length) {
      return false;
    }
    await this.save({ version: 1, subscriptions: filtered });
    return true;
  }
}

export function normalizeWebPushSubscriptionPayload(value: unknown): WebPushSubscriptionPayload {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new Error("Invalid push subscription");
  }
  const record = value as Record<string, unknown>;
  return {
    endpoint: normalizeEndpoint(record.endpoint),
    keys: normalizeKeys(record.keys),
  };
}

function normalizeEndpoint(value: unknown): string {
  if (typeof value !== "string" || !value.trim()) {
    throw new Error("Invalid push subscription endpoint");
  }
  const endpoint = value.trim();
  if (endpoint.length > MAX_ENDPOINT_LENGTH) {
    throw new Error("Invalid push subscription endpoint");
  }
  let url: URL;
  try {
    url = new URL(endpoint);
  } catch {
    throw new Error("Invalid push subscription endpoint");
  }
  if (url.protocol !== "https:") {
    throw new Error("Invalid push subscription endpoint");
  }
  return endpoint;
}

function normalizeKeys(value: unknown): WebPushSubscriptionKeys {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new Error("Invalid push subscription keys");
  }
  const record = value as Record<string, unknown>;
  return {
    p256dh: normalizeBoundedString(record.p256dh, "p256dh", MAX_KEY_LENGTH),
    auth: normalizeBoundedString(record.auth, "auth", MAX_KEY_LENGTH),
  };
}

function normalizeBoundedString(value: unknown, key: string, maxLength: number): string {
  if (typeof value !== "string" || !value.trim() || value.trim().length > maxLength) {
    throw new Error(`Invalid push subscription ${key}`);
  }
  return value.trim();
}

function normalizeOptionalBoundedString(value: unknown, key: string, maxLength: number): string | undefined {
  if (value == null) {
    return undefined;
  }
  return normalizeBoundedString(value, key, maxLength);
}
