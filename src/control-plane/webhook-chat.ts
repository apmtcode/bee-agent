import type { WebhookChatReplyTarget, WebhookChatSessionMetadata } from "../harness/types.js";

export type WebhookChatInboundMessage = {
  text: string;
  senderId: string;
  senderName?: string;
};

export type WebhookChatPayload = {
  deliveryId?: string;
  conversationId: string;
  threadId?: string;
  message: WebhookChatInboundMessage;
  reply: WebhookChatReplyTarget;
};

export type NormalizedWebhookChatPayload = {
  deliveryId?: string;
  conversationId: string;
  threadId?: string;
  text: string;
  senderId: string;
  senderName?: string;
  reply: WebhookChatReplyTarget;
};

export type WebhookChatReplyRequest = {
  url: string;
  headers: Record<string, string>;
  body: string;
};

const MAX_HEADER_COUNT = 16;
const MAX_RECENT_DELIVERY_IDS = 20;

export function normalizeWebhookChatPayload(payload: Record<string, unknown>): NormalizedWebhookChatPayload {
  const conversationId = getRequiredString(payload, "conversationId");
  const threadId = getOptionalString(payload, "threadId");
  const deliveryId = getOptionalString(payload, "deliveryId");
  const message = getRecord(payload, "message");
  const reply = getRecord(payload, "reply");
  const text = getRequiredString(message, "text");
  const senderId = getRequiredString(message, "senderId");
  const senderName = getOptionalString(message, "senderName");
  return {
    conversationId,
    ...(threadId ? { threadId } : {}),
    ...(deliveryId ? { deliveryId } : {}),
    text,
    senderId,
    ...(senderName ? { senderName } : {}),
    reply: normalizeReplyTarget(reply),
  };
}

export function buildWebhookChatRemoteId(channel: string, payload: Pick<NormalizedWebhookChatPayload, "conversationId" | "threadId">): string {
  const normalizedChannel = normalizeChannel(channel);
  const binding = payload.threadId ?? payload.conversationId;
  return `${normalizedChannel}:${binding}`;
}

export function buildWebhookChatSessionMetadata(
  channel: string,
  remoteId: string,
  payload: NormalizedWebhookChatPayload,
  existing?: WebhookChatSessionMetadata,
): WebhookChatSessionMetadata {
  const recentDeliveryIds = rememberDeliveryId(existing?.recentDeliveryIds, payload.deliveryId);
  return {
    channel: normalizeChannel(channel),
    remoteId,
    conversationId: payload.conversationId,
    ...(payload.threadId ? { threadId: payload.threadId } : {}),
    senderId: payload.senderId,
    ...(payload.senderName ? { senderName: payload.senderName } : {}),
    reply: payload.reply,
    ...(recentDeliveryIds.length > 0 ? { recentDeliveryIds } : {}),
    lastInboundAt: new Date().toISOString(),
    ...(existing?.lastOutboundAt ? { lastOutboundAt: existing.lastOutboundAt } : {}),
  };
}

export function hasWebhookChatDeliveryId(metadata: WebhookChatSessionMetadata | undefined, deliveryId: string | undefined): boolean {
  if (!deliveryId) {
    return false;
  }
  return Boolean(metadata?.recentDeliveryIds?.includes(deliveryId));
}

export function markWebhookChatReplyDelivered(
  metadata: WebhookChatSessionMetadata,
  deliveredAt = new Date().toISOString(),
): WebhookChatSessionMetadata {
  return {
    ...metadata,
    lastOutboundAt: deliveredAt,
  };
}

export function buildWebhookChatReplyRequest(params: {
  metadata: WebhookChatSessionMetadata;
  sessionId: string;
  text: string;
}): WebhookChatReplyRequest {
  const headers: Record<string, string> = {
    "content-type": "application/json",
    ...params.metadata.reply.headers,
  };
  return {
    url: params.metadata.reply.url,
    headers,
    body: JSON.stringify({
      sessionId: params.sessionId,
      channel: params.metadata.channel,
      conversationId: params.metadata.conversationId,
      ...(params.metadata.threadId ? { threadId: params.metadata.threadId } : {}),
      recipient: {
        senderId: params.metadata.senderId,
        ...(params.metadata.senderName ? { senderName: params.metadata.senderName } : {}),
      },
      text: params.text,
    }),
  };
}

function normalizeReplyTarget(value: Record<string, unknown>): WebhookChatReplyTarget {
  const url = getRequiredString(value, "url");
  if (!/^https?:\/\//.test(url)) {
    throw new Error("Invalid reply.url: must start with http:// or https://");
  }
  const headersValue = value.headers;
  if (headersValue == null) {
    return { url };
  }
  if (!isRecord(headersValue)) {
    throw new Error("Invalid reply.headers: expected object");
  }
  const entries = Object.entries(headersValue);
  if (entries.length > MAX_HEADER_COUNT) {
    throw new Error(`Invalid reply.headers: maximum ${MAX_HEADER_COUNT} headers`);
  }
  const headers: Record<string, string> = {};
  for (const [key, headerValue] of entries) {
    if (!key.trim()) {
      throw new Error("Invalid reply.headers: header names must be non-empty");
    }
    if (typeof headerValue !== "string" || !headerValue.trim()) {
      throw new Error(`Invalid reply.headers.${key}: expected non-empty string`);
    }
    headers[key] = headerValue.trim();
  }
  return { url, headers };
}

function normalizeChannel(channel: string): string {
  const normalized = channel.trim().toLowerCase();
  if (!normalized || !/^[a-z0-9._-]+$/.test(normalized)) {
    throw new Error(`Invalid webhook chat channel: ${channel}`);
  }
  return normalized;
}

function rememberDeliveryId(existing: string[] | undefined, deliveryId: string | undefined): string[] {
  const next = [...(existing ?? [])];
  if (deliveryId) {
    next.push(deliveryId);
  }
  if (next.length <= MAX_RECENT_DELIVERY_IDS) {
    return next;
  }
  return next.slice(next.length - MAX_RECENT_DELIVERY_IDS);
}

function getRecord(value: Record<string, unknown>, key: string): Record<string, unknown> {
  const field = value[key];
  if (!isRecord(field)) {
    throw new Error(`Invalid ${key}: expected object`);
  }
  return field;
}

function getRequiredString(value: Record<string, unknown>, key: string): string {
  const field = value[key];
  if (typeof field !== "string" || !field.trim()) {
    throw new Error(`Invalid ${key}: expected non-empty string`);
  }
  return field.trim();
}

function getOptionalString(value: Record<string, unknown>, key: string): string | undefined {
  const field = value[key];
  if (field == null) {
    return undefined;
  }
  if (typeof field !== "string" || !field.trim()) {
    throw new Error(`Invalid ${key}: expected non-empty string`);
  }
  return field.trim();
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}
