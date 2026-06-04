import { describe, expect, it } from "vitest";
import {
  buildWebhookChatRemoteId,
  buildWebhookChatReplyRequest,
  buildWebhookChatSessionMetadata,
  hasWebhookChatDeliveryId,
  markWebhookChatReplyDelivered,
  normalizeWebhookChatPayload,
} from "./webhook-chat.js";

describe("normalizeWebhookChatPayload", () => {
  it("normalizes a valid inbound payload", () => {
    expect(normalizeWebhookChatPayload({
      deliveryId: "delivery-1",
      conversationId: "conversation-1",
      threadId: "thread-1",
      message: {
        text: "hello operator",
        senderId: "user-1",
        senderName: "Pat",
      },
      reply: {
        url: "https://example.test/reply",
        headers: { authorization: "Bearer token" },
      },
    })).toEqual({
      deliveryId: "delivery-1",
      conversationId: "conversation-1",
      threadId: "thread-1",
      text: "hello operator",
      senderId: "user-1",
      senderName: "Pat",
      reply: {
        url: "https://example.test/reply",
        headers: { authorization: "Bearer token" },
      },
    });
  });

  it("rejects invalid payloads", () => {
    expect(() => normalizeWebhookChatPayload({})).toThrow("Invalid conversationId");
    expect(() => normalizeWebhookChatPayload({
      conversationId: "conversation-1",
      message: { text: "hello", senderId: "user-1" },
      reply: { url: "ftp://example.test/reply" },
    })).toThrow("Invalid reply.url");
  });
});

describe("webhook chat helpers", () => {
  it("derives deterministic remote ids and session metadata", () => {
    const payload = normalizeWebhookChatPayload({
      deliveryId: "delivery-1",
      conversationId: "conversation-1",
      threadId: "thread-1",
      message: { text: "hello", senderId: "user-1", senderName: "Pat" },
      reply: { url: "https://example.test/reply" },
    });
    const remoteId = buildWebhookChatRemoteId("Local-Web", payload);
    expect(remoteId).toBe("local-web:thread-1");

    const metadata = buildWebhookChatSessionMetadata("Local-Web", remoteId, payload);
    expect(metadata).toMatchObject({
      channel: "local-web",
      remoteId,
      conversationId: "conversation-1",
      threadId: "thread-1",
      senderId: "user-1",
      senderName: "Pat",
      reply: { url: "https://example.test/reply" },
      recentDeliveryIds: ["delivery-1"],
    });
  });

  it("tracks duplicate deliveries and reply timestamps", () => {
    const payload = normalizeWebhookChatPayload({
      deliveryId: "delivery-1",
      conversationId: "conversation-1",
      message: { text: "hello", senderId: "user-1" },
      reply: { url: "https://example.test/reply" },
    });
    const metadata = buildWebhookChatSessionMetadata("local-web", "local-web:conversation-1", payload);
    expect(hasWebhookChatDeliveryId(metadata, "delivery-1")).toBe(true);
    expect(hasWebhookChatDeliveryId(metadata, "delivery-2")).toBe(false);
    expect(markWebhookChatReplyDelivered(metadata, "2026-06-04T00:00:00.000Z")).toMatchObject({
      lastOutboundAt: "2026-06-04T00:00:00.000Z",
    });
  });

  it("builds outbound callback requests", () => {
    const metadata = buildWebhookChatSessionMetadata(
      "local-web",
      "local-web:conversation-1",
      normalizeWebhookChatPayload({
        conversationId: "conversation-1",
        threadId: "thread-1",
        message: { text: "hello", senderId: "user-1", senderName: "Pat" },
        reply: { url: "https://example.test/reply", headers: { authorization: "Bearer token" } },
      }),
    );
    expect(buildWebhookChatReplyRequest({
      metadata,
      sessionId: "session-1",
      text: "reply text",
    })).toEqual({
      url: "https://example.test/reply",
      headers: {
        "content-type": "application/json",
        authorization: "Bearer token",
      },
      body: JSON.stringify({
        sessionId: "session-1",
        channel: "local-web",
        conversationId: "conversation-1",
        threadId: "thread-1",
        recipient: { senderId: "user-1", senderName: "Pat" },
        text: "reply text",
      }),
    });
  });
});
