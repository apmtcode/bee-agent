import { describe, expect, it } from "vitest";
import {
  buildAnthropicMessagesRequest,
  forwardAnthropicMessagesRequest,
  forwardAnthropicMessagesStreamRequest,
  normalizeAnthropicMessagesPayload,
  resolveAnthropicAuthConfig,
} from "./anthropic-client.js";

describe("anthropic-client", () => {
  it("prefers ANTHROPIC_API_KEY and keeps ANTHROPIC_AUTH_TOKEN as bearer auth", () => {
    const config = resolveAnthropicAuthConfig((name) => ({
      ANTHROPIC_API_KEY: "primary-key",
      ANTHROPIC_AUTH_TOKEN: "oauth-token",
      ANTHROPIC_BASE_URL: "https://example.anthropic.test/",
    })[name]);

    expect(config).toEqual({
      apiKey: "primary-key",
      authToken: "oauth-token",
      baseUrl: "https://example.anthropic.test/",
    });

    expect(buildAnthropicMessagesRequest({ body: "{}", config })).toEqual({
      url: "https://example.anthropic.test/v1/messages",
      body: "{}",
      headers: {
        "x-api-key": "primary-key",
        authorization: "Bearer oauth-token",
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
      },
    });
  });

  it("forwards incoming anthropic-beta headers", () => {
    const config = resolveAnthropicAuthConfig((name) => ({
      ANTHROPIC_API_KEY: "primary-key",
    })[name]);

    expect(buildAnthropicMessagesRequest({
      body: "{}",
      config,
      headers: {
        "Anthropic-Beta": "tools-2024-04-04,files-api-2025-04-14",
      },
    })).toEqual({
      url: "https://api.anthropic.com/v1/messages",
      body: "{}",
      headers: {
        "x-api-key": "primary-key",
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
        "anthropic-beta": "tools-2024-04-04,files-api-2025-04-14",
      },
    });
  });

  it("falls back to ANTHROPIC_AUTH_TOKEN for x-api-key material when no API key exists", () => {
    const config = resolveAnthropicAuthConfig((name) => ({
      ANTHROPIC_AUTH_TOKEN: "token-only",
    })[name]);

    expect(config).toEqual({
      apiKey: "token-only",
      authToken: "token-only",
      baseUrl: "https://api.anthropic.com",
    });
  });

  it("normalizes Anthropc messages by stripping orphaned tool blocks and merging same-role messages", () => {
    expect(normalizeAnthropicMessagesPayload({
      model: "claude-opus-4-7",
      messages: [
        {
          role: "assistant",
          content: [
            { type: "tool_use", id: "tool-orphan", name: "bash", input: { command: "pwd" } },
          ],
        },
        {
          role: "user",
          content: [
            { type: "tool_result", tool_use_id: "tool-keep", content: "ok" },
          ],
        },
        {
          role: "user",
          content: "follow-up",
        },
        {
          role: "assistant",
          content: [
            { type: "tool_use", id: "tool-keep", name: "bash", input: { command: "ls" } },
          ],
        },
      ],
    })).toEqual({
      model: "claude-opus-4-7",
      messages: [
        {
          role: "user",
          content: [
            { type: "text", text: "follow-up" },
          ],
        },
      ],
    });
  });

  it("preserves response metadata for non-streaming requests", async () => {
    const response = await forwardAnthropicMessagesRequest({
      body: JSON.stringify({ model: "claude-opus-4-7", stream: false }),
      fetchImpl: async () => new Response(JSON.stringify({ id: "msg_1" }), {
        status: 200,
        headers: {
          "content-type": "application/json",
          "request-id": "req_123",
          "anthropic-ratelimit-requests-remaining": "42",
          "retry-after": "1",
        },
      }),
      env: (name) => ({
        ANTHROPIC_API_KEY: "meta-key",
      })[name],
    });

    expect(response).toEqual({
      status: 200,
      headers: {
        "content-type": "application/json",
        "request-id": "req_123",
        "anthropic-ratelimit-requests-remaining": "42",
        "retry-after": "1",
      },
      body: JSON.stringify({ id: "msg_1" }),
    });
  });

  it("forwards streaming requests without consuming the SSE body", async () => {
    const stream = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(new TextEncoder().encode("event: message_start\n\n"));
        controller.close();
      },
    });
    const fetchImpl = async (input: string | URL | Request, init?: RequestInit) => {
      expect(String(input)).toBe("https://example.anthropic.test/v1/messages");
      expect(init?.headers).toMatchObject({
        "x-api-key": "stream-key",
        authorization: "Bearer stream-token",
      });
      return new Response(stream, {
        status: 200,
        headers: {
          "content-type": "text/event-stream",
          "request-id": "req_stream_123",
          "anthropic-request-id": "anth_req_stream_123",
        },
      });
    };

    const response = await forwardAnthropicMessagesStreamRequest({
      body: JSON.stringify({ model: "claude-opus-4-7", stream: true }),
      fetchImpl,
      env: (name) => ({
        ANTHROPIC_API_KEY: "stream-key",
        ANTHROPIC_AUTH_TOKEN: "stream-token",
        ANTHROPIC_BASE_URL: "https://example.anthropic.test",
      })[name],
    });

    expect(response.status).toBe(200);
    expect(response.headers).toEqual({
      "content-type": "text/event-stream",
      "request-id": "req_stream_123",
      "anthropic-request-id": "anth_req_stream_123",
    });
    expect(response.bodyStream).toBe(stream);
  });

  it("throws when Anthropic credentials are missing", () => {
    expect(() => resolveAnthropicAuthConfig(() => undefined)).toThrow("Missing Anthropic API key.");
  });
});
