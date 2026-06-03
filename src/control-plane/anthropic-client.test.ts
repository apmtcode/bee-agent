import { describe, expect, it } from "vitest";
import {
  buildAnthropicMessagesRequest,
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

    expect(buildAnthropicMessagesRequest("{}", config)).toEqual({
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

  it("throws when Anthropic credentials are missing", () => {
    expect(() => resolveAnthropicAuthConfig(() => undefined)).toThrow("Missing Anthropic API key.");
  });
});
