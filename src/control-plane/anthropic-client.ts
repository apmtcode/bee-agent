const DEFAULT_ANTHROPIC_BASE_URL = "https://api.anthropic.com";
const ANTHROPIC_VERSION = "2023-06-01";

export type EnvReader = (name: string) => string | undefined;

export type AnthropicAuthConfig = {
  apiKey?: string;
  authToken?: string;
  baseUrl: string;
};

export type AnthropicHttpRequest = {
  body: string;
  headers: Record<string, string>;
  url: string;
};

export type AnthropicMessageForwardResponse = {
  status: number;
  headers: Record<string, string>;
  body: string;
};

export type AnthropicMessageStreamForwardResponse = {
  status: number;
  headers: Record<string, string>;
  bodyStream: ReadableStream<Uint8Array> | null;
};

function readNonEmptyEnv(env: EnvReader, names: string[]): string | undefined {
  for (const name of names) {
    const value = env(name)?.trim();
    if (value) {
      return value;
    }
  }
  return undefined;
}

export function resolveAnthropicAuthConfig(env: EnvReader = readProcessEnv): AnthropicAuthConfig {
  const apiKey = readNonEmptyEnv(env, ["ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"]);
  if (!apiKey) {
    throw new Error("Missing Anthropic API key.");
  }
  return {
    apiKey,
    authToken: readNonEmptyEnv(env, ["ANTHROPIC_AUTH_TOKEN"]),
    baseUrl: readNonEmptyEnv(env, ["ANTHROPIC_BASE_URL"]) ?? DEFAULT_ANTHROPIC_BASE_URL,
  };
}

export function buildAnthropicMessagesRequest(body: string, config: AnthropicAuthConfig): AnthropicHttpRequest {
  const headers: Record<string, string> = {
    "anthropic-version": ANTHROPIC_VERSION,
    "content-type": "application/json",
  };
  if (config.apiKey) {
    headers["x-api-key"] = config.apiKey;
  }
  if (config.authToken) {
    headers.authorization = `Bearer ${config.authToken}`;
  }
  return {
    body,
    headers,
    url: `${config.baseUrl.replace(/\/+$/, "")}/v1/messages`,
  };
}

export async function forwardAnthropicMessagesRequest(params: {
  body: string;
  fetchImpl?: typeof fetch;
  env?: EnvReader;
}): Promise<AnthropicMessageForwardResponse> {
  const request = buildAnthropicMessagesRequest(params.body, resolveAnthropicAuthConfig(params.env));
  const response = await (params.fetchImpl ?? fetch)(request.url, {
    method: "POST",
    headers: request.headers,
    body: request.body,
    signal: AbortSignal.timeout(30_000),
  });
  return {
    status: response.status,
    headers: copyAnthropicResponseHeaders(response, "application/json"),
    body: await response.text(),
  };
}

export async function forwardAnthropicMessagesStreamRequest(params: {
  body: string;
  fetchImpl?: typeof fetch;
  env?: EnvReader;
}): Promise<AnthropicMessageStreamForwardResponse> {
  const request = buildAnthropicMessagesRequest(params.body, resolveAnthropicAuthConfig(params.env));
  const response = await (params.fetchImpl ?? fetch)(request.url, {
    method: "POST",
    headers: request.headers,
    body: request.body,
    signal: AbortSignal.timeout(30_000),
  });
  return {
    status: response.status,
    headers: copyAnthropicResponseHeaders(response, "text/event-stream"),
    bodyStream: response.body,
  };
}

function copyAnthropicResponseHeaders(response: Response, defaultContentType: string): Record<string, string> {
  const headers: Record<string, string> = {
    "content-type": response.headers.get("content-type") ?? defaultContentType,
  };
  for (const name of ["request-id", "anthropic-request-id", "anthropic-organization-id", "anthropic-ratelimit-requests-limit", "anthropic-ratelimit-requests-remaining", "anthropic-ratelimit-requests-reset", "retry-after"]) {
    const value = response.headers.get(name);
    if (value) {
      headers[name] = value;
    }
  }
  return headers;
}

function readProcessEnv(name: string): string | undefined {
  return process.env[name];
}
