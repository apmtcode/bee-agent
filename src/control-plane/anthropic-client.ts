const DEFAULT_ANTHROPIC_BASE_URL = "https://api.anthropic.com";
const ANTHROPIC_VERSION = "2023-06-01";

type AnthropicMessageRole = "user" | "assistant";

type AnthropicMessageBlock = {
  type?: unknown;
  text?: unknown;
  id?: unknown;
  tool_use_id?: unknown;
  [key: string]: unknown;
};

type AnthropicMessage = {
  role: AnthropicMessageRole;
  content: AnthropicMessageBlock[];
};

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

export function buildAnthropicMessagesRequest(params: {
  body: string;
  config: AnthropicAuthConfig;
  headers?: Record<string, string | undefined>;
}): AnthropicHttpRequest {
  const headers: Record<string, string> = {
    "anthropic-version": ANTHROPIC_VERSION,
    "content-type": "application/json",
  };
  if (params.config.apiKey) {
    headers["x-api-key"] = params.config.apiKey;
  }
  if (params.config.authToken) {
    headers.authorization = `Bearer ${params.config.authToken}`;
  }
  const betaHeader = readIncomingHeader(params.headers, "anthropic-beta");
  if (betaHeader) {
    headers["anthropic-beta"] = betaHeader;
  }
  return {
    body: params.body,
    headers,
    url: `${params.config.baseUrl.replace(/\/+$/, "")}/v1/messages`,
  };
}

export function normalizeAnthropicMessagesPayload(payload: Record<string, unknown>): Record<string, unknown> {
  if (!Array.isArray(payload.messages)) {
    return payload;
  }
  return {
    ...payload,
    messages: normalizeAnthropicMessages(payload.messages),
  };
}

export async function forwardAnthropicMessagesRequest(params: {
  body: string;
  fetchImpl?: typeof fetch;
  env?: EnvReader;
  headers?: Record<string, string | undefined>;
}): Promise<AnthropicMessageForwardResponse> {
  const request = buildAnthropicMessagesRequest({
    body: params.body,
    config: resolveAnthropicAuthConfig(params.env),
    headers: params.headers,
  });
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
  headers?: Record<string, string | undefined>;
}): Promise<AnthropicMessageStreamForwardResponse> {
  const request = buildAnthropicMessagesRequest({
    body: params.body,
    config: resolveAnthropicAuthConfig(params.env),
    headers: params.headers,
  });
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
  for (const name of [
    "request-id",
    "anthropic-request-id",
    "anthropic-organization-id",
    "anthropic-ratelimit-requests-limit",
    "anthropic-ratelimit-requests-remaining",
    "anthropic-ratelimit-requests-reset",
    "retry-after",
  ]) {
    const value = response.headers.get(name);
    if (value) {
      headers[name] = value;
    }
  }
  return headers;
}

function readIncomingHeader(
  headers: Record<string, string | undefined> | undefined,
  name: string,
): string | undefined {
  if (!headers) {
    return undefined;
  }
  const target = name.toLowerCase();
  for (const [key, value] of Object.entries(headers)) {
    if (key.toLowerCase() === target && value?.trim()) {
      return value.trim();
    }
  }
  return undefined;
}

function normalizeAnthropicMessages(messages: unknown[]): AnthropicMessage[] {
  const normalized = messages.flatMap((message) => normalizeAnthropicMessage(message));

  const precedingToolUseIds = new Set<string>();
  for (const message of normalized) {
    if (message.role === "assistant") {
      for (const block of message.content) {
        if (block.type === "tool_use" && typeof block.id === "string" && block.id.trim()) {
          precedingToolUseIds.add(block.id);
        }
      }
      continue;
    }
    message.content = message.content.filter(
      (block) => block.type !== "tool_result" || (typeof block.tool_use_id === "string" && precedingToolUseIds.has(block.tool_use_id)),
    );
  }

  const toolResultIds = new Set<string>();
  for (const message of normalized) {
    if (message.role !== "user") {
      continue;
    }
    for (const block of message.content) {
      if (block.type === "tool_result" && typeof block.tool_use_id === "string" && block.tool_use_id.trim()) {
        toolResultIds.add(block.tool_use_id);
      }
    }
  }

  for (const message of normalized) {
    if (message.role !== "assistant") {
      continue;
    }
    message.content = message.content.filter(
      (block) => block.type !== "tool_use" || (typeof block.id === "string" && toolResultIds.has(block.id)),
    );
  }

  const merged: AnthropicMessage[] = [];
  for (const message of normalized.filter((item) => item.content.length > 0)) {
    const previous = merged.at(-1);
    if (previous && previous.role === message.role) {
      previous.content = [...previous.content, ...message.content];
      continue;
    }
    merged.push({
      role: message.role,
      content: [...message.content],
    });
  }

  return merged;
}

function normalizeAnthropicMessage(message: unknown): AnthropicMessage[] {
  if (!message || typeof message !== "object") {
    return [];
  }
  const role = (message as { role?: unknown }).role;
  if (role !== "user" && role !== "assistant") {
    return [];
  }
  return [{
    role,
    content: normalizeAnthropicContent((message as { content?: unknown }).content),
  }];
}

function normalizeAnthropicContent(content: unknown): AnthropicMessageBlock[] {
  if (Array.isArray(content)) {
    const blocks = content.flatMap((block) => normalizeAnthropicBlock(block));
    return blocks.length > 0 ? blocks : [textBlock("(empty message)")];
  }
  if (typeof content === "string") {
    return [textBlock(content.trim() || "(empty message)")];
  }
  return [textBlock("(empty message)")];
}

function normalizeAnthropicBlock(block: unknown): AnthropicMessageBlock[] {
  if (typeof block === "string") {
    return [textBlock(block.trim() || "(empty message)")];
  }
  if (!block || typeof block !== "object") {
    return [];
  }
  const candidate = { ...(block as AnthropicMessageBlock) };
  if (candidate.type === "text") {
    candidate.text = typeof candidate.text === "string" && candidate.text.trim()
      ? candidate.text
      : "(empty message)";
  }
  return [candidate];
}

function textBlock(text: string): AnthropicMessageBlock {
  return {
    type: "text",
    text,
  };
}

function readProcessEnv(name: string): string | undefined {
  return process.env[name];
}
