import fs from "node:fs/promises";
import path from "node:path";
import { ensureParentDir } from "../shared/fs.js";

export type DeliveryTarget = { kind: "local" } | { kind: "webhook"; url: string; headers?: Record<string, string> };

export type CronDeliveryConfig = {
  onSuccess?: DeliveryTarget[];
  onFailure?: DeliveryTarget[];
};

export type DeliveryAttemptStatus = "sent" | "suppressed" | "failed";
export type DeliverySummaryStatus = "sent" | "suppressed" | "failed" | "partial_failed";

export type DeliveryAttemptResult = {
  target: DeliveryTarget;
  status: DeliveryAttemptStatus;
  attemptedAt: string;
  error?: string;
};

export type CronTerminalDeliveryPayload = {
  kind: "cron-terminal-state";
  cron: {
    jobId: string;
    runId: string;
    cron: string;
    prompt: string;
    recurring: boolean;
    status: string;
    outcome?: string;
    summary?: string;
    error?: string;
    scheduledFor?: string;
    startedAt?: string;
    finishedAt?: string;
    sessionId?: string;
    operatorRunId?: string;
  };
};

export type PushNotificationDeliveryPayload = {
  kind: "push-notification";
  notification: {
    id: string;
    status: string;
    message: string;
    createdAt: string;
    sessionId?: string;
  };
};

export type DeliveryPayload = CronTerminalDeliveryPayload | PushNotificationDeliveryPayload;

function nowIso(): string {
  return new Date().toISOString();
}

function normalizeError(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

async function appendJsonLine(filePath: string, value: unknown): Promise<void> {
  await ensureParentDir(filePath);
  await fs.appendFile(filePath, `${JSON.stringify(value)}\n`, { encoding: "utf8", mode: 0o600 });
}

export function summarizeDeliveryResults(results: DeliveryAttemptResult[]): {
  lastDeliveryStatus: DeliverySummaryStatus;
  lastDeliveryError?: string;
} {
  if (results.length === 0) {
    return { lastDeliveryStatus: "suppressed" };
  }
  const statuses = new Set(results.map((result) => result.status));
  const firstError = results.find((result) => typeof result.error === "string")?.error;
  if (statuses.size === 1) {
    const [status] = [...statuses];
    return {
      lastDeliveryStatus: status === "sent" || status === "suppressed" || status === "failed" ? status : "failed",
      ...(firstError ? { lastDeliveryError: firstError } : {}),
    };
  }
  return {
    lastDeliveryStatus: "partial_failed",
    ...(firstError ? { lastDeliveryError: firstError } : {}),
  };
}

export class OperatorDeliveryService {
  private readonly localDeliveryFile: string;
  private readonly fetchImpl: typeof fetch;

  constructor(
    rootDir: string,
    options: {
      fetchImpl?: typeof fetch;
    } = {},
  ) {
    this.localDeliveryFile = path.join(rootDir, "delivery-local.jsonl");
    this.fetchImpl = options.fetchImpl ?? fetch;
  }

  async deliver(targets: DeliveryTarget[], payload: DeliveryPayload): Promise<DeliveryAttemptResult[]> {
    const results: DeliveryAttemptResult[] = [];
    for (const target of targets) {
      results.push(await this.deliverToTarget(target, payload));
    }
    return results;
  }

  private async deliverToTarget(target: DeliveryTarget, payload: DeliveryPayload): Promise<DeliveryAttemptResult> {
    const attemptedAt = nowIso();
    try {
      if (target.kind === "local") {
        await appendJsonLine(this.localDeliveryFile, { attemptedAt, target, payload });
        return { target, status: "sent", attemptedAt };
      }
      const response = await this.fetchImpl(target.url, {
        method: "POST",
        headers: {
          "content-type": "application/json",
          ...(target.headers ?? {}),
        },
        body: JSON.stringify(payload),
        signal: AbortSignal.timeout(5_000),
      });
      if (!response.ok) {
        return {
          target,
          status: "failed",
          attemptedAt,
          error: `Webhook responded with ${response.status}.`,
        };
      }
      return { target, status: "sent", attemptedAt };
    } catch (error) {
      return {
        target,
        status: "failed",
        attemptedAt,
        error: normalizeError(error),
      };
    }
  }
}
