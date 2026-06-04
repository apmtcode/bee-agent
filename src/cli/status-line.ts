import { spawn } from "node:child_process";
import type { Writable } from "node:stream";
import process from "node:process";
import type { StandaloneOperatorRuntime } from "../orchestrator/operator-runtime.js";
import { resolveOperatorCliExecutionConfig, type OperatorCliRuntimeConfig, type OperatorCliStatusLineConfig } from "./config.js";

export type OperatorCliStatusLineSnapshot = {
  version: 1;
  sessionId: string;
  cwd: string;
  currentDate: string;
  permissionMode: string;
  session: {
    status: string;
  };
  approvals: {
    pending: number;
  };
  tasks: {
    total: number;
    pending: number;
    inProgress: number;
    completed: number;
  };
  activeRun?: {
    id: string;
    status: string;
    title: string;
  };
  activeTask?: {
    id: string;
    status: string;
    title: string;
    kind: string;
  };
};

export type OperatorCliStatusLineControllerOptions = {
  config?: OperatorCliStatusLineConfig;
  stdout: Writable & { isTTY?: boolean; columns?: number };
  buildSnapshot: () => Promise<OperatorCliStatusLineSnapshot>;
  onRenderedLineChange?: (line?: string) => void;
  commandTimeoutMs?: number;
  maxOutputBytes?: number;
};

export async function buildOperatorCliStatusLineSnapshot(params: {
  runtime: StandaloneOperatorRuntime;
  sessionId: string;
  cwd: string;
  currentDate: string;
  config?: OperatorCliRuntimeConfig;
}): Promise<OperatorCliStatusLineSnapshot> {
  const [session, activeRun, approvals, activeTask, tasks] = await Promise.all([
    params.runtime.getSession(params.sessionId),
    params.runtime.getActiveRun(params.sessionId),
    params.runtime.listApprovals(params.sessionId),
    params.runtime.getActiveBackgroundTask(params.sessionId),
    params.runtime.listTasks(params.sessionId),
  ]);

  return {
    version: 1,
    sessionId: params.sessionId,
    cwd: params.cwd,
    currentDate: params.currentDate,
    permissionMode: resolveOperatorCliExecutionConfig(params.config).permissionMode,
    session: {
      status: session?.status ?? "unknown",
    },
    approvals: {
      pending: approvals.length,
    },
    tasks: {
      total: tasks.length,
      pending: tasks.filter((task) => task.status === "pending").length,
      inProgress: tasks.filter((task) => task.status === "in_progress").length,
      completed: tasks.filter((task) => task.status === "completed").length,
    },
    ...(activeRun
      ? {
          activeRun: {
            id: activeRun.id,
            status: activeRun.status,
            title: activeRun.title,
          },
        }
      : {}),
    ...(activeTask
      ? {
          activeTask: {
            id: activeTask.id,
            status: activeTask.status,
            title: activeTask.title,
            kind: activeTask.kind,
          },
        }
      : {}),
  };
}

export async function runOperatorCliStatusLineCommand(params: {
  command: string;
  cwd: string;
  snapshot: OperatorCliStatusLineSnapshot;
  padding?: number;
  width?: number;
  timeoutMs?: number;
  maxOutputBytes?: number;
}): Promise<string | undefined> {
  const command = params.command.trim();
  if (!command) {
    return undefined;
  }

  const stdout = await executeStatusLineCommand({
    command,
    cwd: params.cwd,
    snapshot: params.snapshot,
    timeoutMs: params.timeoutMs,
    maxOutputBytes: params.maxOutputBytes,
  });
  return normalizeOperatorCliStatusLineOutput(stdout, {
    padding: params.padding,
    width: params.width,
  });
}

export function normalizeOperatorCliStatusLineOutput(
  output: string,
  options: {
    padding?: number;
    width?: number;
  } = {},
): string | undefined {
  const firstLine = output
    .split(/\r?\n/)
    .map((line) => line.replace(/\s+/g, " ").trim())
    .find((line) => line.length > 0);
  if (!firstLine) {
    return undefined;
  }

  const padding = normalizePadding(options.padding);
  const padded = padding > 0 ? `${" ".repeat(padding)}${firstLine}${" ".repeat(padding)}` : firstLine;
  const width = typeof options.width === "number" && Number.isFinite(options.width) ? Math.max(0, Math.floor(options.width)) : undefined;
  if (!width || padded.length <= width) {
    return padded;
  }
  if (width <= 1) {
    return padded.slice(0, width);
  }
  return `${padded.slice(0, width - 1)}…`;
}

export function formatOperatorCliPrompt(statusLine: string | undefined, basePrompt = "operator> "): string {
  return statusLine ? `${statusLine}\n${basePrompt}` : basePrompt;
}

export class OperatorCliStatusLineController {
  private config?: OperatorCliStatusLineConfig;
  private timer?: NodeJS.Timeout;
  private disposed = false;
  private inFlight = false;
  private pendingRefresh = false;
  private renderedLine?: string;

  constructor(private readonly options: OperatorCliStatusLineControllerOptions) {
    this.config = options.config;
  }

  get currentLine(): string | undefined {
    return this.renderedLine;
  }

  async start(): Promise<void> {
    this.resetTimer();
    await this.refresh();
  }

  async updateConfig(config?: OperatorCliStatusLineConfig): Promise<void> {
    this.config = config;
    this.resetTimer();
    if (!config) {
      this.setRenderedLine(undefined);
      return;
    }
    await this.refresh();
  }

  async refresh(): Promise<void> {
    if (this.disposed) {
      return;
    }
    if (this.inFlight) {
      this.pendingRefresh = true;
      return;
    }
    this.inFlight = true;
    try {
      if (!this.config || !this.options.stdout.isTTY) {
        this.setRenderedLine(undefined);
        return;
      }
      const snapshot = await this.options.buildSnapshot();
      const nextLine = await runOperatorCliStatusLineCommand({
        command: this.config.command,
        cwd: snapshot.cwd,
        snapshot,
        padding: this.config.padding,
        width: this.options.stdout.columns,
        timeoutMs: this.options.commandTimeoutMs,
        maxOutputBytes: this.options.maxOutputBytes,
      });
      this.setRenderedLine(nextLine);
    } finally {
      this.inFlight = false;
      if (this.pendingRefresh) {
        this.pendingRefresh = false;
        queueMicrotask(() => {
          void this.refresh();
        });
      }
    }
  }

  async stop(): Promise<void> {
    this.disposed = true;
    this.clearTimer();
    this.setRenderedLine(undefined);
  }

  private setRenderedLine(line?: string): void {
    if (this.renderedLine === line) {
      return;
    }
    this.renderedLine = line;
    this.options.onRenderedLineChange?.(line);
  }

  private resetTimer(): void {
    this.clearTimer();
    if (!this.config || !this.options.stdout.isTTY) {
      return;
    }
    const refreshMs = resolveStatusLineRefreshMs(this.config.refreshInterval);
    this.timer = setInterval(() => {
      void this.refresh();
    }, refreshMs);
    this.timer.unref?.();
  }

  private clearTimer(): void {
    if (!this.timer) {
      return;
    }
    clearInterval(this.timer);
    this.timer = undefined;
  }
}

async function executeStatusLineCommand(params: {
  command: string;
  cwd: string;
  snapshot: OperatorCliStatusLineSnapshot;
  timeoutMs?: number;
  maxOutputBytes?: number;
}): Promise<string> {
  const timeoutMs = typeof params.timeoutMs === "number" && Number.isFinite(params.timeoutMs)
    ? Math.max(50, Math.floor(params.timeoutMs))
    : 1000;
  const maxOutputBytes = typeof params.maxOutputBytes === "number" && Number.isFinite(params.maxOutputBytes)
    ? Math.max(256, Math.floor(params.maxOutputBytes))
    : 16 * 1024;

  return await new Promise((resolve) => {
    const child = spawn("bash", ["-lc", params.command], {
      cwd: params.cwd,
      env: {
        ...process.env,
        OPERATOR_STATUSLINE_SESSION_ID: params.snapshot.sessionId,
        OPERATOR_STATUSLINE_CWD: params.snapshot.cwd,
      },
      stdio: ["pipe", "pipe", "pipe"],
    });

    let stdout = "";
    let settled = false;
    let timedOut = false;
    let overflowed = false;

    const settle = (value: string): void => {
      if (settled) {
        return;
      }
      settled = true;
      clearTimeout(timer);
      resolve(value);
    };

    const timer = setTimeout(() => {
      timedOut = true;
      child.kill("SIGTERM");
      setTimeout(() => {
        if (!child.killed) {
          child.kill("SIGKILL");
        }
      }, 100).unref?.();
    }, timeoutMs);
    timer.unref?.();

    child.stdout.on("data", (chunk: Buffer | string) => {
      stdout += chunk.toString();
      if (Buffer.byteLength(stdout, "utf8") > maxOutputBytes) {
        overflowed = true;
        child.kill("SIGTERM");
      }
    });

    child.on("error", () => {
      settle("");
    });
    child.on("close", () => {
      if (timedOut) {
        settle("");
        return;
      }
      if (overflowed) {
        settle(truncateUtf8WithEllipsis(stdout, maxOutputBytes));
        return;
      }
      settle(stdout);
    });

    child.stdin.end(`${JSON.stringify(params.snapshot)}\n`);
  });
}

function resolveStatusLineRefreshMs(refreshInterval?: number): number {
  if (typeof refreshInterval !== "number" || !Number.isFinite(refreshInterval) || refreshInterval <= 0) {
    return 1000;
  }
  return Math.max(250, Math.floor(refreshInterval * 1000));
}

function normalizePadding(padding?: number): number {
  if (typeof padding !== "number" || !Number.isFinite(padding) || padding <= 0) {
    return 0;
  }
  return Math.max(0, Math.floor(padding));
}

function truncateUtf8(value: string, maxBytes: number): string {
  if (Buffer.byteLength(value, "utf8") <= maxBytes) {
    return value;
  }
  let end = value.length;
  while (end > 0 && Buffer.byteLength(value.slice(0, end), "utf8") > maxBytes) {
    end -= 1;
  }
  return value.slice(0, end);
}

function truncateUtf8WithEllipsis(value: string, maxBytes: number): string {
  if (Buffer.byteLength(value, "utf8") <= maxBytes) {
    return value;
  }
  const ellipsis = "…";
  const ellipsisBytes = Buffer.byteLength(ellipsis, "utf8");
  if (maxBytes <= ellipsisBytes) {
    return truncateUtf8(ellipsis, maxBytes);
  }
  return `${truncateUtf8(value, maxBytes - ellipsisBytes)}${ellipsis}`;
}
