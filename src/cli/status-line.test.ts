import process from "node:process";
import { afterEach, describe, expect, it } from "vitest";
import { buildOperatorCliStatusLineSnapshot, formatOperatorCliPrompt, normalizeOperatorCliStatusLineOutput, OperatorCliStatusLineController, runOperatorCliStatusLineCommand, type OperatorCliStatusLineSnapshot } from "./status-line.js";
import { StandaloneOperatorRuntime } from "../orchestrator/operator-runtime.js";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";

const tempDirs: string[] = [];

async function makeTempDir(): Promise<string> {
  const dir = await fs.mkdtemp(path.join(os.tmpdir(), "operator-cli-status-line-"));
  tempDirs.push(dir);
  return dir;
}

afterEach(async () => {
  await Promise.all(tempDirs.splice(0).map((dir) => fs.rm(dir, { recursive: true, force: true })));
});

function createSnapshot(overrides: Partial<OperatorCliStatusLineSnapshot> = {}): OperatorCliStatusLineSnapshot {
  return {
    version: 1,
    sessionId: "session-1",
    cwd: "/tmp/project",
    currentDate: "2026-06-04",
    permissionMode: "default",
    session: { status: "idle" },
    approvals: { pending: 0 },
    tasks: { total: 0, pending: 0, inProgress: 0, completed: 0 },
    ...overrides,
  };
}

describe("normalizeOperatorCliStatusLineOutput", () => {
  it("keeps the first non-empty normalized line", () => {
    expect(normalizeOperatorCliStatusLineOutput("\n  hello   world  \nsecond line\n")).toBe("hello world");
  });

  it("applies padding and width truncation", () => {
    expect(normalizeOperatorCliStatusLineOutput("hello world", { padding: 1 })).toBe(" hello world ");
    expect(normalizeOperatorCliStatusLineOutput("hello world", { width: 5 })).toBe("hell…");
    expect(normalizeOperatorCliStatusLineOutput("hello", { width: 1 })).toBe("h");
  });

  it("returns undefined for empty output", () => {
    expect(normalizeOperatorCliStatusLineOutput("\n   \n")).toBeUndefined();
  });
});

describe("formatOperatorCliPrompt", () => {
  it("prepends the status line when present", () => {
    expect(formatOperatorCliPrompt("status", "operator> ")).toBe("status\noperator> ");
    expect(formatOperatorCliPrompt(undefined, "operator> ")).toBe("operator> ");
  });
});

describe("runOperatorCliStatusLineCommand", () => {
  it("passes snapshot json on stdin", async () => {
    const cwd = await makeTempDir();
    const output = await runOperatorCliStatusLineCommand({
      command: "node -e \"let data=''; process.stdin.on('data', chunk => data += chunk); process.stdin.on('end', () => { const parsed = JSON.parse(data); process.stdout.write(parsed.sessionId + ' ' + parsed.permissionMode); });\"",
      cwd,
      snapshot: createSnapshot({ permissionMode: "acceptEdits" }),
    });
    expect(output).toBe("session-1 acceptEdits");
  });

  it("returns undefined on failing or timing out commands", async () => {
    const cwd = await makeTempDir();
    await expect(runOperatorCliStatusLineCommand({
      command: "exit 1",
      cwd,
      snapshot: createSnapshot(),
    })).resolves.toBeUndefined();
    await expect(runOperatorCliStatusLineCommand({
      command: "sleep 1",
      cwd,
      snapshot: createSnapshot(),
      timeoutMs: 50,
    })).resolves.toBeUndefined();
  });

  it("truncates oversized command output", async () => {
    const cwd = await makeTempDir();
    const longText = "x".repeat(400);
    const output = await runOperatorCliStatusLineCommand({
      command: `python3 - <<'PY'\nprint('${longText}')\nPY`,
      cwd,
      snapshot: createSnapshot(),
      maxOutputBytes: 256,
    });
    expect(output).toBe("x".repeat(253) + "…");
  });
});

describe("buildOperatorCliStatusLineSnapshot", () => {
  it("collects session runtime state", async () => {
    const rootDir = await makeTempDir();
    const runtime = new StandaloneOperatorRuntime({ rootDir });
    const session = await runtime.startSession({ title: "status", cwd: rootDir, agentId: "operator-cli" });
    await runtime.createTask({ sessionId: session.id, subject: "pending task", description: "pending task" });
    const inProgressTask = await runtime.createTask({ sessionId: session.id, subject: "active task", description: "active task" });
    await runtime.updateTask(inProgressTask.id, { status: "in_progress" });
    await runtime.promptApproval({ sessionId: session.id, title: "Approve", summary: "Need approval" });
    const run = await runtime.startRun({ sessionId: session.id, title: "Current run" });

    const snapshot = await buildOperatorCliStatusLineSnapshot({
      runtime,
      sessionId: session.id,
      cwd: rootDir,
      currentDate: "2026-06-04",
      config: { merged: { permissionMode: "acceptEdits" }, loadedEntries: [] },
    });

    expect(snapshot).toMatchObject({
      version: 1,
      sessionId: session.id,
      cwd: rootDir,
      currentDate: "2026-06-04",
      permissionMode: "acceptEdits",
      session: { status: session.status },
      approvals: { pending: 1 },
      tasks: { total: 2, pending: 1, inProgress: 1, completed: 0 },
      activeRun: { id: run.id, status: run.status, title: "Current run" },
    });
  });
});

describe("OperatorCliStatusLineController", () => {
  it("refreshes rendered line and coalesces duplicate refreshes", async () => {
    const rendered: Array<string | undefined> = [];
    let buildCount = 0;
    const cwd = await makeTempDir();
    const stdout = process.stdout as typeof process.stdout & { columns?: number };
    stdout.isTTY = true;
    stdout.columns = 80;
    const controller = new OperatorCliStatusLineController({
      config: { type: "command", command: "printf ready" },
      stdout,
      buildSnapshot: async () => {
        buildCount += 1;
        return createSnapshot({ cwd });
      },
      onRenderedLineChange: (line) => {
        rendered.push(line);
      },
    });

    await controller.start();
    expect(controller.currentLine).toBe("ready");

    await Promise.all([controller.refresh(), controller.refresh()]);
    expect(controller.currentLine).toBe("ready");
    expect(buildCount).toBeGreaterThanOrEqual(2);

    await controller.updateConfig(undefined);
    expect(controller.currentLine).toBeUndefined();
    expect(rendered.at(-1)).toBeUndefined();
  });

  it("stays disabled on non-tty stdout", async () => {
    const controller = new OperatorCliStatusLineController({
      config: { type: "command", command: "printf ready" },
      stdout: { isTTY: false, columns: 80 },
      buildSnapshot: async () => createSnapshot(),
    });

    await controller.start();
    expect(controller.currentLine).toBeUndefined();
  });
});
