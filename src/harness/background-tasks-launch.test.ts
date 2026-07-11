import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";

import { FileBackgroundTaskStore } from "./background-tasks.js";

// These tests exercise the REAL launcher (a spawned bash script), so they must
// clear the suite-wide dry-launch flag. The launch script now hands the state
// payload to python3 as a single argv value and writes atomically, so commands
// containing quotes, `#`, and `\n` escapes round-trip into valid state JSON —
// the previous `printf | sed` munging corrupted them.
describe("background task real launcher", () => {
  const dryRunBefore = process.env.OPENCLAW_BACKGROUND_TASK_DRY_RUN;
  let rootDir: string;

  beforeEach(async () => {
    delete process.env.OPENCLAW_BACKGROUND_TASK_DRY_RUN;
    rootDir = await fs.mkdtemp(path.join(os.tmpdir(), "bg-launch-"));
  });

  afterEach(async () => {
    if (dryRunBefore === undefined) {
      delete process.env.OPENCLAW_BACKGROUND_TASK_DRY_RUN;
    } else {
      process.env.OPENCLAW_BACKGROUND_TASK_DRY_RUN = dryRunBefore;
    }
    await fs.rm(rootDir, { recursive: true, force: true });
  });

  it("writes a valid, parseable state file for a command with quotes and newlines", async () => {
    const store = new FileBackgroundTaskStore(path.join(rootDir, "background-tasks.json"));
    // Single quotes, a shell comment char, and \n escapes — all fatal to the
    // old sed-based substitution.
    const command = "printf 'first'\\''s\\nsecond\\n' # it's fine";
    const task = await store.start({
      sessionId: "session-1",
      title: "Quoted command",
      command,
      kind: "task",
    });

    const statePath = path.join(rootDir, task.execution.stateFile);
    const state = await waitForTerminalState(statePath);

    // The command survived the launch pipeline byte-for-byte and the file is
    // valid JSON that reflects a terminal (completed) status.
    expect(state.command).toBe(command);
    expect(state.status).toBe("completed");
    expect(state.taskId).toBe(task.id);
    expect(typeof state.pid).toBe("number");
  });
});

async function waitForTerminalState(statePath: string, timeoutMs = 5000): Promise<Record<string, unknown>> {
  const deadline = Date.now() + timeoutMs;
  let lastError: unknown;
  while (Date.now() < deadline) {
    try {
      const raw = await fs.readFile(statePath, "utf8");
      const parsed = JSON.parse(raw) as Record<string, unknown>;
      if (parsed.status === "completed" || parsed.status === "failed") {
        return parsed;
      }
    } catch (error) {
      lastError = error;
    }
    await delay(50);
  }
  throw new Error(`state file never reached a terminal status: ${String(lastError)}`);
}

function delay(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
