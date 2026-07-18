import { execFileSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { describe, expect, it } from "vitest";

import {
  BackgroundTaskExecutionService,
  type BackgroundTaskExecutionState,
  type BackgroundTaskRecord,
} from "./background-tasks.js";

function hasTool(tool: string): boolean {
  try {
    execFileSync(tool, ["--version"], { stdio: "ignore" });
    return true;
  } catch {
    return false;
  }
}

// The launch script shells out to `bash` and `python3`; skip end-to-end
// execution where either is unavailable (the render logic is still exercised
// by the unit tests). Both are present in the CI/cloud environment.
const canRun = hasTool("bash") && hasTool("python3");

async function makeTempDir(): Promise<string> {
  return await fs.promises.mkdtemp(path.join(os.tmpdir(), "launch-script-"));
}

function buildTaskRecord(rootDir: string, command: string): BackgroundTaskRecord {
  const now = new Date("2026-01-01T00:00:00.000Z").toISOString();
  const dir = "background-tasks/task-1";
  return {
    version: 1,
    id: "task-1",
    title: "Integration task",
    kind: "task",
    command,
    cwd: rootDir,
    createdAt: now,
    updatedAt: now,
    status: "running",
    execution: {
      version: 1,
      preparedAt: now,
      lastUpdatedAt: now,
      workingDirectory: rootDir,
      outputFile: `${dir}/output.log`,
      stateFile: `${dir}/state.json`,
      launchScript: `${dir}/run.sh`,
    },
  };
}

describe.skipIf(!canRun)("background task launch script (end-to-end)", () => {
  it("writes valid, atomic JSON state for a command with single quotes and newlines", async () => {
    const rootDir = await makeTempDir();
    const service = new BackgroundTaskExecutionService(rootDir);
    // This command previously corrupted the state JSON: the shell `printf | sed`
    // substitution broke bash double-quote nesting on the literal `"$$"` pattern,
    // mangling the `command` field and leaving `pid` unsubstituted.
    const command = "printf 'line-1\nline-2\n'";
    const task = buildTaskRecord(rootDir, command);

    await service.writeArtifacts(task);
    const scriptPath = path.join(rootDir, task.execution.launchScript);
    execFileSync("bash", [scriptPath], { cwd: rootDir });

    const stateRaw = await fs.promises.readFile(
      path.join(rootDir, task.execution.stateFile),
      "utf8",
    );
    // The core regression guard: this would throw SyntaxError before the fix.
    const state = JSON.parse(stateRaw) as BackgroundTaskExecutionState;

    expect(state.status).toBe("completed");
    expect(state.exitCode).toBe(0);
    expect(typeof state.pid).toBe("number");
    expect(Number.isFinite(state.pid)).toBe(true);
    // The command round-trips exactly — quotes and newlines intact.
    expect(state.command).toBe(command);
    expect(state.taskId).toBe(task.id);

    // Atomic writes leave no temp files behind.
    const stateDir = path.join(rootDir, "background-tasks/task-1");
    const leftovers = (await fs.promises.readdir(stateDir)).filter((name) => name.includes(".tmp"));
    expect(leftovers).toEqual([]);

    // The command actually ran and produced output.
    const output = await fs.promises.readFile(
      path.join(rootDir, task.execution.outputFile),
      "utf8",
    );
    expect(output).toContain("line-1");
    expect(output).toContain("line-2");
  });

  it("records a non-zero exit code as a failed state", async () => {
    const rootDir = await makeTempDir();
    const service = new BackgroundTaskExecutionService(rootDir);
    const task = buildTaskRecord(rootDir, "exit 3");

    await service.writeArtifacts(task);
    const scriptPath = path.join(rootDir, task.execution.launchScript);
    let threw = false;
    try {
      execFileSync("bash", [scriptPath], { cwd: rootDir, stdio: "ignore" });
    } catch {
      threw = true; // `set -e` + `exit "$exit_code"` propagates the failure.
    }
    expect(threw).toBe(true);

    const state = JSON.parse(
      await fs.promises.readFile(path.join(rootDir, task.execution.stateFile), "utf8"),
    ) as BackgroundTaskExecutionState;
    expect(state.status).toBe("failed");
    expect(state.exitCode).toBe(3);
    expect(state.error).toContain("non-zero");
  });
});
