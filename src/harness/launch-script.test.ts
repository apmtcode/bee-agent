import { spawn } from "node:child_process";
import { promises as fs } from "node:fs";
import os from "node:os";
import path from "node:path";
import { describe, expect, it } from "vitest";

import { renderLaunchScript, type BackgroundTaskRecord } from "./background-tasks.js";

function makeTask(command: string, overrides: Partial<BackgroundTaskRecord> = {}): BackgroundTaskRecord {
  const id = "11111111-2222-3333-4444-555555555555";
  const baseDir = `background-tasks/${id}`;
  return {
    version: 1,
    id,
    title: "launch-script probe",
    command,
    cwd: ".",
    kind: "task",
    createdAt: "2026-01-01T00:00:00.000Z",
    updatedAt: "2026-01-01T00:00:00.000Z",
    status: "planned",
    execution: {
      version: 1,
      preparedAt: "2026-01-01T00:00:00.000Z",
      lastUpdatedAt: "2026-01-01T00:00:00.000Z",
      workingDirectory: baseDir,
      outputFile: `${baseDir}/output.log`,
      stateFile: `${baseDir}/state.json`,
      launchScript: `${baseDir}/run.sh`,
    },
    ...overrides,
  } as BackgroundTaskRecord;
}

async function runScript(task: BackgroundTaskRecord, rootDir: string): Promise<number> {
  const scriptPath = path.join(rootDir, task.execution.launchScript);
  await fs.mkdir(path.dirname(scriptPath), { recursive: true });
  await fs.writeFile(scriptPath, renderLaunchScript(task), { mode: 0o700 });
  return await new Promise<number>((resolve, reject) => {
    const child = spawn(scriptPath, [], { cwd: rootDir, stdio: "ignore" });
    child.on("error", reject);
    child.on("exit", (code) => resolve(code ?? -1));
  });
}

describe("renderLaunchScript state writer", () => {
  // Commands full of shell/JSON hazards used to corrupt state.json because it
  // was built with `printf | sed`. The Python writer must round-trip them.
  const hazardous = "printf 'line-1\nline-2\n' && echo \"quoted $HOME\" # trailing";

  it("writes valid JSON state with a numeric pid for a completed task", async () => {
    const rootDir = await fs.mkdtemp(path.join(os.tmpdir(), "launch-ok-"));
    const task = makeTask(hazardous);
    const exitCode = await runScript(task, rootDir);
    expect(exitCode).toBe(0);

    const raw = await fs.readFile(path.join(rootDir, task.execution.stateFile), "utf8");
    const state = JSON.parse(raw) as Record<string, unknown>;
    expect(state.taskId).toBe(task.id);
    expect(state.status).toBe("completed");
    // pid must be an actual number, not the literal string "$$".
    expect(typeof state.pid).toBe("number");
    expect(state.pid as number).toBeGreaterThan(0);
    // The command round-trips exactly despite the embedded quotes/newlines.
    expect(state.command).toBe(hazardous);
    expect(state.exitCode).toBe(0);
  });

  it("records a failed state with the real exit code and valid JSON", async () => {
    const rootDir = await fs.mkdtemp(path.join(os.tmpdir(), "launch-fail-"));
    const task = makeTask("exit 7");
    const exitCode = await runScript(task, rootDir);
    expect(exitCode).toBe(7);

    const raw = await fs.readFile(path.join(rootDir, task.execution.stateFile), "utf8");
    const state = JSON.parse(raw) as Record<string, unknown>;
    expect(state.status).toBe("failed");
    expect(state.exitCode).toBe(7);
    expect(typeof state.pid).toBe("number");
  });
});
