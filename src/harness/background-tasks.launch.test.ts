import { execFile } from "node:child_process";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { promisify } from "node:util";
import { randomUUID } from "node:crypto";
import { afterEach, describe, expect, it } from "vitest";
import { FileBackgroundTaskStore } from "./background-tasks.js";

const run = promisify(execFile);
const tempDirs: string[] = [];

async function makeTempDir(): Promise<string> {
  const dir = await fs.mkdtemp(path.join(os.tmpdir(), "bg-launch-"));
  tempDirs.push(dir);
  return dir;
}

afterEach(async () => {
  await Promise.all(tempDirs.splice(0).map((dir) => fs.rm(dir, { recursive: true, force: true })));
});

describe("background task launch script", () => {
  // Regression: the launch script builds the initial state.json in shell. A
  // shell-quoting bug (the wrong single-quote escape) plus a fragile `sed`
  // substitution used to corrupt the JSON whenever the command contained single
  // quotes or newlines, leaving an unparseable state file that broke recovery.
  it("writes valid, parseable state JSON for a command with quotes and newlines", async () => {
    const rootDir = await makeTempDir();
    const store = new FileBackgroundTaskStore(path.join(rootDir, "tasks.json"));
    const command = "printf 'line-1\nline-2\n'";

    const task = await store.start({ sessionId: "s1", title: "quoted", command, kind: "task" });

    // The default (inert) spawn does not run the script, so run it ourselves to
    // exercise the real shell that a production launch would execute.
    const scriptPath = path.join(rootDir, task.execution.launchScript);
    await run("bash", [scriptPath], { cwd: rootDir });

    const statePath = path.join(rootDir, task.execution.stateFile);
    const raw = await fs.readFile(statePath, "utf8");

    // Must be valid JSON (the bug produced a SyntaxError here).
    const state = JSON.parse(raw) as Record<string, unknown>;
    expect(state.taskId).toBe(task.id);
    expect(state.status).toBe("completed");
    expect(state.exitCode).toBe(0);
    // pid is converted to a real integer, not the literal placeholder "$$".
    expect(typeof state.pid).toBe("number");
    // The command round-trips exactly, quotes and newlines intact.
    expect(state.command).toBe(command);
  });
});
