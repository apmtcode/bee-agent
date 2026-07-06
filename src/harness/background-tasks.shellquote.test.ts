import { execFileSync } from "node:child_process";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import {
  BackgroundTaskExecutionService,
  type BackgroundTaskRecord,
} from "./background-tasks.js";

// Regression test for a shellQuote bug: single quotes in a task command were
// escaped as `"'"'"'` instead of the correct POSIX idiom `'"'"'`, injecting
// stray `"` characters that corrupted the generated state.json (invalid JSON,
// breaking background-task recovery). This exercises the *generated shell*
// end-to-end through real bash, so the corruption cannot silently return.

const tempDirs: string[] = [];

async function makeTempDir(): Promise<string> {
  const dir = await fs.mkdtemp(path.join(os.tmpdir(), "bg-shellquote-"));
  tempDirs.push(dir);
  return dir;
}

afterEach(async () => {
  await Promise.all(tempDirs.splice(0).map((dir) => fs.rm(dir, { recursive: true, force: true })));
});

function taskWithCommand(command: string): BackgroundTaskRecord {
  return {
    version: 1,
    id: "quote-test-0001",
    title: "quoting",
    kind: "task",
    command,
    cwd: ".",
    createdAt: "2026-01-01T00:00:00.000Z",
    updatedAt: "2026-01-01T00:00:00.000Z",
    status: "planned",
    execution: {
      version: 1,
      preparedAt: "2026-01-01T00:00:00.000Z",
      lastUpdatedAt: "2026-01-01T00:00:00.000Z",
      workingDirectory: ".",
      outputFile: "output.log",
      stateFile: "state.json",
      launchScript: "run.sh",
      processId: 4242,
    },
  };
}

describe("background-task launch-script shell quoting", () => {
  // These commands all contain single quotes — the character shellQuote must
  // escape correctly for the embedded JSON payload to survive bash re-parsing.
  const commands = [
    "printf 'line-1\nline-2\n'",
    "echo 'it''s a test'",
    "bash -lc 'echo \"nested $HOME\"'",
    "grep -r 'needle' .",
  ];

  it.each(commands)("produces a valid JSON state file for: %s", async (command) => {
    const rootDir = await makeTempDir();
    const service = new BackgroundTaskExecutionService(rootDir);
    const task = taskWithCommand(command);
    await service.writeArtifacts(task);

    const script = await fs.readFile(path.join(rootDir, task.execution.launchScript), "utf8");
    // Run only the state-writing prologue (mkdir + started_at + printf|sed),
    // not the task command itself — we are validating the generated JSON.
    const prologue = script.split("\n").slice(0, 5).join("\n");
    execFileSync("bash", ["-c", prologue], { cwd: rootDir });

    const raw = await fs.readFile(path.join(rootDir, task.execution.stateFile), "utf8");
    const parsed = JSON.parse(raw); // throws on the pre-fix corruption
    expect(parsed.command).toBe(command);
    expect(parsed.status).toBe("running");
    expect(parsed.taskId).toBe(task.id);
  });
});
