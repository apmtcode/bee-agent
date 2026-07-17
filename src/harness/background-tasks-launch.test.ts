import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { FileBackgroundTaskStore } from "./background-tasks.js";
import { createSynchronousSpawnBackgroundProcess } from "./background-tasks-testing.js";

const tempDirs: string[] = [];

async function makeTempDir(): Promise<string> {
  const dir = await fs.mkdtemp(path.join(os.tmpdir(), "background-launch-"));
  tempDirs.push(dir);
  return dir;
}

afterEach(async () => {
  await Promise.all(tempDirs.splice(0).map((dir) => fs.rm(dir, { recursive: true, force: true })));
});

describe("background task launch script", () => {
  // Regression: the launch script's shell quoting must survive commands that
  // contain single quotes (previously the escape sequence was `"'"'"'` instead
  // of `'"'"'`, which corrupted the emitted state JSON), and the process id must
  // be substituted as a real number (previously `sed "s/"$$"/$$/g"` broke bash
  // double-quoting and left pid as the literal string "$$").
  it("writes valid state JSON with a numeric pid for a single-quoted command", async () => {
    const rootDir = await makeTempDir();
    const store = new FileBackgroundTaskStore(
      path.join(rootDir, "background-tasks.json"),
      createSynchronousSpawnBackgroundProcess(),
    );

    const command = "printf 'line-1\nline-2\n'";
    const task = await store.start({ title: "quoting", command, kind: "task" });

    // readState throws on malformed JSON, so a successful read proves the state
    // file is well-formed even though the command carried single quotes/newlines.
    const state = await store.executionService.readState(task);
    expect(state).toBeDefined();
    if (!state) {
      throw new Error("expected execution state");
    }

    expect(typeof state.pid).toBe("number");
    expect(Number.isFinite(state.pid)).toBe(true);
    expect(state.pid).toBeGreaterThan(0);
    expect(state.status).toBe("completed");
    expect(state.exitCode).toBe(0);
    // The command round-trips exactly, including the embedded single quotes.
    expect(state.command).toBe(command);

    const output = await store.executionService.readOutput(task, { lineLimit: 10 });
    expect(output).toContain("line-1");
    expect(output).toContain("line-2");
  });
});
