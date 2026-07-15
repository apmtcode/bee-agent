import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { FileBackgroundTaskStore } from "./background-tasks.js";
import {
  SyntheticBackgroundExecutor,
  createNoopBackgroundSpawn,
  simulateTrivialCommandOutput,
} from "./synthetic-background-executor.js";

const tempDirs: string[] = [];

async function makeStore(driver = new SyntheticBackgroundExecutor()): Promise<FileBackgroundTaskStore> {
  const dir = await fs.mkdtemp(path.join(os.tmpdir(), "synthetic-executor-"));
  tempDirs.push(dir);
  return new FileBackgroundTaskStore(path.join(dir, "background-tasks.json"), undefined, undefined, driver);
}

afterEach(async () => {
  await Promise.all(tempDirs.splice(0).map((dir) => fs.rm(dir, { recursive: true, force: true })));
});

describe("simulateTrivialCommandOutput", () => {
  it("interprets printf and echo without running a shell", () => {
    expect(simulateTrivialCommandOutput("printf ok")).toBe("ok");
    expect(simulateTrivialCommandOutput("printf monitor-ok")).toBe("monitor-ok");
    expect(simulateTrivialCommandOutput("printf 'line-1\\nline-2\\n'")).toBe("line-1\nline-2\n");
    expect(simulateTrivialCommandOutput('echo "hello"')).toBe("hello\n");
  });

  it("returns nothing for commands it cannot safely simulate", () => {
    expect(simulateTrivialCommandOutput("tail -f app.log")).toBe("");
    expect(simulateTrivialCommandOutput("node build.js")).toBe("");
  });
});

describe("SyntheticBackgroundExecutor via FileBackgroundTaskStore", () => {
  it("keeps a launched task running with simulated output and no real process", async () => {
    const driver = new SyntheticBackgroundExecutor({ startPid: 4242 });
    const store = await makeStore(driver);

    const task = await store.start({ title: "smoke", command: "printf ok", kind: "task" });
    expect(task.status).toBe("running");
    expect(task.execution.processId).toBe(4242);

    // The driver reports the fake pid as alive, so a sync reconcile keeps it
    // running instead of flipping it to missing-process.
    const synced = await store.sync(task.id);
    expect(synced?.status).toBe("running");

    const output = await store.executionService.readOutput(task);
    expect(output).toContain("starting task");
    expect(output).toContain("ok");

    // Simulating process exit lets a later reconcile observe the running state
    // as orphaned.
    driver.complete(4242);
    expect(driver.isProcessRunning(4242)).toBe(false);
    const reconciled = await store.sync(task.id);
    expect(reconciled?.status).toBe("failed");
  });

  it("does not execute long-running commands", async () => {
    const store = await makeStore();
    const task = await store.start({ title: "watch", command: "tail -f app.log", kind: "monitor" });
    // If the command had actually been spawned, start() would block forever.
    expect(task.status).toBe("running");
    const output = await store.executionService.readOutput(task);
    expect(output).toBe("starting monitor " + task.id);
  });
});

describe("createNoopBackgroundSpawn", () => {
  it("hands out incrementing pids and writes no files", async () => {
    const dir = await fs.mkdtemp(path.join(os.tmpdir(), "noop-spawn-"));
    tempDirs.push(dir);
    const store = new FileBackgroundTaskStore(
      path.join(dir, "background-tasks.json"),
      createNoopBackgroundSpawn(700),
      () => false,
    );

    const task = await store.start({ title: "noop", command: "printf hi", kind: "task" });
    expect(task.execution.processId).toBe(700);

    // No detached process ran, so no execution state file exists yet — the
    // manual-state tests rely on exactly this.
    const state = await store.executionService.readState(task);
    expect(state).toBeUndefined();
  });
});
