import { execFileSync } from "node:child_process";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import {
  BackgroundTaskExecutionService,
  FileBackgroundTaskStore,
  type BackgroundTaskExecutionState,
} from "./background-tasks.js";

const tempDirs: string[] = [];

async function makeTempDir(): Promise<string> {
  const dir = await fs.mkdtemp(path.join(os.tmpdir(), "background-task-"));
  tempDirs.push(dir);
  return dir;
}

afterEach(async () => {
  await Promise.all(tempDirs.splice(0).map((dir) => fs.rm(dir, { recursive: true, force: true })));
});

describe("FileBackgroundTaskStore", () => {
  it("starts tasks, persists output, syncs terminal state, and reloads", async () => {
    const rootDir = await makeTempDir();
    const filePath = path.join(rootDir, "background-tasks.json");
    const spawned: Array<{ command: string; cwd: string }> = [];
    const store = new FileBackgroundTaskStore(
      filePath,
      (command, _args, options) => {
        spawned.push({ command, cwd: options.cwd });
        return { pid: 4242, unref() {} };
      },
      () => true,
    );

    const task = await store.start({
      sessionId: "sess-1",
      title: "Collect logs",
      command: "printf 'line-1\nline-2\n'",
      cwd: rootDir,
      kind: "task",
      metadata: { source: "test" },
    });

    expect(task.status).toBe("running");
    expect(task.execution.processId).toBe(4242);
    expect(spawned).toEqual([
      {
        command: path.join(rootDir, task.execution.launchScript),
        cwd: rootDir,
      },
    ]);
    await expect(store.get(task.id)).resolves.toMatchObject({
      id: task.id,
      sessionId: "sess-1",
      kind: "task",
      status: "running",
      metadata: { source: "test" },
    });

    const runningState: BackgroundTaskExecutionState = {
      version: 1,
      taskId: task.id,
      kind: "task",
      status: "running",
      pid: 4242,
      startedAt: "2026-01-01T00:00:00.000Z",
      updatedAt: "2026-01-01T00:00:00.000Z",
      outputFile: task.execution.outputFile,
      cwd: rootDir,
      command: "printf 'line-1\\nline-2\\n'",
    };
    await store.executionService.writeState(task, runningState);
    await store.executionService.writeOutput(task, "line-1\nline-2\nline-3\n");
    await expect(store.sync(task.id)).resolves.toMatchObject({
      id: task.id,
      status: "running",
      execution: { processId: 4242 },
    });
    await expect(store.getExecutionState(task.id)).resolves.toMatchObject({
      taskId: task.id,
      status: "running",
      pid: 4242,
    });
    await expect(store.getOutput(task.id, 2)).resolves.toBe("line-2\nline-3");

    await store.executionService.writeState(task, {
      ...runningState,
      status: "completed",
      updatedAt: "2026-01-01T00:10:00.000Z",
      completedAt: "2026-01-01T00:10:00.000Z",
      exitCode: 0,
    });
    await expect(store.sync(task.id)).resolves.toMatchObject({
      id: task.id,
      status: "completed",
      execution: {
        completedAt: "2026-01-01T00:10:00.000Z",
        exitCode: 0,
      },
    });

    const reloaded = new FileBackgroundTaskStore(filePath);
    await expect(reloaded.get(task.id)).resolves.toMatchObject({
      id: task.id,
      status: "completed",
      execution: { exitCode: 0 },
    });
  });

  it("cancels running tasks and records cancelled state", async () => {
    const rootDir = await makeTempDir();
    const filePath = path.join(rootDir, "background-tasks.json");
    const store = new FileBackgroundTaskStore(filePath, () => ({ pid: 9898, unref() {} }));

    const task = await store.start({
      title: "Watch logs",
      command: "tail -f app.log",
      cwd: rootDir,
      kind: "monitor",
    });
    await store.executionService.writeState(task, {
      version: 1,
      taskId: task.id,
      kind: "monitor",
      status: "running",
      pid: 9898,
      startedAt: "2026-01-01T00:00:00.000Z",
      updatedAt: "2026-01-01T00:00:00.000Z",
      outputFile: task.execution.outputFile,
      cwd: rootDir,
      command: "tail -f app.log",
    });

    const originalKill = process.kill;
    const kills: Array<{ pid: number; signal?: NodeJS.Signals }> = [];
    process.kill = ((pid: number, signal?: NodeJS.Signals | number) => {
      kills.push({ pid, signal: typeof signal === "string" ? signal : undefined });
      return true;
    }) as typeof process.kill;
    try {
      await expect(store.cancel(task.id, { signal: "SIGTERM" })).resolves.toMatchObject({
        id: task.id,
        status: "cancelled",
        execution: {
          signal: "SIGTERM",
          error: "background task terminated by SIGTERM",
        },
      });
    } finally {
      process.kill = originalKill;
    }

    expect(kills).toEqual([{ pid: -9898, signal: "SIGTERM" }]);
    await expect(store.getExecutionState(task.id)).resolves.toMatchObject({
      taskId: task.id,
      status: "cancelled",
      signal: "SIGTERM",
    });
  });

  it("recovers running tasks from persisted state after restart", async () => {
    const rootDir = await makeTempDir();
    const filePath = path.join(rootDir, "background-tasks.json");
    const store = new FileBackgroundTaskStore(filePath, () => ({ pid: 7777, unref() {} }), () => true);

    const task = await store.start({
      sessionId: "sess-1",
      title: "Collect logs",
      command: "printf 'line-1'",
      cwd: rootDir,
    });
    await store.executionService.writeState(task, {
      version: 1,
      taskId: task.id,
      kind: "task",
      status: "running",
      pid: 7777,
      startedAt: "2026-01-01T00:00:00.000Z",
      updatedAt: "2026-01-01T00:00:00.000Z",
      outputFile: task.execution.outputFile,
      cwd: rootDir,
      command: task.command,
    });

    const reloaded = new FileBackgroundTaskStore(filePath, () => ({ pid: 7777, unref() {} }), () => true);
    await expect(reloaded.recover(task.id)).resolves.toMatchObject({
      changed: true,
      reason: "state-running",
      task: {
        id: task.id,
        status: "running",
        execution: { processId: 7777 },
      },
    });
  });

  it("fails recovered tasks whose process disappeared", async () => {
    const rootDir = await makeTempDir();
    const filePath = path.join(rootDir, "background-tasks.json");
    const store = new FileBackgroundTaskStore(filePath, () => ({ pid: 5555, unref() {} }), () => false);

    const task = await store.start({
      sessionId: "sess-2",
      title: "Watch logs",
      command: "tail -f app.log",
      cwd: rootDir,
      kind: "monitor",
    });
    await store.executionService.writeState(task, {
      version: 1,
      taskId: task.id,
      kind: "monitor",
      status: "running",
      pid: 5555,
      startedAt: "2026-01-01T00:00:00.000Z",
      updatedAt: "2026-01-01T00:00:00.000Z",
      outputFile: task.execution.outputFile,
      cwd: rootDir,
      command: task.command,
    });

    const reloaded = new FileBackgroundTaskStore(filePath, () => ({ pid: 5555, unref() {} }), () => false);
    const recovered = await reloaded.recover(task.id);
    expect(recovered).toMatchObject({
      changed: true,
      reason: "missing-process",
      task: {
        id: task.id,
        status: "failed",
        execution: {
          processId: 5555,
          error: "background task process 5555 is no longer running",
          exitCode: 1,
        },
      },
    });
    await expect(reloaded.getExecutionState(task.id)).resolves.toMatchObject({
      taskId: task.id,
      status: "failed",
      error: "background task process 5555 is no longer running",
    });
  });

  it("rebinds running tasks that lost their state file but still have a live pid", async () => {
    const rootDir = await makeTempDir();
    const filePath = path.join(rootDir, "background-tasks.json");
    const store = new FileBackgroundTaskStore(filePath, () => ({ pid: 2222, unref() {} }), () => true);

    const task = await store.start({
      sessionId: "sess-3",
      title: "Tail logs",
      command: "tail -f app.log",
      cwd: rootDir,
      kind: "monitor",
    });
    const statePath = path.join(rootDir, task.execution.stateFile);
    await fs.rm(statePath, { force: true });

    const reloaded = new FileBackgroundTaskStore(filePath, () => ({ pid: 2222, unref() {} }), () => true);
    const recovered = await reloaded.recover(task.id);
    expect(recovered).toMatchObject({
      changed: true,
      reason: "process-running",
      task: {
        id: task.id,
        status: "running",
        execution: { processId: 2222 },
      },
    });
    await expect(reloaded.getExecutionState(task.id)).resolves.toMatchObject({
      taskId: task.id,
      status: "running",
      pid: 2222,
    });
  });

  it("recovers tasks by session and supports reason filtering upstream", async () => {
    const rootDir = await makeTempDir();
    const filePath = path.join(rootDir, "background-tasks.json");
    const store = new FileBackgroundTaskStore(
      filePath,
      () => ({ pid: 3333, unref() {} }),
      (pid) => pid === 3333,
    );

    const running = await store.start({
      sessionId: "sess-a",
      title: "Tail A",
      command: "tail -f a.log",
      cwd: rootDir,
      kind: "monitor",
    });
    const missing = await store.start({
      sessionId: "sess-a",
      title: "Tail B",
      command: "tail -f b.log",
      cwd: rootDir,
      kind: "monitor",
    });
    const other = await store.start({
      sessionId: "sess-b",
      title: "Tail C",
      command: "tail -f c.log",
      cwd: rootDir,
      kind: "monitor",
    });

    await store.executionService.writeState(running, {
      version: 1,
      taskId: running.id,
      kind: "monitor",
      status: "running",
      pid: 3333,
      startedAt: "2026-01-01T00:00:00.000Z",
      updatedAt: "2026-01-01T00:00:00.000Z",
      outputFile: running.execution.outputFile,
      cwd: rootDir,
      command: running.command,
    });
    await store.executionService.writeState(missing, {
      version: 1,
      taskId: missing.id,
      kind: "monitor",
      status: "running",
      pid: 4444,
      startedAt: "2026-01-01T00:00:00.000Z",
      updatedAt: "2026-01-01T00:00:00.000Z",
      outputFile: missing.execution.outputFile,
      cwd: rootDir,
      command: missing.command,
    });
    await store.executionService.writeState(other, {
      version: 1,
      taskId: other.id,
      kind: "monitor",
      status: "running",
      pid: 5555,
      startedAt: "2026-01-01T00:00:00.000Z",
      updatedAt: "2026-01-01T00:00:00.000Z",
      outputFile: other.execution.outputFile,
      cwd: rootDir,
      command: other.command,
    });

    const reloaded = new FileBackgroundTaskStore(filePath, () => ({ pid: 3333, unref() {} }), (pid) => pid === 3333);
    const recovered = await reloaded.recoverBySession("sess-a");
    expect(recovered).toHaveLength(2);
    expect(recovered).toEqual([
      expect.objectContaining({ task: expect.objectContaining({ id: running.id }), reason: "state-running" }),
      expect.objectContaining({ task: expect.objectContaining({ id: missing.id }), reason: "missing-process" }),
    ]);
    await expect(reloaded.get(other.id)).resolves.toMatchObject({ id: other.id, status: "running" });
  });
});

describe("BackgroundTaskExecutionService", () => {
  it("writes launch artifacts and reads task output", async () => {
    const rootDir = await makeTempDir();
    const store = new FileBackgroundTaskStore(path.join(rootDir, "background-tasks.json"), () => ({ pid: 1111, unref() {} }));
    const task = await store.start({
      title: "Summarize build",
      command: "printf 'done'",
      cwd: rootDir,
    });
    const service = new BackgroundTaskExecutionService(rootDir, () => ({ pid: 1111, unref() {} }));

    await expect(fs.readFile(path.join(rootDir, task.execution.launchScript), "utf8")).resolves.toContain("bash -lc");
    await service.writeOutput(task, "alpha\nbeta\ngamma\n");
    await expect(service.readOutput(task, { lineLimit: 1 })).resolves.toBe("gamma");
  });

  // Regression: the launch script's initial "running" state was written with a
  // brittle `printf … | sed` pipeline that mangled any command containing
  // quotes or newlines into invalid JSON, crashing every later readState. This
  // launches a real script whose command is full of shell-hostile characters
  // and asserts the persisted state is valid, faithfully round-tripped JSON.
  it("persists valid state JSON for commands containing quotes and newlines", async () => {
    const rootDir = await makeTempDir();
    // A no-op spawn keeps `start()` from launching a real detached process; we
    // then execute the generated launch script synchronously so the assertion
    // is deterministic. Real bash + python3 (hard deps of the subsystem) back
    // the script's initial + completion state writers.
    const store = new FileBackgroundTaskStore(
      path.join(rootDir, "background-tasks.json"),
      () => ({ pid: 4242, unref() {} }),
      () => false,
    );
    const service = new BackgroundTaskExecutionService(rootDir);
    const nastyCommand = `echo "double\\"quote"; echo 'single quote'; printf 'line-1\nline-2\n'`;
    const task = await store.start({
      title: "Quote torture",
      command: nastyCommand,
      cwd: rootDir,
      kind: "task",
    });

    // Run the real run.sh. Its initial "running" write used to emit invalid
    // JSON for quote/newline commands, so readState would throw (the regression
    // this guards). We only care that a valid, faithful state is persisted — the
    // command's own exit status is irrelevant here and intentionally ignored.
    try {
      execFileSync("bash", [path.join(rootDir, task.execution.launchScript)], { cwd: rootDir, stdio: "ignore" });
    } catch {
      // The launched command may exit non-zero in a minimal shell; the state
      // writers still run and must leave valid JSON behind.
    }

    // readState JSON.parses the file; a malformed payload throws here.
    const state = await service.readState(task);
    expect(state).toBeDefined();
    // A terminal state was recorded (completed on success, failed otherwise).
    expect(["completed", "failed"]).toContain(state?.status);
    // The command survives the round-trip through JSON exactly, quotes and all.
    expect(state?.command).toBe(nastyCommand);
    expect(typeof state?.pid).toBe("number");
  });
});
