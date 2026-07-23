import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it } from "vitest";

import { FileBackgroundTaskStore } from "./background-tasks.js";
import { noopBackgroundSpawn, synchronousBackgroundSpawn } from "./background-tasks.testkit.js";

const tempDirs: string[] = [];

async function makeTempDir(): Promise<string> {
  const dir = await fs.mkdtemp(path.join(os.tmpdir(), "bee-testkit-"));
  tempDirs.push(dir);
  return dir;
}

afterEach(async () => {
  await Promise.all(tempDirs.splice(0).map((dir) => fs.rm(dir, { recursive: true, force: true })));
});

describe("background-tasks testkit spawners", () => {
  it("noopBackgroundSpawn records a running task but writes no state file", async () => {
    const rootDir = await makeTempDir();
    const store = new FileBackgroundTaskStore(path.join(rootDir, "background-tasks.json"), noopBackgroundSpawn, () => false);

    const task = await store.start({ title: "noop", command: "printf ok", kind: "task" });
    expect(task.status).toBe("running");
    expect(task.execution.processId).toBeGreaterThan(0);

    // The simulated process launches nothing, so no state file exists yet and
    // reading it degrades gracefully to `undefined` rather than throwing.
    const state = await store.executionService.readState(task);
    expect(state).toBeUndefined();
  });

  it("synchronousBackgroundSpawn settles a fast command before returning", async () => {
    const rootDir = await makeTempDir();
    const store = new FileBackgroundTaskStore(path.join(rootDir, "background-tasks.json"), synchronousBackgroundSpawn());

    const task = await store.start({ title: "fast", command: "printf hello-world", kind: "task" });

    // Because the launch script ran synchronously, its terminal state and
    // captured output are already on disk with no polling.
    const state = await store.executionService.readState(task);
    expect(state?.status).toBe("completed");
    expect(state?.exitCode).toBe(0);

    const output = await store.executionService.readOutput(task);
    expect(output).toContain("hello-world");
  });
});
