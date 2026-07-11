import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import {
  createSimulatedBackgroundSpawn,
  simulatedProcessLiveness,
} from "./simulated-background-process.js";

const tempDirs: string[] = [];

async function makeTempDir(): Promise<string> {
  const dir = await fs.mkdtemp(path.join(os.tmpdir(), "sim-bg-"));
  tempDirs.push(dir);
  return dir;
}

afterEach(async () => {
  await Promise.all(tempDirs.splice(0).map((dir) => fs.rm(dir, { recursive: true, force: true })));
});

describe("createSimulatedBackgroundSpawn", () => {
  it("returns the configured pid and never writes files without output", async () => {
    const dir = await makeTempDir();
    const launchScript = path.join(dir, "run.sh");
    await fs.writeFile(launchScript, "#!/usr/bin/env bash\n");

    const spawn = createSimulatedBackgroundSpawn({ pid: 7 });
    const child = spawn(launchScript, [], { cwd: dir, env: {}, stdio: "ignore", detached: true });

    expect(child.pid).toBe(7);
    expect(() => child.unref()).not.toThrow();
    await expect(fs.readFile(path.join(dir, "output.log"), "utf8")).rejects.toMatchObject({ code: "ENOENT" });
    await expect(fs.readFile(path.join(dir, "state.json"), "utf8")).rejects.toMatchObject({ code: "ENOENT" });
  });

  it("seeds the sibling output.log when output is provided", async () => {
    const dir = await makeTempDir();
    const launchScript = path.join(dir, "run.sh");
    await fs.writeFile(launchScript, "#!/usr/bin/env bash\n");

    const spawn = createSimulatedBackgroundSpawn({ pid: 99, output: "hello\nworld\n" });
    const child = spawn(launchScript, [], { cwd: dir, env: {}, stdio: "ignore", detached: true });

    expect(child.pid).toBe(99);
    await expect(fs.readFile(path.join(dir, "output.log"), "utf8")).resolves.toBe("hello\nworld\n");
  });

  it("defaults the pid to 4242", () => {
    const spawn = createSimulatedBackgroundSpawn();
    const child = spawn("/tmp/does-not-need-to-exist/run.sh", [], {
      cwd: "/tmp",
      env: {},
      stdio: "ignore",
      detached: true,
    });
    expect(child.pid).toBe(4242);
  });
});

describe("simulatedProcessLiveness", () => {
  it("reports only the enumerated pids as alive", () => {
    const isRunning = simulatedProcessLiveness(10, 20);
    expect(isRunning(10)).toBe(true);
    expect(isRunning(20)).toBe(true);
    expect(isRunning(30)).toBe(false);
  });

  it("treats an empty set as nothing alive", () => {
    const isRunning = simulatedProcessLiveness();
    expect(isRunning(1)).toBe(false);
  });
});
