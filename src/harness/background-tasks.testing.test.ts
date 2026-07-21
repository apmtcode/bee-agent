import { describe, expect, it } from "vitest";

import { createInertBackgroundSpawn } from "./background-tasks.testing.js";

describe("createInertBackgroundSpawn", () => {
  const spawnArgs = [
    "/tmp/launch.sh",
    [] as string[],
    { cwd: "/tmp", env: {}, stdio: "ignore", detached: true } as const,
  ] as const;

  it("hands out distinct, stable pids and reports them as running", () => {
    const inert = createInertBackgroundSpawn();
    const first = inert.spawn(...spawnArgs);
    const second = inert.spawn(...spawnArgs);

    expect(first.pid).toBe(100000);
    expect(second.pid).toBe(100001);
    expect(first.pid).not.toBe(second.pid);
    expect(inert.isProcessRunning(first.pid as number)).toBe(true);
    expect(inert.isProcessRunning(second.pid as number)).toBe(true);
    expect([...inert.runningPids]).toEqual([100000, 100001]);
  });

  it("treats unknown pids (e.g. a simulated dead process) as not running", () => {
    const inert = createInertBackgroundSpawn();
    inert.spawn(...spawnArgs);
    expect(inert.isProcessRunning(999999)).toBe(false);
  });

  it("never launches a real process — unref is a no-op and no throw", () => {
    const inert = createInertBackgroundSpawn(500);
    const child = inert.spawn(...spawnArgs);
    expect(child.pid).toBe(500);
    expect(() => child.unref()).not.toThrow();
  });

  it("honors a custom starting pid to keep the fake range clear", () => {
    const inert = createInertBackgroundSpawn(7000);
    expect(inert.spawn(...spawnArgs).pid).toBe(7000);
    expect(inert.isProcessRunning(7000)).toBe(true);
    expect(inert.isProcessRunning(6999)).toBe(false);
  });
});
