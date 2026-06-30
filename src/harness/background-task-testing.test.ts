import { describe, expect, it } from "vitest";
import { createInMemoryBackgroundProcesses } from "./background-task-testing.js";

describe("InMemoryBackgroundProcesses", () => {
  it("hands out live fake PIDs above the OS pid range", () => {
    const procs = createInMemoryBackgroundProcesses();
    const first = procs.spawnProcess("/launch.sh", [], {
      cwd: "/",
      env: {},
      stdio: "ignore",
      detached: true,
    });
    const second = procs.spawnProcess("/launch.sh", [], {
      cwd: "/",
      env: {},
      stdio: "ignore",
      detached: true,
    });

    expect(first.pid).toBeGreaterThan(4_194_304);
    expect(second.pid).toBe((first.pid ?? 0) + 1);
    expect(procs.isProcessRunning(first.pid ?? 0)).toBe(true);
    expect(procs.isProcessRunning(second.pid ?? 0)).toBe(true);
    // unref must be callable (background tasks call it after spawning).
    expect(() => first.unref()).not.toThrow();
  });

  it("reports unknown PIDs as not running", () => {
    const procs = createInMemoryBackgroundProcesses();
    expect(procs.isProcessRunning(1234)).toBe(false);
  });

  it("simulates a single process exiting", () => {
    const procs = createInMemoryBackgroundProcesses();
    const child = procs.spawnProcess("/launch.sh", [], { cwd: "/", env: {}, stdio: "ignore", detached: true });
    const pid = child.pid ?? 0;
    expect(procs.isProcessRunning(pid)).toBe(true);
    procs.kill(pid);
    expect(procs.isProcessRunning(pid)).toBe(false);
  });

  it("simulates every tracked process exiting and tracks issued PIDs", () => {
    const procs = createInMemoryBackgroundProcesses();
    const a = procs.spawnProcess("/a.sh", [], { cwd: "/", env: {}, stdio: "ignore", detached: true }).pid ?? 0;
    const b = procs.spawnProcess("/b.sh", [], { cwd: "/", env: {}, stdio: "ignore", detached: true }).pid ?? 0;
    expect(procs.pids()).toEqual([a, b]);
    procs.killAll();
    expect(procs.isProcessRunning(a)).toBe(false);
    expect(procs.isProcessRunning(b)).toBe(false);
  });
});
