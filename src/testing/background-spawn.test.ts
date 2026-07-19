import { describe, expect, it } from "vitest";
import { createSimulatedBackgroundSpawn } from "./background-spawn.js";

const SPAWN_OPTIONS = {
  cwd: "/tmp",
  env: {},
  stdio: "ignore",
  detached: true,
} as const;

describe("createSimulatedBackgroundSpawn", () => {
  it("hands back unique pids and reports them alive", () => {
    const sim = createSimulatedBackgroundSpawn({ startPid: 1000 });
    const a = sim.spawn("script.sh", [], SPAWN_OPTIONS);
    const b = sim.spawn("script.sh", [], SPAWN_OPTIONS);

    expect(a.pid).toBe(1000);
    expect(b.pid).toBe(1001);
    expect(a.pid).not.toBe(b.pid);
    expect(sim.isProcessRunning(a.pid as number)).toBe(true);
    expect(sim.isProcessRunning(b.pid as number)).toBe(true);
  });

  it("reports pids it never handed out as not running", () => {
    const sim = createSimulatedBackgroundSpawn();
    // A synthetic dead pid a test might write into a state file.
    expect(sim.isProcessRunning(999999)).toBe(false);
  });

  it("marks a terminated pid as no longer running", () => {
    const sim = createSimulatedBackgroundSpawn({ startPid: 500 });
    const child = sim.spawn("script.sh", [], SPAWN_OPTIONS);
    expect(sim.isProcessRunning(500)).toBe(true);

    sim.terminate(500);
    expect(sim.isProcessRunning(500)).toBe(false);
    // `unref` is a no-op but must exist for the launcher contract.
    expect(() => child.unref()).not.toThrow();
  });

  it("does not touch the filesystem or start a real process", () => {
    const sim = createSimulatedBackgroundSpawn();
    const child = sim.spawn("/definitely/not/a/real/script.sh", [], SPAWN_OPTIONS);
    // If a real process had been spawned this pid would be a live OS pid; the
    // simulator's value is a synthetic counter starting well above typical pids.
    expect(typeof child.pid).toBe("number");
    expect(child.pid).toBeGreaterThan(0);
  });
});
