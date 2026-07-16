import { describe, expect, it } from "vitest";
import { createSimulatedBackgroundExecution } from "./simulated-background.js";

const spawnOptions = {
  cwd: "/tmp",
  env: {},
  stdio: "ignore" as const,
  detached: true as const,
};

describe("createSimulatedBackgroundExecution", () => {
  it("hands out monotonic pids and never returns a real process handle", () => {
    const sim = createSimulatedBackgroundExecution({ startPid: 500 });
    const first = sim.spawnProcess("/launch.sh", [], spawnOptions);
    const second = sim.spawnProcess("/launch.sh", [], spawnOptions);
    expect(first.pid).toBe(500);
    expect(second.pid).toBe(501);
    // unref must be callable (production code calls child.unref()).
    expect(() => first.unref()).not.toThrow();
  });

  it("reports spawned pids as alive by default and honours markStopped", () => {
    const sim = createSimulatedBackgroundExecution();
    const child = sim.spawnProcess("/launch.sh", [], spawnOptions);
    expect(child.pid).toBeTypeOf("number");
    const pid = child.pid as number;
    expect(sim.isProcessRunning(pid)).toBe(true);
    sim.markStopped(pid);
    expect(sim.isProcessRunning(pid)).toBe(false);
    sim.markRunning(pid);
    expect(sim.isProcessRunning(pid)).toBe(true);
  });

  it("treats freshly spawned pids as dead when aliveOnSpawn is false", () => {
    const sim = createSimulatedBackgroundExecution({ aliveOnSpawn: false });
    const child = sim.spawnProcess("/launch.sh", [], spawnOptions);
    const pid = child.pid as number;
    expect(sim.isProcessRunning(pid)).toBe(false);
    // A never-spawned pid is likewise not running.
    expect(sim.isProcessRunning(999999)).toBe(false);
  });
});
