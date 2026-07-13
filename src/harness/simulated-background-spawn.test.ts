import { describe, expect, it } from "vitest";
import { createSimulatedBackgroundSpawn, type SimulatedBackgroundLaunch } from "./simulated-background-spawn.js";

const spawnOptions = {
  cwd: "/tmp/work",
  env: {} as NodeJS.ProcessEnv,
  stdio: "ignore" as const,
  detached: true as const,
};

describe("createSimulatedBackgroundSpawn", () => {
  it("hands back deterministic, monotonically increasing fake pids", () => {
    const spawn = createSimulatedBackgroundSpawn({ basePid: 500 });
    const first = spawn("/launch-a.sh", [], spawnOptions);
    const second = spawn("/launch-b.sh", [], spawnOptions);
    expect(first.pid).toBe(500);
    expect(second.pid).toBe(501);
  });

  it("defaults to a high base pid and exposes a no-op unref", () => {
    const spawn = createSimulatedBackgroundSpawn();
    const child = spawn("/launch.sh", [], spawnOptions);
    expect(child.pid).toBe(100_000);
    // unref must be callable and must not throw (nothing to detach).
    expect(() => child.unref()).not.toThrow();
  });

  it("reports every simulated launch to onLaunch in order without forking a process", () => {
    const launches: SimulatedBackgroundLaunch[] = [];
    const spawn = createSimulatedBackgroundSpawn({ basePid: 42, onLaunch: (launch) => launches.push(launch) });
    spawn("/launch-1.sh", ["--flag"], spawnOptions);
    spawn("/launch-2.sh", [], { ...spawnOptions, cwd: "/other" });
    expect(launches).toEqual([
      { command: "/launch-1.sh", args: ["--flag"], cwd: "/tmp/work", pid: 42 },
      { command: "/launch-2.sh", args: [], cwd: "/other", pid: 43 },
    ]);
  });
});
