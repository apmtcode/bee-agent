import { describe, expect, it } from "vitest";
import { createSimulatedBackgroundSpawn } from "./simulated-spawn.js";

const launchOptions = { cwd: "/tmp", env: {}, stdio: "ignore", detached: true } as const;

describe("createSimulatedBackgroundSpawn", () => {
  it("hands out deterministic, monotonically increasing pids without real work", () => {
    const spawn = createSimulatedBackgroundSpawn({ startPid: 5000 });
    const first = spawn("run.sh", [], launchOptions);
    const second = spawn("run.sh", [], launchOptions);
    expect(first.pid).toBe(5000);
    expect(second.pid).toBe(5001);
    expect(() => first.unref()).not.toThrow();
  });

  it("defaults to a high start pid that will not collide with real OS pids", () => {
    const spawn = createSimulatedBackgroundSpawn();
    expect(spawn("run.sh", [], launchOptions).pid).toBeGreaterThanOrEqual(1_000_000);
  });

  it("reports each simulated launch to the onLaunch hook", () => {
    const launches: Array<{ command: string; args: string[] }> = [];
    const spawn = createSimulatedBackgroundSpawn({
      onLaunch: (command, args) => launches.push({ command, args }),
    });
    spawn("/tmp/task/run.sh", ["--flag"], launchOptions);
    expect(launches).toEqual([{ command: "/tmp/task/run.sh", args: ["--flag"] }]);
  });
});
