import { describe, expect, it } from "vitest";
import { createSimulatedBackgroundProcess } from "./simulated-process.js";

const spawnOptions = {
  cwd: "/tmp",
  env: {},
  stdio: "ignore" as const,
  detached: true as const,
};

function requirePid(handle: { pid?: number }): number {
  if (typeof handle.pid !== "number") {
    throw new Error("expected simulated spawn to return a numeric pid");
  }
  return handle.pid;
}

describe("createSimulatedBackgroundProcess", () => {
  it("returns stable, unique fake PIDs and records launches", () => {
    const backend = createSimulatedBackgroundProcess();
    const first = backend.spawn("/tmp/launch-1.sh", [], spawnOptions);
    const second = backend.spawn("/tmp/launch-2.sh", [], spawnOptions);

    expect(typeof first.pid).toBe("number");
    expect(first.pid).not.toBe(second.pid);
    expect(() => first.unref()).not.toThrow();

    expect(backend.launches).toEqual([
      { command: "/tmp/launch-1.sh", pid: first.pid },
      { command: "/tmp/launch-2.sh", pid: second.pid },
    ]);
  });

  it("uses fake PIDs above any real Linux pid_max so stray kills yield ESRCH", () => {
    const backend = createSimulatedBackgroundProcess();
    const pid = requirePid(backend.spawn("/tmp/launch.sh", [], spawnOptions));
    // Default Linux pid_max is 4194304; a real kill(2) on this PID must miss.
    expect(pid).toBeGreaterThan(4_194_304);
  });

  it("reports spawned PIDs as running when alive (the default)", () => {
    const backend = createSimulatedBackgroundProcess();
    const pid = requirePid(backend.spawn("/tmp/launch.sh", [], spawnOptions));
    expect(backend.isProcessRunning(pid)).toBe(true);
    expect(backend.isProcessRunning(pid + 1)).toBe(false);
  });

  it("reports spawned PIDs as not running when alive is false", () => {
    const backend = createSimulatedBackgroundProcess({ alive: false });
    const pid = requirePid(backend.spawn("/tmp/launch.sh", [], spawnOptions));
    expect(backend.isProcessRunning(pid)).toBe(false);
  });

  it("honours a custom base PID", () => {
    const backend = createSimulatedBackgroundProcess({ basePid: 5_000_000 });
    const pid = requirePid(backend.spawn("/tmp/launch.sh", [], spawnOptions));
    expect(pid).toBe(5_000_001);
  });
});
