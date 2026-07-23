import { describe, expect, it } from "vitest";
import { createInMemoryBackgroundSpawn } from "./spawn-stub.js";

const spawnOptions = {
  cwd: "/work",
  env: {} as NodeJS.ProcessEnv,
  stdio: "ignore" as const,
  detached: true as const,
};

describe("createInMemoryBackgroundSpawn", () => {
  it("hands out stable, monotonically increasing synthetic pids", () => {
    const spawn = createInMemoryBackgroundSpawn(1000);
    const first = spawn("/run.sh", [], spawnOptions);
    const second = spawn("/run.sh", [], spawnOptions);
    expect(first.pid).toBe(1000);
    expect(second.pid).toBe(1001);
  });

  it("records each launch without forking a real process", () => {
    const spawn = createInMemoryBackgroundSpawn();
    spawn("/a.sh", ["--flag"], { ...spawnOptions, cwd: "/first" });
    spawn("/b.sh", [], { ...spawnOptions, cwd: "/second" });
    expect(spawn.launches).toEqual([
      { pid: 900000, command: "/a.sh", args: ["--flag"], cwd: "/first" },
      { pid: 900001, command: "/b.sh", args: [], cwd: "/second" },
    ]);
  });

  it("returns a callable unref() that is a no-op", () => {
    const spawn = createInMemoryBackgroundSpawn();
    const child = spawn("/run.sh", [], spawnOptions);
    expect(() => child.unref()).not.toThrow();
  });

  it("copies the args array so later caller mutation does not corrupt the record", () => {
    const spawn = createInMemoryBackgroundSpawn();
    const args = ["--one"];
    spawn("/run.sh", args, spawnOptions);
    args.push("--two");
    expect(spawn.launches[0]?.args).toEqual(["--one"]);
  });
});
