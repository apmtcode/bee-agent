import { describe, expect, it } from "vitest";
import { createInertSpawn } from "./inert-spawn.js";

const SPAWN_OPTS = {
  cwd: "/tmp",
  env: {},
  stdio: "ignore" as const,
  detached: true as const,
};

describe("createInertSpawn", () => {
  it("returns a handle with a numeric pid and a no-op unref, launching no process", () => {
    const spawn = createInertSpawn();
    const handle = spawn("/bin/true", [], SPAWN_OPTS);
    expect(typeof handle.pid).toBe("number");
    expect(handle.pid).toBeGreaterThan(0);
    // unref must exist and be callable without throwing.
    expect(() => handle.unref()).not.toThrow();
  });

  it("allocates a fresh, monotonically increasing pid per call so tasks never collide", () => {
    const spawn = createInertSpawn(1000);
    const first = spawn("/bin/true", [], SPAWN_OPTS).pid;
    const second = spawn("/bin/true", [], SPAWN_OPTS).pid;
    const third = spawn("/bin/true", [], SPAWN_OPTS).pid;
    expect(first).toBe(1000);
    expect(second).toBe(1001);
    expect(third).toBe(1002);
  });

  it("gives independent counters per factory instance", () => {
    const a = createInertSpawn(500);
    const b = createInertSpawn(500);
    expect(a("/bin/true", [], SPAWN_OPTS).pid).toBe(500);
    expect(b("/bin/true", [], SPAWN_OPTS).pid).toBe(500);
    expect(a("/bin/true", [], SPAWN_OPTS).pid).toBe(501);
  });
});
