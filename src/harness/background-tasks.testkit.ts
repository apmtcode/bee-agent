import type { SpawnBackgroundProcess } from "./background-tasks.js";

/**
 * A deterministic {@link SpawnBackgroundProcess} stand-in for tests.
 *
 * The real launcher spawns a detached bash process that asynchronously
 * rewrites the task's state file. In tests that assert on synthetic state
 * (or on process liveness via an injected `isProcessRunning`), that real
 * process races the assertions and clobbers the state — making the suite
 * flaky and dependent on host process scheduling. This stand-in performs no
 * work: it hands back a unique fake pid and a no-op `unref()`, so the task
 * record's declared status and any `writeState`/`isProcessRunning` overrides
 * remain authoritative.
 */
export function mockSpawnBackgroundProcess(startPid = 1_000_000): SpawnBackgroundProcess {
  let next = startPid;
  return () => {
    const pid = next;
    next += 1;
    return { pid, unref() {} };
  };
}
