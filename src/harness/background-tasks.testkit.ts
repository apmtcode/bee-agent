import { spawnSync } from "node:child_process";

import type { SpawnBackgroundProcess } from "./background-tasks.js";

/**
 * Deterministic {@link SpawnBackgroundProcess} implementations for tests.
 *
 * The production launch path spawns a real detached OS subprocess (a bash
 * launch script). That is inherently non-deterministic under a heavily
 * parallel test run: the subprocess may not have written its state file yet
 * when an assertion reads it, may lose a spawn to resource contention, or may
 * tear a state write against a concurrent reader. Injecting one of these
 * spawners via the `backgroundTaskSpawnProcess` seam removes that
 * non-determinism while still exercising the real store/reconcile logic.
 */

let noopPidCounter = 900_000;

/**
 * A spawner that launches nothing. The background-task store still records the
 * task as `running` with the returned pid, but no state file is ever written by
 * a subprocess — so tests retain full control over the on-disk state via
 * `executionService.writeState`, and no torn/partial writes can occur.
 *
 * Use this when a test manually simulates process outcomes (the common case)
 * or expects a "still launching, no state yet" task (e.g. a long-running
 * command whose state file should be absent).
 */
export const noopBackgroundSpawn: SpawnBackgroundProcess = () => {
  noopPidCounter += 1;
  const pid = noopPidCounter;
  return {
    pid,
    unref() {
      /* nothing to unref for a simulated process */
    },
  };
};

/**
 * A spawner that runs the launch script **synchronously** (blocking until it
 * exits) via {@link spawnSync}, so by the time the store's `launch()` returns
 * the task has fully settled: output captured and a terminal state written.
 *
 * Only safe for fast-completing commands. A `timeoutMs` bound (default 4s)
 * guards against an accidental long-running command wedging the test; on
 * timeout the child is killed and control returns.
 */
export function synchronousBackgroundSpawn(options: { timeoutMs?: number } = {}): SpawnBackgroundProcess {
  const timeout = options.timeoutMs ?? 4_000;
  return (command, args, spawnOptions) => {
    const result = spawnSync(command, args, {
      cwd: spawnOptions.cwd,
      env: spawnOptions.env,
      stdio: "ignore",
      timeout,
      killSignal: "SIGKILL",
    });
    noopPidCounter += 1;
    const pid = typeof result.pid === "number" && result.pid > 0 ? result.pid : noopPidCounter;
    return {
      pid,
      unref() {
        /* the synchronous child has already exited */
      },
    };
  };
}
