import type { IsProcessRunning, SpawnBackgroundProcess } from "./background-tasks.js";

/**
 * Deterministic background-task execution doubles for tests.
 *
 * The production launcher spawns a real, detached OS process that writes the
 * task's execution state and output asynchronously. Tests that assert on both a
 * task's *output* and its *liveness* therefore race the real subprocess, making
 * them timing-dependent and flaky across environments. These helpers let a test
 * own the state/output transitions explicitly: `fakeSpawnProcess` launches
 * nothing (so no real process ever races the test's own writes), and
 * `constantIsProcessRunning` gives the test control over the liveness probe.
 */

/** A launcher that never starts a real process; returns a stable fake pid. */
export function fakeSpawnProcess(pid = 987654): SpawnBackgroundProcess {
  return () => ({ pid, unref() {} });
}

/** A liveness probe that always reports the same answer. */
export function constantIsProcessRunning(running: boolean): IsProcessRunning {
  return () => running;
}
