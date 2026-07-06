import type { IsProcessRunning, SpawnBackgroundProcess } from "./background-tasks.js";

/**
 * Deterministic, in-memory stand-in for the background-task process backend.
 *
 * The real {@link FileBackgroundTaskStore} spawns a detached OS launch script
 * that asynchronously writes `state.json`/`output.log` for each task. In tests
 * (and in the cloud self-evolution sandbox) that real process races the test's
 * own state reads/writes and its liveness is environment dependent — a
 * `sleep 5` task may or may not be observable via `process.kill(-pid, 0)`, and a
 * fast `printf` task's non-atomic state write can be read mid-truncation. Both
 * make background-task suites flaky.
 *
 * This fake removes the real OS process entirely: {@link spawn} hands out a
 * synthetic pid and records it as "alive", and {@link isProcessRunning} reports
 * liveness purely from that in-memory set. Because no launch script runs, a task
 * only has a `state.json` when a test writes one explicitly, which is exactly
 * what the reconciliation/health logic under test consumes. Pids the fake never
 * handed out (e.g. a hard-coded `999999` used to force a "missing-process"
 * degradation) report as dead, so deliberate failure paths still exercise.
 */
export type FakeBackgroundProcessBackend = {
  /** Injectable `SpawnBackgroundProcess` that never launches a real process. */
  spawn: SpawnBackgroundProcess;
  /** Injectable `IsProcessRunning` backed by the set of handed-out pids. */
  isProcessRunning: IsProcessRunning;
  /** The synthetic pids currently considered alive. */
  aliveFakePids: Set<number>;
  /** Mark a previously handed-out fake pid as terminated. */
  terminate(pid: number): void;
};

export function createFakeBackgroundProcessBackend(startPid = 4_000_001): FakeBackgroundProcessBackend {
  const aliveFakePids = new Set<number>();
  let nextPid = startPid;

  const spawn: SpawnBackgroundProcess = () => {
    const pid = nextPid;
    nextPid += 1;
    aliveFakePids.add(pid);
    return {
      pid,
      unref() {
        // no-op: nothing is actually running
      },
    };
  };

  const isProcessRunning: IsProcessRunning = (pid) => aliveFakePids.has(pid);

  return {
    spawn,
    isProcessRunning,
    aliveFakePids,
    terminate(pid: number) {
      aliveFakePids.delete(pid);
    },
  };
}
