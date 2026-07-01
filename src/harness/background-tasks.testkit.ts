import type { IsProcessRunning, SpawnBackgroundProcess } from "./background-tasks.js";

/**
 * Deterministic test doubles for the background-task execution seam.
 *
 * The real {@link SpawnBackgroundProcess} launches a detached OS subprocess
 * that asynchronously writes its own `state.json`/`output.log`. In tests that
 * assert on background-task state, that real process races the state files the
 * test writes by hand — the task can flip to `completed`/`failed`/`missing-process`
 * at an unpredictable moment, making assertions like `control=active` or
 * "task still running" flaky.
 *
 * These doubles let a test opt out of real process launching: the spawn returns
 * a stable fake pid and starts nothing, so no state file appears unless the test
 * writes one itself, and the liveness probe answers deterministically for that
 * pid. Production code keeps its real defaults (these are wired only via the
 * injectable constructor/option seams).
 */

/** Stable, implausible pid returned by {@link spawnFakeBackgroundProcess}. */
export const FAKE_BACKGROUND_PID = 424242;

/**
 * A spawn double that launches no real process. It returns {@link FAKE_BACKGROUND_PID}
 * and a no-op `unref`, so `startBackgroundTask` records a running task without any
 * detached subprocess writing competing state.
 */
export const spawnFakeBackgroundProcess: SpawnBackgroundProcess = () => ({
  pid: FAKE_BACKGROUND_PID,
  unref() {},
});

/**
 * Liveness probe that reports the fake background pid as running and everything
 * else as dead — so a task launched via {@link spawnFakeBackgroundProcess} stays
 * "running", while manually-written states with other pids reconcile to
 * missing-process exactly as they would in production.
 */
export const isFakeBackgroundProcessRunning: IsProcessRunning = (pid) => pid === FAKE_BACKGROUND_PID;
