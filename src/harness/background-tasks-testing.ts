import { spawnSync } from "node:child_process";
import type { SpawnBackgroundProcess } from "./background-tasks.js";

// Deterministic spawn implementations for tests and other environments that
// must not depend on the timing of real detached OS processes.
//
// The production default (`spawn(..., { detached: true })`) is fire-and-forget:
// `launch()` returns immediately while the launch script asynchronously writes
// the task's running/completed state and command output. Tests that read that
// state/output right after starting a task therefore race the child process,
// which surfaces as flaky "state still running" (→ remote status "degraded")
// or "output not yet written" assertions. These helpers remove the race.

let fakePidCounter = 900000;

function nextFakePid(): number {
  fakePidCounter += 1;
  return fakePidCounter;
}

/**
 * Runs the launch script to completion **synchronously**, so the task's state
 * file and output are fully written before `launch()` returns. Use for tests
 * that assert on real command output/state. Only safe for commands that
 * terminate on their own (a non-terminating command such as `tail -f` would
 * block forever — use {@link createNoopSpawnBackgroundProcess} there).
 */
export function createSynchronousSpawnBackgroundProcess(): SpawnBackgroundProcess {
  return (command, args, options) => {
    spawnSync(command, args, {
      cwd: options.cwd,
      env: options.env,
      stdio: "ignore",
    });
    return { pid: nextFakePid(), unref() {} };
  };
}

/**
 * Returns a fake pid without launching anything. Use for tests that drive task
 * state manually (e.g. recovery/lifecycle tests) or that start non-terminating
 * commands whose real execution is irrelevant to the assertion.
 */
export function createNoopSpawnBackgroundProcess(): SpawnBackgroundProcess {
  return () => ({ pid: nextFakePid(), unref() {} });
}
