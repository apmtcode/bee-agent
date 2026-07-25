import { spawnSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";

import type { SpawnBackgroundProcess } from "./background-tasks.js";

/**
 * Test/simulation helpers for the background-task subsystem.
 *
 * The production launcher spawns a *detached* process that writes its state
 * file asynchronously. In the cloud/CI environment we run in, that real process
 * races with test-controlled `writeState` calls and with wall-clock-sensitive
 * diagnostics, making otherwise-correct tests flaky. These helpers let a test
 * swap in a deterministic spawn implementation via the runtime's
 * `backgroundTaskSpawnProcess` seam, so behaviour is reproducible without
 * depending on OS scheduling.
 *
 * They are intentionally shipped in `src/` (not a `*.test.ts` file) so multiple
 * test files — and any future simulated harness — can share one implementation.
 */

let simulatedPidCounter = 40_000;

function nextSimulatedPid(): number {
  simulatedPidCounter += 1;
  return simulatedPidCounter;
}

/**
 * A spawn that starts no process at all. The background-task *record* is still
 * created (status `running`), but no state file is written by a background
 * process, so the test remains the single authority over execution state. Use
 * this when a test drives state transitions itself and only needs the task to
 * exist / stay "running".
 */
export function createNoopBackgroundSpawn(): SpawnBackgroundProcess {
  return () => ({ pid: nextSimulatedPid(), unref() {} });
}

/**
 * A spawn that runs the generated launch script *synchronously* and returns
 * only once it has exited. This exercises the real launcher (state + output
 * files, atomic writes, quoting) but removes the async race: by the time
 * `start()` resolves, the command's output and terminal state are already
 * persisted. Use this when a test asserts on output the command actually
 * produced (e.g. `printf ok`).
 */
export function createSynchronousBackgroundSpawn(): SpawnBackgroundProcess {
  return (command, args, options) => {
    const result = spawnSync(command, args, {
      cwd: options.cwd,
      env: options.env,
      stdio: "ignore",
    });
    return { pid: result.pid ?? nextSimulatedPid(), unref() {} };
  };
}

/**
 * A spawn that runs the launcher synchronously — so the command's real output
 * lands on disk — and then rewrites the terminal state back to `running`. This
 * models the common intermediate reality of a task that has already logged
 * output but is still alive. Pair it with `backgroundTaskIsProcessRunning: () =>
 * true` so sync/recovery keep the task active. Use it when a test needs both a
 * command's actual output *and* the task to remain the session's active task.
 */
export function createRunningWithOutputBackgroundSpawn(): SpawnBackgroundProcess {
  return (command, args, options) => {
    const result = spawnSync(command, args, {
      cwd: options.cwd,
      env: options.env,
      stdio: "ignore",
    });
    // The launcher writes state.json as a sibling of the run.sh it was handed.
    const stateFile = path.join(path.dirname(command), "state.json");
    try {
      const state = JSON.parse(fs.readFileSync(stateFile, "utf8"));
      state.status = "running";
      delete state.completedAt;
      delete state.exitCode;
      delete state.error;
      fs.writeFileSync(stateFile, `${JSON.stringify(state, null, 2)}\n`);
    } catch {
      // Best effort: if the launcher never wrote a state file, leave it be.
    }
    return { pid: result.pid ?? nextSimulatedPid(), unref() {} };
  };
}
