import { spawnSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import type { SpawnBackgroundProcess } from "./background-tasks.js";

/**
 * Deterministic spawn seams for the background-task subsystem, used by tests.
 *
 * The production spawn detaches a real OS process that asynchronously writes the
 * task's execution-state and output files. That makes any assertion about a
 * freshly-started task inherently racy: whether the state file has landed (and
 * whether the process is still alive) depends on host scheduling. These fakes
 * remove that nondeterminism.
 */

let fakePidCounter = 100_000;

function nextFakePid(): number {
  fakePidCounter += 1;
  return fakePidCounter;
}

/**
 * A spawn that launches nothing. No execution-state or output file is written,
 * so `readState()` resolves to `undefined` and the caller's own `writeState()`
 * calls are authoritative. Ideal for tests that drive task state explicitly and
 * only care about the in-memory record produced by `markStarted`.
 */
export function createNoopBackgroundSpawn(): SpawnBackgroundProcess {
  return () => ({ pid: nextFakePid(), unref() {} });
}

/**
 * A spawn that runs the generated launch script to completion synchronously, so
 * by the time `startBackgroundTask` returns, the output file and a terminal
 * ("completed"/"failed") execution state are fully written. Deterministic for
 * fast commands (e.g. `printf ...`); avoid for long-running commands since it
 * blocks. Returns the child pid reported by `spawnSync` when present, otherwise
 * a synthetic one, so `markStarted` always records a numeric process id.
 */
export function createSynchronousBackgroundSpawn(): SpawnBackgroundProcess {
  return (command, args, options) => {
    const result = spawnSync(command, args, {
      cwd: options.cwd,
      env: options.env,
      stdio: "ignore",
    });
    return { pid: typeof result.pid === "number" ? result.pid : nextFakePid(), unref() {} };
  };
}

/**
 * A spawn for tests that need BOTH real command output AND a task that stays
 * "running" until explicitly stopped (e.g. watch/sync/stop flows). It runs the
 * launch script synchronously so the output file is populated, then rewrites the
 * execution-state file (`<baseDir>/state.json`, colocated with the launch
 * script) back to a "running" status. Pair with `backgroundTaskIsProcessRunning:
 * () => true` so reconciliation keeps the task alive. Use only for fast commands.
 */
export function createRunningBackgroundSpawn(): SpawnBackgroundProcess {
  return (command, args, options) => {
    const result = spawnSync(command, args, {
      cwd: options.cwd,
      env: options.env,
      stdio: "ignore",
    });
    const pid = typeof result.pid === "number" ? result.pid : nextFakePid();
    const stateFile = path.join(path.dirname(command), "state.json");
    try {
      const state = JSON.parse(fs.readFileSync(stateFile, "utf8")) as Record<string, unknown>;
      state.status = "running";
      state.pid = pid;
      delete state.completedAt;
      delete state.exitCode;
      delete state.signal;
      delete state.error;
      fs.writeFileSync(stateFile, `${JSON.stringify(state, null, 2)}\n`, { encoding: "utf8" });
    } catch {
      // If the script never wrote a state file (e.g. a blocked command), leave
      // things as-is; the caller's assertions will surface any real problem.
    }
    return { pid, unref() {} };
  };
}
