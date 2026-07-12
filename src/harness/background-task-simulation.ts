import { execFileSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import type { SpawnBackgroundProcess } from "./background-tasks.js";

/**
 * Deterministic background-task spawn stubs for tests (and dry-run scenarios).
 *
 * The real spawn machinery launches a detached shell that writes state/output
 * files asynchronously. In a test (or cloud CI) that makes background-task
 * behaviour depend on real process scheduling: whether the shell has flushed
 * its state file, whether the process is still alive, etc. These stubs remove
 * that nondeterminism so tests exercise the recovery/control logic instead of
 * the OS scheduler.
 */

const DEFAULT_SIMULATED_PID = 999_999;

/**
 * A spawn that launches nothing: it records a stable pid but never writes a
 * state or output file. Tasks stay in whatever status the caller sets, and the
 * caller (test) is free to seed state/output deterministically via the
 * execution service. Pair with `backgroundTaskIsProcessRunning` to control how
 * recovery treats the task.
 */
export function createIdleBackgroundSpawn(pid: number = DEFAULT_SIMULATED_PID): SpawnBackgroundProcess {
  return () => ({ pid, unref() {} });
}

/**
 * A spawn that runs the generated launch script synchronously — so the real
 * command output is captured deterministically — then removes the terminal
 * state file it wrote. With `backgroundTaskIsProcessRunning: () => true`, the
 * task therefore looks like a live process that has already produced output,
 * which is exactly what the CLI watch/monitor flows assert against without any
 * dependence on process timing. Only use with fast, self-terminating commands.
 */
export function createReplayBackgroundSpawn(pid: number = DEFAULT_SIMULATED_PID): SpawnBackgroundProcess {
  return (scriptPath, _args, options) => {
    try {
      execFileSync("bash", [scriptPath], {
        cwd: options.cwd,
        env: options.env,
        stdio: "ignore",
      });
    } catch {
      // Non-zero exit is fine; the script still wrote output + a state file.
    }
    const stateFile = path.join(path.dirname(scriptPath), "state.json");
    try {
      fs.rmSync(stateFile, { force: true });
    } catch {
      // Best effort; absence of the state file is the desired end state.
    }
    return { pid, unref() {} };
  };
}
