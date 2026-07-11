import fs from "node:fs";
import path from "node:path";
import type { IsProcessRunning, SpawnBackgroundProcess } from "./background-tasks.js";

/**
 * Options for {@link createSimulatedBackgroundSpawn}.
 */
export interface SimulatedBackgroundSpawnOptions {
  /**
   * The PID reported for the simulated process. Pair this with a matching
   * {@link IsProcessRunning} (e.g. `() => true`) when the task should be treated
   * as alive. Defaults to `4242`.
   */
  pid?: number;
  /**
   * When provided, this content is written to the task's `output.log` so
   * `readOutput()` returns deterministic text without executing a real command.
   * The output file lives next to the launch script (`<dir>/output.log`), which
   * is the layout `createBackgroundTaskRecord` produces.
   */
  output?: string;
}

/**
 * Builds a {@link SpawnBackgroundProcess} that never touches the real OS.
 *
 * Real background-task launches shell out to a detached process that writes its
 * execution state and output asynchronously (and relies on process-group
 * signalling for liveness). None of that is deterministic — or even reliable —
 * inside a sandboxed/cloud runner, which makes any test that starts a real task
 * flaky. This simulated spawn returns a fixed PID immediately and, optionally,
 * seeds the task's output file, so callers get a repeatable "running" task
 * whose lifecycle they control via the injected {@link IsProcessRunning}.
 *
 * It intentionally does NOT write a state file: leaving `state.json` absent lets
 * the store's `reconcileMissingState` path decide liveness purely from the
 * injected `IsProcessRunning`, and lets tests that need a specific execution
 * state write it explicitly.
 */
export function createSimulatedBackgroundSpawn(
  options: SimulatedBackgroundSpawnOptions = {},
): SpawnBackgroundProcess {
  const pid = options.pid ?? 4242;
  return (launchScriptPath: string) => {
    if (typeof options.output === "string") {
      const dir = path.dirname(launchScriptPath);
      fs.writeFileSync(path.join(dir, "output.log"), options.output);
    }
    return { pid, unref() {} };
  };
}

/** An {@link IsProcessRunning} that reports the given PIDs as alive. */
export function simulatedProcessLiveness(...alivePids: number[]): IsProcessRunning {
  const alive = new Set(alivePids);
  return (pid: number) => alive.has(pid);
}
