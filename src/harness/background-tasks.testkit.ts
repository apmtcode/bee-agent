import type { SpawnBackgroundProcess } from "./background-tasks.js";

/**
 * Deterministic, in-memory stand-in for the real `spawn` used when a background
 * task is launched.
 *
 * The production launcher spawns a detached shell that runs the task command and
 * writes `state.json` asynchronously (see `renderLaunchScript`). Tests that drive
 * execution state explicitly — via `executionService.writeState` /
 * `writeOutput` — must NOT also launch that real process: it races the test's own
 * writes (a half-written `state.json` fails `JSON.parse`) and leaves real
 * `sleep`/`tail -f` children running for the lifetime of the suite.
 *
 * Inject the result as `backgroundTaskSpawnProcess` on `StandaloneOperatorRuntime`
 * / `OperatorCliApp` / `FileBackgroundTaskStore` so `launch()` returns a stable
 * fake pid and touches nothing on disk or in the process table.
 *
 * The returned pid is intentionally stable and > 0 so `markStarted` records a
 * "running" task exactly as it would in production; pair with
 * `backgroundTaskIsProcessRunning: () => false` to control reconciliation.
 */
export function createNoopBackgroundSpawn(pid = 424242): SpawnBackgroundProcess {
  return () => ({ pid, unref() {} });
}
