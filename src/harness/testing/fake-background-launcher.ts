import fs from "node:fs";
import path from "node:path";

import type { IsProcessRunning, SpawnBackgroundProcess } from "../background-tasks.js";

export interface FakeBackgroundLauncherOptions {
  /** First synthetic pid to hand out; each launch increments from here. */
  basePid?: number;
  /**
   * Canned stdout appended to each launched task's `output.log`. Omit when a
   * test writes output itself (or never reads it).
   */
  output?: string;
}

export interface FakeBackgroundLauncher {
  /** Drop-in for `backgroundTaskSpawnProcess` — never touches the OS. */
  spawn: SpawnBackgroundProcess;
  /**
   * Drop-in for `backgroundTaskIsProcessRunning`. Reports `true` only for pids
   * this launcher issued, so a task started here stays "alive" until a test
   * writes a terminal state (or a foreign pid) for it.
   */
  isProcessRunning: IsProcessRunning;
  /** The set of pids currently considered alive (issued and not reaped). */
  readonly livePids: ReadonlySet<number>;
}

/**
 * Deterministic stand-in for the production detached-bash launcher used by the
 * background-task subsystem.
 *
 * In production, launching a task spawns an OS process that writes `state.json`
 * and `output.log` asynchronously. Under the cloud/CI test runner that write
 * races with the test's own assertions — surfacing as corrupt JSON reads and
 * premature `missing-process` transitions that flake depending on machine
 * speed. This launcher removes the race entirely: `spawn` returns a synthetic
 * pid (optionally seeding canned output) without starting anything, and
 * `isProcessRunning` is pid-aware so lifecycle state is driven only by what the
 * test explicitly writes.
 */
export function createFakeBackgroundLauncher(
  options: FakeBackgroundLauncherOptions = {},
): FakeBackgroundLauncher {
  const livePids = new Set<number>();
  let nextPid = options.basePid ?? 800000;

  const spawn: SpawnBackgroundProcess = (command) => {
    nextPid += 1;
    const pid = nextPid;
    livePids.add(pid);
    if (options.output !== undefined) {
      try {
        const outputPath = path.join(path.dirname(String(command)), "output.log");
        fs.appendFileSync(outputPath, options.output);
      } catch {
        // Best-effort: tests that never read output don't depend on this seed.
      }
    }
    return { pid, unref() {} };
  };

  return {
    spawn,
    isProcessRunning: (pid: number) => livePids.has(pid),
    livePids,
  };
}
