import type { IsProcessRunning, SpawnBackgroundProcess } from "./background-tasks.js";

/**
 * Deterministic spawn/liveness doubles for background-task tests.
 *
 * The production launcher spawns a *real* detached shell that asynchronously
 * writes `state.json` (running → completed/failed) while it runs the task's
 * command. In tests that is a source of nondeterminism: the launcher's writes
 * race with the test's own `writeState`/`writeOutput` calls, and the moment a
 * short command exits is timing-dependent. Torn reads of a half-written
 * `state.json` can even crash recovery.
 *
 * `createMockBackgroundSpawn` replaces that real process with a no-op: it hands
 * back a deterministic, incrementing fake pid and never launches anything, so
 * the only writer of `state.json` is the test itself. The paired
 * `isProcessRunning` reports exactly the pids this double handed out as alive,
 * which lets a test keep started tasks "running" while still marking an
 * explicitly-chosen sentinel pid (e.g. one written via `writeState`) as dead.
 */
export interface MockBackgroundSpawn {
  /** Drop-in for `backgroundTaskSpawnProcess`; never runs a real process. */
  readonly spawn: SpawnBackgroundProcess;
  /** Reports pids handed out by {@link spawn} as alive, all others as dead. */
  readonly isProcessRunning: IsProcessRunning;
  /** The set of pids this double has handed out and still considers alive. */
  readonly livePids: Set<number>;
  /** Pids handed out by {@link spawn}, in launch order. */
  readonly pids: number[];
  /** Mark a handed-out pid as no longer running (e.g. to simulate an exit). */
  readonly kill: (pid: number) => void;
  /** Mark the most recently launched task's process as no longer running. */
  readonly killLast: () => void;
}

export function createMockBackgroundSpawn(startPid = 4100): MockBackgroundSpawn {
  const livePids = new Set<number>();
  const pids: number[] = [];
  let nextPid = startPid;
  const spawn: SpawnBackgroundProcess = () => {
    const pid = nextPid;
    nextPid += 1;
    livePids.add(pid);
    pids.push(pid);
    return {
      pid,
      unref() {
        /* no-op: nothing to detach for a simulated process */
      },
    };
  };
  return {
    spawn,
    isProcessRunning: (pid) => livePids.has(pid),
    livePids,
    pids,
    kill: (pid) => {
      livePids.delete(pid);
    },
    killLast: () => {
      const last = pids.at(-1);
      if (last !== undefined) {
        livePids.delete(last);
      }
    },
  };
}
