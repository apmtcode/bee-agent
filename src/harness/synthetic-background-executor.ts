import type {
  BackgroundExecutionContext,
  BackgroundExecutionDriver,
  BackgroundTaskExecutionState,
  BackgroundTaskRecord,
  SpawnBackgroundProcess,
} from "./background-tasks.js";

/**
 * Maps a shell command to the stdout a real run *would* produce, WITHOUT
 * executing a shell. This keeps simulation deterministic and — critically —
 * safe: a real `spawnSync` on something like `tail -f app.log` would block
 * forever, whereas here it simply yields no simulated output.
 *
 * Only the trivial, side-effect-free `printf`/`echo` forms are interpreted
 * (enough to exercise the background-task pipeline in tests). Anything else
 * returns an empty string.
 */
export type CommandOutputSimulator = (command: string) => string;

const SIMPLE_ESCAPES: Record<string, string> = {
  n: "\n",
  t: "\t",
  r: "\r",
  "\\": "\\",
};

function stripOuterQuotes(value: string): string {
  if (value.length >= 2) {
    const first = value[0];
    const last = value[value.length - 1];
    if ((first === "'" && last === "'") || (first === '"' && last === '"')) {
      return value.slice(1, -1);
    }
  }
  return value;
}

function unescape(value: string): string {
  let out = "";
  for (let i = 0; i < value.length; i += 1) {
    const ch = value[i];
    if (ch === "\\" && i + 1 < value.length) {
      const next = value[i + 1] ?? "";
      if (next in SIMPLE_ESCAPES) {
        out += SIMPLE_ESCAPES[next];
        i += 1;
        continue;
      }
    }
    out += ch;
  }
  return out;
}

/**
 * Deterministically simulate `printf <arg>` / `echo <arg>` output. `printf`
 * emits the argument verbatim (with `\n`/`\t` expanded); `echo` appends a
 * trailing newline. Unrecognised commands produce no output.
 */
export function simulateTrivialCommandOutput(command: string): string {
  const trimmed = command.trim();
  const printfMatch = /^printf\s+([\s\S]+)$/.exec(trimmed);
  if (printfMatch?.[1]) {
    return unescape(stripOuterQuotes(printfMatch[1].trim()));
  }
  const echoMatch = /^echo\s+([\s\S]+)$/.exec(trimmed);
  if (echoMatch?.[1]) {
    return `${unescape(stripOuterQuotes(echoMatch[1].trim()))}\n`;
  }
  return "";
}

export type SyntheticBackgroundExecutorOptions = {
  /** First pid the executor hands out; subsequent launches increment it. */
  startPid?: number;
  /** Deterministic timestamp stamped into the simulated `running` state. */
  startedAt?: string;
  /** Override how a command is turned into simulated stdout. */
  simulateCommand?: CommandOutputSimulator;
};

/**
 * In-process {@link BackgroundExecutionDriver} that stands in for a detached OS
 * process. On launch it records a `running` execution state and the output the
 * command would print, then reports the (fake) pid as alive until explicitly
 * completed/failed. Because nothing is spawned and no command is executed,
 * behaviour is fully deterministic — no timing races, no dependency on the
 * host shell — which is exactly what the cloud self-evolution environment
 * needs to validate the background-task and movement-replay pipelines.
 */
export class SyntheticBackgroundExecutor implements BackgroundExecutionDriver {
  private nextPid: number;
  private readonly startedAt: string;
  private readonly simulateCommand: CommandOutputSimulator;
  private readonly livePids = new Set<number>();
  private readonly tasksByPid = new Map<number, BackgroundTaskRecord>();

  constructor(options: SyntheticBackgroundExecutorOptions = {}) {
    this.nextPid = options.startPid ?? 900000;
    this.startedAt = options.startedAt ?? "2026-01-01T00:00:00.000Z";
    this.simulateCommand = options.simulateCommand ?? simulateTrivialCommandOutput;
  }

  async launch(task: BackgroundTaskRecord, context: BackgroundExecutionContext): Promise<{ pid: number }> {
    const pid = this.nextPid;
    this.nextPid += 1;
    this.livePids.add(pid);
    this.tasksByPid.set(pid, task);

    await context.appendOutput(`starting ${task.kind} ${task.id}\n`);
    const simulated = this.simulateCommand(task.command);
    if (simulated) {
      await context.appendOutput(simulated.endsWith("\n") ? simulated : `${simulated}\n`);
    }

    await context.writeState(this.runningState(task, pid));
    return { pid };
  }

  isProcessRunning(pid: number): boolean {
    return this.livePids.has(pid);
  }

  /** Simulate the process exiting; after this `isProcessRunning` returns false. */
  complete(pid: number): void {
    this.livePids.delete(pid);
  }

  /** Currently-alive simulated pids (useful for driving completion in tests). */
  livePidList(): number[] {
    return [...this.livePids];
  }

  private runningState(task: BackgroundTaskRecord, pid: number): BackgroundTaskExecutionState {
    return {
      version: 1,
      taskId: task.id,
      kind: task.kind,
      status: "running",
      pid,
      startedAt: this.startedAt,
      updatedAt: this.startedAt,
      outputFile: task.execution.outputFile,
      cwd: task.cwd,
      command: task.command,
    };
  }
}

/**
 * A {@link SpawnBackgroundProcess} that never spawns anything and writes no
 * files. Use it when a test manages execution state manually (via
 * `executionService.writeState`) and just needs the real detached process out
 * of the way so it cannot race and clobber those writes.
 */
export function createNoopBackgroundSpawn(startPid = 800000): SpawnBackgroundProcess {
  let pid = startPid;
  return () => {
    const current = pid;
    pid += 1;
    return {
      pid: current,
      unref() {
        /* no-op: nothing to detach */
      },
    };
  };
}
