import fs from "node:fs/promises";
import path from "node:path";
import { spawn } from "node:child_process";
import { randomUUID } from "node:crypto";
import { ensureParentDir, readJsonFile, writeJsonAtomic } from "../shared/fs.js";

export type BackgroundTaskKind = "task" | "monitor";
export type BackgroundTaskStatus = "planned" | "running" | "completed" | "failed" | "cancelled";

export type BackgroundTaskExecution = {
  version: 1;
  preparedAt: string;
  startedAt?: string;
  completedAt?: string;
  failedAt?: string;
  lastUpdatedAt: string;
  workingDirectory: string;
  outputFile: string;
  stateFile: string;
  launchScript: string;
  processId?: number;
  exitCode?: number;
  signal?: string;
  error?: string;
};

export type BackgroundTaskRecord = {
  version: 1;
  id: string;
  sessionId?: string;
  title: string;
  kind: BackgroundTaskKind;
  command: string;
  cwd: string;
  createdAt: string;
  updatedAt: string;
  status: BackgroundTaskStatus;
  metadata?: Record<string, unknown>;
  execution: BackgroundTaskExecution;
};

export type BackgroundTaskStoreShape = {
  version: 1;
  tasks: BackgroundTaskRecord[];
};

export type StartBackgroundTaskParams = {
  sessionId?: string;
  title: string;
  command: string;
  cwd?: string;
  kind?: BackgroundTaskKind;
  metadata?: Record<string, unknown>;
};

export type MarkBackgroundTaskStartedParams = {
  pid: number;
};

export type FailBackgroundTaskParams = {
  reason: string;
  exitCode?: number;
  signal?: string;
  completedAt?: string;
};

export type CompleteBackgroundTaskParams = {
  exitCode?: number;
  completedAt?: string;
};

export type StopBackgroundTaskParams = {
  signal?: NodeJS.Signals;
};

export type ReadBackgroundTaskOutputOptions = {
  lineLimit?: number;
  offset?: number;
};

export type BackgroundTaskExecutionState = {
  version: 1;
  taskId: string;
  kind: BackgroundTaskKind;
  status: "running" | "completed" | "failed" | "cancelled";
  pid: number;
  startedAt: string;
  updatedAt: string;
  completedAt?: string;
  exitCode?: number;
  signal?: string;
  error?: string;
  outputFile: string;
  cwd: string;
  command: string;
};

export type SpawnBackgroundProcess = (
  command: string,
  args: string[],
  options: {
    cwd: string;
    env: NodeJS.ProcessEnv;
    stdio: "ignore";
    detached: true;
  },
) => { pid?: number; unref(): void };

export type IsProcessRunning = (pid: number) => boolean;

export type BackgroundTaskRecoveryReason =
  | "unchanged"
  | "state-running"
  | "state-completed"
  | "state-failed"
  | "state-cancelled"
  | "process-running"
  | "missing-process"
  | "missing-state";

export type BackgroundTaskRecoveryResult = {
  task: BackgroundTaskRecord;
  changed: boolean;
  reason: BackgroundTaskRecoveryReason;
};

const EMPTY_STORE: BackgroundTaskStoreShape = {
  version: 1,
  tasks: [],
};

function nowIso(): string {
  return new Date().toISOString();
}

function defaultIsProcessRunning(pid: number): boolean {
  if (!Number.isFinite(pid) || pid <= 0) {
    return false;
  }
  try {
    process.kill(process.platform === "win32" ? pid : -pid, 0);
    return true;
  } catch (error) {
    const code = (error as NodeJS.ErrnoException).code;
    if (code === "ESRCH") {
      return false;
    }
    if (code === "EPERM") {
      return true;
    }
    throw error;
  }
}

export class BackgroundTaskExecutionService {
  constructor(
    private readonly rootDir: string,
    private readonly spawnProcess: SpawnBackgroundProcess = (command, args, options) => spawn(command, args, options),
    private readonly isProcessRunningImpl: IsProcessRunning = defaultIsProcessRunning,
  ) {}

  async writeArtifacts(task: BackgroundTaskRecord): Promise<void> {
    const launchScriptPath = path.join(this.rootDir, task.execution.launchScript);
    await ensureParentDir(launchScriptPath);
    await fs.writeFile(launchScriptPath, renderLaunchScript(task), {
      encoding: "utf8",
      mode: 0o700,
    });
  }

  async launch(task: BackgroundTaskRecord): Promise<{ pid: number }> {
    const launchScriptPath = path.join(this.rootDir, task.execution.launchScript);
    const child = this.spawnProcess(launchScriptPath, [], {
      cwd: this.rootDir,
      env: {
        ...process.env,
        OPENCLAW_BACKGROUND_TASK_ID: task.id,
        OPENCLAW_BACKGROUND_TASK_KIND: task.kind,
      },
      stdio: "ignore",
      detached: true,
    });
    child.unref();
    if (typeof child.pid !== "number") {
      throw new Error(`Failed to launch background task ${task.id}`);
    }
    return { pid: child.pid };
  }

  isProcessRunning(pid: number): boolean {
    return this.isProcessRunningImpl(pid);
  }

  async stop(task: BackgroundTaskRecord, params: StopBackgroundTaskParams = {}): Promise<BackgroundTaskExecutionState> {
    const state = await this.readState(task);
    const pid = task.execution.processId ?? state?.pid;
    if (typeof pid !== "number") {
      throw new Error(`Background task ${task.id} is not running`);
    }
    const signal = params.signal ?? "SIGTERM";
    try {
      process.kill(process.platform === "win32" ? pid : -pid, signal);
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code !== "ESRCH") {
        throw error;
      }
    }
    const timestamp = nowIso();
    const cancelledState: BackgroundTaskExecutionState = {
      version: 1,
      taskId: task.id,
      kind: task.kind,
      status: "cancelled",
      pid,
      startedAt: state?.startedAt ?? task.execution.startedAt ?? timestamp,
      updatedAt: timestamp,
      completedAt: timestamp,
      exitCode: state?.exitCode,
      signal,
      error: `background task terminated by ${signal}`,
      outputFile: task.execution.outputFile,
      cwd: task.cwd,
      command: task.command,
    };
    await this.writeState(task, cancelledState);
    return cancelledState;
  }

  async writeState(task: BackgroundTaskRecord, state: BackgroundTaskExecutionState): Promise<void> {
    await writeJsonAtomic(path.join(this.rootDir, task.execution.stateFile), state);
  }

  async readState(task: BackgroundTaskRecord): Promise<BackgroundTaskExecutionState | undefined> {
    return await readJsonFile<BackgroundTaskExecutionState | undefined>(
      path.join(this.rootDir, task.execution.stateFile),
      undefined,
    );
  }

  async writeOutput(task: BackgroundTaskRecord, content: string): Promise<void> {
    const outputPath = path.join(this.rootDir, task.execution.outputFile);
    await ensureParentDir(outputPath);
    await fs.writeFile(outputPath, content, "utf8");
  }

  async readOutput(
    task: BackgroundTaskRecord,
    options: ReadBackgroundTaskOutputOptions = {},
  ): Promise<string | undefined> {
    try {
      const content = await fs.readFile(path.join(this.rootDir, task.execution.outputFile), "utf8");
      const lineLimit = options.lineLimit ?? 200;
      const offset = Math.max(0, options.offset ?? 0);
      const lines = content.split("\n");
      const trimmed = lines.at(-1) === "" ? lines.slice(0, -1) : lines;
      if (lineLimit <= 0) {
        return trimmed.slice(offset).join("\n");
      }
      const visible = trimmed.slice(offset);
      return visible.slice(-lineLimit).join("\n");
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code === "ENOENT") {
        return undefined;
      }
      throw error;
    }
  }
}

export class FileBackgroundTaskStore {
  readonly executionService: BackgroundTaskExecutionService;
  private readonly rootDir: string;

  constructor(filePath: string, spawnProcess?: SpawnBackgroundProcess, isProcessRunning?: IsProcessRunning) {
    this.filePath = filePath;
    this.rootDir = path.dirname(filePath);
    this.executionService = new BackgroundTaskExecutionService(this.rootDir, spawnProcess, isProcessRunning);
  }

  private readonly filePath: string;

  async load(): Promise<BackgroundTaskStoreShape> {
    return await readJsonFile(this.filePath, EMPTY_STORE);
  }

  async save(store: BackgroundTaskStoreShape): Promise<void> {
    await writeJsonAtomic(this.filePath, store);
  }

  async list(): Promise<BackgroundTaskRecord[]> {
    const store = await this.load();
    return [...store.tasks].sort((a, b) => a.createdAt.localeCompare(b.createdAt));
  }

  async listBySession(sessionId: string): Promise<BackgroundTaskRecord[]> {
    const tasks = await this.list();
    return tasks.filter((task) => task.sessionId === sessionId);
  }

  async get(taskId: string): Promise<BackgroundTaskRecord | undefined> {
    const store = await this.load();
    return store.tasks.find((task) => task.id === taskId);
  }

  async start(params: StartBackgroundTaskParams): Promise<BackgroundTaskRecord> {
    const task = createBackgroundTaskRecord({
      id: randomUUID(),
      title: params.title,
      command: params.command,
      cwd: params.cwd ?? this.rootDir,
      kind: params.kind ?? "task",
      sessionId: params.sessionId,
      metadata: params.metadata,
    });
    const store = await this.load();
    store.tasks.push(task);
    await this.save(store);

    try {
      await this.executionService.writeArtifacts(task);
      const launched = await this.executionService.launch(task);
      return (await this.markStarted(task.id, { pid: launched.pid })) ?? task;
    } catch (error) {
      const failed = await this.fail(task.id, {
        reason: error instanceof Error ? error.message : String(error),
      });
      if (!failed) {
        throw error;
      }
      return failed;
    }
  }

  async markStarted(
    taskId: string,
    params: MarkBackgroundTaskStartedParams,
  ): Promise<BackgroundTaskRecord | undefined> {
    return await this.update(taskId, (task) => {
      const startedAt = task.execution.startedAt ?? nowIso();
      return {
        ...task,
        status: "running",
        updatedAt: startedAt,
        execution: {
          ...task.execution,
          startedAt,
          completedAt: undefined,
          failedAt: undefined,
          lastUpdatedAt: startedAt,
          processId: params.pid,
          exitCode: undefined,
          signal: undefined,
          error: undefined,
        },
      };
    });
  }

  async complete(
    taskId: string,
    params: CompleteBackgroundTaskParams = {},
  ): Promise<BackgroundTaskRecord | undefined> {
    return await this.update(taskId, (task) => {
      const completedAt = params.completedAt ?? nowIso();
      return {
        ...task,
        status: "completed",
        updatedAt: completedAt,
        execution: {
          ...task.execution,
          completedAt,
          failedAt: undefined,
          lastUpdatedAt: completedAt,
          exitCode: params.exitCode ?? task.execution.exitCode ?? 0,
          signal: undefined,
          error: undefined,
        },
      };
    });
  }

  async fail(
    taskId: string,
    params: FailBackgroundTaskParams,
  ): Promise<BackgroundTaskRecord | undefined> {
    return await this.update(taskId, (task) => {
      const failedAt = params.completedAt ?? nowIso();
      return {
        ...task,
        status: "failed",
        updatedAt: failedAt,
        execution: {
          ...task.execution,
          completedAt: undefined,
          failedAt,
          lastUpdatedAt: failedAt,
          exitCode: params.exitCode,
          signal: params.signal,
          error: params.reason,
        },
      };
    });
  }

  async cancel(taskId: string, params: StopBackgroundTaskParams = {}): Promise<BackgroundTaskRecord | undefined> {
    const task = await this.get(taskId);
    if (!task) {
      return undefined;
    }
    const state = await this.executionService.stop(task, params);
    return await this.update(taskId, (current) => applyExecutionState(current, state));
  }

  async sync(taskId: string): Promise<BackgroundTaskRecord | undefined> {
    const task = await this.get(taskId);
    if (!task) {
      return undefined;
    }
    return (await this.reconcileTask(task)).task;
  }

  async recover(taskId: string): Promise<BackgroundTaskRecoveryResult | undefined> {
    const task = await this.get(taskId);
    return task ? await this.reconcileTask(task) : undefined;
  }

  async recoverAll(): Promise<BackgroundTaskRecoveryResult[]> {
    const tasks = await this.list();
    const recovered: BackgroundTaskRecoveryResult[] = [];
    for (const task of tasks) {
      recovered.push(await this.reconcileTask(task));
    }
    return recovered;
  }

  async recoverBySession(sessionId: string): Promise<BackgroundTaskRecoveryResult[]> {
    const tasks = await this.listBySession(sessionId);
    const recovered: BackgroundTaskRecoveryResult[] = [];
    for (const task of tasks) {
      recovered.push(await this.reconcileTask(task));
    }
    return recovered;
  }

  async getExecutionState(taskId: string): Promise<BackgroundTaskExecutionState | undefined> {
    const task = await this.sync(taskId);
    return task ? await this.executionService.readState(task) : undefined;
  }

  async getOutput(taskId: string, options: number | ReadBackgroundTaskOutputOptions = {}): Promise<string | undefined> {
    const task = await this.get(taskId);
    if (!task) {
      return undefined;
    }
    const readOptions = typeof options === "number" ? { lineLimit: options } : options;
    return await this.executionService.readOutput(task, readOptions);
  }

  private async reconcileTask(task: BackgroundTaskRecord): Promise<BackgroundTaskRecoveryResult> {
    const state = await this.executionService.readState(task);
    if (state) {
      return await this.reconcilePersistedState(task, state);
    }
    return await this.reconcileMissingState(task);
  }

  private async reconcilePersistedState(
    task: BackgroundTaskRecord,
    state: BackgroundTaskExecutionState,
  ): Promise<BackgroundTaskRecoveryResult> {
    if (state.status === "running") {
      if (this.executionService.isProcessRunning(state.pid)) {
        return await this.persistRecoveredTask(task, applyExecutionState(task, state), "state-running");
      }
      const failedState = createFailedExecutionState(task, state.pid, {
        startedAt: state.startedAt,
        exitCode: state.exitCode ?? 1,
        reason: state.error ?? `background task process ${state.pid} is no longer running`,
      });
      await this.executionService.writeState(task, failedState);
      return await this.persistRecoveredTask(task, applyExecutionState(task, failedState), "missing-process");
    }

    return await this.persistRecoveredTask(task, applyExecutionState(task, state), recoveryReasonForState(state.status));
  }

  private async reconcileMissingState(task: BackgroundTaskRecord): Promise<BackgroundTaskRecoveryResult> {
    if (task.status === "planned") {
      const failed = await this.fail(task.id, {
        reason: "background task never wrote execution state",
      });
      return {
        task: failed ?? task,
        changed: Boolean(failed),
        reason: "missing-state",
      };
    }

    if (task.status === "running") {
      const pid = task.execution.processId;
      if (typeof pid === "number" && this.executionService.isProcessRunning(pid)) {
        const runningState = createRunningExecutionState(task, pid);
        await this.executionService.writeState(task, runningState);
        return await this.persistRecoveredTask(task, applyExecutionState(task, runningState), "process-running", true);
      }

      const failed = await this.fail(task.id, {
        reason:
          typeof pid === "number"
            ? `background task process ${pid} is no longer running`
            : "background task has no tracked process",
        exitCode: 1,
      });
      return {
        task: failed ?? task,
        changed: Boolean(failed),
        reason: "missing-process",
      };
    }

    return {
      task,
      changed: false,
      reason: "unchanged",
    };
  }

  private async persistRecoveredTask(
    previous: BackgroundTaskRecord,
    next: BackgroundTaskRecord,
    reason: BackgroundTaskRecoveryReason,
    forceChanged = false,
  ): Promise<BackgroundTaskRecoveryResult> {
    if (!forceChanged && !hasTaskChanged(previous, next)) {
      return {
        task: previous,
        changed: false,
        reason: "unchanged",
      };
    }
    const saved = await this.update(previous.id, () => next);
    return {
      task: saved ?? next,
      changed: true,
      reason,
    };
  }

  private async update(
    taskId: string,
    apply: (task: BackgroundTaskRecord) => BackgroundTaskRecord,
  ): Promise<BackgroundTaskRecord | undefined> {
    const store = await this.load();
    const index = store.tasks.findIndex((task) => task.id === taskId);
    if (index < 0) {
      return undefined;
    }
    const updated = apply(store.tasks[index]);
    store.tasks[index] = updated;
    await this.save(store);
    return updated;
  }
}

function createBackgroundTaskRecord(params: {
  id: string;
  title: string;
  command: string;
  cwd: string;
  kind: BackgroundTaskKind;
  sessionId?: string;
  metadata?: Record<string, unknown>;
}): BackgroundTaskRecord {
  const createdAt = nowIso();
  const baseDir = `background-tasks/${params.id}`;
  return {
    version: 1,
    id: params.id,
    title: params.title,
    command: params.command,
    cwd: params.cwd,
    kind: params.kind,
    createdAt,
    updatedAt: createdAt,
    status: "planned",
    ...(params.sessionId ? { sessionId: params.sessionId } : {}),
    ...(params.metadata ? { metadata: params.metadata } : {}),
    execution: {
      version: 1,
      preparedAt: createdAt,
      lastUpdatedAt: createdAt,
      workingDirectory: baseDir,
      outputFile: `${baseDir}/output.log`,
      stateFile: `${baseDir}/state.json`,
      launchScript: `${baseDir}/run.sh`,
    },
  };
}

function createRunningExecutionState(task: BackgroundTaskRecord, pid: number): BackgroundTaskExecutionState {
  const updatedAt = nowIso();
  return {
    version: 1,
    taskId: task.id,
    kind: task.kind,
    status: "running",
    pid,
    startedAt: task.execution.startedAt ?? task.updatedAt,
    updatedAt,
    outputFile: task.execution.outputFile,
    cwd: task.cwd,
    command: task.command,
  };
}

function createFailedExecutionState(
  task: BackgroundTaskRecord,
  pid: number,
  params: { startedAt?: string; exitCode?: number; signal?: string; reason: string },
): BackgroundTaskExecutionState {
  const completedAt = nowIso();
  return {
    version: 1,
    taskId: task.id,
    kind: task.kind,
    status: "failed",
    pid,
    startedAt: params.startedAt ?? task.execution.startedAt ?? task.updatedAt,
    updatedAt: completedAt,
    completedAt,
    exitCode: params.exitCode ?? 1,
    signal: params.signal,
    error: params.reason,
    outputFile: task.execution.outputFile,
    cwd: task.cwd,
    command: task.command,
  };
}

function recoveryReasonForState(
  status: BackgroundTaskExecutionState["status"],
): Extract<BackgroundTaskRecoveryReason, "state-running" | "state-completed" | "state-failed" | "state-cancelled"> {
  if (status === "running") {
    return "state-running";
  }
  if (status === "completed") {
    return "state-completed";
  }
  if (status === "cancelled") {
    return "state-cancelled";
  }
  return "state-failed";
}

function hasTaskChanged(previous: BackgroundTaskRecord, next: BackgroundTaskRecord): boolean {
  return JSON.stringify(previous) !== JSON.stringify(next);
}

function applyExecutionState(task: BackgroundTaskRecord, state: BackgroundTaskExecutionState): BackgroundTaskRecord {
  if (state.status === "running") {
    return {
      ...task,
      status: "running",
      updatedAt: state.updatedAt,
      execution: {
        ...task.execution,
        processId: state.pid,
        startedAt: state.startedAt,
        completedAt: undefined,
        failedAt: undefined,
        lastUpdatedAt: state.updatedAt,
        exitCode: undefined,
        signal: undefined,
        error: undefined,
      },
    };
  }
  if (state.status === "completed") {
    return {
      ...task,
      status: "completed",
      updatedAt: state.updatedAt,
      execution: {
        ...task.execution,
        processId: state.pid,
        startedAt: state.startedAt,
        completedAt: state.completedAt ?? state.updatedAt,
        failedAt: undefined,
        lastUpdatedAt: state.updatedAt,
        exitCode: state.exitCode,
        signal: undefined,
        error: undefined,
      },
    };
  }
  if (state.status === "cancelled") {
    return {
      ...task,
      status: "cancelled",
      updatedAt: state.updatedAt,
      execution: {
        ...task.execution,
        processId: state.pid,
        startedAt: state.startedAt,
        completedAt: state.completedAt ?? state.updatedAt,
        failedAt: undefined,
        lastUpdatedAt: state.updatedAt,
        exitCode: state.exitCode,
        signal: state.signal,
        error: state.error,
      },
    };
  }
  return {
    ...task,
    status: "failed",
    updatedAt: state.updatedAt,
    execution: {
      ...task.execution,
      processId: state.pid,
      startedAt: state.startedAt,
      completedAt: undefined,
      failedAt: state.completedAt ?? state.updatedAt,
      lastUpdatedAt: state.updatedAt,
      exitCode: state.exitCode,
      signal: state.signal,
      error: state.error ?? `background task exited with code ${state.exitCode ?? "unknown"}`,
    },
  };
}

function renderLaunchScript(task: BackgroundTaskRecord): string {
  const quotedStatePath = shellQuote(task.execution.stateFile);
  const quotedOutputFile = shellQuote(task.execution.outputFile);
  const quotedCwd = shellQuote(task.cwd);
  const quotedCommand = shellQuote(task.command);
  const quotedStatePayload = shellQuote(
    JSON.stringify({
      version: 1,
      taskId: task.id,
      kind: task.kind,
      status: "running",
      pid: "$$",
      startedAt: "__OPENCLAW_STARTED_AT__",
      updatedAt: "__OPENCLAW_STARTED_AT__",
      outputFile: task.execution.outputFile,
      cwd: task.cwd,
      command: task.command,
    }),
  );

  return [
    "#!/usr/bin/env bash",
    "set -euo pipefail",
    `mkdir -p $(dirname ${quotedStatePath}) $(dirname ${quotedOutputFile})`,
    "started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    `__state_tmp=${quotedStatePath}.$$.tmp`,
    `printf '%s' ${quotedStatePayload} | sed "s/__OPENCLAW_STARTED_AT__/$started_at/g; s/\"\$\$\"/$$/g" > "$__state_tmp" && mv -f "$__state_tmp" ${quotedStatePath}`,
    `printf '%s\n' "starting ${task.kind} ${task.id}" >> ${quotedOutputFile}`,
    `if cd ${quotedCwd} && bash -lc ${quotedCommand} >> ${quotedOutputFile} 2>&1; then`,
    "  completed_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    `  python3 - ${quotedStatePath} $$ "$completed_at" 0 <<'PY'`,
    ...renderStateWriterPython("completed"),
    "PY",
    "else",
    "  exit_code=$?",
    "  completed_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    `  python3 - ${quotedStatePath} $$ "$completed_at" "$exit_code" <<'PY'`,
    ...renderStateWriterPython("failed"),
    "PY",
    '  exit "$exit_code"',
    "fi",
    "",
  ].join("\n");
}

function renderStateWriterPython(status: BackgroundTaskExecutionState["status"]): string[] {
  return [
    "import json",
    "import pathlib",
    "import sys",
    "state_path = pathlib.Path(sys.argv[1])",
    "pid = int(sys.argv[2])",
    "timestamp = sys.argv[3]",
    "exit_code = int(sys.argv[4])",
    "state = json.loads(state_path.read_text())",
    `state['status'] = '${status}'`,
    "state['pid'] = pid",
    "state['updatedAt'] = timestamp",
    "state['completedAt'] = timestamp",
    "state['exitCode'] = exit_code",
    `state['error'] = None if '${status}' == 'completed' else f'background task exited non-zero ({exit_code})'`,
    "tmp_path = state_path.with_name(state_path.name + f'.{pid}.tmp')",
    "tmp_path.write_text(json.dumps(state, indent=2) + '\\n')",
    "tmp_path.replace(state_path)",
  ];
}

function shellQuote(value: string): string {
  return `'${value.replaceAll(`'`, `'"'"'`)}'`;
}
