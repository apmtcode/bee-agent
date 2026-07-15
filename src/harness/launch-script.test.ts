import { execFileSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import {
  renderLaunchScript,
  shellQuote,
  type BackgroundTaskExecutionState,
  type BackgroundTaskRecord,
} from "./background-tasks.js";

const tempDirs: string[] = [];

function makeTempDir(): string {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "launch-script-"));
  tempDirs.push(dir);
  return dir;
}

afterEach(() => {
  for (const dir of tempDirs.splice(0)) {
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

// Runs `printf '%s' <shellQuote(value)>` in a real shell and returns what the
// shell actually produced. Round-tripping through bash is the ground truth for
// whether the quoting is POSIX-correct.
function shellRoundTrip(value: string): string {
  return execFileSync("bash", ["-c", `printf '%s' ${shellQuote(value)}`], {
    encoding: "utf8",
  });
}

describe("shellQuote", () => {
  it("round-trips values containing single quotes without corruption", () => {
    for (const value of [
      "printf 'line-1\nline-2\n'",
      "git commit -m 'fix: bug'",
      "echo 'a'\"'\"'b'",
      "plain-no-quotes",
      "nested 'quotes' and \"double\" and $vars and `ticks`",
    ]) {
      expect(shellRoundTrip(value)).toBe(value);
    }
  });
});

describe("renderLaunchScript", () => {
  it("produces a valid state file with a numeric pid and the exact command for a single-quoted command", () => {
    const rootDir = makeTempDir();
    const baseDir = "background-tasks/task-1";
    const command = "printf '%s' 'hello '\"'\"'world'\"'\"''";
    const task = {
      version: 1,
      id: "task-1",
      title: "quoting regression",
      kind: "task",
      command,
      cwd: rootDir,
      createdAt: "2026-01-01T00:00:00.000Z",
      updatedAt: "2026-01-01T00:00:00.000Z",
      status: "running",
      execution: {
        version: 1,
        preparedAt: "2026-01-01T00:00:00.000Z",
        lastUpdatedAt: "2026-01-01T00:00:00.000Z",
        workingDirectory: rootDir,
        outputFile: `${baseDir}/output.log`,
        stateFile: `${baseDir}/state.json`,
        launchScript: `${baseDir}/launch.sh`,
      },
    } satisfies BackgroundTaskRecord;

    const scriptPath = path.join(rootDir, task.execution.launchScript);
    fs.mkdirSync(path.dirname(scriptPath), { recursive: true });
    fs.writeFileSync(scriptPath, renderLaunchScript(task), { mode: 0o700 });

    // Run synchronously to completion — deterministic, no racing detached child.
    execFileSync("bash", [scriptPath], { cwd: rootDir });

    const raw = fs.readFileSync(path.join(rootDir, task.execution.stateFile), "utf8");
    // Must be valid JSON — the shellQuote bug used to corrupt this file.
    const state = JSON.parse(raw) as BackgroundTaskExecutionState;

    expect(state.command).toBe(command);
    expect(typeof state.pid).toBe("number");
    expect(Number.isFinite(state.pid)).toBe(true);
    expect(state.pid).toBeGreaterThan(0);
    expect(state.status).toBe("completed");

    const output = fs.readFileSync(path.join(rootDir, task.execution.outputFile), "utf8");
    expect(output).toContain("hello 'world'");
  });

  it("writes a numeric pid even in the initial running-state substitution", () => {
    const rootDir = makeTempDir();
    const baseDir = "background-tasks/task-2";
    const task = {
      version: 1,
      id: "task-2",
      title: "pid substitution",
      kind: "monitor",
      // A command that lingers just long enough to observe the running state,
      // then exits on its own so the synchronous run terminates.
      command: "sleep 0.2",
      cwd: rootDir,
      createdAt: "2026-01-01T00:00:00.000Z",
      updatedAt: "2026-01-01T00:00:00.000Z",
      status: "running",
      execution: {
        version: 1,
        preparedAt: "2026-01-01T00:00:00.000Z",
        lastUpdatedAt: "2026-01-01T00:00:00.000Z",
        workingDirectory: rootDir,
        outputFile: `${baseDir}/output.log`,
        stateFile: `${baseDir}/state.json`,
        launchScript: `${baseDir}/launch.sh`,
      },
    } satisfies BackgroundTaskRecord;

    const script = renderLaunchScript(task);
    // The generated bash must escape the pid placeholder's quotes so the JSON
    // string `"__OPENCLAW_PID__"` becomes the bare numeric `$$`, not a broken
    // `s/<pid>/<pid>/g` no-op. Assert the substitution is present and correct.
    expect(script).toContain('s/\\"__OPENCLAW_PID__\\"/$$/g');
    expect(script).not.toContain('"pid":"$$"');

    const scriptPath = path.join(rootDir, task.execution.launchScript);
    fs.mkdirSync(path.dirname(scriptPath), { recursive: true });
    fs.writeFileSync(scriptPath, script, { mode: 0o700 });
    execFileSync("bash", [scriptPath], { cwd: rootDir });

    const state = JSON.parse(
      fs.readFileSync(path.join(rootDir, task.execution.stateFile), "utf8"),
    ) as BackgroundTaskExecutionState;
    expect(typeof state.pid).toBe("number");
    expect(state.pid).toBeGreaterThan(0);
  });
});
