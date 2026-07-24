import { spawn } from "node:child_process";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import {
  renderAtomicPythonStateWriter,
  renderAtomicShellStateWrite,
  shellQuote,
} from "./launch-script.js";

const tempDirs: string[] = [];

afterEach(async () => {
  await Promise.all(tempDirs.splice(0).map((dir) => fs.rm(dir, { recursive: true, force: true })));
});

async function makeTempDir(): Promise<string> {
  const dir = await fs.mkdtemp(path.join(os.tmpdir(), "launch-script-"));
  tempDirs.push(dir);
  return dir;
}

function runScript(scriptPath: string): Promise<number> {
  return new Promise((resolve, reject) => {
    const child = spawn("bash", [scriptPath], { stdio: "ignore" });
    child.on("error", reject);
    child.on("exit", (code) => resolve(code ?? -1));
  });
}

describe("shellQuote", () => {
  it("wraps a plain value and escapes embedded single quotes", () => {
    expect(shellQuote("plain")).toBe("'plain'");
    expect(shellQuote("it's")).toBe(`'it'"'"'s'`);
  });
});

describe("renderAtomicShellStateWrite", () => {
  it("writes producer output to a temp file then renames it into place", async () => {
    const dir = await makeTempDir();
    const statePath = path.join(dir, "state.json");
    const scriptPath = path.join(dir, "write.sh");
    const script = [
      "#!/usr/bin/env bash",
      "set -euo pipefail",
      ...renderAtomicShellStateWrite(shellQuote(statePath), `printf '%s' '{"status":"running"}'`),
      "",
    ].join("\n");
    await fs.writeFile(scriptPath, script, { mode: 0o700 });

    expect(await runScript(scriptPath)).toBe(0);
    expect(JSON.parse(await fs.readFile(statePath, "utf8"))).toEqual({ status: "running" });
    // The temp file must not linger after the atomic rename.
    const leftovers = (await fs.readdir(dir)).filter((name) => name.includes(".tmp"));
    expect(leftovers).toEqual([]);
  });
});

describe("renderAtomicPythonStateWriter", () => {
  it("loads, mutates, and atomically replaces the state file (completed)", async () => {
    const dir = await makeTempDir();
    const statePath = path.join(dir, "state.json");
    await fs.writeFile(statePath, JSON.stringify({ version: 1, status: "running", pid: 1 }));
    const scriptPath = path.join(dir, "finish.sh");
    const script = [
      "#!/usr/bin/env bash",
      "set -euo pipefail",
      `python3 - ${shellQuote(statePath)} 4321 "2026-01-01T00:00:00Z" 0 <<'PY'`,
      ...renderAtomicPythonStateWriter("completed", "None"),
      "PY",
      "",
    ].join("\n");
    await fs.writeFile(scriptPath, script, { mode: 0o700 });

    expect(await runScript(scriptPath)).toBe(0);
    const state = JSON.parse(await fs.readFile(statePath, "utf8"));
    expect(state).toMatchObject({ status: "completed", pid: 4321, exitCode: 0, error: null });
    expect((await fs.readdir(dir)).filter((name) => name.includes(".tmp"))).toEqual([]);
  });

  it("records the failure error expression when status is failed", async () => {
    const dir = await makeTempDir();
    const statePath = path.join(dir, "state.json");
    await fs.writeFile(statePath, JSON.stringify({ version: 1, status: "running", pid: 1 }));
    const scriptPath = path.join(dir, "fail.sh");
    const script = [
      "#!/usr/bin/env bash",
      "set -euo pipefail",
      `python3 - ${shellQuote(statePath)} 7 "2026-01-01T00:00:00Z" 3 <<'PY'`,
      ...renderAtomicPythonStateWriter("failed", "f'exited non-zero ({exit_code})'"),
      "PY",
      "",
    ].join("\n");
    await fs.writeFile(scriptPath, script, { mode: 0o700 });

    expect(await runScript(scriptPath)).toBe(0);
    const state = JSON.parse(await fs.readFile(statePath, "utf8"));
    expect(state).toMatchObject({ status: "failed", exitCode: 3, error: "exited non-zero (3)" });
  });

  it("never exposes a torn state file to a concurrent reader under repeated writes", async () => {
    const dir = await makeTempDir();
    const statePath = path.join(dir, "state.json");
    // Seed a large, valid state so a non-atomic writer would produce observable
    // torn reads mid-write; the atomic rename must prevent that entirely.
    const seed = { version: 1, status: "running", pid: 1, note: "x".repeat(4096) };
    await fs.writeFile(statePath, JSON.stringify(seed));
    const scriptPath = path.join(dir, "spin.sh");
    const script = [
      "#!/usr/bin/env bash",
      "set -euo pipefail",
      "for i in $(seq 1 40); do",
      ...renderAtomicShellStateWrite(
        shellQuote(statePath),
        `printf '{"version":1,"status":"running","pid":%s,"note":"%s"}' "$i" "$(printf 'y%.0s' $(seq 1 4096))"`,
      ).map((line) => `  ${line}`),
      "done",
      "",
    ].join("\n");
    await fs.writeFile(scriptPath, script, { mode: 0o700 });

    let writerDone = false;
    const writer = runScript(scriptPath).then((code) => {
      writerDone = true;
      return code;
    });
    let tornReads = 0;
    let reads = 0;
    while (!writerDone) {
      try {
        const raw = await fs.readFile(statePath, "utf8");
        reads += 1;
        try {
          JSON.parse(raw);
        } catch {
          tornReads += 1;
        }
      } catch {
        // ENOENT is impossible with atomic rename, but ignore any transient read error.
      }
    }
    expect(await writer).toBe(0);
    expect(reads).toBeGreaterThan(0);
    expect(tornReads).toBe(0);
  });
});
