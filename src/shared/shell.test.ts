import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { describe, expect, it } from "vitest";
import { singleQuote } from "./shell.js";

const execFileAsync = promisify(execFile);

describe("singleQuote", () => {
  it("wraps plain values in single quotes", () => {
    expect(singleQuote("printf hello")).toBe("'printf hello'");
  });

  it("escapes embedded single quotes with the canonical POSIX idiom", () => {
    // Regression: a malformed escape (`"'"'"'`) here corrupted the JSON state
    // payloads that background-task/training launch scripts write, so recovery
    // failed to parse them. The correct escape is `'"'"'`.
    expect(singleQuote("printf 'line-1\\nline-2\\n'")).toBe(
      `'printf '"'"'line-1\\nline-2\\n'"'"''`,
    );
  });

  it("round-trips arbitrary bytes through a real shell", async () => {
    const cases = [
      "printf 'line-1\\nline-2\\n'",
      `it's a "quote" $HOME \\backslash\\`,
      "nested 'a'b'c' quotes",
      "trailing quote'",
    ];
    for (const value of cases) {
      const { stdout } = await execFileAsync("bash", ["-c", `printf '%s' ${singleQuote(value)}`]);
      expect(stdout).toBe(value);
    }
  });

  it("keeps a JSON payload valid after the launch-script sed pass", async () => {
    // Mirrors the exact pattern the background-task launch script uses: a JSON
    // state payload (whose `command` field contains single quotes) is emitted
    // via `printf` and rewritten with `sed`, then must still JSON.parse.
    const payload = JSON.stringify({
      version: 1,
      status: "running",
      pid: "$$",
      startedAt: "__STAMP__",
      command: "printf 'line-1\\nline-2\\n'",
    });
    const script = `printf '%s' ${singleQuote(payload)} | sed "s/__STAMP__/2026-01-01T00:00:00Z/g; s/\\"\\$\\$\\"/4242/g"`;
    const { stdout } = await execFileAsync("bash", ["-c", script]);
    const parsed = JSON.parse(stdout) as { command: string; pid: number; startedAt: string };
    expect(parsed.command).toBe("printf 'line-1\\nline-2\\n'");
    expect(parsed.pid).toBe(4242);
    expect(parsed.startedAt).toBe("2026-01-01T00:00:00Z");
  });
});
