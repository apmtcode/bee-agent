import { spawnSync } from "node:child_process";
import { describe, expect, it } from "vitest";

import { shellQuote } from "./background-tasks.js";

/**
 * Round-trip a value through a POSIX shell: `printf '%s' <shellQuote(value)>`
 * must reproduce the original bytes exactly. Returns undefined when no POSIX
 * shell is available so the assertion can be skipped in that (rare) case.
 */
function shellRoundTrip(value: string): string | undefined {
  const result = spawnSync("/bin/sh", ["-c", `printf '%s' ${shellQuote(value)}`], {
    encoding: "utf8",
  });
  if (result.error) {
    return undefined;
  }
  return result.stdout;
}

describe("shellQuote", () => {
  const cases: Array<[string, string]> = [
    ["plain", "hello-world"],
    ["single quote", "printf 'line-1'"],
    ["multiple single quotes", "a'b'c'd"],
    ["json payload with quotes", JSON.stringify({ command: "printf 'line-1\nline-2\n'" })],
    ["double quotes", 'say "hi"'],
    ["mixed quotes", `it's a "test"`],
    ["dollar and backtick", "cost is $5 `whoami`"],
    ["backslashes", "a\\b\\'c"],
    ["empty", ""],
  ];

  it.each(cases)("round-trips %s through a POSIX shell", (_label, value) => {
    const out = shellRoundTrip(value);
    if (out === undefined) {
      return; // no /bin/sh in this environment
    }
    expect(out).toBe(value);
  });

  it("uses the correct POSIX escape sequence for embedded single quotes", () => {
    // The value a'b must become '<open>a'"'"'b'<close>, NOT the broken
    // '"'"'"'-based form that regressed background-task state writes.
    expect(shellQuote("a'b")).toBe(`'a'"'"'b'`);
  });

  it("produces a state JSON payload that parses after shell evaluation", () => {
    const payload = JSON.stringify({
      version: 1,
      taskId: "task-123",
      status: "running",
      command: "printf 'line-1\nline-2\n'",
      cwd: "/tmp/some 'dir'",
    });
    const out = shellRoundTrip(payload);
    if (out === undefined) {
      return;
    }
    // The whole point: the payload survives the shell and is still valid JSON.
    expect(() => JSON.parse(out)).not.toThrow();
    expect(JSON.parse(out)).toEqual(JSON.parse(payload));
  });
});
