import { execFileSync } from "node:child_process";
import { describe, expect, it } from "vitest";

import { shellSingleQuote } from "./shell.js";

/**
 * Round-trip a value through `printf %s <quoted>` in a real shell and assert the
 * shell reproduces the byte-exact input. If the quoting is wrong, the shell
 * either errors or emits a corrupted value.
 */
function shellRoundTrip(value: string): string {
  return execFileSync("bash", ["-c", `printf %s ${shellSingleQuote(value)}`], {
    encoding: "utf8",
  });
}

describe("shellSingleQuote", () => {
  const cases: Array<[string, string]> = [
    ["empty", ""],
    ["plain", "abc"],
    ["single quote", "abc'def"],
    ["leading/trailing quotes", "'wrapped'"],
    ["printf command", "printf 'line-1\nline-2\n'"],
    ["double quotes", 'say "hello"'],
    ["mixed quotes", `it's a "test"`],
    ["dollar and backtick", "$HOME `whoami` ${x}"],
    ["backslashes", "a\\b\\'c"],
    ["spaces and semicolons", "rm -rf /; echo pwned"],
  ];

  for (const [name, value] of cases) {
    it(`round-trips ${name} through a real shell`, () => {
      expect(shellRoundTrip(value)).toBe(value);
    });
  }

  it("produces valid JSON when a quoted JSON payload contains single quotes", () => {
    const payload = JSON.stringify({
      command: "printf 'line-1\nline-2\n'",
      note: "it's fine",
    });
    const emitted = execFileSync("bash", ["-c", `printf %s ${shellSingleQuote(payload)}`], {
      encoding: "utf8",
    });
    const parsed = JSON.parse(emitted) as { command: string; note: string };
    expect(parsed.command).toBe("printf 'line-1\nline-2\n'");
    expect(parsed.note).toBe("it's fine");
  });
});
