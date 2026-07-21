import { execFileSync } from "node:child_process";
import { describe, expect, it } from "vitest";
import { posixSingleQuote } from "./shell.js";

describe("posixSingleQuote", () => {
  it("wraps a plain word in single quotes", () => {
    expect(posixSingleQuote("hello")).toBe("'hello'");
  });

  it("escapes an embedded single quote with the canonical sequence", () => {
    // Regression guard: the previous buggy implementation produced
    // '\''a"\'"\'"\'b'\'' (a leading double-quote), which injected a spurious
    // `"` and corrupted any JSON payload embedded in a generated shell script.
    expect(posixSingleQuote("a'b")).toBe(`'a'"'"'b'`);
  });

  it("preserves newlines, dollars, backticks and double quotes verbatim through bash", () => {
    const samples = [
      `printf 'line-1\nline-2\n'`,
      `{"pid":"$$","command":"echo 'hi'"}`,
      "weird `backtick` and \"double\" and $HOME and \\backslash",
      "trailing single quote'",
      "'leading single quote",
      "''double''",
    ];
    for (const sample of samples) {
      const roundTripped = execFileSync("bash", ["-c", `printf '%s' ${posixSingleQuote(sample)}`], {
        encoding: "utf8",
      });
      expect(roundTripped).toBe(sample);
    }
  });

  it("keeps an embedded JSON state payload parseable after a bash printf", () => {
    const payload = JSON.stringify({
      version: 1,
      status: "running",
      command: `printf 'line-1\nline-2\n'`,
    });
    const written = execFileSync("bash", ["-c", `printf '%s' ${posixSingleQuote(payload)}`], {
      encoding: "utf8",
    });
    expect(() => JSON.parse(written)).not.toThrow();
    expect(JSON.parse(written)).toEqual(JSON.parse(payload));
  });
});
