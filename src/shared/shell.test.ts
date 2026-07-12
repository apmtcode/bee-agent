import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { describe, expect, it } from "vitest";
import { shellQuote, shellQuoteCommand } from "./shell.js";

const run = promisify(execFile);

/**
 * Echo a value back through a real POSIX shell using the quoted form, proving
 * that `printf '%s'` on the quoted word reproduces the original bytes exactly.
 */
async function roundTrip(value: string): Promise<string> {
  const { stdout } = await run("bash", ["-c", `printf '%s' ${shellQuote(value)}`]);
  return stdout;
}

describe("shellQuote", () => {
  it("wraps a plain word in single quotes", () => {
    expect(shellQuote("hello")).toBe("'hello'");
  });

  it("escapes an embedded single quote with the canonical sequence", () => {
    // The historical bug used "'\"'\"'\"'" (6 chars) which produced an
    // unbalanced, corrupt word. The correct escape is '"'"' (5 chars).
    expect(shellQuote("a'b")).toBe(`'a'"'"'b'`);
  });

  it("round-trips a command containing single quotes through a real shell", async () => {
    const value = "printf 'line-1\nline-2\n'";
    await expect(roundTrip(value)).resolves.toBe(value);
  });

  it.each([
    "plain",
    "with spaces",
    "single 'quote'",
    'double "quote"',
    "mixed 'a' and \"b\"",
    "back`tick` and $dollar and ${brace}",
    "semis; pipes | and && ops",
    "newline\nand\ttab",
    "trailing quote'",
    "'leading quote",
    "'''",
  ])("round-trips %j through a real shell", async (value) => {
    await expect(roundTrip(value)).resolves.toBe(value);
  });
});

describe("shellQuoteCommand", () => {
  it("quotes each argv entry independently and preserves word boundaries", () => {
    expect(shellQuoteCommand(["printf", "%s\n", "a b"])).toBe(`'printf' '%s\n' 'a b'`);
  });

  it("round-trips an argv with quotes and spaces into distinct arguments", async () => {
    const argv = ["a b", "it's", 'say "hi"'];
    const { stdout } = await run("bash", ["-c", `for a in ${shellQuoteCommand(argv)}; do printf '[%s]' "$a"; done`]);
    expect(stdout).toBe("[a b][it's][say \"hi\"]");
  });
});
