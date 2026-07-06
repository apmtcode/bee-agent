import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { describe, expect, it } from "vitest";
import { shellQuote } from "./background-tasks.js";

const run = promisify(execFile);

/**
 * Round-trip a value through bash: `printf '%s' <shellQuote(value)>` must emit
 * the original bytes verbatim. This guards the launch-script generator against
 * the class of bug where an embedded single quote injected a stray character
 * and corrupted the JSON state file the script writes.
 */
async function roundTrip(value: string): Promise<string> {
  const { stdout } = await run("bash", ["-c", `printf '%s' ${shellQuote(value)}`]);
  return stdout;
}

describe("shellQuote", () => {
  const cases = [
    "plain",
    "with spaces",
    "printf 'line-1\nline-2\n'",
    `has "double" quotes`,
    "mixed 'single' and \"double\"",
    "dollar $$ and $VAR and ${BRACE}",
    "back\\slash and `backtick`",
    "semicolon; pipe | amp & redirect > <",
    "unicode ✓ é 漢字",
    "'leading and trailing'",
    "",
  ];

  for (const value of cases) {
    it(`round-trips ${JSON.stringify(value)} through bash unchanged`, async () => {
      expect(await roundTrip(value)).toBe(value);
    });
  }

  it("does not inject a stray character for embedded single quotes", () => {
    // The buggy version produced `'x'"'"'y'` for `x'y` (an extra double quote);
    // the correct `'\''` idiom produces `'x'\''y'`.
    expect(shellQuote("x'y")).toBe(`'x'\\''y'`);
  });
});
