import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { describe, expect, it } from "vitest";
import { shellQuote } from "./background-tasks.js";

const execFileAsync = promisify(execFile);

/**
 * Run `printf %s <quoted>` through a real bash and return exactly what the
 * shell reconstructs. If shellQuote is correct, this must equal the input
 * verbatim for any string — the whole point of quoting.
 */
async function bashRoundTrip(value: string): Promise<string> {
  const { stdout } = await execFileAsync("bash", ["-c", `printf %s ${shellQuote(value)}`]);
  return stdout;
}

describe("shellQuote", () => {
  const cases = [
    "plain",
    "with spaces",
    "single'quote",
    "many'''quotes",
    "double\"quote",
    "mixed'\"both",
    "printf 'line-1\nline-2\n'",
    `{"command":"printf 'hi'","cwd":"/tmp"}`, // JSON payload containing a quote
    "$HOME `whoami` $(id) ${x}", // no expansion must occur inside single quotes
    "back\\slash",
    "newline\nembedded",
    "tab\tembedded",
  ];

  for (const value of cases) {
    it(`round-trips ${JSON.stringify(value)} through bash unchanged`, async () => {
      expect(await bashRoundTrip(value)).toBe(value);
    });
  }

  it("never injects a stray double-quote when escaping a single-quote", async () => {
    // Regression: the old idiom `"'"'"'` turned a'b into a"'b through bash.
    expect(await bashRoundTrip("a'b")).toBe("a'b");
  });

  it("preserves a JSON string value that embeds a single quote", async () => {
    const payload = JSON.stringify({ command: "printf 'x'", pid: "$$" });
    const reconstructed = await bashRoundTrip(payload);
    // Must remain parseable and structurally identical after a shell round-trip.
    expect(() => JSON.parse(reconstructed)).not.toThrow();
    expect(JSON.parse(reconstructed)).toEqual(JSON.parse(payload));
  });
});
