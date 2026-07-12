/**
 * POSIX-safe shell quoting helpers.
 *
 * These are shared by every code path that renders an executable shell launch
 * script (background tasks, local training runner). Keeping a single
 * implementation avoids the class of bug where one copy escapes single quotes
 * incorrectly and silently corrupts the generated script.
 */

/**
 * Wrap an arbitrary string as a single POSIX shell word.
 *
 * The value is wrapped in single quotes; any embedded single quote is emitted
 * as the canonical `'"'"'` sequence (close the single-quoted span, emit a
 * double-quoted `'`, then reopen the single-quoted span). Passing the result to
 * `printf '%s'`/`eval` reproduces the original bytes exactly, including quotes,
 * newlines, and shell metacharacters.
 */
export function shellQuote(value: string): string {
  return `'${value.replaceAll("'", `'"'"'`)}'`;
}

/**
 * Quote a full argv into a single space-separated shell command string. Each
 * argument is individually {@link shellQuote}d so the reconstructed command
 * preserves word boundaries regardless of embedded whitespace or quotes.
 */
export function shellQuoteCommand(argv: readonly string[]): string {
  return argv.map((arg) => shellQuote(arg)).join(" ");
}
