/**
 * POSIX-safe shell quoting helpers.
 *
 * These are shared by every subsystem that renders a shell launch script
 * (background tasks, local training runners). Keeping a single implementation
 * here prevents the classes of bug where two hand-rolled copies of the same
 * quoting logic silently diverge — a malformed single-quote escape corrupts the
 * JSON state payloads those scripts write, which then fail to parse on recovery.
 */

/**
 * Wrap `value` so it is a single, literal shell token.
 *
 * Every embedded single quote is escaped using the canonical POSIX idiom:
 * close the quoted run, emit an escaped quote inside double quotes, then reopen
 * the quoted run — i.e. `'` becomes `'"'"'`. The result is safe to interpolate
 * into a `bash`/`sh` command line and round-trips any byte sequence, including
 * quotes, newlines, `$`, and backslashes.
 */
export function singleQuote(value: string): string {
  return `'${value.replaceAll("'", `'"'"'`)}'`;
}
