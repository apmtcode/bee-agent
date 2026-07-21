/**
 * POSIX shell quoting helpers.
 *
 * These are the single source of truth for embedding arbitrary strings into
 * generated shell scripts (background-task and training launchers). Two
 * independent copies of this logic previously existed; one carried a subtle
 * single-quote-escaping bug (`"'"'"'` instead of `'"'"'`) that corrupted any
 * generated JSON payload whenever a command contained a single quote. Keeping
 * one tested implementation prevents that divergence from recurring.
 */

/**
 * Wrap `value` in single quotes so it is a single, literal shell word.
 *
 * A single-quoted POSIX string cannot itself contain a single quote, so each
 * embedded `'` is emitted as the canonical escape sequence `'"'"'` — close the
 * single-quoted run, add a double-quoted single quote, then reopen. Any byte
 * (newlines, `$`, backticks, `"`, backslashes) is preserved verbatim.
 *
 * @example
 * posixSingleQuote("a'b")            // => '\'a\'"\'"\'b\''  i.e.  'a'"'"'b'
 * posixSingleQuote("printf 'x'")     // => 'printf '"'"'x'"'"''
 */
export function posixSingleQuote(value: string): string {
  return `'${value.replaceAll(`'`, `'"'"'`)}'`;
}
