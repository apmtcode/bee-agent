/**
 * POSIX-safe single-quoting for embedding an arbitrary string as one shell word.
 *
 * The value is wrapped in single quotes; any embedded single quote is escaped
 * with the canonical `'\''` sequence (close-quote, backslash-escaped quote,
 * reopen-quote). This is robust for *any* input — including JSON payloads whose
 * values themselves contain single quotes — so a launch script can carry a
 * command verbatim without corrupting the surrounding shell or JSON.
 *
 * Historically this helper was duplicated across the background-task and
 * training launch-script renderers, and one copy used an incorrect escape
 * (`"'"'"'`) that injected a spurious double-quote for every embedded single
 * quote — producing invalid JSON state files whenever a task command contained
 * a single quote (e.g. `printf 'hi'`). Centralizing the correct implementation
 * here prevents that class of bug from recurring.
 */
export function shellSingleQuote(value: string): string {
  return `'${value.replaceAll("'", `'\\''`)}'`;
}
