/**
 * Helpers for the detached launch scripts used by the background-task runner and
 * the local training runner. Both scripts persist a JSON *state* file that other
 * processes (recovery, sync, status polling) read concurrently while the script
 * is still running.
 *
 * The critical invariant is that those state writes must be **atomic**: a reader
 * must never observe a truncated or half-written file. A plain shell redirect
 * (`> state.json`) or Python `write_text` truncates the target first, so a read
 * landing mid-write yields torn bytes and crashes `JSON.parse`. Writing to a
 * unique temp file and then `rename(2)`/`os.replace`-ing it into place makes the
 * swap atomic on POSIX, mirroring `writeJsonAtomic` in ./fs.ts.
 */

/** POSIX single-quote a value for safe interpolation into a shell script. */
export function shellQuote(value: string): string {
  return `'${value.replaceAll(`'`, `'"'"'`)}'`;
}

/**
 * Render shell lines that run `producer` (a command emitting the state JSON on
 * stdout) and write its output atomically to `quotedStatePath` (already
 * shell-quoted): the bytes go to a PID-unique temp file, then `mv -f` renames it
 * into place. rename(2) is atomic, so concurrent readers see either the old
 * complete file or the new complete file — never a partial one.
 */
export function renderAtomicShellStateWrite(quotedStatePath: string, producer: string): string[] {
  return [
    `__state_tmp=${quotedStatePath}.$$.tmp`,
    `${producer} > "$__state_tmp"`,
    `mv -f "$__state_tmp" ${quotedStatePath}`,
  ];
}

/**
 * Render an inline Python program (body only — the caller wraps it in a heredoc)
 * that loads the current state file, applies the terminal status, and writes it
 * back **atomically** via a temp file + `os.replace`. `errorExpression` is a
 * Python expression evaluated for the `error` field when `status === "failed"`
 * (it may reference `exit_code`); for `"completed"` the error is set to `None`.
 */
export function renderAtomicPythonStateWriter(
  status: "completed" | "failed",
  errorExpression: string,
): string[] {
  return [
    "import json",
    "import os",
    "import pathlib",
    "import sys",
    "state_path = pathlib.Path(sys.argv[1])",
    "pid = int(sys.argv[2])",
    "timestamp = sys.argv[3]",
    "exit_code = int(sys.argv[4])",
    "state = json.loads(state_path.read_text())",
    `state['status'] = '${status}'`,
    "state['pid'] = pid",
    "state['updatedAt'] = timestamp",
    "state['completedAt'] = timestamp",
    "state['exitCode'] = exit_code",
    `state['error'] = ${status === "completed" ? "None" : errorExpression}`,
    "tmp_path = state_path.with_name(state_path.name + f'.{pid}.tmp')",
    "tmp_path.write_text(json.dumps(state, indent=2) + '\\n')",
    "os.replace(tmp_path, state_path)",
  ];
}
