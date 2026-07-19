import fs from "node:fs";
import path from "node:path";

export async function ensureParentDir(filePath: string): Promise<void> {
  await fs.promises.mkdir(path.dirname(filePath), { recursive: true });
}

export type ReadJsonFileOptions = {
  /**
   * When true, a malformed/partially-written file is treated the same as a
   * missing one (the fallback is returned) instead of throwing. Use this only
   * for *volatile* files that a live external process may be mid-writing
   * non-atomically (e.g. background-task / training execution-state files), so a
   * status or recovery read observing a half-written file does not crash. Do NOT
   * use it for persistent stores or config, where a parse error is a real fault
   * that should surface.
   */
  tolerateParseErrors?: boolean;
};

function cloneFallback<T>(fallback: T): T {
  if (fallback === undefined) {
    return fallback;
  }
  return JSON.parse(JSON.stringify(fallback)) as T;
}

export async function readJsonFile<T>(
  filePath: string,
  fallback: T,
  options: ReadJsonFileOptions = {},
): Promise<T> {
  let raw: string;
  try {
    raw = await fs.promises.readFile(filePath, "utf8");
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") {
      return cloneFallback(fallback);
    }
    throw error;
  }

  if (!raw.trim()) {
    return cloneFallback(fallback);
  }

  try {
    return JSON.parse(raw) as T;
  } catch (error) {
    if (options.tolerateParseErrors && error instanceof SyntaxError) {
      return cloneFallback(fallback);
    }
    throw error;
  }
}

let writeCounter = 0;

export async function writeJsonAtomic(filePath: string, value: unknown): Promise<void> {
  await ensureParentDir(filePath);
  writeCounter += 1;
  const tempPath = `${filePath}.${process.pid}.${writeCounter}.tmp`;
  await fs.promises.writeFile(tempPath, `${JSON.stringify(value, null, 2)}\n`, {
    encoding: "utf8",
    mode: 0o600,
  });
  await fs.promises.rename(tempPath, filePath);
}
