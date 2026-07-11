import fs from "node:fs";
import path from "node:path";

export async function ensureParentDir(filePath: string): Promise<void> {
  await fs.promises.mkdir(path.dirname(filePath), { recursive: true });
}

export interface ReadJsonFileOptions {
  /**
   * When true, a malformed/partially-written file is treated like a missing
   * one and the fallback is returned instead of throwing. Useful for state
   * files that a concurrently-running (or crashed-mid-write) process may leave
   * truncated. Defaults to false to preserve strict parsing for callers that
   * want corruption surfaced.
   */
  tolerateParseErrors?: boolean;
}

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
  try {
    const raw = await fs.promises.readFile(filePath, "utf8");
    if (!raw.trim()) {
      return cloneFallback(fallback);
    }
    return JSON.parse(raw) as T;
  } catch (error) {
    const isMissing = (error as NodeJS.ErrnoException).code === "ENOENT";
    const isParseError = error instanceof SyntaxError;
    if (isMissing || (isParseError && options.tolerateParseErrors)) {
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
