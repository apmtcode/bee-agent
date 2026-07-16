import fs from "node:fs";
import path from "node:path";

export async function ensureParentDir(filePath: string): Promise<void> {
  await fs.promises.mkdir(path.dirname(filePath), { recursive: true });
}

function cloneFallback<T>(fallback: T): T {
  if (fallback === undefined) {
    return fallback;
  }
  return JSON.parse(JSON.stringify(fallback)) as T;
}

/**
 * Read and parse a JSON file. Writers in this codebase use {@link writeJsonAtomic}
 * (stage + rename), so a reader normally sees either the old or the new complete
 * file — never a torn one. As defense-in-depth against a foreign non-atomic writer
 * mid-truncate, a parse failure on a non-empty file is retried a bounded number of
 * times before surfacing, so a transient partial read does not crash a caller.
 */
export async function readJsonFile<T>(filePath: string, fallback: T, retries = 3): Promise<T> {
  let lastParseError: unknown;
  for (let attempt = 0; attempt <= retries; attempt += 1) {
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
      lastParseError = error;
      if (attempt < retries) {
        await new Promise((resolve) => setTimeout(resolve, 2 ** attempt));
      }
    }
  }
  throw lastParseError;
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
