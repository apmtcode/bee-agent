import fs from "node:fs";
import path from "node:path";

export async function ensureParentDir(filePath: string): Promise<void> {
  await fs.promises.mkdir(path.dirname(filePath), { recursive: true });
}

/**
 * Number of times {@link readJsonFile} re-reads a file whose contents fail to
 * parse as JSON. A parse failure on an existing, non-empty file almost always
 * means a *torn read*: a concurrent writer that does not write atomically
 * (e.g. a detached shell/python process truncating-then-writing its state file)
 * was observed mid-write. Such a window closes within milliseconds, so a short
 * bounded retry turns a spurious crash into a correct read. A genuinely corrupt
 * file still surfaces the SyntaxError after the retries are exhausted.
 */
const TORN_READ_RETRIES = 5;
const TORN_READ_DELAY_MS = 4;

function cloneFallback<T>(fallback: T): T {
  if (fallback === undefined) {
    return fallback;
  }
  return JSON.parse(JSON.stringify(fallback)) as T;
}

function delay(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

export async function readJsonFile<T>(filePath: string, fallback: T): Promise<T> {
  let lastParseError: unknown;
  for (let attempt = 0; attempt <= TORN_READ_RETRIES; attempt += 1) {
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
      // An empty file is the transient state left by a truncate-then-write.
      // Retry briefly in case a writer is mid-write; fall back once settled.
      if (attempt < TORN_READ_RETRIES) {
        await delay(TORN_READ_DELAY_MS);
        continue;
      }
      return cloneFallback(fallback);
    }
    try {
      return JSON.parse(raw) as T;
    } catch (error) {
      lastParseError = error;
      if (attempt < TORN_READ_RETRIES) {
        await delay(TORN_READ_DELAY_MS);
        continue;
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
