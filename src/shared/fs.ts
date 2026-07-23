import fs from "node:fs";
import path from "node:path";

export async function ensureParentDir(filePath: string): Promise<void> {
  await fs.promises.mkdir(path.dirname(filePath), { recursive: true });
}

// A concurrent writer that truncates-and-rewrites a file in place (e.g. a
// detached shell/python background-task process using `>` redirection) can be
// observed mid-write by a reader, yielding a syntactically-invalid JSON
// snapshot. Because such writes complete in well under a millisecond, a small
// bounded retry lets the reader re-read the settled file instead of crashing
// the caller with a spurious SyntaxError.
const TORN_READ_RETRIES = 5;
const TORN_READ_DELAY_MS = 4;

function cloneFallback<T>(fallback: T): T {
  if (fallback === undefined) {
    return fallback;
  }
  return JSON.parse(JSON.stringify(fallback)) as T;
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
      return cloneFallback(fallback);
    }
    try {
      return JSON.parse(raw) as T;
    } catch (error) {
      if (!(error instanceof SyntaxError)) {
        throw error;
      }
      // Likely a torn read of an in-progress non-atomic write; re-read shortly.
      lastParseError = error;
      if (attempt < TORN_READ_RETRIES) {
        await delay(TORN_READ_DELAY_MS);
      }
    }
  }
  throw lastParseError;
}

function delay(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
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
