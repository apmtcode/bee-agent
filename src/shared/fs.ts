import fs from "node:fs";
import path from "node:path";

export async function ensureParentDir(filePath: string): Promise<void> {
  await fs.promises.mkdir(path.dirname(filePath), { recursive: true });
}

/**
 * Number of times {@link readJsonFile} re-reads a file after a JSON parse
 * failure before giving up. A parse failure on an existing file is almost
 * always a *torn read* — the file is mid-rewrite by another process (e.g. the
 * background-task launch script) and we caught it between truncate and flush.
 * Retrying a few times with a tiny backoff lets the writer finish, turning a
 * spurious throw into a correct read. Genuine corruption still surfaces after
 * the retries are exhausted.
 */
const TORN_READ_RETRIES = 3;
const TORN_READ_BACKOFF_MS = 5;

function cloneFallback<T>(fallback: T): T {
  if (fallback === undefined) {
    return fallback;
  }
  return JSON.parse(JSON.stringify(fallback)) as T;
}

export async function readJsonFile<T>(filePath: string, fallback: T): Promise<T> {
  for (let attempt = 0; ; attempt += 1) {
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
      // An empty file is the transient state a truncate-then-write leaves
      // behind. Treat it as a torn read and retry before falling back.
      if (attempt < TORN_READ_RETRIES) {
        await delay(TORN_READ_BACKOFF_MS);
        continue;
      }
      return cloneFallback(fallback);
    }

    try {
      return JSON.parse(raw) as T;
    } catch (error) {
      if (error instanceof SyntaxError && attempt < TORN_READ_RETRIES) {
        await delay(TORN_READ_BACKOFF_MS);
        continue;
      }
      throw error;
    }
  }
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
