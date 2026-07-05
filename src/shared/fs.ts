import fs from "node:fs";
import path from "node:path";

export async function ensureParentDir(filePath: string): Promise<void> {
  await fs.promises.mkdir(path.dirname(filePath), { recursive: true });
}

export async function readJsonFile<T>(filePath: string, fallback: T): Promise<T> {
  try {
    const raw = await fs.promises.readFile(filePath, "utf8");
    if (!raw.trim()) {
      if (fallback === undefined) {
        return fallback;
      }
      return JSON.parse(JSON.stringify(fallback)) as T;
    }
    return JSON.parse(raw) as T;
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") {
      if (fallback === undefined) {
        return fallback;
      }
      return JSON.parse(JSON.stringify(fallback)) as T;
    }
    throw error;
  }
}

/**
 * Like {@link readJsonFile}, but also returns the fallback when the file
 * contains malformed/partial JSON instead of throwing. Use this for files that
 * may be written by an external process with a non-atomic writer, where a
 * concurrent read can observe a truncated document. Callers that own the
 * writer (and write atomically) should prefer the strict {@link readJsonFile}
 * so genuine corruption surfaces instead of being silently swallowed.
 */
export async function tryReadJsonFile<T>(filePath: string, fallback: T): Promise<T> {
  try {
    return await readJsonFile(filePath, fallback);
  } catch (error) {
    if (error instanceof SyntaxError) {
      if (fallback === undefined) {
        return fallback;
      }
      return JSON.parse(JSON.stringify(fallback)) as T;
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
