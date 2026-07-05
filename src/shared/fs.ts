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
 * Like {@link readJsonFile} but also tolerates a malformed/partially-written
 * file by returning the fallback instead of throwing a `SyntaxError`.
 *
 * State files (background-task/training execution state) are written by a
 * detached shell process while other parts of the system may concurrently
 * read them. A reader that catches the file mid-write would otherwise crash
 * with a JSON parse error. For these transient, reconstructable files an
 * unreadable snapshot is best treated as "no state yet" so callers fall back
 * to reconciliation rather than propagating a hard failure.
 */
export async function readJsonFileResilient<T>(filePath: string, fallback: T): Promise<T> {
  try {
    return await readJsonFile<T>(filePath, fallback);
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
