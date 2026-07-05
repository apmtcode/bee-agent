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
 * Read a JSON file that may be written concurrently by another (possibly
 * non-Node) process. Writers should still write atomically (temp file + rename)
 * so a reader only ever sees complete content; this helper is defense-in-depth:
 * on a transient parse failure it retries a few times, and if the file is still
 * unreadable it degrades to `fallback` instead of throwing, so a single corrupt
 * state file can never crash a recovery sweep. ENOENT/empty behave exactly like
 * {@link readJsonFile}.
 */
export async function readJsonFileResilient<T>(
  filePath: string,
  fallback: T,
  options: { attempts?: number; delayMs?: number } = {},
): Promise<T> {
  const attempts = Math.max(1, options.attempts ?? 3);
  const delayMs = Math.max(0, options.delayMs ?? 5);
  let lastError: unknown;
  for (let attempt = 0; attempt < attempts; attempt += 1) {
    try {
      return await readJsonFile(filePath, fallback);
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code === "ENOENT") {
        throw error;
      }
      lastError = error;
      if (attempt < attempts - 1 && delayMs > 0) {
        await new Promise((resolve) => setTimeout(resolve, delayMs));
      }
    }
  }
  void lastError;
  if (fallback === undefined) {
    return fallback;
  }
  return JSON.parse(JSON.stringify(fallback)) as T;
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
