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
 * Read and parse a JSON file. Writers in this project write atomically
 * (temp file + rename), so a well-behaved reader should never observe a torn
 * document. As defense-in-depth against any non-atomic writer (including
 * external processes), a `SyntaxError` is retried a few times with a short
 * backoff before propagating — a genuinely corrupt file still surfaces the
 * error, while a transient mid-write read resolves on retry.
 */
export async function readJsonFile<T>(filePath: string, fallback: T): Promise<T> {
  const maxAttempts = 4;
  for (let attempt = 1; ; attempt += 1) {
    try {
      const raw = await fs.promises.readFile(filePath, "utf8");
      if (!raw.trim()) {
        return cloneFallback(fallback);
      }
      return JSON.parse(raw) as T;
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code === "ENOENT") {
        return cloneFallback(fallback);
      }
      if (error instanceof SyntaxError && attempt < maxAttempts) {
        await new Promise((resolve) => setTimeout(resolve, attempt * 5));
        continue;
      }
      throw error;
    }
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
