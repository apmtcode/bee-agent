import fs from "node:fs";
import path from "node:path";

export async function ensureParentDir(filePath: string): Promise<void> {
  await fs.promises.mkdir(path.dirname(filePath), { recursive: true });
}

export type ReadJsonFileOptions = {
  /**
   * How many extra attempts to make when a non-empty file fails to parse. A
   * parse failure on a file that exists is almost always a concurrent partial
   * write — state files written by external shell/python launch scripts are
   * not atomic — so we re-read briefly before giving up. Genuine corruption
   * still throws after the final attempt. Default: 4 (5 attempts total).
   */
  parseRetries?: number;
};

export async function readJsonFile<T>(
  filePath: string,
  fallback: T,
  options: ReadJsonFileOptions = {},
): Promise<T> {
  const maxAttempts = Math.max(1, (options.parseRetries ?? 4) + 1);
  for (let attempt = 0; attempt < maxAttempts; attempt++) {
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
    } catch (parseError) {
      if (attempt < maxAttempts - 1) {
        await delay(2 * (attempt + 1));
        continue;
      }
      throw parseError;
    }
  }
  // Unreachable: the loop either returns or throws on the final attempt.
  return cloneFallback(fallback);
}

function cloneFallback<T>(fallback: T): T {
  if (fallback === undefined) {
    return fallback;
  }
  return JSON.parse(JSON.stringify(fallback)) as T;
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
