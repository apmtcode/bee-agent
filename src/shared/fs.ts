import fs from "node:fs";
import path from "node:path";

export async function ensureParentDir(filePath: string): Promise<void> {
  await fs.promises.mkdir(path.dirname(filePath), { recursive: true });
}

export interface ReadJsonFileOptions {
  /**
   * Number of extra attempts to make when the file's contents fail to parse as
   * JSON. State files can be written by external processes we launch (e.g. the
   * background-task runner). Writes are made atomic via temp-file + rename, but
   * a defensive bounded retry protects readers from any residual partial-write
   * window on platforms where rename atomicity is weaker. Defaults to 0.
   */
  parseRetries?: number;
  /** Delay in milliseconds between parse retries. Defaults to 5ms. */
  parseRetryDelayMs?: number;
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
  const parseRetries = Math.max(0, options.parseRetries ?? 0);
  const parseRetryDelayMs = options.parseRetryDelayMs ?? 5;
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
      return cloneFallback(fallback);
    }
    try {
      return JSON.parse(raw) as T;
    } catch (error) {
      if (!(error instanceof SyntaxError) || attempt >= parseRetries) {
        throw error;
      }
      await new Promise((resolve) => setTimeout(resolve, parseRetryDelayMs));
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
