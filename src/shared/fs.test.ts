import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { readJsonFile, writeJsonAtomic } from "./fs.js";

const tempDirs: string[] = [];

async function makeTempDir(): Promise<string> {
  const dir = await fs.mkdtemp(path.join(os.tmpdir(), "shared-fs-"));
  tempDirs.push(dir);
  return dir;
}

afterEach(async () => {
  await Promise.all(tempDirs.splice(0).map((dir) => fs.rm(dir, { recursive: true, force: true })));
});

describe("readJsonFile", () => {
  it("returns a cloned fallback when the file is missing", async () => {
    const dir = await makeTempDir();
    const fallback = { tasks: [] as string[] };
    const value = await readJsonFile(path.join(dir, "missing.json"), fallback);
    expect(value).toEqual(fallback);
    value.tasks.push("x");
    expect(fallback.tasks).toEqual([]);
  });

  it("returns a cloned fallback for an empty/whitespace file", async () => {
    const dir = await makeTempDir();
    const filePath = path.join(dir, "empty.json");
    await fs.writeFile(filePath, "   \n", "utf8");
    expect(await readJsonFile(filePath, { ok: true })).toEqual({ ok: true });
  });

  it("round-trips a value written atomically", async () => {
    const dir = await makeTempDir();
    const filePath = path.join(dir, "state.json");
    await writeJsonAtomic(filePath, { version: 1, items: [1, 2, 3] });
    expect(await readJsonFile(filePath, {})).toEqual({ version: 1, items: [1, 2, 3] });
  });

  it("recovers when a torn partial write is repaired before retries are exhausted", async () => {
    const dir = await makeTempDir();
    const filePath = path.join(dir, "torn.json");
    // Simulate a foreign non-atomic writer: file currently holds a truncated,
    // unparseable fragment, then becomes valid JSON shortly after.
    await fs.writeFile(filePath, '{"version":1,"status":"run', "utf8");
    setTimeout(() => {
      void fs.writeFile(filePath, '{"version":1,"status":"running"}', "utf8");
    }, 5);
    expect(await readJsonFile(filePath, undefined, 5)).toEqual({ version: 1, status: "running" });
  });

  it("throws the parse error when a file stays corrupt past all retries", async () => {
    const dir = await makeTempDir();
    const filePath = path.join(dir, "corrupt.json");
    await fs.writeFile(filePath, "{not valid json", "utf8");
    await expect(readJsonFile(filePath, undefined, 2)).rejects.toBeInstanceOf(SyntaxError);
  });
});
