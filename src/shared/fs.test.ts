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
  await Promise.all(
    tempDirs.splice(0).map((dir) => fs.rm(dir, { recursive: true, force: true, maxRetries: 5, retryDelay: 50 })),
  );
});

describe("readJsonFile / writeJsonAtomic", () => {
  it("round-trips a value written atomically", async () => {
    const dir = await makeTempDir();
    const file = path.join(dir, "nested", "state.json");
    await writeJsonAtomic(file, { version: 1, items: ["a", "b"] });
    await expect(readJsonFile(file, undefined)).resolves.toEqual({ version: 1, items: ["a", "b"] });
  });

  it("returns a cloned fallback for a missing or empty file", async () => {
    const dir = await makeTempDir();
    const fallback = { items: [] as string[] };
    const missing = await readJsonFile(path.join(dir, "missing.json"), fallback);
    expect(missing).toEqual({ items: [] });
    expect(missing).not.toBe(fallback); // defensive clone, not the shared reference

    const emptyPath = path.join(dir, "empty.json");
    await fs.writeFile(emptyPath, "   \n", "utf8");
    await expect(readJsonFile(emptyPath, fallback)).resolves.toEqual({ items: [] });
  });

  it("recovers from a torn read once a concurrent writer settles the file", async () => {
    const dir = await makeTempDir();
    const file = path.join(dir, "torn.json");
    // Simulate a non-atomic writer being observed mid-write: a truncated JSON
    // snapshot on disk, replaced by the complete document a moment later.
    await fs.writeFile(file, '{ "status": "run', "utf8");
    setTimeout(() => {
      void fs.writeFile(file, JSON.stringify({ status: "running", pid: 42 }), "utf8");
    }, 6);
    await expect(readJsonFile(file, undefined)).resolves.toEqual({ status: "running", pid: 42 });
  });

  it("throws when a file stays syntactically invalid past the retry budget", async () => {
    const dir = await makeTempDir();
    const file = path.join(dir, "corrupt.json");
    await fs.writeFile(file, "{ not json", "utf8");
    await expect(readJsonFile(file, undefined)).rejects.toThrow(SyntaxError);
  });

  it("never observes a torn read under concurrent atomic writers and readers", async () => {
    const dir = await makeTempDir();
    const file = path.join(dir, "contended.json");
    await writeJsonAtomic(file, { seq: 0 });
    const work: Promise<unknown>[] = [];
    for (let i = 1; i <= 40; i += 1) {
      work.push(writeJsonAtomic(file, { seq: i, payload: "x".repeat(200) }));
      work.push(readJsonFile(file, { seq: -1 }));
    }
    // All reads resolve to a complete document; none reject on a partial write.
    const results = await Promise.all(work);
    for (const result of results) {
      if (result && typeof result === "object" && "seq" in result) {
        expect(typeof (result as { seq: number }).seq).toBe("number");
      }
    }
  });
});
