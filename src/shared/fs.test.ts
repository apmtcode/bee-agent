import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { readJsonFile, writeJsonAtomic } from "./fs.js";

const tempDirs: string[] = [];

async function makeTempDir(): Promise<string> {
  const dir = await fs.promises.mkdtemp(path.join(os.tmpdir(), "bee-fs-"));
  tempDirs.push(dir);
  return dir;
}

afterEach(async () => {
  await Promise.all(
    tempDirs.splice(0).map((dir) => fs.promises.rm(dir, { recursive: true, force: true })),
  );
});

describe("readJsonFile", () => {
  it("returns a deep clone of the fallback when the file is missing", async () => {
    const dir = await makeTempDir();
    const fallback = { version: 1, tasks: [] as string[] };
    const first = await readJsonFile(path.join(dir, "missing.json"), fallback);
    expect(first).toEqual(fallback);
    first.tasks.push("mutated");
    // Fallback must not be mutated through the returned clone.
    expect(fallback.tasks).toEqual([]);
  });

  it("returns undefined fallback as-is for a missing file", async () => {
    const dir = await makeTempDir();
    await expect(readJsonFile(path.join(dir, "missing.json"), undefined)).resolves.toBeUndefined();
  });

  it("parses well-formed JSON", async () => {
    const dir = await makeTempDir();
    const file = path.join(dir, "state.json");
    await writeJsonAtomic(file, { status: "completed", pid: 42 });
    await expect(readJsonFile(file, undefined)).resolves.toEqual({ status: "completed", pid: 42 });
  });

  it("recovers when a concurrent writer flips a torn file into valid JSON before retries run out", async () => {
    const dir = await makeTempDir();
    const file = path.join(dir, "state.json");
    // Simulate a torn read: the file currently holds a truncated fragment.
    await fs.promises.writeFile(file, "{\"status\": \"comp", "utf8");
    // A concurrent writer finishes the atomic write shortly after.
    setTimeout(() => {
      void writeJsonAtomic(file, { status: "completed" });
    }, 8);
    await expect(readJsonFile(file, undefined)).resolves.toEqual({ status: "completed" });
  });

  it("recovers when an empty (mid-truncate) file is filled before retries run out", async () => {
    const dir = await makeTempDir();
    const file = path.join(dir, "state.json");
    await fs.promises.writeFile(file, "", "utf8");
    setTimeout(() => {
      void writeJsonAtomic(file, { status: "running" });
    }, 8);
    await expect(readJsonFile(file, { status: "fallback" })).resolves.toEqual({ status: "running" });
  });

  it("throws on genuinely corrupt JSON once retries are exhausted", async () => {
    const dir = await makeTempDir();
    const file = path.join(dir, "state.json");
    await fs.promises.writeFile(file, "{not json at all", "utf8");
    await expect(readJsonFile(file, undefined)).rejects.toBeInstanceOf(SyntaxError);
  });

  it("falls back for a persistently empty file once retries are exhausted", async () => {
    const dir = await makeTempDir();
    const file = path.join(dir, "state.json");
    await fs.promises.writeFile(file, "   \n", "utf8");
    await expect(readJsonFile(file, { version: 1 })).resolves.toEqual({ version: 1 });
  });
});
