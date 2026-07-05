import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { readJsonFile, readJsonFileResilient, writeJsonAtomic } from "./fs.js";

const tempDirs: string[] = [];

async function makeTempDir(): Promise<string> {
  const dir = await fs.mkdtemp(path.join(os.tmpdir(), "shared-fs-"));
  tempDirs.push(dir);
  return dir;
}

afterEach(async () => {
  await Promise.all(tempDirs.splice(0).map((dir) => fs.rm(dir, { recursive: true, force: true })));
});

describe("readJsonFileResilient", () => {
  it("reads well-formed JSON like readJsonFile", async () => {
    const dir = await makeTempDir();
    const file = path.join(dir, "state.json");
    await writeJsonAtomic(file, { status: "running", pid: 42 });

    await expect(readJsonFileResilient(file, undefined)).resolves.toEqual({ status: "running", pid: 42 });
  });

  it("returns the fallback for a missing file", async () => {
    const dir = await makeTempDir();
    const file = path.join(dir, "missing.json");

    await expect(readJsonFileResilient(file, { status: "unknown" })).resolves.toEqual({ status: "unknown" });
    await expect(readJsonFileResilient(file, undefined)).resolves.toBeUndefined();
  });

  it("tolerates a malformed / partially-written file instead of throwing", async () => {
    const dir = await makeTempDir();
    const file = path.join(dir, "partial.json");
    // Simulate a reader catching a non-atomic writer mid-write: a truncated
    // JSON snapshot on disk. The strict reader throws; the resilient one does
    // not, so a transient partial read never crashes state reconciliation.
    await fs.writeFile(file, '{"status":"running","pid":4', "utf8");

    await expect(readJsonFile(file, undefined)).rejects.toBeInstanceOf(SyntaxError);
    await expect(readJsonFileResilient(file, undefined)).resolves.toBeUndefined();
    await expect(readJsonFileResilient(file, { status: "unknown" })).resolves.toEqual({ status: "unknown" });
  });

  it("returns an independent clone of an object fallback", async () => {
    const dir = await makeTempDir();
    const file = path.join(dir, "partial.json");
    await fs.writeFile(file, "{ not json", "utf8");
    const fallback = { retries: 0 };

    const first = await readJsonFileResilient(file, fallback);
    (first as { retries: number }).retries = 5;
    const second = await readJsonFileResilient(file, fallback);

    expect(second).toEqual({ retries: 0 });
    expect(fallback.retries).toBe(0);
  });
});
