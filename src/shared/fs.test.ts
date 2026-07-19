import { mkdtemp, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { describe, expect, it } from "vitest";
import { readJsonFile, writeJsonAtomic } from "./fs.js";

async function makeTempDir(): Promise<string> {
  return await mkdtemp(path.join(tmpdir(), "bee-fs-"));
}

describe("readJsonFile", () => {
  it("reads well-formed JSON", async () => {
    const dir = await makeTempDir();
    const file = path.join(dir, "state.json");
    await writeJsonAtomic(file, { value: 42 });
    expect(await readJsonFile(file, undefined)).toEqual({ value: 42 });
  });

  it("returns the fallback for a missing file", async () => {
    const dir = await makeTempDir();
    expect(await readJsonFile(path.join(dir, "missing.json"), { ok: true })).toEqual({ ok: true });
  });

  it("returns a deep clone of the fallback (callers cannot mutate the shared default)", async () => {
    const dir = await makeTempDir();
    const fallback = { list: [1] };
    const first = await readJsonFile(path.join(dir, "missing.json"), fallback);
    (first as { list: number[] }).list.push(2);
    expect(fallback.list).toEqual([1]);
  });

  it("throws on malformed JSON by default (persistent stores must surface corruption)", async () => {
    const dir = await makeTempDir();
    const file = path.join(dir, "corrupt.json");
    await writeFile(file, '{"version":1,"pid":', "utf8");
    await expect(readJsonFile(file, undefined)).rejects.toBeInstanceOf(SyntaxError);
  });

  it("tolerates a transient half-written file when tolerateParseErrors is set", async () => {
    const dir = await makeTempDir();
    const file = path.join(dir, "volatile.json");
    // Simulates a non-atomic writer observed mid-write.
    await writeFile(file, '{"version":1,"status":"running","pid":', "utf8");
    expect(await readJsonFile(file, undefined, { tolerateParseErrors: true })).toBeUndefined();
  });

  it("still returns real data when tolerateParseErrors is set and the file is complete", async () => {
    const dir = await makeTempDir();
    const file = path.join(dir, "volatile.json");
    await writeJsonAtomic(file, { status: "completed" });
    expect(await readJsonFile(file, undefined, { tolerateParseErrors: true })).toEqual({ status: "completed" });
  });
});
