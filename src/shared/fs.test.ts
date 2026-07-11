import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { readJsonFile, writeJsonAtomic } from "./fs.js";

describe("readJsonFile", () => {
  let dir: string;

  beforeEach(async () => {
    dir = await fs.mkdtemp(path.join(os.tmpdir(), "bee-fs-"));
  });

  afterEach(async () => {
    await fs.rm(dir, { recursive: true, force: true });
  });

  it("returns a cloned fallback when the file is missing", async () => {
    const fallback = { count: 0, items: [] as string[] };
    const result = await readJsonFile(path.join(dir, "missing.json"), fallback);
    expect(result).toEqual(fallback);
    expect(result).not.toBe(fallback);
  });

  it("returns undefined fallback without cloning when the file is missing", async () => {
    const result = await readJsonFile<{ id: string } | undefined>(path.join(dir, "missing.json"), undefined);
    expect(result).toBeUndefined();
  });

  it("returns the fallback for an empty file", async () => {
    const file = path.join(dir, "empty.json");
    await fs.writeFile(file, "   \n", "utf8");
    expect(await readJsonFile(file, { ok: true })).toEqual({ ok: true });
  });

  it("throws on malformed JSON by default", async () => {
    const file = path.join(dir, "corrupt.json");
    await fs.writeFile(file, '{"partial": ', "utf8");
    await expect(readJsonFile(file, { ok: true })).rejects.toBeInstanceOf(SyntaxError);
  });

  it("returns the fallback on malformed JSON when tolerateParseErrors is set", async () => {
    const file = path.join(dir, "corrupt.json");
    await fs.writeFile(file, '{"partial": ', "utf8");
    const result = await readJsonFile(file, { ok: true }, { tolerateParseErrors: true });
    expect(result).toEqual({ ok: true });
  });

  it("still parses valid JSON when tolerance is enabled", async () => {
    const file = path.join(dir, "valid.json");
    await writeJsonAtomic(file, { status: "running", pid: 1234 });
    const result = await readJsonFile(file, undefined, { tolerateParseErrors: true });
    expect(result).toEqual({ status: "running", pid: 1234 });
  });
});
