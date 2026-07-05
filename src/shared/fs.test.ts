import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { readJsonFile, writeJsonAtomic } from "./fs.js";

let dir: string;

beforeEach(async () => {
  dir = await fs.promises.mkdtemp(path.join(os.tmpdir(), "bee-fs-"));
});

afterEach(async () => {
  await fs.promises.rm(dir, { recursive: true, force: true });
});

describe("readJsonFile", () => {
  it("reads a well-formed JSON file", async () => {
    const file = path.join(dir, "state.json");
    await writeJsonAtomic(file, { status: "running", pid: 42 });
    expect(await readJsonFile(file, undefined)).toEqual({ status: "running", pid: 42 });
  });

  it("returns a cloned fallback when the file is missing", async () => {
    const fallback = { tasks: [] as string[] };
    const result = await readJsonFile(path.join(dir, "missing.json"), fallback);
    expect(result).toEqual({ tasks: [] });
    expect(result).not.toBe(fallback);
  });

  it("returns the fallback for an empty file", async () => {
    const file = path.join(dir, "empty.json");
    await fs.promises.writeFile(file, "");
    expect(await readJsonFile(file, { ok: true })).toEqual({ ok: true });
  });

  it("recovers when a concurrent partial write is completed mid-read", async () => {
    const file = path.join(dir, "state.json");
    // Simulate a non-atomic external writer: truncate to a partial JSON payload,
    // then complete it shortly after — exactly the launch-script state-file race.
    await fs.promises.writeFile(file, '{"status":"run');
    const complete = { status: "completed", pid: 7 };
    setTimeout(() => {
      void fs.promises.writeFile(file, JSON.stringify(complete));
    }, 4);

    expect(await readJsonFile(file, undefined)).toEqual(complete);
  });

  it("throws on genuinely corrupt JSON after exhausting retries", async () => {
    const file = path.join(dir, "corrupt.json");
    await fs.promises.writeFile(file, "{not json at all");
    await expect(readJsonFile(file, undefined, { parseRetries: 1 })).rejects.toThrow();
  });
});
