import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { readJsonFile, writeJsonAtomic } from "./fs.js";

describe("readJsonFile", () => {
  let dir: string;

  beforeEach(async () => {
    dir = await fs.promises.mkdtemp(path.join(os.tmpdir(), "bee-fs-"));
  });

  afterEach(async () => {
    await fs.promises.rm(dir, { recursive: true, force: true });
  });

  it("parses valid JSON", async () => {
    const file = path.join(dir, "state.json");
    await writeJsonAtomic(file, { id: "abc", status: "running" });
    await expect(readJsonFile(file, undefined)).resolves.toEqual({ id: "abc", status: "running" });
  });

  it("returns a cloned fallback for a missing file without mutating the original", async () => {
    const fallback = { tasks: [] as string[] };
    const result = await readJsonFile(path.join(dir, "missing.json"), fallback);
    expect(result).toEqual({ tasks: [] });
    // The returned value must be a clone — mutating it must not touch the fallback.
    result.tasks.push("mutated");
    expect(fallback.tasks).toEqual([]);
  });

  it("returns fallback for an empty file", async () => {
    const file = path.join(dir, "empty.json");
    await fs.promises.writeFile(file, "");
    await expect(readJsonFile(file, { ok: true })).resolves.toEqual({ ok: true });
  });

  it("heals a torn read: retries until a concurrent non-atomic writer finishes", async () => {
    const file = path.join(dir, "torn.json");
    // Simulate a truncate-then-write writer caught mid-write: a partial,
    // unparseable JSON fragment is on disk when the read begins.
    await fs.promises.writeFile(file, '{"id":"abc","status":"run');
    const readPromise = readJsonFile<{ id: string; status: string }>(file, { id: "", status: "" });
    // The writer completes shortly after; the bounded retry must observe it.
    setTimeout(() => {
      fs.writeFileSync(file, JSON.stringify({ id: "abc", status: "completed" }));
    }, 2);
    await expect(readPromise).resolves.toEqual({ id: "abc", status: "completed" });
  });

  it("throws for a persistently corrupt file after exhausting retries", async () => {
    const file = path.join(dir, "corrupt.json");
    await fs.promises.writeFile(file, "{not valid json ]]]");
    await expect(readJsonFile(file, undefined)).rejects.toThrow(SyntaxError);
  });
});
