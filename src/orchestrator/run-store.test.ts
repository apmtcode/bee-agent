import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { buildExecutionGraph } from "./execution-graph.js";
import { FileRunStore } from "./run-store.js";

const tempDirs: string[] = [];

async function makeTempDir(): Promise<string> {
  const dir = await fs.mkdtemp(path.join(os.tmpdir(), "run-store-"));
  tempDirs.push(dir);
  return dir;
}

afterEach(async () => {
  await Promise.all(tempDirs.splice(0).map((dir) => fs.rm(dir, { recursive: true, force: true })));
});

describe("FileRunStore", () => {
  it("creates runs, updates status, and builds an execution graph", async () => {
    const dir = await makeTempDir();
    const store = new FileRunStore(path.join(dir, "runs.json"));
    const parent = await store.create({ sessionId: "sess-1", title: "Parent", status: "running" });
    const child = await store.create({
      sessionId: "sess-1",
      title: "Child",
      parentRunId: parent.id,
      status: "pending",
    });

    await expect(store.update(child.id, { status: "completed" })).resolves.toMatchObject({ status: "completed" });
    await expect(store.listBySession("sess-1")).resolves.toHaveLength(2);
    await expect(store.listChildren(parent.id)).resolves.toEqual([expect.objectContaining({ id: child.id })]);

    const graph = buildExecutionGraph(await store.list());
    expect(graph).toHaveLength(1);
    expect(graph[0]?.run.id).toBe(parent.id);
    expect(graph[0]?.children[0]?.run.id).toBe(child.id);
  });
});
