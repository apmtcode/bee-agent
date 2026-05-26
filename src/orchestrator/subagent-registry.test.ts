import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { FileSubagentRegistry } from "./subagent-registry.js";

const tempDirs: string[] = [];

async function makeTempDir(): Promise<string> {
  const dir = await fs.mkdtemp(path.join(os.tmpdir(), "subagent-registry-"));
  tempDirs.push(dir);
  return dir;
}

afterEach(async () => {
  await Promise.all(tempDirs.splice(0).map((dir) => fs.rm(dir, { recursive: true, force: true })));
});

describe("FileSubagentRegistry", () => {
  it("registers subagents, updates status, and restores active entries", async () => {
    const rootDir = await makeTempDir();
    const registry = new FileSubagentRegistry(rootDir);
    const entry = await registry.register({
      sessionId: "sess-parent",
      parentRunId: "run-parent",
      childSessionId: "sess-child",
      title: "Investigate deploy",
    });

    expect(entry.parentRunId).toBe("run-parent");
    await expect(registry.updateStatus(entry.runId, "running")).resolves.toMatchObject({ status: "running" });
    await expect(registry.listByParent("run-parent")).resolves.toEqual([
      expect.objectContaining({ runId: entry.runId, childSessionId: "sess-child" }),
    ]);

    const restored = new FileSubagentRegistry(rootDir);
    await expect(restored.listActive()).resolves.toEqual([
      expect.objectContaining({ runId: entry.runId, status: "running" }),
    ]);
    await expect(restored.getRun(entry.runId)).resolves.toMatchObject({ parentRunId: "run-parent" });
  });
});
