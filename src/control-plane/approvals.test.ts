import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { FileApprovalStore } from "./approval-store.js";
import { InMemoryApprovalManager } from "./approvals.js";

const tempDirs: string[] = [];

async function makeTempDir(): Promise<string> {
  const dir = await fs.mkdtemp(path.join(os.tmpdir(), "approval-store-"));
  tempDirs.push(dir);
  return dir;
}

afterEach(async () => {
  await Promise.all(tempDirs.splice(0).map((dir) => fs.rm(dir, { recursive: true, force: true })));
});

describe("InMemoryApprovalManager", () => {
  it("creates, lists, resolves, waits, and persists approvals", async () => {
    const dir = await makeTempDir();
    const approvals = new InMemoryApprovalManager(new FileApprovalStore(path.join(dir, "approvals.json")));
    await approvals.loadPersisted();
    const request = await approvals.create({
      title: "Run browser action",
      summary: "Click the deploy button",
      sessionId: "sess-1",
      timeoutMs: 5_000,
    });

    expect(approvals.list()).toEqual([request]);
    const waiter = approvals.waitForDecision(request.id);
    const resolution = await approvals.resolve(request.id, "approved", "operator");

    expect(resolution).toMatchObject({ id: request.id, decision: "approved", resolvedBy: "operator" });
    await expect(waiter).resolves.toMatchObject({ id: request.id, decision: "approved" });
    expect(approvals.list()).toEqual([]);

    const restored = new InMemoryApprovalManager(new FileApprovalStore(path.join(dir, "approvals.json")));
    await restored.loadPersisted();
    expect(restored.list()).toEqual([]);
  });
});
