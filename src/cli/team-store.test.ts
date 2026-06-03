import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { FileOperatorCliTeamStore } from "./team-store.js";

const tempDirs: string[] = [];

async function makeTempDir(): Promise<string> {
  const dir = await fs.mkdtemp(path.join(os.tmpdir(), "operator-cli-team-store-"));
  tempDirs.push(dir);
  return dir;
}

afterEach(async () => {
  await Promise.all(tempDirs.splice(0).map((dir) => fs.rm(dir, { recursive: true, force: true, maxRetries: 5, retryDelay: 50 })));
});

describe("FileOperatorCliTeamStore", () => {
  it("creates lists and deletes teams", async () => {
    const rootDir = await makeTempDir();
    const store = new FileOperatorCliTeamStore(rootDir);

    const created = await store.create({ name: "reviewers", description: "Review team", taskSessionId: "session-reviewers" });
    expect(created.name).toBe("reviewers");
    expect(created.description).toBe("Review team");
    expect(created.taskSessionId).toBe("session-reviewers");

    await expect(store.get("reviewers")).resolves.toEqual(
      expect.objectContaining({ name: "reviewers", description: "Review team", taskSessionId: "session-reviewers" }),
    );

    await expect(store.list()).resolves.toEqual([
      expect.objectContaining({ name: "reviewers", description: "Review team", taskSessionId: "session-reviewers" }),
    ]);

    const deleted = await store.delete("reviewers");
    expect(deleted).toMatchObject({ name: "reviewers" });
    await expect(store.list()).resolves.toEqual([]);
  });

  it("keeps team names unique", async () => {
    const rootDir = await makeTempDir();
    const store = new FileOperatorCliTeamStore(rootDir);

    const first = await store.create({ name: "operators" });
    const second = await store.create({ name: "operators", description: "Duplicate" });
    expect(second.id).toBe(first.id);
    await expect(store.list()).resolves.toHaveLength(1);
  });

  it("backfills a missing task session id for an existing team", async () => {
    const rootDir = await makeTempDir();
    const store = new FileOperatorCliTeamStore(rootDir);

    const first = await store.create({ name: "operators" });
    const second = await store.create({ name: "operators", taskSessionId: "session-operators" });
    expect(second.id).toBe(first.id);
    expect(second.taskSessionId).toBe("session-operators");
    await expect(store.get("operators")).resolves.toEqual(expect.objectContaining({ taskSessionId: "session-operators" }));
  });
});
