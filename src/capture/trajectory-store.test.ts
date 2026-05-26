import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { buildTrajectorySpan } from "./trajectory.js";
import { FileTrajectoryStore } from "./trajectory-store.js";

const tempDirs: string[] = [];

async function makeTempDir(): Promise<string> {
  const dir = await fs.mkdtemp(path.join(os.tmpdir(), "trajectory-store-"));
  tempDirs.push(dir);
  return dir;
}

afterEach(async () => {
  await Promise.all(tempDirs.splice(0).map((dir) => fs.rm(dir, { recursive: true, force: true })));
});

describe("FileTrajectoryStore", () => {
  it("persists trajectories by session", async () => {
    const dir = await makeTempDir();
    const filePath = path.join(dir, "trajectories.json");
    const store = new FileTrajectoryStore(filePath);

    const first = buildTrajectorySpan({
      id: "traj-1",
      sessionId: "sess-1",
      outcome: { status: "success", summary: "done" },
    });
    const second = buildTrajectorySpan({
      id: "traj-2",
      sessionId: "sess-2",
      outcome: { status: "success", summary: "other" },
    });

    await store.add(first);
    await store.add(second);

    await expect(store.list()).resolves.toHaveLength(2);
    await expect(store.listBySession("sess-1")).resolves.toEqual([first]);
    await expect(new FileTrajectoryStore(filePath).get("traj-2")).resolves.toEqual(second);
  });

  it("stores reviewed redactions and lists approved trajectories", async () => {
    const dir = await makeTempDir();
    const filePath = path.join(dir, "trajectories.json");
    const store = new FileTrajectoryStore(filePath);

    const trajectory = buildTrajectorySpan({
      id: "traj-reviewed",
      sessionId: "sess-1",
      observations: [{ kind: "observation", ts: 1, source: "browser", summary: "opened billing" }],
      actions: [{ kind: "action", ts: 2, tool: "browser", summary: "clicked pay" }],
    });
    await store.add(trajectory);

    await expect(
      store.review(trajectory.id, {
        status: "approved",
        reviewedBy: "reviewer",
        reviewNote: "safe subset",
        redactedObservations: [{ ts: 11, source: "browser", summary: "opened reviewed screen" }],
        redactedActions: [{ ts: 12, tool: "browser", summary: "clicked reviewed button" }],
        redactedTranscript: [{ ts: 13, role: "assistant", content: "reviewed transcript" }],
      }),
    ).resolves.toMatchObject({
      id: trajectory.id,
      review: {
        status: "approved",
        reviewedBy: "reviewer",
        reviewNote: "safe subset",
        reviewedAt: expect.any(String),
        redactedObservations: [{ ts: 11, source: "browser", summary: "opened reviewed screen" }],
        redactedActions: [{ ts: 12, tool: "browser", summary: "clicked reviewed button" }],
        redactedTranscript: [{ ts: 13, role: "assistant", content: "reviewed transcript" }],
      },
    });

    await expect(store.listReviewed()).resolves.toEqual([
      expect.objectContaining({ id: trajectory.id, review: expect.objectContaining({ status: "approved" }) }),
    ]);
    await expect(store.listReviewed("rejected")).resolves.toEqual([]);
  });
});
