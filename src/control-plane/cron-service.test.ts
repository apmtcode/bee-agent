import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { OperatorCronService } from "./cron-service.js";

const tempDirs: string[] = [];

async function makeTempDir(): Promise<string> {
  const dir = await fs.mkdtemp(path.join(os.tmpdir(), "cron-service-"));
  tempDirs.push(dir);
  return dir;
}

afterEach(async () => {
  await Promise.all(tempDirs.splice(0).map((dir) => fs.rm(dir, { recursive: true, force: true })));
});

describe("OperatorCronService", () => {
  it("creates, lists, runs, and deletes cron jobs", async () => {
    const service = new OperatorCronService(await makeTempDir());
    const recurring = await service.createJob({ cron: "* * * * *", prompt: "do recurring" });
    const oneShot = await service.createJob({ cron: "* * * * *", prompt: "do once", recurring: false });

    await expect(service.listJobs()).resolves.toHaveLength(2);
    await expect(service.tick(new Date(Date.now() + 120_000))).resolves.toHaveLength(2);
    await expect(service.listRuns()).resolves.toHaveLength(2);

    const jobsAfterTick = await service.listJobs();
    expect(jobsAfterTick.map((job) => job.id)).toContain(recurring.id);
    expect(jobsAfterTick.map((job) => job.id)).not.toContain(oneShot.id);

    await expect(service.deleteJob(recurring.id)).resolves.toBe(true);
    await expect(service.listJobs()).resolves.toEqual([]);
  });
});
