import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { OperatorCliApp, parseSlashCommand } from "./app.js";

const tempDirs: string[] = [];

async function makeTempDir(): Promise<string> {
  const dir = await fs.mkdtemp(path.join(os.tmpdir(), "operator-cli-app-"));
  tempDirs.push(dir);
  return dir;
}

afterEach(async () => {
  await Promise.all(tempDirs.splice(0).map((dir) => fs.rm(dir, { recursive: true, force: true })));
});

describe("parseSlashCommand", () => {
  it("parses supported slash commands", () => {
    expect(parseSlashCommand("/status")).toEqual({ kind: "status" });
    expect(parseSlashCommand("/recall deploy bug")).toEqual({ kind: "recall", query: "deploy bug" });
    expect(parseSlashCommand("hello")).toBeUndefined();
  });
});

describe("OperatorCliApp", () => {
  it("dispatches status and runtime-backed commands", async () => {
    const rootDir = await makeTempDir();
    const app = new OperatorCliApp({ rootDir, cwd: rootDir, currentDate: "2026-05-25" });
    const session = await app.runtime.startSession({ title: "CLI test", cwd: rootDir, agentId: "operator-cli" });

    await app.runtime.recordTurn({
      sessionId: session.id,
      userText: "check deploy",
      assistantText: "deploy checked",
      trajectorySummary: "Deploy checked for recall",
    });

    const status = await app.dispatchSlashCommand({ kind: "status" }, session.id);
    expect(status).toContain(`session=${session.id}`);
    expect(status).toContain("instructionFiles=");

    const recall = await app.dispatchSlashCommand({ kind: "recall", query: "deploy" }, session.id);
    expect(recall).toContain("Deploy checked for recall");
  });

  it("runs executable skills from slash commands", async () => {
    const rootDir = await makeTempDir();
    const app = new OperatorCliApp({ rootDir, cwd: rootDir, currentDate: "2026-05-25" });
    const session = await app.runtime.startSession({ title: "CLI skill", cwd: rootDir, agentId: "operator-cli" });
    const recorded = await app.runtime.recordTurn({
      sessionId: session.id,
      userText: "summarize deploy",
      assistantText: "deploy summarized",
      trajectorySummary: "Deploy summary workflow for CLI skill",
    });
    if (!recorded.skillCandidate) {
      throw new Error("expected skill candidate");
    }
    await app.runtime.reviewSkillCandidate(recorded.skillCandidate.id, "reviewed", "reusable");
    const promoted = await app.runtime.promoteSkill(recorded.skillCandidate.id);
    if (!promoted) {
      throw new Error("expected promoted skill");
    }
    const executable = await app.runtime.createExecutableSkillFromPromoted({
      promotedSkillId: promoted.id,
      steps: [{ id: "summary", title: "Summarize", kind: "summary", content: "deploy summary" }],
    });
    if (!executable) {
      throw new Error("expected executable skill");
    }

    const output = await app.dispatchSlashCommand({ kind: "run-skill", skillId: executable.id }, session.id);
    expect(output).toContain("completed");
    expect(output).toContain("deploy summary");
  });
});
