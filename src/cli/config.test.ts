import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import {
  OperatorCliConfigLoader,
  deepMergeConfig,
  resolveOperatorCliExecutionConfig,
  resolveOperatorCliStatusLineConfig,
  setProjectPermissionModeConfig,
  setProjectStatusLineConfig,
} from "./config.js";

const tempDirs: string[] = [];

async function makeTempDir(): Promise<string> {
  const dir = await fs.mkdtemp(path.join(os.tmpdir(), "operator-cli-config-"));
  tempDirs.push(dir);
  return dir;
}

afterEach(async () => {
  await Promise.all(tempDirs.splice(0).map((dir) => fs.rm(dir, { recursive: true, force: true })));
});

describe("OperatorCliConfigLoader", () => {
  it("loads user, project, and local settings with precedence", async () => {
    const root = await makeTempDir();
    const cwd = path.join(root, "project");
    const home = path.join(root, "home", ".claude");
    await fs.mkdir(path.join(cwd, ".claude"), { recursive: true });
    await fs.mkdir(home, { recursive: true });

    await fs.writeFile(path.join(home, "settings.json"), '{"model":"sonnet","env":{"A":"1"},"hooks":{"PreToolUse":["base"]}}');
    await fs.writeFile(path.join(cwd, ".claude", "settings.json"), '{"env":{"B":"2"},"hooks":{"PostToolUse":["project"]}}');
    await fs.writeFile(path.join(cwd, ".claude", "settings.local.json"), '{"model":"opus","permissionMode":"acceptEdits"}');

    const loaded = await new OperatorCliConfigLoader(cwd, home).load();

    expect(loaded.loadedEntries.map((entry) => entry.source)).toEqual(["user", "project", "local"]);
    expect(loaded.merged).toMatchObject({
      model: "opus",
      permissionMode: "acceptEdits",
      env: { A: "1", B: "2" },
      hooks: { PreToolUse: ["base"], PostToolUse: ["project"] },
    });
  });

  it("deep merges nested config objects", () => {
    expect(
      deepMergeConfig(
        { env: { A: "1" }, hooks: { PreToolUse: ["base"] } },
        { env: { B: "2" }, hooks: { PostToolUse: ["project"] } },
      ),
    ).toEqual({ env: { A: "1", B: "2" }, hooks: { PreToolUse: ["base"], PostToolUse: ["project"] } });
  });

  it("extracts execution policy settings from merged config", () => {
    expect(
      resolveOperatorCliExecutionConfig({
        model: "opus",
        permissionMode: "acceptEdits",
        hooks: {
          PreToolUse: ["echo pre-tool"],
          PreCommand: ["echo pre-command"],
          PostCommand: ["echo post"],
          SessionStart: ["echo session-start"],
          ApprovalResolved: ["echo approval-resolved"],
          Notification: ["echo unsupported"],
        },
      }),
    ).toEqual({
      permissionMode: "acceptEdits",
      hooks: {
        PreToolUse: ["echo pre-tool"],
        PreCommand: ["echo pre-command"],
        PostCommand: ["echo post"],
        SessionStart: ["echo session-start"],
        SessionEnd: [],
        ApprovalRequested: [],
        ApprovalResolved: ["echo approval-resolved"],
      },
      unsupportedHookKeys: ["Notification"],
    });
  });

  it("keeps PostToolUse aliased to PostCommand", () => {
    expect(
      resolveOperatorCliExecutionConfig({
        hooks: {
          PostToolUse: ["echo alias"],
        },
      }),
    ).toEqual({
      permissionMode: "default",
      hooks: {
        PreToolUse: [],
        PreCommand: [],
        PostCommand: ["echo alias"],
        SessionStart: [],
        SessionEnd: [],
        ApprovalRequested: [],
        ApprovalResolved: [],
      },
      unsupportedHookKeys: [],
    });
  });

  it("surfaces invalid permission modes", () => {
    expect(
      resolveOperatorCliExecutionConfig({
        permissionMode: "dangerously-unsupported",
      }),
    ).toEqual({
      permissionMode: "default",
      invalidPermissionMode: "dangerously-unsupported",
      hooks: {
        PreToolUse: [],
        PreCommand: [],
        PostCommand: [],
        SessionStart: [],
        SessionEnd: [],
        ApprovalRequested: [],
        ApprovalResolved: [],
      },
      unsupportedHookKeys: [],
    });
  });

  it("extracts status line command settings from merged config", () => {
    expect(resolveOperatorCliStatusLineConfig({
      statusLine: {
        type: "command",
        command: "node .claude/statusline.js",
        padding: 1,
        refreshInterval: 5,
        hideVimModeIndicator: true,
      },
    })).toEqual({
      type: "command",
      command: "node .claude/statusline.js",
      padding: 1,
      refreshInterval: 5,
      hideVimModeIndicator: true,
    });
  });

  it("writes and removes project status line settings", async () => {
    const cwd = await makeTempDir();
    await setProjectStatusLineConfig(cwd, { type: "command", command: "node .claude/statusline.js" });
    await expect(fs.readFile(path.join(cwd, ".claude", "settings.local.json"), "utf8")).resolves.toContain("statusLine");
    await expect(fs.readFile(path.join(cwd, ".claude", "settings.local.json"), "utf8")).resolves.toContain("node .claude/statusline.js");

    await setProjectStatusLineConfig(cwd, undefined);
    const removed = JSON.parse(await fs.readFile(path.join(cwd, ".claude", "settings.local.json"), "utf8")) as Record<string, unknown>;
    expect(removed).not.toHaveProperty("statusLine");
  });

  it("writes and removes project plan mode settings", async () => {
    const cwd = await makeTempDir();
    await setProjectPermissionModeConfig(cwd, "plan", "acceptEdits");
    const enabled = JSON.parse(await fs.readFile(path.join(cwd, ".claude", "settings.local.json"), "utf8")) as Record<string, unknown>;
    expect(enabled).toMatchObject({
      permissionMode: "plan",
      operator: { previousPermissionMode: "acceptEdits" },
    });

    await setProjectPermissionModeConfig(cwd, "default", undefined);
    const disabled = JSON.parse(await fs.readFile(path.join(cwd, ".claude", "settings.local.json"), "utf8")) as Record<string, unknown>;
    expect(disabled).toMatchObject({ permissionMode: "default" });
    expect(disabled).not.toHaveProperty("operator");
  });
});
