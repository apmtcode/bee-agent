import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import {
  OperatorCliConfigLoader,
  deepMergeConfig,
  resolveOperatorCliExecutionConfig,
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
          PreToolUse: ["echo pre"],
          PostCommand: ["echo post"],
          SessionStart: ["echo unsupported"],
        },
      }),
    ).toEqual({
      permissionMode: "acceptEdits",
      hooks: {
        PreCommand: ["echo pre"],
        PostCommand: ["echo post"],
      },
      unsupportedHookKeys: ["SessionStart"],
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
      hooks: { PreCommand: [], PostCommand: [] },
      unsupportedHookKeys: [],
    });
  });
});
