import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { FilePluginManifestRegistry } from "./manifest-registry.js";
import { OperatorPluginRuntimeManager } from "./runtime.js";

const tempDirs: string[] = [];

async function makeTempDir(): Promise<string> {
  const dir = await fs.mkdtemp(path.join(os.tmpdir(), "plugin-runtime-"));
  tempDirs.push(dir);
  return dir;
}

afterEach(async () => {
  delete (globalThis as Record<string, unknown>).__operatorRuntimeActivations;
  await Promise.all(tempDirs.splice(0).map((dir) => fs.rm(dir, { recursive: true, force: true })));
});

describe("OperatorPluginRuntimeManager", () => {
  it("lists enabled plugins and activates each plugin once", async () => {
    const rootDir = await makeTempDir();
    const registry = new FilePluginManifestRegistry(path.join(rootDir, "plugins.json"), rootDir);
    const entrypoint = path.join(rootDir, "runtime-plugin.mjs");

    await fs.writeFile(
      entrypoint,
      [
        "export default {",
        "  async activate() {",
        "    globalThis.__operatorRuntimeActivations = (globalThis.__operatorRuntimeActivations ?? 0) + 1;",
        "  },",
        "  getSkillStepHandler(kind) {",
        "    if (kind !== 'summary') return undefined;",
        "    return async ({ step }) => ({ output: step.content ?? 'plugin-summary' });",
        "  },",
        "};",
        "",
      ].join("\n"),
      "utf8",
    );

    await registry.register({
      manifest: {
        id: "runtime-plugin",
        version: "1.0.0",
        name: "Runtime Plugin",
        description: "Provides a standalone plugin runtime",
        entrypoint,
        capabilities: ["skill-provider"],
      },
    });

    const manager = new OperatorPluginRuntimeManager(registry);
    await expect(manager.listEnabledPlugins("skill-provider")).resolves.toEqual([
      expect.objectContaining({ manifest: expect.objectContaining({ id: "runtime-plugin" }) }),
    ]);

    await manager.activate("runtime-plugin");
    await manager.activate("runtime-plugin");

    expect(manager.isActivated("runtime-plugin")).toBe(true);
    expect((globalThis as Record<string, unknown>).__operatorRuntimeActivations).toBe(1);
    await expect(manager.getSkillStepHandler("summary")).resolves.toBeTypeOf("function");
    await expect(manager.getSkillStepHandler("command")).resolves.toBeUndefined();
  });
});
