import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { FilePluginManifestRegistry } from "./manifest-registry.js";

const tempDirs: string[] = [];

async function makeTempDir(): Promise<string> {
  const dir = await fs.mkdtemp(path.join(os.tmpdir(), "plugin-registry-"));
  tempDirs.push(dir);
  return dir;
}

afterEach(async () => {
  await Promise.all(tempDirs.splice(0).map((dir) => fs.rm(dir, { recursive: true, force: true })));
});

describe("FilePluginManifestRegistry", () => {
  it("registers, lists, filters, enables, and lazily activates plugins", async () => {
    const rootDir = await makeTempDir();
    const registryFile = path.join(rootDir, "plugins.json");
    const entrypoint = path.join(rootDir, "sample-plugin.mjs");

    await fs.writeFile(
      entrypoint,
      [
        "globalThis.__operatorPluginLoads = (globalThis.__operatorPluginLoads ?? 0) + 1;",
        "export default {",
        "  async activate() {",
        "    globalThis.__operatorPluginActivations = (globalThis.__operatorPluginActivations ?? 0) + 1;",
        "  },",
        "};",
        "",
      ].join("\n"),
      "utf8",
    );

    const registry = new FilePluginManifestRegistry(registryFile, rootDir);
    const record = await registry.register({
      manifest: {
        id: "sample-plugin",
        version: "1.0.0",
        name: "Sample Plugin",
        description: "Exposes a standalone tool",
        entrypoint,
        capabilities: ["tool", "training-runtime"],
      },
    });

    expect(record.enabled).toBe(true);
    await expect(registry.list()).resolves.toEqual([
      expect.objectContaining({ manifest: expect.objectContaining({ id: "sample-plugin" }) }),
    ]);
    await expect(registry.listByCapability("tool")).resolves.toEqual([
      expect.objectContaining({ manifest: expect.objectContaining({ id: "sample-plugin" }) }),
    ]);
    await expect(registry.setEnabled("sample-plugin", false)).resolves.toMatchObject({ enabled: false });

    expect((globalThis as Record<string, unknown>).__operatorPluginLoads).toBeUndefined();
    const runtime = await registry.activate("sample-plugin");
    expect(typeof runtime.activate).toBe("function");
    expect((globalThis as Record<string, unknown>).__operatorPluginLoads).toBe(1);
  });
});
