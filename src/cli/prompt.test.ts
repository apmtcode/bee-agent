import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { discoverPromptContext } from "./prompt.js";

const tempDirs: string[] = [];

async function makeTempDir(): Promise<string> {
  const dir = await fs.mkdtemp(path.join(os.tmpdir(), "operator-cli-prompt-"));
  tempDirs.push(dir);
  return dir;
}

afterEach(async () => {
  await Promise.all(tempDirs.splice(0).map((dir) => fs.rm(dir, { recursive: true, force: true })));
});

describe("discoverPromptContext", () => {
  it("discovers instruction files in stable ancestor order", async () => {
    const root = await makeTempDir();
    const project = path.join(root, "project");
    const nested = path.join(project, "src", "feature");
    await fs.mkdir(path.join(project, ".claude"), { recursive: true });
    await fs.mkdir(nested, { recursive: true });

    await fs.writeFile(path.join(root, "CLAUDE.md"), "root instructions\n");
    await fs.writeFile(path.join(project, "CLAUDE.local.md"), "local project instructions\n");
    await fs.writeFile(path.join(project, ".claude", "CLAUDE.md"), "project claude instructions\n");

    const context = await discoverPromptContext({
      cwd: nested,
      currentDate: "2026-05-25",
      includeGitStatus: false,
    });

    expect(context.project.instructionFiles.map((file) => path.basename(file.path))).toEqual([
      "CLAUDE.md",
      "CLAUDE.local.md",
      "CLAUDE.md",
    ]);
    expect(context.sections.join("\n\n")).toContain("root instructions");
    expect(context.sections.join("\n\n")).toContain("project claude instructions");
  });
});
