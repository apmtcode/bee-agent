import fs from "node:fs/promises";
import path from "node:path";
import { execFile } from "node:child_process";
import { promisify } from "node:util";
import type { OperatorCliRuntimeConfig } from "./config.js";

const execFileAsync = promisify(execFile);

export type OperatorCliContextFile = {
  path: string;
  content: string;
};

export type OperatorCliProjectContext = {
  cwd: string;
  currentDate: string;
  gitStatus?: string;
  instructionFiles: OperatorCliContextFile[];
};

export type OperatorCliPromptContext = {
  project: OperatorCliProjectContext;
  config?: OperatorCliRuntimeConfig;
  sections: string[];
};

export async function discoverPromptContext(params: {
  cwd: string;
  currentDate: string;
  config?: OperatorCliRuntimeConfig;
  includeGitStatus?: boolean;
}): Promise<OperatorCliPromptContext> {
  const project: OperatorCliProjectContext = {
    cwd: params.cwd,
    currentDate: params.currentDate,
    instructionFiles: await discoverInstructionFiles(params.cwd),
    ...(params.includeGitStatus === false ? {} : { gitStatus: await readGitStatus(params.cwd) }),
  };

  const sections = [
    renderEnvironmentSection(project),
    renderProjectSection(project),
    ...(project.instructionFiles.length > 0 ? [renderInstructionFiles(project.instructionFiles)] : []),
    ...(params.config ? [renderConfigSection(params.config)] : []),
  ];

  return { project, config: params.config, sections };
}

async function discoverInstructionFiles(cwd: string): Promise<OperatorCliContextFile[]> {
  const directories: string[] = [];
  let cursor = path.resolve(cwd);
  while (true) {
    directories.push(cursor);
    const parent = path.dirname(cursor);
    if (parent === cursor) {
      break;
    }
    cursor = parent;
  }
  directories.reverse();

  const files: OperatorCliContextFile[] = [];
  for (const directory of directories) {
    for (const candidate of [
      path.join(directory, "CLAUDE.md"),
      path.join(directory, "CLAUDE.local.md"),
      path.join(directory, ".claude", "CLAUDE.md"),
    ]) {
      const content = await readOptionalText(candidate);
      if (content) {
        files.push({ path: candidate, content });
      }
    }
  }
  return files;
}

async function readOptionalText(filePath: string): Promise<string | undefined> {
  try {
    const content = await fs.readFile(filePath, "utf8");
    const trimmed = content.trim();
    return trimmed.length > 0 ? trimmed : undefined;
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") {
      return undefined;
    }
    throw error;
  }
}

async function readGitStatus(cwd: string): Promise<string | undefined> {
  try {
    const { stdout } = await execFileAsync("git", ["--no-optional-locks", "status", "--short", "--branch"], { cwd });
    const trimmed = stdout.trim();
    return trimmed.length > 0 ? trimmed : undefined;
  } catch {
    return undefined;
  }
}

function renderEnvironmentSection(project: OperatorCliProjectContext): string {
  return ["# Environment context", ` - Working directory: ${project.cwd}`, ` - Date: ${project.currentDate}`].join("\n");
}

function renderProjectSection(project: OperatorCliProjectContext): string {
  return [
    "# Project context",
    ` - Today's date is ${project.currentDate}.`,
    ...(project.gitStatus ? ["", "Git status snapshot:", project.gitStatus] : []),
  ].join("\n");
}

function renderInstructionFiles(files: OperatorCliContextFile[]): string {
  return [
    "# Claude instructions",
    ...files.flatMap((file) => [`## ${file.path}`, file.content]),
  ].join("\n\n");
}

function renderConfigSection(config: OperatorCliRuntimeConfig): string {
  return [
    "# Runtime config",
    ...config.loadedEntries.map((entry) => ` - ${entry.source}: ${entry.path}`),
  ].join("\n");
}
