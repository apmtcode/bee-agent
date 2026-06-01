import path from "node:path";
import { readJsonFile } from "../shared/fs.js";

export type OperatorCliConfigSource = "user" | "project" | "local";

export type OperatorCliConfigEntry = {
  source: OperatorCliConfigSource;
  path: string;
};

export type OperatorCliConfigValue = null | boolean | number | string | OperatorCliConfigObject | OperatorCliConfigValue[];

export type OperatorCliConfigObject = {
  [key: string]: OperatorCliConfigValue;
};

export type OperatorCliRuntimeConfig = {
  merged: OperatorCliConfigObject;
  loadedEntries: OperatorCliConfigEntry[];
};

export type OperatorCliPermissionMode = "default" | "acceptEdits" | "bypassPermissions" | "plan";
export type OperatorCliHookEvent =
  | "PreCommand"
  | "PostCommand"
  | "SessionStart"
  | "SessionEnd"
  | "ApprovalRequested"
  | "ApprovalResolved";

export type OperatorCliExecutionConfig = {
  permissionMode: OperatorCliPermissionMode;
  invalidPermissionMode?: string;
  hooks: Record<OperatorCliHookEvent, string[]>;
  unsupportedHookKeys: string[];
};

const SUPPORTED_PERMISSION_MODES = new Set<OperatorCliPermissionMode>([
  "default",
  "acceptEdits",
  "bypassPermissions",
  "plan",
]);

export class OperatorCliConfigLoader {
  constructor(
    private readonly cwd: string,
    private readonly configHome = process.env.CLAUDE_CONFIG_HOME ?? path.join(process.env.HOME ?? ".", ".claude"),
  ) {}

  discover(): OperatorCliConfigEntry[] {
    return [
      { source: "user", path: path.join(this.configHome, "settings.json") },
      { source: "project", path: path.join(this.cwd, ".claude", "settings.json") },
      { source: "local", path: path.join(this.cwd, ".claude", "settings.local.json") },
    ];
  }

  async load(): Promise<OperatorCliRuntimeConfig> {
    const merged: OperatorCliConfigObject = {};
    const loadedEntries: OperatorCliConfigEntry[] = [];

    for (const entry of this.discover()) {
      const parsed = await readOptionalConfigObject(entry.path);
      if (!parsed) {
        continue;
      }
      deepMergeConfig(merged, parsed);
      loadedEntries.push(entry);
    }

    return { merged, loadedEntries };
  }
}

export function resolveOperatorCliExecutionConfig(
  config?: OperatorCliRuntimeConfig | OperatorCliConfigObject,
): OperatorCliExecutionConfig {
  const merged = !config ? {} : "merged" in config ? config.merged : config;
  const rawPermissionMode = merged.permissionMode;
  const permissionMode =
    typeof rawPermissionMode === "string" && SUPPORTED_PERMISSION_MODES.has(rawPermissionMode as OperatorCliPermissionMode)
      ? (rawPermissionMode as OperatorCliPermissionMode)
      : "default";

  const hooks: Record<OperatorCliHookEvent, string[]> = {
    PreCommand: [],
    PostCommand: [],
    SessionStart: [],
    SessionEnd: [],
    ApprovalRequested: [],
    ApprovalResolved: [],
  };
  const unsupportedHookKeys: string[] = [];
  const rawHooks = merged.hooks;
  if (isConfigObject(rawHooks)) {
    for (const [key, value] of Object.entries(rawHooks)) {
      const commands = readHookCommandList(value);
      switch (key) {
        case "PreCommand":
        case "PreToolUse":
          if (!commands) {
            unsupportedHookKeys.push(`${key} (invalid)`);
            continue;
          }
          hooks.PreCommand.push(...commands);
          continue;
        case "PostCommand":
        case "PostToolUse":
          if (!commands) {
            unsupportedHookKeys.push(`${key} (invalid)`);
            continue;
          }
          hooks.PostCommand.push(...commands);
          continue;
        case "SessionStart":
        case "SessionEnd":
        case "ApprovalRequested":
        case "ApprovalResolved":
          if (!commands) {
            unsupportedHookKeys.push(`${key} (invalid)`);
            continue;
          }
          hooks[key].push(...commands);
          continue;
        default:
          unsupportedHookKeys.push(key);
      }
    }
  }

  return {
    permissionMode,
    ...(typeof rawPermissionMode === "string" && !SUPPORTED_PERMISSION_MODES.has(rawPermissionMode as OperatorCliPermissionMode)
      ? { invalidPermissionMode: rawPermissionMode }
      : {}),
    hooks: {
      PreCommand: [...new Set(hooks.PreCommand)],
      PostCommand: [...new Set(hooks.PostCommand)],
      SessionStart: [...new Set(hooks.SessionStart)],
      SessionEnd: [...new Set(hooks.SessionEnd)],
      ApprovalRequested: [...new Set(hooks.ApprovalRequested)],
      ApprovalResolved: [...new Set(hooks.ApprovalResolved)],
    },
    unsupportedHookKeys,
  };
}

async function readOptionalConfigObject(filePath: string): Promise<OperatorCliConfigObject | undefined> {
  const value = await readJsonFile<unknown | undefined>(filePath, undefined);
  if (value === undefined) {
    return undefined;
  }
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new Error(`${filePath}: top-level settings value must be a JSON object`);
  }
  return value as OperatorCliConfigObject;
}

function isConfigObject(value: OperatorCliConfigValue | undefined): value is OperatorCliConfigObject {
  return !!value && typeof value === "object" && !Array.isArray(value);
}

function readHookCommandList(value: OperatorCliConfigValue | undefined): string[] | undefined {
  if (typeof value === "string") {
    return value.trim() ? [value.trim()] : [];
  }
  if (Array.isArray(value) && value.every((item) => typeof item === "string")) {
    return value.map((item) => item.trim()).filter(Boolean);
  }
  return undefined;
}

export function deepMergeConfig(target: OperatorCliConfigObject, source: OperatorCliConfigObject): OperatorCliConfigObject {
  for (const [key, value] of Object.entries(source)) {
    const existing = target[key];
    if (isConfigObject(existing) && isConfigObject(value)) {
      deepMergeConfig(existing, value);
      continue;
    }
    target[key] = cloneConfigValue(value);
  }
  return target;
}

function cloneConfigValue(value: OperatorCliConfigValue): OperatorCliConfigValue {
  if (Array.isArray(value)) {
    return value.map((item) => cloneConfigValue(item));
  }
  if (isConfigObject(value)) {
    return Object.fromEntries(Object.entries(value).map(([key, item]) => [key, cloneConfigValue(item)]));
  }
  return value;
}
