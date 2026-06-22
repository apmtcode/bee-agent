import path from "node:path";
import { readJsonFile, writeJsonAtomic } from "../shared/fs.js";

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

export type OperatorCliStatusLineConfig = {
  type: "command";
  command: string;
  padding?: number;
  refreshInterval?: number;
  hideVimModeIndicator?: boolean;
};

export type OperatorCliPermissionMode = "default" | "acceptEdits" | "bypassPermissions" | "plan";
export type OperatorCliHookEvent =
  | "PreToolUse"
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

/**
 * Normalize the two accepted shapes (a loaded runtime config or a bare merged
 * object) into a plain config object. A plain `"merged" in config` check is
 * insufficient because `OperatorCliRuntimeConfig` is structurally assignable to
 * the index-signature `OperatorCliConfigObject`, so the narrowing collapses to
 * `OperatorCliConfigValue`. Discriminate on the runtime shape instead.
 */
function resolveMergedConfig(
  config?: OperatorCliRuntimeConfig | OperatorCliConfigObject,
): OperatorCliConfigObject {
  if (!config) {
    return {};
  }
  const candidate = config as OperatorCliRuntimeConfig;
  if (isConfigObject(candidate.merged) && Array.isArray(candidate.loadedEntries)) {
    return candidate.merged;
  }
  return config as OperatorCliConfigObject;
}

export function resolveOperatorCliExecutionConfig(
  config?: OperatorCliRuntimeConfig | OperatorCliConfigObject,
): OperatorCliExecutionConfig {
  const merged = resolveMergedConfig(config);
  const rawPermissionMode = merged.permissionMode;
  const permissionMode =
    typeof rawPermissionMode === "string" && SUPPORTED_PERMISSION_MODES.has(rawPermissionMode as OperatorCliPermissionMode)
      ? (rawPermissionMode as OperatorCliPermissionMode)
      : "default";

  const hooks: Record<OperatorCliHookEvent, string[]> = {
    PreToolUse: [],
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
        case "PreToolUse":
        case "PreCommand":
          if (!commands) {
            unsupportedHookKeys.push(`${key} (invalid)`);
            continue;
          }
          hooks[key].push(...commands);
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
      PreToolUse: [...new Set(hooks.PreToolUse)],
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

export function resolveOperatorCliStatusLineConfig(
  config?: OperatorCliRuntimeConfig | OperatorCliConfigObject,
): OperatorCliStatusLineConfig | undefined {
  const merged = resolveMergedConfig(config);
  const rawStatusLine = merged.statusLine;
  if (!isConfigObject(rawStatusLine) || rawStatusLine.type !== "command" || typeof rawStatusLine.command !== "string") {
    return undefined;
  }
  const command = rawStatusLine.command.trim();
  if (!command) {
    return undefined;
  }
  const padding = typeof rawStatusLine.padding === "number" ? rawStatusLine.padding : undefined;
  const refreshInterval = typeof rawStatusLine.refreshInterval === "number" ? rawStatusLine.refreshInterval : undefined;
  const hideVimModeIndicator = typeof rawStatusLine.hideVimModeIndicator === "boolean" ? rawStatusLine.hideVimModeIndicator : undefined;
  return {
    type: "command",
    command,
    ...(padding === undefined ? {} : { padding }),
    ...(refreshInterval === undefined ? {} : { refreshInterval }),
    ...(hideVimModeIndicator === undefined ? {} : { hideVimModeIndicator }),
  };
}

export async function setProjectStatusLineConfig(cwd: string, statusLine?: OperatorCliStatusLineConfig): Promise<void> {
  await updateProjectLocalConfig(cwd, (next) => {
    if (statusLine) {
      next.statusLine = cloneConfigValue(statusLine) as OperatorCliConfigValue;
    } else {
      delete next.statusLine;
    }
  });
}

export async function setProjectPermissionModeConfig(
  cwd: string,
  permissionMode?: OperatorCliPermissionMode,
  previousPermissionMode?: Exclude<OperatorCliPermissionMode, "plan">,
): Promise<void> {
  await updateProjectLocalConfig(cwd, (next) => {
    if (permissionMode) {
      next.permissionMode = permissionMode;
    } else {
      delete next.permissionMode;
    }
    if (previousPermissionMode) {
      next.operator = {
        ...(isConfigObject(next.operator) ? next.operator : {}),
        previousPermissionMode,
      };
      return;
    }
    if (!isConfigObject(next.operator)) {
      return;
    }
    const operator = { ...next.operator };
    delete operator.previousPermissionMode;
    if (Object.keys(operator).length === 0) {
      delete next.operator;
      return;
    }
    next.operator = operator;
  });
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

async function updateProjectLocalConfig(
  cwd: string,
  update: (next: OperatorCliConfigObject) => void,
): Promise<void> {
  const filePath = path.join(cwd, ".claude", "settings.local.json");
  const current = await readOptionalConfigObject(filePath) ?? {};
  const next = { ...current };
  update(next);
  await writeJsonAtomic(filePath, next);
}
