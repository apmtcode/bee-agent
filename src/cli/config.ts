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
