import path from "node:path";
import { pathToFileURL } from "node:url";
import { readJsonFile, writeJsonAtomic } from "../shared/fs.js";
import {
  isOperatorPluginManifest,
  type OperatorPluginCapability,
  type OperatorPluginManifest,
  type OperatorPluginRuntime,
} from "./manifest.js";

export type RegisteredPluginRecord = {
  manifest: OperatorPluginManifest;
  registeredAt: string;
  enabled: boolean;
};

export type PluginRegistryShape = {
  version: 1;
  plugins: RegisteredPluginRecord[];
};

export type RegisterPluginParams = {
  manifest: OperatorPluginManifest;
  enabled?: boolean;
};

const EMPTY_REGISTRY: PluginRegistryShape = {
  version: 1,
  plugins: [],
};

function nowIso(): string {
  return new Date().toISOString();
}

function resolveEntrypoint(rootDir: string, entrypoint: string): string {
  return path.isAbsolute(entrypoint) ? entrypoint : path.join(rootDir, entrypoint);
}

export class FilePluginManifestRegistry {
  constructor(
    private readonly filePath: string,
    private readonly rootDir: string,
  ) {}

  async load(): Promise<PluginRegistryShape> {
    return await readJsonFile(this.filePath, EMPTY_REGISTRY);
  }

  async save(store: PluginRegistryShape): Promise<void> {
    await writeJsonAtomic(this.filePath, store);
  }

  async register(params: RegisterPluginParams): Promise<RegisteredPluginRecord> {
    if (!isOperatorPluginManifest(params.manifest)) {
      throw new Error("Invalid plugin manifest");
    }
    const store = await this.load();
    const existingIndex = store.plugins.findIndex((item) => item.manifest.id === params.manifest.id);
    const record: RegisteredPluginRecord = {
      manifest: params.manifest,
      registeredAt: nowIso(),
      enabled: params.enabled ?? params.manifest.enabledByDefault ?? true,
    };
    if (existingIndex >= 0) {
      store.plugins[existingIndex] = record;
    } else {
      store.plugins.push(record);
    }
    await this.save(store);
    return record;
  }

  async list(): Promise<RegisteredPluginRecord[]> {
    const store = await this.load();
    return [...store.plugins].sort((a, b) => a.manifest.id.localeCompare(b.manifest.id));
  }

  async get(pluginId: string): Promise<RegisteredPluginRecord | undefined> {
    const store = await this.load();
    return store.plugins.find((item) => item.manifest.id === pluginId);
  }

  async setEnabled(pluginId: string, enabled: boolean): Promise<RegisteredPluginRecord | undefined> {
    const store = await this.load();
    const index = store.plugins.findIndex((item) => item.manifest.id === pluginId);
    if (index < 0) {
      return undefined;
    }
    const updated: RegisteredPluginRecord = {
      ...store.plugins[index],
      enabled,
    };
    store.plugins[index] = updated;
    await this.save(store);
    return updated;
  }

  async listByCapability(capability: OperatorPluginCapability): Promise<RegisteredPluginRecord[]> {
    const plugins = await this.list();
    return plugins.filter((item) => item.manifest.capabilities.includes(capability));
  }

  async activate(pluginId: string): Promise<OperatorPluginRuntime> {
    const record = await this.get(pluginId);
    if (!record) {
      throw new Error(`Unknown plugin: ${pluginId}`);
    }
    const modulePath = resolveEntrypoint(this.rootDir, record.manifest.entrypoint);
    const runtimeModule = await import(pathToFileURL(modulePath).href);
    const runtime = runtimeModule.default ?? runtimeModule.runtime;
    if (!runtime || typeof runtime.activate !== "function") {
      throw new Error(`Invalid plugin runtime: ${pluginId}`);
    }
    return runtime as OperatorPluginRuntime;
  }
}
