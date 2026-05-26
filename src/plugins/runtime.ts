import type { OperatorPluginCapability, OperatorPluginRuntime } from "./manifest.js";
import {
  FilePluginManifestRegistry,
  type RegisteredPluginRecord,
} from "./manifest-registry.js";

export class OperatorPluginRuntimeManager {
  private readonly activated = new Map<string, OperatorPluginRuntime>();

  constructor(private readonly registry: FilePluginManifestRegistry) {}

  async listPlugins(): Promise<RegisteredPluginRecord[]> {
    return await this.registry.list();
  }

  async listEnabledPlugins(capability?: OperatorPluginCapability): Promise<RegisteredPluginRecord[]> {
    const plugins = capability
      ? await this.registry.listByCapability(capability)
      : await this.registry.list();
    return plugins.filter((item) => item.enabled);
  }

  async activate(pluginId: string): Promise<OperatorPluginRuntime> {
    const existing = this.activated.get(pluginId);
    if (existing) {
      return existing;
    }
    const runtime = await this.registry.activate(pluginId);
    await runtime.activate();
    this.activated.set(pluginId, runtime);
    return runtime;
  }

  isActivated(pluginId: string): boolean {
    return this.activated.has(pluginId);
  }
}
