export type OperatorPluginCapability =
  | "tool"
  | "memory-enricher"
  | "capture-source"
  | "skill-provider"
  | "training-runtime";

export type OperatorPluginManifest = {
  id: string;
  version: string;
  name: string;
  description: string;
  entrypoint: string;
  capabilities: OperatorPluginCapability[];
  enabledByDefault?: boolean;
  metadata?: Record<string, unknown>;
};

export type OperatorPluginRuntime = {
  activate(): Promise<void>;
};

export function isOperatorPluginManifest(value: unknown): value is OperatorPluginManifest {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return false;
  }
  const candidate = value as Record<string, unknown>;
  return (
    typeof candidate.id === "string" &&
    candidate.id.length > 0 &&
    typeof candidate.version === "string" &&
    candidate.version.length > 0 &&
    typeof candidate.name === "string" &&
    candidate.name.length > 0 &&
    typeof candidate.description === "string" &&
    candidate.description.length > 0 &&
    typeof candidate.entrypoint === "string" &&
    candidate.entrypoint.length > 0 &&
    Array.isArray(candidate.capabilities) &&
    candidate.capabilities.every((item) =>
      item === "tool" ||
      item === "memory-enricher" ||
      item === "capture-source" ||
      item === "skill-provider" ||
      item === "training-runtime",
    )
  );
}
