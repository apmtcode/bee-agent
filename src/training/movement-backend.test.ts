import { describe, expect, it } from "vitest";
import type { MovementToken } from "./movement-model.js";
import {
  MockMovementTrainingBackend,
  MovementTrainingBackendRegistry,
  createDefaultMovementBackendRegistry,
  type MovementModelArtifact,
  type MovementTrainingBackend,
} from "./movement-backend.js";

function obs(source: string, summary: string): MovementToken {
  return { role: "observation", source, summary };
}
function act(tool: string, summary: string): MovementToken {
  return { role: "action", tool, summary };
}

const flow: MovementToken[] = [
  obs("window", "open browser"),
  act("mouse", "click(new-tab)"),
  act("keyboard", "type(url)"),
  act("keyboard", "press(enter)"),
];

describe("MockMovementTrainingBackend", () => {
  it("trains an artifact and reloads it to repeat the recorded movements", async () => {
    const backend = new MockMovementTrainingBackend();
    const artifact = await backend.train(
      { sequences: [{ tokens: flow }] },
      { createdAt: "2026-07-18T00:00:00.000Z", metadata: { purpose: "test" } },
    );

    expect(artifact.backendId).toBe("mock");
    expect(artifact.format).toBe("movement-ngram/v1");
    expect(artifact.sequenceCount).toBe(1);
    expect(artifact.actionCount).toBe(3);
    expect(artifact.createdAt).toBe("2026-07-18T00:00:00.000Z");
    expect(artifact.metadata).toEqual({ purpose: "test" });
    expect(artifact.model).toBeDefined();

    const handle = backend.load(artifact);
    const produced = handle.infer({ seed: [obs("window", "open browser")] });
    expect(produced.map((token) => token.summary)).toEqual([
      "click(new-tab)",
      "type(url)",
      "press(enter)",
    ]);
  });

  it("survives a JSON serialization round-trip of the artifact", async () => {
    const backend = new MockMovementTrainingBackend();
    const artifact = await backend.train({ sequences: [{ tokens: flow }] });
    const roundTripped = JSON.parse(JSON.stringify(artifact)) as MovementModelArtifact;
    const handle = backend.load(roundTripped);
    expect(handle.infer({ seed: [obs("window", "open browser")] }).map((t) => t.summary)).toEqual([
      "click(new-tab)",
      "type(url)",
      "press(enter)",
    ]);
  });

  it("exposes fidelity evaluation through the handle", async () => {
    const backend = new MockMovementTrainingBackend();
    const artifact = await backend.train({ sequences: [{ tokens: flow }] });
    const handle = backend.load(artifact);
    const fidelity = handle.evaluate({ tokens: flow });
    expect(fidelity.predicted).toBe(3);
    expect(fidelity.correct).toBe(3);
    expect(fidelity.accuracy).toBe(1);
  });

  it("throws when asked to load an artifact with no in-process model", () => {
    const backend = new MockMovementTrainingBackend();
    const artifact: MovementModelArtifact = {
      version: 1,
      backendId: "mock",
      format: "native/gguf",
      sequenceCount: 1,
      actionCount: 1,
      nativeArtifactPath: "/tmp/model.gguf",
    };
    expect(() => backend.load(artifact)).toThrow(/no in-process model/);
  });
});

describe("MovementTrainingBackendRegistry", () => {
  it("registers, resolves, and lists backends", () => {
    const registry = createDefaultMovementBackendRegistry();
    expect(registry.has("mock")).toBe(true);
    expect(registry.get("mock").id).toBe("mock");
    expect(registry.list().map((backend) => backend.id)).toContain("mock");
  });

  it("throws on an unknown backend id", () => {
    const registry = createDefaultMovementBackendRegistry();
    expect(() => registry.get("nope")).toThrow(/Unknown movement training backend/);
  });

  it("loads an artifact using the backend that produced it", async () => {
    const registry = createDefaultMovementBackendRegistry();
    const artifact = await registry.get("mock").train({ sequences: [{ tokens: flow }] });
    const handle = registry.load(artifact);
    expect(handle.backendId).toBe("mock");
    expect(handle.infer({ seed: [obs("window", "open browser")] })).toHaveLength(3);
  });

  it("accepts a custom backend implementing the interface (pluggability)", () => {
    const custom: MovementTrainingBackend = {
      id: "native-mlx",
      kind: "native-deferred",
      train: async () => ({
        version: 1,
        backendId: "native-mlx",
        format: "native/gguf",
        sequenceCount: 0,
        actionCount: 0,
        nativeArtifactPath: "/models/mlx.gguf",
      }),
      load: () => {
        throw new Error("native inference runs on-device");
      },
    };
    const registry = new MovementTrainingBackendRegistry().register(custom);
    expect(registry.get("native-mlx").kind).toBe("native-deferred");
  });
});
