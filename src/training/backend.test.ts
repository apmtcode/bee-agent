import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import {
  MockMovementTrainingBackend,
  TrainingBackendRegistry,
  createDefaultTrainingBackendRegistry,
  replaysFromExport,
  type TrainingBackend,
} from "./backend.js";
import { predictAction } from "./policy-model.js";
import type { ReviewedExportManifest } from "./export-manifest.js";

let tempDir: string;

beforeEach(async () => {
  tempDir = await fs.mkdtemp(path.join(os.tmpdir(), "bee-backend-"));
});

afterEach(async () => {
  await fs.rm(tempDir, { recursive: true, force: true });
});

const replays = [
  {
    events: [
      { kind: "observation" as const, source: "window", summary: "Editor focused on main.ts" },
      { kind: "action" as const, tool: "keyboard", summary: "type import" },
    ],
  },
];

describe("MockMovementTrainingBackend", () => {
  it("advertises cloud-capable in-process execution", () => {
    const backend = new MockMovementTrainingBackend();
    expect(backend.descriptor.supportsCloudExecution).toBe(true);
    expect(backend.descriptor.kind).toBe("in-process");
  });

  it("trains a model artifact fully in-process and reports metrics", async () => {
    const backend = new MockMovementTrainingBackend();
    const result = await backend.train({ jobId: "job-1", mode: "sft", replays, outputDir: tempDir });

    expect(result.status).toBe("completed");
    expect(result.metrics).toMatchObject({ observationCount: 1, actionCount: 1, transitionCount: 1 });
    expect(result.metrics?.selfConsistency).toBe(1);

    const written = JSON.parse(await fs.readFile(result.modelPath!, "utf8"));
    expect(written.kind).toBe("frequency-nextaction");
  });

  it("round-trips the artifact so inference works after loading", async () => {
    const backend = new MockMovementTrainingBackend();
    const result = await backend.train({ jobId: "job-2", mode: "sft", replays, outputDir: tempDir });
    const model = await backend.loadModel(result.modelPath!);
    expect(model).toBeDefined();
    const prediction = predictAction(model!, { source: "window", summary: "Editor focused on main.ts" });
    expect(prediction.action).toMatchObject({ tool: "keyboard", summary: "type import" });
  });

  it("produces a byte-identical artifact for identical input (deterministic)", async () => {
    const backend = new MockMovementTrainingBackend();
    const a = path.join(tempDir, "a");
    const b = path.join(tempDir, "b");
    await backend.train({ jobId: "job", mode: "sft", replays, outputDir: a });
    await backend.train({ jobId: "job", mode: "sft", replays, outputDir: b });
    expect(await fs.readFile(path.join(a, "model.json"), "utf8")).toBe(
      await fs.readFile(path.join(b, "model.json"), "utf8"),
    );
  });
});

describe("replaysFromExport", () => {
  it("extracts observation/action streams from a reviewed export, dropping transcript", () => {
    const manifest = {
      replays: [
        {
          sessionId: "s1",
          trajectoryIds: ["t1"],
          eventCount: 3,
          events: [
            { kind: "transcript", ts: 1, messageId: "m1", role: "user", content: "hello" },
            { kind: "observation", ts: 2, trajectoryId: "t1", source: "window", summary: "open" },
            { kind: "action", ts: 3, trajectoryId: "t1", tool: "mouse", summary: "click" },
          ],
        },
      ],
    } as Pick<ReviewedExportManifest, "replays">;

    const converted = replaysFromExport(manifest);
    expect(converted).toHaveLength(1);
    expect(converted[0].events).toEqual([
      { kind: "observation", source: "window", summary: "open" },
      { kind: "action", tool: "mouse", summary: "click" },
    ]);
  });
});

describe("TrainingBackendRegistry", () => {
  it("registers, resolves, and defaults backends", () => {
    const registry = createDefaultTrainingBackendRegistry();
    expect(registry.getDefault().descriptor.id).toBe("mock-movement");
    expect(registry.has("mock-movement")).toBe(true);
    expect(registry.listCloudCapable()).toHaveLength(1);
  });

  it("rejects duplicate ids", () => {
    const registry = new TrainingBackendRegistry();
    registry.register(new MockMovementTrainingBackend());
    expect(() => registry.register(new MockMovementTrainingBackend())).toThrow(/already registered/);
  });

  it("throws for unknown backend ids", () => {
    const registry = new TrainingBackendRegistry();
    expect(() => registry.get("nope")).toThrow(/Unknown training backend/);
  });

  it("honours an explicit makeDefault override", () => {
    const registry = new TrainingBackendRegistry();
    const secondary: TrainingBackend = {
      descriptor: {
        id: "on-device-stub",
        runtime: "mlx",
        targetPlatform: "apple-silicon",
        kind: "local-process",
        supportsCloudExecution: false,
        supportedModes: ["sft"],
      },
      train: async (request) => ({ backendId: "on-device-stub", jobId: request.jobId, status: "completed" }),
    };
    registry.register(new MockMovementTrainingBackend());
    registry.register(secondary, { makeDefault: true });
    expect(registry.getDefault().descriptor.id).toBe("on-device-stub");
    expect(registry.listCloudCapable().map((backend) => backend.descriptor.id)).toEqual(["mock-movement"]);
  });
});
