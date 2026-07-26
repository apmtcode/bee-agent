import { describe, expect, it } from "vitest";
import { buildTrajectorySpan } from "../capture/trajectory.js";
import type { TrajectorySpan } from "../capture/trajectory.js";
import type { ReplayManifest } from "../capture/replay.js";
import {
  MovementBackendRegistry,
  evaluateNextTokenAccuracy,
  extractMovementSequences,
  extractMovementSequencesFromReplay,
  movementActionToken,
  type MovementDataset,
} from "./movement-model.js";
import {
  SimulatedMovementModelBackend,
  createDefaultMovementBackendRegistry,
  createSimulatedMovementBackend,
} from "./simulated-backend.js";

function actionSpan(id: string, gestures: Array<{ tool: string; gesture: string; target?: string }>): TrajectorySpan {
  return buildTrajectorySpan({
    id,
    sessionId: "sess-move",
    actions: gestures.map((g, index) => ({
      kind: "action" as const,
      tool: g.tool,
      summary: `${g.gesture} ${g.target ?? ""}`.trim(),
      ts: 1000 + index,
      metadata: { gesture: g.gesture, ...(g.target ? { target: g.target } : {}) },
    })),
  });
}

describe("movementActionToken", () => {
  it("produces stable, low-cardinality, normalized tokens", () => {
    expect(movementActionToken({ tool: "Device", gesture: "Tap", target: "Submit Button" })).toBe(
      "device:tap:submit-button",
    );
    expect(movementActionToken({ tool: "device", gesture: "swipe", direction: "left" })).toBe("device:swipe:left");
    expect(movementActionToken({ tool: "device" })).toBe("device");
  });
});

describe("extractMovementSequences", () => {
  it("reads tokens from trajectory action metadata", () => {
    const dataset = extractMovementSequences([
      actionSpan("t1", [
        { tool: "device", gesture: "tap", target: "search" },
        { tool: "device", gesture: "type", target: "query" },
      ]),
    ]);
    expect(dataset.sequences).toHaveLength(1);
    expect(dataset.sequences[0]?.tokens).toEqual(["device:tap:search", "device:type:query"]);
  });

  it("prefers reviewed redacted actions over raw actions", () => {
    const span = actionSpan("t1", [{ tool: "device", gesture: "tap", target: "raw-secret" }]);
    span.review = {
      status: "approved",
      reviewedAt: new Date(0).toISOString(),
      reviewedBy: "reviewer",
      redactedActions: [{ ts: 1, tool: "device", summary: "tap redacted" }],
    };
    const dataset = extractMovementSequences([span]);
    expect(dataset.sequences[0]?.tokens).toEqual(["device:tap"]);
  });

  it("skips trajectories with no movement actions", () => {
    const empty = buildTrajectorySpan({ id: "empty", sessionId: "s" });
    expect(extractMovementSequences([empty]).sequences).toHaveLength(0);
  });
});

describe("extractMovementSequencesFromReplay", () => {
  it("groups action events per trajectory", () => {
    const manifest: ReplayManifest = {
      version: 1,
      sessionId: "s",
      trajectoryIds: ["a", "b"],
      eventCount: 3,
      events: [
        { kind: "action", ts: 1, trajectoryId: "a", tool: "device", summary: "tap thing" },
        { kind: "observation", ts: 2, trajectoryId: "a", source: "device", summary: "screen" },
        { kind: "action", ts: 3, trajectoryId: "a", tool: "device", summary: "swipe up" },
        { kind: "action", ts: 4, trajectoryId: "b", tool: "device", summary: "scroll down" },
      ],
    };
    const dataset = extractMovementSequencesFromReplay(manifest);
    expect(dataset.sequences).toEqual([
      { id: "a", tokens: ["device:tap", "device:swipe"] },
      { id: "b", tokens: ["device:scroll"] },
    ]);
  });
});

describe("SimulatedMovementModelBackend", () => {
  const dataset: MovementDataset = {
    sequences: [
      { id: "s1", tokens: ["open", "search", "type", "submit"] },
      { id: "s2", tokens: ["open", "search", "type", "submit"] },
      { id: "s3", tokens: ["open", "search", "scroll", "submit"] },
    ],
  };

  it("trains a serializable artifact tagged with the backend id", async () => {
    const backend = new SimulatedMovementModelBackend();
    const artifact = await backend.train(dataset, { order: 2 });
    expect(artifact.backendId).toBe("simulated-ngram");
    expect(artifact.sequenceCount).toBe(3);
    expect(artifact.tokenCount).toBe(12);
    expect(artifact.vocabulary).toEqual(["open", "scroll", "search", "submit", "type"]);
    // Round-trips through JSON (proves it is a real serializable model artifact).
    expect(() => JSON.parse(JSON.stringify(artifact))).not.toThrow();
  });

  it("reproduces the dominant recorded movement chain (training = repeat)", async () => {
    const backend = createSimulatedMovementBackend();
    const model = backend.load(await backend.train(dataset, { order: 2 }));
    // From the seed "open" the argmax rollout must reproduce the majority chain.
    expect(model.generate({ seed: ["open"], steps: 3 })).toEqual(["search", "type", "submit"]);
  });

  it("ranks the next movement by observed frequency", async () => {
    const backend = createSimulatedMovementBackend();
    const model = backend.load(await backend.train(dataset, { order: 2 }));
    const predictions = model.predictNext(["search"]);
    expect(predictions[0]?.token).toBe("type"); // seen 2x vs scroll 1x after "search"
    expect(predictions[0]?.probability).toBeCloseTo(2 / 3, 6);
    expect(predictions.map((p) => p.token)).toEqual(["type", "scroll"]);
  });

  it("generalizes to an unseen prefix via backoff to a seen suffix", async () => {
    const backend = createSimulatedMovementBackend();
    const model = backend.load(await backend.train(dataset, { order: 2 }));
    // "close search" was never seen as a bigram; backoff to unigram "search"
    // still yields the plausible recorded continuation.
    const predictions = model.predictNext(["close", "search"]);
    expect(predictions[0]?.token).toBe("type");
    expect(predictions[0]?.matchedOrder).toBe(1);
  });

  it("is deterministic: identical training yields identical artifacts", async () => {
    const backend = new SimulatedMovementModelBackend();
    const a = await backend.train(dataset, { order: 3 });
    const b = await backend.train(dataset, { order: 3 });
    expect(JSON.stringify(a)).toEqual(JSON.stringify(b));
  });

  it("stops generation on a stop token", async () => {
    const backend = createSimulatedMovementBackend();
    const model = backend.load(await backend.train(dataset, { order: 2 }));
    expect(model.generate({ seed: ["open"], steps: 10, stop: ["submit"] })).toEqual(["search", "type", "submit"]);
  });

  it("handles an empty dataset without throwing", async () => {
    const backend = createSimulatedMovementBackend();
    const model = backend.load(await backend.train({ sequences: [] }, { order: 2 }));
    expect(model.predictNext(["anything"])).toEqual([]);
    expect(model.generate({ steps: 5 })).toEqual([]);
  });
});

describe("evaluateNextTokenAccuracy", () => {
  it("measures held-out generalization fidelity", async () => {
    const train: MovementDataset = {
      sequences: [
        { id: "a", tokens: ["open", "search", "type", "submit"] },
        { id: "b", tokens: ["open", "search", "type", "submit"] },
      ],
    };
    const heldOut: MovementDataset = {
      sequences: [{ id: "c", tokens: ["open", "search", "type", "submit"] }],
    };
    const backend = createSimulatedMovementBackend();
    const model = backend.load(await backend.train(train, { order: 2 }));
    const result = evaluateNextTokenAccuracy(model, heldOut);
    expect(result.total).toBe(3);
    expect(result.top1Accuracy).toBe(1); // every transition was learned
  });
});

describe("MovementBackendRegistry", () => {
  it("registers and resolves backends by id", () => {
    const registry = new MovementBackendRegistry().register(createSimulatedMovementBackend());
    expect(registry.has("simulated-ngram")).toBe(true);
    expect(registry.list()).toEqual(["simulated-ngram"]);
    expect(registry.get("simulated-ngram").id).toBe("simulated-ngram");
    expect(() => registry.get("nope")).toThrow(/unknown movement backend/);
  });

  it("createDefaultMovementBackendRegistry seeds the built-in backend and loads artifacts", async () => {
    const registry = createDefaultMovementBackendRegistry();
    const backend = registry.get("simulated-ngram");
    const artifact = await backend.train({ sequences: [{ id: "s", tokens: ["a", "b"] }] });
    const model = registry.load(artifact);
    expect(model.backendId).toBe("simulated-ngram");
    expect(model.generate({ seed: ["a"], steps: 1 })).toEqual(["b"]);
  });
});
