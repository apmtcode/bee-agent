import { describe, expect, it } from "vitest";
import {
  MarkovMovementBackend,
  createDefaultMovementBackend,
} from "./backends/markov-movement-backend.js";
import {
  MovementModelBackendRegistry,
  buildMovementDataset,
  encodeMovementToken,
  evaluateMovementModel,
  tokenizeReplayManifest,
  tokenizeTrajectory,
  type MovementDataset,
  type MovementSequence,
} from "./movement-model.js";
import { buildTrajectorySpan, type TrajectorySpan } from "../capture/trajectory.js";
import { buildReplayManifest } from "../capture/replay.js";

function syntheticSequence(id: string, tokens: string[]): MovementSequence {
  return { id, tokens };
}

function syntheticDataset(sequences: MovementSequence[]): MovementDataset {
  return { version: 1, sequences };
}

describe("encodeMovementToken", () => {
  it("normalizes whitespace and case so equivalent movements collapse", () => {
    expect(encodeMovementToken("Device", "Tapped   Submit")).toBe("device:tapped submit");
    expect(encodeMovementToken("device", "tapped submit")).toBe(
      encodeMovementToken("DEVICE", " tapped  submit "),
    );
  });
});

describe("tokenizeTrajectory", () => {
  it("orders actions by timestamp and encodes them", () => {
    const trajectory: TrajectorySpan = buildTrajectorySpan({
      id: "t1",
      sessionId: "s1",
      actions: [
        { kind: "action", tool: "device", summary: "tapped B", ts: 20 },
        { kind: "action", tool: "device", summary: "tapped A", ts: 10 },
      ],
    });
    expect(tokenizeTrajectory(trajectory).tokens).toEqual(["device:tapped a", "device:tapped b"]);
  });

  it("prefers redacted actions from a review when present", () => {
    const trajectory: TrajectorySpan = {
      ...buildTrajectorySpan({
        id: "t2",
        sessionId: "s1",
        actions: [{ kind: "action", tool: "device", summary: "raw secret", ts: 5 }],
      }),
      review: {
        status: "approved",
        reviewedAt: "2026-01-01T00:00:00.000Z",
        reviewedBy: "tester",
        redactedActions: [{ ts: 5, tool: "device", summary: "tapped login" }],
      },
    };
    expect(tokenizeTrajectory(trajectory).tokens).toEqual(["device:tapped login"]);
  });
});

describe("tokenizeReplayManifest", () => {
  it("extracts per-trajectory action sequences in time order", () => {
    const trajectory = buildTrajectorySpan({
      id: "tr",
      sessionId: "s1",
      actions: [
        { kind: "action", tool: "device", summary: "open app", ts: 1 },
        { kind: "action", tool: "device", summary: "type query", ts: 2 },
      ],
    });
    const manifest = buildReplayManifest({ sessionId: "s1", transcript: [], trajectories: [trajectory] });
    const sequences = tokenizeReplayManifest(manifest);
    expect(sequences).toEqual([
      { id: "tr", tokens: ["device:open app", "device:type query"] },
    ]);
  });
});

describe("buildMovementDataset", () => {
  it("drops trajectories with no actions", () => {
    const withActions = buildTrajectorySpan({
      id: "a",
      sessionId: "s",
      actions: [{ kind: "action", tool: "device", summary: "tap", ts: 1 }],
    });
    const empty = buildTrajectorySpan({ id: "b", sessionId: "s" });
    const dataset = buildMovementDataset([withActions, empty]);
    expect(dataset.sequences).toHaveLength(1);
    expect(dataset.sequences[0]?.id).toBe("a");
  });
});

describe("MarkovMovementBackend", () => {
  it("trains a serializable model with the expected envelope", async () => {
    const backend = new MarkovMovementBackend();
    const dataset = syntheticDataset([syntheticSequence("s1", ["a", "b", "c"])]);
    const model = await backend.train(dataset, { order: 2 });
    expect(model.backendId).toBe("markov-ngram");
    expect(model.order).toBe(2);
    expect(model.vocabulary).toEqual(["a", "b", "c"]);
    expect(model.metadata.sequenceCount).toBe(1);
    expect(model.metadata.tokenCount).toBe(3);
    // Model survives a JSON round-trip (persistence requirement).
    const roundTripped = JSON.parse(JSON.stringify(model));
    expect(backend.predict(roundTripped, ["a"]).token).toBe("b");
  });

  it("repeats a recorded movement sequence exactly (objective 2c)", async () => {
    const backend = createDefaultMovementBackend();
    const recorded = ["open:app", "click:menu", "click:settings", "toggle:dark-mode"];
    const model = await backend.train(syntheticDataset([syntheticSequence("rec", recorded)]), { order: 3 });
    const replayed = backend.generate(model, [recorded[0] as string], recorded.length - 1);
    expect([recorded[0], ...replayed]).toEqual(recorded);
  });

  it("generalizes to a new-but-related context via back-off (objective 2d)", async () => {
    const backend = new MarkovMovementBackend();
    // In every training context, "open:editor" is followed by "type:text".
    const dataset = syntheticDataset([
      syntheticSequence("s1", ["focus:window-a", "open:editor", "type:text"]),
      syntheticSequence("s2", ["focus:window-b", "open:editor", "type:text"]),
      syntheticSequence("s3", ["scroll:down", "open:editor", "type:text"]),
    ]);
    const model = await backend.train(dataset, { order: 3 });
    // A novel prefix never seen during training, ending in the known cue.
    const prediction = backend.predict(model, ["resize:window-c", "open:editor"]);
    expect(prediction.token).toBe("type:text");
    // It had to back off below full order because the trigram was unseen.
    expect(prediction.backoffOrder).toBeLessThan(3);
  });

  it("ranks continuations by probability with deterministic tie-breaks", async () => {
    const backend = new MarkovMovementBackend();
    const dataset = syntheticDataset([
      syntheticSequence("s1", ["start", "go-b"]),
      syntheticSequence("s2", ["start", "go-b"]),
      syntheticSequence("s3", ["start", "go-a"]),
    ]);
    const model = await backend.train(dataset, { order: 1 });
    const prediction = backend.predict(model, ["start"]);
    expect(prediction.ranked.map((candidate) => candidate.token)).toEqual(["go-b", "go-a"]);
    expect(prediction.ranked[0]?.probability).toBeGreaterThan(prediction.ranked[1]?.probability ?? 0);
  });

  it("returns an empty prediction for an empty model", async () => {
    const backend = new MarkovMovementBackend();
    const model = await backend.train(syntheticDataset([]));
    const prediction = backend.predict(model, ["anything"]);
    expect(prediction.token).toBeUndefined();
    expect(prediction.ranked).toEqual([]);
  });

  it("stops generation when no continuation is known", async () => {
    const backend = new MarkovMovementBackend();
    const model = await backend.train(syntheticDataset([syntheticSequence("s", ["a", "b"])]), { order: 1 });
    // "b" was never followed by anything, and the unigram only knows a,b.
    const generated = backend.generate(model, ["b"], 5);
    expect(generated.length).toBeLessThanOrEqual(5);
  });
});

describe("evaluateMovementModel (generalization harness)", () => {
  it("reports high next-token accuracy on held-out related sequences", async () => {
    const backend = new MarkovMovementBackend();
    const train = syntheticDataset([
      syntheticSequence("s1", ["a", "b", "c", "d"]),
      syntheticSequence("s2", ["a", "b", "c", "d"]),
      syntheticSequence("s3", ["x", "b", "c", "d"]),
    ]);
    const model = await backend.train(train, { order: 2 });
    // Held-out sequence sharing the b->c->d structure under a novel prefix.
    const heldOut: MovementSequence[] = [syntheticSequence("eval", ["z", "b", "c", "d"])];
    const evaluation = evaluateMovementModel(backend, model, heldOut);
    expect(evaluation.predictions).toBe(3);
    expect(evaluation.nextTokenAccuracy).toBeGreaterThanOrEqual(2 / 3);
    expect(evaluation.backoffRate).toBeGreaterThan(0);
  });

  it("scores a perfectly memorized sequence at 100% accuracy", async () => {
    const backend = new MarkovMovementBackend();
    const seq = syntheticSequence("s", ["m1", "m2", "m3", "m4"]);
    const model = await backend.train(syntheticDataset([seq]), { order: 3 });
    const evaluation = evaluateMovementModel(backend, model, [seq]);
    expect(evaluation.nextTokenAccuracy).toBe(1);
  });
});

describe("MovementModelBackendRegistry", () => {
  it("registers, resolves, and lists backends", () => {
    const registry = new MovementModelBackendRegistry();
    const backend = new MarkovMovementBackend();
    registry.register(backend);
    expect(registry.get("markov-ngram")).toBe(backend);
    expect(registry.require("markov-ngram")).toBe(backend);
    expect(registry.list()).toEqual([backend]);
  });

  it("throws a clear error for an unknown backend", () => {
    const registry = new MovementModelBackendRegistry();
    expect(() => registry.require("nope")).toThrow(/unknown movement-model backend: nope/);
  });
});
