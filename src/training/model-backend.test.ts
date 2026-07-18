import { describe, expect, it } from "vitest";
import type { ReplayManifest } from "../capture/replay.js";
import type { TrajectorySpan } from "../capture/trajectory.js";
import {
  DEFAULT_MOVEMENT_BACKEND_ID,
  MOVEMENT_END_TOKEN,
  NGramMovementBackend,
  evaluateNextTokenAccuracy,
  getMovementBackend,
  listMovementBackends,
  registerMovementBackend,
  slugifyMovement,
  synthesizeMovementExamples,
  tokenizeAction,
  tokenizeReplayManifest,
  tokenizeTrajectorySpan,
  type LocalModelBackend,
  type MovementTrainingExample,
  type SerializedMovementModel,
} from "./model-backend.js";

const CONFIG = { order: 2 };

async function trainOn(examples: MovementTrainingExample[], order = 2) {
  return await new NGramMovementBackend().train(examples, { order });
}

describe("NGramMovementBackend", () => {
  it("repeats a memorized movement sequence exactly (replay)", async () => {
    const example: MovementTrainingExample = {
      sessionId: "s1",
      tokens: ["device:open_app", "device:tap_compose", "device:type_body", "device:tap_send"],
    };
    const model = await trainOn([example]);
    const generated = model.generate(["device:open_app"], 10);
    expect(generated).toEqual(["device:tap_compose", "device:type_body", "device:tap_send"]);
  });

  it("stops generation at the end boundary", async () => {
    const model = await trainOn([{ sessionId: "s", tokens: ["a", "b"] }]);
    const out = model.generate(["a"], 100);
    expect(out).toEqual(["b"]);
    expect(out).not.toContain(MOVEMENT_END_TOKEN);
  });

  it("predicts deterministically with lexicographic tie-breaking", async () => {
    // "a" is followed by "z" and "b" equally often -> lexicographically smaller "b" wins.
    const model = await trainOn(
      [
        { sessionId: "1", tokens: ["a", "z"] },
        { sessionId: "2", tokens: ["a", "b"] },
      ],
      1,
    );
    const prediction = model.predictNext(["a"]);
    expect(prediction?.token).toBe("b");
    expect(prediction?.matchedOrder).toBe(1);
    expect(prediction?.backoff).toBe(false);
  });

  it("generalizes to an unseen prefix via backoff", async () => {
    // The model never saw "warmup" before "tap_compose", but backoff to the
    // unigram/shorter context still predicts the dominant next movement.
    const model = await trainOn([
      { sessionId: "1", tokens: ["open_app", "tap_compose", "tap_send"] },
      { sessionId: "2", tokens: ["open_app", "tap_compose", "tap_send"] },
      { sessionId: "3", tokens: ["scroll", "tap_compose", "tap_send"] },
    ]);
    const prediction = model.predictNext(["warmup", "tap_compose"]);
    expect(prediction?.token).toBe("tap_send");
    // "warmup tap_compose" was never seen at order 2, so it backed off.
    expect(prediction?.backoff).toBe(true);
    expect(prediction?.matchedOrder).toBeLessThan(2);
  });

  it("returns undefined for an empty, untrained model", async () => {
    const model = await trainOn([]);
    expect(model.predictNext(["anything"])).toBeUndefined();
    expect(model.generate([], 5)).toEqual([]);
  });

  it("round-trips through serialize/load with identical predictions", async () => {
    const examples = synthesizeMovementExamples({
      templates: [
        { name: "compose", steps: ["open", "compose", "type", "send"] },
        { name: "search", steps: ["open", "search", "type", "submit"] },
      ],
      variantsPerTemplate: 6,
      seed: 7,
    });
    const backend = new NGramMovementBackend();
    const model = await backend.train(examples, CONFIG);
    const serialized: SerializedMovementModel = model.serialize();
    const restored = backend.load(serialized);

    expect(restored.vocabulary).toEqual(model.vocabulary);
    expect(restored.config).toEqual(model.config);
    for (const context of [["open"], ["open", "compose"], ["search"], ["warmup", "type"]]) {
      expect(restored.predictNext(context)).toEqual(model.predictNext(context));
    }
    // Serialized form must survive a JSON boundary (it is written to disk).
    expect(() => JSON.parse(JSON.stringify(serialized))).not.toThrow();
  });
});

describe("movement backend registry", () => {
  it("exposes the default n-gram backend", () => {
    expect(listMovementBackends()).toContain(DEFAULT_MOVEMENT_BACKEND_ID);
    expect(getMovementBackend().id).toBe(DEFAULT_MOVEMENT_BACKEND_ID);
  });

  it("allows registering and selecting a custom backend (pluggable seam)", () => {
    const stub: LocalModelBackend = {
      id: "stub-backend",
      async train() {
        throw new Error("not implemented");
      },
      load() {
        throw new Error("not implemented");
      },
    };
    registerMovementBackend(stub);
    expect(getMovementBackend("stub-backend")).toBe(stub);
    expect(listMovementBackends()).toContain("stub-backend");
  });

  it("throws a helpful error for an unknown backend", () => {
    expect(() => getMovementBackend("does-not-exist")).toThrow(/no movement model backend/);
  });
});

describe("dataset adapters", () => {
  it("slugifies free-text summaries into stable tokens", () => {
    expect(slugifyMovement("Tapped Compose Button!")).toBe("tapped_compose_button");
    expect(slugifyMovement("   ")).toBe("unknown");
  });

  it("tokenizes a trajectory span ordered by timestamp", () => {
    const span: TrajectorySpan = {
      id: "t1",
      sessionId: "s1",
      createdAt: "2026-01-01T00:00:00.000Z",
      captureTier: "app",
      observations: [],
      actions: [
        { kind: "action", tool: "device", summary: "tapped send", ts: 30 },
        { kind: "action", tool: "device", summary: "tapped compose", ts: 10 },
      ],
    };
    const example = tokenizeTrajectorySpan(span);
    expect(example.sessionId).toBe("s1");
    expect(example.tokens).toEqual(["device:tapped_compose", "device:tapped_send"]);
  });

  it("prefers redacted review actions when present (export-safe)", () => {
    const span: TrajectorySpan = {
      id: "t2",
      sessionId: "s2",
      createdAt: "2026-01-01T00:00:00.000Z",
      captureTier: "app",
      observations: [],
      actions: [{ kind: "action", tool: "device", summary: "raw secret", ts: 5 }],
      review: {
        status: "approved",
        reviewedAt: "2026-01-02T00:00:00.000Z",
        reviewedBy: "reviewer",
        redactedActions: [{ tool: "device", summary: "tapped ok", ts: 5 }],
      },
    };
    expect(tokenizeTrajectorySpan(span).tokens).toEqual(["device:tapped_ok"]);
  });

  it("tokenizes a replay manifest's action events only", () => {
    const manifest: ReplayManifest = {
      version: 1,
      sessionId: "s3",
      trajectoryIds: ["t"],
      eventCount: 3,
      events: [
        { kind: "observation", ts: 1, trajectoryId: "t", source: "device", summary: "screen" },
        { kind: "action", ts: 2, trajectoryId: "t", tool: "device", summary: "tap a" },
        { kind: "action", ts: 3, trajectoryId: "t", tool: "device", summary: "tap b" },
      ],
    };
    expect(tokenizeReplayManifest(manifest).tokens).toEqual(["device:tap_a", "device:tap_b"]);
  });

  it("produces the same token via tokenizeAction", () => {
    expect(tokenizeAction("Device", "Tapped Send")).toBe("device:tapped_send");
  });
});

describe("synthetic streams + generalization eval", () => {
  it("is deterministic for a fixed seed", () => {
    const templates = [{ name: "flow", steps: ["a", "b", "c", "d"] }];
    const first = synthesizeMovementExamples({ templates, variantsPerTemplate: 5, seed: 42 });
    const second = synthesizeMovementExamples({ templates, variantsPerTemplate: 5, seed: 42 });
    expect(first).toEqual(second);
    expect(first).toHaveLength(5);
  });

  it("trains on synthetic variants and generalizes to held-out ones", async () => {
    // Distinct openers make the task branch resolvable from context; the model
    // must still generalize over the drop/duplicate noise in held-out variants.
    const templates = [
      { name: "compose", steps: ["open_compose", "tap_compose", "type_subject", "type_body", "tap_send"] },
      { name: "reply", steps: ["open_thread", "tap_message", "tap_reply", "type_body", "tap_send"] },
      { name: "search", steps: ["open_search", "tap_search", "type_query", "tap_result"] },
    ];
    const all = synthesizeMovementExamples({ templates, variantsPerTemplate: 30, seed: 123 });
    // Interleave so train/held-out both cover every template.
    const train = all.filter((_, i) => i % 5 !== 0);
    const heldOut = all.filter((_, i) => i % 5 === 0);

    const model = await trainOn(train, 3);
    const eval_ = evaluateNextTokenAccuracy(model, heldOut);

    expect(eval_.predictions).toBeGreaterThan(0);
    // Held-out variants were never seen, yet share structure -> the model
    // predicts most next-movements correctly despite drop/duplicate noise
    // (well above the ~1/vocab random baseline).
    expect(eval_.accuracy).toBeGreaterThan(0.7);
    // Some predictions must have relied on backoff (proof of generalization,
    // not pure memorization of full-order contexts).
    expect(eval_.backoffs).toBeGreaterThan(0);
  });

  it("scores a memorized example at perfect accuracy", async () => {
    const example = { sessionId: "s", tokens: ["a", "b", "c"] };
    const model = await trainOn([example], 3);
    const eval_ = evaluateNextTokenAccuracy(model, [example]);
    expect(eval_.accuracy).toBe(1);
  });
});
