import { describe, expect, it } from "vitest";
import type { ReplayManifest } from "../capture/replay.js";
import {
  buildMovementSequences,
  createDefaultMovementBackendRegistry,
  evaluateMovementModel,
  MovementBackendRegistry,
  NgramMovementBackend,
  rolloutMovements,
  tokenizeReplayEvent,
  type MovementToken,
} from "./backend.js";

const backend = new NgramMovementBackend("mock");

async function train(sequences: MovementToken[][], maxOrder = 2) {
  return await backend.train({ sequences, maxOrder });
}

describe("tokenizeReplayEvent", () => {
  it("reduces each event kind to its coarse movement class", () => {
    expect(tokenizeReplayEvent({ kind: "action", ts: 1, trajectoryId: "t", tool: "Bash", summary: "ls -a" })).toBe(
      "action:Bash",
    );
    expect(
      tokenizeReplayEvent({ kind: "observation", ts: 1, trajectoryId: "t", source: "window", summary: "focus" }),
    ).toBe("observation:window");
    expect(tokenizeReplayEvent({ kind: "transcript", ts: 1, messageId: "m", role: "user", content: "hi" })).toBe(
      "transcript:user",
    );
  });

  it("maps two events that differ only in free-text summary to the same token", () => {
    const a = tokenizeReplayEvent({ kind: "action", ts: 1, trajectoryId: "t", tool: "Click", summary: "at (10,10)" });
    const b = tokenizeReplayEvent({ kind: "action", ts: 2, trajectoryId: "t", tool: "Click", summary: "at (99,42)" });
    expect(a).toBe(b);
  });
});

describe("buildMovementSequences", () => {
  it("derives one ordered token sequence per replay manifest", () => {
    const replay: Pick<ReplayManifest, "events"> = {
      events: [
        { kind: "observation", ts: 1, trajectoryId: "t", source: "window", summary: "focus editor" },
        { kind: "action", ts: 2, trajectoryId: "t", tool: "Click", summary: "menu" },
        { kind: "action", ts: 3, trajectoryId: "t", tool: "Type", summary: "hello" },
      ],
    };
    expect(buildMovementSequences([replay])).toEqual([["observation:window", "action:Click", "action:Type"]]);
  });
});

describe("NgramMovementBackend training + inference", () => {
  it("reproduces a recorded movement exactly (memorized next step)", async () => {
    const seq: MovementToken[] = ["observation:window", "action:Click", "action:Type", "action:Submit"];
    const model = await train([seq]);
    const prediction = backend.predict(model, ["observation:window", "action:Click"]);
    expect(prediction.token).toBe("action:Type");
    expect(prediction.order).toBe(2);
    expect(prediction.confidence).toBe(1);
  });

  it("greedy rollout regenerates the full recorded trajectory from a seed", async () => {
    const seq: MovementToken[] = ["action:Open", "action:Click", "action:Type", "action:Submit"];
    const model = await train([seq]);
    const generated = rolloutMovements(backend, model, ["action:Open"], 3);
    expect(generated).toEqual(["action:Click", "action:Type", "action:Submit"]);
  });

  it("generalizes to an unseen context by backing off to a shorter n-gram", async () => {
    // "action:Type" is always followed by "action:Submit" across the corpus,
    // regardless of what preceded the Type. A never-seen full context should
    // still recover Submit via order-1 backoff.
    const model = await train([
      ["action:Open", "action:Type", "action:Submit"],
      ["action:Focus", "action:Type", "action:Submit"],
    ]);
    const prediction = backend.predict(model, ["observation:brand-new", "action:Type"]);
    expect(prediction.token).toBe("action:Submit");
    expect(prediction.order).toBeLessThan(2); // it generalized, did not memorize
  });

  it("breaks ties deterministically by token order", async () => {
    // From "action:Start", Bravo and Alpha each occur once → tie broken A→Z.
    const model = await train([
      ["action:Start", "action:Bravo"],
      ["action:Start", "action:Alpha"],
    ]);
    const prediction = backend.predict(model, ["action:Start"]);
    expect(prediction.token).toBe("action:Alpha");
    expect(prediction.confidence).toBeCloseTo(0.5);
    expect(prediction.alternatives[0]?.token).toBe("action:Bravo");
  });

  it("falls back to the unconditional distribution for a cold-start context", async () => {
    const model = await train([["action:A", "action:A", "action:A", "action:B"]]);
    const prediction = backend.predict(model, []);
    expect(prediction.order).toBe(0);
    expect(prediction.token).toBe("action:A"); // most frequent token overall
  });

  it("returns a null prediction from an empty model", async () => {
    const model = await train([]);
    const prediction = backend.predict(model, ["anything"]);
    expect(prediction.token).toBeNull();
    expect(prediction.confidence).toBe(0);
    expect(prediction.order).toBe(-1);
  });

  it("produces a deterministic, JSON-serializable model artifact", async () => {
    const sequences: MovementToken[][] = [["action:A", "action:B", "action:A", "action:B"]];
    const first = await train(sequences);
    const second = await train(sequences);
    expect(second).toEqual(first);
    const roundTripped = JSON.parse(JSON.stringify(first));
    expect(roundTripped).toEqual(first);
    expect(first.vocabulary).toEqual(["action:A", "action:B"]);
    expect(first.tokenCount).toBe(4);
    expect(first.sequenceCount).toBe(1);
  });
});

describe("evaluateMovementModel", () => {
  it("scores perfect next-token accuracy on the training sequence", async () => {
    const seq: MovementToken[] = ["action:A", "action:B", "action:C", "action:D"];
    const model = await train([seq]);
    const evaluation = evaluateMovementModel(backend, model, [seq]);
    expect(evaluation.accuracy).toBe(1);
    expect(evaluation.predictions).toBe(3);
  });

  it("measures generalization on held-out but related sequences via backoff", async () => {
    const model = await train([
      ["action:Login", "action:Type", "action:Submit"],
      ["action:Search", "action:Type", "action:Submit"],
    ]);
    // Held-out: a novel prefix leading into the learned Type→Submit habit.
    const heldOut: MovementToken[][] = [["action:Filter", "action:Type", "action:Submit"]];
    const evaluation = evaluateMovementModel(backend, model, heldOut);
    expect(evaluation.correct).toBeGreaterThan(0);
    expect(evaluation.generalizedFraction).toBeGreaterThan(0);
  });
});

describe("MovementBackendRegistry", () => {
  it("resolves the default mock backend and lists it", () => {
    const registry = createDefaultMovementBackendRegistry();
    expect(registry.has("mock")).toBe(true);
    expect(registry.list()).toContain("mock");
    expect(registry.get("mock")).toBeInstanceOf(NgramMovementBackend);
  });

  it("throws a helpful error for an unregistered backend", () => {
    const registry = new MovementBackendRegistry();
    expect(() => registry.get("real-device")).toThrowError(/Unknown movement backend "real-device"/);
  });

  it("lets a custom backend be registered under its own name", async () => {
    const registry = createDefaultMovementBackendRegistry().register(new NgramMovementBackend("device-lora"));
    expect(registry.list()).toEqual(["device-lora", "mock"]);
    const model = await registry.get("device-lora").train({ sequences: [["action:A", "action:B"]] });
    expect(model.backend).toBe("device-lora");
  });
});
