import { describe, expect, it } from "vitest";
import type { ReplayManifest } from "../capture/replay.js";
import {
  MarkovMovementBackend,
  MovementBackendRegistry,
  buildMovementDataset,
  createDefaultMovementBackendRegistry,
  restoreMovementModel,
  tokenizeReplayEvent,
  type MovementDataset,
} from "./movement-model.js";

function replay(events: ReplayManifest["events"]): Pick<ReplayManifest, "events"> {
  return { events };
}

describe("tokenizeReplayEvent", () => {
  it("derives stable tokens per event kind", () => {
    expect(tokenizeReplayEvent({ kind: "action", ts: 1, trajectoryId: "t", tool: "browser", summary: "" })).toBe(
      "action:browser",
    );
    expect(
      tokenizeReplayEvent({ kind: "observation", ts: 1, trajectoryId: "t", source: "screen", summary: "" }),
    ).toBe("observation:screen");
    expect(
      tokenizeReplayEvent({ kind: "transcript", ts: 1, messageId: "m", role: "assistant", content: "" }),
    ).toBe("transcript:assistant");
  });
});

describe("buildMovementDataset", () => {
  it("produces one token sequence per non-empty replay", () => {
    const dataset = buildMovementDataset([
      replay([
        { kind: "observation", ts: 1, trajectoryId: "t", source: "screen", summary: "" },
        { kind: "action", ts: 2, trajectoryId: "t", tool: "mouse", summary: "" },
      ]),
      replay([]),
    ]);
    expect(dataset.sequences).toEqual([["observation:screen", "action:mouse"]]);
  });
});

describe("MarkovMovementBackend", () => {
  const dataset: MovementDataset = {
    sequences: [
      ["observation:screen", "action:mouse", "action:keyboard", "action:submit"],
    ],
  };

  it("repeats a recorded movement sequence exactly (objective 2c)", () => {
    const model = new MarkovMovementBackend().train(dataset, { order: 2 });
    const continuation = model.generate(["observation:screen"], 3);
    expect(continuation).toEqual(["action:mouse", "action:keyboard", "action:submit"]);

    const fidelity = model.evaluate(dataset.sequences[0]!);
    expect(fidelity.accuracy).toBe(1);
    expect(fidelity.correct).toBe(fidelity.steps);
  });

  it("generalizes to a new-but-related context via backoff (objective 2d)", () => {
    // Two sequences share the "action:mouse -> action:keyboard" transition.
    const model = new MarkovMovementBackend().train(
      {
        sequences: [
          ["observation:window", "action:mouse", "action:keyboard"],
          ["observation:menu", "action:mouse", "action:keyboard"],
        ],
      },
      { order: 2 },
    );

    // Context never seen at full order-2 ("observation:screen" + "action:mouse"),
    // but the order-1 suffix "action:mouse" is known -> backs off and predicts.
    const prediction = model.predictNext(["observation:screen", "action:mouse"]);
    expect(prediction).toBeDefined();
    expect(prediction!.token).toBe("action:keyboard");
    expect(prediction!.backedOff).toBe(true);
    expect(prediction!.contextUsed).toEqual(["action:mouse"]);
  });

  it("reports zero-step fidelity for trivial sequences without dividing by zero", () => {
    const model = new MarkovMovementBackend().train(dataset);
    const report = model.evaluate(["observation:screen"]);
    expect(report).toEqual({ steps: 0, correct: 0, accuracy: 0, backoffRate: 0 });
  });

  it("returns undefined predictions from an empty (untrained) model", () => {
    const model = new MarkovMovementBackend().train({ sequences: [] });
    expect(model.predictNext(["anything"])).toBeUndefined();
    expect(model.generate(["anything"], 5)).toEqual([]);
    expect(model.vocabulary).toEqual([]);
  });

  it("is deterministic: identical dataset produces identical serialized models", () => {
    const backend = new MarkovMovementBackend();
    expect(backend.train(dataset, { order: 2 }).toJSON()).toEqual(
      backend.train(dataset, { order: 2 }).toJSON(),
    );
  });

  it("round-trips through serialization", () => {
    const model = new MarkovMovementBackend().train(dataset, { order: 2 });
    const restored = restoreMovementModel(model.toJSON());
    expect(restored.toJSON()).toEqual(model.toJSON());
    expect(restored.generate(["observation:screen"], 3)).toEqual(
      model.generate(["observation:screen"], 3),
    );
  });
});

describe("MovementBackendRegistry", () => {
  it("resolves the default mock backend and rejects unknown ids", () => {
    const registry = createDefaultMovementBackendRegistry();
    expect(registry.list()).toEqual(["markov-mock"]);
    expect(registry.resolve("markov-mock").id).toBe("markov-mock");
    expect(() => registry.resolve("mlx-lora")).toThrow(/Unknown movement-model backend/);
  });

  it("accepts pluggable custom backends", () => {
    const registry = new MovementBackendRegistry().register({
      id: "stub",
      train: () => new MarkovMovementBackend().train({ sequences: [] }),
    });
    expect(registry.has("stub")).toBe(true);
    expect(registry.resolve("stub").id).toBe("stub");
  });
});
