import { describe, expect, it } from "vitest";
import {
  MarkovMovementBackend,
  MOVEMENT_START_TOKEN,
  buildMovementDataset,
  buildMovementDatasetFromReplays,
  defaultMovementTokenizer,
  evaluateNextMovement,
  generateMovements,
  splitMovementDataset,
  type MovementActionLike,
} from "./movement-model.js";

/**
 * Synthetic movement-stream generator. Stands in for real on-device capture:
 * deterministically produces device-gesture actions following a repeatable
 * "motif" so training/eval are reproducible in the cloud. No clock, no random.
 */
function syntheticGestureStream(motif: string[][]): { id: string; actions: MovementActionLike[] }[] {
  return motif.map((gestures, index) => ({
    id: `session-${index}`,
    actions: gestures.map((spec, step) => {
      const [gesture, qualifier] = spec.split(":");
      return {
        tool: "device",
        summary: `${gesture} ${qualifier ?? ""}`.trim(),
        ts: step,
        metadata: qualifier
          ? { gesture, [gesture === "swipe" || gesture === "scroll" ? "direction" : "target"]: qualifier }
          : { gesture },
      } satisfies MovementActionLike;
    }),
  }));
}

describe("defaultMovementTokenizer", () => {
  it("prefers structured gesture metadata over prose", () => {
    expect(
      defaultMovementTokenizer({ tool: "device", summary: "swiped up", metadata: { gesture: "swipe", direction: "up" } }),
    ).toBe("swipe:up");
    expect(
      defaultMovementTokenizer({ tool: "device", summary: "tapped Send", metadata: { gesture: "tap", target: "Send" } }),
    ).toBe("tap:send");
  });

  it("falls back to a tool:summary slug for non-gesture actions", () => {
    expect(defaultMovementTokenizer({ tool: "Bash", summary: "Run tests" })).toBe("bash:run-tests");
  });
});

describe("buildMovementDataset", () => {
  it("builds sequences and a sorted vocabulary, dropping empty sources", () => {
    const dataset = buildMovementDataset({
      order: 2,
      sources: [
        ...syntheticGestureStream([["tap:a", "swipe:up", "tap:b"]]),
        { id: "empty", actions: [] },
      ],
    });
    expect(dataset.sequences).toHaveLength(1);
    expect(dataset.sequences[0].tokens).toEqual(["tap:a", "swipe:up", "tap:b"]);
    expect(dataset.vocabulary).toEqual(["swipe:up", "tap:a", "tap:b"]);
    expect(dataset.order).toBe(2);
  });

  it("clamps an invalid order to the default", () => {
    const dataset = buildMovementDataset({ order: 0, sources: syntheticGestureStream([["tap:a"]]) });
    expect(dataset.order).toBe(3);
  });
});

describe("buildMovementDatasetFromReplays", () => {
  it("extracts only action events into movement sequences", () => {
    const dataset = buildMovementDatasetFromReplays({
      order: 2,
      replays: [
        {
          sessionId: "s1",
          events: [
            { kind: "transcript" },
            { kind: "observation", summary: "screen" },
            { kind: "action", tool: "device", summary: "swiped up", metadata: { gesture: "swipe", direction: "up" } },
            { kind: "action", tool: "device", summary: "tapped Send", metadata: { gesture: "tap", target: "Send" } },
          ],
        },
      ],
    });
    expect(dataset.sequences[0].tokens).toEqual(["swipe:up", "tap:send"]);
  });
});

describe("MarkovMovementBackend", () => {
  const backend = new MarkovMovementBackend();

  it("reproduces a recorded movement exactly from seen context", () => {
    const dataset = buildMovementDataset({
      order: 3,
      sources: syntheticGestureStream([["tap:a", "swipe:up", "tap:b", "scroll:down"]]),
    });
    const policy = backend.train(dataset);

    // With full context the next movement is deterministic and confident.
    const prediction = backend.predict(policy, ["tap:a", "swipe:up", "tap:b"]);
    expect(prediction.token).toBe("scroll:down");
    expect(prediction.confidence).toBe(1);
    expect(prediction.contextOrderUsed).toBe(3);
  });

  it("predicts the first movement via the START context", () => {
    const dataset = buildMovementDataset({ sources: syntheticGestureStream([["tap:a", "swipe:up"]]) });
    const policy = backend.train(dataset);
    const prediction = backend.predict(policy, []);
    expect(prediction.token).toBe("tap:a");
    // START is recorded in the transitions bucket for the empty context.
    expect(policy.transitions[MOVEMENT_START_TOKEN]["tap:a"]).toBe(1);
  });

  it("backs off to shorter context for novel prefixes (generalization)", () => {
    // Two motifs share the bigram (swipe:up -> tap:send); a novel 3-gram prefix
    // must still predict the shared continuation by backing off.
    const dataset = buildMovementDataset({
      order: 3,
      sources: syntheticGestureStream([
        ["tap:a", "swipe:up", "tap:send"],
        ["scroll:down", "swipe:up", "tap:send"],
      ]),
    });
    const policy = backend.train(dataset);

    const prediction = backend.predict(policy, ["tap:z", "swipe:up"]);
    expect(prediction.token).toBe("tap:send");
    expect(prediction.contextOrderUsed).toBe(1); // full 2-token context "tap:z swipe:up" unseen -> backoff to 1
  });

  it("returns an empty prediction for an untrained/empty policy", () => {
    const policy = backend.train(buildMovementDataset({ sources: [] }));
    const prediction = backend.predict(policy, ["tap:a"]);
    expect(prediction.token).toBeUndefined();
    expect(prediction.confidence).toBe(0);
    expect(prediction.distribution).toEqual([]);
  });

  it("produces a JSON-serializable policy (round-trips through JSON)", () => {
    const dataset = buildMovementDataset({ sources: syntheticGestureStream([["tap:a", "swipe:up"]]) });
    const policy = backend.train(dataset);
    const restored = JSON.parse(JSON.stringify(policy));
    expect(backend.predict(restored, ["tap:a"])).toEqual(backend.predict(policy, ["tap:a"]));
  });
});

describe("generateMovements", () => {
  const backend = new MarkovMovementBackend();

  it("regenerates the dominant recorded path (repeat objective)", () => {
    const dataset = buildMovementDataset({
      order: 3,
      sources: syntheticGestureStream([["tap:a", "swipe:up", "tap:b", "scroll:down"]]),
    });
    const policy = backend.train(dataset);
    const generated = generateMovements({ backend, policy, maxLength: 4 });
    expect(generated.tokens).toEqual(["tap:a", "swipe:up", "tap:b", "scroll:down"]);
  });

  it("continues from a seed and stops below the confidence floor", () => {
    const dataset = buildMovementDataset({
      order: 2,
      sources: syntheticGestureStream([
        ["tap:a", "swipe:up", "tap:b"],
        ["tap:a", "swipe:up", "tap:c"],
      ]),
    });
    const policy = backend.train(dataset);
    // After "tap:a swipe:up" the next token splits 50/50 -> below a 0.9 floor.
    const generated = generateMovements({
      backend,
      policy,
      seed: ["tap:a", "swipe:up"],
      maxLength: 5,
      minConfidence: 0.9,
    });
    expect(generated.tokens).toEqual([]);
  });
});

describe("evaluateNextMovement + splitMovementDataset", () => {
  const backend = new MarkovMovementBackend();

  it("scores perfect fidelity when held-out sequences repeat a learned motif", () => {
    const motif = ["tap:home", "swipe:up", "tap:app", "scroll:down", "tap:done"];
    // Ten identical repetitions -> holdout is drawn from the same motif.
    const dataset = buildMovementDataset({
      order: 3,
      sources: syntheticGestureStream(Array.from({ length: 10 }, () => [...motif])),
    });
    const { train, holdout } = splitMovementDataset(dataset, 4);
    expect(holdout.length).toBeGreaterThan(0);

    const policy = backend.train(train);
    const result = evaluateNextMovement({ backend, policy, sequences: holdout });

    expect(result.accuracy).toBe(1);
    expect(result.total).toBe(holdout.length * motif.length);
    expect(result.averageConfidence).toBeGreaterThan(0);
  });

  it("generalizes above chance on held-out related-but-novel motifs", () => {
    // A shared skeleton (open -> menu -> settings -> <mid> -> save -> confirm)
    // where only the single middle movement varies. The exact sequences are
    // never trained on, but backoff recovers the shared prefix AND the shared
    // "save -> confirm" tail across the novel middle token.
    const sources = [
      { id: "train-a", actions: gestures(["tap:open", "tap:menu", "tap:settings", "swipe:up", "tap:save", "tap:confirm"]) },
      { id: "holdout-a", actions: gestures(["tap:open", "tap:menu", "tap:settings", "swipe:down", "tap:save", "tap:confirm"]) },
      { id: "train-b", actions: gestures(["tap:open", "tap:menu", "tap:settings", "scroll:up", "tap:save", "tap:confirm"]) },
      { id: "holdout-b", actions: gestures(["tap:open", "tap:menu", "tap:settings", "scroll:down", "tap:save", "tap:confirm"]) },
    ];
    const dataset = buildMovementDataset({ order: 2, sources });
    const { train, holdout } = splitMovementDataset(dataset, 2);
    const policy = backend.train(train);
    const result = evaluateNextMovement({ backend, policy, sequences: holdout });

    // Only the novel middle step (and the one action after it) are unpredictable;
    // the shared 3-token prefix and "save -> confirm" tail are recovered -> ~0.67.
    expect(result.accuracy).toBeGreaterThan(0.5);
  });

  it("reports zeros for an empty held-out set", () => {
    const policy = backend.train(buildMovementDataset({ sources: syntheticGestureStream([["tap:a"]]) }));
    const result = evaluateNextMovement({ backend, policy, sequences: [] });
    expect(result).toMatchObject({ total: 0, correct: 0, accuracy: 0, averageConfidence: 0 });
  });
});

function gestures(specs: string[]): MovementActionLike[] {
  return specs.map((spec, step) => {
    const [gesture, qualifier] = spec.split(":");
    return {
      tool: "device",
      summary: spec,
      ts: step,
      metadata: qualifier
        ? { gesture, [gesture === "swipe" || gesture === "scroll" ? "direction" : "target"]: qualifier }
        : { gesture },
    };
  });
}
