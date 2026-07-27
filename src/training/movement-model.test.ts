import { describe, expect, it } from "vitest";
import type { TrajectorySpan } from "../capture/trajectory.js";
import {
  MarkovMovementBackend,
  MovementModelRegistry,
  buildMovementDataset,
  tokenizeAction,
  type MovementDataset,
} from "./movement-model.js";

function trajectory(id: string, gestures: Array<{ gesture: string; direction?: string }>): TrajectorySpan {
  return {
    id,
    sessionId: `session-${id}`,
    createdAt: "2026-07-27T00:00:00.000Z",
    captureTier: "app",
    observations: [],
    actions: gestures.map((g, index) => ({
      kind: "action",
      tool: "device",
      summary: `${g.gesture} ${g.direction ?? ""}`.trim(),
      ts: index,
      metadata: { gesture: g.gesture, ...(g.direction ? { direction: g.direction } : {}) },
    })),
  };
}

describe("tokenizeAction", () => {
  it("prefers structured gesture metadata over free-text summary", () => {
    expect(
      tokenizeAction({ tool: "device", summary: "swiped down on the feed", metadata: { gesture: "swipe", direction: "down" } }),
    ).toBe("device:swipe:down");
  });

  it("falls back to a slug of the summary when no structured metadata exists", () => {
    expect(tokenizeAction({ tool: "shell", summary: "Run npm test" })).toBe("shell:run-npm-test");
  });

  it("collapses semantically identical movements to the same token", () => {
    const a = tokenizeAction({ tool: "device", summary: "swiped downward", metadata: { gesture: "swipe", direction: "down" } });
    const b = tokenizeAction({ tool: "device", summary: "a downward swipe gesture", metadata: { gesture: "swipe", direction: "down" } });
    expect(a).toBe(b);
  });
});

describe("buildMovementDataset", () => {
  it("derives one token sequence per trajectory with actions", () => {
    const dataset = buildMovementDataset([
      trajectory("t1", [{ gesture: "tap" }, { gesture: "swipe", direction: "up" }]),
      trajectory("t2", []),
    ]);
    expect(dataset.sequences).toHaveLength(1);
    expect(dataset.sequences[0]).toEqual({ trajectoryId: "t1", tokens: ["device:tap", "device:swipe:up"] });
  });

  it("uses redacted actions when a trajectory has been reviewed", () => {
    const base = trajectory("t3", [{ gesture: "tap" }]);
    base.review = {
      status: "approved",
      reviewedAt: "2026-07-27T00:00:00.000Z",
      reviewedBy: "reviewer",
      redactedActions: [{ ts: 0, tool: "device", summary: "tapped compose" }],
    };
    const dataset = buildMovementDataset([base]);
    expect(dataset.sequences[0].tokens).toEqual(["device:tapped-compose"]);
  });
});

const REPEAT_DATASET: MovementDataset = {
  version: 1,
  sequences: [
    { trajectoryId: "a", tokens: ["device:tap", "device:swipe:down", "device:tap", "device:type"] },
    { trajectoryId: "b", tokens: ["device:tap", "device:swipe:down", "device:tap", "device:type"] },
    { trajectoryId: "c", tokens: ["device:tap", "device:swipe:down", "device:tap", "device:type"] },
  ],
};

describe("MarkovMovementBackend", () => {
  it("repeats a recorded movement sequence from its seed", () => {
    const model = new MarkovMovementBackend().train(REPEAT_DATASET, { order: 2 });
    const produced = model.generate([], 4);
    expect(produced).toEqual(["device:tap", "device:swipe:down", "device:tap", "device:type"]);
  });

  it("predicts the most likely next token deterministically", () => {
    const model = new MarkovMovementBackend().train(REPEAT_DATASET, { order: 2 });
    const prediction = model.predictNext(["device:tap", "device:swipe:down"]);
    expect(prediction.token).toBe("device:tap");
    expect(prediction.probability).toBeGreaterThan(0);
    expect(prediction.backedOff).toBe(false);
  });

  it("generalizes to an unseen context via backoff instead of failing", () => {
    // Train on two related routines that share the "swipe:down -> tap" motif.
    const dataset: MovementDataset = {
      version: 1,
      sequences: [
        { trajectoryId: "a", tokens: ["device:open", "device:swipe:down", "device:tap"] },
        { trajectoryId: "b", tokens: ["device:launch", "device:swipe:down", "device:tap"] },
      ],
    };
    const model = new MarkovMovementBackend().train(dataset, { order: 2 });
    // "device:scroll" was never followed by anything; high-order context is unseen,
    // but backoff to the shared "swipe:down -> tap" bigram still yields a related move.
    const prediction = model.predictNext(["device:scroll", "device:swipe:down"]);
    expect(prediction.token).toBe("device:tap");
    // The exact 2-gram context was never seen, so the model backs off to the
    // shared "swipe:down -> tap" bigram — generalization, not failure.
    expect(prediction.backedOff).toBe(true);
    expect(prediction.order).toBe(1);
    const noContext = model.predictNext(["device:totally-unknown"]);
    expect(noContext.token).toBeDefined(); // unigram backoff still predicts something
    expect(noContext.backedOff).toBe(true);
  });

  it("scores an in-distribution sequence higher than an out-of-distribution one", () => {
    const model = new MarkovMovementBackend().train(REPEAT_DATASET, { order: 2 });
    const inDist = model.score(["device:tap", "device:swipe:down", "device:tap", "device:type"]);
    const outDist = model.score(["device:type", "device:type", "device:type", "device:type"]);
    expect(inDist).toBeGreaterThan(outDist);
    expect(Number.isFinite(inDist)).toBe(true);
    expect(Number.isFinite(outDist)).toBe(true);
  });

  it("serializes to a stable, replayable snapshot", () => {
    const model = new MarkovMovementBackend().train(REPEAT_DATASET, { order: 2 });
    const first = model.serialize();
    const second = new MarkovMovementBackend().train(REPEAT_DATASET, { order: 2 }).serialize();
    expect(first).toEqual(second);
    expect(first.backend).toBe("markov");
    expect(first.vocabulary).toContain("device:type");
  });

  it("keeps context orders distinct so token sequences do not collide", () => {
    const dataset: MovementDataset = {
      version: 1,
      sequences: [
        { trajectoryId: "a", tokens: ["a", "b", "z"] },
        { trajectoryId: "b", tokens: ["ab", "y"] },
      ],
    };
    const model = new MarkovMovementBackend().train(dataset, { order: 2 });
    // Context ["a","b"] must predict "z", not "y" (which follows the single token "ab").
    expect(model.predictNext(["a", "b"]).token).toBe("z");
    expect(model.predictNext(["ab"]).token).toBe("y");
  });
});

describe("MovementModelRegistry", () => {
  it("registers the markov backend as an in-process default", () => {
    const registry = new MovementModelRegistry();
    expect(registry.has("markov")).toBe(true);
    expect(registry.list()).toEqual(["axolotl", "markov", "mlx"]);
    const model = registry.train("markov", REPEAT_DATASET, { order: 2 });
    expect(model.backend).toBe("markov");
  });

  it("exposes on-device backends as seams that fail loudly in the cloud", () => {
    const registry = new MovementModelRegistry();
    expect(() => registry.train("mlx", REPEAT_DATASET)).toThrow(/on-device runtime/);
    expect(() => registry.train("axolotl", REPEAT_DATASET)).toThrow(/on-device runtime/);
  });

  it("throws a helpful error for an unknown backend", () => {
    const registry = new MovementModelRegistry();
    expect(() => registry.get("gpt")).toThrow(/unknown movement model backend/);
  });
});
