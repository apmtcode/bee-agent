import { describe, expect, it } from "vitest";
import {
  MarkovMovementBackend,
  buildMovementDataset,
  buildMovementDatasetFromReplays,
  movementTokenFromAction,
  movementTokenKey,
  type MovementModelBackend,
  type MovementToken,
} from "./movement-model.js";
import type { ReplayTimelineEvent } from "../capture/replay.js";

function tokens(...specs: Array<[string, string, string?]>): MovementToken[] {
  return specs.map(([tool, action, target]) => ({ tool, action, ...(target ? { target } : {}) }));
}

describe("movement-model dataset builders", () => {
  it("drops empty sequences and counts tokens", () => {
    const dataset = buildMovementDataset([
      tokens(["device", "tap", "a"], ["device", "swipe:down"]),
      [],
      tokens(["device", "type"]),
    ]);
    expect(dataset.sampleCount).toBe(2);
    expect(dataset.tokenCount).toBe(3);
  });

  it("derives a movement token from a captured action + metadata", () => {
    const token = movementTokenFromAction({
      tool: "device",
      summary: "swiped down",
      metadata: { gesture: "swipe", direction: "down", target: "feed" },
    });
    expect(token).toEqual({ tool: "device", action: "swipe:down", target: "feed" });
  });

  it("falls back to the summary verb when metadata is absent", () => {
    const token = movementTokenFromAction({ tool: "browser", summary: "Clicked Submit button" });
    expect(token).toEqual({ tool: "browser", action: "clicked" });
  });

  it("groups replay action events into per-trajectory sequences in ts order", () => {
    const events: ReplayTimelineEvent[] = [
      { kind: "action", ts: 3, trajectoryId: "t1", tool: "device", summary: "typed name" },
      { kind: "observation", ts: 1, trajectoryId: "t1", source: "device", summary: "app active" },
      { kind: "action", ts: 1, trajectoryId: "t1", tool: "device", summary: "tapped field" },
      { kind: "action", ts: 2, trajectoryId: "t2", tool: "device", summary: "scrolled down" },
    ];
    const dataset = buildMovementDatasetFromReplays([{ events }]);
    expect(dataset.sampleCount).toBe(2);
    const t1 = dataset.sequences.find((seq) => seq[0]?.action === "tapped");
    expect(t1?.map((token) => token.action)).toEqual(["tapped", "typed"]);
  });
});

describe("MarkovMovementBackend", () => {
  const backend: MovementModelBackend = new MarkovMovementBackend();

  it("reproduces a recorded movement sequence from its prefix", async () => {
    const recorded = tokens(["device", "tap", "search"], ["device", "type", "query"], ["device", "tap", "result"]);
    const model = await backend.train(buildMovementDataset([recorded]));

    const replayed = model.generate([recorded[0]], { maxSteps: 10 });
    expect(replayed).toEqual(recorded.slice(1));

    // The learned end-sentinel makes generation terminate rather than loop.
    expect(replayed.length).toBe(2);
  });

  it("predicts the exact next movement with backoffDepth 0 for a seen context", async () => {
    const recorded = tokens(["device", "tap", "a"], ["device", "swipe:down"], ["device", "tap", "b"]);
    const model = await backend.train(buildMovementDataset([recorded]), { order: 2 });

    const prediction = model.predictNext([recorded[0], recorded[1]]);
    expect(prediction.token).toEqual(recorded[2]);
    expect(prediction.backoffDepth).toBe(0);
    expect(prediction.novel).toBe(false);
    expect(prediction.probability).toBe(1);
  });

  it("generalizes to a novel context via backoff to learned local transitions", async () => {
    // Two related recordings: both have `swipe` followed by a "confirm" style tap.
    const model = await backend.train(
      buildMovementDataset([
        tokens(["device", "tap", "feed"], ["device", "swipe:down"], ["device", "tap", "confirm"]),
        tokens(["device", "type", "note"], ["device", "swipe:down"], ["device", "tap", "confirm"]),
      ]),
      { order: 2 },
    );

    // A context whose bigram `[open, swipe:down]` was never recorded verbatim:
    const novelContext = tokens(["device", "open", "menu"], ["device", "swipe:down"]);
    const prediction = model.predictNext(novelContext);

    // It backs off to the unigram context `[swipe:down]` and still predicts the
    // learned continuation — generalization to a related-but-new movement.
    expect(prediction.novel).toBe(true);
    expect(prediction.backoffDepth).toBeGreaterThan(0);
    expect(prediction.token).toEqual({ tool: "device", action: "tap", target: "confirm" });
  });

  it("is deterministic: identical training yields identical predictions", async () => {
    const data = buildMovementDataset([
      tokens(["device", "tap", "x"], ["device", "type", "y"]),
      tokens(["device", "tap", "x"], ["device", "swipe:up"]),
    ]);
    const a = await backend.train(data);
    const b = await backend.train(data);
    // Context [tap x] leads to a 50/50 split; deterministic tie-break must agree.
    const pa = a.predictNext(tokens(["device", "tap", "x"]));
    const pb = b.predictNext(tokens(["device", "tap", "x"]));
    expect(pa.token).toEqual(pb.token);
    expect(pa.candidates.map((c) => c.token)).toEqual(pb.candidates.map((c) => c.token));
  });

  it("returns weighted candidates summing to ~1 over the matched context", async () => {
    const model = await backend.train(
      buildMovementDataset([
        tokens(["device", "tap", "x"], ["device", "type", "y"]),
        tokens(["device", "tap", "x"], ["device", "type", "y"]),
        tokens(["device", "tap", "x"], ["device", "swipe:up"]),
      ]),
    );
    const prediction = model.predictNext(tokens(["device", "tap", "x"]));
    expect(prediction.token).toEqual({ tool: "device", action: "type", target: "y" });
    expect(prediction.probability).toBeCloseTo(2 / 3, 6);
    const total = prediction.candidates.reduce((sum, candidate) => sum + candidate.probability, 0);
    expect(total).toBeCloseTo(1, 6);
  });

  it("round-trips through serialize/load with identical behaviour", async () => {
    const recorded = tokens(["device", "tap", "a"], ["device", "type", "b"], ["device", "tap", "c"]);
    const model = await backend.train(buildMovementDataset([recorded]));
    const restored = backend.load(JSON.parse(JSON.stringify(model.serialize())));

    expect(restored.backendId).toBe(model.backendId);
    expect(restored.order).toBe(model.order);
    expect(restored.generate([recorded[0]])).toEqual(model.generate([recorded[0]]));
  });

  it("returns an empty prediction for a model trained on no sequences", async () => {
    const model = await backend.train(buildMovementDataset([]));
    const prediction = model.predictNext(tokens(["device", "tap"]));
    expect(prediction.token).toBeUndefined();
    expect(prediction.candidates).toEqual([]);
    expect(model.generate(tokens(["device", "tap"]))).toEqual([]);
  });

  it("produces stable token keys for identical tokens", () => {
    expect(movementTokenKey({ tool: "device", action: "tap", target: "x" })).toBe(
      movementTokenKey({ tool: "device", action: "tap", target: "x" }),
    );
    expect(movementTokenKey({ tool: "device", action: "tap" })).not.toBe(
      movementTokenKey({ tool: "device", action: "tap", target: "x" }),
    );
  });
});
