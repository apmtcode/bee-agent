import { describe, expect, it } from "vitest";
import { MarkovMovementBackend } from "./backend.js";
import {
  MOVEMENT_BOS,
  MOVEMENT_EOS,
  labelToken,
  createCoordinateQuantizer,
  type MovementDataset,
} from "./events.js";
import { generateMovementDataset, type SyntheticTaskSpec } from "./synthetic.js";

const quantizer = createCoordinateQuantizer({ width: 1280, height: 800, cols: 16, rows: 10 });

function datasetOf(specs: SyntheticTaskSpec[], seed = 3): MovementDataset {
  return generateMovementDataset({ seed, specs, quantizer, jitter: 4 });
}

describe("MarkovMovementBackend", () => {
  it("trains a model that exposes a serializable snapshot", async () => {
    const dataset = datasetOf([
      { task: "open-and-type", target: { x: 300, y: 200, windowId: "editor" }, typeLength: 2 },
    ]);
    const model = await new MarkovMovementBackend().train(dataset, { maxOrder: 3 });
    const snapshot = model.snapshot();
    expect(snapshot.backendId).toBe("markov-backoff");
    expect(snapshot.order).toBe(3);
    expect(snapshot.vocabularySize).toBeGreaterThan(0);
    expect(snapshot.contextCount).toBeGreaterThan(0);
  });

  it("predicts the intent token right after BOS", async () => {
    const dataset = datasetOf([
      { task: "menu-select", target: { x: 100, y: 60, windowId: "menu" } },
    ]);
    const model = await new MarkovMovementBackend().train(dataset);
    const prediction = model.predictNextToken([MOVEMENT_BOS]);
    expect(prediction?.token).toBe(labelToken("menu-select"));
    expect(prediction?.probability).toBeGreaterThan(0);
  });

  it("backs off to shorter contexts when the full context is novel", async () => {
    const dataset = datasetOf([
      { task: "open-and-type", target: { x: 300, y: 200, windowId: "editor" }, typeLength: 3 },
    ]);
    const model = await new MarkovMovementBackend().train(dataset, { maxOrder: 4 });
    // A context the model never saw verbatim, but whose tail (kd:char) it did.
    const prediction = model.predictNextToken(["totally", "unseen", "prefix", "kd:char"]);
    expect(prediction).toBeDefined();
    expect(prediction!.order).toBeLessThan(4);
    expect(prediction!.token).toBe("ku:char");
  });

  it("generation is deterministic and terminates at EOS", async () => {
    const dataset = datasetOf([
      { task: "scroll-and-click", target: { x: 640, y: 400, windowId: "list" }, scrollDown: true },
    ]);
    const model = await new MarkovMovementBackend().train(dataset);
    const seed = [MOVEMENT_BOS, labelToken("scroll-and-click")];
    const first = model.generate(seed, 50);
    const second = model.generate(seed, 50);
    expect(first).toEqual(second);
    expect(first.at(-1)).toBe(MOVEMENT_EOS);
    expect(first.length).toBeLessThanOrEqual(50 + seed.length);
  });

  it("returns undefined when there is nothing to predict from an empty model", async () => {
    const model = await new MarkovMovementBackend().train({ quantizer, trajectories: [] });
    expect(model.predictNextToken([MOVEMENT_BOS])).toBeUndefined();
    expect(model.generate([MOVEMENT_BOS], 5)).toEqual([MOVEMENT_BOS]);
  });
});
