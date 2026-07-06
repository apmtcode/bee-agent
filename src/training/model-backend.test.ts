import { describe, expect, it } from "vitest";
import type { ReplayTimelineEvent } from "../capture/replay.js";
import {
  LocalModelBackendRegistry,
  MOVEMENT_EOS,
  NGramMovementBackend,
  createDefaultModelBackendRegistry,
  datasetFromReplayEvents,
  evaluateMovementModel,
  tokenizeReplayEvents,
  type MovementDataset,
} from "./model-backend.js";

const backend = new NGramMovementBackend();

// A small, deterministic synthetic "movement" workflow: open a file, edit it,
// run tests, then commit. Repeated with a related variant to give the model
// something to generalize from.
const OPEN_EDIT_TEST_COMMIT = ["action:open", "action:edit", "action:test", "action:commit"];
const OPEN_EDIT_TEST_PUSH = ["action:open", "action:edit", "action:test", "action:push"];

describe("tokenizeReplayEvents", () => {
  it("maps action/observation events to kind-prefixed tokens and skips transcript by default", () => {
    const events: ReplayTimelineEvent[] = [
      { kind: "transcript", ts: 1, messageId: "m1", role: "user", content: "hi" },
      { kind: "observation", ts: 2, trajectoryId: "t1", source: "screen", summary: "window opened" },
      { kind: "action", ts: 3, trajectoryId: "t1", tool: "click", summary: "clicked button" },
    ];
    expect(tokenizeReplayEvents(events)).toEqual(["observation:screen", "action:click"]);
    expect(tokenizeReplayEvents(events, { includeTranscript: true })).toEqual([
      "transcript:user",
      "observation:screen",
      "action:click",
    ]);
  });

  it("builds a dataset from replay events, dropping empties", () => {
    const dataset = datasetFromReplayEvents([
      [{ kind: "action", ts: 1, trajectoryId: "t", tool: "open", summary: "" }],
      [{ kind: "transcript", ts: 1, messageId: "m", role: "user", content: "" }],
    ]);
    expect(dataset.sequences).toEqual([["action:open"]]);
  });
});

describe("NGramMovementBackend train + repeat", () => {
  it("reproduces a recorded movement exactly when seeded with its start (part c)", async () => {
    const model = await backend.train({ sequences: [OPEN_EDIT_TEST_COMMIT] });
    const result = model.generate(["action:open"], 10);
    expect(result.continuation).toEqual(["action:edit", "action:test", "action:commit"]);
    expect(result.stopped).toBe(true);
    expect(result.sequence).toEqual(OPEN_EDIT_TEST_COMMIT);
  });

  it("reproduces a whole recording from an empty seed via the BOS anchor", async () => {
    const model = await backend.train({ sequences: [OPEN_EDIT_TEST_COMMIT] });
    const result = model.generate([], 10);
    expect(result.sequence).toEqual(OPEN_EDIT_TEST_COMMIT);
    expect(result.stopped).toBe(true);
  });

  it("exposes vocabulary without the EOS marker", async () => {
    const model = await backend.train({ sequences: [OPEN_EDIT_TEST_COMMIT] });
    expect(model.vocabulary).not.toContain(MOVEMENT_EOS);
    expect(model.vocabulary).toContain("action:open");
  });

  it("predicts the most frequent continuation and reports probability", async () => {
    // "edit" follows "open" twice; "test" follows "open" once → open→edit wins.
    const dataset: MovementDataset = {
      sequences: [OPEN_EDIT_TEST_COMMIT, OPEN_EDIT_TEST_PUSH, ["action:open", "action:test"]],
    };
    const model = await backend.train(dataset);
    const prediction = model.predictNext(["action:open"]);
    expect(prediction?.token).toBe("action:edit");
    expect(prediction?.probability).toBeCloseTo(2 / 3, 5);
    expect(prediction?.alternatives[0]?.token).toBe("action:test");
  });
});

describe("NGramMovementBackend generalization (part d)", () => {
  it("predicts via suffix backoff for an unseen-but-related context", async () => {
    // Train only on the commit variant. A novel order-3 prefix ("launch,edit,
    // test") was never seen, so the model backs off to the order-2 suffix
    // "edit,test" → commit that it did learn.
    const model = await backend.train({ sequences: [OPEN_EDIT_TEST_COMMIT] }, { order: 3 });
    const prediction = model.predictNext(["action:launch", "action:edit", "action:test"]);
    expect(prediction?.token).toBe("action:commit");
    expect(prediction?.exact).toBe(false);
    expect(prediction?.backoffOrder).toBe(2);
  });

  it("backs off to a shorter suffix when the full-order context is novel", async () => {
    const model = await backend.train(
      { sequences: [["a", "b", "c", "d"], ["x", "b", "c", "e"]] },
      { order: 3 },
    );
    // Context "z,q,c": order-3 "z,q,c" unseen, order-2 "q,c" unseen, order-1 "c"
    // seen → follows to whichever token is more frequent (d and e tie → lexicographic "d").
    const prediction = model.predictNext(["z", "q", "c"]);
    expect(prediction?.token).toBe("d");
    expect(prediction?.exact).toBe(false);
    expect(prediction?.backoffOrder).toBe(1);
  });
});

describe("evaluateMovementModel", () => {
  it("scores perfect fidelity on a memorized sequence", async () => {
    const model = await backend.train({ sequences: [OPEN_EDIT_TEST_COMMIT] });
    const result = evaluateMovementModel(model, { sequences: [OPEN_EDIT_TEST_COMMIT] });
    expect(result.accuracy).toBe(1);
    expect(result.predictions).toBe(OPEN_EDIT_TEST_COMMIT.length + 1); // + EOS
  });

  it("credits generalized (backoff) matches on held-out related sequences", async () => {
    const model = await backend.train({ sequences: [OPEN_EDIT_TEST_COMMIT] }, { order: 2 });
    const result = evaluateMovementModel(model, { sequences: [OPEN_EDIT_TEST_PUSH] });
    // The shared "open,edit,test" prefix is predicted correctly; the divergent
    // tail (push vs commit) is not — so accuracy is partial but positive.
    expect(result.accuracy).toBeGreaterThan(0);
    expect(result.accuracy).toBeLessThan(1);
  });

  it("returns zeroed metrics for an empty held-out set", async () => {
    const model = await backend.train({ sequences: [OPEN_EDIT_TEST_COMMIT] });
    expect(evaluateMovementModel(model, { sequences: [] })).toEqual({
      predictions: 0,
      correct: 0,
      accuracy: 0,
      generalizedShare: 0,
    });
  });
});

describe("snapshot round-trip", () => {
  it("restores an identical model from its snapshot", async () => {
    const dataset: MovementDataset = { sequences: [OPEN_EDIT_TEST_COMMIT, OPEN_EDIT_TEST_PUSH] };
    const model = await backend.train(dataset, { order: 3 });
    const restored = backend.restore(model.snapshot());
    expect(restored.order).toBe(model.order);
    expect(restored.vocabulary).toEqual(model.vocabulary);
    expect(restored.generate(["action:open"], 10).continuation).toEqual(
      model.generate(["action:open"], 10).continuation,
    );
    // Snapshot is JSON-serializable.
    expect(() => JSON.stringify(model.snapshot())).not.toThrow();
  });
});

describe("LocalModelBackendRegistry", () => {
  it("provides the ngram backend by default and is pluggable", () => {
    const registry = createDefaultModelBackendRegistry();
    expect(registry.list()).toContain("ngram");
    expect(registry.has("ngram")).toBe(true);
    expect(registry.get("ngram")).toBeInstanceOf(NGramMovementBackend);
  });

  it("throws for an unknown backend id", () => {
    const registry = new LocalModelBackendRegistry();
    expect(() => registry.get("mlx-real")).toThrow(/unknown local model backend/);
  });

  it("accepts a swapped-in custom backend behind the same interface", async () => {
    const custom = new NGramMovementBackend();
    Object.defineProperty(custom, "id", { value: "custom", configurable: true });
    const registry = new LocalModelBackendRegistry([custom]);
    expect(registry.list()).toEqual(["custom"]);
    const model = await registry.get("custom").train({ sequences: [OPEN_EDIT_TEST_COMMIT] });
    expect(model.generate(["action:open"], 5).stopped).toBe(true);
  });
});
