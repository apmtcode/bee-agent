import { describe, expect, it } from "vitest";
import {
  DEFAULT_MODEL_BACKEND,
  MOVEMENT_END_TOKEN,
  createModelBackend,
  defaultModelBackend,
  listModelBackends,
  movementDatasetFromReplays,
  movementDatasetFromTrajectories,
  movementSequenceFromReplay,
  tokenizeMovementAction,
  type MovementDataset,
  type TrainedMovementModel,
} from "./model-backend.js";
// Import the backend module so it self-registers under "markov".
import { createMarkovBackend } from "./backends/markov-backend.js";
import { buildTrajectorySpan } from "../capture/trajectory.js";

function dataset(sequences: string[][]): MovementDataset {
  return { sequences: sequences.map((tokens, index) => ({ id: `seq-${index}`, tokens })) };
}

describe("movement tokenization + dataset builders", () => {
  it("tokenizes to tool level by default and folds a summary slug when asked", () => {
    expect(tokenizeMovementAction({ tool: "mouse.click", summary: "OK button" })).toBe("mouse.click");
    expect(tokenizeMovementAction({ tool: "mouse.click", summary: "OK button" }, { includeSummary: true })).toBe(
      "mouse.click#ok-button",
    );
  });

  it("builds an ordered token sequence from a trajectory's actions", () => {
    const trajectory = buildTrajectorySpan({
      id: "t1",
      sessionId: "s1",
      actions: [
        { kind: "action", tool: "focus", summary: "", ts: 30 },
        { kind: "action", tool: "open_menu", summary: "", ts: 10 },
        { kind: "action", tool: "click", summary: "", ts: 20 },
      ],
    });
    const built = movementDatasetFromTrajectories([trajectory]);
    expect(built.sequences).toHaveLength(1);
    // Sorted by timestamp, not insertion order.
    expect(built.sequences[0]!.tokens).toEqual(["open_menu", "click", "focus"]);
  });

  it("extracts action tokens from a replay manifest and drops non-action events", () => {
    const sequence = movementSequenceFromReplay({
      sessionId: "s2",
      events: [
        { kind: "transcript", ts: 5, messageId: "m1", role: "user", content: "hi" },
        { kind: "observation", ts: 6, trajectoryId: "t", source: "screen", summary: "menu open" },
        { kind: "action", ts: 7, trajectoryId: "t", tool: "move", summary: "to item" },
        { kind: "action", ts: 8, trajectoryId: "t", tool: "click", summary: "item" },
      ],
    });
    expect(sequence.tokens).toEqual(["move", "click"]);
  });

  it("drops empty sequences from replay datasets", () => {
    const built = movementDatasetFromReplays([
      { sessionId: "empty", events: [{ kind: "transcript", ts: 1, messageId: "m", role: "user", content: "x" }] },
    ]);
    expect(built.sequences).toHaveLength(0);
  });
});

describe("backend registry", () => {
  it("registers the default markov backend and can create it by name", () => {
    expect(listModelBackends()).toContain(DEFAULT_MODEL_BACKEND);
    expect(createModelBackend(DEFAULT_MODEL_BACKEND).name).toBe("markov");
    expect(defaultModelBackend().name).toBe("markov");
  });

  it("throws a helpful error for an unknown backend", () => {
    expect(() => createModelBackend("does-not-exist")).toThrow(/unknown model backend/);
  });
});

describe("markov backend — replay fidelity (objective 2c)", () => {
  it("reproduces a recorded movement sequence deterministically", async () => {
    const backend = createMarkovBackend();
    const model = await backend.train(dataset([["focus", "open_menu", "move_to_item", "click"]]), { order: 2 });

    // Seeded with the opening movement, it replays the rest exactly.
    const replayed = backend.generate(model, { seed: ["focus"] });
    expect(replayed).toEqual(["open_menu", "move_to_item", "click"]);

    // From scratch it reconstructs the whole trajectory.
    expect(backend.generate(model)).toEqual(["focus", "open_menu", "move_to_item", "click"]);
  });

  it("predicts the argmax next movement with a normalized distribution", async () => {
    const backend = createMarkovBackend();
    const model = await backend.train(
      dataset([
        ["a", "b", "c"],
        ["a", "b", "c"],
        ["a", "b", "d"],
      ]),
      { order: 2 },
    );
    const prediction = backend.predict(model, ["a", "b"]);
    expect(prediction.token).toBe("c");
    expect(prediction.contextOrderUsed).toBe(2);
    const total = prediction.distribution.reduce((sum, entry) => sum + entry.probability, 0);
    expect(total).toBeCloseTo(1, 10);
    expect(prediction.distribution.find((entry) => entry.token === "c")?.probability).toBeCloseTo(2 / 3, 10);
  });
});

describe("markov backend — generalization (objective 2d)", () => {
  it("backs off to a shorter observed context for an unseen prefix", async () => {
    const backend = createMarkovBackend();
    // Full bigram context [c, x] was never seen, but [x] -> {y, z} was.
    const model = await backend.train(
      dataset([
        ["a", "x", "y"],
        ["b", "x", "z"],
      ]),
      { order: 2 },
    );
    const prediction = backend.predict(model, ["c", "x"]);
    // Generalizes via unigram back-off; tie broken lexically -> "y".
    expect(prediction.token).toBe("y");
    expect(prediction.contextOrderUsed).toBe(1);
  });

  it("terminates generation at the learned end boundary", async () => {
    const backend = createMarkovBackend();
    const model = await backend.train(dataset([["step"]]), { order: 2 });
    const out = backend.generate(model, { seed: ["step"] });
    expect(out).toEqual([]);
    expect(model.vocabulary).not.toContain(MOVEMENT_END_TOKEN);
  });
});

describe("markov backend — persistence + determinism", () => {
  it("survives a JSON serialization round-trip", async () => {
    const backend = createMarkovBackend();
    const model = await backend.train(dataset([["p", "q", "r"]]), { order: 2 });
    const rehydrated = JSON.parse(JSON.stringify(model)) as TrainedMovementModel;
    expect(backend.generate(rehydrated, { seed: ["p"] })).toEqual(["q", "r"]);
  });

  it("produces identical models across runs (no clocks/randomness)", async () => {
    const backend = createMarkovBackend();
    const data = dataset([
      ["a", "b", "c"],
      ["a", "b", "d"],
    ]);
    const first = await backend.train(data, { order: 2 });
    const second = await backend.train(data, { order: 2 });
    expect(JSON.stringify(first)).toEqual(JSON.stringify(second));
  });
});
