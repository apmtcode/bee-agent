import { describe, expect, it } from "vitest";
import {
  MarkovMovementBackend,
  UnavailableLocalMovementBackend,
  bucketizeSummary,
  createMovementBackend,
  datasetFromTrajectories,
  defaultMovementBackendRegistry,
  generateSyntheticMovementDataset,
  movementToken,
  tokenizeReplayManifest,
  tokenizeTrajectory,
  type MovementDataset,
  type MovementToken,
} from "./movement-model.js";
import { buildReplayManifest } from "../capture/replay.js";
import { buildTrajectorySpan } from "../capture/trajectory.js";

function seq(id: string, tokens: MovementToken[]): { id: string; tokens: MovementToken[] } {
  return { id, tokens };
}

describe("bucketizeSummary", () => {
  it("normalizes free text to a stable low-cardinality bucket", () => {
    expect(bucketizeSummary("Click New Tab")).toBe("click-new-tab");
    expect(bucketizeSummary("Scroll down 240 pixels!!!")).toBe("scroll-down-#");
    expect(bucketizeSummary("")).toBe("noop");
  });

  it("caps the bucket at three words so cardinality stays bounded", () => {
    expect(bucketizeSummary("one two three four five")).toBe("one-two-three");
  });
});

describe("MarkovMovementBackend reproduction", () => {
  it("reproduces a recorded movement sequence exactly via argmax decoding", async () => {
    const tokens = [
      movementToken("mouse", "move to toolbar"),
      movementToken("mouse", "click new tab"),
      movementToken("keyboard", "type url"),
      movementToken("keyboard", "press enter"),
    ];
    const dataset: MovementDataset = { sequences: [seq("a", tokens)] };
    const model = await new MarkovMovementBackend().train(dataset, { order: 3 });

    const generated = model.generate([tokens[0]!], { maxTokens: 16 });
    expect([tokens[0]!, ...generated]).toEqual(tokens);
  });

  it("stops generating at the learned end of a movement", async () => {
    const tokens = [movementToken("mouse", "click"), movementToken("mouse", "release")];
    const model = await new MarkovMovementBackend().train({ sequences: [seq("a", tokens)] });
    const generated = model.generate([tokens[0]!], { maxTokens: 50 });
    // Bounded output: it does not run forever, it ends where the recording ended.
    expect(generated).toEqual([tokens[1]!]);
  });
});

describe("MarkovMovementBackend generalization", () => {
  it("generalizes to a new-but-related context via back-off", async () => {
    // Two sequences share the continuation "b -> c" after different heads.
    const b = movementToken("keyboard", "type");
    const c = movementToken("keyboard", "enter");
    const head1 = movementToken("mouse", "focus");
    const head2 = movementToken("window", "restore");
    const dataset: MovementDataset = {
      sequences: [seq("s1", [head1, b, c]), seq("s2", [head2, b, c])],
    };
    const model = await new MarkovMovementBackend().train(dataset, { order: 2 });

    // A never-before-seen head, followed by b: full-order context is unseen,
    // so the model backs off to the unigram context "b" and still predicts c.
    const novelHead = movementToken("mouse", "hover");
    const prediction = model.predictNext([novelHead, b]);
    expect(prediction?.token).toEqual(c);
    expect(prediction?.backoffOrder).toBeLessThan(2);
  });

  it("returns undefined when there is no context signal at all", async () => {
    const model = await new MarkovMovementBackend().train({ sequences: [] });
    expect(model.predictNext([movementToken("mouse", "click")])).toBeUndefined();
    expect(model.generate([movementToken("mouse", "click")])).toEqual([]);
  });
});

describe("snapshot round-trip", () => {
  it("restores an identical model from its snapshot", async () => {
    const dataset = generateSyntheticMovementDataset({ seed: 7, sequenceCount: 6 });
    const backend = new MarkovMovementBackend();
    const model = await backend.train(dataset, { order: 3 });
    const restored = backend.restore(model.snapshot());

    expect(restored.snapshot()).toEqual(model.snapshot());

    const seed = dataset.sequences[0]!.tokens.slice(0, 1);
    expect(restored.generate(seed, { maxTokens: 20 })).toEqual(model.generate(seed, { maxTokens: 20 }));
  });
});

describe("tokenization", () => {
  it("tokenizes a trajectory's action stream in timestamp order", () => {
    const trajectory = buildTrajectorySpan({
      id: "t1",
      sessionId: "s1",
      actions: [
        { kind: "action", tool: "mouse", summary: "Click Save", ts: 20 },
        { kind: "action", tool: "keyboard", summary: "Type name", ts: 10 },
      ],
    });
    const tokenized = tokenizeTrajectory(trajectory);
    expect(tokenized.tokens).toEqual([
      { tool: "keyboard", action: "type-name" },
      { tool: "mouse", action: "click-save" },
    ]);
  });

  it("tokenizes only action events from a replay manifest", () => {
    const trajectory = buildTrajectorySpan({
      id: "t1",
      sessionId: "s1",
      observations: [{ kind: "observation", source: "screen", summary: "menu open", ts: 5 }],
      actions: [{ kind: "action", tool: "mouse", summary: "Click item", ts: 6 }],
    });
    const manifest = buildReplayManifest({ sessionId: "s1", transcript: [], trajectories: [trajectory] });
    expect(tokenizeReplayManifest(manifest).tokens).toEqual([{ tool: "mouse", action: "click-item" }]);
  });

  it("builds a dataset from trajectories, dropping empty ones", () => {
    const withActions = buildTrajectorySpan({
      id: "t1",
      sessionId: "s1",
      actions: [{ kind: "action", tool: "mouse", summary: "Click", ts: 1 }],
    });
    const empty = buildTrajectorySpan({ id: "t2", sessionId: "s1" });
    const dataset = datasetFromTrajectories([withActions, empty]);
    expect(dataset.sequences.map((sequence) => sequence.id)).toEqual(["t1"]);
  });
});

describe("backend registry", () => {
  it("resolves the deterministic mock by default", () => {
    expect(createMovementBackend().id).toBe("markov-ngram");
  });

  it("exposes on-device backends that fail loudly in the cloud", async () => {
    const registry = defaultMovementBackendRegistry();
    expect([...registry.keys()]).toContain("mlx-lora");
    const backend = createMovementBackend("mlx-lora", registry);
    expect(backend).toBeInstanceOf(UnavailableLocalMovementBackend);
    await expect(backend.train({ sequences: [] })).rejects.toThrow(/on-device/);
  });

  it("throws on an unknown backend id", () => {
    expect(() => createMovementBackend("does-not-exist")).toThrow(/Unknown movement backend/);
  });
});

describe("synthetic dataset", () => {
  it("is deterministic for a given seed", () => {
    const a = generateSyntheticMovementDataset({ seed: 42, sequenceCount: 5, sequenceLength: 6 });
    const b = generateSyntheticMovementDataset({ seed: 42, sequenceCount: 5, sequenceLength: 6 });
    expect(a).toEqual(b);
    expect(a.sequences).toHaveLength(5);
    expect(a.sequences[0]!.tokens).toHaveLength(6);
  });

  it("produces a bounded, in-vocabulary continuation from a synthetic seed", async () => {
    const dataset = generateSyntheticMovementDataset({ seed: 3, sequenceCount: 8, sequenceLength: 6 });
    const model = await new MarkovMovementBackend().train(dataset, { order: 3 });
    const vocab = new Set(dataset.sequences.flatMap((s) => s.tokens.map((t) => `${t.tool}|${t.action}`)));

    const sample = dataset.sequences[0]!;
    const continued = model.generate(sample.tokens.slice(0, 2), { maxTokens: 20 });
    // Inference terminates (does not hit the cap) and only emits known movements.
    expect(continued.length).toBeLessThan(20);
    for (const token of continued) {
      expect(vocab.has(`${token.tool}|${token.action}`)).toBe(true);
    }
  });
});
