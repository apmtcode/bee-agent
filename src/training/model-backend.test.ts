import { describe, expect, it } from "vitest";
import {
  MarkovMovementBackend,
  MovementModelBackendRegistry,
  MOVEMENT_EOS,
  createDefaultMovementBackendRegistry,
  createMovementDatasetFromExport,
  createMovementDatasetFromReplays,
  evaluateMovementModel,
  type MovementDataset,
} from "./model-backend.js";
import type { ExportedReplayManifest, ReviewedExportManifest } from "./export-manifest.js";

function dataset(examples: Array<{ id: string; tokens: string[] }>): MovementDataset {
  return { version: 1, examples };
}

describe("MarkovMovementBackend", () => {
  it("trains a deterministic, JSON-serialisable model", async () => {
    const backend = new MarkovMovementBackend();
    const ds = dataset([{ id: "t1", tokens: ["focus", "move", "click"] }]);

    const modelA = await backend.train(ds, { order: 3 });
    const modelB = await backend.train(ds, { order: 3 });

    // Determinism: two trainings on the same data are byte-identical.
    expect(JSON.stringify(modelA)).toEqual(JSON.stringify(modelB));
    // Round-trips through JSON (it is persisted as a training artifact).
    expect(JSON.parse(JSON.stringify(modelA))).toEqual(modelA);
    expect(modelA.backend).toBe("markov-mock");
    expect(modelA.trainedExampleCount).toBe(1);
    expect(modelA.vocabulary).toEqual(["click", "focus", "move"]);
    expect(modelA.vocabulary).not.toContain(MOVEMENT_EOS);
  });

  it("repeats a recorded movement exactly (memorisation)", async () => {
    const backend = new MarkovMovementBackend();
    const recorded = ["focus-window", "move-to", "mouse-down", "drag", "mouse-up"];
    const model = await backend.train(dataset([{ id: "t1", tokens: recorded }]), { order: 5 });

    // From an empty seed the model seeds itself from the learned start token and
    // reproduces the whole movement, stopping at the natural end.
    expect(backend.generate(model)).toEqual(recorded);
  });

  it("predicts the next movement token from context", async () => {
    const backend = new MarkovMovementBackend();
    const model = await backend.train(dataset([{ id: "t1", tokens: ["a", "b", "c", "d"] }]), { order: 3 });

    expect(backend.predictNext(model, ["a"])).toBe("b");
    expect(backend.predictNext(model, ["a", "b"])).toBe("c");
    expect(backend.predictNext(model, ["a", "b", "c"])).toBe("d");
  });

  it("never returns the EOS sentinel from predictNext", async () => {
    const backend = new MarkovMovementBackend();
    const model = await backend.train(dataset([{ id: "t1", tokens: ["a", "b"] }]), { order: 3 });

    // Context ["a","b"] is only ever followed by EOS in training; predictNext
    // must back off to a concrete movement rather than surface the sentinel.
    const predicted = backend.predictNext(model, ["a", "b"]);
    expect(predicted).not.toBe(MOVEMENT_EOS);
    expect(predicted).toBeDefined();
  });

  it("generalises to a new-but-related movement via backoff", async () => {
    const backend = new MarkovMovementBackend();
    // Two related recorded movements share the sub-sequence [open, type].
    const model = await backend.train(
      dataset([
        { id: "t1", tokens: ["open", "type", "save"] },
        { id: "t2", tokens: ["open", "type", "close"] },
      ]),
      { order: 3 },
    );

    // A novel prefix the model never saw verbatim: ["focus","open","type"].
    // The full-order context is unknown, so it backs off to the shared suffix
    // [open, type] and predicts a learned continuation — a related movement.
    const next = backend.predictNext(model, ["focus", "open", "type"]);
    expect(["save", "close"]).toContain(next);
    // Deterministic tie-break (lexical) picks "close".
    expect(next).toBe("close");
  });

  it("falls back to the global unigram for an entirely unseen context", async () => {
    const backend = new MarkovMovementBackend();
    const model = await backend.train(
      dataset([
        { id: "t1", tokens: ["click", "click", "click"] },
        { id: "t2", tokens: ["click", "scroll"] },
      ]),
      { order: 2 },
    );

    // "zzz" appears nowhere; backoff bottoms out at the unigram distribution,
    // where "click" is the most frequent token.
    expect(backend.predictNext(model, ["zzz"])).toBe("click");
  });

  it("returns an empty sequence for an empty model", async () => {
    const backend = new MarkovMovementBackend();
    const model = await backend.train(dataset([]), { order: 3 });
    expect(model.trainedExampleCount).toBe(0);
    expect(backend.generate(model)).toEqual([]);
    expect(backend.predictNext(model, ["anything"])).toBeUndefined();
  });

  it("honours maxLength and includeSeed in generate", async () => {
    const backend = new MarkovMovementBackend();
    const model = await backend.train(dataset([{ id: "t1", tokens: ["a", "b", "c", "d", "e"] }]), { order: 5 });

    expect(backend.generate(model, { seed: ["a"], maxLength: 2 })).toEqual(["a", "b", "c"]);
    expect(backend.generate(model, { seed: ["a"], maxLength: 2, includeSeed: false })).toEqual(["b", "c"]);
  });
});

describe("MovementModelBackendRegistry", () => {
  it("registers, resolves and lists backends", () => {
    const registry = new MovementModelBackendRegistry().register(new MarkovMovementBackend());
    expect(registry.list()).toEqual(["markov-mock"]);
    expect(registry.get("markov-mock")).toBeInstanceOf(MarkovMovementBackend);
    expect(registry.get("nope")).toBeUndefined();
    expect(() => registry.require("nope")).toThrow(/Unknown movement-model backend/);
    expect(registry.require("markov-mock").name).toBe("markov-mock");
  });

  it("default registry ships the reference backend", () => {
    expect(createDefaultMovementBackendRegistry().list()).toEqual(["markov-mock"]);
  });

  it("accepts a pluggable custom backend behind the same interface", async () => {
    const custom = {
      name: "always-noop",
      async train() {
        return {
          backend: "always-noop",
          version: 1 as const,
          order: 1,
          vocabulary: [],
          transitions: {},
          starts: {},
          trainedExampleCount: 0,
        };
      },
      predictNext: () => "noop",
      generate: () => ["noop"],
    };
    const registry = createDefaultMovementBackendRegistry().register(custom);
    expect(registry.list()).toEqual(["always-noop", "markov-mock"]);
    expect(registry.require("always-noop").generate({} as never)).toEqual(["noop"]);
  });
});

describe("createMovementDatasetFromReplays", () => {
  function actionReplay(): ExportedReplayManifest {
    return {
      sessionId: "s1",
      trajectoryIds: ["t1", "t2"],
      eventCount: 5,
      events: [
        { kind: "transcript", ts: 1, messageId: "m1", role: "user", content: "go" },
        { kind: "action", ts: 4, trajectoryId: "t1", tool: "click", summary: "click ok" },
        { kind: "action", ts: 2, trajectoryId: "t1", tool: "move", summary: "move to (1,2)" },
        { kind: "observation", ts: 3, trajectoryId: "t1", source: "screen", summary: "dialog" },
        { kind: "action", ts: 5, trajectoryId: "t2", tool: "scroll", summary: "scroll down" },
      ],
    };
  }

  it("groups action events per trajectory, ordered by timestamp", () => {
    const ds = createMovementDatasetFromReplays([actionReplay()]);
    // Examples sorted by trajectory id; tokens sorted by ts; non-action events dropped.
    expect(ds.examples).toEqual([
      { id: "t1", tokens: ["move", "click"] },
      { id: "t2", tokens: ["scroll"] },
    ]);
  });

  it("supports a custom tokenizer for a richer movement vocabulary", () => {
    const ds = createMovementDatasetFromReplays([actionReplay()], (event) => `${event.tool}:${event.summary}`);
    expect(ds.examples[0].tokens).toEqual(["move:move to (1,2)", "click:click ok"]);
  });

  it("derives a dataset from a reviewed export manifest", () => {
    const manifest = {
      replays: [actionReplay()],
    } as unknown as ReviewedExportManifest;
    const ds = createMovementDatasetFromExport(manifest);
    expect(ds.examples.map((example) => example.id)).toEqual(["t1", "t2"]);
  });
});

describe("evaluateMovementModel", () => {
  it("reports perfect fidelity when reproducing the training set", async () => {
    const backend = new MarkovMovementBackend();
    const ds = dataset([
      { id: "t1", tokens: ["open", "type", "save"] },
      { id: "t2", tokens: ["focus", "click", "close"] },
    ]);
    const model = await backend.train(ds, { order: 4 });

    const evaluation = evaluateMovementModel(backend, model, ds);
    expect(evaluation.backend).toBe("markov-mock");
    expect(evaluation.exampleCount).toBe(2);
    expect(evaluation.exactMatchCount).toBe(2);
    expect(evaluation.meanAccuracy).toBe(1);
    expect(evaluation.perExample.every((entry) => entry.exactMatch)).toBe(true);
  });

  it("measures generalisation on a held-out related movement", async () => {
    const backend = new MarkovMovementBackend();
    const train = dataset([
      { id: "t1", tokens: ["open", "type", "save"] },
      { id: "t2", tokens: ["open", "type", "save"] },
    ]);
    const model = await backend.train(train, { order: 3 });

    // Held-out sequence starts the same then diverges; the shared prefix should
    // be reproduced, so accuracy is partial but positive — evidence of transfer.
    const holdout = dataset([{ id: "h1", tokens: ["open", "type", "save", "extra"] }]);
    const evaluation = evaluateMovementModel(backend, model, holdout);
    expect(evaluation.perExample[0].accuracy).toBeGreaterThanOrEqual(0.75);
    expect(evaluation.perExample[0].accuracy).toBeLessThan(1);
  });
});
