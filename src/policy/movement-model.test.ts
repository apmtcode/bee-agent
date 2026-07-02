import { describe, expect, it } from "vitest";
import {
  NgramMovementBackend,
  buildMovementSamples,
  buildMovementSamplesFromReplays,
  defaultVerbObject,
  evaluateMovementBackend,
  measureGeneralization,
  renderMovementSummary,
  type MovementPolicyBackend,
  type MovementSample,
} from "./movement-model.js";
import { synthesizeWorkflowDataset, synthesizeWorkflowReplay } from "./synthetic.js";

describe("defaultVerbObject", () => {
  it("splits a summary into verb and object", () => {
    expect(defaultVerbObject("save report")).toEqual({ verb: "save", object: "report" });
    expect(defaultVerbObject("typed into search box")).toEqual({ verb: "typed", object: "into search box" });
    expect(defaultVerbObject("SAVE")).toEqual({ verb: "save" });
    expect(defaultVerbObject("  ")).toEqual({ verb: "" });
  });
});

describe("buildMovementSamples", () => {
  it("emits one sample per action with the preceding context and recent object", () => {
    const samples = buildMovementSamples(synthesizeWorkflowReplay({ target: "report" }));
    expect(samples.map((s) => s.label)).toEqual([
      { tool: "device", verb: "open", object: "report" },
      { tool: "device", verb: "type", object: "report" },
      { tool: "device", verb: "save", object: "report" },
    ]);
    // Every action's object is recoverable from context (established by `select`).
    expect(samples.every((s) => s.context.object === "report")).toBe(true);
    // Context grows with the timeline.
    expect(samples[0]?.context.tokens.length).toBe(2);
    expect(samples[2]?.context.tokens.length).toBe(5);
  });

  it("respects the context window bound", () => {
    const samples = buildMovementSamples(synthesizeWorkflowReplay({ target: "report" }), { contextWindow: 2 });
    expect(samples[2]?.context.tokens.length).toBe(2);
  });
});

describe("NgramMovementBackend repeat fidelity (objective 2c)", () => {
  it("re-derives every recorded movement it was trained on", () => {
    const replay = synthesizeWorkflowReplay({ target: "report" });
    const samples = buildMovementSamples(replay);
    const backend = new NgramMovementBackend(4);
    backend.train(samples);

    const result = evaluateMovementBackend(backend, samples);
    expect(result.total).toBe(3);
    expect(result.correct).toBe(3);
    expect(result.accuracy).toBe(1);
    expect(result.unpredicted).toBe(0);
    expect(result.meanConfidence).toBe(1);
  });

  it("predicts a concrete, slot-filled movement", () => {
    const samples = buildMovementSamples(synthesizeWorkflowReplay({ target: "report" }));
    const backend = new NgramMovementBackend(4);
    backend.train(samples);

    const prediction = backend.predict(samples[2]!.context);
    expect(prediction).toMatchObject({ tool: "device", verb: "save", object: "report", summary: "save report" });
    expect(prediction?.backoffOrder).toBe(4); // capped at the backend's n-gram order
  });
});

describe("NgramMovementBackend generalization (objective 2d)", () => {
  it("transfers a learned workflow shape to an unseen target via backoff + slot fill", () => {
    const trainSamples = buildMovementSamplesFromReplays(
      synthesizeWorkflowDataset(["report", "invoice", "ticket"]),
    );
    const testSamples = buildMovementSamples(synthesizeWorkflowReplay({ target: "memo" }));

    const backend = new NgramMovementBackend(4);
    backend.train(trainSamples);

    const generalization = measureGeneralization(backend, trainSamples, testSamples);
    // "memo" was never in training, yet the object-free shape transfers.
    expect(generalization.novel).toBe(3);
    expect(generalization.novelCorrect).toBe(3);
    expect(generalization.generalizationAccuracy).toBe(1);
    expect(generalization.accuracy).toBe(1);

    // Concretely: the held-out "save" step is predicted with the new object.
    const savePrediction = backend.predict(testSamples[2]!.context);
    expect(savePrediction?.summary).toBe("save memo");
  });

  it("backs off to a shorter suffix when the full context is unseen", () => {
    const backend = new NgramMovementBackend(4);
    // Trained only on the context ["a","b"] -> save.
    backend.train([{ context: { tokens: ["a", "b"], object: "doc" }, label: { tool: "device", verb: "save", object: "doc" } }]);

    // Query with a novel leading token: order-2 key ["z","b"] misses, so the
    // model falls back to the order-1 suffix ["b"] it does know.
    const prediction = backend.predict({ tokens: ["z", "b"], object: "note" });
    expect(prediction?.verb).toBe("save");
    expect(prediction?.summary).toBe("save note");
    expect(prediction?.backoffOrder).toBe(1);
  });

  it("returns undefined when nothing has been trained", () => {
    const backend = new NgramMovementBackend();
    expect(backend.predict({ tokens: ["obs␟os␟focus"] })).toBeUndefined();
  });
});

describe("NgramMovementBackend snapshot round-trip", () => {
  it("serializes deterministically and reloads to an equivalent model", () => {
    const samples = buildMovementSamplesFromReplays(synthesizeWorkflowDataset(["report", "invoice"]));
    const backend = new NgramMovementBackend(4);
    backend.train(samples);

    const snapshot = backend.snapshot();
    expect(snapshot.backendId).toBe("ngram-movement");
    // JSON-serializable and stable ordering.
    expect(JSON.stringify(backend.snapshot())).toBe(JSON.stringify(snapshot));

    const reloaded = NgramMovementBackend.fromSnapshot(snapshot);
    const test = buildMovementSamples(synthesizeWorkflowReplay({ target: "memo" }));
    expect(evaluateMovementBackend(reloaded, test)).toEqual(evaluateMovementBackend(backend, test));
    expect(reloaded.snapshot()).toEqual(snapshot);
  });
});

describe("pluggable backend contract", () => {
  it("evaluates any MovementPolicyBackend implementation", () => {
    // A trivial constant backend proving the interface is honored, not the ngram.
    class ConstantBackend implements MovementPolicyBackend {
      readonly backendId = "constant";
      private label = { tool: "device", verb: "save" };
      train(samples: MovementSample[]): void {
        if (samples[0]) {
          this.label = { tool: samples[0].label.tool, verb: samples[0].label.verb };
        }
      }
      predict(context: { tokens: string[]; object?: string }) {
        return {
          tool: this.label.tool,
          verb: this.label.verb,
          summary: renderMovementSummary(this.label.verb, context.object),
          ...(context.object ? { object: context.object } : {}),
          confidence: 1,
          backoffOrder: 0,
          support: 1,
        };
      }
      snapshot() {
        return { version: 1 as const, backendId: this.backendId, order: 0, entries: [] };
      }
    }

    const backend: MovementPolicyBackend = new ConstantBackend();
    backend.train([{ context: { tokens: [] }, label: { tool: "device", verb: "open", object: "x" } }]);
    const result = evaluateMovementBackend(backend, [
      { context: { tokens: [], object: "y" }, label: { tool: "device", verb: "open", object: "y" } },
    ]);
    expect(result.total).toBe(1);
    expect(result.correct).toBe(1);
  });
});
