import { describe, expect, it } from "vitest";
import { buildTrajectorySpan, type TrajectorySpan } from "../capture/trajectory.js";
import {
  MarkovMovementBackend,
  MovementModelBackendRegistry,
  buildMovementDataset,
  canonicalize,
  createDefaultMovementBackendRegistry,
  evaluateMovementModel,
  type MovementDataset,
  type MovementSequence,
  type MovementToken,
} from "./movement-model.js";

function token(key: string, kind: MovementToken["kind"] = "action"): MovementToken {
  return { key, kind, label: key };
}

function seq(id: string, keys: string[]): MovementSequence {
  return { id, tokens: keys.map((key) => token(key)) };
}

describe("canonicalize", () => {
  it("collapses phrasing, numbers, and punctuation into a stable symbol", () => {
    expect(canonicalize("Tapped Submit button")).toBe("tapped_submit_button");
    expect(canonicalize("tapped submit!")).toBe("tapped_submit");
    expect(canonicalize("opened file /var/log/9.txt")).toBe("opened_file_var_log");
  });

  it("never returns an empty symbol", () => {
    expect(canonicalize("123 456")).toBe("#_#");
    expect(canonicalize("")).toBe("_");
  });
});

describe("buildMovementDataset", () => {
  it("interleaves observations and actions in timestamp order", () => {
    const trajectory: TrajectorySpan = buildTrajectorySpan({
      id: "t1",
      sessionId: "s1",
      observations: [
        { kind: "observation", source: "os", summary: "focused Editor", ts: 10 },
        { kind: "observation", source: "device", summary: "app active", ts: 30 },
      ],
      actions: [{ kind: "action", tool: "device", summary: "tapped Save", ts: 20 }],
    });

    const dataset = buildMovementDataset([trajectory]);
    expect(dataset.sequences).toHaveLength(1);
    expect(dataset.sequences[0]!.tokens.map((t) => t.key)).toEqual([
      "obs:os:focused_editor",
      "act:device:tapped_save",
      "obs:device:app_active",
    ]);
  });
});

describe("MarkovMovementBackend recall", () => {
  const backend = new MarkovMovementBackend();

  it("reproduces a recorded movement sequence exactly from its start", () => {
    const dataset: MovementDataset = { version: 1, sequences: [seq("a", ["open", "type", "save", "close"])] };
    const model = backend.train(dataset, { order: 3 });
    const generated = backend.generate(model, [token("open")], { maxTokens: 10 });
    expect(generated.map((t) => t.key)).toEqual(["type", "save", "close"]);
  });

  it("predicts the most frequent continuation deterministically", () => {
    const dataset: MovementDataset = {
      version: 1,
      sequences: [seq("a", ["open", "save"]), seq("b", ["open", "save"]), seq("c", ["open", "quit"])],
    };
    const model = backend.train(dataset, { order: 1 });
    const prediction = backend.predictNext(model, [token("open")]);
    expect(prediction?.token.key).toBe("save");
    expect(prediction?.probability).toBeCloseTo(2 / 3);
    expect(prediction?.matchedOrder).toBe(1);
    expect(prediction?.distribution.map((d) => d.token.key)).toEqual(["save", "quit"]);
  });
});

describe("MarkovMovementBackend generalization (back-off)", () => {
  const backend = new MarkovMovementBackend();

  it("backs off to a shorter context for an unseen prefix", () => {
    // "save" is always followed by "close" across recorded data...
    const dataset: MovementDataset = {
      version: 1,
      sequences: [seq("a", ["open", "save", "close"]), seq("b", ["edit", "save", "close"])],
    };
    const model = backend.train(dataset, { order: 2 });
    // ...so a novel prefix ["reopen", "save"] (never seen verbatim) should
    // still generalize to "close" via the order-1 context "save".
    const prediction = backend.predictNext(model, [token("reopen"), token("save")]);
    expect(prediction?.token.key).toBe("close");
    expect(prediction?.matchedOrder).toBe(1);
  });

  it("falls back to the global distribution when no context matches", () => {
    const dataset: MovementDataset = { version: 1, sequences: [seq("a", ["ping", "ping", "pong"])] };
    const model = backend.train(dataset, { order: 2 });
    const prediction = backend.predictNext(model, [token("totally-unseen")]);
    expect(prediction?.matchedOrder).toBe(0);
    expect(prediction?.token.key).toBe("ping"); // most frequent overall
  });
});

describe("MarkovMovementBackend generate", () => {
  const backend = new MarkovMovementBackend();

  it("honours maxTokens and a stop key", () => {
    const dataset: MovementDataset = { version: 1, sequences: [seq("a", ["a", "b", "c", "d", "e"])] };
    const model = backend.train(dataset, { order: 4 });
    expect(backend.generate(model, [token("a")], { maxTokens: 2 }).map((t) => t.key)).toEqual(["b", "c"]);
    expect(backend.generate(model, [token("a")], { stopKeys: ["c"], maxTokens: 10 }).map((t) => t.key)).toEqual([
      "b",
      "c",
    ]);
  });
});

describe("model serialization", () => {
  it("round-trips a trained model through serialize/deserialize", () => {
    const backend = new MarkovMovementBackend();
    const dataset: MovementDataset = { version: 1, sequences: [seq("a", ["open", "save", "close"])] };
    const model = backend.train(dataset, { order: 2 });
    const restored = backend.deserialize(backend.serialize(model));
    expect(backend.generate(restored, [token("open")], { maxTokens: 5 }).map((t) => t.key)).toEqual([
      "save",
      "close",
    ]);
  });

  it("rejects a foreign backend payload", () => {
    const backend = new MarkovMovementBackend();
    expect(() => backend.deserialize(JSON.stringify({ backend: "other" }))).toThrow(/expected markov/);
  });
});

describe("MovementModelBackendRegistry", () => {
  it("registers, resolves, and lists backends", () => {
    const registry = createDefaultMovementBackendRegistry();
    expect(registry.has("markov")).toBe(true);
    expect(registry.list()).toEqual(["markov"]);
    expect(registry.get("markov").name).toBe("markov");
  });

  it("throws a helpful error for an unknown backend", () => {
    const registry = new MovementModelBackendRegistry();
    expect(() => registry.get("nope")).toThrow(/unknown movement-model backend: nope/);
  });
});

describe("evaluateMovementModel", () => {
  const backend = new MarkovMovementBackend();

  it("reports perfect reproduction on the training distribution", () => {
    const sequences = [seq("a", ["open", "save", "close"]), seq("b", ["open", "save", "close"])];
    const model = backend.train({ version: 1, sequences }, { order: 3 });
    const result = evaluateMovementModel(backend, model, sequences, { seedTokens: 1 });
    expect(result.accuracy).toBe(1);
    expect(result.exactSequenceMatch).toBe(1);
    expect(result.scoredTokens).toBe(4);
  });

  it("measures partial generalization to related held-out sequences", () => {
    // Train on prefixes that all end "save"->"close"; hold out a novel prefix.
    const train = [seq("a", ["open", "save", "close"]), seq("b", ["edit", "save", "close"])];
    const model = backend.train({ version: 1, sequences: train }, { order: 2 });
    const heldOut = [seq("h", ["reopen", "save", "close"])];
    const result = evaluateMovementModel(backend, model, heldOut, { seedTokens: 2 });
    // Seeded with ["reopen","save"], the model should still predict "close".
    expect(result.matchedTokens).toBe(1);
    expect(result.accuracy).toBe(1);
  });

  it("skips sequences too short to score", () => {
    const model = backend.train({ version: 1, sequences: [seq("a", ["x", "y"])] }, { order: 2 });
    const result = evaluateMovementModel(backend, model, [seq("s", ["only"])], { seedTokens: 1 });
    expect(result.sequences).toBe(0);
    expect(result.accuracy).toBe(0);
  });
});
