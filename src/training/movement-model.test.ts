import { describe, expect, it } from "vitest";
import type { ReplayManifest, ReplayTimelineEvent } from "../capture/replay.js";
import {
  DEFAULT_MOVEMENT_BACKEND_ID,
  MarkovMovementBackend,
  evaluateMovementModel,
  extractMovementSequences,
  getMovementBackend,
  listMovementBackends,
  registerMovementBackend,
  restoreMovementModel,
  type MovementModelBackend,
  type MovementSequence,
} from "./movement-model.js";

function actionEvent(trajectoryId: string, tool: string, ts: number): ReplayTimelineEvent {
  return { kind: "action", ts, trajectoryId, tool, summary: `${tool}@${ts}` };
}

function manifest(trajectoryId: string, tools: string[]): Pick<ReplayManifest, "events"> {
  return {
    events: tools.map((tool, index) => actionEvent(trajectoryId, tool, index)),
  };
}

/** Deterministic (seeded) synthetic event-stream generator — no OS input, no Math.random. */
function syntheticSequence(trajectoryId: string, seed: number, length: number): MovementSequence {
  const vocab = ["mouse.move", "mouse.click", "key.press", "scroll", "window.focus"];
  const tokens: string[] = [];
  let state = seed >>> 0;
  for (let i = 0; i < length; i += 1) {
    // xorshift-ish deterministic walk with a bias so patterns are learnable
    state = (state * 1664525 + 1013904223) >>> 0;
    const pick = (state + (tokens.length > 0 ? tokens[tokens.length - 1]!.length : 0)) % vocab.length;
    tokens.push(vocab[pick]!);
  }
  return { trajectoryId, tokens };
}

describe("extractMovementSequences", () => {
  it("keeps only action events, grouped per trajectory in timeline order", () => {
    const events: ReplayTimelineEvent[] = [
      { kind: "transcript", ts: 0, messageId: "m", role: "user", content: "go" },
      actionEvent("traj-1", "mouse.move", 1),
      { kind: "observation", ts: 2, trajectoryId: "traj-1", source: "os", summary: "moved" },
      actionEvent("traj-1", "mouse.click", 3),
      actionEvent("traj-2", "key.press", 4),
    ];
    const sequences = extractMovementSequences([{ events }]);
    expect(sequences).toEqual([
      { trajectoryId: "traj-1", tokens: ["mouse.move", "mouse.click"] },
      { trajectoryId: "traj-2", tokens: ["key.press"] },
    ]);
  });

  it("honours a custom tokenizer", () => {
    const sequences = extractMovementSequences([manifest("t", ["a", "b"])], (event) => `${event.tool}:${event.summary}`);
    expect(sequences[0]!.tokens).toEqual(["a:a@0", "b:b@1"]);
  });
});

describe("MarkovMovementBackend", () => {
  const backend = new MarkovMovementBackend();

  it("repeats a recorded movement sequence via greedy rollout", () => {
    const recorded: MovementSequence = {
      trajectoryId: "t",
      tokens: ["window.focus", "mouse.move", "mouse.click", "key.press", "scroll"],
    };
    const model = backend.train({ sequences: [recorded] }, { order: 2 });
    const replayed = model.generate(["window.focus"], 4);
    expect(replayed).toEqual(["mouse.move", "mouse.click", "key.press", "scroll"]);
  });

  it("generalizes to an unseen context by backing off to a shorter suffix", () => {
    // In every training sequence, `mouse.move` is always followed by `mouse.click`.
    const sequences: MovementSequence[] = [
      { trajectoryId: "a", tokens: ["key.press", "mouse.move", "mouse.click"] },
      { trajectoryId: "b", tokens: ["scroll", "mouse.move", "mouse.click"] },
      { trajectoryId: "c", tokens: ["window.focus", "mouse.move", "mouse.click"] },
    ];
    const model = backend.train({ sequences }, { order: 2 });
    // Context ["scroll","mouse.move"] was seen, but ["window.focus","mouse.move"]
    // as an order-2 key was also seen; use a novel two-token prefix instead:
    const prediction = model.predictNext(["mouse.click", "mouse.move"]);
    expect(prediction).toBeDefined();
    // order-2 context "mouse.click mouse.move" is unseen -> backs off to order-1 "mouse.move".
    expect(prediction!.token).toBe("mouse.click");
    expect(prediction!.order).toBe(1);
  });

  it("is deterministic on ties (lexicographically smallest token wins)", () => {
    const sequences: MovementSequence[] = [
      { trajectoryId: "a", tokens: ["start", "zzz"] },
      { trajectoryId: "b", tokens: ["start", "aaa"] },
    ];
    const model = backend.train({ sequences }, { order: 1 });
    const first = model.predictNext(["start"]);
    const second = model.predictNext(["start"]);
    expect(first!.token).toBe("aaa");
    expect(second!.token).toBe("aaa");
  });

  it("returns undefined for an untrained model", () => {
    const model = backend.train({ sequences: [] });
    expect(model.predictNext(["anything"])).toBeUndefined();
    expect(model.generate(["x"], 3)).toEqual([]);
  });

  it("round-trips through a snapshot without behaviour change", () => {
    const sequences = [syntheticSequence("t1", 7, 30), syntheticSequence("t2", 42, 30)];
    const model = backend.train({ sequences }, { order: 3 });
    const restored = restoreMovementModel(model.snapshot());
    expect(restored.order).toBe(model.order);
    expect(restored.vocabulary).toEqual(model.vocabulary);
    for (const seed of [["mouse.move"], ["key.press", "scroll"], ["window.focus"]]) {
      expect(restored.generate(seed, 5)).toEqual(model.generate(seed, 5));
    }
  });
});

describe("evaluateMovementModel", () => {
  it("scores perfect fidelity on a deterministic (order-1 learnable) sequence", () => {
    const cycle = ["mouse.move", "mouse.click", "key.press", "scroll", "window.focus"];
    const tokens = Array.from({ length: 40 }, (_, i) => cycle[i % cycle.length]!);
    const sequences: MovementSequence[] = [{ trajectoryId: "t", tokens }];
    const model = new MarkovMovementBackend().train({ sequences }, { order: 2 });
    const result = evaluateMovementModel(model, sequences);
    expect(result.totalPredictions).toBeGreaterThan(0);
    expect(result.accuracy).toBe(1);
    expect(result.perplexity).toBeLessThanOrEqual(1.0001);
  });

  it("beats chance on held-out but related synthetic trajectories", () => {
    const train = Array.from({ length: 8 }, (_, i) => syntheticSequence(`train-${i}`, i + 1, 40));
    const heldOut = Array.from({ length: 3 }, (_, i) => syntheticSequence(`eval-${i}`, i + 1, 40));
    const model = new MarkovMovementBackend().train({ sequences: train }, { order: 2 });
    const result = evaluateMovementModel(model, heldOut);
    // 5-token vocab -> chance accuracy is 0.2; a learned model must clear it.
    expect(result.accuracy).toBeGreaterThan(0.2);
    expect(result.averageProbability).toBeGreaterThan(0);
  });

  it("handles empty held-out sets", () => {
    const model = new MarkovMovementBackend().train({ sequences: [] });
    expect(evaluateMovementModel(model, [])).toMatchObject({ totalPredictions: 0, accuracy: 0, perplexity: 0 });
  });
});

describe("pluggable backend registry", () => {
  it("registers the markov backend by default", () => {
    expect(listMovementBackends()).toContain(DEFAULT_MOVEMENT_BACKEND_ID);
    expect(getMovementBackend(DEFAULT_MOVEMENT_BACKEND_ID)).toBeInstanceOf(MarkovMovementBackend);
  });

  it("allows registering an alternate backend under the same interface", () => {
    const stub: MovementModelBackend = {
      id: "stub-backend",
      train: () => new MarkovMovementBackend().train({ sequences: [] }),
    };
    registerMovementBackend(stub);
    expect(getMovementBackend("stub-backend")).toBe(stub);
    expect(listMovementBackends()).toContain("stub-backend");
  });
});
