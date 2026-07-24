import { describe, expect, it } from "vitest";
import type { ReplayTimelineEvent } from "../capture/replay.js";
import {
  MarkovMovementBackend,
  evaluateMovementModel,
  loadMovementModel,
  sequencesFromReplays,
  tokenizeReplayAction,
  type MovementSequence,
} from "./model-backend.js";

const backend = new MarkovMovementBackend();

/** A small synthetic corpus of "open app -> navigate -> confirm" style flows. */
const CORPUS: MovementSequence[] = [
  ["tap:launcher", "tap:search", "type:query", "tap:result", "tap:confirm"],
  ["tap:launcher", "tap:search", "type:query", "tap:result", "tap:confirm"],
  ["tap:launcher", "tap:settings", "swipe:down", "tap:toggle", "tap:confirm"],
];

describe("MarkovMovementBackend", () => {
  it("repeats a recorded movement sequence verbatim from its seed", async () => {
    const model = await backend.train({ sequences: CORPUS }, { order: 3 });
    const generated = model.generate(["tap:launcher", "tap:search"]);
    expect(generated).toEqual(["type:query", "tap:result", "tap:confirm"]);
  });

  it("uses full-order context when the prefix was seen verbatim", async () => {
    const model = await backend.train({ sequences: CORPUS }, { order: 3 });
    const prediction = model.predictNext(["tap:search", "type:query"]);
    expect(prediction?.token).toBe("tap:result");
    expect(prediction?.backoffOrder).toBe(2);
    expect(prediction?.probability).toBeGreaterThan(0.9);
  });

  it("generalizes to an unseen-but-related context via backoff", async () => {
    const model = await backend.train({ sequences: CORPUS }, { order: 3 });
    // This exact 2-gram context never occurred, but "tap:toggle" -> "tap:confirm"
    // and "tap:result" -> "tap:confirm" did, so backing off still predicts confirm.
    const prediction = model.predictNext(["tap:unknown-screen", "tap:toggle"]);
    expect(prediction?.token).toBe("tap:confirm");
    expect(prediction?.backoffOrder).toBeLessThan(2); // had to back off
  });

  it("falls back to the unigram distribution for a fully cold context", async () => {
    const model = await backend.train({ sequences: CORPUS }, { order: 3 });
    const prediction = model.predictNext([]);
    // No context to condition on -> backs off to the order-0 unigram frequency
    // and returns a real (non-sentinel) max-frequency movement token.
    expect(prediction?.backoffOrder).toBe(0);
    expect(prediction?.token).not.toContain("END");
    expect(model.vocabulary()).toContain(prediction?.token);
    // tap:launcher and tap:confirm both occur 3x (the modal frequency); the
    // deterministic tie-break picks the lexicographically-first real token.
    expect(prediction?.token).toBe("tap:confirm");
  });

  it("is deterministic across repeated training runs", async () => {
    const a = await backend.train({ sequences: CORPUS }, { order: 3 });
    const b = await backend.train({ sequences: CORPUS }, { order: 3 });
    expect(a.toJSON()).toEqual(b.toJSON());
    expect(a.generate(["tap:launcher"])).toEqual(b.generate(["tap:launcher"]));
  });

  it("terminates generation at the learned end of a sequence", async () => {
    const model = await backend.train({ sequences: CORPUS }, { order: 3 });
    const generated = model.generate(["tap:launcher", "tap:settings", "swipe:down", "tap:toggle"], { steps: 32 });
    expect(generated).toEqual(["tap:confirm"]);
  });

  it("ranks candidate continuations by probability, most likely first", async () => {
    const model = await backend.train({ sequences: CORPUS }, { order: 3 });
    const ranked = model.rankNext(["tap:launcher"]);
    expect(ranked.length).toBeGreaterThan(1);
    expect(ranked[0]?.token).toBe("tap:search"); // 2 of 3 corpus flows go launcher->search
    for (let i = 1; i < ranked.length; i += 1) {
      expect(ranked[i - 1]!.probability).toBeGreaterThanOrEqual(ranked[i]!.probability);
    }
  });

  it("scores a corpus sequence as more likely than random noise", async () => {
    const model = await backend.train({ sequences: CORPUS }, { order: 3 });
    const known = model.logLikelihood(CORPUS[0]!);
    const noise = model.logLikelihood(["swipe:down", "tap:launcher", "type:query", "tap:launcher"]);
    expect(known).toBeGreaterThan(noise);
  });

  it("excludes the internal END sentinel from the public vocabulary", async () => {
    const model = await backend.train({ sequences: CORPUS }, { order: 3 });
    const vocab = model.vocabulary();
    expect(vocab).toContain("tap:confirm");
    expect(vocab.some((token) => token.includes("END"))).toBe(false);
  });

  it("round-trips through serialization without behavioural change", async () => {
    const model = await backend.train({ sequences: CORPUS }, { order: 3 });
    const restored = loadMovementModel(model.toJSON());
    expect(restored.toJSON()).toEqual(model.toJSON());
    expect(restored.generate(["tap:launcher", "tap:search"])).toEqual(
      model.generate(["tap:launcher", "tap:search"]),
    );
    expect(restored.predictNext(["tap:launcher"])?.token).toBe(model.predictNext(["tap:launcher"])?.token);
  });

  it("handles an empty corpus without throwing", async () => {
    const model = await backend.train({ sequences: [] });
    expect(model.predictNext(["anything"])).toBeUndefined();
    expect(model.generate(["anything"])).toEqual([]);
    expect(model.vocabulary()).toEqual([]);
  });
});

describe("sequencesFromReplays", () => {
  const action = (ts: number, tool: string, summary: string): ReplayTimelineEvent => ({
    kind: "action",
    ts,
    trajectoryId: "traj-1",
    tool,
    summary,
  });
  const observation = (ts: number): ReplayTimelineEvent => ({
    kind: "observation",
    ts,
    trajectoryId: "traj-1",
    source: "device",
    summary: "app active",
  });

  it("extracts one time-ordered action sequence per replay, ignoring observations", () => {
    const dataset = sequencesFromReplays([
      {
        events: [
          action(30, "device", "tapped confirm"),
          observation(5),
          action(10, "device", "tapped launcher"),
          action(20, "device", "typed query"),
        ],
      },
    ]);
    expect(dataset.sequences).toEqual([
      ["device:tapped launcher", "device:typed query", "device:tapped confirm"],
    ]);
  });

  it("drops replays that contain no actions", () => {
    const dataset = sequencesFromReplays([{ events: [observation(1), observation(2)] }]);
    expect(dataset.sequences).toEqual([]);
  });

  it("supports a custom tokenizer", () => {
    const dataset = sequencesFromReplays(
      [{ events: [action(1, "device", "tapped launcher")] }],
      { tokenize: (event) => event.summary.toUpperCase() },
    );
    expect(dataset.sequences).toEqual([["TAPPED LAUNCHER"]]);
  });

  it("tokenizeReplayAction normalizes whitespace in the summary", () => {
    expect(tokenizeReplayAction({ kind: "action", ts: 1, trajectoryId: "t", tool: "device", summary: "swiped   up" })).toBe(
      "device:swiped up",
    );
  });
});

describe("evaluateMovementModel", () => {
  it("reports high next-token accuracy on held-out related sequences", async () => {
    const model = await backend.train({ sequences: CORPUS }, { order: 3 });
    // Held-out flow that recombines familiar transitions (generalization target).
    const heldOut: MovementSequence[] = [["tap:launcher", "tap:search", "type:query", "tap:result", "tap:confirm"]];
    const report = evaluateMovementModel(model, heldOut);
    expect(report.sequences).toBe(1);
    expect(report.predictedTokens).toBe(4);
    expect(report.nextTokenAccuracy).toBe(1);
    expect(report.perplexity).toBeGreaterThan(0);
    expect(Number.isFinite(report.perplexity)).toBe(true);
  });

  it("penalizes an unrelated held-out sequence with lower accuracy", async () => {
    const model = await backend.train({ sequences: CORPUS }, { order: 3 });
    const related = evaluateMovementModel(model, [["tap:launcher", "tap:search", "type:query"]]);
    const unrelated = evaluateMovementModel(model, [["swipe:up", "tap:random", "type:gibberish"]]);
    expect(related.nextTokenAccuracy).toBeGreaterThan(unrelated.nextTokenAccuracy);
    expect(unrelated.perplexity).toBeGreaterThan(related.perplexity);
  });

  it("returns zeroed metrics for an empty held-out set", async () => {
    const model = await backend.train({ sequences: CORPUS });
    const report = evaluateMovementModel(model, []);
    expect(report).toEqual({
      sequences: 0,
      predictedTokens: 0,
      correctTokens: 0,
      nextTokenAccuracy: 0,
      perplexity: 0,
    });
  });
});
