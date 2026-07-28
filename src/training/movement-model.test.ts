import { describe, expect, it } from "vitest";
import type { ReplayManifest } from "../capture/replay.js";
import {
  buildMovementDataset,
  buildMovementDatasetFromTrajectories,
  createMovementModelBackend,
  evaluateReplayFidelity,
  MarkovMovementBackend,
  movementTokenKey,
  parseMovementToken,
  partitionSequences,
  synthesizeMovementSequences,
  tokenizeReplayManifest,
  type MovementSequence,
  type MovementToken,
} from "./movement-model.js";
import { buildTrajectorySpan } from "../capture/trajectory.js";

function actionEvent(trajectoryId: string, tool: string, summary: string, ts: number) {
  return { kind: "action" as const, ts, trajectoryId, tool, summary };
}

function manifest(sessionId: string, trajectoryId: string, actions: Array<[string, string]>): ReplayManifest {
  return {
    version: 1,
    sessionId,
    trajectoryIds: [trajectoryId],
    eventCount: actions.length,
    events: actions.map(([tool, summary], index) => actionEvent(trajectoryId, tool, summary, index)),
  };
}

describe("parseMovementToken", () => {
  it("splits a summary into verb + normalized target and drops stopwords", () => {
    expect(parseMovementToken("device", "tapped the Send button")).toEqual({
      tool: "device",
      verb: "tapped",
      target: "send button",
    });
  });

  it("handles verb-only summaries", () => {
    expect(parseMovementToken("device", "swiped")).toEqual({ tool: "device", verb: "swiped" });
  });

  it("falls back to a placeholder for empty summaries", () => {
    expect(parseMovementToken("", "")).toEqual({ tool: "unknown", verb: "act" });
  });
});

describe("tokenizeReplayManifest / buildMovementDataset", () => {
  it("keeps only action events in order", () => {
    const replay: ReplayManifest = {
      version: 1,
      sessionId: "s1",
      trajectoryIds: ["t1"],
      eventCount: 3,
      events: [
        { kind: "observation", ts: 0, trajectoryId: "t1", source: "device", summary: "app active" },
        actionEvent("t1", "device", "tapped compose button", 1),
        { kind: "transcript", ts: 2, messageId: "m1", role: "user", content: "hi" },
        actionEvent("t1", "device", "typed body field", 3),
      ],
    };
    const tokens = tokenizeReplayManifest(replay);
    expect(tokens.map((t) => t.verb)).toEqual(["tapped", "typed"]);
  });

  it("drops empty sequences", () => {
    const dataset = buildMovementDataset([
      manifest("s1", "t1", [["device", "tapped send button"]]),
      { version: 1, sessionId: "s2", trajectoryIds: ["t2"], eventCount: 0, events: [] },
    ]);
    expect(dataset.sequences).toHaveLength(1);
    expect(dataset.sequences[0]?.id).toBe("t1");
  });
});

describe("buildMovementDatasetFromTrajectories", () => {
  it("sorts actions by timestamp and tokenizes them", () => {
    const span = buildTrajectorySpan({
      id: "t1",
      sessionId: "s1",
      actions: [
        { kind: "action", tool: "device", summary: "tapped send button", ts: 5 },
        { kind: "action", tool: "device", summary: "typed body field", ts: 1 },
      ],
    });
    const dataset = buildMovementDatasetFromTrajectories([span]);
    expect(dataset.sequences[0]?.tokens.map((t) => t.verb)).toEqual(["typed", "tapped"]);
  });
});

describe("MarkovMovementBackend", () => {
  const backend = new MarkovMovementBackend(2);

  const dataset = buildMovementDataset([
    manifest("s1", "t1", [
      ["device", "tapped compose button"],
      ["device", "typed body field"],
      ["device", "tapped send button"],
    ]),
    manifest("s2", "t2", [
      ["device", "tapped compose button"],
      ["device", "typed body field"],
      ["device", "tapped send button"],
    ]),
  ]);

  it("trains a model with per-order transition tables and a vocabulary", () => {
    const model = backend.train(dataset);
    expect(model.backend).toBe("markov");
    expect(model.order).toBe(2);
    expect(model.orders).toHaveLength(3);
    expect(model.sequenceCount).toBe(2);
    expect(model.tokenCount).toBe(6);
    expect(Object.keys(model.vocabulary)).toContain("device:tapped:compose button");
  });

  it("predicts the recorded next movement (repeat capability)", () => {
    const model = backend.train(dataset);
    const context: MovementToken[] = [
      { tool: "device", verb: "tapped", target: "compose button" },
      { tool: "device", verb: "typed", target: "body field" },
    ];
    const prediction = backend.predictNext(model, context);
    expect(prediction?.token).toEqual({ tool: "device", verb: "tapped", target: "send button" });
    expect(prediction?.probability).toBeGreaterThan(0);
    expect(prediction?.backoffOrder).toBe(2);
  });

  it("regenerates the full recorded movement sequence from a prime", () => {
    const model = backend.train(dataset);
    const generated = backend.generate(model, {
      prime: [{ tool: "device", verb: "tapped", target: "compose button" }],
      maxSteps: 5,
    });
    expect(generated.map(movementTokenKey)).toEqual([
      "device:tapped:compose button",
      "device:typed:body field",
      "device:tapped:send button",
    ]);
  });

  it("is deterministic across repeated training and generation", () => {
    const a = backend.train(dataset);
    const b = backend.train(dataset);
    expect(JSON.stringify(a)).toBe(JSON.stringify(b));
    expect(backend.generate(a, { prime: [{ tool: "device", verb: "tapped", target: "compose button" }] })).toEqual(
      backend.generate(b, { prime: [{ tool: "device", verb: "tapped", target: "compose button" }] }),
    );
  });

  it("backs off to a shorter context for unseen histories (generalization)", () => {
    const model = backend.train(dataset);
    // Context never observed verbatim, but "typed body field" -> follows via order-1 backoff.
    const prediction = backend.predictNext(model, [
      { tool: "device", verb: "shortcut", target: "unknown thing" },
      { tool: "device", verb: "typed", target: "body field" },
    ]);
    expect(prediction).toBeDefined();
    expect(prediction?.backoffOrder).toBeLessThan(2);
    expect(prediction?.token.target).toBe("send button");
  });

  it("returns undefined when the model has no data", () => {
    const empty = backend.train({ version: 1, sequences: [] });
    expect(backend.predictNext(empty, [])).toBeUndefined();
    expect(backend.generate(empty)).toEqual([]);
  });

  it("stops generation on a cycle rather than looping forever", () => {
    const loopy = buildMovementDataset([
      manifest("s1", "t1", [
        ["device", "scrolled down"],
        ["device", "scrolled down"],
        ["device", "scrolled down"],
      ]),
    ]);
    const model = backend.train(loopy);
    const generated = backend.generate(model, {
      prime: [{ tool: "device", verb: "scrolled", target: "down" }],
      maxSteps: 50,
      cycleGuard: 3,
    });
    expect(generated.length).toBeLessThan(50);
  });
});

describe("createMovementModelBackend", () => {
  it("returns a markov backend by default", () => {
    expect(createMovementModelBackend().id).toBe("markov");
    expect(createMovementModelBackend("markov", { order: 3 })).toBeInstanceOf(MarkovMovementBackend);
  });
});

describe("synthetic generation + generalization eval", () => {
  it("produces related-but-distinct sequences deterministically", () => {
    const first = synthesizeMovementSequences({ variants: 3 });
    const second = synthesizeMovementSequences({ variants: 3 });
    expect(JSON.stringify(first)).toBe(JSON.stringify(second));
    expect(first.length).toBeGreaterThan(3);
    // Variants of the same task share their first verb but can differ in target.
    const compose = first.filter((s) => s.id.startsWith("compose-message"));
    expect(compose).toHaveLength(3);
    expect(new Set(compose.map((s) => s.tokens[0]?.verb))).toEqual(new Set(["tapped"]));
  });

  it("generalizes to held-out related sequences above chance", () => {
    const backend = createMovementModelBackend("markov", { order: 2 });
    const sequences = synthesizeMovementSequences({ variants: 6 });
    const { train, heldOut } = partitionSequences(sequences, 3);
    expect(train.length).toBeGreaterThan(0);
    expect(heldOut.length).toBeGreaterThan(0);

    const model = backend.train({ version: 1, sequences: train });
    const report = evaluateReplayFidelity(backend, model, heldOut);

    expect(report.sequences).toBe(heldOut.length);
    expect(report.predictions).toBeGreaterThan(0);
    // Held-out sequences were never trained on, yet backoff should recover most
    // next-movements from the shared task structure.
    expect(report.accuracy).toBeGreaterThan(0.5);
  });

  it("predicts every non-cold-start step of a memorized sequence exactly", () => {
    const backend = createMovementModelBackend();
    const sequences: MovementSequence[] = synthesizeMovementSequences({ variants: 1 });
    const model = backend.train({ version: 1, sequences });

    // Cold-start (empty-context) prediction falls back to the global prior and
    // is not memorizable to one sequence, but every subsequent step is.
    for (const sequence of sequences) {
      for (let position = 1; position < sequence.tokens.length; position += 1) {
        const prediction = backend.predictNext(model, sequence.tokens.slice(0, position));
        expect(prediction?.tokenKey).toBe(movementTokenKey(sequence.tokens[position]!));
      }
    }
  });
});
