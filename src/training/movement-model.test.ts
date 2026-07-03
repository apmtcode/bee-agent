import { describe, expect, it } from "vitest";
import type { ReplayManifest } from "../capture/replay.js";
import { buildTrajectorySpan, type TrajectoryAction } from "../capture/trajectory.js";
import {
  MarkovMovementBackend,
  buildMovementDataset,
  evaluateMovementModel,
  splitMovementDataset,
  synthesizeMovementSequences,
  tokenizeAction,
  tokenizeReplayManifest,
  tokenizeTrajectory,
  type MovementDataset,
} from "./movement-model.js";

function action(tool: string, summary: string, metadata?: Record<string, unknown>, ts = 0): TrajectoryAction {
  return { kind: "action", tool, summary, ts, ...(metadata ? { metadata } : {}) };
}

describe("tokenizeAction", () => {
  it("prefers structured gesture metadata over free-text summary", () => {
    expect(tokenizeAction(action("device", "swiped to the left", { gesture: "swipe", direction: "left" }))).toBe(
      "device:swipe:left",
    );
    expect(tokenizeAction(action("device", "tapped Login", { gesture: "tap", target: "Login" }))).toBe(
      "device:tap:login",
    );
  });

  it("falls back to the summary's leading verb when metadata is absent", () => {
    expect(tokenizeAction(action("os", "opened settings window"))).toBe("os:opened");
  });

  it("collapses differently-phrased but structurally identical movements", () => {
    const a = tokenizeAction(action("device", "typed the search query", { gesture: "type", target: "search" }));
    const b = tokenizeAction(action("device", "entered text", { gesture: "type", target: "Search" }));
    expect(a).toBe(b);
  });
});

describe("tokenization sources", () => {
  it("orders trajectory actions by timestamp", () => {
    const span = buildTrajectorySpan({
      id: "t1",
      sessionId: "s1",
      actions: [
        action("device", "send", { gesture: "tap", target: "send" }, 30),
        action("device", "focus", { gesture: "tap", target: "compose" }, 10),
        action("device", "body", { gesture: "type", target: "body" }, 20),
      ],
    });
    expect(tokenizeTrajectory(span)).toEqual(["device:tap:compose", "device:type:body", "device:tap:send"]);
  });

  it("extracts only action events from a replay manifest, in ts order", () => {
    const manifest: ReplayManifest = {
      version: 1,
      sessionId: "s1",
      trajectoryIds: ["t1"],
      eventCount: 3,
      events: [
        { kind: "observation", ts: 5, trajectoryId: "t1", source: "os", summary: "focus" },
        { kind: "action", ts: 20, trajectoryId: "t1", tool: "device", summary: "scrolled down" },
        { kind: "action", ts: 10, trajectoryId: "t1", tool: "device", summary: "tapped result" },
      ],
    };
    expect(tokenizeReplayManifest(manifest)).toEqual(["device:tapped", "device:scrolled"]);
  });
});

function sequenceSpan(id: string, tokens: string[]) {
  return buildTrajectorySpan({
    id,
    sessionId: "s",
    actions: tokens.map((token, index) => {
      const [, verb] = token.split(":");
      return action(token.split(":")[0] ?? "device", verb ?? "act", { gesture: verb }, index);
    }),
  });
}

describe("MarkovMovementBackend — repeat recorded movements (objective 2c)", () => {
  it("reproduces a recorded sequence from its first movement", () => {
    const dataset: MovementDataset = {
      sequences: [{ id: "rec", tokens: ["a:1", "b:2", "c:3", "d:4", "e:5"] }],
    };
    const model = new MarkovMovementBackend().train(dataset, { order: 3 });
    const replayed = model.generate(["a:1"], { maxSteps: 10 });
    expect(replayed).toEqual(["a:1", "b:2", "c:3", "d:4", "e:5"]);
  });

  it("stops generating exactly at the recorded end (no runaway back-off)", () => {
    const dataset: MovementDataset = { sequences: [{ id: "rec", tokens: ["x:1", "y:2", "z:3"] }] };
    const model = new MarkovMovementBackend().train(dataset, { order: 2 });
    // minMatchedOrder default 1 means: only continue while real context matches.
    expect(model.generate(["x:1"])).toEqual(["x:1", "y:2", "z:3"]);
  });
});

describe("MarkovMovementBackend — generalize to new but related movements (objective 2d)", () => {
  it("predicts a learned continuation for a prefix it never saw as a whole", () => {
    // Two recordings share the "open -> search -> type" motif in different contexts.
    const dataset: MovementDataset = {
      sequences: [
        { id: "s1", tokens: ["app:launch", "ui:open", "ui:search", "ui:type", "ui:submit"] },
        { id: "s2", tokens: ["dock:click", "ui:open", "ui:search", "ui:type", "ui:submit"] },
      ],
    };
    const model = new MarkovMovementBackend().train(dataset, { order: 3 });
    // A brand-new prefix that ends in a shared motif; exact full context unseen.
    const prediction = model.predictNext(["home:tap", "ui:open", "ui:search"]);
    expect(prediction?.token).toBe("ui:type");
    expect(prediction?.matchedOrder).toBeGreaterThanOrEqual(1);
  });
});

describe("serialization round-trip", () => {
  it("restores an identical model from a snapshot", () => {
    const dataset = synthesizeMovementSequences({ seed: 7, sequences: 6 });
    const model = new MarkovMovementBackend().train(dataset, { order: 3 });
    const restored = MarkovMovementBackend.fromSnapshot(model.serialize());

    expect(restored.order).toBe(model.order);
    expect(restored.vocabulary).toEqual(model.vocabulary);
    for (const sequence of dataset.sequences) {
      for (let i = 1; i < sequence.tokens.length; i += 1) {
        const context = sequence.tokens.slice(0, i);
        expect(restored.predictNext(context)?.token).toBe(model.predictNext(context)?.token);
      }
    }
  });
});

describe("evaluateMovementModel + synthetic generator", () => {
  it("is deterministic for a fixed seed", () => {
    const a = synthesizeMovementSequences({ seed: 42 });
    const b = synthesizeMovementSequences({ seed: 42 });
    expect(a).toEqual(b);
  });

  it("generalizes on held-out related sequences well above a unigram baseline", () => {
    const dataset = synthesizeMovementSequences({ seed: 3, sequences: 24 });
    const { train, heldOut } = splitMovementDataset(dataset, 3);
    expect(heldOut.sequences.length).toBeGreaterThan(0);

    const backend = new MarkovMovementBackend();
    const model = backend.train(train, { order: 3 });
    const evaluation = evaluateMovementModel(model, heldOut);

    // Unigram baseline: order-0 model (always predicts the single most frequent token).
    const unigram = backend.train(train, { order: 0 });
    const baseline = evaluateMovementModel(unigram, heldOut);

    expect(evaluation.coverage).toBe(1);
    expect(evaluation.accuracy).toBeGreaterThan(baseline.accuracy);
    expect(evaluation.accuracy).toBeGreaterThan(0.5);
  });

  it("reports zeroed metrics for an empty held-out set", () => {
    const model = new MarkovMovementBackend().train({ sequences: [] });
    const evaluation = evaluateMovementModel(model, { sequences: [] });
    expect(evaluation).toMatchObject({ predictions: 0, correct: 0, accuracy: 0, coverage: 0 });
  });
});

describe("buildMovementDataset", () => {
  it("combines trajectories and replay manifests and drops empty sequences", () => {
    const span = sequenceSpan("t1", ["device:tap", "device:type"]);
    const empty = buildTrajectorySpan({ id: "t2", sessionId: "s", actions: [] });
    const manifest: ReplayManifest = {
      version: 1,
      sessionId: "sess",
      trajectoryIds: [],
      eventCount: 1,
      events: [{ kind: "action", ts: 1, trajectoryId: "t", tool: "os", summary: "ran build" }],
    };
    const dataset = buildMovementDataset({ trajectories: [span, empty], manifests: [manifest] });
    expect(dataset.sequences.map((s) => s.id)).toEqual(["t1", "sess:replay"]);
  });
});
