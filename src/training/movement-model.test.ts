import { describe, expect, it } from "vitest";
import type { TrajectoryAction, TrajectorySpan } from "../capture/trajectory.js";
import {
  MarkovMovementModelBackend,
  MOVEMENT_EOS,
  actionToMovementToken,
  buildMovementDataset,
  evaluateMovementModel,
  loadMovementModel,
  tokenizeTrajectory,
  type MovementSequence,
} from "./movement-model.js";

function action(
  tool: string,
  summary: string,
  ts: number,
  metadata?: Record<string, unknown>,
): TrajectoryAction {
  return { kind: "action", tool, summary, ts, ...(metadata ? { metadata } : {}) };
}

function span(id: string, actions: TrajectoryAction[]): TrajectorySpan {
  return {
    id,
    sessionId: `session-${id}`,
    createdAt: "2026-07-15T00:00:00.000Z",
    captureTier: "operator",
    observations: [],
    actions,
  };
}

/** A canonical "open editor -> type -> save" movement, sharable across spans. */
function editWorkflow(idPrefix: string, target: string): TrajectorySpan {
  return span(`${idPrefix}`, [
    action("device", "tapped Files", 1, { gesture: "tap", target: "Files" }),
    action("device", "typed", 2, { gesture: "type", target }),
    action("device", "triggered Save", 3, { gesture: "shortcut", target: "Save" }),
  ]);
}

/**
 * A workflow whose *opening* screen varies but whose tail is a fixed, shared
 * "type body -> save -> close" sequence. Backoff generalizes over the shared
 * tail even when the opening token is novel.
 */
function openThenEdit(idPrefix: string, openTarget: string): TrajectorySpan {
  return span(`${idPrefix}`, [
    action("device", `tapped ${openTarget}`, 1, { gesture: "tap", target: openTarget }),
    action("device", "typed body", 2, { gesture: "type", target: "body" }),
    action("device", "triggered Save", 3, { gesture: "shortcut", target: "Save" }),
    action("device", "triggered Close", 4, { gesture: "shortcut", target: "Close" }),
  ]);
}

describe("actionToMovementToken", () => {
  it("collapses structurally identical actions to the same token", () => {
    const a = action("device", "tapped Login", 1, { gesture: "tap", target: "Login" });
    const b = action("device", "tapped the login button", 5, { gesture: "tap", target: "Login" });
    expect(actionToMovementToken(a)).toBe("device:tap:Login");
    expect(actionToMovementToken(a)).toBe(actionToMovementToken(b));
  });

  it("uses direction and event metadata when target is absent", () => {
    expect(actionToMovementToken(action("device", "scrolled down", 1, { gesture: "scroll", direction: "down" }))).toBe(
      "device:scroll:down",
    );
    expect(actionToMovementToken(action("browser", "navigated", 1, { action: "navigate" }))).toBe("browser:navigate");
  });

  it("falls back to a normalized summary when metadata lacks a verb", () => {
    expect(actionToMovementToken(action("tool", "Did A Thing", 1))).toBe("tool:did-a-thing");
    expect(actionToMovementToken(action("tool", "Did A Thing", 1), { useSummaryFallback: false })).toBe("tool:tool");
  });
});

describe("tokenizeTrajectory / buildMovementDataset", () => {
  it("orders tokens by timestamp regardless of insertion order", () => {
    const s = span("t1", [
      action("device", "third", 30, { gesture: "tap", target: "C" }),
      action("device", "first", 10, { gesture: "tap", target: "A" }),
      action("device", "second", 20, { gesture: "tap", target: "B" }),
    ]);
    expect(tokenizeTrajectory(s).tokens).toEqual(["device:tap:A", "device:tap:B", "device:tap:C"]);
  });

  it("drops trajectories with no actions", () => {
    const dataset = buildMovementDataset([editWorkflow("a", "hello"), span("empty", [])]);
    expect(dataset.sequences).toHaveLength(1);
    expect(dataset.sequences[0]!.trajectoryId).toBe("a");
  });
});

describe("MarkovMovementModelBackend training + inference", () => {
  it("repeats a recorded movement sequence exactly", () => {
    const dataset = buildMovementDataset([editWorkflow("rec", "hello world")]);
    const model = new MarkovMovementModelBackend(2).train(dataset);
    // Generating from scratch reproduces the recorded continuation to EOS.
    expect(model.generate([])).toEqual([
      "device:tap:Files",
      "device:type:hello world",
      "device:shortcut:Save",
    ]);
  });

  it("predicts the recorded next movement given a prefix", () => {
    const dataset = buildMovementDataset([editWorkflow("rec", "hello")]);
    const model = new MarkovMovementModelBackend(2).train(dataset);
    const prediction = model.predictNext(["device:tap:Files", "device:type:hello"]);
    expect(prediction?.token).toBe("device:shortcut:Save");
    expect(prediction?.order).toBe(2);
    expect(prediction?.probability).toBeGreaterThan(0);
  });

  it("stops generation at EOS instead of looping forever", () => {
    const dataset = buildMovementDataset([editWorkflow("rec", "x")]);
    const model = new MarkovMovementModelBackend(2).train(dataset);
    const out = model.generate([], { maxSteps: 100 });
    expect(out).not.toContain(MOVEMENT_EOS);
    expect(out.length).toBeLessThanOrEqual(3);
  });

  it("generalizes: a novel opening screen backs off to the shared tail", () => {
    // Both workflows share the "type body -> Save" tail but open different
    // screens. A never-seen opening screen makes the 2-gram context unseen, so
    // the model must back off to the shared 1-gram (type body -> Save).
    const dataset = buildMovementDataset([
      openThenEdit("a", "Files"),
      openThenEdit("b", "Recent"),
    ]);
    const model = new MarkovMovementModelBackend(2).train(dataset);
    const prediction = model.predictNext(["device:tap:Brand-New-Screen", "device:type:body"]);
    expect(prediction?.token).toBe("device:shortcut:Save");
    expect(prediction?.order).toBeLessThan(2); // matched a backed-off context
  });

  it("returns undefined when there is nothing to predict from", () => {
    const model = new MarkovMovementModelBackend(2).train({ version: 1, sequences: [] });
    expect(model.predictNext(["anything"])).toBeUndefined();
    expect(model.generate([])).toEqual([]);
    expect(model.vocabulary()).toEqual([]);
  });

  it("breaks ties deterministically (lexicographic) and is order-stable", () => {
    // Same context 'start' is followed once by 'z...' and once by 'a...'.
    const dataset = buildMovementDataset([
      span("t1", [action("k", "start", 1, { action: "start" }), action("k", "z", 2, { action: "z-branch" })]),
      span("t2", [action("k", "start", 1, { action: "start" }), action("k", "a", 2, { action: "a-branch" })]),
    ]);
    const backend = new MarkovMovementModelBackend(1);
    const p1 = backend.train(dataset).predictNext(["k:start"]);
    const p2 = backend.train(dataset).predictNext(["k:start"]);
    expect(p1?.token).toBe("k:a-branch"); // lexicographically smaller wins ties
    expect(p2).toEqual(p1);
  });

  it("rejects invalid model orders", () => {
    expect(() => new MarkovMovementModelBackend(0)).toThrow(/positive integer/);
    expect(() => new MarkovMovementModelBackend(1.5)).toThrow(/positive integer/);
  });
});

describe("serialization round-trip", () => {
  it("reloads to an equivalent model", () => {
    const dataset = buildMovementDataset([editWorkflow("a", "alpha"), editWorkflow("b", "beta")]);
    const trained = new MarkovMovementModelBackend(2).train(dataset);
    const reloaded = loadMovementModel(JSON.parse(JSON.stringify(trained.serialize())));
    expect(reloaded.vocabulary()).toEqual(trained.vocabulary());
    expect(reloaded.generate([])).toEqual(trained.generate([]));
    expect(reloaded.serialize()).toEqual(trained.serialize());
  });

  it("rejects unknown serialized model kinds", () => {
    expect(() => loadMovementModel({ kind: "bogus" } as never)).toThrow(/Unsupported/);
  });
});

describe("evaluateMovementModel (generalization harness)", () => {
  it("scores perfect accuracy replaying a held-in sequence", () => {
    const train = buildMovementDataset([editWorkflow("a", "alpha")]);
    const model = new MarkovMovementModelBackend(2).train(train);
    const heldOut: MovementSequence[] = train.sequences;
    const evalResult = evaluateMovementModel(model, heldOut);
    expect(evalResult.samples).toBeGreaterThan(0);
    expect(evalResult.coverage).toBe(1);
    expect(evalResult.topOneAccuracy).toBe(1);
  });

  it("still covers and partially predicts a related held-out sequence", () => {
    const train = buildMovementDataset([openThenEdit("a", "Files"), openThenEdit("b", "Recent")]);
    const model = new MarkovMovementModelBackend(2).train(train);
    // Held-out variant with a new opening screen but the same edit/save/close tail.
    const heldOut = buildMovementDataset([openThenEdit("c", "Unseen-Screen")]).sequences;
    const evalResult = evaluateMovementModel(model, heldOut);
    expect(evalResult.coverage).toBe(1); // back-off always finds *some* context
    expect(evalResult.topOneAccuracy).toBeGreaterThan(0.5); // shared tail generalizes
  });

  it("reports zeros for an empty held-out set", () => {
    const model = new MarkovMovementModelBackend(2).train({ version: 1, sequences: [] });
    expect(evaluateMovementModel(model, [])).toEqual({
      samples: 0,
      topOneAccuracy: 0,
      coverage: 0,
      meanOrder: 0,
    });
  });
});
