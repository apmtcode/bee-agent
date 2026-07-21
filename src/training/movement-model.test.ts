import { describe, expect, it } from "vitest";
import { buildTrajectorySpan, type TrajectoryAction } from "../capture/trajectory.js";
import {
  NGramMovementBackend,
  abstractTokenKey,
  buildMovementDataset,
  evaluateMovementModel,
  extractMovementTokens,
  generateMovementSequence,
  movementTokenKey,
  splitMovementDataset,
  trainMovementModel,
  type MovementSequence,
  type MovementToken,
} from "./movement-model.js";

function action(tool: string, gesture: string, target: string | undefined, ts: number): TrajectoryAction {
  return {
    kind: "action",
    tool,
    summary: `${gesture} ${target ?? ""}`.trim(),
    ts,
    metadata: { gesture, ...(target ? { target } : {}) },
  };
}

function span(id: string, actions: TrajectoryAction[]) {
  return buildTrajectorySpan({ id, sessionId: `s-${id}`, actions });
}

/** A repeatable "open menu -> pick item -> confirm" flow, per target. */
function menuFlow(id: string, item: string): TrajectoryAction[] {
  return [
    action("device", "tap", "menu-button", 1),
    action("device", "tap", item, 2),
    action("device", "tap", "confirm", 3),
  ].map((a, index) => ({ ...a, summary: `${a.summary} ${id}`, ts: index + 1 }));
}

describe("movement token extraction", () => {
  it("derives structured tokens from trajectory actions", () => {
    const tokens = extractMovementTokens(span("t1", menuFlow("t1", "settings")));
    expect(tokens).toEqual<MovementToken[]>([
      { tool: "device", action: "tap", target: "menu-button" },
      { tool: "device", action: "tap", target: "settings" },
      { tool: "device", action: "tap", target: "confirm" },
    ]);
  });

  it("falls back to the tool verb when no gesture metadata is present", () => {
    const tokens = extractMovementTokens(
      span("t2", [{ kind: "action", tool: "browser", summary: "navigate", ts: 1 }]),
    );
    expect(tokens[0]).toEqual({ tool: "browser", action: "browser" });
  });

  it("builds a dataset and skips empty trajectories", () => {
    const dataset = buildMovementDataset([
      span("t1", menuFlow("t1", "settings")),
      span("empty", []),
    ]);
    expect(dataset.sequences).toHaveLength(1);
    expect(dataset.sequences[0]!.trajectoryId).toBe("t1");
  });
});

describe("NGramMovementBackend — repeat recorded movements", () => {
  it("deterministically repeats a seen sequence from its seed", async () => {
    const dataset = buildMovementDataset([span("t1", menuFlow("t1", "settings"))]);
    const model = await trainMovementModel(dataset, { order: 3 });
    const backend = new NGramMovementBackend();

    const seed: MovementToken[] = [{ tool: "device", action: "tap", target: "menu-button" }];
    const first = backend.predict(model, seed);
    expect(first.token).toEqual({ tool: "device", action: "tap", target: "settings" });
    expect(first.layer).toBe("exact");
    expect(first.confidence).toBe(1);

    const generated = generateMovementSequence(model, seed, 5, { backend });
    expect(generated).toEqual([
      { tool: "device", action: "tap", target: "settings" },
      { tool: "device", action: "tap", target: "confirm" },
    ]);
  });

  it("is stable across two independent trainings (deterministic)", async () => {
    const dataset = buildMovementDataset([
      span("a", menuFlow("a", "settings")),
      span("b", menuFlow("b", "profile")),
    ]);
    const m1 = await trainMovementModel(dataset);
    const m2 = await trainMovementModel(dataset);
    expect(m2).toEqual(m1);
  });

  it("reports unigram fallback when the context is unknown", async () => {
    const dataset = buildMovementDataset([span("t1", menuFlow("t1", "settings"))]);
    const model = await trainMovementModel(dataset);
    const prediction = new NGramMovementBackend().predict(model, [
      { tool: "browser", action: "swipe", target: "nowhere" },
    ]);
    expect(prediction.layer).toBe("unigram");
    expect(prediction.token).toBeDefined();
  });

  it("returns layer 'none' for an untrained model", () => {
    const empty = { ...emptyModel() };
    const prediction = new NGramMovementBackend().predict(empty, [
      { tool: "device", action: "tap", target: "x" },
    ]);
    expect(prediction.layer).toBe("none");
    expect(prediction.token).toBeUndefined();
  });
});

describe("NGramMovementBackend — generalize to related movements", () => {
  it("predicts the next movement *kind* for an unseen target via the abstract layer", async () => {
    // Train on menu flows over several items; then seed a *new* item.
    const dataset = buildMovementDataset([
      span("a", menuFlow("a", "settings")),
      span("b", menuFlow("b", "profile")),
      span("c", menuFlow("c", "billing")),
    ]);
    const model = await trainMovementModel(dataset, { order: 2 });
    const backend = new NGramMovementBackend();

    // Unseen full context (new item "notifications") but familiar structure:
    // after tapping menu-button then a fresh item, the flow always confirms.
    const context: MovementToken[] = [
      { tool: "device", action: "tap", target: "menu-button" },
      { tool: "device", action: "tap", target: "notifications" },
    ];
    const prediction = backend.predict(model, context);
    expect(prediction.token?.action).toBe("tap");
    // It confirms next — generalized from the recurring structure.
    expect(prediction.token?.target).toBe("confirm");
    expect(["exact", "backoff", "abstract"]).toContain(prediction.layer);
  });
});

describe("evaluateMovementModel — generalization harness", () => {
  it("scores next-movement accuracy on held-out related sequences", async () => {
    const dataset = buildMovementDataset(
      ["settings", "profile", "billing", "notifications", "privacy", "about"].map((item, index) =>
        span(`t${index}`, menuFlow(`t${index}`, item)),
      ),
    );
    const { train, heldOut } = splitMovementDataset(dataset, 3);
    expect(train.sequences.length).toBeGreaterThan(0);
    expect(heldOut.length).toBeGreaterThan(0);

    const model = await trainMovementModel(train, { order: 2 });
    const result = evaluateMovementModel(model, heldOut);

    expect(result.predictions).toBeGreaterThan(0);
    // The recurring confirm-step generalizes, so action-level accuracy is high.
    expect(result.actionAccuracy).toBeGreaterThanOrEqual(result.accuracy);
    expect(result.actionAccuracy).toBeGreaterThan(0.5);
    const layerTotal = Object.values(result.layerCounts).reduce((sum, n) => sum + n, 0);
    expect(layerTotal).toBe(result.predictions);
  });

  it("returns zeroed metrics for empty held-out input", async () => {
    const model = await trainMovementModel(buildMovementDataset([span("t", menuFlow("t", "x"))]));
    const result = evaluateMovementModel(model, [] as MovementSequence[]);
    expect(result).toMatchObject({ predictions: 0, correct: 0, accuracy: 0, actionAccuracy: 0 });
  });
});

describe("token keys", () => {
  it("distinguishes full vs abstract views", () => {
    const token: MovementToken = { tool: "device", action: "tap", target: "confirm" };
    expect(movementTokenKey(token)).not.toBe(abstractTokenKey(token));
    expect(abstractTokenKey(token)).toBe(abstractTokenKey({ tool: "device", action: "tap", target: "other" }));
  });
});

function emptyModel() {
  return {
    version: 1 as const,
    backend: "ngram-markov",
    order: 3,
    transitions: {},
    abstractTransitions: {},
    unigram: {},
    tokenByKey: {},
    trainedSequences: 0,
    trainedTokens: 0,
  };
}
