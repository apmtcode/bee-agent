import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { buildTrajectorySpan, type TrajectoryAction, type TrajectorySpan } from "../capture/trajectory.js";
import {
  buildMovementDataset,
  evaluateMovementModel,
  loadMovementModel,
  MarkovMovementBackend,
  MOVEMENT_END_TOKEN,
  MOVEMENT_START_TOKEN,
  MovementModelRegistry,
  saveMovementModel,
  tokenizeMovementAction,
  tokenizeTrajectory,
  type MovementModelArtifact,
} from "./movement-model.js";

function gesture(kind: string, target: string | undefined, ts: number, direction?: string): TrajectoryAction {
  return {
    kind: "action",
    tool: "device",
    summary: `${kind} ${target ?? direction ?? ""}`.trim(),
    ts,
    metadata: {
      gesture: kind,
      ...(target ? { target } : {}),
      ...(direction ? { direction } : {}),
    },
  };
}

/**
 * Synthetic event-stream generator: emit a trajectory whose actions follow a
 * fixed grammar so we can validate capture → dataset → train → replay round
 * trips without any real OS input.
 */
function syntheticTrajectory(id: string, steps: Array<[string, string]>, startTs = 1000): TrajectorySpan {
  return buildTrajectorySpan({
    id,
    sessionId: `sess-${id}`,
    actions: steps.map(([kind, target], index) => gesture(kind, target, startTs + index * 10)),
  });
}

describe("tokenizeMovementAction", () => {
  it("prefers structured gesture/target metadata", () => {
    expect(tokenizeMovementAction(gesture("tap", "Submit Button", 1))).toBe("device:tap:submit-button");
  });

  it("falls back to direction when no target is present", () => {
    expect(tokenizeMovementAction(gesture("swipe", undefined, 1, "up"))).toBe("device:swipe:up");
  });

  it("falls back to the tool name when there is no gesture metadata", () => {
    expect(
      tokenizeMovementAction({ kind: "action", tool: "browser", summary: "clicked", ts: 1 }),
    ).toBe("browser");
  });
});

describe("tokenizeTrajectory", () => {
  it("orders tokens chronologically", () => {
    const trajectory = buildTrajectorySpan({
      id: "t1",
      sessionId: "s1",
      actions: [gesture("type", "search", 30), gesture("tap", "field", 10), gesture("tap", "go", 20)],
    });
    expect(tokenizeTrajectory(trajectory).tokens).toEqual([
      "device:tap:field",
      "device:tap:go",
      "device:type:search",
    ]);
  });

  it("uses redacted actions when a review redacted them", () => {
    const trajectory: TrajectorySpan = {
      ...buildTrajectorySpan({ id: "t1", sessionId: "s1", actions: [gesture("tap", "secret", 10)] }),
      review: {
        status: "approved",
        reviewedAt: "2026-01-01T00:00:00.000Z",
        reviewedBy: "reviewer",
        redactedActions: [{ ts: 10, tool: "device", summary: "tapped redacted" }],
      },
    };
    // Redacted actions have no metadata, so they tokenize to the tool name.
    expect(tokenizeTrajectory(trajectory).tokens).toEqual(["device"]);
  });
});

describe("buildMovementDataset", () => {
  it("drops empty sequences and builds a sorted vocabulary", () => {
    const dataset = buildMovementDataset([
      syntheticTrajectory("a", [
        ["tap", "menu"],
        ["tap", "settings"],
      ]),
      buildTrajectorySpan({ id: "empty", sessionId: "s", actions: [] }),
    ]);
    expect(dataset.sequences).toHaveLength(1);
    expect(dataset.vocabulary).toEqual(["device:tap:menu", "device:tap:settings"]);
  });
});

describe("MarkovMovementBackend", () => {
  const backend = new MarkovMovementBackend();

  it("repeats a recorded movement exactly when seeded with its prefix", () => {
    const trajectory = syntheticTrajectory("open-settings", [
      ["tap", "menu"],
      ["tap", "settings"],
      ["swipe", "down"],
      ["tap", "wifi"],
    ]);
    const dataset = buildMovementDataset([trajectory]);
    const predictor = backend.createPredictor(backend.train(dataset));
    const expected = tokenizeTrajectory(trajectory).tokens;

    const replayed = predictor.generate([expected[0]], 16);
    expect(replayed).toEqual(expected);
  });

  it("generalizes to a new-but-related movement composed from observed transitions", () => {
    // Two trajectories share the `tap:menu -> tap:settings` transition but diverge
    // afterwards. A seed that starts like the second should follow its branch.
    const dataset = buildMovementDataset([
      syntheticTrajectory("a", [
        ["tap", "menu"],
        ["tap", "settings"],
        ["tap", "wifi"],
      ]),
      syntheticTrajectory("b", [
        ["tap", "menu"],
        ["tap", "settings"],
        ["tap", "bluetooth"],
        ["swipe", "down"],
      ]),
    ]);
    const artifact = backend.train(dataset, { order: 2 });
    const predictor = backend.createPredictor(artifact);

    // Unseen exact starting bigram context resolves via back-off + order-2 branch.
    const continued = predictor.generate(["device:tap:menu", "device:tap:settings", "device:tap:bluetooth"], 8);
    expect(continued).toEqual([
      "device:tap:menu",
      "device:tap:settings",
      "device:tap:bluetooth",
      "device:swipe:down",
    ]);
  });

  it("backs off to a shorter context when the full window was never seen", () => {
    const dataset = buildMovementDataset([
      syntheticTrajectory("a", [
        ["tap", "a"],
        ["tap", "b"],
        ["tap", "c"],
      ]),
    ]);
    const predictor = backend.createPredictor(backend.train(dataset, { order: 2 }));
    // Context ["<unseen>", "device:tap:b"] has no order-2 match, backs off to
    // the unigram context ["device:tap:b"] which was followed by tap:c.
    const prediction = predictor.predictNext(["device:tap:zzz", "device:tap:b"]);
    expect(prediction?.token).toBe("device:tap:c");
    expect(prediction?.contextLength).toBe(1);
  });

  it("is deterministic under tie-breaks", () => {
    const dataset = buildMovementDataset([
      syntheticTrajectory("a", [["tap", "root"], ["tap", "left"]]),
      syntheticTrajectory("b", [["tap", "root"], ["tap", "right"]]),
    ]);
    const predictor = backend.createPredictor(backend.train(dataset, { order: 1 }));
    const first = predictor.predictNext(["device:tap:root"]);
    const second = predictor.predictNext(["device:tap:root"]);
    // Equal probability (1 each) -> stable alphabetical tie-break picks "left".
    expect(first?.token).toBe("device:tap:left");
    expect(second).toEqual(first);
  });

  it("returns undefined for a wholly unknown context", () => {
    const predictor = backend.createPredictor(backend.train(buildMovementDataset([])));
    expect(predictor.predictNext(["nothing"])) .toBeUndefined();
  });

  it("stops generating at the end sentinel and never emits sentinels", () => {
    const trajectory = syntheticTrajectory("short", [["tap", "only"]]);
    const predictor = backend.createPredictor(backend.train(buildMovementDataset([trajectory])));
    const generated = predictor.generate(["device:tap:only"], 32);
    expect(generated).toEqual(["device:tap:only"]);
    expect(generated).not.toContain(MOVEMENT_START_TOKEN);
    expect(generated).not.toContain(MOVEMENT_END_TOKEN);
  });
});

describe("evaluateMovementModel", () => {
  const backend = new MarkovMovementBackend();

  it("reports perfect metrics replaying a single trained trajectory", () => {
    const trajectories = [
      syntheticTrajectory("a", [["tap", "menu"], ["tap", "settings"], ["tap", "wifi"]]),
    ];
    const dataset = buildMovementDataset(trajectories);
    const predictor = backend.createPredictor(backend.train(dataset, { order: 2 }));
    const evaluation = evaluateMovementModel(
      predictor,
      trajectories.map((trajectory) => tokenizeTrajectory(trajectory)),
    );
    expect(evaluation.evaluatedSequences).toBe(1);
    expect(evaluation.nextTokenAccuracy).toBe(1);
    expect(evaluation.exactReplayRate).toBe(1);
  });

  it("still replays each branch exactly when trained on branching trajectories", () => {
    // Shared `<start>` makes the first-token prediction ambiguous, so next-token
    // accuracy is below 1, but seeded free-running replay recovers each branch.
    const trajectories = [
      syntheticTrajectory("a", [["tap", "home"], ["tap", "settings"], ["tap", "wifi"]]),
      syntheticTrajectory("b", [["tap", "camera"], ["tap", "photos"], ["swipe", "up"]]),
    ];
    const dataset = buildMovementDataset(trajectories);
    const predictor = backend.createPredictor(backend.train(dataset, { order: 2 }));
    const evaluation = evaluateMovementModel(
      predictor,
      trajectories.map((trajectory) => tokenizeTrajectory(trajectory)),
    );
    expect(evaluation.evaluatedSequences).toBe(2);
    expect(evaluation.nextTokenAccuracy).toBeGreaterThan(0.5);
    expect(evaluation.exactReplayRate).toBe(1);
  });

  it("scores lower on held-out trajectories with unseen transitions", () => {
    const train = [syntheticTrajectory("a", [["tap", "menu"], ["tap", "settings"], ["tap", "wifi"]])];
    const heldOut = [syntheticTrajectory("z", [["tap", "menu"], ["tap", "camera"], ["tap", "shutter"]])];
    const predictor = backend.createPredictor(backend.train(buildMovementDataset(train), { order: 2 }));
    const evaluation = evaluateMovementModel(
      predictor,
      heldOut.map((trajectory) => tokenizeTrajectory(trajectory)),
    );
    expect(evaluation.nextTokenAccuracy).toBeLessThan(1);
    expect(evaluation.exactReplayRate).toBe(0);
  });
});

describe("MovementModelRegistry", () => {
  it("resolves a predictor for an artifact by kind", () => {
    const registry = new MovementModelRegistry();
    expect(registry.list()).toContain("markov");
    const dataset = buildMovementDataset([syntheticTrajectory("a", [["tap", "x"], ["tap", "y"]])]);
    const artifact = new MarkovMovementBackend().train(dataset);
    const predictor = registry.predictorFor(artifact);
    expect(predictor.predictNext(["device:tap:x"])?.token).toBe("device:tap:y");
  });

  it("throws for an artifact whose backend is not registered", () => {
    const registry = new MovementModelRegistry([]);
    const artifact: MovementModelArtifact = {
      version: 1,
      kind: "phantom",
      order: 1,
      vocabulary: [],
      sequenceCount: 0,
      tokenCount: 0,
      transitions: {},
    };
    expect(() => registry.predictorFor(artifact)).toThrow(/phantom/);
  });
});

describe("persistence", () => {
  let dir: string;

  beforeEach(async () => {
    dir = await mkdtemp(path.join(tmpdir(), "movement-model-"));
  });

  afterEach(async () => {
    await rm(dir, { recursive: true, force: true });
  });

  it("round-trips an artifact through disk with identical predictions", async () => {
    const backend = new MarkovMovementBackend();
    const dataset = buildMovementDataset([
      syntheticTrajectory("a", [["tap", "menu"], ["tap", "settings"], ["swipe", "down"]]),
    ]);
    const artifact = backend.train(dataset, { order: 2 });
    const file = path.join(dir, "model.json");
    await saveMovementModel(file, artifact);
    const loaded = await loadMovementModel(file);
    expect(loaded).toEqual(artifact);

    const before = backend.createPredictor(artifact).generate(["device:tap:menu"], 8);
    const after = backend.createPredictor(loaded!).generate(["device:tap:menu"], 8);
    expect(after).toEqual(before);
  });

  it("returns undefined when no model file exists", async () => {
    expect(await loadMovementModel(path.join(dir, "missing.json"))).toBeUndefined();
  });
});
