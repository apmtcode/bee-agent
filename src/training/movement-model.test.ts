import { describe, expect, it } from "vitest";
import { buildReplayManifest } from "../capture/replay.js";
import { buildTrajectorySpan, type TrajectoryAction } from "../capture/trajectory.js";
import {
  MOVEMENT_END_TOKEN,
  MovementModelBackendRegistry,
  NgramMovementBackend,
  buildMovementDatasetFromReplays,
  buildMovementDatasetFromTrajectories,
  evaluateMovementModel,
  movementTokenFromAction,
  trainMovementModel,
  type MovementModelBackend,
  type MovementSequence,
} from "./movement-model.js";

function action(tool: string, summary: string, ts: number): TrajectoryAction {
  return { kind: "action", tool, summary, ts };
}

function trajectory(id: string, actions: TrajectoryAction[]) {
  return buildTrajectorySpan({ id, sessionId: `session-${id}`, actions });
}

describe("movement dataset construction", () => {
  it("canonicalizes action tokens deterministically", () => {
    expect(movementTokenFromAction("Device", "  Tapped   Submit ")).toBe("device::tapped submit");
    expect(movementTokenFromAction("device", "tapped submit")).toBe(
      movementTokenFromAction("Device", "Tapped Submit"),
    );
  });

  it("builds ordered token sequences from trajectories with an end marker", () => {
    const dataset = buildMovementDatasetFromTrajectories([
      trajectory("t1", [
        action("device", "tapped compose", 2),
        action("device", "typed subject", 3),
        action("device", "tapped send", 1),
      ]),
    ]);

    // Sorted by ts, terminated with the end token.
    expect(dataset.sequences[0]!.tokens).toEqual([
      "device::tapped send",
      "device::tapped compose",
      "device::typed subject",
      MOVEMENT_END_TOKEN,
    ]);
    expect(dataset.vocabulary).not.toContain(MOVEMENT_END_TOKEN);
    expect(dataset.vocabulary).toContain("device::tapped send");
  });

  it("extracts only action events from replay manifests", () => {
    const replay = buildReplayManifest({
      sessionId: "s1",
      transcript: [],
      trajectories: [
        {
          ...trajectory("t1", [action("device", "tapped open", 5)]),
          observations: [{ kind: "observation", source: "device", summary: "mail active", ts: 4 }],
        },
      ],
    });
    const dataset = buildMovementDatasetFromReplays([replay]);
    expect(dataset.sequences[0]!.tokens).toEqual(["device::tapped open", MOVEMENT_END_TOKEN]);
  });

  it("drops empty sequences", () => {
    const dataset = buildMovementDatasetFromTrajectories([trajectory("empty", [])]);
    expect(dataset.sequences).toHaveLength(0);
    expect(dataset.vocabulary).toHaveLength(0);
  });
});

describe("n-gram movement model — recorded-movement repetition (objective 2c)", () => {
  it("reproduces a recorded movement exactly from its opening action", () => {
    const recorded = trajectory("t1", [
      action("device", "tap app icon", 1),
      action("device", "tap compose", 2),
      action("device", "type recipient", 3),
      action("device", "tap send", 4),
    ]);
    const dataset = buildMovementDatasetFromTrajectories([recorded]);
    const model = trainMovementModel(dataset, { order: 3 });

    const rollout = model.generate(["device::tap app icon"]);
    expect(rollout).toEqual([
      "device::tap compose",
      "device::type recipient",
      "device::tap send",
    ]);
  });

  it("prefers the most frequently recorded continuation with deterministic tie-breaks", () => {
    // Two recordings share the same first move but diverge; the more frequent
    // continuation must win, deterministically.
    const dataset = buildMovementDatasetFromTrajectories([
      trajectory("a", [action("device", "open", 1), action("device", "save", 2)]),
      trajectory("b", [action("device", "open", 1), action("device", "save", 2)]),
      trajectory("c", [action("device", "open", 1), action("device", "discard", 2)]),
    ]);
    const model = trainMovementModel(dataset, { order: 2 });
    const prediction = model.predictNext(["device::open"]);
    expect(prediction?.token).toBe("device::save");
    expect(prediction?.confidence).toBeCloseTo(2 / 3);
    expect(prediction?.generalized).toBe(false);
  });
});

describe("n-gram movement model — generalization to new-but-related movements (objective 2d)", () => {
  it("predicts a next action for an unseen context via suffix backoff", () => {
    // Train: two flows that both end "... -> paste -> save".
    const dataset = buildMovementDatasetFromTrajectories([
      trajectory("a", [action("app", "copy", 1), action("app", "paste", 2), action("app", "save", 3)]),
      trajectory("b", [action("app", "select all", 1), action("app", "paste", 2), action("app", "save", 3)]),
    ]);
    const model = trainMovementModel(dataset, { order: 3 });

    // A *new* related context the model never saw verbatim: begins with an
    // unseen action, then "paste". It should still generalize "-> save".
    const prediction = model.predictNext(["app::cut", "app::paste"]);
    expect(prediction?.token).toBe("app::save");
    expect(prediction?.generalized).toBe(true);
    expect(prediction?.matchedOrder).toBeLessThan(2);
  });

  it("scores generalization on held-out related sequences", () => {
    const train: MovementSequence[] = [
      { id: "a", tokens: ["x", "y", "z", MOVEMENT_END_TOKEN] },
      { id: "b", tokens: ["w", "y", "z", MOVEMENT_END_TOKEN] },
    ];
    const backend = new NgramMovementBackend();
    const model = backend.train({ version: 1, sequences: train, vocabulary: ["w", "x", "y", "z"] }, { order: 2 });

    // Held-out sequence shares the "y -> z" structure but with a novel prefix.
    const evaluation = evaluateMovementModel(model, [{ id: "held", tokens: ["v", "y", "z", MOVEMENT_END_TOKEN] }]);
    expect(evaluation.samples).toBe(4);
    // "y"->"z" and "z"->"<end>" should be recovered by backoff.
    expect(evaluation.correct).toBeGreaterThanOrEqual(2);
    expect(evaluation.accuracy).toBeGreaterThan(0);
    expect(evaluation.generalizationRate).toBeGreaterThan(0);
  });
});

describe("model serialization + pluggable backend registry", () => {
  it("round-trips a trained model through serialize/restore with identical predictions", () => {
    const dataset = buildMovementDatasetFromTrajectories([
      trajectory("a", [action("device", "open", 1), action("device", "scroll", 2), action("device", "tap", 3)]),
    ]);
    const model = trainMovementModel(dataset, { order: 3 });
    const registry = new MovementModelBackendRegistry();
    const restored = registry.restore(model.serialize());

    expect(restored.backendId).toBe(model.backendId);
    expect(restored.order).toBe(model.order);
    expect(restored.generate(["device::open"])).toEqual(model.generate(["device::open"]));
  });

  it("lists registered backends and rejects unknown ids", () => {
    const registry = new MovementModelBackendRegistry();
    expect(registry.list()).toContain("ngram-local");
    expect(registry.has("ngram-local")).toBe(true);
    expect(() => registry.get("does-not-exist")).toThrow(/Unknown movement-model backend/);
  });

  it("accepts a custom pluggable backend under the same interface", () => {
    const stub: MovementModelBackend = {
      id: "stub-backend",
      train: () => ({
        backendId: "stub-backend",
        order: 1,
        predictNext: () => ({ token: "fixed", confidence: 1, matchedOrder: 0, generalized: true }),
        generate: () => ["fixed"],
        serialize: () => ({ backendId: "stub-backend", order: 1, vocabulary: [], transitions: {} }),
      }),
      restore: () => {
        throw new Error("not needed");
      },
    };
    const registry = new MovementModelBackendRegistry([new NgramMovementBackend(), stub]);
    expect(registry.list()).toEqual(["ngram-local", "stub-backend"]);
    const model = registry.get("stub-backend").train({ version: 1, sequences: [], vocabulary: [] });
    expect(model.generate([])).toEqual(["fixed"]);
  });
});

describe("synthetic event-stream round-trip", () => {
  it("captures → dataset → train → replays the whole recorded stream", () => {
    // Simulate a recorded local-movement stream (no real OS needed).
    const steps = ["focus window", "move cursor", "click button", "type text", "press enter"];
    const actions = steps.map((summary, index) => action("os", summary, index + 1));
    const dataset = buildMovementDatasetFromTrajectories([trajectory("flow", actions)]);
    const model = trainMovementModel(dataset, { order: 4 });

    const rollout = model.generate([movementTokenFromAction("os", steps[0]!)]);
    expect(rollout).toEqual(steps.slice(1).map((summary) => movementTokenFromAction("os", summary)));
  });
});
