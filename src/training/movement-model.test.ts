import { describe, expect, it } from "vitest";
import type { ReplayManifest } from "../capture/replay.js";
import { buildTrajectorySpan, type TrajectorySpan } from "../capture/trajectory.js";
import {
  DEFAULT_MOVEMENT_BACKEND,
  MarkovBackoffMovementBackend,
  MovementBackendRegistry,
  buildMovementDataset,
  buildMovementDatasetFromReplays,
  evaluateMovementModel,
  loadMovementModel,
  movementBackendRegistry,
  tokensFromTrajectory,
  type MovementModelBackend,
} from "./movement-model.js";

/** A synthetic "open mail → tap compose → type subject → tap send" flow. */
function mailFlow(id: string, target: string, base = 1_000): TrajectorySpan {
  return buildTrajectorySpan({
    id,
    sessionId: `session-${id}`,
    captureTier: "app",
    observations: [
      { kind: "observation", source: "device", summary: `mail on ${target} inbox`, ts: base },
      { kind: "observation", source: "device", summary: `mail on ${target} compose`, ts: base + 20 },
    ],
    actions: [
      { kind: "action", tool: "device", summary: "tapped compose", ts: base + 10 },
      { kind: "action", tool: "device", summary: "typed subject", ts: base + 30 },
      { kind: "action", tool: "device", summary: "tapped send", ts: base + 40 },
    ],
  });
}

describe("buildMovementDataset", () => {
  it("emits one step per action with the preceding tokens as context", () => {
    const dataset = buildMovementDataset([mailFlow("t1", "acme")]);
    expect(dataset.steps).toHaveLength(3);
    // The first action ("tapped compose") is preceded by one inbox observation.
    const first = dataset.steps[0];
    expect(first.action.label).toBe("tapped compose");
    expect(first.context.map((token) => token.label)).toEqual(["mail on acme inbox"]);
  });

  it("orders interleaved observations and actions by timestamp", () => {
    const tokens = tokensFromTrajectory(mailFlow("t1", "acme"));
    expect(tokens.map((token) => token.label)).toEqual([
      "mail on acme inbox",
      "tapped compose",
      "mail on acme compose",
      "typed subject",
      "tapped send",
    ]);
  });
});

describe("MarkovBackoffMovementBackend — replay recorded movements", () => {
  it("reproduces the recorded action sequence for a trained flow", () => {
    const backend = new MarkovBackoffMovementBackend();
    const model = backend.train(buildMovementDataset([mailFlow("t1", "acme")]));

    // Replay: walking the recorded contexts should reproduce recorded actions.
    for (const step of buildMovementDataset([mailFlow("t1", "acme")]).steps) {
      const prediction = model.predict({ history: step.context });
      expect(prediction).toBeDefined();
      expect(prediction?.source).toBe("exact");
      expect(prediction?.tool).toBe(step.action.channel);
      expect(prediction?.summary).toBe(step.action.label);
      expect(prediction?.confidence).toBeGreaterThan(0);
    }
  });

  it("learns the modal action when several trajectories share a context", () => {
    const backend = new MarkovBackoffMovementBackend();
    const model = backend.train(
      buildMovementDataset([mailFlow("t1", "acme"), mailFlow("t2", "acme"), mailFlow("t3", "acme")]),
    );
    const prediction = model.predict({
      history: tokensFromTrajectory(mailFlow("q", "acme")).slice(0, 1),
    });
    expect(prediction?.summary).toBe("tapped compose");
    expect(prediction?.confidence).toBe(1);
  });
});

describe("MarkovBackoffMovementBackend — generalize to related movements", () => {
  it("backs off to the prototype for an unseen-but-related context", () => {
    const backend = new MarkovBackoffMovementBackend();
    // Train on "acme"; query a brand-new "globex" whose exact context labels
    // were never seen, so exact recall misses and backoff must generalize.
    const model = backend.train(buildMovementDataset([mailFlow("t1", "acme"), mailFlow("t2", "acme")]));

    const relatedContext = tokensFromTrajectory(mailFlow("new", "globex")).slice(0, 1);
    const prediction = model.predict({ history: relatedContext });

    expect(prediction).toBeDefined();
    expect(prediction?.source).toBe("backoff");
    // The leading observation prototype ("mail") most often preceded a "tapped"
    // action, so generalization still yields a plausible movement.
    expect(prediction?.prototype).toBe("tapped");
  });

  it("falls back to the global prior when even the prototype is unseen", () => {
    const backend = new MarkovBackoffMovementBackend();
    const model = backend.train(buildMovementDataset([mailFlow("t1", "acme")]));
    const prediction = model.predict({
      history: [{ role: "observation", channel: "browser", label: "unheard of", prototype: "quux" }],
    });
    expect(prediction).toBeDefined();
    expect(prediction?.source).toBe("prior");
  });

  it("returns undefined for a model trained on an empty dataset", () => {
    const model = new MarkovBackoffMovementBackend().train({ steps: [] });
    expect(model.predict({ history: [] })).toBeUndefined();
  });
});

describe("evaluateMovementModel — generalization eval harness", () => {
  it("scores exact replay fidelity at 1.0 on the training trajectory", () => {
    const train = [mailFlow("t1", "acme"), mailFlow("t2", "acme")];
    const model = new MarkovBackoffMovementBackend().train(buildMovementDataset(train));
    const result = evaluateMovementModel(model, [mailFlow("t3", "acme")]);
    expect(result.total).toBe(3);
    expect(result.exactMatchRate).toBe(1);
    expect(result.generalizedSteps).toBe(0);
  });

  it("reports prototype-level generalization on held-out related trajectories", () => {
    const train = [mailFlow("t1", "acme"), mailFlow("t2", "acme")];
    const model = new MarkovBackoffMovementBackend().train(buildMovementDataset(train));
    // Held-out "globex" flow: the two observation-preceded steps carry
    // target-specific context labels never seen in training, so they require
    // prototype backoff; the action-preceded step ("typed subject") is
    // target-independent and still recalled exactly. Generalization fidelity is
    // never worse than exact recall.
    const result = evaluateMovementModel(model, [mailFlow("h1", "globex")]);
    expect(result.total).toBe(3);
    expect(result.generalizedSteps).toBe(2);
    expect(result.prototypeMatchRate).toBeGreaterThanOrEqual(result.exactMatchRate);
    expect(result.prototypeMatchRate).toBeGreaterThanOrEqual(2 / 3);
  });
});

describe("dataset from replay manifests", () => {
  it("builds equivalent steps from a replay timeline", () => {
    const replay: ReplayManifest = {
      version: 1,
      sessionId: "s1",
      trajectoryIds: ["t1"],
      eventCount: 3,
      events: [
        { kind: "observation", ts: 1, trajectoryId: "t1", source: "device", summary: "mail on acme inbox" },
        { kind: "action", ts: 2, trajectoryId: "t1", tool: "device", summary: "tapped compose" },
        { kind: "transcript", ts: 3, messageId: "m1", role: "assistant", content: "noise" },
      ],
    };
    const dataset = buildMovementDatasetFromReplays([replay]);
    expect(dataset.steps).toHaveLength(1);
    expect(dataset.steps[0].action.label).toBe("tapped compose");
    // Transcript events are context noise and must not become steps.
    expect(dataset.steps[0].context.map((token) => token.label)).toEqual(["mail on acme inbox"]);
  });
});

describe("serialization + registry", () => {
  it("round-trips a policy through serialize/loadMovementModel", () => {
    const model = new MarkovBackoffMovementBackend().train(buildMovementDataset([mailFlow("t1", "acme")]));
    const restored = loadMovementModel(model.serialize());
    const context = tokensFromTrajectory(mailFlow("q", "acme")).slice(0, 1);
    expect(restored.predict({ history: context })).toEqual(model.predict({ history: context }));
  });

  it("exposes the mock backend by default and supports pluggable registration", () => {
    expect(movementBackendRegistry.get(DEFAULT_MOVEMENT_BACKEND)?.name).toBe("markov-backoff");

    const registry = new MovementBackendRegistry().register(new MarkovBackoffMovementBackend());
    const fake: MovementModelBackend = {
      name: "mlx",
      train: (dataset) => new MarkovBackoffMovementBackend().train(dataset),
    };
    registry.register(fake);
    expect(registry.list()).toEqual(["markov-backoff", "mlx"]);
    expect(registry.require("mlx").name).toBe("mlx");
    expect(() => registry.require("nope")).toThrow(/unknown movement backend/);
  });
});
