import { describe, expect, it } from "vitest";
import type { ReplayManifest } from "../capture/replay.js";
import type { TrajectorySpan } from "../capture/trajectory.js";
import {
  MarkovMovementPolicyBackend,
  buildMovementSamplesFromReplay,
  buildMovementSamplesFromTrajectories,
  createMovementPolicyBackend,
  evaluateMovementPolicy,
  movementActionToken,
  rolloutMovementPolicy,
  type MovementSample,
} from "./movement-policy.js";

/**
 * Deterministic synthetic movement-stream generator. Models a small mouse/
 * keyboard workflow: within each app the operator repeats a canonical action
 * chain (open -> focus -> submit). A seeded LCG picks screens so we can hold
 * out unseen-but-related screens to test generalization.
 */
function makeLcg(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state * 1664525 + 1013904223) >>> 0;
    return state / 0x100000000;
  };
}

function generateSyntheticSamples(params: {
  app: string;
  screens: string[];
  seed: number;
  chains: number;
}): MovementSample[] {
  const random = makeLcg(params.seed);
  const chain = [
    movementActionToken("device", "tapped open"),
    movementActionToken("device", "typed into field"),
    movementActionToken("device", "tapped submit"),
  ];
  const samples: MovementSample[] = [];
  for (let index = 0; index < params.chains; index += 1) {
    const screen = params.screens[Math.floor(random() * params.screens.length)]!;
    let previousAction: string | undefined;
    for (const action of chain) {
      samples.push({
        context: { app: params.app, screen, ...(previousAction ? { previousAction } : {}) },
        action,
      });
      previousAction = action;
    }
  }
  return samples;
}

describe("MarkovMovementPolicyBackend", () => {
  it("learns and repeats a recorded movement chain", () => {
    const backend = new MarkovMovementPolicyBackend();
    const samples = generateSyntheticSamples({ app: "mail", screens: ["inbox"], seed: 1, chains: 5 });
    const model = backend.train(samples);

    const first = backend.predict(model, { app: "mail", screen: "inbox" });
    expect(first?.action).toBe(movementActionToken("device", "tapped open"));
    expect(first?.backoffLevel).toBe("exact");
    expect(first?.confidence).toBe(1);

    const second = backend.predict(model, {
      app: "mail",
      screen: "inbox",
      previousAction: movementActionToken("device", "tapped open"),
    });
    expect(second?.action).toBe(movementActionToken("device", "typed into field"));
  });

  it("generalizes to a new but related screen via app-level backoff", () => {
    const backend = createMovementPolicyBackend("markov");
    // Train only on "inbox" and "compose"; evaluate on an unseen "archive" screen.
    const train = generateSyntheticSamples({ app: "mail", screens: ["inbox", "compose"], seed: 7, chains: 20 });
    const model = backend.train(train);

    const prediction = backend.predict(model, {
      app: "mail",
      screen: "archive",
      previousAction: movementActionToken("device", "tapped open"),
    });

    expect(prediction).toBeDefined();
    expect(prediction?.action).toBe(movementActionToken("device", "typed into field"));
    // Exact (app, "archive", prev) was never seen, so it must have backed off.
    expect(prediction?.backoffLevel).toBe("app");
  });

  it("produces deterministic tie-broken predictions", () => {
    const backend = new MarkovMovementPolicyBackend();
    const samples: MovementSample[] = [
      { context: { app: "a" }, action: "z:one" },
      { context: { app: "a" }, action: "a:two" },
    ];
    const model = backend.train(samples);
    const p1 = backend.predict(model, { app: "a" });
    const p2 = backend.predict(model, { app: "a" });
    expect(p1).toEqual(p2);
    // Tie broken lexicographically -> "a:two" wins over "z:one".
    expect(p1?.action).toBe("a:two");
    expect(p1?.confidence).toBe(0.5);
  });

  it("returns undefined when nothing was learned", () => {
    const backend = new MarkovMovementPolicyBackend();
    const model = backend.train([]);
    expect(backend.predict(model, { app: "unknown" })).toBeUndefined();
  });
});

describe("createMovementPolicyBackend", () => {
  it("throws for an unknown backend id", () => {
    expect(() => createMovementPolicyBackend("mlx-lora")).toThrow(/Unknown movement policy backend/);
  });
});

describe("buildMovementSamplesFromReplay", () => {
  it("threads observation context and previous action into each sample", () => {
    const manifest: ReplayManifest = {
      version: 1,
      sessionId: "session-1",
      trajectoryIds: ["traj-1"],
      eventCount: 4,
      events: [
        { kind: "observation", ts: 1, trajectoryId: "traj-1", source: "mail", summary: "inbox" },
        { kind: "action", ts: 2, trajectoryId: "traj-1", tool: "device", summary: "tapped open" },
        { kind: "observation", ts: 3, trajectoryId: "traj-1", source: "mail", summary: "compose" },
        { kind: "action", ts: 4, trajectoryId: "traj-1", tool: "device", summary: "tapped submit" },
      ],
    };
    const samples = buildMovementSamplesFromReplay(manifest);
    expect(samples).toEqual([
      { context: { app: "mail", screen: "inbox" }, action: "device:tapped open" },
      {
        context: { app: "mail", screen: "compose", previousAction: "device:tapped open" },
        action: "device:tapped submit",
      },
    ]);
  });
});

describe("buildMovementSamplesFromTrajectories", () => {
  it("prefers redacted review data and orders by timestamp", () => {
    const span: TrajectorySpan = {
      id: "traj-1",
      sessionId: "session-1",
      createdAt: "2026-07-13T00:00:00.000Z",
      captureTier: "operator",
      observations: [{ kind: "observation", source: "raw", summary: "raw", ts: 1 }],
      actions: [{ kind: "action", tool: "device", summary: "raw action", ts: 2 }],
      review: {
        status: "approved",
        reviewedAt: "2026-07-13T00:00:00.000Z",
        reviewedBy: "reviewer",
        redactedObservations: [{ ts: 1, source: "mail", summary: "inbox" }],
        redactedActions: [
          { ts: 2, tool: "device", summary: "tapped open" },
          { ts: 3, tool: "device", summary: "tapped submit" },
        ],
      },
    };
    const samples = buildMovementSamplesFromTrajectories([span]);
    expect(samples).toEqual([
      { context: { app: "mail", screen: "inbox" }, action: "device:tapped open" },
      {
        context: { app: "mail", screen: "inbox", previousAction: "device:tapped open" },
        action: "device:tapped submit",
      },
    ]);
  });
});

describe("rolloutMovementPolicy", () => {
  it("autonomously extends a learned chain from a start context", () => {
    const backend = new MarkovMovementPolicyBackend();
    const samples = generateSyntheticSamples({ app: "mail", screens: ["inbox"], seed: 3, chains: 8 });
    const model = backend.train(samples);

    const rollout = rolloutMovementPolicy(backend, model, { app: "mail", screen: "inbox" }, 3);
    expect(rollout.map((step) => step.prediction.action)).toEqual([
      "device:tapped open",
      "device:typed into field",
      "device:tapped submit",
    ]);
    // Each step feeds its prediction forward as the next previousAction.
    expect(rollout[1]?.context.previousAction).toBe("device:tapped open");
  });

  it("stops early when the model cannot predict", () => {
    const backend = new MarkovMovementPolicyBackend();
    const model = backend.train([]);
    expect(rolloutMovementPolicy(backend, model, { app: "mail" }, 5)).toEqual([]);
  });
});

describe("evaluateMovementPolicy", () => {
  it("scores accuracy and backoff usage on held-out related contexts", () => {
    const backend = new MarkovMovementPolicyBackend();
    const train = generateSyntheticSamples({ app: "mail", screens: ["inbox", "compose"], seed: 11, chains: 30 });
    const model = backend.train(train);

    // Held-out samples on an unseen screen -> must be answered via backoff.
    const heldOut = generateSyntheticSamples({ app: "mail", screens: ["settings"], seed: 99, chains: 10 });
    const evaluation = evaluateMovementPolicy(backend, model, heldOut);

    expect(evaluation.total).toBe(heldOut.length);
    expect(evaluation.accuracy).toBe(1);
    expect(evaluation.abstained).toBe(0);
    // Generalization: correct answers came from app/global backoff, not exact.
    expect(evaluation.backoffBreakdown.exact).toBe(0);
    expect(evaluation.backoffBreakdown.app + evaluation.backoffBreakdown.global).toBe(heldOut.length);
  });

  it("reports zero accuracy for an empty held-out set", () => {
    const backend = new MarkovMovementPolicyBackend();
    const model = backend.train([{ context: { app: "a" }, action: "x:y" }]);
    expect(evaluateMovementPolicy(backend, model, []).accuracy).toBe(0);
  });
});
