import { describe, expect, it } from "vitest";
import { buildTrajectorySpan, type TrajectoryAction, type TrajectorySpan } from "../capture/trajectory.js";
import {
  MOVEMENT_END,
  MarkovMovementBackend,
  buildMovementDataset,
  deriveTrajectoryContext,
  evaluateMovementPolicy,
  tokenizeAction,
  type MovementSequence,
} from "./movement-policy.js";

let clock = 1;

function action(tool: string, summary: string, metadata?: Record<string, unknown>): TrajectoryAction {
  return { kind: "action", tool, summary, ts: clock++, ...(metadata ? { metadata } : {}) };
}

/** Synthetic, deterministic gesture stream: an app the operator drives repeatedly. */
function syntheticTrajectory(id: string, appName: string, gestures: [string, string][]): TrajectorySpan {
  return buildTrajectorySpan({
    id,
    sessionId: `sess-${id}`,
    observations: [
      { kind: "observation", source: "device", summary: `${appName} active`, ts: clock++, metadata: { appName } },
    ],
    actions: gestures.map(([gesture, target]) => action("device", `${gesture} ${target}`, { gesture, target })),
  });
}

describe("movement dataset", () => {
  it("tokenizes gestures canonically and derives app context", () => {
    const trajectory = syntheticTrajectory("t", "notes", [
      ["tap", "compose"],
      ["type", "body"],
    ]);
    expect(deriveTrajectoryContext(trajectory)).toBe("notes");
    expect(tokenizeAction(trajectory.actions[0])).toBe("device:tap#compose");
    expect(tokenizeAction(trajectory.actions[1])).toBe("device:type#body");
  });

  it("builds ordered sequences and skips action-free trajectories", () => {
    const withActions = syntheticTrajectory("a", "notes", [["tap", "compose"]]);
    const empty = buildTrajectorySpan({ id: "b", sessionId: "sess-b" });
    const dataset = buildMovementDataset([withActions, empty], { order: 2 });
    expect(dataset.order).toBe(2);
    expect(dataset.sequences).toHaveLength(1);
    expect(dataset.sequences[0]).toMatchObject({ id: "a", context: "notes" });
  });
});

describe("MarkovMovementBackend", () => {
  const compose: [string, string][] = [
    ["tap", "compose"],
    ["type", "title"],
    ["type", "body"],
    ["tap", "send"],
  ];

  it("repeats a recorded movement sequence from its start", () => {
    const dataset = buildMovementDataset([syntheticTrajectory("t1", "notes", compose)], { order: 2 });
    const model = new MarkovMovementBackend().train(dataset);

    const replayed = model.rollout({ context: "notes", maxSteps: 10 });
    expect(replayed).toEqual(dataset.sequences[0].tokens);

    // Terminal state is learned: the last token predicts END.
    const end = model.predict({ context: "notes", history: dataset.sequences[0].tokens });
    expect(end?.token).toBe(MOVEMENT_END);
  });

  it("continues a primed partial movement", () => {
    const dataset = buildMovementDataset([syntheticTrajectory("t1", "notes", compose)], { order: 2 });
    const model = new MarkovMovementBackend().train(dataset);
    const rest = model.rollout({ context: "notes", prime: ["device:tap#compose"], maxSteps: 10 });
    expect(rest).toEqual(["device:type#title", "device:type#body", "device:tap#send"]);
  });

  it("generalizes to a new-but-related context via cross-context backoff", () => {
    // Train on two apps that share the same "compose → send" movement grammar.
    const dataset = buildMovementDataset(
      [
        syntheticTrajectory("mail", "mail", compose),
        syntheticTrajectory("notes", "notes", compose),
      ],
      { order: 2 },
    );
    const model = new MarkovMovementBackend().train(dataset);

    // "chat" was never recorded, but the pooled grammar still drives the movement.
    const prediction = model.predict({ context: "chat", history: ["device:tap#compose"] });
    expect(prediction?.backoff).toBe("global");
    expect(prediction?.token).toBe("device:type#title");

    const rollout = model.rollout({ context: "chat", prime: ["device:tap#compose"], maxSteps: 10 });
    expect(rollout).toEqual(["device:type#title", "device:type#body", "device:tap#send"]);
  });

  it("survives a serialize/load round-trip with identical inference", () => {
    const dataset = buildMovementDataset([syntheticTrajectory("t1", "notes", compose)], { order: 2 });
    const backend = new MarkovMovementBackend();
    const model = backend.train(dataset);
    const restored = backend.load(JSON.parse(JSON.stringify(model.serialize())));

    expect(restored.order).toBe(model.order);
    expect(restored.rollout({ context: "notes", maxSteps: 10 })).toEqual(
      model.rollout({ context: "notes", maxSteps: 10 }),
    );
  });

  it("returns undefined when nothing was ever trained", () => {
    const model = new MarkovMovementBackend().train({ version: 1, order: 2, sequences: [] });
    expect(model.predict({ context: "anything", history: [] })).toBeUndefined();
    expect(model.rollout({ context: "anything", maxSteps: 5 })).toEqual([]);
  });
});

describe("evaluateMovementPolicy", () => {
  const compose: [string, string][] = [
    ["tap", "compose"],
    ["type", "title"],
    ["tap", "send"],
  ];

  it("scores next-token fidelity on held-out sequences and reports generalization", () => {
    const trainingDataset = buildMovementDataset(
      [syntheticTrajectory("mail", "mail", compose), syntheticTrajectory("notes", "notes", compose)],
      { order: 2 },
    );
    const model = new MarkovMovementBackend().train(trainingDataset);

    // Held-out sequence in an unseen context exercises cross-context generalization.
    const heldOut: MovementSequence[] = [
      { id: "held", context: "chat", tokens: ["device:tap#compose", "device:type#title", "device:tap#send"] },
    ];
    const evaluation = evaluateMovementPolicy(model, heldOut);

    // 3 tokens + END = 4 predictions, all recoverable from the shared grammar.
    expect(evaluation.totalPredictions).toBe(4);
    expect(evaluation.accuracy).toBe(1);
    expect(evaluation.generalizedPredictions).toBe(4);
    expect(evaluation.perContext.chat.accuracy).toBe(1);
  });
});
