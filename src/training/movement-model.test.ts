import { describe, expect, it } from "vitest";
import {
  MOVEMENT_END_TOOL,
  MarkovMovementBackend,
  buildMovementDataset,
  evaluateGeneralization,
  movementActionKey,
  rolloutMovements,
  splitMovementDataset,
  type MovementSequence,
} from "./movement-model.js";
import {
  defaultMovementTemplates,
  synthesizeMovementTrajectories,
} from "./movement-synth.js";

describe("buildMovementDataset", () => {
  it("turns recorded trajectories into ordered context→action sequences", () => {
    const [trajectory] = synthesizeMovementTrajectories(
      [defaultMovementTemplates()[0]!],
      { perTemplate: 1, seed: 7 },
    );
    const dataset = buildMovementDataset([trajectory!]);
    expect(dataset.sequences).toHaveLength(1);
    const sequence = dataset.sequences[0]!;
    // Four recorded actions + one terminal "stop" example.
    expect(sequence.examples).toHaveLength(5);
    expect(sequence.examples[4]!.action.tool).toBe(MOVEMENT_END_TOOL);
    // First action has no predecessor; later actions carry the previous tool.
    expect(sequence.examples[0]!.context.lastTool).toBeUndefined();
    expect(sequence.examples[1]!.context.lastTool).toBe("device");
    // App context comes from the opening observation.
    expect(sequence.examples[0]!.context.app).toBe("mail");
    expect(sequence.outcome).toBe("success");
  });
});

describe("MarkovMovementBackend", () => {
  const backend = new MarkovMovementBackend();

  it("repeats a recorded movement chain from its start context", () => {
    const template = defaultMovementTemplates()[0]!;
    const trajectories = synthesizeMovementTrajectories([template], { perTemplate: 8, seed: 3 });
    const dataset = buildMovementDataset(trajectories);
    const model = backend.train(dataset);

    const rollout = rolloutMovements(backend, model, { app: template.app, platform: template.platform }, {
      maxSteps: 6,
    });

    // The greedy chain should reconstruct the recorded step order.
    const targets = rollout.map((step) => step.action.target);
    expect(targets).toEqual(["compose-button", "recipient-field", "body-field", "send-button"]);
    expect(rollout[0]!.backoffLevel).toBeGreaterThanOrEqual(0);
    expect(rollout[0]!.confidence).toBeGreaterThan(0);
  });

  it("predicts a plausible action for a related-but-unseen context via backoff", () => {
    const template = defaultMovementTemplates()[1]!;
    const trajectories = synthesizeMovementTrajectories([template], { perTemplate: 6, seed: 11 });
    const model = backend.train(buildMovementDataset(trajectories));

    // Unseen app but the same last-tool signal → backoff still fires.
    const prediction = backend.predict(model, { app: "unknown-app", lastTool: "device" });
    expect(prediction.action).toBeDefined();
    expect(prediction.backoffLevel).toBeGreaterThan(0);
  });

  it("abstains cleanly when it has no matching signal", () => {
    const model = backend.train({ version: 1, sequences: [] });
    const prediction = backend.predict(model, { app: "whatever" });
    expect(prediction.action).toBeUndefined();
    expect(prediction.backoffLevel).toBe(-1);
    expect(prediction.confidence).toBe(0);
  });

  it("round-trips through serialize/deserialize without changing predictions", () => {
    const trajectories = synthesizeMovementTrajectories(defaultMovementTemplates(), { perTemplate: 4, seed: 5 });
    const model = backend.train(buildMovementDataset(trajectories));
    const restored = backend.deserialize(backend.serialize(model));

    const context = { app: "mail", lastTool: "device", lastGesture: "tap" };
    expect(backend.predict(restored, context)).toEqual(backend.predict(model, context));
  });

  it("weights successful trajectories over failed ones", () => {
    const success: MovementSequence = {
      trajectoryId: "s",
      sessionId: "sess",
      outcome: "success",
      examples: [{ context: { app: "editor" }, action: { tool: "device", gesture: "tap", target: "save", summary: "tapped save" } }],
    };
    const failure: MovementSequence = {
      trajectoryId: "f",
      sessionId: "sess",
      outcome: "failure",
      examples: [{ context: { app: "editor" }, action: { tool: "device", gesture: "tap", target: "discard", summary: "tapped discard" } }],
    };
    const model = backend.train({ version: 1, sequences: [success, failure] });
    const prediction = backend.predict(model, { app: "editor" });
    expect(prediction.action?.target).toBe("save");
  });
});

describe("movementActionKey", () => {
  it("is stable and distinguishes differing tokens", () => {
    const a = { tool: "device", gesture: "tap", target: "x", summary: "tapped x" };
    const b = { tool: "device", gesture: "tap", target: "y", summary: "tapped y" };
    expect(movementActionKey(a)).toBe(movementActionKey({ ...a, summary: "different summary" }));
    expect(movementActionKey(a)).not.toBe(movementActionKey(b));
  });
});

describe("evaluateGeneralization", () => {
  const backend = new MarkovMovementBackend();

  it("scores replay fidelity on held-out related trajectories above chance", () => {
    const trajectories = synthesizeMovementTrajectories(defaultMovementTemplates(), {
      perTemplate: 12,
      seed: 42,
      variationRate: 0.35,
    });
    const dataset = buildMovementDataset(trajectories);
    const { train, heldOut } = splitMovementDataset(dataset, 4);
    expect(heldOut.length).toBeGreaterThan(0);

    const model = backend.train(train);
    const report = evaluateGeneralization(backend, model, heldOut);

    expect(report.total).toBeGreaterThan(0);
    // The model has never seen the held-out trajectories, but they are related,
    // so it should recover the tool nearly always and beat chance on exact match.
    expect(report.toolAccuracy).toBeGreaterThan(0.8);
    expect(report.exactAccuracy).toBeGreaterThan(0.4);
    expect(report.abstentionRate).toBeLessThan(0.2);
    expect(report.meanConfidence).toBeGreaterThan(0);
  });

  it("reports full abstention for an empty model", () => {
    const model = backend.train({ version: 1, sequences: [] });
    const heldOut: MovementSequence[] = [
      { trajectoryId: "t", sessionId: "s", examples: [{ context: { app: "a" }, action: { tool: "device", summary: "x" } }] },
    ];
    const report = evaluateGeneralization(backend, model, heldOut);
    expect(report.abstentionRate).toBe(1);
    expect(report.toolAccuracy).toBe(0);
  });
});

describe("synthesizeMovementTrajectories", () => {
  it("is deterministic for a fixed seed", () => {
    const a = synthesizeMovementTrajectories(defaultMovementTemplates(), { perTemplate: 3, seed: 99 });
    const b = synthesizeMovementTrajectories(defaultMovementTemplates(), { perTemplate: 3, seed: 99 });
    expect(a).toEqual(b);
  });

  it("varies with the seed", () => {
    const a = synthesizeMovementTrajectories(defaultMovementTemplates(), { perTemplate: 3, seed: 1, variationRate: 0.5 });
    const b = synthesizeMovementTrajectories(defaultMovementTemplates(), { perTemplate: 3, seed: 2, variationRate: 0.5 });
    expect(a).not.toEqual(b);
  });
});
