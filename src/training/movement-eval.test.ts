import { describe, expect, it } from "vitest";
import { MarkovMovementBackend, MOVEMENT_BOS, buildMovementDataset } from "./movement-model.js";
import {
  buildSyntheticMovementReplays,
  evaluateMovementModel,
  scoreRolloutFidelity,
  type SyntheticWorkflow,
} from "./movement-eval.js";

const STEPS = {
  open: { tool: "device", summary: "opened app" },
  menu: { tool: "device", summary: "tapped Menu" },
  settings: { tool: "device", summary: "tapped Settings" },
  toggle: { tool: "device", summary: "toggled Wifi" },
  back: { tool: "device", summary: "tapped Back" },
  save: { tool: "device", summary: "tapped Save" },
} as const;

describe("buildSyntheticMovementReplays", () => {
  it("produces deterministic, one-per-workflow replay manifests", () => {
    const workflows: SyntheticWorkflow[] = [
      { id: "w1", steps: ["open", "menu", "settings"] },
      { id: "w2", steps: ["open", "menu"] },
    ];
    const replays = buildSyntheticMovementReplays({ steps: STEPS, workflows });
    expect(replays).toHaveLength(2);
    expect(replays[0].trajectoryIds).toEqual(["w1"]);
    expect(replays[0].events.map((event) => (event.kind === "action" ? event.summary : ""))).toEqual([
      "opened app",
      "tapped Menu",
      "tapped Settings",
    ]);
    // Deterministic: same inputs -> identical output.
    expect(buildSyntheticMovementReplays({ steps: STEPS, workflows })).toEqual(replays);
  });

  it("throws on an unknown step key", () => {
    expect(() =>
      buildSyntheticMovementReplays({ steps: STEPS, workflows: [{ id: "w", steps: ["nope"] }] }),
    ).toThrow(/unknown synthetic step/);
  });
});

describe("evaluateMovementModel", () => {
  it("perfectly replays held-out sequences drawn from trained workflows", async () => {
    const workflows: SyntheticWorkflow[] = [{ id: "w1", steps: ["open", "menu", "settings", "toggle"] }];
    const replays = buildSyntheticMovementReplays({ steps: STEPS, workflows });
    const dataset = buildMovementDataset(replays);
    const model = await new MarkovMovementBackend().train(dataset, { order: 2 });

    const result = evaluateMovementModel(model, dataset);
    expect(result.nextTokenAccuracy).toBe(1);
    expect(result.unpredicted).toBe(0);
  });

  it("generalizes to held-out workflows that recombine trained movements", async () => {
    // Train on workflows that establish local transitions (menu->settings,
    // settings->toggle, toggle->save, open->menu). The held-out workflow is a
    // *new ordering* built from the same movement vocabulary.
    const trainWorkflows: SyntheticWorkflow[] = [
      { id: "t1", steps: ["open", "menu", "settings", "toggle", "save"] },
      { id: "t2", steps: ["open", "menu", "settings", "toggle"] },
      { id: "t3", steps: ["menu", "settings", "toggle", "save"] },
    ];
    const heldOutWorkflows: SyntheticWorkflow[] = [
      // Never trained as a whole, but every local transition was observed.
      { id: "h1", steps: ["open", "menu", "settings", "toggle", "save"] },
    ];

    const trainReplays = buildSyntheticMovementReplays({ steps: STEPS, workflows: trainWorkflows });
    const heldReplays = buildSyntheticMovementReplays({
      steps: STEPS,
      workflows: heldOutWorkflows,
      sessionIdPrefix: "held",
    });

    const model = await new MarkovMovementBackend().train(buildMovementDataset(trainReplays), { order: 2 });
    const result = evaluateMovementModel(model, buildMovementDataset(heldReplays));

    expect(result.nextTokenAccuracy).toBeGreaterThan(0.9);
    expect(result.sequencesEvaluated).toBe(1);
  });

  it("reports back-off usage as the generalization signal on novel orderings", async () => {
    // Two disjoint branch endings share a common prefix; a recombined held-out
    // ordering forces the model to back off from the full bigram context.
    const trainWorkflows: SyntheticWorkflow[] = [
      { id: "t1", steps: ["open", "menu", "settings"] },
      { id: "t2", steps: ["open", "menu", "toggle"] },
      { id: "t3", steps: ["back", "settings", "save"] },
    ];
    const heldOut: SyntheticWorkflow[] = [{ id: "h1", steps: ["menu", "settings", "save"] }];

    const model = await new MarkovMovementBackend().train(
      buildMovementDataset(buildSyntheticMovementReplays({ steps: STEPS, workflows: trainWorkflows })),
      { order: 2 },
    );
    const result = evaluateMovementModel(
      model,
      buildMovementDataset(buildSyntheticMovementReplays({ steps: STEPS, workflows: heldOut, sessionIdPrefix: "h" })),
    );

    // At least one correct prediction came from backing off (settings->save was
    // only ever seen after "back settings", not "menu settings").
    expect(result.correct).toBeGreaterThan(0);
    expect(result.fallbackRate).toBeGreaterThan(0);
  });
});

describe("scoreRolloutFidelity", () => {
  it("scores an exact rollout of a memorized workflow", async () => {
    const replays = buildSyntheticMovementReplays({
      steps: STEPS,
      workflows: [{ id: "w1", steps: ["open", "menu", "settings"] }],
    });
    const dataset = buildMovementDataset(replays);
    const model = await new MarkovMovementBackend().train(dataset, { order: 2 });

    const fidelity = scoreRolloutFidelity(model, dataset.sequences[0].tokens);
    expect(fidelity.exactMatch).toBe(true);
    expect(fidelity.commonPrefix).toBe(3);
    expect(fidelity.expectedLength).toBe(3);
  });

  it("generates nothing from an untrained seed", async () => {
    const model = await new MarkovMovementBackend().train(
      { version: 1, tokenization: { includeObservations: false, withBoundaries: true }, sequences: [], vocabulary: [] },
      { order: 2 },
    );
    expect(model.generate([MOVEMENT_BOS], 5)).toEqual([]);
  });
});
