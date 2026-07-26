import { describe, expect, it } from "vitest";
import { buildMovementDataset } from "./dataset.js";
import { MarkovMovementBackend } from "./markov-backend.js";
import { evaluateMovementModel } from "./eval.js";
import {
  generateRelatedTrajectory,
  generateWorkflowTrajectories,
  type WorkflowSpec,
} from "./synthetic.js";

const WORKFLOW: WorkflowSpec = {
  appName: "mail",
  steps: [
    { tool: "device", gesture: "tap", target: "compose" },
    { tool: "device", gesture: "type", target: "recipient" },
    { tool: "device", gesture: "type", target: "body" },
    { tool: "device", gesture: "tap", target: "send" },
  ],
};

async function trainOnWorkflow(order = 2) {
  const trajectories = generateWorkflowTrajectories(WORKFLOW, 5);
  const dataset = buildMovementDataset(trajectories, { order, requireApproved: true });
  const model = await new MarkovMovementBackend().train(dataset);
  return { trajectories, dataset, model };
}

describe("MarkovMovementBackend", () => {
  it("reproduces a recorded movement sequence with full-context prediction", async () => {
    const { trajectories, model } = await trainOnWorkflow();

    // Reproduction fidelity: on the trained workflow, every step should be the
    // top-1 prediction, using the full requested context (no backoff).
    const result = evaluateMovementModel(model, trajectories.slice(0, 1), { order: 2, requireApproved: true });
    expect(result.accuracy).toBe(1);
    expect(result.generalizationRate).toBe(0);
  });

  it("predicts the next movement given a partial history", async () => {
    const { model } = await trainOnWorkflow();
    const prediction = model.predict({
      appContext: "mail",
      history: ["device:tap:compose", "device:type:recipient"],
    });
    expect(prediction?.action).toBe("device:type:body");
    expect(prediction?.probability).toBeGreaterThan(0);
    expect(prediction?.backoffOrder).toBe(2);
  });

  it("generalises to a related, never-seen trajectory via backoff", async () => {
    const { model } = await trainOnWorkflow();
    const related = generateRelatedTrajectory(WORKFLOW, { seed: 7 });

    const result = evaluateMovementModel(model, [related], { order: 2, requireApproved: true });
    // It never saw this exact sequence, yet the shared vocabulary + app context
    // let it recover most movements — and it reports having generalised.
    expect(result.total).toBeGreaterThan(0);
    expect(result.topKAccuracy).toBeGreaterThanOrEqual(0.5);
    expect(result.generalizationRate).toBeGreaterThan(0);
  });

  it("backs off to the app-agnostic unigram for an unseen app context", async () => {
    const { model } = await trainOnWorkflow();
    const prediction = model.predict({ appContext: "totally-unknown-app", history: [] });
    expect(prediction).toBeDefined();
    expect(prediction?.backoffOrder).toBe(-1);
  });

  it("returns undefined when the model is empty", async () => {
    const model = await new MarkovMovementBackend().train(buildMovementDataset([], { order: 2 }));
    expect(model.predict({ history: [] })).toBeUndefined();
  });

  it("round-trips through serialize/load with identical predictions", async () => {
    const backend = new MarkovMovementBackend();
    const { model } = await trainOnWorkflow();
    const context = { appContext: "mail", history: ["device:tap:compose"] };

    const before = model.predict(context);
    const reloaded = backend.load(JSON.parse(JSON.stringify(model.serialize())));
    const after = reloaded.predict(context);

    expect(after).toEqual(before);
  });

  it("applies additive smoothing without changing the argmax", async () => {
    const trajectories = generateWorkflowTrajectories(WORKFLOW, 5);
    const dataset = buildMovementDataset(trajectories, { order: 2, requireApproved: true });
    const smoothed = await new MarkovMovementBackend().train(dataset, { smoothing: 0.5 });
    const prediction = smoothed.predict({ appContext: "mail", history: ["device:tap:compose"] });
    expect(prediction?.action).toBe("device:type:recipient");
    // Smoothing pulls probability mass toward the vocabulary, so it stays < 1.
    expect(prediction!.probability).toBeLessThan(1);
    expect(prediction!.probability).toBeGreaterThan(0);
  });
});
