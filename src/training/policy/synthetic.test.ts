import { describe, expect, it } from "vitest";
import { buildMovementDataset } from "./dataset.js";
import { actionKey } from "./model.js";
import { NgramMovementBackend } from "./ngram-backend.js";
import { evaluateMovementModel, generateSyntheticTrajectories, mailComposeFlow } from "./synthetic.js";

const backend = new NgramMovementBackend();

describe("generateSyntheticTrajectories", () => {
  it("produces one deterministic trajectory per variant", () => {
    const a = generateSyntheticTrajectories(mailComposeFlow(), { variants: ["compose", "reply"] });
    const b = generateSyntheticTrajectories(mailComposeFlow(), { variants: ["compose", "reply"] });
    expect(a).toEqual(b);
    expect(a).toHaveLength(2);
    expect(a[0].id).toBe("mail-{v}-compose");
    expect(a[0].observations).toHaveLength(3);
    expect(a[0].actions).toHaveLength(3);
  });

  it("substitutes the variant into observations and actions", () => {
    const [traj] = generateSyntheticTrajectories(mailComposeFlow(), { variants: ["reply"] });
    expect(traj.observations[0].summary).toContain("reply option focused");
    expect(traj.actions[0].metadata?.target).toBe("reply");
    expect(traj.observations[1].summary).toBe("reply editor open");
  });

  it("attaches an outcome reward when requested", () => {
    const [traj] = generateSyntheticTrajectories(mailComposeFlow(), { variants: ["compose"], reward: 2 });
    expect(traj.outcome).toMatchObject({ status: "success", reward: 2 });
  });
});

describe("evaluateMovementModel — generalization harness", () => {
  it("scores held-out variants and reports generalized hits", () => {
    const train = generateSyntheticTrajectories(mailComposeFlow(), { variants: ["compose", "reply", "archive"] });
    const heldOut = generateSyntheticTrajectories(mailComposeFlow(), { variants: ["forward"] });
    const model = backend.train(buildMovementDataset(train));

    const result = evaluateMovementModel(backend, model, heldOut);
    expect(result.total).toBe(3);
    // The shared final "tap send" step must be recovered on a wholly unseen variant.
    expect(result.correct).toBeGreaterThanOrEqual(1);
    expect(result.accuracy).toBeGreaterThan(0);
    // At least one held-out step is answered via generalization, not exact recall.
    expect(result.byMethod.generalized.total).toBeGreaterThanOrEqual(1);
  });

  it("scores 100% when evaluated on its own training data", () => {
    const train = generateSyntheticTrajectories(mailComposeFlow(), { variants: ["compose", "reply"] });
    const model = backend.train(buildMovementDataset(train));
    const result = evaluateMovementModel(backend, model, train);
    expect(result.accuracy).toBe(1);
    expect(result.byMethod.exact.total).toBe(result.total);
  });

  it("recovers the shared terminal action across every held-out variant", () => {
    const train = generateSyntheticTrajectories(mailComposeFlow(), { variants: ["compose", "reply", "archive", "flag"] });
    const model = backend.train(buildMovementDataset(train));
    for (const variant of ["forward", "snooze", "pin"]) {
      const [heldOut] = generateSyntheticTrajectories(mailComposeFlow(), { variants: [variant] });
      const dataset = buildMovementDataset([heldOut]);
      const sendExample = dataset.examples.find((e) => actionKey(e.action) === "device tap:send");
      const prediction = backend.predict(model, sendExample!.context);
      expect(prediction.action && actionKey(prediction.action)).toBe("device tap:send");
    }
  });
});
