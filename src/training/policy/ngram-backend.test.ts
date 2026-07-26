import { describe, expect, it } from "vitest";
import { buildMovementDataset } from "./dataset.js";
import { actionKey, type MovementContext } from "./model.js";
import { NGRAM_BACKEND_ID, NgramMovementBackend, rolloutMovements } from "./ngram-backend.js";
import { generateSyntheticTrajectories, mailComposeFlow } from "./synthetic.js";

const backend = new NgramMovementBackend();

function trainOnMail(variants: string[]) {
  const trajectories = generateSyntheticTrajectories(mailComposeFlow(), { variants });
  return backend.train(buildMovementDataset(trajectories), { trainedAt: "2026-07-26T00:00:00Z" });
}

describe("NgramMovementBackend.train", () => {
  it("stamps backend metadata and the action vocabulary", () => {
    const model = trainOnMail(["compose", "reply"]);
    expect(model.backend).toBe(NGRAM_BACKEND_ID);
    expect(model.trainedAt).toBe("2026-07-26T00:00:00Z");
    expect(model.exampleCount).toBeGreaterThan(0);
    expect(model.actionVocabulary).toContain("device tap:send");
  });

  it("produces a null timestamp for a deterministic artifact when unstamped", () => {
    const trajectories = generateSyntheticTrajectories(mailComposeFlow(), { variants: ["compose"] });
    expect(backend.train(buildMovementDataset(trajectories)).trainedAt).toBeNull();
  });
});

describe("NgramMovementBackend.predict — exact recall", () => {
  it("recalls the exact next action for a seen context", () => {
    const trajectories = generateSyntheticTrajectories(mailComposeFlow(), { variants: ["compose"] });
    const dataset = buildMovementDataset(trajectories);
    const model = backend.train(dataset);

    // First example's context (mail focused) should recall "tap compose".
    const prediction = backend.predict(model, dataset.examples[0].context);
    expect(prediction.method).toBe("exact");
    expect(prediction.action && actionKey(prediction.action)).toBe("device tap:compose");
    expect(prediction.confidence).toBeGreaterThan(0);
  });
});

describe("NgramMovementBackend.predict — generalization", () => {
  it("performs a new-but-related movement it never saw verbatim", () => {
    // Train on two variants; hold out a third, structurally identical variant.
    const model = trainOnMail(["compose", "reply"]);
    const heldOut = generateSyntheticTrajectories(mailComposeFlow(), { variants: ["forward"] });
    const dataset = buildMovementDataset(heldOut);

    // The final step of every mail flow ends in "tap send"; the shared tokens
    // (mail, draft-ready, prev:type) should let the model generalize to it.
    const sendExample = dataset.examples.find((e) => actionKey(e.action) === "device tap:send");
    expect(sendExample).toBeDefined();
    const prediction = backend.predict(model, sendExample!.context);
    expect(prediction.method).toBe("generalized");
    expect(prediction.action && actionKey(prediction.action)).toBe("device tap:send");
  });

  it("falls back to the prior when the context has no known tokens", () => {
    const model = trainOnMail(["compose"]);
    const alien: MovementContext = { tokens: ["src:unknown", "quantum", "widget"] };
    const prediction = backend.predict(model, alien);
    expect(prediction.method).toBe("prior");
    expect(prediction.action).toBeDefined();
  });

  it("returns method 'none' for an empty model", () => {
    const empty = backend.train({ version: 1, examples: [], actionVocabulary: [] });
    expect(backend.predict(empty, { tokens: ["anything"] })).toMatchObject({ method: "none", action: undefined });
  });
});

describe("NgramMovementBackend serialize round-trip", () => {
  it("re-imports to an identical model with identical predictions", () => {
    const model = trainOnMail(["compose", "reply"]);
    const restored = backend.deserialize(backend.serialize(model));
    expect(restored).toEqual(model);

    const trajectories = generateSyntheticTrajectories(mailComposeFlow(), { variants: ["compose"] });
    const context = buildMovementDataset(trajectories).examples[0].context;
    expect(backend.predict(restored, context)).toEqual(backend.predict(model, context));
  });

  it("rejects a serialized model from a different backend", () => {
    const foreign = JSON.stringify({ backend: "some-other-backend", state: {} });
    expect(() => backend.deserialize(foreign)).toThrow(/backend mismatch/);
  });
});

describe("rolloutMovements", () => {
  it("replays the learned action chain autoregressively", () => {
    const trajectories = generateSyntheticTrajectories(mailComposeFlow(), { variants: ["compose"] });
    const dataset = buildMovementDataset(trajectories);
    const model = backend.train(dataset);

    const steps = rolloutMovements(backend, model, dataset.examples[0].context, { maxSteps: 3 });
    expect(steps.length).toBeGreaterThanOrEqual(1);
    // The first predicted action of the mail flow is tapping compose.
    expect(steps[0].prediction.action && actionKey(steps[0].prediction.action)).toBe("device tap:compose");
    // Each subsequent context carries the previous action forward.
    if (steps.length > 1) {
      expect(steps[1].context.tokens).toContain("prev:device tap:compose");
    }
  });

  it("stops early when confidence drops below the floor", () => {
    const model = trainOnMail(["compose"]);
    const steps = rolloutMovements(backend, model, { tokens: ["totally", "unknown"] }, { maxSteps: 5, minConfidence: 1.1 });
    // A prior/generalized guess never reaches confidence 1.1, so it stops at step 0.
    expect(steps).toHaveLength(1);
  });
});
