import { describe, expect, it } from "vitest";
import { MarkovMovementBackend, decodeToken } from "./backend.js";
import {
  generateSyntheticDataset,
  tokenizeSequence,
  type MovementDataset,
  type MovementSequence,
} from "./event.js";

const backend = new MarkovMovementBackend();

const formFill: MovementSequence = {
  id: "form-fill-fixed",
  appId: "browser",
  label: "form-fill",
  events: [
    { t: 0, type: "move", x: 400, y: 200, target: "name-field" },
    { t: 120, type: "click", button: "left", target: "name-field" },
    { t: 300, type: "type", key: "text" },
    { t: 900, type: "move", x: 500, y: 360, target: "submit-button" },
    { t: 1020, type: "click", button: "left", target: "submit-button" },
  ],
};

describe("movement/backend MarkovMovementBackend", () => {
  it("trains a serializable artifact with correct stats", async () => {
    const dataset: MovementDataset = { version: 1, sequences: [formFill] };
    const model = await backend.train(dataset, { order: 3 });
    expect(model.backend).toBe("markov");
    expect(model.order).toBe(3);
    expect(model.stats.sequenceCount).toBe(1);
    expect(model.stats.eventCount).toBe(5);
    // Round-trips through JSON (persistable like every other store).
    expect(JSON.parse(JSON.stringify(model))).toEqual(model);
  });

  it("repeats a recorded movement sequence exactly (replay/repeat)", async () => {
    const model = await backend.train({ version: 1, sequences: [formFill] }, { order: 3 });
    const generated = await backend.generate(
      model,
      { history: [] },
      { maxSteps: formFill.events.length },
    );
    const generatedTokens = tokenizeSequence({ id: "gen", events: generated }, model.tokenize);
    expect(generatedTokens).toEqual(tokenizeSequence(formFill, model.tokenize));
  });

  it("predicts the recorded next event with high confidence from a seen prefix", async () => {
    const model = await backend.train({ version: 1, sequences: [formFill] }, { order: 3 });
    const prediction = await backend.predict(model, {
      history: formFill.events.slice(0, 2),
    });
    expect(prediction.token).toBe("type:text");
    expect(prediction.confidence).toBe(1);
    expect(prediction.event).toMatchObject({ type: "type", key: "text" });
    expect(prediction.backoffOrder).toBeGreaterThan(0);
  });

  it("backs off to shorter context for an unseen prefix and still predicts", async () => {
    // Train on two related sequences so a shared bigram exists.
    const other: MovementSequence = {
      id: "other",
      events: [
        { t: 0, type: "click", button: "left", target: "name-field" },
        { t: 100, type: "type", key: "text" },
      ],
    };
    const model = await backend.train({ version: 1, sequences: [formFill, other] }, { order: 3 });
    // A never-seen high-order context (a scroll immediately before the click was
    // never recorded), but the bigram "click name-field -> type" was.
    const prediction = await backend.predict(model, {
      history: [
        { t: 0, type: "scroll", dy: 120 },
        { t: 100, type: "click", button: "left", target: "name-field" },
      ],
    });
    // Should still predict "type:text" via lower-order backoff.
    expect(prediction.token).toBe("type:text");
    expect(prediction.backoffOrder).toBeLessThan(model.order);
  });

  it("returns zero-confidence when nothing was trained", async () => {
    const model = await backend.train({ version: 1, sequences: [] }, { order: 2 });
    const prediction = await backend.predict(model, { history: [] });
    expect(prediction.confidence).toBe(0);
    expect(prediction.token).toBeUndefined();
    expect(prediction.distribution).toEqual([]);
  });

  it("produces a deterministic, descending-sorted distribution", async () => {
    const dataset = generateSyntheticDataset({ seed: 3, count: 20 });
    const model = await backend.train(dataset, { order: 2 });
    const prediction = await backend.predict(model, { history: [] });
    const probs = prediction.distribution.map((entry) => entry.probability);
    const sorted = [...probs].sort((a, b) => b - a);
    expect(probs).toEqual(sorted);
    const total = probs.reduce((sum, value) => sum + value, 0);
    expect(total).toBeCloseTo(1, 6);
  });

  it("decodes tokens back into concrete events", () => {
    expect(decodeToken("click:left:@submit-button")).toMatchObject({
      type: "click",
      button: "left",
      target: "submit-button",
    });
    expect(decodeToken("scroll:down")).toMatchObject({ type: "scroll", dy: 1 });
    expect(decodeToken("move:#6,3")).toMatchObject({ type: "move", x: 6, y: 3 });
  });

  it("stops generation at a provided stop token", async () => {
    const model = await backend.train({ version: 1, sequences: [formFill] }, { order: 3 });
    const generated = await backend.generate(
      model,
      { history: [] },
      { maxSteps: 10, stopToken: "type:text" },
    );
    const tokens = tokenizeSequence({ id: "g", events: generated }, model.tokenize);
    expect(tokens).not.toContain("type:text");
  });
});
