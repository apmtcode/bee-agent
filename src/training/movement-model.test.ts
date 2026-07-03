import { describe, expect, it } from "vitest";
import type { ReplayManifest } from "../capture/replay.js";
import {
  MarkovMovementBackend,
  MOVEMENT_END_TOKEN,
  type MovementModelBackend,
  type MovementSequence,
  defaultMovementBackend,
  evaluateReplayFidelity,
  movementTokensFromReplay,
  movementTokenForEvent,
} from "./movement-model.js";

function seq(...tokens: string[]): MovementSequence {
  return tokens;
}

describe("movementTokensFromReplay", () => {
  const manifest: Pick<ReplayManifest, "events"> = {
    events: [
      { kind: "observation", ts: 1, trajectoryId: "t1", source: "window", summary: "focus editor" },
      { kind: "action", ts: 2, trajectoryId: "t1", tool: "mouse.move", summary: "to (10,20)" },
      { kind: "transcript", ts: 3, messageId: "m1", role: "assistant", content: "typing" },
      { kind: "action", ts: 4, trajectoryId: "t1", tool: "keyboard.type", summary: "hello" },
    ],
  };

  it("emits action tokens by default in timeline order", () => {
    expect(movementTokensFromReplay(manifest)).toEqual([
      "action:mouse.move",
      "action:keyboard.type",
    ]);
  });

  it("can include other kinds when asked", () => {
    expect(movementTokensFromReplay(manifest, { include: ["observation", "action", "transcript"] })).toEqual([
      "obs:window",
      "action:mouse.move",
      "msg:assistant",
      "action:keyboard.type",
    ]);
  });

  it("derives a stable token per event kind", () => {
    expect(movementTokenForEvent({ kind: "action", ts: 0, trajectoryId: "t", tool: "click", summary: "" })).toBe(
      "action:click",
    );
    expect(movementTokenForEvent({ kind: "observation", ts: 0, trajectoryId: "t", source: "app", summary: "" })).toBe(
      "obs:app",
    );
  });
});

describe("MarkovMovementBackend — repeat recorded movements (objective 2c)", () => {
  it("reproduces a recorded movement exactly from an empty seed", () => {
    const recorded = seq("action:focus", "action:type", "action:enter");
    const model = new MarkovMovementBackend().train([recorded]);
    expect(model.generate([])).toEqual(recorded);
  });

  it("continues a partial recorded prefix to completion", () => {
    const recorded = seq("action:focus", "action:type", "action:enter");
    const model = new MarkovMovementBackend().train([recorded]);
    expect(model.generate(seq("action:focus"))).toEqual(seq("action:type", "action:enter"));
  });

  it("predicts the recorded next step with full confidence when unambiguous", () => {
    const model = new MarkovMovementBackend().train([seq("action:a", "action:b")]);
    const prediction = model.predictNext(seq("action:a"));
    expect(prediction.token).toBe("action:b");
    expect(prediction.confidence).toBe(1);
  });
});

describe("MarkovMovementBackend — generalize to new but related movements (objective 2d)", () => {
  it("continues a novel prefix that ends in a familiar sub-sequence via backoff", () => {
    // Two recorded movements share the tail "open -> save". A brand-new prefix
    // that has never been seen but ends in "open" should still predict "save".
    const model = new MarkovMovementBackend().train([
      seq("action:launch", "action:open", "action:save"),
      seq("action:focus", "action:open", "action:save"),
    ]);
    const prediction = model.predictNext(seq("action:navigate", "action:open"));
    expect(prediction.token).toBe("action:save");
    // Matched only the last token, not the full (unseen) prefix.
    expect(prediction.order).toBe(1);
  });

  it("prefers the longest matching context (memorization) over backoff", () => {
    // After "b", "c" is more common globally, but after "a b" the recorded
    // continuation is "d". The higher-order context must win.
    const model = new MarkovMovementBackend().train([
      seq("action:a", "action:b", "action:d"),
      seq("action:x", "action:b", "action:c"),
      seq("action:y", "action:b", "action:c"),
    ]);
    expect(model.predictNext(seq("action:a", "action:b")).token).toBe("action:d");
    expect(model.predictNext(seq("action:z", "action:b")).token).toBe("action:c");
  });

  it("returns no continuation for a wholly unfamiliar context", () => {
    const model = new MarkovMovementBackend().train([seq("action:a", "action:b")]);
    const prediction = model.predictNext(seq("action:unseen"));
    // Backs off to the unigram distribution rather than failing outright.
    expect(prediction.order).toBe(0);
    expect(prediction.token).not.toBeNull();
  });
});

describe("MarkovMovementBackend — determinism & config", () => {
  it("is deterministic across repeated training and prediction", () => {
    const dataset = [seq("action:a", "action:b"), seq("action:a", "action:c")];
    const first = new MarkovMovementBackend().train(dataset).predictNext(seq("action:a"));
    const second = new MarkovMovementBackend().train(dataset).predictNext(seq("action:a"));
    expect(second).toEqual(first);
  });

  it("breaks ties deterministically by count then token order", () => {
    // "a" is followed once by "c" and once by "b" -> equal counts, so the
    // alphabetically-smaller token wins.
    const model = new MarkovMovementBackend().train([seq("action:a", "action:c"), seq("action:a", "action:b")]);
    expect(model.predictNext(seq("action:a")).token).toBe("action:b");
  });

  it("respects maxOrder: order 1 cannot distinguish longer contexts", () => {
    const dataset = [seq("action:a", "action:b", "action:d"), seq("action:x", "action:b", "action:c")];
    const lowOrder = new MarkovMovementBackend().train(dataset, { maxOrder: 1 });
    // With only bigrams, "b" -> whichever tail is more frequent (tie -> token order "c").
    expect(lowOrder.predictNext(seq("action:a", "action:b")).order).toBe(1);
  });

  it("caps rollout length via maxSteps", () => {
    // A self-looping movement would generate forever without the cap.
    const model = new MarkovMovementBackend().train([seq("action:tick", "action:tick", "action:tick")]);
    const generated = model.generate(seq("action:tick"), { maxSteps: 5 });
    expect(generated.length).toBeLessThanOrEqual(5);
  });
});

describe("snapshot round-trip", () => {
  it("restores an identical model from its snapshot", () => {
    const backend = new MarkovMovementBackend();
    const trained = backend.train([seq("action:a", "action:b", "action:c")]);
    const restored = backend.restore(JSON.parse(JSON.stringify(trained.toJSON())));
    expect(restored.generate([])).toEqual(trained.generate([]));
    expect(restored.predictNext(seq("action:a"))).toEqual(trained.predictNext(seq("action:a")));
  });

  it("records the end marker in learned counts", () => {
    const snapshot = new MarkovMovementBackend().train([seq("action:a")]).toJSON();
    expect(snapshot.counts[""]).toHaveProperty(MOVEMENT_END_TOKEN);
    expect(snapshot.sequenceCount).toBe(1);
  });
});

describe("pluggability", () => {
  it("defaultMovementBackend satisfies the MovementModelBackend contract", () => {
    const backend: MovementModelBackend = defaultMovementBackend();
    expect(backend.name).toBe("markov");
    const model = backend.train([seq("action:a", "action:b")]);
    expect(model.backend).toBe("markov");
  });

  it("an alternative backend can implement the same interface", () => {
    // A trivial always-null backend proves the seam is swappable.
    const nullBackend: MovementModelBackend = {
      name: "null",
      train: () => ({
        backend: "null",
        predictNext: () => ({ token: null, confidence: 0, order: 0, alternatives: [] }),
        generate: () => [],
        toJSON: () => ({ version: 1, backend: "null", maxOrder: 0, counts: {}, sequenceCount: 0 }),
      }),
      restore: () => nullBackend.train([]),
    };
    const model = nullBackend.train([seq("action:a")]);
    expect(model.generate([])).toEqual([]);
  });
});

describe("evaluateReplayFidelity — generalization eval harness", () => {
  it("scores a memorized sequence at perfect fidelity", () => {
    const recorded = seq("action:a", "action:b", "action:c");
    const model = new MarkovMovementBackend().train([recorded]);
    const fidelity = evaluateReplayFidelity(model, recorded);
    expect(fidelity.accuracy).toBe(1);
    expect(fidelity.matched).toBe(fidelity.total);
  });

  it("scores a held-out related sequence between 0 and 1", () => {
    const model = new MarkovMovementBackend().train([
      seq("action:launch", "action:open", "action:save"),
      seq("action:focus", "action:open", "action:save"),
    ]);
    // Held-out but related: new head, familiar "open -> save" tail.
    const heldOut = seq("action:navigate", "action:open", "action:save");
    const fidelity = evaluateReplayFidelity(model, heldOut);
    expect(fidelity.accuracy).toBeGreaterThan(0);
    expect(fidelity.accuracy).toBeLessThanOrEqual(1);
    expect(fidelity.total).toBe(heldOut.length + 1);
  });
});
