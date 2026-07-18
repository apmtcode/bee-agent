import { describe, expect, it } from "vitest";
import { generateSyntheticMovementTrajectories } from "../capture/synthetic.js";
import { buildTrajectorySpan } from "../capture/trajectory.js";
import {
  MarkovMovementBackend,
  restoreMarkovMovementBackend,
  type MovementPolicyBackend,
} from "./policy-backend.js";

function trainedBackend(): MovementPolicyBackend {
  const trajectories = generateSyntheticMovementTrajectories(
    [
      {
        name: "open-and-save",
        steps: ["focus-window", "click-menu", "click-save", "type-filename", "press-enter"],
        observationSource: "editor",
      },
    ],
    { repeat: 3 },
  );
  const backend = new MarkovMovementBackend({ maxOrder: 3 });
  backend.fit(trajectories);
  return backend;
}

describe("MarkovMovementBackend", () => {
  it("returns undefined before any training", () => {
    const backend = new MarkovMovementBackend();
    expect(backend.predict({ history: ["focus-window"] })).toBeUndefined();
  });

  it("recalls the recorded next movement from full context", () => {
    const backend = trainedBackend();
    const prediction = backend.predict({ history: ["click-menu", "click-save"] });
    expect(prediction?.tool).toBe("type-filename");
    expect(prediction?.source).toBe("recall");
    expect(prediction?.confidence).toBe(1);
  });

  it("replays a full recorded movement sequence via rollout", () => {
    const backend = trainedBackend();
    const rollout = backend.rollout({ history: ["focus-window"] }, 4);
    expect(rollout.map((step) => step.tool)).toEqual([
      "click-menu",
      "click-save",
      "type-filename",
      "press-enter",
    ]);
  });

  it("generalises to a related-but-unseen prefix by backing off", () => {
    // "click-save" was only ever seen after "click-menu"; here the immediate
    // history is unseen at full order, so the model must back off to the
    // shorter context that it does know.
    const backend = trainedBackend();
    const prediction = backend.predict({ history: ["some-unseen-move", "click-save"] });
    expect(prediction?.tool).toBe("type-filename");
    expect(prediction?.source).toBe("backoff");
  });

  it("falls back to the unigram prior when no context matches", () => {
    const backend = new MarkovMovementBackend({ maxOrder: 2 });
    backend.fit(
      generateSyntheticMovementTrajectories([
        { name: "a", steps: ["x", "y", "y", "y"] },
      ]),
    );
    const prediction = backend.predict({ history: ["totally-unknown"] });
    expect(prediction?.source).toBe("prior");
    // "y" is the most frequent action overall, so the prior favours it.
    expect(prediction?.tool).toBe("y");
  });

  it("breaks ties deterministically by lexical order", () => {
    const backend = new MarkovMovementBackend({ maxOrder: 1 });
    backend.fit([
      buildTrajectorySpan({
        id: "t1",
        sessionId: "s1",
        actions: [
          { kind: "action", tool: "start", summary: "start", ts: 1 },
          { kind: "action", tool: "beta", summary: "b", ts: 2 },
        ],
      }),
      buildTrajectorySpan({
        id: "t2",
        sessionId: "s1",
        actions: [
          { kind: "action", tool: "start", summary: "start", ts: 1 },
          { kind: "action", tool: "alpha", summary: "a", ts: 2 },
        ],
      }),
    ]);
    // start -> {alpha:1, beta:1}; tie broken to the lexically smaller "alpha".
    expect(backend.predict({ history: ["start"] })?.tool).toBe("alpha");
  });

  it("carries a representative summary for the predicted tool", () => {
    const backend = new MarkovMovementBackend({ maxOrder: 1 });
    backend.fit([
      buildTrajectorySpan({
        id: "t1",
        sessionId: "s1",
        actions: [
          { kind: "action", tool: "focus", summary: "focus", ts: 1 },
          { kind: "action", tool: "save", summary: "Save the document", ts: 2 },
        ],
      }),
    ]);
    expect(backend.predict({ history: ["focus"] })?.summary).toBe("Save the document");
  });

  it("reports training info", () => {
    const backend = trainedBackend();
    const info = backend.info();
    expect(info.kind).toBe("markov");
    expect(info.trainedTrajectories).toBe(3);
    expect(info.trainedActions).toBe(15);
    expect(info.vocabulary).toBe(5);
  });

  it("round-trips through a snapshot without changing predictions", () => {
    const backend = trainedBackend();
    const snapshot = backend.snapshot();
    const restored = restoreMarkovMovementBackend(JSON.parse(JSON.stringify(snapshot)));

    for (const history of [["focus-window"], ["click-menu", "click-save"], ["press-enter"]]) {
      expect(restored.predict({ history })).toEqual(backend.predict({ history }));
    }
    expect(restored.info()).toEqual(backend.info());
  });

  it("rejects snapshots from an unknown backend kind", () => {
    const snapshot = trainedBackend().snapshot();
    expect(() =>
      restoreMarkovMovementBackend({ ...snapshot, kind: "transformer" }),
    ).toThrow(/unsupported movement policy snapshot kind/);
  });

  it("prefers review-redacted actions over raw actions when present", () => {
    const span = buildTrajectorySpan({
      id: "t1",
      sessionId: "s1",
      actions: [
        { kind: "action", tool: "secret", summary: "secret", ts: 1 },
        { kind: "action", tool: "leaked", summary: "leaked", ts: 2 },
      ],
    });
    span.review = {
      status: "approved",
      reviewedAt: "2026-01-01T00:00:00Z",
      reviewedBy: "reviewer",
      redactedActions: [
        { ts: 1, tool: "safe-start", summary: "safe-start" },
        { ts: 2, tool: "safe-next", summary: "safe-next" },
      ],
    };
    const backend = new MarkovMovementBackend({ maxOrder: 1 });
    backend.fit([span]);
    expect(backend.predict({ history: ["safe-start"] })?.tool).toBe("safe-next");
    // The raw (unreviewed) tokens must never enter the trained vocabulary; only
    // the redacted actions are learned.
    expect(backend.info().vocabulary).toBe(2);
    const priorPrediction = backend.predict({ history: ["secret"] });
    expect(["safe-start", "safe-next"]).toContain(priorPrediction?.tool);
  });
});

describe("generateSyntheticMovementTrajectories", () => {
  it("expands programs deterministically with an injectable step clock", () => {
    const [span] = generateSyntheticMovementTrajectories(
      [{ name: "demo", steps: ["a", "b"], observationSource: "app" }],
      { startTs: 1000, stepMillis: 10 },
    );
    expect(span?.observations[0]).toMatchObject({ source: "app", ts: 1000 });
    expect(span?.actions.map((action) => action.ts)).toEqual([1010, 1020]);
    expect(span?.actions.map((action) => action.tool)).toEqual(["a", "b"]);
  });

  it("repeats programs into distinct spans", () => {
    const spans = generateSyntheticMovementTrajectories([{ name: "p", steps: ["a"] }], {
      repeat: 2,
    });
    expect(spans.map((span) => span.id)).toEqual(["p-0", "p-1"]);
  });
});
