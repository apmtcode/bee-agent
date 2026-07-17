import { describe, expect, it } from "vitest";
import {
  NgramMovementPolicyBackend,
  evaluateNextTokenFidelity,
  type MovementDataset,
  type MovementSequence,
} from "./movement-policy.js";
import { generateSyntheticMovementDataset, movementSequencesFromReplays } from "./movement-synthetic.js";
import type { ReplayManifest } from "../capture/replay.js";

describe("NgramMovementPolicyBackend", () => {
  it("repeats recorded movements with perfect teacher-forced fidelity", () => {
    const sequence: MovementSequence = {
      id: "recorded-1",
      tokens: [
        "pointer.move",
        "pointer.down",
        "pointer.up",
        "key.down",
        "key.up",
        "scroll.begin",
        "scroll.delta",
        "scroll.end",
      ],
    };
    const model = new NgramMovementPolicyBackend({ maxOrder: 3 }).train({ sequences: [sequence] });

    const fidelity = evaluateNextTokenFidelity(model, [sequence]);
    expect(fidelity.accuracy).toBe(1);
    expect(fidelity.coverage).toBe(1);

    // Rolling out from the recorded seed reproduces the recorded tail exactly.
    const generated = model.rollout(sequence.tokens.slice(0, 2), sequence.tokens.length - 2);
    expect(generated).toEqual(sequence.tokens.slice(2));
  });

  it("marks the highest matched order as an exact prediction", () => {
    const model = new NgramMovementPolicyBackend({ maxOrder: 2 }).train({
      sequences: [{ id: "s", tokens: ["a", "b", "c", "a", "b", "c"] }],
    });
    const prediction = model.predict(["a", "b"]);
    expect(prediction.token).toBe("c");
    expect(prediction.source).toBe("exact");
    expect(prediction.contextOrder).toBe(2);
    expect(prediction.confidence).toBeCloseTo(1);
  });

  it("generalizes to unseen-but-related movements via context backoff", () => {
    // Two related recordings that share the same local structure.
    const dataset: MovementDataset = {
      sequences: [
        { id: "a", tokens: ["pointer.move", "pointer.down", "pointer.up", "key.down", "key.up"] },
        { id: "b", tokens: ["window.focus", "pointer.move", "pointer.down", "pointer.up", "key.down"] },
      ],
    };
    const model = new NgramMovementPolicyBackend({ maxOrder: 3 }).train(dataset);

    // A prefix never seen verbatim, but whose suffix ("pointer.down") was always
    // followed by "pointer.up" in training. Backoff must recover that.
    const prediction = model.predict(["scroll.end", "pointer.move", "pointer.down"]);
    expect(prediction.token).toBe("pointer.up");
    expect(prediction.source).toBe("backoff");
    expect(prediction.contextOrder).toBeGreaterThanOrEqual(1);
  });

  it("returns an empty prediction when it has no evidence at all", () => {
    const model = new NgramMovementPolicyBackend().train({ sequences: [] });
    const prediction = model.predict(["anything"]);
    expect(prediction.token).toBeUndefined();
    expect(prediction.source).toBe("none");
    expect(model.rollout(["seed"], 5)).toEqual([]);
  });

  it("generalizes on held-out synthetic trajectories better than a unigram baseline", () => {
    const train = generateSyntheticMovementDataset({ seed: 7, sequenceCount: 40 });
    const heldOut = generateSyntheticMovementDataset({
      seed: 101,
      sequenceCount: 12,
      idPrefix: "held-out",
    });

    const model = new NgramMovementPolicyBackend({ maxOrder: 3 }).train(train);
    const unigram = new NgramMovementPolicyBackend({ maxOrder: 0 }).train(train);

    const modelFidelity = evaluateNextTokenFidelity(model, heldOut.sequences);
    const unigramFidelity = evaluateNextTokenFidelity(unigram, heldOut.sequences);

    // The held-out sequences are novel but share motif structure, so the
    // context model must generalize meaningfully and beat the context-free
    // baseline. Deterministic seeds keep these thresholds stable.
    expect(modelFidelity.coverage).toBe(1);
    expect(modelFidelity.accuracy).toBeGreaterThan(0.6);
    expect(modelFidelity.accuracy).toBeGreaterThan(unigramFidelity.accuracy);
  });
});

describe("movementSequencesFromReplays", () => {
  it("extracts action tokens from replay manifests and drops empty ones", () => {
    const replays: ReplayManifest[] = [
      {
        version: 1,
        sessionId: "session-1",
        trajectoryIds: ["traj-1"],
        eventCount: 3,
        events: [
          { kind: "transcript", ts: 1, messageId: "m1", role: "user", content: "go" },
          { kind: "observation", ts: 2, trajectoryId: "traj-1", source: "screen", summary: "window" },
          { kind: "action", ts: 3, trajectoryId: "traj-1", tool: "pointer.down", summary: "press at 10,20" },
          { kind: "action", ts: 4, trajectoryId: "traj-1", tool: "pointer.up", summary: "release" },
        ],
      },
      {
        version: 1,
        sessionId: "session-empty",
        trajectoryIds: ["traj-2"],
        eventCount: 1,
        events: [{ kind: "observation", ts: 1, trajectoryId: "traj-2", source: "screen", summary: "idle" }],
      },
    ];

    const sequences = movementSequencesFromReplays(replays);
    expect(sequences).toEqual([{ id: "session-1", tokens: ["pointer.down", "pointer.up"] }]);
  });

  it("produces a model that repeats replay-derived movements", () => {
    const replays: ReplayManifest[] = [
      {
        version: 1,
        sessionId: "session-1",
        trajectoryIds: ["traj-1"],
        eventCount: 4,
        events: [
          { kind: "action", ts: 1, trajectoryId: "traj-1", tool: "pointer.move", summary: "" },
          { kind: "action", ts: 2, trajectoryId: "traj-1", tool: "pointer.down", summary: "" },
          { kind: "action", ts: 3, trajectoryId: "traj-1", tool: "pointer.up", summary: "" },
          { kind: "action", ts: 4, trajectoryId: "traj-1", tool: "key.down", summary: "" },
        ],
      },
    ];
    const sequences = movementSequencesFromReplays(replays);
    const model = new NgramMovementPolicyBackend().train({ sequences });
    expect(evaluateNextTokenFidelity(model, sequences).accuracy).toBe(1);
  });
});
