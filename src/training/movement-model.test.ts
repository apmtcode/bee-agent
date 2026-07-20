import { describe, expect, it } from "vitest";
import { buildReplayManifest } from "../capture/replay.js";
import { buildTrajectorySpan } from "../capture/trajectory.js";
import {
  MarkovMovementBackend,
  MOVEMENT_END_TOKEN,
  datasetFromReplayManifests,
  evaluateNextTokenFidelity,
  generateSyntheticMovementDataset,
  replayManifestToSequence,
  trainMovementModel,
  trajectoryToMovementSequence,
  type MovementDataset,
} from "./movement-model.js";

const REPEAT_DATASET: MovementDataset = {
  sequences: [
    { id: "a", tokens: ["act:device:tapped Search", "act:device:typed query", "act:device:tapped Result"] },
    { id: "b", tokens: ["act:device:tapped Search", "act:device:typed query", "act:device:tapped Result"] },
  ],
};

describe("MarkovMovementBackend training", () => {
  it("is deterministic — same dataset yields identical artifacts", () => {
    const backend = new MarkovMovementBackend();
    const first = backend.train(REPEAT_DATASET);
    const second = backend.train(REPEAT_DATASET);
    expect(JSON.stringify(first)).toEqual(JSON.stringify(second));
    expect(first.sequenceCount).toBe(2);
    expect(first.vocabulary).toContain("act:device:tapped Search");
    expect(first.vocabulary).not.toContain(MOVEMENT_END_TOKEN);
  });

  it("records the end token as a transition target", () => {
    const { artifact } = trainMovementModel(REPEAT_DATASET);
    const endSeen = Object.values(artifact.transitions).some((dist) =>
      Object.keys(dist).includes(MOVEMENT_END_TOKEN),
    );
    expect(endSeen).toBe(true);
  });
});

describe("movement repetition (objective 2c)", () => {
  it("reproduces a recorded movement from an empty prompt", () => {
    const { inference } = trainMovementModel(REPEAT_DATASET);
    const generated = inference.generate([]);
    expect(generated).toEqual([
      "act:device:tapped Search",
      "act:device:typed query",
      "act:device:tapped Result",
    ]);
  });

  it("stops at end-of-sequence rather than looping", () => {
    const { inference } = trainMovementModel(REPEAT_DATASET);
    const generated = inference.generate([], { maxSteps: 50 });
    expect(generated.length).toBe(3);
    expect(inference.predictNext(REPEAT_DATASET.sequences[0]!.tokens).token).toBeNull();
  });
});

describe("movement generalization (objective 2d)", () => {
  it("completes a related-but-unseen prefix via n-gram backoff", () => {
    // Two skills share a common middle motif; the model should generalize the
    // shared continuation onto a novel combination it never saw verbatim.
    const dataset: MovementDataset = {
      sequences: [
        { id: "login", tokens: ["act:os:focused App", "act:device:tapped Field", "act:device:typed value", "act:device:tapped Submit"] },
        { id: "search", tokens: ["act:os:focused Browser", "act:device:tapped Field", "act:device:typed value", "act:device:tapped Submit"] },
      ],
    };
    const { inference } = trainMovementModel(dataset, new MarkovMovementBackend(2));
    // Novel opening app, then the shared "tapped Field" — backoff should predict
    // the learned continuation "typed value".
    const prediction = inference.predictNext(["act:os:focused Settings", "act:device:tapped Field"]);
    expect(prediction.token).toBe("act:device:typed value");
    expect(prediction.backoffOrder).toBeGreaterThanOrEqual(1);
  });

  it("falls back to end when it has no learned context at all", () => {
    const { inference } = trainMovementModel(REPEAT_DATASET);
    const empty = new MarkovMovementBackend().createInference({
      version: 1,
      backend: "markov-ngram",
      order: 3,
      vocabulary: [],
      transitions: {},
      sequenceCount: 0,
      transitionCount: 0,
    });
    expect(empty.predictNext(["anything"]).token).toBeNull();
    expect(empty.predictNext(["anything"]).backoffOrder).toBe(-1);
    // Sanity: the real model, via its START-padded generation path, still emits.
    expect(inference.generate([]).length).toBeGreaterThan(0);
  });
});

describe("synthetic streams + fidelity harness", () => {
  it("generates reproducible structured datasets from a seed", () => {
    const first = generateSyntheticMovementDataset({ sequenceCount: 8, seed: 42 });
    const second = generateSyntheticMovementDataset({ sequenceCount: 8, seed: 42 });
    expect(first).toEqual(second);
    expect(first.sequences).toHaveLength(8);
    const different = generateSyntheticMovementDataset({ sequenceCount: 8, seed: 7 });
    expect(JSON.stringify(different)).not.toEqual(JSON.stringify(first));
  });

  it("achieves high next-token fidelity on held-out synthetic motifs", () => {
    const train = generateSyntheticMovementDataset({ sequenceCount: 40, seed: 1, motifsPerSequence: 3 });
    const heldOut = generateSyntheticMovementDataset({ sequenceCount: 10, seed: 999, motifsPerSequence: 3 });
    const { inference } = trainMovementModel(train, new MarkovMovementBackend(3));
    const report = evaluateNextTokenFidelity(inference, heldOut.sequences);
    // The held-out sequences are novel concatenations of the same motifs, so a
    // trigram model should predict the vast majority of intra-motif steps.
    expect(report.total).toBeGreaterThan(0);
    expect(report.accuracy).toBeGreaterThan(0.6);
    expect(report.perSequence).toHaveLength(10);
  });
});

describe("tokenizers bridge capture -> dataset", () => {
  it("tokenizes a replay manifest's action events in order", () => {
    const span = buildTrajectorySpan({
      id: "t1",
      sessionId: "s1",
      actions: [
        { kind: "action", tool: "device", summary: "tapped A", ts: 2 },
        { kind: "action", tool: "device", summary: "tapped B", ts: 1 },
      ],
      observations: [{ kind: "observation", source: "os", summary: "focused App", ts: 0 }],
    });
    const manifest = buildReplayManifest({ sessionId: "s1", transcript: [], trajectories: [span] });
    const sequence = replayManifestToSequence(manifest);
    expect(sequence.tokens).toEqual(["act:device:tapped B", "act:device:tapped A"]);

    const withObs = replayManifestToSequence(manifest, { includeObservations: true });
    expect(withObs.tokens[0]).toBe("obs:os:focused App");
  });

  it("builds a dataset from multiple manifests and can train on it", () => {
    const makeSpan = (id: string, target: string) =>
      buildTrajectorySpan({
        id,
        sessionId: id,
        actions: [
          { kind: "action", tool: "device", summary: "tapped Menu", ts: 1 },
          { kind: "action", tool: "device", summary: `tapped ${target}`, ts: 2 },
        ],
      });
    const manifests = [
      buildReplayManifest({ sessionId: "s1", transcript: [], trajectories: [makeSpan("s1", "One")] }),
      buildReplayManifest({ sessionId: "s2", transcript: [], trajectories: [makeSpan("s2", "One")] }),
    ];
    const dataset = datasetFromReplayManifests(manifests);
    expect(dataset.sequences).toHaveLength(2);
    const { inference } = trainMovementModel(dataset);
    expect(inference.generate([])).toEqual(["act:device:tapped Menu", "act:device:tapped One"]);
  });

  it("orders interleaved observations and actions by timestamp when requested", () => {
    const span = buildTrajectorySpan({
      id: "t",
      sessionId: "s",
      observations: [{ kind: "observation", source: "os", summary: "focused", ts: 1 }],
      actions: [{ kind: "action", tool: "device", summary: "tapped", ts: 2 }],
    });
    const sequence = trajectoryToMovementSequence(span, { includeObservations: true });
    expect(sequence.tokens).toEqual(["obs:os:focused", "act:device:tapped"]);
  });
});
