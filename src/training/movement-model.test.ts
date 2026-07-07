import { describe, expect, it } from "vitest";
import { buildTrajectorySpan } from "../capture/trajectory.js";
import { buildReplayManifest } from "../capture/replay.js";
import {
  MOVEMENT_END_TOKEN,
  MOVEMENT_START_TOKEN,
  MarkovMovementBackend,
  buildMovementDatasetFromReplays,
  buildMovementDatasetFromTrajectories,
  evaluateMovementModel,
  generateMovementSequence,
  generateSyntheticMovementDataset,
  slugifyMovementSummary,
  splitMovementDataset,
  tokenizeMovementAction,
} from "./movement-model.js";

describe("movement tokenization", () => {
  it("produces a stable tool:slug token", () => {
    expect(tokenizeMovementAction({ tool: "Browser", summary: "Clicked Deploy!" })).toBe("browser:clicked-deploy");
    expect(tokenizeMovementAction({ tool: "device", summary: "  swiped   up  " })).toBe("device:swiped-up");
  });

  it("falls back to the tool when the summary slugifies to empty", () => {
    expect(tokenizeMovementAction({ tool: "keyboard", summary: "!!!" })).toBe("keyboard");
    expect(slugifyMovementSummary("--Hi There--")).toBe("hi-there");
  });
});

describe("dataset builders", () => {
  it("orders trajectory actions by timestamp", () => {
    const trajectory = buildTrajectorySpan({
      id: "traj-1",
      sessionId: "sess-1",
      actions: [
        { kind: "action", tool: "browser", summary: "clicked submit", ts: 30 },
        { kind: "action", tool: "browser", summary: "opened page", ts: 10 },
        { kind: "action", tool: "keyboard", summary: "typed message", ts: 20 },
      ],
    });
    const dataset = buildMovementDatasetFromTrajectories([trajectory]);
    expect(dataset.sequences).toEqual([
      {
        trajectoryId: "traj-1",
        tokens: ["browser:opened-page", "keyboard:typed-message", "browser:clicked-submit"],
      },
    ]);
  });

  it("derives sequences from replay manifest action events", () => {
    const trajectory = buildTrajectorySpan({
      id: "traj-9",
      sessionId: "sess-9",
      actions: [
        { kind: "action", tool: "device", summary: "tapped icon", ts: 5 },
        { kind: "action", tool: "device", summary: "confirmed action", ts: 8 },
      ],
    });
    const replay = buildReplayManifest({ sessionId: "sess-9", transcript: [], trajectories: [trajectory] });
    const dataset = buildMovementDatasetFromReplays([replay]);
    expect(dataset.sequences).toEqual([
      { trajectoryId: "traj-9", tokens: ["device:tapped-icon", "device:confirmed-action"] },
    ]);
  });

  it("skips trajectories with no actions", () => {
    const empty = buildTrajectorySpan({ id: "t", sessionId: "s", actions: [] });
    expect(buildMovementDatasetFromTrajectories([empty]).sequences).toEqual([]);
  });
});

describe("MarkovMovementBackend — repeat recorded movements", () => {
  it("reproduces a recorded sequence exactly via free generation", async () => {
    const dataset = {
      version: 1 as const,
      sequences: [{ tokens: ["a:open", "b:act", "c:confirm"] }],
    };
    const backend = new MarkovMovementBackend();
    const artifact = await backend.train(dataset, { order: 2 });

    expect(artifact.backendId).toBe("markov");
    expect(artifact.vocabulary).toEqual(["a:open", "b:act", "c:confirm"]);
    expect(artifact.sequenceCount).toBe(1);

    const result = await generateMovementSequence(backend, artifact);
    expect(result.completed).toBe(true);
    expect(result.tokens).toEqual(["a:open", "b:act", "c:confirm"]);
  });

  it("predicts the first movement from the START context", async () => {
    const backend = new MarkovMovementBackend();
    const artifact = await backend.train({ version: 1, sequences: [{ tokens: ["x:go", "y:stop"] }] });
    const prediction = backend.predictNext(artifact, []);
    expect(prediction?.token).toBe("x:go");
    expect(prediction?.confidence).toBe(1);
  });

  it("emits END after the last recorded token", async () => {
    const backend = new MarkovMovementBackend();
    const artifact = await backend.train({ version: 1, sequences: [{ tokens: ["only:one"] }] });
    const prediction = backend.predictNext(artifact, ["only:one"]);
    expect(prediction?.token).toBe(MOVEMENT_END_TOKEN);
  });
});

describe("MarkovMovementBackend — generalize to related movements", () => {
  it("backs off to a shorter context for an unseen prefix", async () => {
    // Two sequences share the sub-path "b:act" -> "c:confirm". A novel prefix
    // ending in "b:act" should still predict "c:confirm" via back-off.
    const dataset = {
      version: 1 as const,
      sequences: [
        { tokens: ["a:open", "b:act", "c:confirm"] },
        { tokens: ["z:launch", "b:act", "c:confirm"] },
      ],
    };
    const backend = new MarkovMovementBackend();
    const artifact = await backend.train(dataset, { order: 2 });

    // Context ["novel:prefix", "b:act"] was never seen at order 2, but the
    // order-1 context ["b:act"] -> "c:confirm" was.
    const prediction = backend.predictNext(artifact, ["novel:prefix", "b:act"]);
    expect(prediction?.token).toBe("c:confirm");
    expect(prediction?.order).toBe(1);
  });

  it("is deterministic across repeated training + generation", async () => {
    const dataset = generateSyntheticMovementDataset({ seed: 7, sequenceCount: 8 });
    const backend = new MarkovMovementBackend();
    const [a, b] = await Promise.all([backend.train(dataset), backend.train(dataset)]);
    expect(a).toEqual(b);
    const genA = await generateMovementSequence(backend, a);
    const genB = await generateMovementSequence(backend, b);
    expect(genA).toEqual(genB);
  });

  it("halts at maxSteps even without an END transition", async () => {
    // A self-loop sequence: "loop" repeats, so END is unlikely early.
    const backend = new MarkovMovementBackend();
    const artifact = await backend.train(
      { version: 1, sequences: [{ tokens: ["loop", "loop", "loop", "loop"] }] },
      { order: 1 },
    );
    const result = await generateMovementSequence(backend, artifact, { maxSteps: 3 });
    expect(result.tokens.length).toBeLessThanOrEqual(3);
  });
});

describe("synthetic generator + eval harness", () => {
  it("generates a reproducible dataset for a given seed", () => {
    const a = generateSyntheticMovementDataset({ seed: 42, sequenceCount: 5 });
    const b = generateSyntheticMovementDataset({ seed: 42, sequenceCount: 5 });
    expect(a).toEqual(b);
    expect(a.sequences).toHaveLength(5);
    expect(a.sequences.every((seq) => seq.tokens.length >= 1)).toBe(true);
  });

  it("splits deterministically into train / held-out folds", () => {
    const dataset = generateSyntheticMovementDataset({ seed: 3, sequenceCount: 8 });
    const { train, heldOut } = splitMovementDataset(dataset, 4);
    expect(train.sequences.length + heldOut.sequences.length).toBe(8);
    expect(heldOut.sequences.length).toBe(2);
  });

  it("achieves high next-token accuracy on held-out related trajectories", async () => {
    const dataset = generateSyntheticMovementDataset({ seed: 11, sequenceCount: 40 });
    const { train, heldOut } = splitMovementDataset(dataset, 4);
    const backend = new MarkovMovementBackend();
    const artifact = await backend.train(train, { order: 2 });
    const report = await evaluateMovementModel(backend, artifact, heldOut);

    expect(report.sequenceCount).toBeGreaterThan(0);
    // The synthetic grammar shares sub-paths, so back-off should predict most
    // held-out movements correctly — well above chance for the vocab size.
    expect(report.nextTokenAccuracy).toBeGreaterThan(0.5);
    expect(report.meanPrefixMatch).toBeGreaterThan(0.3);
  });

  it("reports zeroed metrics for an empty held-out set", async () => {
    const backend = new MarkovMovementBackend();
    const artifact = await backend.train(generateSyntheticMovementDataset({ seed: 1 }));
    const report = await evaluateMovementModel(backend, artifact, { version: 1, sequences: [] });
    expect(report).toEqual({
      sequenceCount: 0,
      tokenCount: 0,
      nextTokenAccuracy: 0,
      exactSequenceMatch: 0,
      meanPrefixMatch: 0,
    });
  });
});

describe("start/end sentinels", () => {
  it("exposes stable sentinel values", () => {
    expect(MOVEMENT_START_TOKEN).toBe("START");
    expect(MOVEMENT_END_TOKEN).toBe("END");
  });
});
