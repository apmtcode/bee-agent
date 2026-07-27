import { describe, expect, it } from "vitest";
import { buildTrajectorySpan } from "../capture/trajectory.js";
import { MarkovMovementBackend } from "./movement-model.js";
import {
  buildMovementDatasetFromReplays,
  buildMovementDatasetFromTrajectories,
  evaluateMovementGeneralization,
  normalizeMovementVerb,
  trajectoryToMovements,
} from "./movement-dataset.js";
import { DEFAULT_MOVEMENT_TASKS, generateSyntheticMovements } from "./movement-synth.js";

describe("normalizeMovementVerb", () => {
  it("reduces a summary to a stable single-verb token component", () => {
    expect(normalizeMovementVerb("Clicked the Submit button")).toBe("clicked");
    expect(normalizeMovementVerb("  scrolled DOWN  ")).toBe("scrolled");
    expect(normalizeMovementVerb("")).toBe("act");
  });
});

describe("buildMovementDatasetFromTrajectories", () => {
  it("orders events by ts and derives canonical tokens", () => {
    const trajectory = buildTrajectorySpan({
      id: "traj-1",
      sessionId: "sess-1",
      observations: [{ kind: "observation", source: "os", summary: "focused editor", ts: 10 }],
      actions: [
        { kind: "action", tool: "keyboard", summary: "typed hello", ts: 30 },
        { kind: "action", tool: "mouse", summary: "clicked save", ts: 20 },
      ],
    });
    const dataset = buildMovementDatasetFromTrajectories([trajectory], { createdAt: "2026-07-27T00:00:00.000Z" });

    expect(dataset.sequences).toHaveLength(1);
    expect(dataset.sequences[0]?.tokens).toEqual([
      "observation:os:focused",
      "action:mouse:clicked",
      "action:keyboard:typed",
    ]);
    expect(dataset.tokenCount).toBe(3);
    expect(dataset.vocabulary).toContain("action:mouse:clicked");
    expect(dataset.vocabulary).toContain("start");
    expect(dataset.vocabulary).toContain("end");
  });

  it("drops sequences below the minimum length", () => {
    const empty = buildTrajectorySpan({ id: "empty", sessionId: "s" });
    const dataset = buildMovementDatasetFromTrajectories([empty], { minSequenceLength: 1 });
    expect(dataset.sequences).toHaveLength(0);
  });
});

describe("buildMovementDatasetFromReplays", () => {
  it("ignores transcript events and keeps observation/action movements", () => {
    const dataset = buildMovementDatasetFromReplays([
      {
        id: "replay-1",
        trajectoryIds: ["traj-1"],
        events: [
          { kind: "transcript", ts: 5, messageId: "m1", role: "user", content: "go" },
          { kind: "observation", ts: 10, trajectoryId: "traj-1", source: "browser", summary: "opened page" },
          { kind: "action", ts: 20, trajectoryId: "traj-1", tool: "mouse", summary: "clicked link" },
        ],
      },
    ]);
    expect(dataset.sequences[0]?.tokens).toEqual(["observation:browser:opened", "action:mouse:clicked"]);
  });
});

describe("trajectoryToMovements", () => {
  it("interleaves observations and actions by timestamp", () => {
    const trajectory = buildTrajectorySpan({
      id: "traj-1",
      sessionId: "sess-1",
      observations: [{ kind: "observation", source: "app", summary: "confirmed dialog", ts: 40 }],
      actions: [{ kind: "action", tool: "mouse", summary: "clicked ok", ts: 30 }],
    });
    expect(trajectoryToMovements(trajectory).map((event) => event.ts)).toEqual([30, 40]);
  });
});

describe("movement generalization pipeline (synthetic)", () => {
  it("round-trips capture → dataset → train → generalize on held-out streams", async () => {
    // Train and eval on disjoint synthetic streams drawn from the SAME task
    // grammar: held-out sequences are new but related, so a model that
    // generalizes (not memorizes) should predict most next-moves correctly.
    const trainStreams = generateSyntheticMovements({ seed: 7, sequenceCount: 40 });
    const evalStreams = generateSyntheticMovements({ seed: 999, sequenceCount: 16 });

    const trainDataset = buildMovementDatasetFromReplays(
      trainStreams.map((stream) => ({ id: stream.id, events: stream.events })),
    );
    const evalDataset = buildMovementDatasetFromReplays(
      evalStreams.map((stream) => ({ id: stream.id, events: stream.events })),
    );

    const model = await new MarkovMovementBackend().train(trainDataset, { order: 2 });
    const report = evaluateMovementGeneralization(model, evalDataset.sequences);

    // The task grammar is deterministic given the leading verb, so once the
    // model has seen every task it should predict held-out next-moves almost
    // perfectly — this is the generalization objective, measured.
    expect(report.sequenceCount).toBe(evalDataset.sequences.length);
    expect(report.accuracy).toBeGreaterThan(0.8);
    // Vocabulary is closed over the task set (no unseen tokens leak in).
    expect(trainDataset.vocabulary.length).toBeGreaterThanOrEqual(DEFAULT_MOVEMENT_TASKS.length);
  });

  it("generates streams deterministically for a given seed", () => {
    expect(generateSyntheticMovements({ seed: 42, sequenceCount: 3 })).toEqual(
      generateSyntheticMovements({ seed: 42, sequenceCount: 3 }),
    );
    // Different seeds diverge.
    const a = JSON.stringify(generateSyntheticMovements({ seed: 1, sequenceCount: 3 }));
    const b = JSON.stringify(generateSyntheticMovements({ seed: 2, sequenceCount: 3 }));
    expect(a).not.toEqual(b);
  });
});
