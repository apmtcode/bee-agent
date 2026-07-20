import { describe, expect, it } from "vitest";
import {
  DeterministicMarkovMovementBackend,
  MarkovMovementModel,
  buildMovementDataset,
  evaluateMovementModel,
  tokenizeReplayEvents,
  tokenizeTrajectory,
  type MovementDataset,
} from "./movement-model.js";
import {
  SYNTHETIC_TASK_FAMILIES,
  generateSyntheticTrajectories,
} from "./synthetic-movements.js";
import { buildTrajectorySpan } from "../capture/trajectory.js";
import type { ReplayTimelineEvent } from "../capture/replay.js";

function span(id: string, tokens: { tool: string; ts: number }[]) {
  return buildTrajectorySpan({
    id,
    sessionId: "s",
    actions: tokens.map((t) => ({ kind: "action" as const, tool: t.tool, summary: t.tool, ts: t.ts })),
  });
}

describe("tokenization", () => {
  it("orders a trajectory's actions and observations by timestamp", () => {
    const trajectory = buildTrajectorySpan({
      id: "t1",
      sessionId: "s",
      actions: [{ kind: "action", tool: "Save File", summary: "save", ts: 30 }],
      observations: [{ kind: "observation", source: "Screen", summary: "focus", ts: 10 }],
    });
    const sequence = tokenizeTrajectory(trajectory);
    expect(sequence.tokens).toEqual(["obs:screen", "act:save-file"]);
  });

  it("can exclude observations", () => {
    const trajectory = buildTrajectorySpan({
      id: "t1",
      sessionId: "s",
      actions: [{ kind: "action", tool: "click", summary: "c", ts: 20 }],
      observations: [{ kind: "observation", source: "screen", summary: "o", ts: 10 }],
    });
    expect(tokenizeTrajectory(trajectory, { includeObservations: false }).tokens).toEqual(["act:click"]);
  });

  it("tokenizes replay timeline events, skipping transcripts by default", () => {
    const events: ReplayTimelineEvent[] = [
      { kind: "transcript", ts: 1, messageId: "m", role: "user", content: "hi" },
      { kind: "action", ts: 2, trajectoryId: "t", tool: "click", summary: "c" },
      { kind: "observation", ts: 3, trajectoryId: "t", source: "screen", summary: "o" },
    ];
    expect(tokenizeReplayEvents("t", events).tokens).toEqual(["act:click", "obs:screen"]);
    expect(tokenizeReplayEvents("t", events, { includeTranscript: true }).tokens).toEqual([
      "msg:user",
      "act:click",
      "obs:screen",
    ]);
  });
});

describe("DeterministicMarkovMovementBackend", () => {
  const backend = new DeterministicMarkovMovementBackend();

  it("repeats a recorded movement sequence verbatim (objective 2c)", async () => {
    const dataset = buildMovementDataset([
      span("t1", [
        { tool: "focus", ts: 1 },
        { tool: "type", ts: 2 },
        { tool: "save", ts: 3 },
        { tool: "close", ts: 4 },
      ]),
    ]);
    const model = await backend.train(dataset);
    const generated = model.generate(["act:focus"], { maxSteps: 10 });
    expect(generated).toEqual(["act:type", "act:save", "act:close"]);
  });

  it("generalizes to a novel-but-related prefix via backoff (objective 2d)", async () => {
    // Two related recordings share the "type -> save" transition, differ in how they start.
    const dataset: MovementDataset = {
      version: 1,
      sequences: [
        { trajectoryId: "a", tokens: ["act:open", "act:type", "act:save"] },
        { trajectoryId: "b", tokens: ["act:launch", "act:type", "act:save"] },
      ],
    };
    const model = await backend.train(dataset, { order: 3 });
    // This exact 2-gram context ("resume" -> "type") was never seen, but the
    // lower-order context ("type" -> ?) was. Backoff should still predict "save".
    const prediction = model.predictNext(["act:resume", "act:type"]);
    expect(prediction?.token).toBe("act:save");
    expect(prediction?.order).toBeLessThan(2); // came from a backed-off context
  });

  it("returns undefined when there is no data at all", async () => {
    const model = await backend.train({ version: 1, sequences: [] });
    expect(model.predictNext(["act:anything"])).toBeUndefined();
    expect(model.generate(["act:anything"])).toEqual([]);
  });

  it("breaks ties deterministically (count desc, then lexicographic)", async () => {
    const dataset: MovementDataset = {
      version: 1,
      sequences: [
        { trajectoryId: "a", tokens: ["act:x", "act:zeta"] },
        { trajectoryId: "b", tokens: ["act:x", "act:alpha"] },
      ],
    };
    const model = await backend.train(dataset, { order: 1 });
    // Equal counts -> lexicographically smaller token wins.
    expect(model.predictNext(["act:x"])?.token).toBe("act:alpha");
  });

  it("round-trips through serialize/deserialize", async () => {
    const dataset = buildMovementDataset([
      span("t1", [
        { tool: "a", ts: 1 },
        { tool: "b", ts: 2 },
        { tool: "c", ts: 3 },
      ]),
    ]);
    const model = await backend.train(dataset, { order: 2 });
    const restored = MarkovMovementModel.deserialize(model.serialize());
    expect(restored.generate(["act:a"], { maxSteps: 5 })).toEqual(model.generate(["act:a"], { maxSteps: 5 }));
    expect(restored.order).toBe(2);
  });

  it("stops generation at the learned end of a sequence (EOS sentinel)", async () => {
    const dataset = buildMovementDataset([
      span("t1", [
        { tool: "a", ts: 1 },
        { tool: "b", ts: 2 },
        { tool: "c", ts: 3 },
      ]),
    ]);
    const model = await backend.train(dataset);
    // No maxSteps clamp needed — the model learned to stop after "c".
    expect(model.generate(["act:a"], { maxSteps: 50 })).toEqual(["act:b", "act:c"]);
  });

  it("honors an explicit stopToken that fires before the natural end", async () => {
    const dataset = buildMovementDataset([
      span("t1", [
        { tool: "x", ts: 1 },
        { tool: "y", ts: 2 },
        { tool: "z", ts: 3 },
        { tool: "w", ts: 4 },
      ]),
    ]);
    const model = await backend.train(dataset);
    expect(model.generate(["act:x"], { stopToken: "act:z" })).toEqual(["act:y", "act:z"]);
  });
});

describe("generalization eval on synthetic trajectories", () => {
  it("generalizes: a same-family model reproduces held-out variants far better than a cross-family baseline", async () => {
    const family = SYNTHETIC_TASK_FAMILIES.fileSave!;
    const other = SYNTHETIC_TASK_FAMILIES.browserSearch!;
    const heldOut = generateSyntheticTrajectories({ family, seed: 999, count: 4, idPrefix: "holdout" }).map((t) =>
      tokenizeTrajectory(t),
    );

    const backend = new DeterministicMarkovMovementBackend();
    const inFamily = await backend.train(
      buildMovementDataset(generateSyntheticTrajectories({ family, seed: 1, count: 16 })),
      { order: 3 },
    );
    const crossFamily = await backend.train(
      buildMovementDataset(generateSyntheticTrajectories({ family: other, seed: 1, count: 16 })),
      { order: 3 },
    );

    const inFamilyEval = evaluateMovementModel(inFamily, heldOut);
    const crossFamilyEval = evaluateMovementModel(crossFamily, heldOut);

    expect(inFamilyEval.sequenceCount).toBe(4);
    // Learning the family's shared movement spine transfers to unseen variants;
    // a model that never saw the family should do much worse.
    expect(inFamilyEval.meanFidelity).toBeGreaterThan(0.5);
    expect(inFamilyEval.meanFidelity).toBeGreaterThan(crossFamilyEval.meanFidelity + 0.3);
  });

  it("is reproducible for a fixed seed", () => {
    const family = SYNTHETIC_TASK_FAMILIES.browserSearch!;
    const a = generateSyntheticTrajectories({ family, seed: 7, count: 3 });
    const b = generateSyntheticTrajectories({ family, seed: 7, count: 3 });
    expect(JSON.stringify(a.map((t) => tokenizeTrajectory(t).tokens))).toEqual(
      JSON.stringify(b.map((t) => tokenizeTrajectory(t).tokens)),
    );
  });
});
