import { describe, expect, it } from "vitest";
import type { ReplayTimelineEvent } from "../capture/replay.js";
import type { TrajectorySpan } from "../capture/trajectory.js";
import {
  BOUNDARY_START,
  buildMovementDatasetFromReplays,
  buildMovementDatasetFromTrajectories,
  createMovementModelBackend,
  decodeMovementToken,
  encodeMovementToken,
  NgramMovementBackend,
  restoreMovementModel,
  type MovementDataset,
  type MovementSequence,
  type ReplaySource,
} from "./model-backend.js";

function actionEvent(ts: number, tool: string, summary: string): ReplayTimelineEvent {
  return { kind: "action", ts, trajectoryId: "traj", tool, summary };
}

function observationEvent(ts: number, source: string, summary: string): ReplayTimelineEvent {
  return { kind: "observation", ts, trajectoryId: "traj", source, summary };
}

const openReplay: ReplaySource = {
  trajectoryIds: ["traj-open"],
  events: [
    observationEvent(1, "browser", "editor focused"),
    actionEvent(2, "keyboard", "cmd+p"),
    actionEvent(3, "keyboard", "type deploy.ts"),
    actionEvent(4, "keyboard", "enter"),
    actionEvent(5, "mouse", "click run"),
  ],
};

describe("movement tokenization", () => {
  it("round-trips action and observation tokens", () => {
    const action = { kind: "action", tool: "keyboard", summary: "cmd+p" } as const;
    const observation = { kind: "observation", source: "browser", summary: "editor focused" } as const;
    expect(decodeMovementToken(encodeMovementToken(action))).toEqual(action);
    expect(decodeMovementToken(encodeMovementToken(observation))).toEqual(observation);
  });

  it("returns undefined for boundary sentinels", () => {
    expect(decodeMovementToken(BOUNDARY_START)).toBeUndefined();
  });
});

describe("buildMovementDatasetFromReplays", () => {
  it("tokenizes events and collects a sorted vocabulary", () => {
    const dataset = buildMovementDatasetFromReplays([openReplay]);
    expect(dataset.sequences).toHaveLength(1);
    expect(dataset.sequences[0]!.tokens).toHaveLength(5);
    expect(dataset.vocabulary).toEqual([...dataset.vocabulary].sort());
    expect(dataset.vocabulary).toContain(BOUNDARY_START);
  });

  it("can exclude observations from the token stream", () => {
    const dataset = buildMovementDatasetFromReplays([openReplay], { includeObservations: false });
    expect(dataset.sequences[0]!.tokens).toHaveLength(4);
    expect(dataset.sequences[0]!.tokens.every((token) => token.startsWith("act"))).toBe(true);
  });
});

describe("NgramMovementBackend — reproduce (objective 2c)", () => {
  it("regenerates a recorded trajectory exactly from its start context", async () => {
    const dataset = buildMovementDatasetFromReplays([openReplay]);
    const model = await new NgramMovementBackend().train(dataset, { order: 3 });
    const startContext = Array.from({ length: model.order - 1 }, () => BOUNDARY_START);

    const generated = model.generate(startContext);
    expect(generated).toEqual(dataset.sequences[0]!.tokens);
  });

  it("predicts the exact next movement given a seen prefix", async () => {
    const dataset = buildMovementDatasetFromReplays([openReplay]);
    const model = await new NgramMovementBackend().train(dataset, { order: 3 });

    const prefix = dataset.sequences[0]!.tokens.slice(0, 3);
    const prediction = model.predictNext(prefix);
    expect(prediction.token).toBe(dataset.sequences[0]!.tokens[3]);
    expect(prediction.probability).toBeGreaterThan(0);
  });

  it("is deterministic: repeated training yields identical predictions", async () => {
    const dataset = buildMovementDatasetFromReplays([openReplay]);
    const backend = new NgramMovementBackend();
    const a = await backend.train(dataset);
    const b = await backend.train(dataset);
    const prefix = dataset.sequences[0]!.tokens.slice(0, 2);
    expect(a.predictNext(prefix)).toEqual(b.predictNext(prefix));
  });
});

describe("NgramMovementBackend — generalize (objective 2d)", () => {
  it("predicts a plausible next movement for an unseen-but-related prefix via backoff", async () => {
    // Two related trajectories share the "type X -> enter -> click run" tail.
    const replayA: ReplaySource = {
      trajectoryIds: ["a"],
      events: [
        actionEvent(1, "keyboard", "cmd+p"),
        actionEvent(2, "keyboard", "type deploy.ts"),
        actionEvent(3, "keyboard", "enter"),
        actionEvent(4, "mouse", "click run"),
      ],
    };
    const replayB: ReplaySource = {
      trajectoryIds: ["b"],
      events: [
        actionEvent(1, "keyboard", "cmd+p"),
        actionEvent(2, "keyboard", "type build.ts"),
        actionEvent(3, "keyboard", "enter"),
        actionEvent(4, "mouse", "click run"),
      ],
    };
    const dataset = buildMovementDatasetFromReplays([replayA, replayB], { includeObservations: false });
    const model = await new NgramMovementBackend().train(dataset, { order: 3 });

    // A brand-new file name never seen in training, but the same "enter" step.
    const novelPrefix = [
      encodeMovementToken({ kind: "action", tool: "keyboard", summary: "type migrate.ts" }),
      encodeMovementToken({ kind: "action", tool: "keyboard", summary: "enter" }),
    ];
    const prediction = model.predictNext(novelPrefix);
    // Highest-order context is unseen; backoff to "enter" -> the shared "click run".
    expect(prediction.token).toBe(
      encodeMovementToken({ kind: "action", tool: "mouse", summary: "click run" }),
    );
    expect(prediction.matchedOrder).toBeLessThan(model.order - 1);
  });

  it("weights higher-reward trajectories more strongly", async () => {
    const base: Omit<TrajectorySpan, "id" | "outcome"> = {
      sessionId: "s",
      createdAt: "2026-01-01T00:00:00.000Z",
      captureTier: "operator",
      observations: [],
      actions: [],
    };
    const success: TrajectorySpan = {
      ...base,
      id: "win",
      actions: [
        { kind: "action", tool: "keyboard", summary: "start", ts: 1 },
        { kind: "action", tool: "mouse", summary: "approve", ts: 2 },
      ],
      outcome: { status: "success", summary: "ok", reward: 5 },
    };
    const failure: TrajectorySpan = {
      ...base,
      id: "lose",
      actions: [
        { kind: "action", tool: "keyboard", summary: "start", ts: 1 },
        { kind: "action", tool: "mouse", summary: "cancel", ts: 2 },
      ],
      outcome: { status: "failure", summary: "no", reward: 0 },
    };
    const dataset = buildMovementDatasetFromTrajectories([success, failure]);
    const model = await new NgramMovementBackend().train(dataset, { order: 2 });

    const prediction = model.predictNext([
      encodeMovementToken({ kind: "action", tool: "keyboard", summary: "start" }),
    ]);
    // "approve" (reward 5) outweighs "cancel" (reward 0 -> weight 1).
    expect(prediction.token).toBe(
      encodeMovementToken({ kind: "action", tool: "mouse", summary: "approve" }),
    );
  });
});

describe("model persistence + registry", () => {
  it("serializes and restores to an identical model", async () => {
    const dataset = buildMovementDatasetFromReplays([openReplay]);
    const model = await new NgramMovementBackend().train(dataset, { order: 3 });
    const snapshot = model.serialize();

    // Snapshot must be plain-JSON serializable.
    const roundTripped = restoreMovementModel(JSON.parse(JSON.stringify(snapshot)));
    const prefix = dataset.sequences[0]!.tokens.slice(0, 2);
    expect(roundTripped.predictNext(prefix)).toEqual(model.predictNext(prefix));
    expect(roundTripped.generate(Array.from({ length: model.order - 1 }, () => BOUNDARY_START))).toEqual(
      model.generate(Array.from({ length: model.order - 1 }, () => BOUNDARY_START)),
    );
  });

  it("createMovementModelBackend returns the deterministic ngram backend", () => {
    expect(createMovementModelBackend().id).toBe("ngram");
    expect(createMovementModelBackend("ngram")).toBeInstanceOf(NgramMovementBackend);
  });

  it("rejects unknown backend snapshots", () => {
    const bad = { backendId: "mystery", version: 1, order: 2, counts: {}, vocabulary: [] };
    expect(() => restoreMovementModel(bad as never)).toThrow(/Unknown movement model backend/);
  });

  it("guards against invalid n-gram order", async () => {
    const dataset: MovementDataset = { version: 1, sequences: [] as MovementSequence[], vocabulary: [] };
    await expect(new NgramMovementBackend().train(dataset, { order: 0 })).rejects.toThrow(/positive integer/);
  });
});
