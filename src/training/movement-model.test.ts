import { describe, expect, it } from "vitest";
import type { ExportedReplayManifest } from "./export-manifest.js";
import {
  NgramMovementBackend,
  buildMovementDatasetFromReplays,
  evaluateMovementModel,
  tokenizeActionSummary,
  type MovementDataset,
  type MovementModelBackend,
  type MovementSequence,
  type MovementPrediction,
  type TrainedMovementModel,
} from "./movement-model.js";

function dataset(sequences: MovementSequence[]): MovementDataset {
  return { version: 1, sequences };
}

describe("NgramMovementBackend training", () => {
  it("builds a sorted vocabulary that excludes the end sentinel", async () => {
    const model = await new NgramMovementBackend().train(
      dataset([{ id: "t1", tokens: ["open:app", "tap:search", "type:query"] }]),
    );
    expect(model.vocabulary).toEqual(["open:app", "tap:search", "type:query"]);
    expect(model.backendId).toBe("ngram-v1");
    expect(model.vocabulary.some((token) => token.includes("end"))).toBe(false);
  });
});

describe("memorization / replay (objective 2c)", () => {
  it("reproduces a recorded movement run exactly from its opening move", async () => {
    const tokens = ["open:app", "tap:search", "type:query", "tap:submit"];
    const model = await new NgramMovementBackend().train(dataset([{ id: "t1", tokens }]));

    const replayed = model.generate([tokens[0]!], { maxSteps: 10 });
    expect(replayed).toEqual(tokens.slice(1));
  });

  it("predicts the recorded next move at full order without back-off", async () => {
    const tokens = ["open:app", "tap:search", "type:query", "tap:submit"];
    const model = await new NgramMovementBackend().train(dataset([{ id: "t1", tokens }]));

    const prediction = model.predictNext(["open:app", "tap:search"]);
    expect(prediction.token).toBe("type:query");
    expect(prediction.fromBackoff).toBe(false);
    expect(prediction.order).toBe(2);
    expect(prediction.confidence).toBeCloseTo(1);
  });

  it("predicts end (undefined) at the true end of a memorized run", async () => {
    const tokens = ["open:app", "tap:submit"];
    const model = await new NgramMovementBackend().train(dataset([{ id: "t1", tokens }]));
    expect(model.predictNext(tokens).token).toBeUndefined();
  });
});

describe("generalization via back-off (objective 2d)", () => {
  it("emits a plausible related move for an unseen prefix by backing off", async () => {
    const model = await new NgramMovementBackend().train(
      dataset([
        { id: "t1", tokens: ["open:mail", "tap:compose", "tap:send"] },
        { id: "t2", tokens: ["open:chat", "tap:compose", "tap:send"] },
      ]),
    );

    // "open:notes" was never recorded, but "tap:compose" is always followed by
    // "tap:send" — the model should generalize the continuation.
    const prediction = model.predictNext(["open:notes", "tap:compose"]);
    expect(prediction.token).toBe("tap:send");
    expect(prediction.fromBackoff).toBe(true);
    expect(prediction.order).toBeLessThan(2);
  });

  it("falls back to global frequency when no context matches at all", async () => {
    const model = await new NgramMovementBackend().train(
      dataset([{ id: "t1", tokens: ["tap:a", "tap:b", "tap:b"] }]),
    );
    const prediction = model.predictNext(["totally:unseen"]);
    expect(prediction.token).toBe("tap:b"); // most frequent token overall
    expect(prediction.order).toBe(0);
    expect(prediction.fromBackoff).toBe(true);
  });
});

describe("determinism", () => {
  it("breaks probability ties lexicographically and is repeatable", async () => {
    const backend = new NgramMovementBackend();
    const data = dataset([
      { id: "t1", tokens: ["x", "q"] },
      { id: "t2", tokens: ["x", "p"] },
    ]);
    const a = (await backend.train(data)).predictNext(["x"]);
    const b = (await backend.train(data)).predictNext(["x"]);
    expect(a).toEqual(b);
    expect(a.token).toBe("p");
    expect(a.candidates.map((candidate) => candidate.token)).toEqual(["p", "q"]);
  });
});

describe("evaluateMovementModel (generalization eval harness)", () => {
  it("scores held-out sequences and splits memorized vs generalized hits", async () => {
    const backend = new NgramMovementBackend();
    const train = dataset([
      { id: "a", tokens: ["open:app", "tap:menu", "tap:settings"] },
      { id: "b", tokens: ["open:home", "tap:menu", "tap:settings"] },
    ]);
    const model = await backend.train(train, { maxOrder: 3 });

    // Held-out but related: same "tap:menu" hub, novel opener. The model has
    // never seen "open:widget", so tokens[2] can only be reached by backing off.
    const heldOut: MovementSequence[] = [{ id: "h", tokens: ["open:widget", "tap:menu", "tap:settings"] }];
    const result = evaluateMovementModel(model, heldOut);

    expect(result.predictions).toBe(2); // predict tokens[1] and tokens[2]
    expect(result.correct).toBeGreaterThan(0);
    expect(result.accuracy).toBeGreaterThan(0);
    expect(result.memorizedCorrect + result.backoffCorrect).toBe(result.correct);
    expect(result.generalizationRate).toBeGreaterThan(0); // at least one hit came from back-off
    expect(result.averageConfidence).toBeGreaterThan(0);
  });

  it("reports zeroed metrics for an empty held-out set", async () => {
    const model = await new NgramMovementBackend().train(dataset([{ id: "t1", tokens: ["a", "b"] }]));
    const result = evaluateMovementModel(model, []);
    expect(result).toMatchObject({ predictions: 0, correct: 0, accuracy: 0, generalizationRate: 0 });
  });
});

describe("buildMovementDatasetFromReplays", () => {
  it("extracts one time-ordered action sequence per trajectory, ignoring non-actions", () => {
    const replays: ExportedReplayManifest[] = [
      {
        sessionId: "s1",
        trajectoryIds: ["traj-b", "traj-a"],
        eventCount: 5,
        events: [
          { kind: "transcript", ts: 1, messageId: "m1", role: "user", content: "hi" },
          { kind: "action", ts: 30, trajectoryId: "traj-a", tool: "device", summary: "tapped submit button" },
          { kind: "observation", ts: 5, trajectoryId: "traj-a", source: "device", summary: "screen" },
          { kind: "action", ts: 10, trajectoryId: "traj-a", tool: "device", summary: "opened app" },
          { kind: "action", ts: 20, trajectoryId: "traj-b", tool: "device", summary: "scrolled down" },
        ],
      },
    ];

    const built = buildMovementDatasetFromReplays(replays);
    expect(built.sequences.map((sequence) => sequence.id)).toEqual(["traj-a", "traj-b"]);
    const trajA = built.sequences.find((sequence) => sequence.id === "traj-a");
    expect(trajA?.tokens).toEqual(["device:opened-app", "device:tapped-submit-button"]);
  });

  it("tokenizes summaries into stable slugs", () => {
    expect(tokenizeActionSummary("device", "Tapped Submit-Button!")).toBe("device:tapped-submit-button");
    expect(tokenizeActionSummary("browser", "")).toBe("browser:action");
  });
});

describe("pluggable backend seam", () => {
  it("lets an alternate backend feed the same eval harness", async () => {
    // A trivial backend proving the interface is enough to swap in another model
    // (e.g. a real on-device neural policy) without touching the harness.
    class ConstantBackend implements MovementModelBackend {
      readonly id = "constant-test";
      async train(data: MovementDataset): Promise<TrainedMovementModel> {
        const token = data.sequences[0]?.tokens[0];
        const vocabulary = [...new Set(data.sequences.flatMap((sequence) => sequence.tokens))].sort();
        const predict = (): MovementPrediction => ({
          token,
          confidence: 1,
          order: 0,
          fromBackoff: false,
          candidates: token ? [{ token, probability: 1 }] : [],
        });
        return {
          backendId: this.id,
          maxOrder: 0,
          vocabulary,
          predictNext: predict,
          generate: () => (token ? [token] : []),
        };
      }
    }

    const model = await new ConstantBackend().train(dataset([{ id: "t1", tokens: ["tap:go", "tap:go"] }]));
    const result = evaluateMovementModel(model, [{ id: "h", tokens: ["tap:go", "tap:go"] }]);
    expect(model.backendId).toBe("constant-test");
    expect(result.correct).toBe(1); // predicts "tap:go" for the second position
  });
});
