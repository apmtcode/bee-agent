import { describe, expect, it } from "vitest";
import type { ReplayManifest } from "../capture/replay.js";
import {
  buildMovementDataset,
  contextFeatures,
  createMovementModelBackend,
  deserializeMovementPolicy,
  evaluateMovementGeneralization,
  extractMovementExamples,
  generateSyntheticMovementReplays,
  MockNgramMovementBackend,
  splitMovementReplays,
} from "./movement-model.js";

function manifest(events: ReplayManifest["events"], sessionId = "s1"): ReplayManifest {
  return {
    version: 1,
    sessionId,
    trajectoryIds: [sessionId],
    eventCount: events.length,
    events,
  };
}

describe("extractMovementExamples", () => {
  it("pairs each action with the current surface and previous action", () => {
    const examples = extractMovementExamples(
      manifest([
        { kind: "observation", ts: 1, trajectoryId: "t", source: "device", summary: "mail active" },
        { kind: "action", ts: 2, trajectoryId: "t", tool: "device", summary: "tapped compose" },
        { kind: "action", ts: 3, trajectoryId: "t", tool: "device", summary: "typed subject" },
      ]),
    );

    expect(examples).toHaveLength(2);
    expect(examples[0]?.context).toEqual({ surface: "device/mail" });
    expect(examples[0]?.action.label).toBe("device:tapped compose");
    expect(examples[1]?.context).toEqual({
      surface: "device/mail",
      previousAction: "device:tapped compose",
    });
  });

  it("ignores transcript events and tracks surface changes", () => {
    const examples = extractMovementExamples(
      manifest([
        { kind: "transcript", ts: 1, messageId: "m", role: "user", content: "go" },
        { kind: "observation", ts: 2, trajectoryId: "t", source: "os", summary: "editor active" },
        { kind: "action", ts: 3, trajectoryId: "t", tool: "os", summary: "focused buffer" },
      ]),
    );

    expect(examples).toHaveLength(1);
    expect(examples[0]?.context.surface).toBe("os/editor");
  });
});

describe("buildMovementDataset", () => {
  it("collects examples and a sorted action vocabulary", () => {
    const dataset = buildMovementDataset([
      manifest([
        { kind: "observation", ts: 1, trajectoryId: "t", source: "device", summary: "chat active" },
        { kind: "action", ts: 2, trajectoryId: "t", tool: "device", summary: "scrolled thread" },
        { kind: "action", ts: 3, trajectoryId: "t", tool: "device", summary: "tapped thread" },
      ]),
    ]);

    expect(dataset.examples).toHaveLength(2);
    expect(dataset.actionVocabulary).toEqual(["device:scrolled thread", "device:tapped thread"]);
  });
});

describe("MockNgramMovementBackend", () => {
  it("predicts the most frequent action for a known context at level 0", async () => {
    const backend = new MockNgramMovementBackend();
    const dataset = buildMovementDataset([
      manifest([
        { kind: "observation", ts: 1, trajectoryId: "t", source: "device", summary: "mail active" },
        { kind: "action", ts: 2, trajectoryId: "t", tool: "device", summary: "tapped compose" },
      ]),
    ]);
    const policy = await backend.train(dataset);

    // No previous action => the full-context feature is `s=…&p=∅`, which was
    // present in training, so it matches at level 0 (full context).
    const prediction = policy.predict({ surface: "device/mail" });
    expect(prediction.action.label).toBe("device:tapped compose");
    expect(prediction.action.tool).toBe("device");
    expect(prediction.backoffLevel).toBe(0);
    expect(prediction.confidence).toBeGreaterThan(0);
  });

  it("backs off to a coarser context for unseen (surface, prev) pairs", async () => {
    const backend = createMovementModelBackend("mock-ngram");
    // Two examples share a surface but have different previous actions.
    const dataset = buildMovementDataset([
      manifest([
        { kind: "observation", ts: 1, trajectoryId: "t", source: "device", summary: "mail active" },
        { kind: "action", ts: 2, trajectoryId: "t", tool: "device", summary: "tapped compose" },
        { kind: "action", ts: 3, trajectoryId: "t", tool: "device", summary: "tapped send" },
      ]),
    ]);
    const policy = await backend.train(dataset);

    // Novel previous action never seen with this surface -> must back off.
    const prediction = policy.predict({ surface: "device/mail", previousAction: "device:unseen" });
    expect(prediction.backoffLevel).toBeGreaterThanOrEqual(1);
    expect(prediction.action.label.startsWith("device:")).toBe(true);
  });

  it("returns a no-knowledge sentinel when trained on nothing", async () => {
    const policy = await new MockNgramMovementBackend().train(buildMovementDataset([]));
    const prediction = policy.predict({ surface: "device/mail" });
    expect(prediction.backoffLevel).toBe(-1);
    expect(prediction.confidence).toBe(0);
  });

  it("serializes and reloads to an identical policy", async () => {
    const dataset = buildMovementDataset([
      manifest([
        { kind: "observation", ts: 1, trajectoryId: "t", source: "device", summary: "mail active" },
        { kind: "action", ts: 2, trajectoryId: "t", tool: "device", summary: "tapped compose" },
      ]),
    ]);
    const policy = await new MockNgramMovementBackend().train(dataset);
    const reloaded = deserializeMovementPolicy(policy.serialize());

    const context = { surface: "device/mail" };
    expect(reloaded.predict(context)).toEqual(policy.predict(context));
  });

  it("is deterministic across repeated training runs", async () => {
    const dataset = buildMovementDataset(
      generateSyntheticMovementReplays({ seed: 7, sessions: 4, stepsPerSession: 6 }),
    );
    const a = (await new MockNgramMovementBackend().train(dataset)).serialize();
    const b = (await new MockNgramMovementBackend().train(dataset)).serialize();
    expect(a).toEqual(b);
  });
});

describe("contextFeatures", () => {
  it("emits four ordered backoff levels", () => {
    expect(contextFeatures({ surface: "device/mail", previousAction: "device:tap" })).toEqual([
      "s=device/mail&p=device:tap",
      "s=device/mail",
      "p=device:tap",
      "*",
    ]);
  });
});

describe("generateSyntheticMovementReplays", () => {
  it("is deterministic for a given seed", () => {
    const a = generateSyntheticMovementReplays({ seed: 42, sessions: 3, stepsPerSession: 5 });
    const b = generateSyntheticMovementReplays({ seed: 42, sessions: 3, stepsPerSession: 5 });
    expect(a).toEqual(b);
  });

  it("produces alternating observation/action timelines", () => {
    const [first] = generateSyntheticMovementReplays({ seed: 1, sessions: 1, stepsPerSession: 3 });
    expect(first?.events.filter((e) => e.kind === "action")).toHaveLength(3);
    expect(first?.events.filter((e) => e.kind === "observation")).toHaveLength(3);
  });
});

describe("end-to-end generalization", () => {
  it("learns structure and generalizes to held-out sessions above chance", async () => {
    const replays = generateSyntheticMovementReplays({ seed: 123, sessions: 40, stepsPerSession: 12 });
    const { train, test } = splitMovementReplays(replays, 0.75);

    const trainDataset = buildMovementDataset(train);
    const policy = await new MockNgramMovementBackend().train(trainDataset);

    const heldOut = buildMovementDataset(test).examples;
    const evaluation = evaluateMovementGeneralization(
      policy,
      heldOut,
      trainDataset.examples.map((example) => example.context),
    );

    // The generative process makes the preferred gesture dominant (70%), so a
    // policy that learned surface/gesture structure must beat random guessing.
    expect(evaluation.total).toBeGreaterThan(0);
    expect(evaluation.accuracy).toBeGreaterThan(0.4);
    // Novel contexts exist in the held-out set and are still predicted via backoff.
    expect(evaluation.generalizationTotal).toBeGreaterThan(0);
    expect(evaluation.generalizationAccuracy).toBeGreaterThan(0.3);
    // No held-out example should hit the no-knowledge sentinel: the global
    // prior (level 3) always catches unseen contexts.
    expect(evaluation.backoffHistogram["-1"] ?? 0).toBe(0);
  });
});
