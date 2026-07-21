import { describe, expect, it } from "vitest";
import {
  ConstantMovementBackend,
  MarkovMovementBackend,
  datasetFromTrajectories,
  evaluateMovementPolicy,
  measureReplayFidelity,
  tokenizeAction,
  tokenizeReplayManifest,
  tokenizeTrajectory,
  type MovementDataset,
  type MovementPolicyBackend,
} from "./movement-policy.js";
import {
  Lcg,
  generateSyntheticMovementDataset,
  splitMovementDataset,
} from "./synthetic-movements.js";
import { buildTrajectorySpan } from "../capture/trajectory.js";
import type { ReplayManifest } from "../capture/replay.js";

function mostFrequentToken(dataset: MovementDataset): string {
  const counts = new Map<string, number>();
  for (const sequence of dataset.sequences) {
    for (const token of sequence.tokens) {
      counts.set(token, (counts.get(token) ?? 0) + 1);
    }
  }
  let best = "";
  let bestCount = -1;
  for (const [token, count] of [...counts.entries()].sort((a, b) => (a[0] < b[0] ? -1 : 1))) {
    if (count > bestCount) {
      bestCount = count;
      best = token;
    }
  }
  return best;
}

describe("tokenizeAction", () => {
  it("builds a compact gesture token from action metadata", () => {
    expect(
      tokenizeAction({
        tool: "device",
        summary: "scrolled down",
        metadata: { gesture: "scroll", direction: "down", target: "list" },
      }),
    ).toBe("device:scroll:down:list");
  });

  it("falls back to tool+summary slug when no gesture metadata", () => {
    expect(tokenizeAction({ tool: "os", summary: "Focused Main Window" })).toBe("os:focused-main-window");
  });
});

describe("tokenizeTrajectory / dataset", () => {
  it("extracts an ordered token stream sorted by timestamp", () => {
    const span = buildTrajectorySpan({
      id: "t1",
      sessionId: "s1",
      actions: [
        { kind: "action", tool: "device", summary: "b", ts: 20, metadata: { gesture: "tap", target: "b" } },
        { kind: "action", tool: "device", summary: "a", ts: 10, metadata: { gesture: "tap", target: "a" } },
      ],
    });
    const sequence = tokenizeTrajectory(span);
    expect(sequence.tokens).toEqual(["device:tap:a", "device:tap:b"]);
  });

  it("drops empty sequences when building a dataset", () => {
    const withActions = buildTrajectorySpan({
      id: "t1",
      sessionId: "s1",
      actions: [{ kind: "action", tool: "device", summary: "a", ts: 1, metadata: { gesture: "tap", target: "a" } }],
    });
    const withoutActions = buildTrajectorySpan({ id: "t2", sessionId: "s1", actions: [] });
    const dataset = datasetFromTrajectories([withActions, withoutActions]);
    expect(dataset.sequences.map((sequence) => sequence.id)).toEqual(["t1"]);
  });

  it("tokenizes replay-manifest action events", () => {
    const manifest: ReplayManifest = {
      version: 1,
      sessionId: "s1",
      trajectoryIds: ["t1"],
      eventCount: 2,
      events: [
        { kind: "observation", ts: 5, trajectoryId: "t1", source: "os", summary: "x" },
        { kind: "action", ts: 10, trajectoryId: "t1", tool: "device", summary: "tapped go" },
      ],
    };
    expect(tokenizeReplayManifest(manifest).tokens).toEqual(["device:tapped-go"]);
  });
});

describe("Lcg", () => {
  it("is deterministic for a fixed seed", () => {
    const a = new Lcg(42);
    const b = new Lcg(42);
    const seqA = Array.from({ length: 5 }, () => a.int(100));
    const seqB = Array.from({ length: 5 }, () => b.int(100));
    expect(seqA).toEqual(seqB);
  });
});

describe("MarkovMovementBackend — repeat recorded movements (objective 2c)", () => {
  it("reproduces a recorded sequence exactly from its seed", () => {
    const sequence = { id: "s", tokens: ["a", "b", "c", "d", "e"] };
    const backend = new MarkovMovementBackend({ order: 2 });
    backend.train({ sequences: [sequence] });
    const fidelity = measureReplayFidelity(backend, sequence, { seedLength: 1 });
    expect(fidelity.produced).toEqual(["b", "c", "d", "e"]);
    expect(fidelity.fidelity).toBe(1);
  });

  it("stops the rollout at the recorded end instead of looping forever", () => {
    const backend = new MarkovMovementBackend({ order: 2 });
    backend.train({ sequences: [{ id: "s", tokens: ["x", "y", "z"] }] });
    // Ask for far more tokens than the recording holds.
    expect(backend.generate(["x"], 50)).toEqual(["y", "z"]);
  });

  it("predicts deterministically (no randomness) across identical runs", () => {
    const dataset = generateSyntheticMovementDataset({ seed: 7, count: 9 });
    const a = new MarkovMovementBackend({ order: 2 });
    const b = new MarkovMovementBackend({ order: 2 });
    a.train(dataset);
    b.train(dataset);
    expect(a.predictNext(["os:focus:editor"])).toEqual(b.predictNext(["os:focus:editor"]));
  });

  it("learns the deterministic structural transitions", () => {
    const dataset = generateSyntheticMovementDataset({ seed: 3, count: 12 });
    const backend = new MarkovMovementBackend({ order: 2 });
    backend.train(dataset);
    // focus:editor is always followed by tap:menu in the edit template.
    expect(backend.predictNext(["os:focus:editor"])?.token).toBe("device:tap:menu");
    // focus:terminal always begins a typed command in run-command.
    expect(backend.predictNext(["os:focus:terminal"])?.token.startsWith("device:type:")).toBe(true);
  });
});

describe("MarkovMovementBackend — generalize to new movements (objective 2d)", () => {
  it("held-out sequences are genuinely unseen during training", () => {
    const dataset = generateSyntheticMovementDataset({ seed: 11, count: 24 });
    const { train, heldOut } = splitMovementDataset(dataset, { holdOutEvery: 4 });
    expect(heldOut.sequences.length).toBeGreaterThan(0);
    const trainIds = new Set(train.sequences.map((sequence) => sequence.id));
    for (const sequence of heldOut.sequences) {
      expect(trainIds.has(sequence.id)).toBe(false);
    }
  });

  it("generalizes above chance and beats a most-frequent-token baseline", () => {
    // Average over several seeds so the claim doesn't hinge on one lucky split.
    // ~20-token vocabulary ⇒ random-guess accuracy ≈ 0.05.
    const seeds = [11, 17, 23, 29, 37];
    const markovAccuracies: number[] = [];
    const baselineAccuracies: number[] = [];
    for (const seed of seeds) {
      const dataset = generateSyntheticMovementDataset({ seed, count: 24 });
      const { train, heldOut } = splitMovementDataset(dataset, { holdOutEvery: 4 });
      const markov = new MarkovMovementBackend({ order: 2 });
      markov.train(train);
      const markovResult = evaluateMovementPolicy(markov, heldOut, { contextOrder: 2 });
      markovAccuracies.push(markovResult.accuracy);

      // Baseline: always emit the single most-frequent token in the training set.
      const baseline = new ConstantMovementBackend(mostFrequentToken(train));
      baselineAccuracies.push(evaluateMovementPolicy(baseline, heldOut).accuracy);

      // Every individual split beats chance by a wide margin.
      expect(markovResult.accuracy).toBeGreaterThan(0.3);
    }
    const mean = (values: number[]) => values.reduce((sum, value) => sum + value, 0) / values.length;
    // The learned policy clearly generalizes better than the frequency baseline.
    expect(mean(markovAccuracies)).toBeGreaterThan(mean(baselineAccuracies) * 1.5);
    expect(mean(markovAccuracies)).toBeGreaterThan(0.35);
  });

  it("backs off to shorter contexts for unseen prefixes", () => {
    const backend = new MarkovMovementBackend({ order: 3 });
    backend.train({ sequences: [{ id: "s", tokens: ["a", "b", "c", "d"] }] });
    // Context ["unseen", "c"] has no order-2 match; backoff to ["c"] ⇒ "d".
    const prediction = backend.predictNext(["unseen", "c"]);
    expect(prediction?.token).toBe("d");
    expect(prediction?.backoffOrder).toBe(1);
  });
});

describe("pluggable backend seam", () => {
  it("honours an alternative backend implementation", () => {
    // Exercised through the interface, exactly as callers see backends.
    const backend: MovementPolicyBackend = new ConstantMovementBackend("device:tap:ok");
    backend.train({ sequences: [] });
    expect(backend.predictNext([])?.token).toBe("device:tap:ok");
    expect(backend.generate([], 3)).toEqual(["device:tap:ok", "device:tap:ok", "device:tap:ok"]);
  });

  it("evaluate + replay helpers operate through the interface, not the class", () => {
    const dataset: MovementDataset = { sequences: [{ id: "s", tokens: ["a", "b", "a", "b"] }] };
    const backend = new MarkovMovementBackend({ order: 1 });
    backend.train(dataset);
    const result = evaluateMovementPolicy(backend, dataset);
    expect(result.totalPredictions).toBe(3);
    expect(result.accuracy).toBeGreaterThan(0.5);
  });
});
