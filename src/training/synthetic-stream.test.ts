import { describe, expect, it } from "vitest";
import {
  createMovementBackend,
  evaluateNextTokenAccuracy,
  generateMovementSequence,
  tokenizeMovementEvent,
  tokenizeTrajectory,
} from "./movement-model.js";
import {
  generateRelatedTrajectory,
  generateSyntheticMovementDataset,
  listSyntheticWorkflows,
} from "./synthetic-stream.js";

describe("generateSyntheticMovementDataset", () => {
  it("is deterministic for a fixed seed", () => {
    const a = generateSyntheticMovementDataset({ seed: 7, trajectoryCount: 5 });
    const b = generateSyntheticMovementDataset({ seed: 7, trajectoryCount: 5 });
    expect(b).toEqual(a);
    expect(a.trajectories).toHaveLength(5);
  });

  it("produces different streams for different seeds", () => {
    const a = generateSyntheticMovementDataset({ seed: 1, trajectoryCount: 8 });
    const b = generateSyntheticMovementDataset({ seed: 2, trajectoryCount: 8 });
    expect(b).not.toEqual(a);
  });

  it("emits alternating observation/action events with increasing timestamps", () => {
    const dataset = generateSyntheticMovementDataset({ seed: 3, trajectoryCount: 1, minSteps: 2 });
    const events = dataset.trajectories[0]!.events;
    expect(events.length).toBeGreaterThanOrEqual(4);
    expect(events[0]!.kind).toBe("observation");
    expect(events[1]!.kind).toBe("action");
    for (let index = 1; index < events.length; index += 1) {
      expect(events[index]!.ts).toBeGreaterThan(events[index - 1]!.ts);
    }
  });

  it("honors a workflow filter", () => {
    const dataset = generateSyntheticMovementDataset({ seed: 5, trajectoryCount: 6, workflows: ["search"] });
    expect(dataset.trajectories.every((trajectory) => trajectory.id.startsWith("search-"))).toBe(true);
  });

  it("throws when no template matches the filter", () => {
    expect(() => generateSyntheticMovementDataset({ seed: 1, trajectoryCount: 1, workflows: ["nope"] })).toThrow();
  });

  it("lists the built-in workflows", () => {
    expect(listSyntheticWorkflows()).toEqual(expect.arrayContaining(["deploy", "search", "compose"]));
  });
});

describe("synthetic round-trip: capture -> dataset -> train -> generalize", () => {
  it("trains on synthetic data and reproduces a full template chain", async () => {
    // Full-length episodes (no truncation) → unambiguous next-token continuations.
    const dataset = generateSyntheticMovementDataset({ seed: 42, trajectoryCount: 40, truncate: false });
    const backend = createMovementBackend();
    const model = await backend.train(dataset, { order: 3 });

    // A clean full deploy template; seed with only its first observation.
    const deploy = generateSyntheticMovementDataset({
      seed: 42,
      trajectoryCount: 12,
      workflows: ["deploy"],
      truncate: false,
    }).trajectories[0]!;
    const tokens = tokenizeTrajectory(deploy);
    const seed = [tokenizeMovementEvent(deploy.events[0]!)];
    const generated = generateMovementSequence(backend, model, seed);
    // The model should recover the recorded action/observation chain.
    expect([seed[0], ...generated]).toEqual(tokens.slice(0, -1));
  });

  it("generalizes to a related trajectory entered from a novel observation", async () => {
    const dataset = generateSyntheticMovementDataset({ seed: 99, trajectoryCount: 60, truncate: false });
    const backend = createMovementBackend();
    const model = await backend.train(dataset, { order: 2 });

    const related = generateRelatedTrajectory({
      seed: 1,
      workflow: "compose",
      noveltyPrefix: "reached via unfamiliar shortcut",
    });
    // The very first observation is novel (never trained on), the rest is real.
    const accuracy = evaluateNextTokenAccuracy(backend, model, related);
    // Backoff means only the first prediction can miss; the chain is otherwise recovered.
    expect(accuracy.accuracy).toBeGreaterThan(0.7);
  });
});
