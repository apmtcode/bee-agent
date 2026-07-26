import { describe, expect, it } from "vitest";
import {
  FORM_FILL_GRAMMAR,
  NAV_MENU_GRAMMAR,
  createSeededRng,
  generateSyntheticTrajectories,
  withNovelTargets,
} from "./synthetic-movements.js";
import { buildMovementDataset, tokenizeAction } from "./movement-model.js";

describe("createSeededRng", () => {
  it("is deterministic for a given seed", () => {
    const a = createSeededRng(42);
    const b = createSeededRng(42);
    const seqA = [a(), a(), a()];
    const seqB = [b(), b(), b()];
    expect(seqA).toEqual(seqB);
    expect(seqA.every((n) => n >= 0 && n < 1)).toBe(true);
  });

  it("differs across seeds", () => {
    expect(createSeededRng(1)()).not.toEqual(createSeededRng(2)());
  });
});

describe("generateSyntheticTrajectories", () => {
  it("produces the requested count with grammar-shaped actions", () => {
    const trajectories = generateSyntheticTrajectories({
      grammar: FORM_FILL_GRAMMAR,
      count: 5,
      rng: createSeededRng(7),
    });
    expect(trajectories).toHaveLength(5);
    for (const trajectory of trajectories) {
      expect(trajectory.actions).toHaveLength(FORM_FILL_GRAMMAR.steps.length);
      // action verbs follow the grammar order
      expect(trajectory.actions.map((a) => tokenizeAction(a).action)).toEqual(
        FORM_FILL_GRAMMAR.steps.map((s) => s.action),
      );
      // every target comes from the step's pool
      trajectory.actions.forEach((a, i) => {
        expect(FORM_FILL_GRAMMAR.steps[i].targets).toContain(tokenizeAction(a).target);
      });
    }
  });

  it("is byte-stable across runs (fixed createdAt + seeded rng)", () => {
    const first = generateSyntheticTrajectories({ grammar: NAV_MENU_GRAMMAR, count: 3, rng: createSeededRng(11) });
    const second = generateSyntheticTrajectories({ grammar: NAV_MENU_GRAMMAR, count: 3, rng: createSeededRng(11) });
    expect(JSON.stringify(first)).toEqual(JSON.stringify(second));
    expect(first[0].createdAt).toBe("1970-01-01T00:00:00.000Z");
  });

  it("can drop an optional tail step for variation", () => {
    const trajectories = generateSyntheticTrajectories({
      grammar: FORM_FILL_GRAMMAR,
      count: 40,
      rng: createSeededRng(3),
      dropTailProbability: 0.5,
    });
    const lengths = new Set(trajectories.map((t) => t.actions.length));
    expect(lengths.has(FORM_FILL_GRAMMAR.steps.length)).toBe(true);
    expect(lengths.has(FORM_FILL_GRAMMAR.steps.length - 1)).toBe(true);
  });

  it("feeds cleanly into the dataset builder", () => {
    const trajectories = generateSyntheticTrajectories({ grammar: FORM_FILL_GRAMMAR, count: 2, rng: createSeededRng(1) });
    const dataset = buildMovementDataset(trajectories);
    expect(dataset.sequences).toHaveLength(2);
    expect(dataset.sequences[0].tokens.length).toBe(FORM_FILL_GRAMMAR.steps.length);
  });
});

describe("withNovelTargets", () => {
  it("keeps structure but disjoints the target pools", () => {
    const novel = withNovelTargets(FORM_FILL_GRAMMAR);
    expect(novel.steps.map((s) => s.action)).toEqual(FORM_FILL_GRAMMAR.steps.map((s) => s.action));
    const originalTargets = new Set(FORM_FILL_GRAMMAR.steps.flatMap((s) => s.targets));
    for (const step of novel.steps) {
      for (const target of step.targets) {
        expect(originalTargets.has(target)).toBe(false);
      }
    }
  });
});
