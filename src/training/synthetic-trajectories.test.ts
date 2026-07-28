import { describe, expect, it } from "vitest";
import {
  generateSyntheticTrajectories,
  listSyntheticFamilies,
} from "./synthetic-trajectories.js";

describe("generateSyntheticTrajectories", () => {
  it("produces the requested count for a known family", () => {
    const trajectories = generateSyntheticTrajectories({
      family: "desktop-file-edit",
      count: 5,
      seed: 1,
    });
    expect(trajectories).toHaveLength(5);
    expect(trajectories.every((trajectory) => trajectory.actions.length > 0)).toBe(true);
  });

  it("is deterministic for a given seed", () => {
    const first = generateSyntheticTrajectories({ family: "web-form-submit", count: 3, seed: 42 });
    const second = generateSyntheticTrajectories({ family: "web-form-submit", count: 3, seed: 42 });
    expect(first).toEqual(second);
  });

  it("varies with the seed", () => {
    const a = generateSyntheticTrajectories({ family: "web-form-submit", count: 2, seed: 1 });
    const b = generateSyntheticTrajectories({ family: "web-form-submit", count: 2, seed: 2 });
    expect(JSON.stringify(a)).not.toEqual(JSON.stringify(b));
  });

  it("emits ordered timestamps and does not read the wall clock", () => {
    const [trajectory] = generateSyntheticTrajectories({
      family: "desktop-file-edit",
      count: 1,
      seed: 7,
      startTs: 0,
    });
    const timestamps = trajectory!.actions.map((action) => action.ts);
    const sorted = [...timestamps].sort((x, y) => x - y);
    expect(timestamps).toEqual(sorted);
    // startTs 0 -> createdAt anchored to the epoch, proving no Date.now() use.
    expect(trajectory!.createdAt).toBe(new Date(0).toISOString());
  });

  it("keeps required steps and only toggles optional ones under novelty", () => {
    const baseline = generateSyntheticTrajectories({
      family: "desktop-file-edit",
      count: 1,
      seed: 3,
      noveltyRate: 0,
    })[0]!;
    // With novelty 0, required steps (focus + a save shortcut) are always present.
    expect(baseline.actions.some((action) => action.tool === "window.focus")).toBe(true);
    expect(baseline.actions.some((action) => action.summary.includes("cmd+s"))).toBe(true);

    const lengths = new Set(
      Array.from({ length: 12 }, (_, seed) =>
        generateSyntheticTrajectories({
          family: "desktop-file-edit",
          count: 1,
          seed,
          noveltyRate: 0.5,
        })[0]!.actions.length,
      ),
    );
    // High novelty toggles optional steps, so sequence lengths vary.
    expect(lengths.size).toBeGreaterThan(1);
  });

  it("throws on an unknown family", () => {
    expect(() =>
      // @ts-expect-error exercising the runtime guard
      generateSyntheticTrajectories({ family: "nope", count: 1, seed: 1 }),
    ).toThrow(/Unknown synthetic family/);
  });

  it("lists its built-in families", () => {
    expect(listSyntheticFamilies()).toEqual(
      expect.arrayContaining(["desktop-file-edit", "web-form-submit"]),
    );
  });
});
