import { describe, expect, it } from "vitest";
import {
  createSeededRandom,
  generateMovementDataset,
  type SyntheticTaskSpec,
} from "./synthetic.js";

describe("createSeededRandom", () => {
  it("is deterministic for a given seed and varies across seeds", () => {
    const a = createSeededRandom(42);
    const b = createSeededRandom(42);
    const c = createSeededRandom(7);
    const seqA = [a(), a(), a()];
    const seqB = [b(), b(), b()];
    expect(seqA).toEqual(seqB);
    expect(seqA).not.toEqual([c(), c(), c()]);
    for (const value of seqA) {
      expect(value).toBeGreaterThanOrEqual(0);
      expect(value).toBeLessThan(1);
    }
  });
});

describe("generateMovementDataset", () => {
  const specs: SyntheticTaskSpec[] = [
    { task: "open-and-type", target: { x: 300, y: 200, windowId: "editor" }, typeLength: 2 },
    { task: "menu-select", target: { x: 100, y: 60, windowId: "menu" } },
    { task: "scroll-and-click", target: { x: 640, y: 400, windowId: "list" }, scrollDown: true },
  ];

  it("produces one labeled trajectory per spec with the expected structure", () => {
    const dataset = generateMovementDataset({ seed: 1, specs });
    expect(dataset.trajectories).toHaveLength(3);

    const [type, menu, scroll] = dataset.trajectories;
    expect(type!.label).toBe("open-and-type");
    // focus + move + down + up + (2 chars * 2 key events)
    expect(type!.events).toHaveLength(1 + 3 + 4);
    expect(type!.events[0]).toMatchObject({ kind: "window-focus", windowId: "editor" });

    expect(menu!.label).toBe("menu-select");
    expect(menu!.events.filter((e) => e.kind === "mouse-down")).toHaveLength(2);

    expect(scroll!.label).toBe("scroll-and-click");
    expect(scroll!.events.filter((e) => e.kind === "scroll")).toHaveLength(2);
  });

  it("is fully reproducible for a fixed seed", () => {
    const first = generateMovementDataset({ seed: 99, specs });
    const second = generateMovementDataset({ seed: 99, specs });
    expect(first.trajectories).toEqual(second.trajectories);
  });

  it("timestamps increase monotonically within a trajectory", () => {
    const dataset = generateMovementDataset({ seed: 5, specs });
    for (const trajectory of dataset.trajectories) {
      for (let i = 1; i < trajectory.events.length; i += 1) {
        expect(trajectory.events[i]!.ts).toBeGreaterThan(trajectory.events[i - 1]!.ts);
      }
    }
  });
});
