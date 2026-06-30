import { describe, expect, it } from "vitest";
import { pointerPath } from "./events.js";
import {
  createSeededRandom,
  generateDragDemonstration,
  generateSyntheticDataset,
} from "./synthetic.js";

describe("synthetic generator", () => {
  it("seeded RNG is deterministic and reproducible", () => {
    const a = createSeededRandom(123);
    const b = createSeededRandom(123);
    const seqA = [a(), a(), a()];
    const seqB = [b(), b(), b()];
    expect(seqA).toEqual(seqB);
    expect(seqA.every((v) => v >= 0 && v < 1)).toBe(true);
  });

  it("builds an ordered drag: down, N moves, up", () => {
    const demo = generateDragDemonstration({
      id: "d",
      intent: "drag",
      start: { x: 0, y: 0 },
      end: { x: 100, y: 100 },
      steps: 4,
    });
    expect(demo.events).toHaveLength(6);
    expect(demo.events[0]!.kind).toBe("pointer-down");
    expect(demo.events.at(-1)!.kind).toBe("pointer-up");
    expect(demo.events.filter((e) => e.kind === "pointer-move")).toHaveLength(4);
    const ts = demo.events.map((e) => e.ts);
    expect(ts).toEqual([...ts].sort((x, y) => x - y));
  });

  it("a line drag has a straight, monotonic path; an arc bows off-axis", () => {
    const line = generateDragDemonstration({
      id: "l",
      intent: "drag",
      start: { x: 0, y: 0 },
      end: { x: 100, y: 0 },
      shape: "line",
      steps: 8,
    });
    expect(pointerPath(line.events).every((p) => Math.abs(p.y) < 1e-9)).toBe(true);

    const arc = generateDragDemonstration({
      id: "a",
      intent: "drag",
      start: { x: 0, y: 0 },
      end: { x: 100, y: 0 },
      shape: "arc",
      arcHeight: 30,
      steps: 8,
    });
    const maxBow = Math.max(...pointerPath(arc.events).map((p) => Math.abs(p.y)));
    expect(maxBow).toBeGreaterThan(10);
  });

  it("generates a labelled dataset deterministically for a given seed", () => {
    const a = generateSyntheticDataset({ intent: "drag", count: 4, seed: 99 });
    const b = generateSyntheticDataset({ intent: "drag", count: 4, seed: 99 });
    expect(a.demonstrations).toHaveLength(4);
    expect(JSON.stringify(a)).toEqual(JSON.stringify(b));
    expect(a.demonstrations.every((d) => d.intent === "drag")).toBe(true);
  });
});
