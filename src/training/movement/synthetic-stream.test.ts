import { describe, expect, it } from "vitest";
import { validateMovementTrajectory } from "./event-schema.js";
import {
  createSeededRng,
  generateClickGesture,
  generateGestureBatch,
  generateTypingGesture,
} from "./synthetic-stream.js";

describe("synthetic-stream", () => {
  it("produces a reproducible PRNG sequence for a fixed seed", () => {
    const a = createSeededRng(42);
    const b = createSeededRng(42);
    const seqA = [a(), a(), a()];
    const seqB = [b(), b(), b()];
    expect(seqA).toEqual(seqB);
    expect(seqA.every((value) => value >= 0 && value < 1)).toBe(true);
  });

  it("generates a click gesture that lands exactly on the target", () => {
    const gesture = generateClickGesture({
      id: "g1",
      start: { x: 0, y: 0 },
      target: { x: 100, y: 50 },
      steps: 10,
    });
    expect(validateMovementTrajectory(gesture)).toEqual([]);
    const moves = gesture.events.filter((event) => event.kind === "pointer-move");
    expect(moves).toHaveLength(10);
    const down = gesture.events.find((event) => event.kind === "pointer-down");
    expect(down).toMatchObject({ x: 100, y: 50, button: "left" });
    // Ends with a matching up event.
    expect(gesture.events.at(-1)).toMatchObject({ kind: "pointer-up", x: 100, y: 50 });
  });

  it("keeps jittered gestures landing on target but off the straight line mid-path", () => {
    const clean = generateClickGesture({ id: "c", start: { x: 0, y: 0 }, target: { x: 100, y: 0 }, steps: 12 });
    const jittered = generateClickGesture({
      id: "j",
      start: { x: 0, y: 0 },
      target: { x: 100, y: 0 },
      steps: 12,
      jitter: 8,
      seed: 7,
    });
    const cleanMoves = clean.events.filter((event) => event.kind === "pointer-move");
    const jitterMoves = jittered.events.filter((event) => event.kind === "pointer-move");
    // Endpoints agree; some interior point differs due to jitter.
    expect(jitterMoves.at(-1)).toMatchObject({ x: 100, y: 0 });
    const anyDifferent = jitterMoves.some(
      (event, index) => Math.abs((event as { y: number }).y - (cleanMoves[index] as { y: number }).y) > 0.01,
    );
    expect(anyDifferent).toBe(true);
  });

  it("generates a typing gesture with paired key events", () => {
    const gesture = generateTypingGesture({ id: "type", text: "hi", keyMs: 100 });
    expect(gesture.events).toHaveLength(4);
    expect(gesture.events.map((event) => event.kind)).toEqual([
      "key-down",
      "key-up",
      "key-down",
      "key-up",
    ]);
    expect(validateMovementTrajectory(gesture)).toEqual([]);
  });

  it("generates a reproducible gesture batch, one per target", () => {
    const targets = [
      { x: 10, y: 10 },
      { x: 200, y: 40 },
      { x: 50, y: 300 },
    ];
    const batchA = generateGestureBatch({ targets, seed: 3 });
    const batchB = generateGestureBatch({ targets, seed: 3 });
    expect(batchA).toHaveLength(3);
    expect(JSON.stringify(batchA)).toEqual(JSON.stringify(batchB));
  });
});
