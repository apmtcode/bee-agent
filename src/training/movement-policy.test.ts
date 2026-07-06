import { describe, expect, it } from "vitest";
import {
  NGramMovementBackend,
  createDefaultMovementPolicyRegistry,
  evaluateMovementPolicy,
  tokenizeMovementEvent,
  tokenizeMovementTrajectory,
  type MovementDataset,
  type MovementEvent,
} from "./movement-policy.js";
import {
  generateSyntheticMovementDataset,
  movementTrajectoryFromReplay,
  movementTrajectoryFromSpan,
} from "./movement-dataset.js";
import { buildTrajectorySpan } from "../capture/trajectory.js";
import { buildReplayManifest } from "../capture/replay.js";

function ev(partial: Partial<MovementEvent> & { ts: number; action: string }): MovementEvent {
  return { channel: "pointer", ...partial };
}

describe("tokenizeMovementEvent", () => {
  it("encodes channel:action:target deterministically", () => {
    expect(tokenizeMovementEvent(ev({ ts: 0, channel: "pointer", action: "click", target: "save" }))).toBe(
      "pointer:click:save",
    );
  });

  it("omits value unless requested", () => {
    const event = ev({ ts: 0, channel: "keyboard", action: "keydown", value: "cmd+s" });
    expect(tokenizeMovementEvent(event)).toBe("keyboard:keydown:");
    expect(tokenizeMovementEvent(event, { includeValue: true })).toBe("keyboard:keydown::cmd+s");
  });

  it("sorts events by timestamp before tokenizing", () => {
    const tokens = tokenizeMovementTrajectory({
      id: "t",
      events: [ev({ ts: 2, action: "b" }), ev({ ts: 1, action: "a" })],
    });
    expect(tokens).toEqual(["pointer:a:", "pointer:b:"]);
  });
});

describe("NGramMovementBackend", () => {
  const dataset: MovementDataset = {
    version: 1,
    trajectories: [
      {
        id: "flow",
        events: [
          ev({ ts: 0, channel: "window", action: "focus", target: "editor" }),
          ev({ ts: 1, channel: "pointer", action: "move", target: "toolbar" }),
          ev({ ts: 2, channel: "pointer", action: "click", target: "save" }),
        ],
      },
    ],
  };

  it("learns the next movement from a recorded sequence", () => {
    const model = new NGramMovementBackend().train(dataset);
    const prediction = model.predict(["window:focus:editor"]);
    expect(prediction?.token).toBe("pointer:move:toolbar");
    expect(prediction?.probability).toBe(1);
    expect(prediction?.matchedOrder).toBe(1);
  });

  it("replays a full recorded movement sequence via rollout", () => {
    const model = new NGramMovementBackend().train(dataset);
    const replayed = model.rollout(["window:focus:editor"], 2);
    expect(replayed).toEqual(["pointer:move:toolbar", "pointer:click:save"]);
  });

  it("generalizes to an unseen context by backing off", () => {
    // Two flows share the "click:save -> keydown" tail but differ in prefix.
    const trained = new NGramMovementBackend().train({
      version: 1,
      trajectories: [
        {
          id: "a",
          events: [
            ev({ ts: 0, action: "click", target: "save" }),
            ev({ ts: 1, channel: "keyboard", action: "keydown", target: "enter" }),
          ],
        },
        {
          id: "b",
          events: [
            ev({ ts: 0, channel: "window", action: "focus", target: "x" }),
            ev({ ts: 1, action: "click", target: "save" }),
            ev({ ts: 2, channel: "keyboard", action: "keydown", target: "enter" }),
          ],
        },
      ],
    });
    // Novel prefix never paired with click:save, but the 1-gram context matches.
    const prediction = trained.predict(["window:focus:zzz", "pointer:click:save"]);
    expect(prediction?.token).toBe("keyboard:keydown:enter");
    expect(prediction?.matchedOrder).toBeLessThanOrEqual(1);
  });

  it("returns undefined for an empty model", () => {
    const model = new NGramMovementBackend().train({ version: 1, trajectories: [] });
    expect(model.predict(["anything"])).toBeUndefined();
    expect(model.rollout(["anything"], 3)).toEqual([]);
  });

  it("round-trips through serialize/load", () => {
    const backend = new NGramMovementBackend();
    const model = backend.train(dataset);
    const snapshot = model.serialize();
    const restored = backend.load(snapshot);
    expect(restored.serialize()).toEqual(snapshot);
    expect(restored.predict(["window:focus:editor"])?.token).toBe("pointer:move:toolbar");
    expect(restored.vocabularySize).toBe(model.vocabularySize);
  });

  it("breaks frequency ties deterministically (lexical)", () => {
    // From context "a", "b" and "c" each follow once -> tie -> lexical winner "b".
    const model = new NGramMovementBackend().train({
      version: 1,
      trajectories: [
        { id: "1", events: [ev({ ts: 0, action: "a" }), ev({ ts: 1, action: "b" })] },
        { id: "2", events: [ev({ ts: 0, action: "a" }), ev({ ts: 1, action: "c" })] },
      ],
    });
    expect(model.predict(["pointer:a:"])?.token).toBe("pointer:b:");
  });
});

describe("registry", () => {
  it("exposes the default ngram backend and rejects unknown ids", () => {
    const registry = createDefaultMovementPolicyRegistry();
    expect(registry.list()).toContain("ngram");
    expect(registry.require("ngram").id).toBe("ngram");
    expect(registry.get("does-not-exist")).toBeUndefined();
    expect(() => registry.require("does-not-exist")).toThrow(/Unknown movement-policy backend/);
  });
});

describe("evaluateMovementPolicy", () => {
  it("scores well above chance on held-out data from the same generator", () => {
    const backend = new NGramMovementBackend();
    const train = generateSyntheticMovementDataset({ seed: 7, trajectoryCount: 40 });
    const heldOut = generateSyntheticMovementDataset({ seed: 999, trajectoryCount: 20 });
    const model = backend.train(train, { maxOrder: 3 });
    const report = evaluateMovementPolicy(model, heldOut);

    expect(report.trajectoriesEvaluated).toBe(20);
    expect(report.predictions).toBeGreaterThan(0);
    expect(report.coverage).toBeGreaterThan(0.9);
    // The flows are highly structured, so a trained n-gram should be strong.
    expect(report.accuracy).toBeGreaterThan(0.7);
  });

  it("a trained model beats an empty (untrained) baseline", () => {
    const backend = new NGramMovementBackend();
    const heldOut = generateSyntheticMovementDataset({ seed: 3, trajectoryCount: 15 });
    const trained = backend.train(generateSyntheticMovementDataset({ seed: 42, trajectoryCount: 30 }));
    const empty = backend.train({ version: 1, trajectories: [] });

    const trainedReport = evaluateMovementPolicy(trained, heldOut);
    const emptyReport = evaluateMovementPolicy(empty, heldOut);
    expect(trainedReport.accuracy).toBeGreaterThan(emptyReport.accuracy);
    expect(emptyReport.predictions).toBe(0);
  });
});

describe("dataset adapters", () => {
  it("derives a movement trajectory from a captured span", () => {
    const span = buildTrajectorySpan({
      id: "span-1",
      sessionId: "s1",
      observations: [{ kind: "observation", ts: 5, source: "screen", summary: "editor open" }],
      actions: [{ kind: "action", ts: 10, tool: "Click", summary: "save button" }],
    });
    const trajectory = movementTrajectoryFromSpan(span);
    expect(trajectory.id).toBe("span-1");
    expect(tokenizeMovementTrajectory(trajectory)).toEqual([
      "observation:screen:editor open",
      "tool:Click:save button",
    ]);
  });

  it("derives a movement trajectory from a replay manifest", () => {
    const span = buildTrajectorySpan({
      id: "span-2",
      sessionId: "s2",
      observations: [{ kind: "observation", ts: 1, source: "screen", summary: "open" }],
      actions: [{ kind: "action", ts: 2, tool: "Type", summary: "hello" }],
    });
    const manifest = buildReplayManifest({ sessionId: "s2", transcript: [], trajectories: [span] });
    const trajectory = movementTrajectoryFromReplay(manifest);
    expect(trajectory.events.map((e) => e.channel)).toContain("tool");
    expect(trajectory.events.length).toBe(manifest.eventCount);
  });

  it("generates identical datasets for identical seeds and differs across seeds", () => {
    expect(generateSyntheticMovementDataset({ seed: 5 })).toEqual(
      generateSyntheticMovementDataset({ seed: 5 }),
    );
    expect(generateSyntheticMovementDataset({ seed: 5 })).not.toEqual(
      generateSyntheticMovementDataset({ seed: 6 }),
    );
  });
});
