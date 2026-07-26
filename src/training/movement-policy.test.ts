import { describe, expect, it } from "vitest";
import type { TrajectoryAction } from "../capture/trajectory.js";
import {
  MarkovMovementBackend,
  buildMovementDataset,
  encodeActionFeature,
  evaluateMovementPolicy,
  rolloutMovementPolicy,
  type MovementTrajectoryInput,
} from "./movement-policy.js";

function action(
  tool: string,
  gesture: string,
  target: string | undefined,
  ts: number,
  direction?: string,
): TrajectoryAction {
  const metadata: Record<string, unknown> = { gesture };
  if (direction) {
    metadata.direction = direction;
  }
  if (target) {
    metadata.target = target;
  }
  return { kind: "action", tool, summary: `${gesture} ${target ?? ""}`.trim(), ts, metadata };
}

// A 4-step "open menu -> tap item -> type field -> tap confirm" idiom, with the
// concrete targets supplied so we can vary them between train and held-out sets.
function openSaveTrajectory(id: string, targets: [string, string, string, string], baseTs = 1000): MovementTrajectoryInput {
  return {
    id,
    actions: [
      action("device", "tap", targets[0], baseTs),
      action("device", "tap", targets[1], baseTs + 250),
      action("device", "type", targets[2], baseTs + 500),
      action("device", "tap", targets[3], baseTs + 750),
    ],
  };
}

describe("encodeActionFeature", () => {
  it("splits into specific (with target) and general (target-dropped) channels", () => {
    const feature = encodeActionFeature(action("device", "tap", "save-button", 0));
    expect(feature.specific).toBe("device|tap|@save-button");
    expect(feature.general).toBe("device|tap");
    expect(feature.tool).toBe("device");
  });

  it("includes direction and omits absent target", () => {
    const feature = encodeActionFeature(action("device", "scroll", undefined, 0, "down"));
    expect(feature.specific).toBe("device|scroll|down");
    expect(feature.general).toBe("device|scroll|down");
  });
});

describe("buildMovementDataset", () => {
  it("sorts actions by timestamp so out-of-order capture is corrected", () => {
    const trajectory: MovementTrajectoryInput = {
      id: "t1",
      actions: [action("device", "type", "b", 500), action("device", "tap", "a", 100)],
    };
    const dataset = buildMovementDataset({ trajectories: [trajectory], order: 2 });
    expect(dataset.sequences[0]?.features.map((f) => f.specific)).toEqual([
      "device|tap|@a",
      "device|type|@b",
    ]);
  });
});

describe("MarkovMovementBackend — verbatim replay", () => {
  it("predicts the exact recorded next movement for a seen context", async () => {
    const backend = new MarkovMovementBackend();
    const dataset = buildMovementDataset({
      trajectories: [openSaveTrajectory("train", ["menu", "item", "field", "confirm"])],
      order: 2,
    });
    const model = await backend.train(dataset);

    // Context = [tap menu, tap item] -> recorded next is [type field].
    const context = dataset.sequences[0]!.features.slice(0, 2);
    const prediction = backend.predict(model, context);
    expect(prediction.level).toBe("specific");
    expect(prediction.next).toBe("device|type|@field");
    expect(prediction.confidence).toBe(1);
  });

  it("rolls out the recorded continuation from a seed", async () => {
    const backend = new MarkovMovementBackend();
    const trajectory = openSaveTrajectory("train", ["menu", "item", "field", "confirm"]);
    const dataset = buildMovementDataset({ trajectories: [trajectory], order: 2 });
    const model = await backend.train(dataset);

    const seed = dataset.sequences[0]!.features.slice(0, 1);
    const generated = rolloutMovementPolicy({ backend, model, seed, steps: 3 });
    expect(generated.map((g) => g.token)).toEqual([
      "device|tap|@item",
      "device|type|@field",
      "device|tap|@confirm",
    ]);
    expect(generated.every((g) => g.level === "specific")).toBe(true);
  });
});

describe("MarkovMovementBackend — generalization to new but related movements", () => {
  it("backs off to the general channel for an unseen target with the same gesture grammar", async () => {
    const backend = new MarkovMovementBackend();
    // Train only on 'menu/item/field/confirm'; evaluate on entirely different
    // targets that never appear in training but share the gesture structure.
    const train = openSaveTrajectory("train", ["fileMenu", "openItem", "nameField", "okButton"]);
    const dataset = buildMovementDataset({ trajectories: [train], order: 2 });
    const model = await backend.train(dataset);

    const heldOut = buildMovementDataset({
      trajectories: [openSaveTrajectory("eval", ["burger", "row", "search", "apply"])],
      order: 2,
    });
    // Context = [tap burger, tap row] -> the specific tokens were never seen,
    // but the (tap -> tap -> type) grammar was, so it should predict 'type'.
    const context = heldOut.sequences[0]!.features.slice(0, 2);
    const prediction = backend.predict(model, context);
    expect(prediction.level).toBe("general");
    expect(prediction.next).toBe("device|type");
  });

  it("scores held-out generalization via the eval harness", async () => {
    const backend = new MarkovMovementBackend();
    const train = [
      openSaveTrajectory("t1", ["a1", "a2", "a3", "a4"], 1000),
      openSaveTrajectory("t2", ["b1", "b2", "b3", "b4"], 5000),
    ];
    const model = await backend.train(buildMovementDataset({ trajectories: train, order: 2 }));

    const heldOut = buildMovementDataset({
      trajectories: [openSaveTrajectory("eval", ["z1", "z2", "z3", "z4"], 9000)],
      order: 2,
    }).sequences;
    const result = evaluateMovementPolicy({ backend, model, heldOut });

    expect(result.total).toBe(3); // positions i=1,2,3
    expect(result.generalHits).toBeGreaterThan(0);
    expect(result.accuracy).toBeGreaterThanOrEqual(2 / 3);
  });
});

describe("MarkovMovementBackend — fallback", () => {
  it("falls back to the most frequent movement when nothing matches", async () => {
    const backend = new MarkovMovementBackend();
    const model = await backend.train(
      buildMovementDataset({
        trajectories: [
          { id: "t", actions: [action("device", "tap", "x", 1), action("device", "tap", "x", 2)] },
        ],
        order: 2,
      }),
    );
    const prediction = backend.predict(model, [encodeActionFeature(action("keyboard", "shortcut", "cmd-z", 0))]);
    expect(prediction.level).toBe("fallback");
    expect(prediction.next).toBe("device|tap|@x");
  });
});
