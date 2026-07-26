import { describe, expect, it } from "vitest";
import {
  DEFAULT_MOVEMENT_TEMPLATES,
  generateSyntheticMovementTrajectories,
  type MovementTemplate,
} from "./movement-synthesis.js";
import {
  MarkovMovementBackend,
  buildMovementDataset,
  evaluateMovementPolicy,
} from "./movement-policy.js";

const TEMPLATES: MovementTemplate[] = [
  {
    name: "save",
    steps: [
      { tool: "device", gesture: "tap", targetSlot: "menu" },
      { tool: "device", gesture: "tap", targetSlot: "item" },
      { tool: "device", gesture: "type", targetSlot: "field" },
      { tool: "device", gesture: "tap", targetSlot: "confirm" },
    ],
  },
];

const VOCAB = {
  menu: ["file-menu", "edit-menu"],
  item: ["save-item", "export-item"],
  field: ["name-field", "path-field"],
  confirm: ["ok-button", "apply-button"],
};

describe("generateSyntheticMovementTrajectories", () => {
  it("is deterministic for a fixed seed", () => {
    const a = generateSyntheticMovementTrajectories({ templates: TEMPLATES, targetVocabulary: VOCAB, count: 5, seed: 42 });
    const b = generateSyntheticMovementTrajectories({ templates: TEMPLATES, targetVocabulary: VOCAB, count: 5, seed: 42 });
    expect(a).toEqual(b);
  });

  it("varies with the seed", () => {
    const a = generateSyntheticMovementTrajectories({ templates: TEMPLATES, targetVocabulary: VOCAB, count: 8, seed: 1 });
    const b = generateSyntheticMovementTrajectories({ templates: TEMPLATES, targetVocabulary: VOCAB, count: 8, seed: 2 });
    const targetsOf = (list: typeof a) => list.flatMap((t) => t.actions.map((x) => x.metadata?.target)).join(",");
    expect(targetsOf(a)).not.toBe(targetsOf(b));
  });

  it("instantiates templates with monotonic timestamps and vocabulary targets", () => {
    const [trajectory] = generateSyntheticMovementTrajectories({
      templates: TEMPLATES,
      targetVocabulary: VOCAB,
      count: 1,
      seed: 7,
    });
    expect(trajectory?.actions).toHaveLength(4);
    const timestamps = trajectory!.actions.map((a) => a.ts);
    expect([...timestamps].sort((x, y) => x - y)).toEqual(timestamps);
    const menuTarget = trajectory!.actions[0]?.metadata?.target;
    expect(VOCAB.menu).toContain(menuTarget);
  });

  it("ships a usable default template library", () => {
    const trajectories = generateSyntheticMovementTrajectories({
      templates: DEFAULT_MOVEMENT_TEMPLATES,
      targetVocabulary: { menu: ["m"], item: ["i"], field: ["f"], confirm: ["c"] },
      count: 3,
      seed: 5,
    });
    expect(trajectories.length).toBe(3);
    expect(trajectories.every((t) => t.actions.length > 0)).toBe(true);
  });
});

describe("synthetic capture -> dataset -> train -> infer round-trip", () => {
  it("learns the gesture grammar and generalizes to a held-out seed", async () => {
    const trajectories = generateSyntheticMovementTrajectories({
      templates: TEMPLATES,
      targetVocabulary: VOCAB,
      count: 20,
      seed: 99,
    });
    const backend = new MarkovMovementBackend();
    const model = await backend.train(buildMovementDataset({ trajectories, order: 2 }));

    // Held out uses targets that never appear in the training vocabulary.
    const heldOut = buildMovementDataset({
      trajectories: generateSyntheticMovementTrajectories({
        templates: TEMPLATES,
        targetVocabulary: {
          menu: ["novel-menu"],
          item: ["novel-item"],
          field: ["novel-field"],
          confirm: ["novel-confirm"],
        },
        count: 4,
        seed: 123,
      }),
      order: 2,
    }).sequences;

    const result = evaluateMovementPolicy({ backend, model, heldOut });
    expect(result.total).toBeGreaterThan(0);
    // Every held-out prediction must be reachable via generalization since the
    // specific targets are all unseen — the grammar itself must carry it.
    expect(result.accuracy).toBe(1);
    expect(result.generalHits).toBe(result.total);
  });
});
