import { describe, expect, it } from "vitest";
import {
  createMovementModelBackend,
  listMovementModelBackends,
  movementContextKeys,
  type MovementDataset,
} from "./movement-model.js";
import { NgramMovementBackend } from "./ngram-backend.js";
import { buildMovementDataset } from "./dataset-builder.js";
import { generateSyntheticDataset, type MovementTemplate } from "./synthetic.js";
import { evaluateReplayFidelity } from "./eval.js";
import { buildTrajectorySpan } from "../capture/trajectory.js";

const TEMPLATES: MovementTemplate[] = [
  {
    name: "open-doc",
    steps: [
      { action: "shortcut:open", target: "cmd+o" },
      { action: "type", target: "path" },
      { action: "tap", target: "confirm" },
    ],
  },
  {
    name: "save-flow",
    steps: [
      { action: "tap", target: "menu" },
      { action: "tap", target: "save" },
      { action: "type", target: "filename" },
    ],
  },
];

describe("movement backend registry", () => {
  it("registers the ngram backend and instantiates it by name", () => {
    expect(listMovementModelBackends()).toContain("ngram");
    const backend = createMovementModelBackend("ngram");
    expect(backend.name).toBe("ngram");
  });

  it("throws for an unknown backend", () => {
    expect(() => createMovementModelBackend("nope")).toThrow(/unknown movement model backend/);
  });
});

describe("movementContextKeys", () => {
  it("produces most-specific-first backoff keys", () => {
    expect(movementContextKeys({ platform: "macos", appId: "Notes", screen: "Editor" })).toEqual([
      "macos|notes|editor",
      "macos|notes",
      "macos",
      "*",
    ]);
  });
});

describe("ngram post-training + inference", () => {
  it("reproduces a demonstrated sequence at its exact context (backoff level 0)", async () => {
    const backend = new NgramMovementBackend();
    const dataset = generateSyntheticDataset({
      seed: 7,
      templates: TEMPLATES,
      contexts: [{ appId: "editor", screen: "main", templates: ["open-doc"], repeats: 4 }],
    });
    const model = await backend.train(dataset);
    expect(model.trainedTrajectories).toBe(4);

    const prediction = await backend.predict(model, {
      context: { platform: "macos", appId: "editor", screen: "main" },
    });
    expect(prediction.steps.map((step) => step.action)).toEqual(["shortcut:open", "type", "tap"]);
    expect(prediction.steps.map((step) => step.target)).toEqual(["cmd+o", "path", "confirm"]);
    expect(prediction.maxBackoffLevel).toBe(0);
    expect(prediction.steps[0].confidence).toBeGreaterThan(0);
  });

  it("generalizes to a novel-but-related screen via app-level backoff", async () => {
    const backend = new NgramMovementBackend();
    // Two known screens of the same app both demonstrate save-flow...
    const dataset = generateSyntheticDataset({
      seed: 11,
      templates: TEMPLATES,
      contexts: [
        { appId: "notes", screen: "editor", templates: ["save-flow"], repeats: 3 },
        { appId: "notes", screen: "review", templates: ["save-flow"], repeats: 3 },
      ],
    });
    const model = await backend.train(dataset);

    // ...predict for a screen never seen in training.
    const prediction = await backend.predict(model, {
      context: { platform: "macos", appId: "notes", screen: "brand-new-draft" },
    });
    expect(prediction.steps.map((step) => step.action)).toEqual(["tap", "tap", "type"]);
    // It had to generalize: no level-0 statistics for the unseen screen.
    expect(prediction.maxBackoffLevel).toBe(1);
  });

  it("refuses to run a model trained by another backend", async () => {
    const backend = new NgramMovementBackend();
    await expect(
      backend.predict(
        { backend: "mlx", version: 1, trainedTrajectories: 0, trainedSteps: 0, weights: {} },
        { context: { platform: "macos", appId: "x" } },
      ),
    ).rejects.toThrow(/cannot run a model trained by/);
  });
});

describe("buildMovementDataset (capture bridge)", () => {
  it("derives action tokens, targets, and context from captured spans", () => {
    const span = buildTrajectorySpan({
      id: "span-1",
      sessionId: "s1",
      observations: [
        {
          kind: "observation",
          source: "device",
          summary: "Notes active",
          ts: 100,
          metadata: { appName: "Notes", platform: "macos", screenTitle: "Editor" },
        },
      ],
      actions: [
        { kind: "action", tool: "device", summary: "swiped down", ts: 120, metadata: { gesture: "swipe", direction: "down" } },
        { kind: "action", tool: "device", summary: "tapped save", ts: 110, metadata: { gesture: "tap", target: "save" } },
      ],
    });

    const dataset = buildMovementDataset([span]);
    expect(dataset.trajectories).toHaveLength(1);
    const trajectory = dataset.trajectories[0];
    expect(trajectory.context).toEqual({ platform: "macos", appId: "Notes", screen: "Editor" });
    // Steps are sorted by ts, so the tap (110) precedes the swipe (120).
    expect(trajectory.steps.map((step) => step.action)).toEqual(["tap", "swipe:down"]);
    expect(trajectory.steps[0].target).toBe("save");
  });

  it("honors approvedOnly filtering", () => {
    const base = {
      sessionId: "s1",
      observations: [
        { kind: "observation" as const, source: "device", summary: "app", ts: 1, metadata: { appName: "A" } },
      ],
      actions: [{ kind: "action" as const, tool: "device", summary: "tap", ts: 2, metadata: { gesture: "tap" } }],
    };
    const approved = buildTrajectorySpan({ id: "ok", ...base });
    approved.review = { status: "approved", reviewedAt: "now", reviewedBy: "me" };
    const pending = buildTrajectorySpan({ id: "no", ...base });

    const dataset = buildMovementDataset([approved, pending], { approvedOnly: true });
    expect(dataset.trajectories.map((trajectory) => trajectory.id)).toEqual(["ok"]);
  });
});

describe("synthetic generator", () => {
  it("is deterministic for a fixed seed", () => {
    const spec = {
      seed: 42,
      templates: TEMPLATES,
      contexts: [{ appId: "editor", templates: ["open-doc"], repeats: 2 }],
    };
    const a = generateSyntheticDataset(spec);
    const b = generateSyntheticDataset(spec);
    expect(a).toEqual(b);
    expect(a.trajectories).toHaveLength(2);
  });

  it("throws on an unknown template reference", () => {
    expect(() =>
      generateSyntheticDataset({ templates: TEMPLATES, contexts: [{ appId: "x", templates: ["ghost"] }] }),
    ).toThrow(/unknown movement template/);
  });
});

describe("evaluateReplayFidelity", () => {
  it("scores full overlap for a memorized trajectory and flags generalization on held-out screens", async () => {
    const backend = new NgramMovementBackend();
    const dataset: MovementDataset = generateSyntheticDataset({
      seed: 3,
      templates: TEMPLATES,
      contexts: [
        { appId: "notes", screen: "editor", templates: ["save-flow"], repeats: 3 },
        { appId: "notes", screen: "review", templates: ["save-flow"], repeats: 3 },
      ],
    });
    const model = await backend.train(dataset);

    const memorized = evaluateReplayFidelity(backend, model, dataset.trajectories[0]);
    const heldOut = evaluateReplayFidelity(backend, model, {
      id: "held-out",
      context: { platform: "macos", appId: "notes", screen: "unseen" },
      steps: dataset.trajectories[0].steps,
    });

    const [m, h] = await Promise.all([memorized, heldOut]);
    expect(m.actionOverlap).toBe(1);
    expect(m.maxBackoffLevel).toBe(0);
    expect(h.actionOverlap).toBe(1);
    expect(h.maxBackoffLevel).toBe(1);
  });
});
