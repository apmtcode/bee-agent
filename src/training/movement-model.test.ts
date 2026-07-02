import { describe, expect, it } from "vitest";
import type { ReplayManifest } from "../capture/replay.js";
import {
  actionLabel,
  buildMovementDataset,
  buildMovementDatasetFromReplays,
  DeterministicMovementBackend,
  evaluateMovementModel,
  movementSequenceFromReplay,
  trainMovementModel,
  type MovementEvent,
  type MovementModel,
  type MovementModelBackend,
} from "./movement-model.js";
import {
  relatedVariant,
  synthesizeRepetitions,
  synthesizeSequence,
  type SyntheticTaskTemplate,
} from "./movement-synthetic.js";

const editTask: SyntheticTaskTemplate = {
  name: "edit-and-save",
  steps: [
    { observationSource: "os", observationSummary: "focused editor", actionTool: "keyboard", actionSummary: "open file" },
    { observationSource: "os", observationSummary: "file panel visible", actionTool: "mouse", actionSummary: "click line 10" },
    { observationSource: "os", observationSummary: "cursor at line 10", actionTool: "keyboard", actionSummary: "type change" },
    { observationSource: "os", observationSummary: "buffer dirty", actionTool: "keyboard", actionSummary: "save file" },
  ],
};

describe("buildMovementDataset", () => {
  it("emits one example per action with the preceding context", () => {
    const sequence = synthesizeSequence(editTask);
    const dataset = buildMovementDataset([sequence]);

    // 4 actions -> 4 examples.
    expect(dataset.examples).toHaveLength(4);
    // Vocabulary is the 4 distinct action labels, sorted.
    expect(dataset.actionVocabulary).toHaveLength(4);
    expect([...dataset.actionVocabulary]).toEqual([...dataset.actionVocabulary].sort());

    // The first example's context is just the first observation.
    expect(dataset.examples[0].context).toHaveLength(1);
    expect(dataset.examples[0].context[0].kind).toBe("observation");
    // The last example's context includes all earlier obs + actions.
    expect(dataset.examples[3].context.length).toBeGreaterThan(dataset.examples[0].context.length);
    expect(dataset.examples[3].action.summary).toBe("save file");
  });

  it("builds a dataset directly from replay manifests", () => {
    const manifest: Pick<ReplayManifest, "events" | "trajectoryIds"> = {
      trajectoryIds: ["traj-1"],
      events: [
        { kind: "transcript", ts: 0, messageId: "m1", role: "user", content: "please edit" },
        { kind: "observation", ts: 1, trajectoryId: "traj-1", source: "os", summary: "focused editor" },
        { kind: "action", ts: 2, trajectoryId: "traj-1", tool: "keyboard", summary: "open file" },
      ],
    };
    const dataset = buildMovementDatasetFromReplays([manifest]);
    expect(dataset.examples).toHaveLength(1);
    expect(dataset.examples[0].action.tool).toBe("keyboard");
    // The transcript event is preserved as an observation in context.
    const sequence = movementSequenceFromReplay(manifest);
    expect(sequence[0]).toMatchObject({ kind: "observation", source: "transcript:user" });
  });
});

describe("DeterministicMovementBackend replay fidelity", () => {
  it("reproduces recorded movements with perfect fidelity on training data", async () => {
    const sequences = synthesizeRepetitions(editTask, 3);
    const dataset = buildMovementDataset(sequences);
    const { model, replayFidelity } = await trainMovementModel({ dataset, trainingSequences: sequences });

    expect(replayFidelity).toBeDefined();
    expect(replayFidelity!.steps).toBe(12); // 4 actions * 3 repetitions
    expect(replayFidelity!.fidelity).toBe(1);
    // Every step after the first observation should resolve via the n-gram model.
    expect(replayFidelity!.bySource.ngram).toBeGreaterThan(0);
    expect(model.serialize().exampleCount).toBe(12);
  });

  it("is deterministic across independent trainings", async () => {
    const sequences = synthesizeRepetitions(editTask, 2);
    const dataset = buildMovementDataset(sequences);
    const backend = new DeterministicMovementBackend();
    const a = await backend.train(dataset);
    const b = await backend.train(dataset);
    const context = sequences[0].events.slice(0, 3);
    expect(a.predict(context)).toEqual(b.predict(context));
  });
});

describe("generalization to new but related movements", () => {
  it("predicts correct next actions on a reworded (unseen) variant via similarity", async () => {
    const trainSequences = synthesizeRepetitions(editTask, 2);
    const dataset = buildMovementDataset(trainSequences);

    // A related task: same actions, reworded observations -> unseen exact context.
    const heldout = [synthesizeSequence(relatedVariant(editTask))];

    const { model, generalization } = await trainMovementModel({
      dataset,
      trainingSequences: trainSequences,
      holdout: heldout,
    });

    expect(generalization).toBeDefined();
    // Tool-level fidelity should generalize well even though exact contexts are new.
    expect(generalization!.toolFidelity).toBeGreaterThanOrEqual(0.75);
    // At least some predictions came from the similarity (generalization) path.
    expect(generalization!.bySource.similarity + generalization!.bySource.ngram).toBe(generalization!.steps);
    expect(generalization!.bySource.similarity).toBeGreaterThan(0);

    // Direct probe: from the reworded first observation, the model still predicts "open file".
    const probeContext: MovementEvent[] = [
      { kind: "observation", source: "os", summary: "focused editor (again)", ts: 0 },
    ];
    const prediction = model.predict(probeContext);
    expect(prediction.action?.tool).toBe("keyboard");
    expect(prediction.action?.summary).toBe("open file");
    expect(prediction.source).toBe("similarity");
    expect(prediction.confidence).toBeGreaterThan(0);
  });

  it("falls back to the global prior when nothing is similar", async () => {
    const dataset = buildMovementDataset(synthesizeRepetitions(editTask, 1));
    const { model } = await trainMovementModel({ dataset });
    const alien: MovementEvent[] = [
      { kind: "observation", source: "browser", summary: "completely unrelated page zzz", ts: 0 },
    ];
    const prediction = model.predict(alien);
    // No token overlap -> prior (or none if empty). Dataset is non-empty so prior.
    expect(prediction.source).toBe("prior");
    expect(prediction.action).toBeDefined();
  });

  it("returns an empty prediction for an empty model", async () => {
    const empty = buildMovementDataset([]);
    const { model } = await trainMovementModel({ dataset: empty });
    const prediction = model.predict([{ kind: "observation", source: "os", summary: "anything", ts: 0 }]);
    expect(prediction.source).toBe("none");
    expect(prediction.action).toBeUndefined();
    expect(prediction.confidence).toBe(0);
  });
});

describe("backend pluggability", () => {
  it("accepts an alternate backend implementing MovementModelBackend", async () => {
    // A trivial constant backend proves the interface seam works end-to-end.
    class ConstantBackend implements MovementModelBackend {
      readonly name = "constant";
      async train(): Promise<MovementModel> {
        return {
          backend: this.name,
          predict: () => ({
            action: { tool: "noop", summary: "constant" },
            label: "noop::constant",
            confidence: 1,
            source: "prior" as const,
            alternatives: [],
          }),
          serialize: () => ({
            backend: this.name,
            version: 1 as const,
            maxOrder: 0,
            exampleCount: 0,
            actionVocabulary: [],
          }),
        };
      }
    }

    const dataset = buildMovementDataset(synthesizeRepetitions(editTask, 1));
    const { model } = await trainMovementModel({ dataset, backend: new ConstantBackend() });
    expect(model.backend).toBe("constant");
    expect(model.predict([]).action?.tool).toBe("noop");
  });
});

describe("evaluateMovementModel", () => {
  it("reports zero steps for sequences with no actions", async () => {
    const dataset = buildMovementDataset(synthesizeRepetitions(editTask, 1));
    const { model } = await trainMovementModel({ dataset });
    const result = evaluateMovementModel(model, [
      { events: [{ kind: "observation", source: "os", summary: "idle", ts: 0 }] },
    ]);
    expect(result.steps).toBe(0);
    expect(result.fidelity).toBe(0);
  });
});

describe("actionLabel", () => {
  it("normalizes whitespace and case so related movements share a label", () => {
    expect(actionLabel({ tool: "Keyboard", summary: "Save  File" })).toBe(
      actionLabel({ tool: "keyboard", summary: "save file" }),
    );
  });
});
