import { describe, expect, it } from "vitest";
import { buildTrajectorySpan } from "../capture/trajectory.js";
import {
  MovementModel,
  datasetFromTrajectories,
  trainMovementModel,
  trajectoryToMovementSequence,
  type MovementDataset,
  type MovementToken,
} from "./movement-model.js";

function obs(source: string, summary: string): MovementToken {
  return { role: "observation", source, summary };
}
function act(tool: string, summary: string): MovementToken {
  return { role: "action", tool, summary };
}

const browserFlow: MovementToken[] = [
  obs("window", "open browser"),
  act("mouse", "click(new-tab)"),
  act("keyboard", "type(url)"),
  act("keyboard", "press(enter)"),
];
const editorFlow: MovementToken[] = [
  obs("window", "open editor"),
  act("mouse", "click(new-tab)"),
  act("keyboard", "type(path)"),
  act("keyboard", "press(enter)"),
];

describe("MovementModel", () => {
  it("repeats a recorded movement sequence exactly (objective 2c)", () => {
    const model = trainMovementModel({ sequences: [{ tokens: browserFlow }] });
    const produced = model.generate([obs("window", "open browser")]);
    expect(produced.map((token) => token.summary)).toEqual([
      "click(new-tab)",
      "type(url)",
      "press(enter)",
    ]);
  });

  it("stops on the learned end-of-sequence rather than looping forever", () => {
    const model = trainMovementModel({ sequences: [{ tokens: browserFlow }] });
    const produced = model.generate([obs("window", "open browser")], 100);
    expect(produced).toHaveLength(3);
  });

  it("generalizes to a novel-but-related observation context (objective 2d)", () => {
    // Two flows share the same action skeleton but different observations.
    const model = trainMovementModel({
      sequences: [{ tokens: browserFlow }, { tokens: editorFlow }],
    });
    // Seed with an observation never seen in training.
    const produced = model.generate([obs("window", "open terminal")]);
    // Backs off to the action skeleton: first action + shared tail structure.
    expect(produced[0]?.summary).toBe("click(new-tab)");
    expect(produced.at(-1)?.summary).toBe("press(enter)");
    expect(produced.length).toBeGreaterThanOrEqual(3);
  });

  it("scores teacher-forced fidelity on a held-out related sequence", () => {
    const model = trainMovementModel({
      sequences: [{ tokens: browserFlow }, { tokens: editorFlow }],
    });
    // Held-out flow reuses the shared skeleton with a new observation.
    const heldOut: MovementToken[] = [
      obs("window", "open notes"),
      act("mouse", "click(new-tab)"),
      act("keyboard", "type(url)"),
      act("keyboard", "press(enter)"),
    ];
    const fidelity = model.evaluateFidelity(heldOut);
    expect(fidelity.predicted).toBe(3);
    // click(new-tab) and press(enter) are shared structure → predicted correctly.
    expect(fidelity.correct).toBeGreaterThanOrEqual(2);
    expect(fidelity.accuracy).toBeCloseTo(fidelity.correct / fidelity.predicted);
  });

  it("returns no prediction and an empty rollout when untrained", () => {
    const model = new MovementModel();
    expect(model.predictNext([])).toBeUndefined();
    expect(model.generate([obs("window", "hello")])).toEqual([]);
  });

  it("round-trips through JSON serialization", () => {
    const model = trainMovementModel({ sequences: [{ tokens: browserFlow }] });
    const restored = MovementModel.fromJSON(JSON.parse(JSON.stringify(model.toJSON())));
    expect(restored.generate([obs("window", "open browser")]).map((t) => t.summary)).toEqual(
      model.generate([obs("window", "open browser")]).map((t) => t.summary),
    );
    expect(restored.getOrder()).toBe(model.getOrder());
  });

  it("builds a movement dataset from captured trajectory spans, ordered by ts", () => {
    const span = buildTrajectorySpan({
      id: "traj-1",
      sessionId: "sess-1",
      observations: [{ kind: "observation", source: "window", summary: "open browser", ts: 10 }],
      actions: [
        { kind: "action", tool: "keyboard", summary: "press(enter)", ts: 40 },
        { kind: "action", tool: "mouse", summary: "click(new-tab)", ts: 20 },
      ],
    });
    const sequence = trajectoryToMovementSequence(span);
    expect(sequence.tokens.map((token) => (token.role === "action" ? token.summary : `obs:${token.summary}`))).toEqual([
      "obs:open browser",
      "click(new-tab)",
      "press(enter)",
    ]);

    const dataset: MovementDataset = datasetFromTrajectories([span]);
    expect(dataset.sequences).toHaveLength(1);
    const model = trainMovementModel(dataset);
    expect(model.generate([obs("window", "open browser")]).map((t) => t.summary)).toEqual([
      "click(new-tab)",
      "press(enter)",
    ]);
  });
});
