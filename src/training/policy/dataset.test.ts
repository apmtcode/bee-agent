import { describe, expect, it } from "vitest";
import { buildTrajectorySpan } from "../../capture/trajectory.js";
import { buildMovementDataset, labelAction, observationTokens, trajectoryToExamples } from "./dataset.js";
import { actionKey, contextKey } from "./model.js";

function span() {
  return buildTrajectorySpan({
    id: "t1",
    sessionId: "s1",
    captureTier: "full",
    observations: [
      { kind: "observation", source: "os", summary: "mail app active", ts: 10, metadata: { event: "focus-changed", appName: "mail" } },
      { kind: "observation", source: "device", summary: "compose editor open", ts: 30, metadata: { appName: "mail", screenTitle: "compose" } },
    ],
    actions: [
      { kind: "action", tool: "device", summary: "tapped compose", ts: 20, metadata: { gesture: "tap", target: "compose" } },
      { kind: "action", tool: "device", summary: "typed into body", ts: 40, metadata: { gesture: "type", target: "body" } },
    ],
  });
}

describe("labelAction", () => {
  it("prefers gesture + target metadata over the summary", () => {
    expect(labelAction({ kind: "action", tool: "device", summary: "tapped compose", ts: 1, metadata: { gesture: "tap", target: "compose" } })).toEqual({
      tool: "device",
      descriptor: "tap:compose",
    });
  });

  it("falls back to a slug of the summary when no discriminating metadata", () => {
    expect(labelAction({ kind: "action", tool: "browser", summary: "Clicked the Submit button", ts: 1 })).toEqual({
      tool: "browser",
      descriptor: "clicked-submit-button",
    });
  });
});

describe("observationTokens", () => {
  it("emits a source token plus summary + metadata feature tokens", () => {
    const tokens = observationTokens({ kind: "observation", source: "os", summary: "mail app active", ts: 1, metadata: { event: "focus-changed", appName: "mail" } });
    expect(tokens).toContain("src:os");
    expect(tokens).toContain("mail");
    expect(tokens).toContain("event:focus-changed");
    expect(tokens).toContain("appName:mail");
  });
});

describe("trajectoryToExamples", () => {
  it("emits one example per action, chronologically, with a prev-action token", () => {
    const examples = trajectoryToExamples(span());
    expect(examples).toHaveLength(2);

    // First action's context: the first observation, no prev action.
    expect(actionKey(examples[0].action)).toBe("device tap:compose");
    expect(examples[0].context.tokens).toContain("event:focus-changed");
    expect(examples[0].context.tokens.some((t) => t.startsWith("prev:"))).toBe(false);

    // Second action's context: the second observation + a prev token for action 1.
    expect(actionKey(examples[1].action)).toBe("device type:body");
    expect(examples[1].context.tokens).toContain("prev:device tap:compose");
    expect(examples[1].context.tokens).toContain("screenTitle:compose");
  });

  it("uses a positive outcome reward as the example weight", () => {
    const rewarded = buildTrajectorySpan({
      id: "t2",
      sessionId: "s1",
      observations: [{ kind: "observation", source: "os", summary: "ready", ts: 1 }],
      actions: [{ kind: "action", tool: "device", summary: "tap go", ts: 2, metadata: { gesture: "tap", target: "go" } }],
      outcome: { status: "success", summary: "done", reward: 3 },
    });
    expect(trajectoryToExamples(rewarded)[0].weight).toBe(3);
  });
});

describe("buildMovementDataset", () => {
  it("collects a sorted, de-duplicated action vocabulary", () => {
    const dataset = buildMovementDataset([span()]);
    expect(dataset.actionVocabulary).toEqual(["device tap:compose", "device type:body"]);
    expect(dataset.examples).toHaveLength(2);
  });

  it("produces stable context keys for equivalent situations", () => {
    const a = buildMovementDataset([span()]);
    const b = buildMovementDataset([span()]);
    expect(contextKey(a.examples[0].context)).toBe(contextKey(b.examples[0].context));
  });
});
