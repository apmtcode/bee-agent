import { describe, expect, it } from "vitest";
import { buildTrajectorySpan } from "../capture/trajectory.js";
import {
  actionToMovementToken,
  buildMovementDataset,
  movementTokenFromKey,
  movementTokenKey,
  normalizeMovementTarget,
  observationToMovementToken,
  tokenizeTrajectory,
  type MovementSequence,
} from "./movement-dataset.js";
import { sequenceToTrajectory, synthesizeMovementSequences } from "./synthetic-movements.js";

describe("normalizeMovementTarget", () => {
  it("slugifies and lowercases targets", () => {
    expect(normalizeMovementTarget("  Send Button ")).toBe("send-button");
    expect(normalizeMovementTarget("Repo/Deploy.sh")).toBe("repo/deploy.sh");
  });

  it("returns wildcard for empty/absent input", () => {
    expect(normalizeMovementTarget(undefined)).toBe("*");
    expect(normalizeMovementTarget("   ")).toBe("*");
    expect(normalizeMovementTarget("!!!")).toBe("*");
  });
});

describe("token keys", () => {
  it("round-trips a token through its canonical key", () => {
    const token = { modality: "pointer", verb: "scroll", target: "*", detail: "down" } as const;
    expect(movementTokenFromKey(movementTokenKey(token))).toEqual(token);
  });

  it("never collides across different token boundaries", () => {
    const a = movementTokenKey({ modality: "pointer", verb: "tap", target: "a" });
    const b = movementTokenKey({ modality: "pointer", verb: "tap", target: "b" });
    expect(a).not.toBe(b);
  });
});

describe("captured event -> movement token", () => {
  it("maps device gestures to pointer/keyboard tokens", () => {
    expect(
      actionToMovementToken({ kind: "action", tool: "device", summary: "", ts: 1, metadata: { gesture: "tap", target: "Send" } }),
    ).toEqual({ modality: "pointer", verb: "tap", target: "send" });
    expect(
      actionToMovementToken({ kind: "action", tool: "device", summary: "", ts: 1, metadata: { gesture: "scroll", direction: "down" } }),
    ).toEqual({ modality: "pointer", verb: "scroll", target: "*", detail: "down" });
    expect(
      actionToMovementToken({ kind: "action", tool: "device", summary: "", ts: 1, metadata: { gesture: "type", target: "body" } }),
    ).toEqual({ modality: "keyboard", verb: "type", target: "body" });
  });

  it("maps os observations to window/command tokens", () => {
    expect(
      observationToMovementToken({ kind: "observation", source: "os", summary: "", ts: 1, metadata: { event: "focus-changed", windowTitle: "Terminal" } }),
    ).toEqual({ modality: "window", verb: "focus", target: "terminal" });
    expect(
      observationToMovementToken({ kind: "observation", source: "os", summary: "", ts: 1, metadata: { event: "command-ran", commandSummary: "git pull" } }),
    ).toEqual({ modality: "command", verb: "run", target: "git-pull" });
  });

  it("ignores observations that are not movements", () => {
    expect(
      observationToMovementToken({ kind: "observation", source: "device", summary: "app active", ts: 1, metadata: {} }),
    ).toBeUndefined();
  });
});

describe("tokenizeTrajectory", () => {
  it("orders observations and actions by timestamp", () => {
    const span = buildTrajectorySpan({
      id: "t1",
      sessionId: "s1",
      observations: [{ kind: "observation", source: "os", summary: "", ts: 10, metadata: { event: "focus-changed", windowTitle: "Mail" } }],
      actions: [
        { kind: "action", tool: "device", summary: "", ts: 30, metadata: { gesture: "tap", target: "send" } },
        { kind: "action", tool: "device", summary: "", ts: 20, metadata: { gesture: "type", target: "body" } },
      ],
    });
    const sequence = tokenizeTrajectory(span);
    expect(sequence.tokens.map((token) => `${token.verb}:${token.target}`)).toEqual(["focus:mail", "type:body", "tap:send"]);
  });

  it("drops empty sequences when building a dataset", () => {
    const empty = buildTrajectorySpan({ id: "empty", sessionId: "s" });
    const dataset = buildMovementDataset([empty]);
    expect(dataset.sequences).toHaveLength(0);
  });
});

describe("capture <-> dataset round-trip", () => {
  const fixture: MovementSequence = {
    id: "fixture",
    tokens: [
      { modality: "window", verb: "open", target: "mail" },
      { modality: "pointer", verb: "tap", target: "compose" },
      { modality: "keyboard", verb: "type", target: "recipient" },
      { modality: "pointer", verb: "scroll", target: "*", detail: "down" },
      { modality: "command", verb: "run", target: "deploy" },
    ],
  };

  it("recovers the exact token sequence from an emitted trajectory", () => {
    const span = sequenceToTrajectory(fixture);
    expect(tokenizeTrajectory(span).tokens).toEqual(fixture.tokens);
  });

  it("round-trips every synthetic sequence losslessly", () => {
    for (const sequence of synthesizeMovementSequences({ seed: 7, variantsPerFamily: 3 })) {
      const span = sequenceToTrajectory(sequence);
      expect(tokenizeTrajectory(span).tokens).toEqual(sequence.tokens);
    }
  });
});
