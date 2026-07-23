import { describe, expect, it } from "vitest";
import {
  LocalModelBackendRegistry,
  buildMovementSequences,
  evaluateReplayFidelity,
  tokenizeReplayEvent,
  type MovementSequence,
} from "./backend.js";
import {
  MarkovMovementBackend,
  createDefaultLocalModelBackendRegistry,
} from "./backends/markov-backend.js";
import type { ReviewedExportManifest } from "./export-manifest.js";

function seq(id: string, tokens: string[]): MovementSequence {
  return { id, tokens };
}

/** Deterministic synthetic movement stream: a repeatable "open → type → save" loop. */
function syntheticSequences(count: number): MovementSequence[] {
  const base = [
    "action:focus-window",
    "action:click",
    "action:type",
    "action:type",
    "action:save",
  ];
  return Array.from({ length: count }, (_, index) =>
    seq(`traj-${index}`, index % 2 === 0 ? base : [...base, "action:confirm"]),
  );
}

describe("movement tokenizer + sequence builder", () => {
  it("maps each replay event kind to a stable token", () => {
    expect(tokenizeReplayEvent({ kind: "action", ts: 1, trajectoryId: "t", tool: "click", summary: "" })).toBe(
      "action:click",
    );
    expect(
      tokenizeReplayEvent({ kind: "observation", ts: 1, trajectoryId: "t", source: "screen", summary: "" }),
    ).toBe("obs:screen");
    expect(
      tokenizeReplayEvent({ kind: "transcript", ts: 1, messageId: "m", role: "assistant", content: "" }),
    ).toBe("msg:assistant");
  });

  it("builds one time-ordered token sequence per reviewed replay", () => {
    const manifest = {
      replays: [
        {
          sessionId: "s1",
          trajectoryIds: ["traj-a"],
          eventCount: 2,
          events: [
            { kind: "observation", ts: 1, trajectoryId: "traj-a", source: "screen", summary: "" },
            { kind: "action", ts: 2, trajectoryId: "traj-a", tool: "click", summary: "" },
          ],
        },
      ],
    } as unknown as ReviewedExportManifest;

    expect(buildMovementSequences(manifest)).toEqual([
      { id: "traj-a", tokens: ["obs:screen", "action:click"] },
    ]);
  });
});

describe("LocalModelBackendRegistry", () => {
  it("registers, resolves, and lists backends; throws on unknown", () => {
    const registry = new LocalModelBackendRegistry().register(new MarkovMovementBackend());
    expect(registry.has("markov")).toBe(true);
    expect(registry.list()).toEqual(["markov"]);
    expect(registry.get("markov").name).toBe("markov");
    expect(() => registry.get("nope")).toThrow(/unknown local-model backend/);
  });

  it("default registry is pre-seeded with the deterministic backend", () => {
    expect(createDefaultLocalModelBackendRegistry().list()).toEqual(["markov"]);
  });
});

describe("MarkovMovementBackend", () => {
  const backend = new MarkovMovementBackend();

  it("trains a deterministic, serializable artifact", async () => {
    const sequences = syntheticSequences(4);
    const a = await backend.train({ jobId: "job-1", mode: "sft", sequences, order: 2 });
    const b = await backend.train({ jobId: "job-1", mode: "sft", sequences, order: 2 });

    expect(a.backend).toBe("markov");
    expect(a.order).toBe(2);
    expect(a.sequenceCount).toBe(4);
    expect(a.vocabulary).toContain("action:save");
    // JSON round-trips (serializable) and is stable across identical runs.
    expect(JSON.parse(JSON.stringify(a.weights))).toEqual(b.weights);
  });

  it("repeats a recorded movement sequence from its opening prefix", async () => {
    const recorded = ["action:focus-window", "action:click", "action:type", "action:save"];
    const model = await backend.train({
      jobId: "job-2",
      mode: "sft",
      sequences: [seq("traj-0", recorded)],
      order: 2,
    });

    const result = await backend.infer(model, { prompt: recorded.slice(0, 1), maxTokens: 16 });
    expect(result.tokens).toEqual(recorded.slice(1));
    expect(result.terminated).toBe(true); // stopped at learned end-of-sequence
  });

  it("generalizes to an unseen but related prompt via context backoff", async () => {
    const model = await backend.train({
      jobId: "job-3",
      mode: "sft",
      sequences: syntheticSequences(6),
      order: 2,
    });

    // "action:type" was always followed by another type or a save in training,
    // even though this exact prompt prefix was never recorded verbatim.
    const result = await backend.infer(model, { prompt: ["action:type"], maxTokens: 4 });
    expect(result.tokens.length).toBeGreaterThan(0);
    expect(result.tokens[0]).toMatch(/^action:(type|save)$/);
  });

  it("falls back to the unigram prior for a wholly unknown prompt", async () => {
    const model = await backend.train({
      jobId: "job-4",
      mode: "sft",
      sequences: syntheticSequences(4),
      order: 2,
    });

    const result = await backend.infer(model, { prompt: ["action:never-seen"], maxTokens: 1 });
    expect(result.usedBackoff).toBe(true);
    expect(model.vocabulary).toContain(result.tokens[0]!);
  });
});

describe("evaluateReplayFidelity", () => {
  const backend = new MarkovMovementBackend();

  it("reports near-perfect fidelity when repeating the training set", async () => {
    const sequences = syntheticSequences(4);
    const model = await backend.train({ jobId: "job-5", mode: "sft", sequences, order: 2 });
    const report = await evaluateReplayFidelity(backend, model, sequences);

    expect(report.sequenceCount).toBe(4);
    expect(report.evaluatedTokens).toBeGreaterThan(0);
    expect(report.accuracy).toBeGreaterThan(0.9);
    expect(report.perSequence).toHaveLength(4);
  });

  it("generalizes above chance to held-out related sequences", async () => {
    const train = syntheticSequences(6);
    const model = await backend.train({ jobId: "job-6", mode: "sft", sequences: train, order: 2 });

    // Held-out sequence: same vocabulary/structure, never trained on verbatim.
    const heldOut = [seq("held-0", ["action:focus-window", "action:click", "action:type", "action:save"])];
    const report = await evaluateReplayFidelity(backend, model, heldOut);

    // Random next-token accuracy over a ~5-symbol vocab would be ~0.2.
    expect(report.accuracy).toBeGreaterThan(0.5);
  });
});
