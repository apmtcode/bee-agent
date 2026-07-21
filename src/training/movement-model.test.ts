import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import type { ReplayTimelineEvent } from "../capture/replay.js";
import {
  MarkovMovementBackend,
  MovementModelTrainer,
  buildMovementDataset,
  tokenizeMovementEvent,
  type MovementDataset,
  type MovementModelBackend,
  type MovementPrediction,
} from "./movement-model.js";

let ts = 0;

function action(tool: string): ReplayTimelineEvent {
  ts += 1;
  return { kind: "action", ts, trajectoryId: "t", tool, summary: `${tool} @${ts}` };
}

function observation(source: string): ReplayTimelineEvent {
  ts += 1;
  return { kind: "observation", ts, trajectoryId: "t", source, summary: `${source} @${ts}` };
}

function tokensOf(events: ReplayTimelineEvent[]): string[] {
  return events.map(tokenizeMovementEvent);
}

beforeEach(() => {
  ts = 0;
});

describe("tokenizeMovementEvent", () => {
  it("produces canonical, text-free tokens per event kind", () => {
    expect(tokenizeMovementEvent(action("mouse.move"))).toBe("action:mouse.move");
    expect(tokenizeMovementEvent(observation("screen"))).toBe("observation:screen");
    expect(
      tokenizeMovementEvent({ kind: "transcript", ts: 1, messageId: "m", role: "user", content: "secret" }),
    ).toBe("transcript:user");
  });
});

describe("MarkovMovementBackend — memorization (repeat recorded movements)", () => {
  it("reproduces a recorded movement sequence exactly via rollout", () => {
    const recorded = [
      observation("screen"),
      action("mouse.move"),
      action("mouse.click"),
      action("keyboard.type"),
    ];
    const dataset: MovementDataset = { version: 1, samples: [{ events: recorded }] };
    const trainer = new MovementModelTrainer(".", new MarkovMovementBackend(2));
    const model = trainer.train(dataset);

    const generated = trainer.generate(model);
    expect(tokensOf(generated)).toEqual(tokensOf(recorded));
  });

  it("advances timestamps monotonically in generated rollouts", () => {
    const recorded = [observation("screen"), action("mouse.move"), action("mouse.click")];
    const trainer = new MovementModelTrainer(".", new MarkovMovementBackend(2));
    const model = trainer.train({ version: 1, samples: [{ events: recorded }] });
    const generated = trainer.generate(model);
    for (let i = 1; i < generated.length; i += 1) {
      expect(generated[i]!.ts).toBeGreaterThan(generated[i - 1]!.ts);
    }
  });
});

describe("MarkovMovementBackend — generalization (new but related movements)", () => {
  const backend = new MarkovMovementBackend(2);

  function trainedModel() {
    const s1 = [observation("app-x"), action("menu.open"), action("menu.select")];
    const s2 = [observation("app-y"), action("menu.open"), action("menu.select")];
    return backend.train({ version: 1, samples: [{ events: s1 }, { events: s2 }] });
  }

  it("predicts the verbatim continuation at full order without backing off", () => {
    const model = trainedModel();
    const prediction = backend.predictNext(model, [observation("app-x"), action("menu.open")]);
    expect(prediction?.token).toBe("action:menu.select");
    expect(prediction?.backedOff).toBe(false);
    expect(prediction?.order).toBe(2);
  });

  it("generalizes an unseen prefix to the learned continuation via suffix back-off", () => {
    const model = trainedModel();
    // app-z was never seen before menu.open, so the order-2 context is novel;
    // the model must back off to the order-1 suffix [menu.open] -> menu.select.
    const prediction = backend.predictNext(model, [observation("app-z"), action("menu.open")]);
    expect(prediction?.token).toBe("action:menu.select");
    expect(prediction?.backedOff).toBe(true);
    expect(prediction?.order).toBe(1);
  });

  it("falls all the way back to order-0 for a wholly unseen context", () => {
    const model = trainedModel();
    // "never.seen.tool" has no order>=1 context, so the model backs off to the
    // order-0 (empty context) distribution and still predicts a plausible move.
    const prediction = backend.predictNext(model, [action("never.seen.tool")]);
    expect(prediction).toBeDefined();
    expect(prediction?.order).toBe(0);
    expect(prediction?.backedOff).toBe(true);
  });
});

describe("MarkovMovementBackend — deterministic tie-breaking", () => {
  it("produces identical predictions across independent trainings", () => {
    const events = [action("b"), action("a"), action("b"), action("c")];
    const backend = new MarkovMovementBackend(1);
    const m1 = backend.train({ version: 1, samples: [{ events }] });
    const m2 = backend.train({ version: 1, samples: [{ events }] });
    const p1 = backend.predictNext(m1, [action("b")]);
    const p2 = backend.predictNext(m2, [action("b")]);
    expect(p1?.token).toBe(p2?.token);
  });
});

describe("serialize / deserialize round-trip", () => {
  it("preserves predictions after (de)serialization", () => {
    const backend = new MarkovMovementBackend(2);
    const model = backend.train({
      version: 1,
      samples: [{ events: [observation("screen"), action("mouse.move"), action("mouse.click")] }],
    });
    const restored = backend.deserialize(backend.serialize(model));
    const original = backend.predictNext(model, [observation("screen"), action("mouse.move")]);
    const roundTripped = backend.predictNext(restored, [observation("screen"), action("mouse.move")]);
    expect(roundTripped?.token).toBe(original?.token);
    expect(roundTripped?.probability).toBe(original?.probability);
  });

  it("rejects deserializing a non-markov artifact", () => {
    const backend = new MarkovMovementBackend();
    expect(() => backend.deserialize(JSON.stringify({ backend: "mlx" }))).toThrow(/markov/i);
  });
});

describe("buildMovementDataset", () => {
  it("keeps non-empty replays and drops empty ones", () => {
    const dataset = buildMovementDataset([
      { events: [action("a")] },
      { events: [] },
      { events: [observation("s"), action("b")] },
    ]);
    expect(dataset.samples).toHaveLength(2);
    expect(dataset.samples[0]!.events).toHaveLength(1);
  });
});

describe("MovementModelTrainer — persistence", () => {
  let dir: string;

  beforeEach(async () => {
    dir = await mkdtemp(path.join(tmpdir(), "bee-movement-"));
  });

  afterEach(async () => {
    await rm(dir, { recursive: true, force: true });
  });

  it("saves and reloads a model with identical behaviour", async () => {
    const trainer = new MovementModelTrainer(dir, new MarkovMovementBackend(2));
    const model = trainer.train({
      version: 1,
      samples: [{ events: [observation("screen"), action("mouse.move"), action("mouse.click")] }],
    });

    const saved = await trainer.save("model.json", model, "2026-07-21T00:00:00.000Z");
    expect(saved.backend).toBe("markov");
    expect(saved.order).toBe(2);
    expect(saved.sampleCount).toBe(1);
    expect(saved.createdAt).toBe("2026-07-21T00:00:00.000Z");

    const reloaded = await trainer.load("model.json");
    expect(reloaded).toBeDefined();
    const before = trainer.predictNext(model, [observation("screen")]);
    const after = trainer.predictNext(reloaded!, [observation("screen")]);
    expect(after?.token).toBe(before?.token);
  });

  it("returns undefined loading a missing model", async () => {
    const trainer = new MovementModelTrainer(dir, new MarkovMovementBackend());
    expect(await trainer.load("nope.json")).toBeUndefined();
  });

  it("refuses to load a model trained by a different backend", async () => {
    const markovTrainer = new MovementModelTrainer(dir, new MarkovMovementBackend());
    const model = markovTrainer.train({ version: 1, samples: [{ events: [action("a")] }] });
    await markovTrainer.save("model.json", model);

    const otherTrainer = new MovementModelTrainer(dir, new ConstantBackend());
    await expect(otherTrainer.load("model.json")).rejects.toThrow(/backend/i);
  });
});

/** A trivial backend proving the {@link MovementModelBackend} seam is pluggable. */
class ConstantBackend implements MovementModelBackend<{ backend: "const" }> {
  readonly id = "const";
  train(): { backend: "const" } {
    return { backend: "const" };
  }
  predictNext(_model: { backend: "const" }, context: ReplayTimelineEvent[]): MovementPrediction | undefined {
    if (context.length > 0) {
      return undefined;
    }
    return {
      token: "action:noop",
      event: { kind: "action", ts: 0, trajectoryId: "const", tool: "noop", summary: "noop" },
      probability: 1,
      order: 0,
      backedOff: false,
    };
  }
  serialize(model: { backend: "const" }): string {
    return JSON.stringify(model);
  }
  deserialize(data: string): { backend: "const" } {
    return JSON.parse(data) as { backend: "const" };
  }
}

describe("Pluggable backend seam", () => {
  it("drives a custom backend through the trainer's rollout loop", () => {
    const trainer = new MovementModelTrainer(".", new ConstantBackend());
    const model = trainer.train({ version: 1, samples: [] });
    const generated = trainer.generate(model, { maxLength: 10 });
    expect(generated).toHaveLength(1);
    expect(generated[0]!.kind).toBe("action");
  });

  it("respects the rollout maxLength guard", () => {
    // A backend that never stops would loop forever without the cap.
    const endless: MovementModelBackend<null> = {
      id: "endless",
      train: () => null,
      predictNext: (_m, context) => ({
        token: "action:tick",
        event: { kind: "action", ts: context.length, trajectoryId: "e", tool: "tick", summary: "tick" },
        probability: 1,
        order: 0,
        backedOff: false,
      }),
      serialize: () => "null",
      deserialize: () => null,
    };
    const trainer = new MovementModelTrainer(".", endless);
    const generated = trainer.generate(null, { maxLength: 5 });
    expect(generated).toHaveLength(5);
  });
});
