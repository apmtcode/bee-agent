import { describe, expect, it } from "vitest";
import {
  MarkovMovementBackend,
  buildMovementDataset,
  describeBackendSeam,
  type MovementDataset,
} from "./model-backend.js";
import type { TrajectorySpan } from "../capture/trajectory.js";

function seq(id: string, tokens: string[]) {
  return { id, tokens };
}

function datasetOf(...sequences: { id: string; tokens: string[] }[]): MovementDataset {
  return { sequences };
}

describe("MarkovMovementBackend", () => {
  it("learns to repeat a recorded movement sequence verbatim", async () => {
    const recorded = ["act:focus", "act:click", "act:type", "act:submit"];
    const model = await new MarkovMovementBackend().train(datasetOf(seq("t1", recorded)));

    // Generating from an empty seed reproduces the whole recorded movement.
    expect(model.generate([], 10)).toEqual(recorded);
  });

  it("continues a recorded sequence from a partial seed", async () => {
    const recorded = ["act:focus", "act:click", "act:type", "act:submit"];
    const model = await new MarkovMovementBackend().train(datasetOf(seq("t1", recorded)));

    expect(model.generate(["act:focus", "act:click"], 10)).toEqual(["act:type", "act:submit"]);
  });

  it("generalizes to an unseen prefix via backoff", async () => {
    // Two recordings share the suffix pattern click -> type -> submit.
    const model = await new MarkovMovementBackend().train(
      datasetOf(
        seq("t1", ["act:openA", "act:click", "act:type", "act:submit"]),
        seq("t2", ["act:openB", "act:click", "act:type", "act:submit"]),
      ),
    );

    // A brand-new opener never paired with click at full order still predicts
    // the learned continuation by backing off to the shorter `click` context.
    const prediction = model.predictNext(["act:openC", "act:click"]);
    expect(prediction?.token).toBe("act:type");
    // Backoff happened: full 2-token context (openC,click) was never seen.
    expect(prediction?.backoffOrder).toBeLessThan(2);
  });

  it("is deterministic across repeated training and prediction", async () => {
    const data = datasetOf(
      seq("a", ["act:x", "act:y", "act:z"]),
      seq("b", ["act:x", "act:y", "act:w"]),
    );
    const first = (await new MarkovMovementBackend().train(data)).predictNext(["act:x", "act:y"]);
    const second = (await new MarkovMovementBackend().train(data)).predictNext(["act:x", "act:y"]);
    expect(first).toEqual(second);
    // Tie (y->z and y->w each seen once): deterministic lexicographic winner.
    expect(first?.token).toBe("act:w");
    expect(first?.alternatives[0]?.token).toBe("act:z");
  });

  it("reports calibrated probabilities and ranked alternatives", async () => {
    const model = await new MarkovMovementBackend().train(
      datasetOf(
        seq("1", ["act:a", "act:b"]),
        seq("2", ["act:a", "act:b"]),
        seq("3", ["act:a", "act:c"]),
      ),
    );
    const prediction = model.predictNext(["act:a"]);
    expect(prediction?.token).toBe("act:b");
    expect(prediction?.probability).toBeCloseTo(2 / 3);
    expect(prediction?.alternatives).toEqual([{ token: "act:c", probability: 1 / 3 }]);
  });

  it("returns undefined for a context with no learned continuation", async () => {
    const model = await new MarkovMovementBackend().train(datasetOf(seq("1", ["act:a"])));
    expect(model.predictNext(["act:unknown"])).toBeUndefined();
  });

  it("never emits BOS/EOS sentinels from generate", async () => {
    const model = await new MarkovMovementBackend().train(datasetOf(seq("1", ["act:a", "act:b"])));
    const produced = model.generate([], 20);
    expect(produced).toEqual(["act:a", "act:b"]);
    expect(produced.some((token) => token.includes("^") || token.includes("$"))).toBe(false);
  });

  it("round-trips through serialize/restore without behaviour change", async () => {
    const data = datasetOf(seq("1", ["act:a", "act:b", "act:c"]));
    const trained = await new MarkovMovementBackend().train(data);
    const restored = MarkovMovementBackend.restore(trained.serialize());
    expect(restored.order).toBe(trained.order);
    expect(restored.generate([], 10)).toEqual(trained.generate([], 10));
    expect(restored.predictNext(["act:a"])).toEqual(trained.predictNext(["act:a"]));
  });

  it("higher order favours verbatim recall over generalization", async () => {
    // At order 1 the two branches after `mid` collide; order 2 separates them.
    const data = datasetOf(
      seq("1", ["act:p1", "act:mid", "act:endA"]),
      seq("2", ["act:p2", "act:mid", "act:endB"]),
    );
    const order2 = await new MarkovMovementBackend().train(data, { order: 2 });
    expect(order2.predictNext(["act:p1", "act:mid"])?.token).toBe("act:endA");
    expect(order2.predictNext(["act:p2", "act:mid"])?.token).toBe("act:endB");
  });
});

describe("buildMovementDataset", () => {
  const trajectory: TrajectorySpan = {
    id: "traj-1",
    sessionId: "s1",
    createdAt: "2026-07-15T00:00:00.000Z",
    captureTier: "operator",
    observations: [{ kind: "observation", source: "screen", summary: "form visible", ts: 5 }],
    actions: [
      { kind: "action", tool: "click", summary: "click submit", ts: 20 },
      { kind: "action", tool: "focus", summary: "focus field", ts: 10 },
    ],
  };

  it("orders actions by timestamp and tokenizes by tool", () => {
    const dataset = buildMovementDataset([trajectory]);
    expect(dataset.sequences).toEqual([{ id: "traj-1", tokens: ["act:focus", "act:click"] }]);
  });

  it("interleaves observations when requested, ordered by ts", () => {
    const dataset = buildMovementDataset([trajectory], { includeObservations: true });
    expect(dataset.sequences[0]?.tokens).toEqual(["obs:screen", "act:focus", "act:click"]);
  });

  it("feeds an end-to-end train + generate round-trip", async () => {
    const dataset = buildMovementDataset([trajectory]);
    const model = await new MarkovMovementBackend().train(dataset);
    expect(model.generate([], 5)).toEqual(["act:focus", "act:click"]);
  });
});

describe("describeBackendSeam", () => {
  it("documents the backend contract as data", () => {
    const seam = describeBackendSeam();
    expect(seam.interface).toBe("MovementModelBackend");
    expect(seam.contract.length).toBeGreaterThan(0);
  });
});
