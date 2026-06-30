import { describe, expect, it } from "vitest";
import { buildMovementDataset } from "./events.js";
import { decodeDatasetJsonl, encodeDatasetJsonl } from "./dataset.js";
import { generateSyntheticDataset } from "./synthetic.js";

describe("movement dataset codec", () => {
  it("round-trips a synthetic dataset losslessly", () => {
    const dataset = generateSyntheticDataset({ intent: "drag", count: 3, seed: 5 });
    const decoded = decodeDatasetJsonl(encodeDatasetJsonl(dataset));
    expect(decoded).toEqual(dataset);
  });

  it("emits one header line plus one line per demonstration", () => {
    const dataset = generateSyntheticDataset({ intent: "drag", count: 2, seed: 1 });
    const lines = encodeDatasetJsonl(dataset).trimEnd().split("\n");
    expect(lines).toHaveLength(3);
    expect(JSON.parse(lines[0]!)).toMatchObject({ kind: "header", version: 1 });
  });

  it("decodes an empty dataset", () => {
    expect(decodeDatasetJsonl("")).toEqual(buildMovementDataset([]));
  });

  it("rejects input without a header record", () => {
    expect(() => decodeDatasetJsonl(`{"id":"x"}`)).toThrow(/header/);
  });

  it("rejects an unsupported version", () => {
    expect(() => decodeDatasetJsonl(`{"kind":"header","version":2}`)).toThrow(/version/);
  });
});
