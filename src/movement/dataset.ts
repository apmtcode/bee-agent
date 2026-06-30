/**
 * Replayable dataset codec.
 *
 * Persists a {@link MovementDataset} as JSON Lines: the first line is a header
 * (`{ kind: "header", version }`), each subsequent line is one demonstration.
 * JSONL keeps datasets append-friendly and streamable for large captures while
 * staying trivially diffable and replayable.
 */

import {
  buildMovementDataset,
  type MovementDataset,
  type MovementDemonstration,
} from "./events.js";

type HeaderLine = { kind: "header"; version: 1 };

export function encodeDatasetJsonl(dataset: MovementDataset): string {
  const header: HeaderLine = { kind: "header", version: dataset.version };
  const lines = [JSON.stringify(header)];
  for (const demonstration of dataset.demonstrations) {
    lines.push(JSON.stringify(demonstration));
  }
  return lines.join("\n") + "\n";
}

export function decodeDatasetJsonl(text: string): MovementDataset {
  const lines = text.split("\n").map((line) => line.trim()).filter((line) => line.length > 0);
  if (lines.length === 0) {
    return buildMovementDataset([]);
  }
  const [headerLine, ...rest] = lines;
  const header = JSON.parse(headerLine!) as Partial<HeaderLine>;
  if (header.kind !== "header") {
    throw new Error("movement dataset: first line must be a header record");
  }
  if (header.version !== 1) {
    throw new Error(`movement dataset: unsupported version ${String(header.version)}`);
  }
  const demonstrations = rest.map((line) => JSON.parse(line) as MovementDemonstration);
  return buildMovementDataset(demonstrations);
}
