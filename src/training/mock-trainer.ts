import fs from "node:fs/promises";
import path from "node:path";
import { writeJsonAtomic, readJsonFile } from "../shared/fs.js";
import { trainMockMovementModel, type MockMovementModel } from "./backends.js";

/**
 * Dataset contract consumed by the mock trainer. A `movements.json` file in the
 * dataset directory carries the recorded movement token sequences (produced by
 * the capture pipeline or the synthetic event-stream generator). Each inner
 * array is one ordered movement session (e.g. per trajectory).
 */
export type MockMovementDataset = {
  version: 1;
  sequences: string[][];
};

export type MockTrainerResult = {
  model: MockMovementModel;
  outPath: string;
  datasetPath: string;
  mode: string;
};

/**
 * Train the mock movement model from a dataset directory and persist it.
 * Fully deterministic and dependency-free so it runs anywhere `node` runs
 * (cloud/CI included). Reads `${datasetDir}/movements.json`; an absent or empty
 * dataset yields an empty (but valid) model rather than throwing.
 */
export async function runMockTrainer(params: {
  datasetDir: string;
  outPath: string;
  mode?: string;
}): Promise<MockTrainerResult> {
  const datasetPath = path.join(params.datasetDir, "movements.json");
  const dataset = await readJsonFile<MockMovementDataset | undefined>(datasetPath, undefined);
  const sequences = Array.isArray(dataset?.sequences) ? dataset.sequences : [];
  const model = trainMockMovementModel(sequences);
  await writeJsonAtomic(params.outPath, model);
  return { model, outPath: params.outPath, datasetPath, mode: params.mode ?? "sft" };
}

/** Minimal argv parser: `--flag value` pairs. */
function parseArgs(argv: string[]): Record<string, string> {
  const args: Record<string, string> = {};
  for (let index = 0; index < argv.length; index += 1) {
    const token = argv[index];
    if (token.startsWith("--")) {
      const key = token.slice(2);
      const value = argv[index + 1];
      if (value !== undefined && !value.startsWith("--")) {
        args[key] = value;
        index += 1;
      } else {
        args[key] = "true";
      }
    }
  }
  return args;
}

async function main(): Promise<void> {
  const args = parseArgs(process.argv.slice(2));
  const datasetDir = args.dataset;
  const outPath = args.out;
  if (!datasetDir || !outPath) {
    process.stderr.write("usage: mock-trainer --dataset <dir> --out <file> [--mode sft|rl]\n");
    process.exitCode = 2;
    return;
  }
  const result = await runMockTrainer({ datasetDir, outPath, mode: args.mode });
  await fs.mkdir(path.dirname(outPath), { recursive: true }).catch(() => undefined);
  process.stdout.write(
    `mock-trainer: trained markov-1 model on ${result.model.sequenceCount} sequence(s), ` +
      `${result.model.transitionCount} transition(s) -> ${outPath}\n`,
  );
}

// CLI entry: only runs when executed directly, not when imported.
const invokedPath = process.argv[1] ? path.resolve(process.argv[1]) : "";
const thisPath = (() => {
  try {
    return path.resolve(new URL(import.meta.url).pathname);
  } catch {
    return "";
  }
})();
if (invokedPath && thisPath && invokedPath === thisPath) {
  void main();
}
