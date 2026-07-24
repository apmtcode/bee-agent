import type { CaptureTier, TrajectorySpan } from "../capture/trajectory.js";
import type { MovementDataset, MovementSequence, MovementStep } from "./movement-model.js";

/**
 * Deterministic synthetic movement-stream generator.
 *
 * The real capture pipeline needs live OS input, which is unavailable in the
 * cloud. This module produces structured, learnable movement streams from a
 * small grammar so the capture -> dataset -> train -> infer -> replay loop and the
 * generalization eval can be validated end-to-end without any real device.
 *
 * Streams are generated from a seed via a linear-congruential RNG, so a given
 * seed always yields the same dataset (reproducible tests, no wall-clock or
 * Math.random dependency).
 */

export type SyntheticTaskFlow = {
  name: string;
  /** Ordered stages; each stage picks one of its step variants per sequence. */
  stages: MovementStep[][];
};

/**
 * A few realistic on-device task flows. Each flow has a stable skeleton with
 * bounded per-stage variation, so an n-gram model can learn the skeleton and
 * generalize across the variants (which is what the eval measures).
 */
export const DEFAULT_SYNTHETIC_FLOWS: SyntheticTaskFlow[] = [
  {
    name: "compose-and-send",
    stages: [
      [{ channel: "os", verb: "focus-changed", qualifier: "mail" }],
      [{ channel: "device", verb: "tap", qualifier: "compose" }],
      [
        { channel: "device", verb: "type", qualifier: "recipient" },
        { channel: "device", verb: "type", qualifier: "subject" },
      ],
      [{ channel: "device", verb: "type", qualifier: "body" }],
      [{ channel: "device", verb: "tap", qualifier: "send" }],
    ],
  },
  {
    name: "browse-and-scroll",
    stages: [
      [{ channel: "os", verb: "window-opened", qualifier: "browser" }],
      [{ channel: "browser", verb: "navigate", qualifier: "docs" }],
      [
        { channel: "device", verb: "scroll", qualifier: "down" },
        { channel: "device", verb: "scroll", qualifier: "up" },
      ],
      [{ channel: "device", verb: "tap", qualifier: "link" }],
      [{ channel: "device", verb: "swipe", qualifier: "left" }],
    ],
  },
  {
    name: "file-edit-save",
    stages: [
      [{ channel: "os", verb: "file-opened", qualifier: "editor" }],
      [{ channel: "device", verb: "tap", qualifier: "line" }],
      [{ channel: "device", verb: "type", qualifier: "edit" }],
      [{ channel: "device", verb: "shortcut", qualifier: "save" }],
      [{ channel: "os", verb: "command-ran", qualifier: "build" }],
    ],
  },
];

function createLcg(seed: number): () => number {
  let state = (Math.abs(Math.trunc(seed)) % 2147483647) || 1;
  return () => {
    state = (state * 48271) % 2147483647;
    return (state - 1) / 2147483646;
  };
}

function pick<T>(items: T[], rng: () => number): T {
  const index = Math.min(items.length - 1, Math.floor(rng() * items.length));
  return items[index]!;
}

export type SyntheticDatasetParams = {
  /** Number of sequences to emit. */
  sequenceCount: number;
  /** RNG seed for reproducibility. */
  seed: number;
  /** Flows to sample from. Defaults to {@link DEFAULT_SYNTHETIC_FLOWS}. */
  flows?: SyntheticTaskFlow[];
  /** Prefix for generated sequence ids. Defaults to "synthetic". */
  idPrefix?: string;
};

/** Generate a reproducible synthetic movement dataset for training/eval. */
export function generateSyntheticMovementDataset(params: SyntheticDatasetParams): MovementDataset {
  const flows = params.flows ?? DEFAULT_SYNTHETIC_FLOWS;
  const rng = createLcg(params.seed);
  const prefix = params.idPrefix ?? "synthetic";
  const sequences: MovementSequence[] = [];

  for (let index = 0; index < params.sequenceCount; index += 1) {
    const flow = pick(flows, rng);
    const steps = flow.stages.map((variants) => pick(variants, rng));
    sequences.push({ id: `${prefix}-${flow.name}-${index}`, steps });
  }

  return { version: 1, sequences };
}

/**
 * Deterministically split a dataset into train/held-out partitions by index,
 * so generalization can be measured on sequences the model never trained on.
 */
export function splitMovementDataset(
  dataset: MovementDataset,
  holdOutFraction: number,
): { train: MovementDataset; heldOut: MovementDataset } {
  const clamped = Math.min(0.9, Math.max(0, holdOutFraction));
  const total = dataset.sequences.length;
  const heldOutCount = Math.floor(total * clamped);
  const train: MovementSequence[] = [];
  const heldOut: MovementSequence[] = [];
  dataset.sequences.forEach((sequence, index) => {
    // Interleave by modulus so both partitions cover all flows.
    const stride = heldOutCount > 0 ? Math.max(1, Math.round(total / heldOutCount)) : 0;
    if (stride > 0 && index % stride === 0 && heldOut.length < heldOutCount) {
      heldOut.push(sequence);
    } else {
      train.push(sequence);
    }
  });
  return { train: { version: 1, sequences: train }, heldOut: { version: 1, sequences: heldOut } };
}

/**
 * Render a movement sequence as a {@link TrajectorySpan}, so synthetic streams
 * can flow through the same capture/export path as real captures.
 */
export function syntheticSequenceToTrajectory(
  sequence: MovementSequence,
  options: { sessionId?: string; captureTier?: CaptureTier; baseTs?: number } = {},
): TrajectorySpan {
  const baseTs = options.baseTs ?? 0;
  return {
    id: sequence.id,
    sessionId: options.sessionId ?? `session-${sequence.id}`,
    createdAt: new Date(baseTs).toISOString(),
    captureTier: options.captureTier ?? "app",
    observations: [],
    actions: sequence.steps.map((step, index) => ({
      kind: "action" as const,
      tool: step.channel,
      summary: step.qualifier ? `${step.verb} ${step.qualifier}` : step.verb,
      ts: baseTs + index,
      metadata: {
        gesture: step.verb,
        ...(step.qualifier ? { target: step.qualifier } : {}),
      },
    })),
  };
}
