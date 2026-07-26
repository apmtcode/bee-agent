import path from "node:path";
import type {
  LocalTrainingExecution,
  LocalTrainingJobManifest,
  RlTrainingConfig,
  SftTrainingConfig,
} from "./job-manifest.js";

/**
 * The runtime-specific slice of a {@link TrainingJobPlan}. A pluggable
 * {@link LocalTrainingBackend} contributes exactly these fields; the runner
 * owns everything platform-agnostic (paths, dataset/replay wiring, launch
 * script rendering). This is the seam that lets bee-agent target different
 * local training stacks — Apple-Silicon MLX today, a mock backend for
 * cloud/CI, and (documented seam) any small on-device model tomorrow.
 */
export type TrainingBackendContribution = {
  /** Backend-specific runtime identifier surfaced in the plan (e.g. "mlx"). */
  runtime: string;
  /** Platform the backend targets (e.g. "apple-silicon", "portable"). */
  targetPlatform: string;
  /** File name (within the artifact dir) the trained model is written to. */
  outputFileName: string;
  /** Argv the launch script executes to run training. */
  command: string[];
  /** Environment variables exported into the training process. */
  environment: Record<string, string>;
};

/**
 * A pluggable local-model training backend. Implementations translate a
 * reviewed training job into a concrete, runnable command for a specific local
 * runtime. Backends must be pure/deterministic given (job, execution) so the
 * generated plan is reproducible and testable without touching a real machine.
 */
export interface LocalTrainingBackend {
  /** Stable identifier for the backend (e.g. "apple-silicon", "mock"). */
  readonly id: string;
  /** Contribute the runtime-specific portion of a training plan. */
  describe(job: LocalTrainingJobManifest, execution: LocalTrainingExecution): TrainingBackendContribution;
}

/**
 * Apple-Silicon backend: the original bee-agent behaviour, extracted verbatim.
 * SFT runs on MLX (`mlx_lm.lora`), RL runs on Axolotl with a replay-manifest
 * reward. Requires an on-device Apple Silicon machine to actually execute.
 */
export class AppleSiliconTrainingBackend implements LocalTrainingBackend {
  readonly id = "apple-silicon";

  describe(job: LocalTrainingJobManifest, execution: LocalTrainingExecution): TrainingBackendContribution {
    const baseEnvironment = {
      OPENCLAW_TRAINING_JOB_ID: job.id,
      OPENCLAW_TRAINING_MODE: job.mode,
      OPENCLAW_TARGET_PLATFORM: job.targetPlatform,
      OPENCLAW_REVIEWED_EXPORT_REQUIRED: "true",
      OPENCLAW_RAW_CAPTURE_ALLOWED: "false",
    };

    if (job.mode === "sft") {
      const config = job.config as SftTrainingConfig;
      return {
        runtime: "mlx",
        targetPlatform: "apple-silicon",
        outputFileName: "model.gguf",
        command: [
          "python3",
          "-m",
          "mlx_lm.lora",
          "--train",
          "--data",
          execution.datasetDir,
          "--adapter-path",
          execution.artifactDir,
          "--learning-rate",
          String(config.learningRate),
          "--batch-size",
          String(config.batchSize),
          "--iters",
          String(config.epochs * 1000),
        ],
        environment: { ...baseEnvironment, OPENCLAW_TRAINING_RUNTIME: "mlx" },
      };
    }

    const config = job.config as RlTrainingConfig;
    return {
      runtime: "axolotl",
      targetPlatform: "apple-silicon",
      outputFileName: "policy.gguf",
      command: [
        "python3",
        "-m",
        "axolotl.cli.train",
        execution.planFile,
        "--reward-model",
        "replay-manifest",
        "--rollouts",
        String(config.rolloutCount),
        "--kl-penalty",
        String(config.klPenalty),
      ],
      environment: { ...baseEnvironment, OPENCLAW_TRAINING_RUNTIME: "axolotl" },
    };
  }
}

/**
 * Deterministic, dependency-free backend for cloud/CI. It runs bee-agent's own
 * bundled trainer (`src/training/mock-trainer.js`) via `node`, so the full
 * prepare → launch → train → artifact seam is exercisable in environments
 * without Python/MLX or a real GPU. The trainer fits a tiny on-device Markov
 * movement model (see {@link trainMockMovementModel}) that can repeat and
 * generalize recorded movement sequences.
 */
export class MockLocalTrainingBackend implements LocalTrainingBackend {
  readonly id = "mock";

  constructor(private readonly trainerModule = "src/training/mock-trainer.js") {}

  describe(job: LocalTrainingJobManifest, execution: LocalTrainingExecution): TrainingBackendContribution {
    const outputFileName = job.mode === "sft" ? "model.json" : "policy.json";
    return {
      runtime: "mock",
      targetPlatform: "portable",
      outputFileName,
      command: [
        "node",
        this.trainerModule,
        "--dataset",
        execution.datasetDir,
        "--out",
        path.posix.join(execution.artifactDir, outputFileName),
        "--mode",
        job.mode,
      ],
      environment: {
        BEE_TRAINING_JOB_ID: job.id,
        BEE_TRAINING_MODE: job.mode,
        BEE_TRAINING_RUNTIME: "mock",
        BEE_TARGET_PLATFORM: "portable",
        BEE_REVIEWED_EXPORT_REQUIRED: "true",
        BEE_RAW_CAPTURE_ALLOWED: "false",
      },
    };
  }
}

// ---------------------------------------------------------------------------
// Tiny on-device movement model (the mock backend's actual "trainable model").
//
// A first-order Markov transition model over movement tokens. It is trivially
// trainable on-device with zero dependencies, fully deterministic, and — being
// a probabilistic sequence model — can both *repeat* recorded movements and
// *generalize* to related-but-unseen movement paths (any transition observed in
// one sequence becomes reachable from any other sequence that reaches its
// source state). This is deliberately small: it validates the train → infer
// pipeline shape so a heavier pluggable model can drop into the same seam.
// ---------------------------------------------------------------------------

export type MockMovementModel = {
  version: 1;
  kind: "markov-1";
  /** sourceToken -> nextToken -> observed transition count. */
  transitions: Record<string, Record<string, number>>;
  /** Tokens observed at the start of a sequence -> count. */
  starts: Record<string, number>;
  vocabulary: string[];
  sequenceCount: number;
  transitionCount: number;
};

/**
 * Train the mock movement model from recorded movement token sequences.
 * Deterministic: identical input always yields an identical model. Tokens are
 * opaque strings (e.g. "mousedown:left", "key:Enter", "window:focus").
 */
export function trainMockMovementModel(sequences: string[][]): MockMovementModel {
  const transitions: Record<string, Record<string, number>> = {};
  const starts: Record<string, number> = {};
  const vocabulary = new Set<string>();
  let transitionCount = 0;

  for (const sequence of sequences) {
    if (sequence.length === 0) {
      continue;
    }
    starts[sequence[0]] = (starts[sequence[0]] ?? 0) + 1;
    for (let index = 0; index < sequence.length; index += 1) {
      const token = sequence[index];
      vocabulary.add(token);
      const next = sequence[index + 1];
      if (next === undefined) {
        continue;
      }
      const row = (transitions[token] ??= {});
      row[next] = (row[next] ?? 0) + 1;
      transitionCount += 1;
    }
  }

  return {
    version: 1,
    kind: "markov-1",
    transitions: sortNestedCounts(transitions),
    starts: sortCounts(starts),
    vocabulary: [...vocabulary].sort(),
    sequenceCount: sequences.filter((sequence) => sequence.length > 0).length,
    transitionCount,
  };
}

/**
 * Predict the most likely next movement token given a preceding context.
 * Uses the last context token (first-order). Ties break lexicographically for
 * determinism. Returns undefined when no transition was ever observed from the
 * context token (the model does not hallucinate unseen transitions).
 */
export function predictNextMovement(model: MockMovementModel, context: string[]): string | undefined {
  const source = context.at(-1);
  if (source === undefined) {
    return argmaxCount(model.starts);
  }
  const row = model.transitions[source];
  if (!row) {
    return undefined;
  }
  return argmaxCount(row);
}

/**
 * Replay a full movement sequence from the model starting at `start` (or the
 * most common recorded start token), following the highest-probability
 * transition at each step. Stops at `maxLength` or when a state has no known
 * outgoing transition. This is how the trained model "repeats" a movement.
 */
export function replayMovementSequence(
  model: MockMovementModel,
  options: { start?: string; maxLength?: number } = {},
): string[] {
  const maxLength = options.maxLength ?? 64;
  const first = options.start ?? argmaxCount(model.starts);
  if (first === undefined) {
    return [];
  }
  const sequence: string[] = [first];
  while (sequence.length < maxLength) {
    const next = predictNextMovement(model, sequence);
    if (next === undefined) {
      break;
    }
    sequence.push(next);
    // Guard against self-loops producing an unbounded identical tail.
    if (next === sequence.at(-2) && next === sequence.at(-3)) {
      break;
    }
  }
  return sequence;
}

function argmaxCount(counts: Record<string, number>): string | undefined {
  let best: string | undefined;
  let bestCount = -Infinity;
  for (const key of Object.keys(counts).sort()) {
    const count = counts[key];
    if (count > bestCount) {
      best = key;
      bestCount = count;
    }
  }
  return best;
}

function sortCounts(counts: Record<string, number>): Record<string, number> {
  const sorted: Record<string, number> = {};
  for (const key of Object.keys(counts).sort()) {
    sorted[key] = counts[key];
  }
  return sorted;
}

function sortNestedCounts(
  nested: Record<string, Record<string, number>>,
): Record<string, Record<string, number>> {
  const sorted: Record<string, Record<string, number>> = {};
  for (const key of Object.keys(nested).sort()) {
    sorted[key] = sortCounts(nested[key]);
  }
  return sorted;
}
