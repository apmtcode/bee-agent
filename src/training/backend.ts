import type { ReviewedExportManifest, TrainingMode } from "./export-manifest.js";

/**
 * Pluggable local-model backend for the movement-learning subsystem.
 *
 * The {@link LocalAppleSiliconTrainingRunner} emits launch scripts that drive an
 * *external* Apple-Silicon training process (mlx / axolotl). That path cannot run
 * in the cloud/CI, so it can never validate the end-to-end "record → post-train →
 * repeat → generalize" loop that standing objective #2 requires.
 *
 * This module defines a small, JSON-serializable backend seam that CAN run
 * in-process: a backend consumes reviewed movement sequences, produces a
 * serializable model artifact, and predicts continuations from a prompt. The
 * default {@link "./backends/markov-backend.js" | MarkovMovementBackend} is a
 * fully deterministic implementation used by tests; a real on-device small-model
 * backend can implement the same interface without touching call sites.
 */

/** A single tokenized movement/action step, e.g. `"action:click"`. */
export type MovementToken = string;

/** Sentinel prepended to every training sequence so short-prompt contexts exist. */
export const SEQUENCE_START_TOKEN = "START";
/** Sentinel appended to every training sequence so the model learns to stop. */
export const SEQUENCE_END_TOKEN = "END";

/** An ordered run of movement tokens derived from one replay/trajectory. */
export type MovementSequence = {
  id: string;
  tokens: MovementToken[];
};

export type LocalModelTrainingRequest = {
  jobId: string;
  mode: TrainingMode;
  sequences: MovementSequence[];
  /** Context length for sequence models; backends may clamp or ignore. */
  order?: number;
};

/** Serializable trained-model artifact. `weights` is backend-defined but JSON-safe. */
export type LocalModelArtifact = {
  backend: string;
  jobId: string;
  version: 1;
  order: number;
  vocabulary: MovementToken[];
  sequenceCount: number;
  tokenCount: number;
  weights: unknown;
};

export type LocalModelInferenceRequest = {
  /** Prefix of already-observed tokens; the model predicts what comes next. */
  prompt: MovementToken[];
  maxTokens?: number;
};

export type LocalModelInferenceResult = {
  tokens: MovementToken[];
  /** True if any prediction fell back to a shorter/global context. */
  usedBackoff: boolean;
  /** True if generation stopped at a learned end-of-sequence rather than maxTokens. */
  terminated: boolean;
};

/**
 * A local-model backend. Implementations MUST be deterministic given identical
 * inputs so cloud/CI runs are reproducible.
 */
export interface LocalModelBackend {
  readonly name: string;
  train(request: LocalModelTrainingRequest): Promise<LocalModelArtifact>;
  infer(
    model: LocalModelArtifact,
    request: LocalModelInferenceRequest,
  ): Promise<LocalModelInferenceResult>;
}

/** Registry so backends are selectable by name (pluggability seam). */
export class LocalModelBackendRegistry {
  private readonly backends = new Map<string, LocalModelBackend>();

  register(backend: LocalModelBackend): this {
    this.backends.set(backend.name, backend);
    return this;
  }

  has(name: string): boolean {
    return this.backends.has(name);
  }

  get(name: string): LocalModelBackend {
    const backend = this.backends.get(name);
    if (!backend) {
      throw new Error(
        `unknown local-model backend "${name}"; registered: ${this.list().join(", ") || "<none>"}`,
      );
    }
    return backend;
  }

  list(): string[] {
    return [...this.backends.keys()].sort();
  }
}

/** Delimiter joining tokens into a context key; not valid inside a token. */
export const CONTEXT_DELIMITER = "";

/** Map a reviewed replay event to a single movement token. */
export function tokenizeReplayEvent(
  event: ReviewedExportManifest["replays"][number]["events"][number],
): MovementToken {
  switch (event.kind) {
    case "action":
      return `action:${event.tool}`;
    case "observation":
      return `obs:${event.source}`;
    case "transcript":
      return `msg:${event.role}`;
  }
}

/**
 * Build one movement sequence per reviewed replay manifest. Events are already
 * time-ordered by {@link buildReplayManifest}, so token order is preserved.
 */
export function buildMovementSequences(manifest: ReviewedExportManifest): MovementSequence[] {
  return manifest.replays.map((replay, index) => ({
    id: replay.trajectoryIds[0] ?? `${replay.sessionId}#${index}`,
    tokens: replay.events.map(tokenizeReplayEvent),
  }));
}

export type SequenceFidelity = {
  id: string;
  tokens: number;
  correct: number;
  accuracy: number;
};

export type ReplayFidelityReport = {
  sequenceCount: number;
  evaluatedTokens: number;
  correctPredictions: number;
  /** Overall next-token accuracy in [0, 1]. */
  accuracy: number;
  perSequence: SequenceFidelity[];
};

/**
 * Generalization / replay-fidelity eval harness. For each sequence, walks every
 * position and asks the backend to predict the next token from the preceding
 * prefix, reporting next-token accuracy. Evaluating on the training set measures
 * *repeat* fidelity; evaluating on held-out related sequences measures
 * *generalization*.
 */
export async function evaluateReplayFidelity(
  backend: LocalModelBackend,
  model: LocalModelArtifact,
  sequences: MovementSequence[],
): Promise<ReplayFidelityReport> {
  const perSequence: SequenceFidelity[] = [];
  let evaluatedTokens = 0;
  let correctPredictions = 0;

  for (const sequence of sequences) {
    let tokens = 0;
    let correct = 0;
    for (let index = 0; index < sequence.tokens.length; index += 1) {
      const prompt = sequence.tokens.slice(0, index);
      const prediction = await backend.infer(model, { prompt, maxTokens: 1 });
      tokens += 1;
      if (prediction.tokens[0] === sequence.tokens[index]) {
        correct += 1;
      }
    }
    evaluatedTokens += tokens;
    correctPredictions += correct;
    perSequence.push({
      id: sequence.id,
      tokens,
      correct,
      accuracy: tokens === 0 ? 0 : correct / tokens,
    });
  }

  return {
    sequenceCount: sequences.length,
    evaluatedTokens,
    correctPredictions,
    accuracy: evaluatedTokens === 0 ? 0 : correctPredictions / evaluatedTokens,
    perSequence,
  };
}
