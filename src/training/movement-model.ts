import type { TrajectorySpan } from "../capture/trajectory.js";
import type { ReplayManifest } from "../capture/replay.js";

/**
 * Local-movement learning: in-process policy model over recorded action
 * sequences.
 *
 * The training runner (`runner.ts`) only *plans* out-of-process training jobs
 * for a real on-device model (mlx/axolotl). This module provides the
 * complementary in-process seam required by standing objective #2(c)/(d):
 * post-train a *local* model that can (c) repeat recorded movements and
 * (d) generalize to new-but-related movements.
 *
 * The model backend is pluggable (`MovementModelBackend`) so a real on-device
 * small model can be dropped in later. The bundled {@link MarkovMovementBackend}
 * is fully deterministic (no clock, no RNG) so it validates the
 * capture -> dataset -> train -> infer round-trip in the cloud/CI with
 * synthetic event streams, exactly as the objective requires.
 */

/** A single normalized movement token, e.g. `"click:button.save"`. */
export type MovementToken = string;

/** An ordered movement sequence (one recorded trajectory / demonstration). */
export type MovementSequence = {
  id: string;
  tokens: MovementToken[];
};

/** A replayable dataset of movement demonstrations. */
export type MovementDataset = {
  version: 1;
  sequences: MovementSequence[];
};

export type MovementTrainingConfig = {
  /** Maximum n-gram context length the model conditions on. Default 3. */
  order?: number;
};

/**
 * Serializable trained-model artifact. Plain JSON so it can be persisted next
 * to the export/replay manifests and reloaded without the backend instance.
 */
export type MovementModelArtifact = {
  version: 1;
  backend: string;
  order: number;
  vocabulary: MovementToken[];
  sequenceCount: number;
  tokenCount: number;
  /**
   * `contextKey -> (nextToken -> count)`. The empty-string key holds unigram
   * counts. Context keys join tokens with the unit-separator control char.
   */
  transitions: Record<string, Record<MovementToken, number>>;
};

export type MovementPrediction = {
  /** Argmax next token, or `undefined` when the model has learned nothing. */
  token: MovementToken | undefined;
  /** P(token | context) for the backoff context actually used, in [0, 1]. */
  confidence: number;
  /** Length of the context suffix that produced the prediction (0 = unigram). */
  backoffOrder: number;
  /** All candidates for the chosen context, most-probable first. */
  alternatives: Array<{ token: MovementToken; probability: number }>;
};

export type MovementGenerateParams = {
  /** Prior context to condition the first prediction on. */
  seed?: MovementToken[];
  /** Hard cap on generated tokens (prevents cyclic-argmax runaway). Default 32. */
  maxSteps?: number;
  /** Stop rollout when this token is produced (it is included in the output). */
  stopToken?: MovementToken;
};

/**
 * Pluggable local-model backend. Swap {@link MarkovMovementBackend} for a real
 * on-device small model by implementing this interface.
 */
export interface MovementModelBackend {
  readonly name: string;
  train(dataset: MovementDataset, config?: MovementTrainingConfig): Promise<MovementModelArtifact>;
  predictNext(
    artifact: MovementModelArtifact,
    context: MovementToken[],
    options?: { topK?: number },
  ): MovementPrediction;
  generate(artifact: MovementModelArtifact, params?: MovementGenerateParams): MovementToken[];
}

const CONTEXT_SEPARATOR = "\u001f"; // unit separator, absent from real tokens
const DEFAULT_ORDER = 3;
const DEFAULT_MAX_STEPS = 32;

function contextKey(tokens: MovementToken[]): string {
  return tokens.join(CONTEXT_SEPARATOR);
}

/**
 * Deterministic n-gram movement policy with stupid-backoff.
 *
 * Learning: counts every context of length `0..order` -> next-token. Inference:
 * uses the longest context suffix seen in training (backing off to shorter
 * contexts, ending at the unigram), and picks the highest-count continuation
 * with a stable lexicographic tie-break. The backoff is what lets it
 * *generalize*: an unseen full context still resolves through a shared shorter
 * suffix instead of failing.
 */
export class MarkovMovementBackend implements MovementModelBackend {
  readonly name = "markov";

  async train(dataset: MovementDataset, config?: MovementTrainingConfig): Promise<MovementModelArtifact> {
    const order = Math.max(1, Math.floor(config?.order ?? DEFAULT_ORDER));
    const transitions: Record<string, Record<MovementToken, number>> = {};
    const vocabulary = new Set<MovementToken>();
    let tokenCount = 0;

    for (const sequence of dataset.sequences) {
      const tokens = sequence.tokens;
      for (let i = 0; i < tokens.length; i += 1) {
        const next = tokens[i]!;
        vocabulary.add(next);
        tokenCount += 1;
        // Condition `next` on every context length 0..order ending at i-1.
        for (let k = 0; k <= order; k += 1) {
          if (i - k < 0) {
            break;
          }
          const key = contextKey(tokens.slice(i - k, i));
          const bucket = (transitions[key] ??= {});
          bucket[next] = (bucket[next] ?? 0) + 1;
        }
      }
    }

    return {
      version: 1,
      backend: this.name,
      order,
      vocabulary: [...vocabulary].sort(),
      sequenceCount: dataset.sequences.length,
      tokenCount,
      transitions,
    };
  }

  predictNext(
    artifact: MovementModelArtifact,
    context: MovementToken[],
    options?: { topK?: number },
  ): MovementPrediction {
    const topK = Math.max(1, Math.floor(options?.topK ?? 5));
    const maxContext = Math.min(context.length, artifact.order);

    for (let k = maxContext; k >= 0; k -= 1) {
      const key = contextKey(context.slice(context.length - k, context.length));
      const bucket = artifact.transitions[key];
      if (!bucket) {
        continue;
      }
      const entries = Object.entries(bucket);
      if (entries.length === 0) {
        continue;
      }
      const total = entries.reduce((sum, [, count]) => sum + count, 0);
      // Highest count wins; lexicographic token order breaks ties for
      // determinism (no RNG, so replay is reproducible).
      const ranked = entries
        .map(([token, count]) => ({ token, probability: total > 0 ? count / total : 0 }))
        .sort((a, b) => (b.probability - a.probability) || (a.token < b.token ? -1 : a.token > b.token ? 1 : 0));
      const best = ranked[0]!;
      return {
        token: best.token,
        confidence: best.probability,
        backoffOrder: k,
        alternatives: ranked.slice(0, topK),
      };
    }

    return { token: undefined, confidence: 0, backoffOrder: 0, alternatives: [] };
  }

  generate(artifact: MovementModelArtifact, params?: MovementGenerateParams): MovementToken[] {
    const maxSteps = Math.max(0, Math.floor(params?.maxSteps ?? DEFAULT_MAX_STEPS));
    const context: MovementToken[] = [...(params?.seed ?? [])];
    const output: MovementToken[] = [];

    for (let step = 0; step < maxSteps; step += 1) {
      const prediction = this.predictNext(artifact, context);
      if (prediction.token === undefined) {
        break;
      }
      output.push(prediction.token);
      context.push(prediction.token);
      if (params?.stopToken !== undefined && prediction.token === params.stopToken) {
        break;
      }
    }

    return output;
  }
}

// ---------------------------------------------------------------------------
// Dataset bridges: recorded capture data -> movement dataset.
// ---------------------------------------------------------------------------

/** Normalize a trajectory action into a stable movement token. */
export function movementTokenFromAction(action: { tool: string; summary: string }): MovementToken {
  return `${action.tool}:${action.summary}`;
}

/** Build a movement sequence from a trajectory's ordered actions. */
export function movementSequenceFromTrajectory(trajectory: TrajectorySpan): MovementSequence {
  const tokens = [...trajectory.actions]
    .sort((a, b) => a.ts - b.ts)
    .map((action) => movementTokenFromAction(action));
  return { id: trajectory.id, tokens };
}

/** Build a movement sequence from a replay manifest's action timeline. */
export function movementSequenceFromReplay(manifest: ReplayManifest): MovementSequence {
  const tokens = manifest.events
    .filter((event): event is Extract<typeof event, { kind: "action" }> => event.kind === "action")
    .map((event) => movementTokenFromAction({ tool: event.tool, summary: event.summary }));
  return { id: manifest.sessionId, tokens };
}

/** Assemble a versioned dataset from movement sequences (empty ones dropped). */
export function buildMovementDataset(sequences: MovementSequence[]): MovementDataset {
  return {
    version: 1,
    sequences: sequences.filter((sequence) => sequence.tokens.length > 0),
  };
}

// ---------------------------------------------------------------------------
// Generalization eval harness.
// ---------------------------------------------------------------------------

export type MovementEvalResult = {
  /** Number of (context -> next) predictions scored. */
  predictions: number;
  /** How many predicted the held-out next token exactly. */
  correct: number;
  /** correct / predictions, in [0, 1] (0 when there was nothing to score). */
  accuracy: number;
  /** Mean confidence the model assigned to its chosen token. */
  meanConfidence: number;
  /** Fraction of predictions that had to back off below full context length. */
  backoffRate: number;
};

/**
 * Next-token accuracy over held-out sequences — the generalization signal for
 * objective #2(d). Train on some demonstrations, evaluate on related-but-unseen
 * ones: high accuracy means the backoff model transfers learned sub-movements.
 */
export function evaluateNextTokenAccuracy(
  backend: MovementModelBackend,
  artifact: MovementModelArtifact,
  sequences: MovementSequence[],
): MovementEvalResult {
  let predictions = 0;
  let correct = 0;
  let confidenceSum = 0;
  let backoffCount = 0;

  for (const sequence of sequences) {
    for (let i = 1; i < sequence.tokens.length; i += 1) {
      const context = sequence.tokens.slice(0, i);
      const prediction = backend.predictNext(artifact, context);
      predictions += 1;
      confidenceSum += prediction.confidence;
      if (prediction.backoffOrder < Math.min(context.length, artifact.order)) {
        backoffCount += 1;
      }
      if (prediction.token === sequence.tokens[i]) {
        correct += 1;
      }
    }
  }

  return {
    predictions,
    correct,
    accuracy: predictions > 0 ? correct / predictions : 0,
    meanConfidence: predictions > 0 ? confidenceSum / predictions : 0,
    backoffRate: predictions > 0 ? backoffCount / predictions : 0,
  };
}
