import type { ReplayManifest } from "../capture/replay.js";
import type { TrajectoryAction, TrajectorySpan } from "../capture/trajectory.js";

/**
 * In-process, cloud-safe movement-model pipeline.
 *
 * Standing objective #2 asks bee-agent to (c) post-train a local model on the
 * recorded movement dataset to repeat movements, and (d) generalize to new but
 * related movements. The Apple-Silicon {@link LocalAppleSiliconTrainingRunner}
 * only *emits launch scripts* for external MLX/axolotl runtimes — it cannot run
 * anywhere the engine actually executes (Anthropic's cloud, CI). This module
 * provides the missing seam: a pluggable {@link LocalModelBackend} interface
 * plus a fully deterministic mock backend that trains and infers entirely
 * in-process, so the capture -> dataset -> train -> replay/generalize loop is
 * exercised by tests without any real OS input or GPU.
 *
 * The mock backend is a k-order Markov model over movement tokens with
 * stupid-backoff: when the exact k-token context was never observed it falls
 * back to progressively shorter contexts, which is precisely the mechanism that
 * lets it *generalize* to related-but-unseen movement prefixes.
 */

/** A compact, replayable descriptor for one captured action, e.g. `tap:submit`. */
export type MovementToken = string;

/** An ordered movement token stream derived from one trajectory / replay. */
export type MovementSequence = {
  id: string;
  tokens: MovementToken[];
};

/** The training corpus consumed by a {@link LocalModelBackend}. */
export type MovementDataset = {
  version: 1;
  sequences: MovementSequence[];
};

/** A trained, JSON-serializable movement model artifact. */
export type TrainedMovementModel = {
  backend: string;
  version: 1;
  /** Maximum context order the model was trained at. */
  order: number;
  vocabulary: MovementToken[];
  /**
   * Transition counts keyed by `"<order><joined-context>"`; each entry
   * maps a candidate next token to the number of times it followed that context.
   */
  transitions: Record<string, Record<MovementToken, number>>;
  sequenceCount: number;
  exampleCount: number;
};

/** One ranked candidate next action. */
export type MovementCandidate = {
  action: MovementToken;
  probability: number;
};

/** The result of asking a trained model for the next movement. */
export type MovementPrediction = {
  /** Highest-probability next token, or `undefined` for an empty model. */
  action: MovementToken | undefined;
  /** Probability of the chosen action at the context order actually used. */
  confidence: number;
  /** The context order the prediction was drawn from (after any backoff). */
  order: number;
  /**
   * True when the exact requested context was unseen and the model had to back
   * off to a shorter context — i.e. the prediction is a generalization rather
   * than a memorized replay.
   */
  generalized: boolean;
  /** All candidate next tokens at the used order, ranked by probability. */
  candidates: MovementCandidate[];
};

export type TrainMovementModelOptions = {
  /** Maximum context order (k). Defaults to 2. Clamped to `>= 0`. */
  order?: number;
};

/**
 * A pluggable local-model backend. The deterministic mock ships here; a real
 * on-device small-model backend can implement the same interface and be
 * registered in a {@link MovementBackendRegistry} without touching call sites.
 */
export interface LocalModelBackend {
  readonly id: string;
  train(dataset: MovementDataset, options?: TrainMovementModelOptions): TrainedMovementModel;
  predict(model: TrainedMovementModel, context: MovementToken[]): MovementPrediction;
}

const CONTEXT_SEPARATOR = "";
const TOKEN_SEPARATOR = "";

function contextKey(order: number, context: MovementToken[]): string {
  return `${order}${CONTEXT_SEPARATOR}${context.join(TOKEN_SEPARATOR)}`;
}

/**
 * Deterministic k-order Markov backend with stupid-backoff. Given identical
 * input it always produces identical models and predictions (ties broken by
 * lexical token order), so it is safe to assert on in CI.
 */
export class DeterministicMarkovBackend implements LocalModelBackend {
  readonly id = "deterministic-markov";

  train(dataset: MovementDataset, options: TrainMovementModelOptions = {}): TrainedMovementModel {
    const order = Math.max(0, Math.trunc(options.order ?? 2));
    const transitions: Record<string, Record<MovementToken, number>> = {};
    const vocabulary = new Set<MovementToken>();
    let exampleCount = 0;

    for (const sequence of dataset.sequences) {
      const tokens = sequence.tokens;
      for (let i = 0; i < tokens.length; i += 1) {
        const next = tokens[i]!;
        vocabulary.add(next);
        const maxContext = Math.min(order, i);
        for (let o = 0; o <= maxContext; o += 1) {
          const context = tokens.slice(i - o, i);
          const key = contextKey(o, context);
          const bucket = (transitions[key] ??= {});
          bucket[next] = (bucket[next] ?? 0) + 1;
          exampleCount += 1;
        }
      }
    }

    return {
      backend: this.id,
      version: 1,
      order,
      vocabulary: [...vocabulary].sort(),
      transitions,
      sequenceCount: dataset.sequences.length,
      exampleCount,
    };
  }

  predict(model: TrainedMovementModel, context: MovementToken[]): MovementPrediction {
    const requestedOrder = Math.min(model.order, context.length);
    for (let o = requestedOrder; o >= 0; o -= 1) {
      const slice = context.slice(context.length - o, context.length);
      const bucket = model.transitions[contextKey(o, slice)];
      if (!bucket) {
        continue;
      }
      const candidates = rankCandidates(bucket);
      if (candidates.length === 0) {
        continue;
      }
      return {
        action: candidates[0]!.action,
        confidence: candidates[0]!.probability,
        order: o,
        generalized: o < requestedOrder,
        candidates,
      };
    }

    return {
      action: undefined,
      confidence: 0,
      order: 0,
      generalized: requestedOrder > 0,
      candidates: [],
    };
  }
}

function rankCandidates(bucket: Record<MovementToken, number>): MovementCandidate[] {
  const total = Object.values(bucket).reduce((sum, count) => sum + count, 0);
  if (total === 0) {
    return [];
  }
  return Object.entries(bucket)
    .map(([action, count]) => ({ action, probability: count / total }))
    .sort((a, b) => {
      if (b.probability !== a.probability) {
        return b.probability - a.probability;
      }
      return a.action < b.action ? -1 : a.action > b.action ? 1 : 0;
    });
}

/**
 * Registry of named backends so a real on-device model can be swapped in for
 * the deterministic mock without changing any call site.
 */
export class MovementBackendRegistry {
  private readonly backends = new Map<string, LocalModelBackend>();

  constructor(backends: LocalModelBackend[] = [new DeterministicMarkovBackend()]) {
    for (const backend of backends) {
      this.register(backend);
    }
  }

  register(backend: LocalModelBackend): void {
    this.backends.set(backend.id, backend);
  }

  get(id: string): LocalModelBackend {
    const backend = this.backends.get(id);
    if (!backend) {
      throw new Error(`unknown movement-model backend: ${id}`);
    }
    return backend;
  }

  has(id: string): boolean {
    return this.backends.has(id);
  }

  list(): string[] {
    return [...this.backends.keys()].sort();
  }
}

/** Convert one captured action into a compact, replayable movement token. */
export function tokenizeMovementAction(action: TrajectoryAction): MovementToken {
  const metadata = action.metadata ?? {};
  const gesture = typeof metadata.gesture === "string" ? metadata.gesture : undefined;
  const target = typeof metadata.target === "string" ? metadata.target : undefined;
  const direction = typeof metadata.direction === "string" ? metadata.direction : undefined;
  const qualifier = target ?? direction ?? firstWord(action.summary);
  const head = gesture ?? action.tool;
  return qualifier ? `${head}:${normalize(qualifier)}` : normalize(head);
}

function firstWord(summary: string): string | undefined {
  const word = summary.trim().split(/\s+/)[0];
  return word ? word : undefined;
}

function normalize(value: string): string {
  return value.trim().toLowerCase().replace(/\s+/g, "-");
}

/** Build a single movement sequence from a captured trajectory span. */
export function movementSequenceFromTrajectory(span: TrajectorySpan): MovementSequence {
  const tokens = [...span.actions]
    .sort((a, b) => a.ts - b.ts)
    .map(tokenizeMovementAction);
  return { id: span.id, tokens };
}

/** Build a single movement sequence from a replay manifest's action events. */
export function movementSequenceFromReplay(manifest: ReplayManifest): MovementSequence {
  const tokens = manifest.events
    .filter((event): event is Extract<typeof event, { kind: "action" }> => event.kind === "action")
    .map((event) => {
      const qualifier = firstWord(event.summary);
      return qualifier ? `${event.tool}:${normalize(qualifier)}` : normalize(event.tool);
    });
  return { id: manifest.sessionId, tokens };
}

/** Assemble a training dataset from captured trajectories (dropping empties). */
export function buildMovementDataset(trajectories: TrajectorySpan[]): MovementDataset {
  const sequences = trajectories
    .map(movementSequenceFromTrajectory)
    .filter((sequence) => sequence.tokens.length > 0);
  return { version: 1, sequences };
}

export type GeneralizationReport = {
  /** Number of next-token predictions evaluated. */
  total: number;
  /** Predictions whose top action matched the held-out ground truth. */
  correct: number;
  /** Correct predictions that required backoff (generalization, not memory). */
  generalizedCorrect: number;
  /** Predictions the model refused (no candidates) — never counted as correct. */
  abstained: number;
  /** `correct / total`, or 0 when `total` is 0. */
  accuracy: number;
  /** `generalizedCorrect / correct`, or 0 when `correct` is 0. */
  generalizationRate: number;
};

/**
 * Held-out generalization eval: for every position in every held-out sequence,
 * ask the model to predict the next token from the preceding context and score
 * exact-match accuracy, tracking how much correctness came from backed-off
 * (generalized) predictions vs. memorized full-order contexts.
 */
export function evaluateGeneralization(
  backend: LocalModelBackend,
  model: TrainedMovementModel,
  heldOut: MovementSequence[],
): GeneralizationReport {
  let total = 0;
  let correct = 0;
  let generalizedCorrect = 0;
  let abstained = 0;

  for (const sequence of heldOut) {
    for (let i = 0; i < sequence.tokens.length; i += 1) {
      const context = sequence.tokens.slice(0, i);
      const expected = sequence.tokens[i]!;
      const prediction = backend.predict(model, context);
      total += 1;
      if (prediction.action === undefined) {
        abstained += 1;
        continue;
      }
      if (prediction.action === expected) {
        correct += 1;
        if (prediction.generalized) {
          generalizedCorrect += 1;
        }
      }
    }
  }

  return {
    total,
    correct,
    generalizedCorrect,
    abstained,
    accuracy: total === 0 ? 0 : correct / total,
    generalizationRate: correct === 0 ? 0 : generalizedCorrect / correct,
  };
}
