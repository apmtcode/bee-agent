/**
 * In-process, pluggable movement-model backend for the local-movement learning
 * subsystem (standing objective #2, parts c + d).
 *
 * The subsystem's on-device story is: record local movements → build a dataset →
 * post-train a small local model that can *repeat* the recorded movements and
 * *generalize* to related ones. The real on-device training happens when the user
 * runs bee-agent locally; this module provides the code + a deterministic
 * in-process backend so the whole pipeline is exercisable (and testable) in the
 * cloud with no OS access.
 *
 * The backend is pluggable: `MovementModelBackend` is the seam a real on-device
 * model (e.g. a small MLX/llama policy) can implement, while
 * `MarkovMovementBackend` is a dependency-free reference implementation that
 * learns an order-k Markov model with stupid-backoff smoothing over a movement
 * vocabulary.
 */

/** A single normalized movement/action step (mouse, keyboard, gesture, tool). */
export type MovementStep = {
  /** Producing surface/tool, e.g. "device", "keyboard", "mouse". */
  tool: string;
  /** Canonical verb, e.g. "tap", "swipe", "type", "click", "shortcut". */
  action: string;
  /** Optional UI target the movement acted on, e.g. "submit-button". */
  target?: string;
  /** Optional direction for swipes/scrolls. */
  direction?: "up" | "down" | "left" | "right";
  /** Optional summarized value (never raw secret content). */
  value?: string;
};

/** One recorded movement sequence (a replayable unit). */
export type MovementSequence = {
  id: string;
  steps: MovementStep[];
};

/** A training dataset: a set of recorded/synthetic movement sequences. */
export type MovementDataset = {
  sequences: MovementSequence[];
};

export type TrainOptions = {
  /** Markov context order (how many prior steps condition the next). Default 2. */
  order?: number;
};

export type GenerateOptions = {
  /** Hard cap on generated steps (safety bound). Default 64. */
  maxSteps?: number;
};

export type MovementPrediction = {
  /** The predicted next step (decoded from the model vocabulary). */
  step: MovementStep;
  /** Estimated probability under the (possibly backed-off) context. */
  probability: number;
  /** Markov order that actually produced the prediction (after backoff). */
  order: number;
  /** True when produced by backing off below the requested order. */
  backoff: boolean;
};

/** Sentinel token marking the end of a sequence, so generation terminates. */
export const MOVEMENT_EOS = "<eos>";

/** Serialized form of a trained Markov model (JSON-safe, persistable). */
export type SerializedMovementModel = {
  version: 1;
  backendId: string;
  order: number;
  /** token -> representative decoded step (EOS omitted). */
  vocabulary: Record<string, MovementStep>;
  /** contextKey -> (nextToken -> count). contextKey "" is the unigram table. */
  transitions: Record<string, Record<string, number>>;
};

/** The trained model surface. A real backend returns its own implementation. */
export interface TrainedMovementModel {
  readonly backendId: string;
  readonly order: number;
  readonly vocabulary: string[];
  /** Predict the single most likely next step given a movement context. */
  predictNext(context: MovementStep[]): MovementPrediction | undefined;
  /** Greedily roll out a full movement sequence from a prefix. */
  generate(prefix: MovementStep[], options?: GenerateOptions): MovementStep[];
  toJSON(): SerializedMovementModel;
}

/** Pluggable training backend. Swap `MarkovMovementBackend` for a real one. */
export interface MovementModelBackend {
  readonly id: string;
  train(dataset: MovementDataset, options?: TrainOptions): Promise<TrainedMovementModel>;
}

/**
 * Canonicalize a step into a stable vocabulary token. Steps that mean the same
 * movement map to the same token, which is what lets the model both repeat and
 * generalize. Field separators are chosen to never collide with the sentinel.
 */
export function tokenizeStep(step: MovementStep): string {
  const parts = [
    step.tool.trim().toLowerCase(),
    step.action.trim().toLowerCase(),
    step.target?.trim().toLowerCase() ?? "",
    step.direction ?? "",
    step.value?.trim().toLowerCase() ?? "",
  ];
  return parts.join("|");
}

function contextKey(tokens: string[]): string {
  return tokens.join(">>");
}

/**
 * Deterministic order-k Markov backend with stupid-backoff smoothing.
 * No randomness, no clock, no OS access — safe for cloud/CI.
 */
export class MarkovMovementBackend implements MovementModelBackend {
  readonly id = "markov";

  async train(dataset: MovementDataset, options: TrainOptions = {}): Promise<TrainedMovementModel> {
    const order = Math.max(1, Math.floor(options.order ?? 2));
    const vocabulary = new Map<string, MovementStep>();
    // transitions[k] maps a k-length contextKey -> Map<nextToken, count>.
    const transitions = new Map<string, Map<string, number>>();

    const bump = (context: string[], next: string): void => {
      const key = contextKey(context);
      let row = transitions.get(key);
      if (!row) {
        row = new Map<string, number>();
        transitions.set(key, row);
      }
      row.set(next, (row.get(next) ?? 0) + 1);
    };

    for (const sequence of dataset.sequences) {
      const tokens: string[] = [];
      for (const step of sequence.steps) {
        const token = tokenizeStep(step);
        vocabulary.set(token, normalizeStep(step));
        tokens.push(token);
      }
      const stream = [...tokens, MOVEMENT_EOS];
      for (let i = 0; i < stream.length; i += 1) {
        const next = stream[i]!;
        // Record every context length from 0 (unigram) up to `order`.
        for (let k = 0; k <= order; k += 1) {
          if (i - k < 0) {
            break;
          }
          bump(stream.slice(i - k, i), next);
        }
      }
    }

    return new MarkovMovementModel(this.id, order, vocabulary, transitions);
  }
}

class MarkovMovementModel implements TrainedMovementModel {
  constructor(
    readonly backendId: string,
    readonly order: number,
    private readonly vocab: Map<string, MovementStep>,
    private readonly transitions: Map<string, Map<string, number>>,
  ) {}

  get vocabulary(): string[] {
    return [...this.vocab.keys()].sort();
  }

  predictNext(context: MovementStep[]): MovementPrediction | undefined {
    const contextTokens = context.map(tokenizeStep);
    const requested = Math.min(this.order, contextTokens.length);
    // Stupid backoff: try the longest context first, shorten on a miss.
    for (let k = requested; k >= 0; k -= 1) {
      const slice = k === 0 ? [] : contextTokens.slice(contextTokens.length - k);
      const row = this.transitions.get(contextKey(slice));
      if (!row || row.size === 0) {
        continue;
      }
      const best = argmax(row);
      if (!best) {
        continue;
      }
      const total = [...row.values()].reduce((sum, count) => sum + count, 0);
      const step = best.token === MOVEMENT_EOS ? undefined : this.vocab.get(best.token);
      if (best.token === MOVEMENT_EOS || !step) {
        // End-of-sequence is the argmax: there is no further movement to make.
        return undefined;
      }
      return {
        step,
        probability: total > 0 ? best.count / total : 0,
        order: k,
        backoff: k < requested,
      };
    }
    return undefined;
  }

  generate(prefix: MovementStep[], options: GenerateOptions = {}): MovementStep[] {
    const maxSteps = Math.max(0, Math.floor(options.maxSteps ?? 64));
    const produced: MovementStep[] = [];
    let context = [...prefix];
    while (produced.length < maxSteps) {
      const prediction = this.predictNext(context);
      if (!prediction) {
        break;
      }
      produced.push(prediction.step);
      context = [...context, prediction.step];
    }
    return produced;
  }

  toJSON(): SerializedMovementModel {
    const vocabulary: Record<string, MovementStep> = {};
    for (const [token, step] of [...this.vocab.entries()].sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0))) {
      vocabulary[token] = step;
    }
    const transitions: Record<string, Record<string, number>> = {};
    for (const [key, row] of [...this.transitions.entries()].sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0))) {
      const serializedRow: Record<string, number> = {};
      for (const [next, count] of [...row.entries()].sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0))) {
        serializedRow[next] = count;
      }
      transitions[key] = serializedRow;
    }
    return { version: 1, backendId: this.backendId, order: this.order, vocabulary, transitions };
  }
}

/** Rehydrate a persisted model without retraining. */
export function deserializeMovementModel(serialized: SerializedMovementModel): TrainedMovementModel {
  const vocab = new Map<string, MovementStep>(Object.entries(serialized.vocabulary));
  const transitions = new Map<string, Map<string, number>>();
  for (const [key, row] of Object.entries(serialized.transitions)) {
    transitions.set(key, new Map(Object.entries(row)));
  }
  return new MarkovMovementModel(serialized.backendId, serialized.order, vocab, transitions);
}

/**
 * Generalization / replay-fidelity eval harness (roadmap item). Given a trained
 * model and held-out sequences, measure how faithfully the model's greedy
 * continuation reproduces each sequence from a seed prefix.
 */
export type ReplayFidelityReport = {
  sequenceCount: number;
  /** Fraction of predicted steps whose token matched the expected step. */
  stepAccuracy: number;
  /** Fraction of sequences reproduced exactly (token-for-token). */
  exactSequenceRate: number;
  perSequence: Array<{
    id: string;
    matchedSteps: number;
    expectedSteps: number;
    exact: boolean;
  }>;
};

export function evaluateReplayFidelity(
  model: TrainedMovementModel,
  heldOut: MovementSequence[],
  options: { seedSteps?: number } = {},
): ReplayFidelityReport {
  const seedSteps = Math.max(0, Math.floor(options.seedSteps ?? 1));
  let totalExpected = 0;
  let totalMatched = 0;
  let exactSequences = 0;
  const perSequence = heldOut.map((sequence) => {
    const seed = sequence.steps.slice(0, seedSteps);
    const expectedTail = sequence.steps.slice(seedSteps);
    const generated = model.generate(seed, { maxSteps: expectedTail.length });
    let matched = 0;
    for (let i = 0; i < expectedTail.length; i += 1) {
      const expectedToken = tokenizeStep(expectedTail[i]!);
      const generatedStep = generated[i];
      if (generatedStep && tokenizeStep(generatedStep) === expectedToken) {
        matched += 1;
      }
    }
    const exact = matched === expectedTail.length && generated.length === expectedTail.length;
    totalExpected += expectedTail.length;
    totalMatched += matched;
    if (exact) {
      exactSequences += 1;
    }
    return { id: sequence.id, matchedSteps: matched, expectedSteps: expectedTail.length, exact };
  });

  return {
    sequenceCount: heldOut.length,
    stepAccuracy: totalExpected > 0 ? totalMatched / totalExpected : 1,
    exactSequenceRate: heldOut.length > 0 ? exactSequences / heldOut.length : 1,
    perSequence,
  };
}

function normalizeStep(step: MovementStep): MovementStep {
  const normalized: MovementStep = {
    tool: step.tool.trim().toLowerCase(),
    action: step.action.trim().toLowerCase(),
  };
  if (step.target !== undefined) {
    normalized.target = step.target;
  }
  if (step.direction !== undefined) {
    normalized.direction = step.direction;
  }
  if (step.value !== undefined) {
    normalized.value = step.value;
  }
  return normalized;
}

function argmax(row: Map<string, number>): { token: string; count: number } | undefined {
  let bestToken: string | undefined;
  let bestCount = -1;
  // Deterministic tie-break: highest count, then lexicographically smallest token.
  for (const [token, count] of [...row.entries()].sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0))) {
    if (count > bestCount) {
      bestCount = count;
      bestToken = token;
    }
  }
  return bestToken === undefined ? undefined : { token: bestToken, count: bestCount };
}
