import type { ReplayManifest, ReplayTimelineEvent } from "../capture/replay.js";

/**
 * Movement-model subsystem (standing objective #2, parts c & d).
 *
 * The capture pipeline (`src/capture`) records movements as ordered
 * {@link ReplayTimelineEvent} timelines. This module turns those timelines into
 * a learnable dataset, defines a *pluggable* local-model backend interface, and
 * ships a deterministic in-process backend that can (c) repeat recorded
 * movements and (d) generalize to related-but-unseen ones — all without any real
 * OS access, so it is fully exercisable in the cloud/CI.
 *
 * The real on-device training path (`src/training/runner.ts`) shells out to
 * mlx/axolotl; that runs only on the user's machine. This backend interface is
 * the seam a real on-device model plugs into: implement {@link MovementModelBackend}
 * and register it in a {@link MovementModelRegistry}.
 */

/** Sentinel prepended to every training sequence so the first real token has context. */
const START_TOKEN = ":start";
/** Sentinel appended to every training sequence so the model learns when a movement ends. */
const END_TOKEN = ":end";
const CONTEXT_SEPARATOR = "";

/** A discrete token summarising a single movement event — the unit the model predicts. */
export type MovementToken = string;

/** One ordered movement sequence (typically one reviewed trajectory or replay). */
export type MovementSequence = {
  id: string;
  tokens: MovementToken[];
};

/** A collection of movement sequences to train or evaluate on. */
export type MovementDataset = {
  sequences: MovementSequence[];
};

/** Prediction context: the movement tokens observed so far (most recent last). */
export type MovementContext = {
  history: MovementToken[];
};

/** A single candidate for the next movement token. `token: undefined` ⇒ end of movement. */
export type MovementCandidate = {
  token: MovementToken | undefined;
  probability: number;
};

/** The model's prediction for the next movement token. */
export type MovementPrediction = {
  /** Argmax next token, or `undefined` when the movement is predicted to end. */
  token: MovementToken | undefined;
  /** Probability mass of the chosen token within the backoff level used (0..1). */
  confidence: number;
  /** Context length (n-gram order) that produced the prediction; 0 = unigram fallback. */
  order: number;
  /** All candidates at the winning backoff level, sorted most-probable first. */
  candidates: MovementCandidate[];
};

export type MovementGenerationOptions = {
  /** Hard cap on generated tokens (defence against a degenerate non-terminating model). */
  maxSteps?: number;
};

/** A trained, ready-to-infer movement model. */
export interface TrainedMovementModel {
  /** Name of the backend that produced this model. */
  readonly backend: string;
  /** Predict the next movement token given prior movement history. */
  predictNext(context: MovementContext): MovementPrediction;
  /**
   * Deterministically generate a full movement continuation from a seed context,
   * stopping at the learned end-of-movement or after `maxSteps`.
   */
  generate(seed: MovementContext, options?: MovementGenerationOptions): MovementToken[];
  /** Serialize to a plain JSON-safe object so the model can be persisted/reloaded. */
  serialize(): SerializedMovementModel;
}

export type MovementTrainingOptions = {
  /** Maximum n-gram context length. Higher = more faithful replay, less generalization. */
  maxOrder?: number;
};

/** Pluggable backend that turns a dataset into a {@link TrainedMovementModel}. */
export interface MovementModelBackend {
  readonly name: string;
  train(dataset: MovementDataset, options?: MovementTrainingOptions): Promise<TrainedMovementModel>;
}

/** JSON-safe serialized form of a Markov movement model. */
export type SerializedMovementModel = {
  version: 1;
  backend: string;
  maxOrder: number;
  /** counts[order][contextKey][token] = observation count. */
  counts: Record<string, Record<string, Record<string, number>>>;
};

// --- Tokenization -----------------------------------------------------------

/** Collapse a replay timeline event into a coarse, learnable movement token. */
export function tokenizeMovementEvent(event: ReplayTimelineEvent): MovementToken {
  switch (event.kind) {
    case "action":
      return `action:${event.tool}`;
    case "observation":
      return `observation:${event.source}`;
    case "transcript":
      return `transcript:${event.role}`;
  }
}

/** Build a training sequence from an ordered list of replay events. */
export function movementSequenceFromEvents(id: string, events: ReplayTimelineEvent[]): MovementSequence {
  return { id, tokens: events.map(tokenizeMovementEvent) };
}

/** Build a training sequence from a replay manifest. */
export function movementSequenceFromReplay(manifest: Pick<ReplayManifest, "sessionId" | "events">): MovementSequence {
  return movementSequenceFromEvents(manifest.sessionId, manifest.events);
}

/** Assemble a dataset from many replay manifests (e.g. a reviewed export's `replays`). */
export function datasetFromReplays(
  replays: Array<Pick<ReplayManifest, "sessionId" | "events">>,
): MovementDataset {
  return { sequences: replays.map(movementSequenceFromReplay) };
}

// --- Deterministic backoff Markov backend -----------------------------------

/**
 * A stupid-backoff n-gram model over movement tokens. Deterministic (argmax with
 * lexicographic tie-break) so training and inference are reproducible in tests
 * and across the cloud/local split. Higher orders reproduce recorded movements
 * verbatim; backoff to shorter contexts is what lets it generalize to related
 * but unseen prefixes.
 */
export class MarkovMovementModel implements TrainedMovementModel {
  readonly backend: string;

  constructor(
    private readonly maxOrder: number,
    /** counts[order] maps a context key → (token → count). */
    private readonly counts: Map<number, Map<string, Map<string, number>>>,
    backend = "markov",
  ) {
    this.backend = backend;
  }

  predictNext(context: MovementContext): MovementPrediction {
    const padded = padHistory(context.history, this.maxOrder);
    for (let order = this.maxOrder; order >= 0; order -= 1) {
      const contextTokens = order === 0 ? [] : padded.slice(padded.length - order);
      const key = contextKey(contextTokens);
      const table = this.counts.get(order)?.get(key);
      if (!table || table.size === 0) {
        continue;
      }
      const candidates = rankCandidates(table);
      const winner = candidates[0];
      return {
        token: fromToken(winner.token),
        confidence: winner.probability,
        order,
        candidates: candidates.map((candidate) => ({
          token: fromToken(candidate.token),
          probability: candidate.probability,
        })),
      };
    }
    return { token: undefined, confidence: 0, order: 0, candidates: [] };
  }

  generate(seed: MovementContext, options: MovementGenerationOptions = {}): MovementToken[] {
    const maxSteps = options.maxSteps ?? 256;
    const history = [...seed.history];
    const generated: MovementToken[] = [];
    for (let step = 0; step < maxSteps; step += 1) {
      const prediction = this.predictNext({ history });
      if (prediction.token === undefined) {
        break;
      }
      generated.push(prediction.token);
      history.push(prediction.token);
    }
    return generated;
  }

  serialize(): SerializedMovementModel {
    const counts: SerializedMovementModel["counts"] = {};
    for (const [order, contexts] of this.counts) {
      const orderKey = String(order);
      counts[orderKey] = {};
      for (const [ctx, tokens] of contexts) {
        counts[orderKey][ctx] = Object.fromEntries(tokens);
      }
    }
    return { version: 1, backend: this.backend, maxOrder: this.maxOrder, counts };
  }

  static deserialize(serialized: SerializedMovementModel): MarkovMovementModel {
    const counts = new Map<number, Map<string, Map<string, number>>>();
    for (const [orderKey, contexts] of Object.entries(serialized.counts)) {
      const order = Number(orderKey);
      const contextMap = new Map<string, Map<string, number>>();
      for (const [ctx, tokens] of Object.entries(contexts)) {
        contextMap.set(ctx, new Map(Object.entries(tokens)));
      }
      counts.set(order, contextMap);
    }
    return new MarkovMovementModel(serialized.maxOrder, counts, serialized.backend);
  }
}

/** Deterministic in-process backend — the cloud/CI-safe default. */
export class MarkovMovementBackend implements MovementModelBackend {
  readonly name = "markov";

  async train(dataset: MovementDataset, options: MovementTrainingOptions = {}): Promise<TrainedMovementModel> {
    const maxOrder = Math.max(0, options.maxOrder ?? 2);
    const counts = new Map<number, Map<string, Map<string, number>>>();
    for (let order = 0; order <= maxOrder; order += 1) {
      counts.set(order, new Map());
    }

    for (const sequence of dataset.sequences) {
      const augmented = [
        ...Array<MovementToken>(maxOrder).fill(START_TOKEN),
        ...sequence.tokens,
        END_TOKEN,
      ];
      // Target every position after the START padding.
      for (let index = maxOrder; index < augmented.length; index += 1) {
        const target = augmented[index]!;
        for (let order = 0; order <= maxOrder; order += 1) {
          const contextTokens = augmented.slice(index - order, index);
          const key = contextKey(contextTokens);
          const table = counts.get(order)!;
          const row = table.get(key) ?? new Map<string, number>();
          row.set(target, (row.get(target) ?? 0) + 1);
          table.set(key, row);
        }
      }
    }

    return new MarkovMovementModel(maxOrder, counts, this.name);
  }
}

// --- Registry (pluggable backends) ------------------------------------------

/** Registry of movement-model backends so the backend is swappable at runtime. */
export class MovementModelRegistry {
  private readonly backends = new Map<string, MovementModelBackend>();

  register(backend: MovementModelBackend): this {
    this.backends.set(backend.name, backend);
    return this;
  }

  has(name: string): boolean {
    return this.backends.has(name);
  }

  get(name: string): MovementModelBackend {
    const backend = this.backends.get(name);
    if (!backend) {
      throw new Error(`unknown movement-model backend: ${name} (registered: ${this.list().join(", ") || "none"})`);
    }
    return backend;
  }

  list(): string[] {
    return [...this.backends.keys()].sort();
  }
}

/** A registry pre-populated with the deterministic in-process backend. */
export function defaultMovementModelRegistry(): MovementModelRegistry {
  return new MovementModelRegistry().register(new MarkovMovementBackend());
}

// --- Generalization eval harness --------------------------------------------

export type MovementSequenceEval = {
  id: string;
  steps: number;
  correct: number;
  accuracy: number;
};

export type MovementEvalResult = {
  sequenceCount: number;
  /** Total next-token predictions made (includes the terminal end-of-movement step). */
  predictedSteps: number;
  correct: number;
  /** Next-token accuracy across all held-out sequences (0..1). */
  accuracy: number;
  perSequence: MovementSequenceEval[];
};

/**
 * Measure replay fidelity / generalization: for every position in each held-out
 * sequence, predict the next token from the true prefix and compare to the actual
 * token (the terminal end-of-movement is scored too). Reports overall and
 * per-sequence next-token accuracy.
 */
export function evaluateMovementModel(
  model: TrainedMovementModel,
  heldOut: MovementDataset,
): MovementEvalResult {
  const perSequence: MovementSequenceEval[] = [];
  let totalSteps = 0;
  let totalCorrect = 0;

  for (const sequence of heldOut.sequences) {
    const targets: Array<MovementToken | undefined> = [...sequence.tokens, undefined];
    let correct = 0;
    for (let index = 0; index < targets.length; index += 1) {
      const history = sequence.tokens.slice(0, index);
      const prediction = model.predictNext({ history });
      if (prediction.token === targets[index]) {
        correct += 1;
      }
    }
    const steps = targets.length;
    perSequence.push({ id: sequence.id, steps, correct, accuracy: steps === 0 ? 0 : correct / steps });
    totalSteps += steps;
    totalCorrect += correct;
  }

  return {
    sequenceCount: heldOut.sequences.length,
    predictedSteps: totalSteps,
    correct: totalCorrect,
    accuracy: totalSteps === 0 ? 0 : totalCorrect / totalSteps,
    perSequence,
  };
}

// --- internal helpers -------------------------------------------------------

function padHistory(history: MovementToken[], maxOrder: number): MovementToken[] {
  if (history.length >= maxOrder) {
    return history;
  }
  return [...Array<MovementToken>(maxOrder - history.length).fill(START_TOKEN), ...history];
}

function contextKey(tokens: MovementToken[]): string {
  return tokens.join(CONTEXT_SEPARATOR);
}

function fromToken(token: MovementToken): MovementToken | undefined {
  return token === END_TOKEN ? undefined : token;
}

function rankCandidates(table: Map<string, number>): Array<{ token: MovementToken; probability: number }> {
  let total = 0;
  for (const count of table.values()) {
    total += count;
  }
  return [...table.entries()]
    .map(([token, count]) => ({ token, probability: total === 0 ? 0 : count / total }))
    .sort((a, b) => {
      if (b.probability !== a.probability) {
        return b.probability - a.probability;
      }
      return a.token < b.token ? -1 : a.token > b.token ? 1 : 0;
    });
}
