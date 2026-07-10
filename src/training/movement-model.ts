import type { ReplayTimelineEvent } from "../capture/replay.js";
import {
  buildMovementDataset,
  movementActionKey,
  type MovementActionToken,
  type MovementContextSignature,
  type MovementDataset,
} from "./movement-dataset.js";

/**
 * Pluggable local-model backend for the movement-learning subsystem.
 *
 * The capture pipeline records movements into replay timelines; {@link
 * buildMovementDataset} reshapes them into supervised (context → action)
 * transitions. A backend learns from those transitions and exposes an inference
 * session that predicts the next action for a given context — the on-device
 * "repeat + generalize the recorded movements" capability from the roadmap.
 *
 * The interface is intentionally backend-agnostic: the real deployment swaps in
 * an on-device small model (e.g. an MLX/GGUF next-action policy), while cloud
 * and CI use {@link DeterministicMarkovMovementBackend}, which trains and infers
 * in-process with zero external dependencies and fully deterministic output.
 */
export interface MovementModelBackend {
  readonly id: string;
  train(dataset: MovementDataset, options?: MovementTrainOptions): Promise<TrainedMovementModel>;
  load(model: TrainedMovementModel): MovementInferenceSession;
}

export type MovementTrainOptions = {
  /** Optional human-facing label recorded in the trained model metadata. */
  label?: string;
};

export type TrainedMovementModel = {
  backendId: string;
  version: 1;
  label?: string;
  contextWindow: number;
  trainedTransitions: number;
  vocabulary: string[];
  /** Backend-specific, JSON-serializable learned parameters. */
  parameters: unknown;
};

export type MovementPrediction = {
  action: MovementActionToken;
  /** Share of the matched context's distribution that chose this action (0..1). */
  confidence: number;
  /** The context the prediction was actually keyed on (may be a backoff key). */
  matchedContext: MovementContextSignature;
  /** True when no exact-context match existed and a coarser key was used. */
  backoff: boolean;
};

export interface MovementInferenceSession {
  /** Predict the next action for a precomputed context signature. */
  predictNext(
    context: MovementContextSignature,
    backoffContext?: MovementContextSignature,
  ): MovementPrediction | undefined;
  /** Predict the next action from a raw window of preceding replay events. */
  predictFromEvents(precedingEvents: ReplayTimelineEvent[]): MovementPrediction | undefined;
}

// --- Deterministic Markov backend (mock / CI-safe default) ------------------

type CountedAction = { action: MovementActionToken; count: number };

type MarkovParameters = {
  kind: "markov-v1";
  /** context signature -> action key -> counted action. */
  contexts: Record<string, Record<string, CountedAction>>;
  /** coarse backoff signature -> action key -> counted action. */
  backoff: Record<string, Record<string, CountedAction>>;
};

/**
 * A count-based Markov next-action model.
 *
 * Training tallies, per context signature, how often each action followed it
 * (plus a coarser backoff table keyed on the single most-recent event).
 * Inference picks the argmax action for a context; on an unseen context it falls
 * back to the coarse table, which is how it generalizes to related-but-unseen
 * situations. Ties break lexicographically by action key so output is fully
 * deterministic — no randomness, no wall-clock, safe to run in the cloud/CI.
 */
export class DeterministicMarkovMovementBackend implements MovementModelBackend {
  readonly id = "markov-v1";

  async train(dataset: MovementDataset, options: MovementTrainOptions = {}): Promise<TrainedMovementModel> {
    const contexts: MarkovParameters["contexts"] = {};
    const backoff: MarkovParameters["backoff"] = {};
    const vocabulary = new Set<string>();

    for (const sequence of dataset.sequences) {
      for (const transition of sequence.transitions) {
        const key = movementActionKey(transition.action);
        vocabulary.add(key);
        tally(contexts, transition.context, key, transition.action);
        tally(backoff, transition.backoffContext, key, transition.action);
      }
    }

    return {
      backendId: this.id,
      version: 1,
      ...(options.label ? { label: options.label } : {}),
      contextWindow: dataset.contextWindow,
      trainedTransitions: dataset.transitionCount,
      vocabulary: [...vocabulary].sort(),
      parameters: { kind: "markov-v1", contexts, backoff } satisfies MarkovParameters,
    };
  }

  load(model: TrainedMovementModel): MovementInferenceSession {
    const parameters = model.parameters as MarkovParameters | undefined;
    if (!parameters || parameters.kind !== "markov-v1") {
      throw new Error(`markov backend cannot load model with parameters kind "${(parameters as { kind?: string } | undefined)?.kind ?? "unknown"}"`);
    }
    const contextWindow = model.contextWindow;
    return {
      predictNext(context, backoffContext) {
        const exact = argmax(parameters.contexts[context]);
        if (exact) {
          return { ...exact, matchedContext: context, backoff: false };
        }
        const fallbackKey = backoffContext ?? deriveBackoffKey(context);
        const coarse = argmax(parameters.backoff[fallbackKey]);
        if (coarse) {
          return { ...coarse, matchedContext: fallbackKey, backoff: true };
        }
        return undefined;
      },
      predictFromEvents(precedingEvents) {
        // Reuse the dataset builder so context signatures are computed
        // identically to training. Append a sentinel action so the builder
        // emits exactly one transition whose context is the supplied window.
        const sentinel: ReplayTimelineEvent = {
          kind: "action",
          ts: Number.MAX_SAFE_INTEGER,
          trajectoryId: "__probe__",
          tool: "__probe__",
          summary: "__probe__",
        };
        const dataset = buildMovementDataset(
          [{ sessionId: "__probe__", trajectoryIds: ["__probe__"], events: [...precedingEvents, sentinel] }],
          { contextWindow },
        );
        const probe = dataset.sequences[0]?.transitions[0];
        if (!probe) {
          return undefined;
        }
        return this.predictNext(probe.context, probe.backoffContext);
      },
    };
  }
}

function tally(
  table: Record<string, Record<string, CountedAction>>,
  contextKey: string,
  actionKey: string,
  action: MovementActionToken,
): void {
  const bucket = (table[contextKey] ??= {});
  const existing = bucket[actionKey];
  if (existing) {
    existing.count += 1;
  } else {
    bucket[actionKey] = { action: { ...action }, count: 1 };
  }
}

function argmax(
  bucket: Record<string, CountedAction> | undefined,
): { action: MovementActionToken; confidence: number } | undefined {
  if (!bucket) {
    return undefined;
  }
  const entries = Object.entries(bucket);
  if (entries.length === 0) {
    return undefined;
  }
  const total = entries.reduce((sum, [, value]) => sum + value.count, 0);
  entries.sort((a, b) => {
    if (a[1].count !== b[1].count) {
      return b[1].count - a[1].count;
    }
    return a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0;
  });
  const [, best] = entries[0];
  return { action: { ...best.action }, confidence: total > 0 ? best.count / total : 0 };
}

function deriveBackoffKey(context: MovementContextSignature): MovementContextSignature {
  const parts = context.split(" | ");
  return parts.at(-1) ?? context;
}

// --- Trainer + generalization eval ------------------------------------------

export type MovementSequenceEval = {
  sequenceId: string;
  transitions: number;
  exactMatches: number;
  backoffMatches: number;
};

export type MovementEvalReport = {
  backendId: string;
  totalTransitions: number;
  exactMatches: number;
  backoffMatches: number;
  misses: number;
  /** (exact + backoff) / total — overall next-action fidelity, 0..1. */
  fidelity: number;
  /** exact / total — fidelity from exact-context matches only, 0..1. */
  exactFidelity: number;
  perSequence: MovementSequenceEval[];
};

/**
 * Orchestrates dataset → backend training → inference, and scores next-action
 * fidelity on a held-out dataset (the generalization signal).
 */
export class MovementModelTrainer {
  constructor(private readonly backend: MovementModelBackend) {}

  async train(dataset: MovementDataset, options?: MovementTrainOptions): Promise<TrainedMovementModel> {
    return await this.backend.train(dataset, options);
  }

  /**
   * Replay `holdout` through the trained model and measure how often it
   * reproduces the recorded action. When `holdout` is the training set this
   * measures reproduction fidelity; when it is a disjoint split it measures
   * generalization to related-but-unseen movements.
   */
  evaluate(model: TrainedMovementModel, holdout: MovementDataset): MovementEvalReport {
    const session = this.backend.load(model);
    let exactMatches = 0;
    let backoffMatches = 0;
    let totalTransitions = 0;
    const perSequence: MovementSequenceEval[] = [];

    for (const sequence of holdout.sequences) {
      let seqExact = 0;
      let seqBackoff = 0;
      for (const transition of sequence.transitions) {
        totalTransitions += 1;
        const prediction = session.predictNext(transition.context, transition.backoffContext);
        if (!prediction) {
          continue;
        }
        const predictedKey = movementActionKey(prediction.action);
        const actualKey = movementActionKey(transition.action);
        if (predictedKey !== actualKey) {
          continue;
        }
        if (prediction.backoff) {
          seqBackoff += 1;
        } else {
          seqExact += 1;
        }
      }
      exactMatches += seqExact;
      backoffMatches += seqBackoff;
      perSequence.push({
        sequenceId: sequence.id,
        transitions: sequence.transitions.length,
        exactMatches: seqExact,
        backoffMatches: seqBackoff,
      });
    }

    const matched = exactMatches + backoffMatches;
    return {
      backendId: model.backendId,
      totalTransitions,
      exactMatches,
      backoffMatches,
      misses: totalTransitions - matched,
      fidelity: totalTransitions > 0 ? matched / totalTransitions : 0,
      exactFidelity: totalTransitions > 0 ? exactMatches / totalTransitions : 0,
      perSequence,
    };
  }
}
