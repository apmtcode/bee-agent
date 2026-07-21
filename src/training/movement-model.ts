// Pluggable local-movement model backend.
//
// Standing objective #2 (local-movement learning subsystem) needs a way to
// *post-train a local model* on recorded movements and have it *repeat* them and
// *generalize* to new-but-related movements. On a real machine that training runs
// on-device (see `LocalAppleSiliconTrainingRunner`, which emits an mlx/axolotl
// launch plan). In the cloud we have no device and no heavyweight ML runtime, so
// this module defines the *seam*: a `MovementModelBackend` interface plus a
// dependency-free, fully deterministic backend that actually trains and infers
// in-process. Tests exercise the whole capture -> dataset -> train -> infer ->
// generalize loop without touching the OS or a GPU.
//
// The deterministic backend is a variable-order Markov model over movement
// tokens with "stupid backoff": it memorises the recorded transitions (so it can
// reproduce a demonstrated sequence exactly) and falls back to shorter context
// suffixes — and finally the global next-token distribution — for contexts it has
// never seen, which is what lets it generalize to related movements.

import type { ReplayManifest, ReplayTimelineEvent } from "../capture/replay.js";
import type { TrajectoryAction, TrajectorySpan } from "../capture/trajectory.js";

/** Separator for composing a context suffix into a lookup key. Unit separator —
 * will not collide with realistic movement tokens. */
const CONTEXT_SEP = "";

/** How an action is reduced to a discrete movement token. Defaults to the tool
 * name, which is the coarse "movement type" (e.g. `mouse.move`, `key.press`);
 * override to fold in more of the summary for finer-grained models. */
export type MovementTokenizer = (action: Pick<TrajectoryAction, "tool" | "summary">) => string;

export const defaultMovementTokenizer: MovementTokenizer = (action) => action.tool;

/** One training row: predict `next` given the preceding `context` tokens. */
export type MovementExample = {
  context: string[];
  next: string;
};

/** A replayable movement dataset derived from recorded action sequences. */
export type MovementDataset = {
  version: 1;
  /** Highest context order (window size) the examples were windowed at. */
  maxOrder: number;
  /** Raw per-demonstration token sequences, preserved for round-trip replay. */
  sequences: string[][];
  /** N-gram windows over the sequences, orders `0..maxOrder`. */
  examples: MovementExample[];
  /** Sorted unique tokens observed across all sequences. */
  vocabulary: string[];
};

export type MovementPredictionCandidate = {
  token: string;
  probability: number;
  count: number;
};

export type MovementPrediction = {
  /** Best next token, or `undefined` when the model is empty. */
  token: string | undefined;
  /** Probability of `token` at the matched backoff order. */
  confidence: number;
  /** Context order actually used after backoff (`context.length` = exact match). */
  order: number;
  /** True when backoff dropped below the full supplied context. */
  backedOff: boolean;
  /** All continuations at the matched order, most likely first. */
  candidates: MovementPredictionCandidate[];
};

export type SerializedMovementModel = {
  version: 1;
  backendId: string;
  maxOrder: number;
  /** Per-order transition tables: order -> contextKey -> [token, count][]. */
  orders: Array<{
    order: number;
    entries: Array<{ key: string; counts: Array<[string, number]> }>;
  }>;
};

export type TrainMovementModelConfig = {
  /** Cap the context order the model conditions on. Defaults to the dataset's. */
  maxOrder?: number;
};

/** A trained model: repeat recorded movements and generalize to related ones. */
export interface TrainedMovementModel {
  readonly backendId: string;
  readonly maxOrder: number;
  /** Predict the next movement token given a context suffix. */
  predict(context: string[]): MovementPrediction;
  /** Roll out `steps` tokens from `seed`, feeding predictions back in. */
  generate(seed: string[], steps: number): string[];
  serialize(): SerializedMovementModel;
}

/** Pluggable training backend. The deterministic one below runs in-process; a
 * real on-device backend (mlx/axolotl) would implement the same shape and stream
 * to a GPU instead. */
export interface MovementModelBackend {
  readonly id: string;
  train(dataset: MovementDataset, config?: TrainMovementModelConfig): Promise<TrainedMovementModel>;
}

// --- dataset construction -------------------------------------------------

/** Window a set of token sequences into n-gram examples (orders 0..maxOrder). */
export function buildMovementDataset(sequences: string[][], maxOrder = 3): MovementDataset {
  if (maxOrder < 0) {
    throw new Error(`maxOrder must be >= 0, received ${maxOrder}`);
  }
  const examples: MovementExample[] = [];
  const vocabulary = new Set<string>();

  for (const sequence of sequences) {
    for (const token of sequence) {
      vocabulary.add(token);
    }
    for (let index = 0; index < sequence.length; index += 1) {
      const contextStart = Math.max(0, index - maxOrder);
      examples.push({
        context: sequence.slice(contextStart, index),
        next: sequence[index]!,
      });
    }
  }

  return {
    version: 1,
    maxOrder,
    sequences: sequences.map((sequence) => [...sequence]),
    examples,
    vocabulary: [...vocabulary].sort(),
  };
}

/** Extract ordered movement-token sequences from trajectory action logs. */
export function buildMovementSequencesFromTrajectories(
  trajectories: TrajectorySpan[],
  tokenize: MovementTokenizer = defaultMovementTokenizer,
): string[][] {
  return trajectories.map((trajectory) =>
    [...trajectory.actions]
      .sort((a, b) => a.ts - b.ts)
      .map((action) => tokenize(action)),
  );
}

/** Extract ordered movement-token sequences from replay manifests (action
 * events only, in timeline order). */
export function buildMovementSequencesFromReplays(
  manifests: ReplayManifest[],
  tokenize: MovementTokenizer = defaultMovementTokenizer,
): string[][] {
  return manifests.map((manifest) =>
    manifest.events
      .filter((event): event is Extract<ReplayTimelineEvent, { kind: "action" }> => event.kind === "action")
      .map((event) => tokenize({ tool: event.tool, summary: event.summary })),
  );
}

// --- deterministic backend ------------------------------------------------

/** In-process variable-order Markov backend with stupid backoff. Deterministic:
 * identical datasets always produce identical models and predictions, with ties
 * broken by first-observed order, so it is safe to assert on in CI. */
export class DeterministicMarkovBackend implements MovementModelBackend {
  readonly id = "deterministic-markov";

  async train(dataset: MovementDataset, config: TrainMovementModelConfig = {}): Promise<TrainedMovementModel> {
    const maxOrder = config.maxOrder ?? dataset.maxOrder;
    if (maxOrder < 0) {
      throw new Error(`maxOrder must be >= 0, received ${maxOrder}`);
    }
    // orders[o] maps a context key (last `o` tokens) -> ordered token counts.
    const orders: Array<Map<string, Map<string, number>>> = Array.from(
      { length: maxOrder + 1 },
      () => new Map<string, Map<string, number>>(),
    );

    for (const sequence of dataset.sequences) {
      for (let index = 0; index < sequence.length; index += 1) {
        const next = sequence[index]!;
        for (let order = 0; order <= maxOrder; order += 1) {
          if (order > index) {
            break;
          }
          const key = contextKey(sequence.slice(index - order, index));
          recordTransition(orders[order]!, key, next);
        }
      }
    }

    return new MarkovMovementModel(this.id, maxOrder, orders);
  }
}

class MarkovMovementModel implements TrainedMovementModel {
  constructor(
    readonly backendId: string,
    readonly maxOrder: number,
    private readonly orders: Array<Map<string, Map<string, number>>>,
  ) {}

  predict(context: string[]): MovementPrediction {
    const startOrder = Math.min(this.maxOrder, context.length);
    for (let order = startOrder; order >= 0; order -= 1) {
      const table = this.orders[order];
      if (!table) {
        continue;
      }
      const key = contextKey(context.slice(context.length - order));
      const counts = table.get(key);
      if (!counts || counts.size === 0) {
        continue;
      }
      const candidates = rankCandidates(counts);
      return {
        token: candidates[0]!.token,
        confidence: candidates[0]!.probability,
        order,
        backedOff: order < context.length,
        candidates,
      };
    }
    return { token: undefined, confidence: 0, order: 0, backedOff: context.length > 0, candidates: [] };
  }

  generate(seed: string[], steps: number): string[] {
    if (steps < 0) {
      throw new Error(`steps must be >= 0, received ${steps}`);
    }
    const generated: string[] = [];
    const context = [...seed];
    for (let step = 0; step < steps; step += 1) {
      const prediction = this.predict(context);
      if (prediction.token === undefined) {
        break;
      }
      generated.push(prediction.token);
      context.push(prediction.token);
    }
    return generated;
  }

  serialize(): SerializedMovementModel {
    return {
      version: 1,
      backendId: this.backendId,
      maxOrder: this.maxOrder,
      orders: this.orders.map((table, order) => ({
        order,
        entries: [...table.entries()]
          .sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0))
          .map(([key, counts]) => ({ key, counts: [...counts.entries()] })),
      })),
    };
  }
}

/** Reconstruct a trained model from its serialized form (portable artifact). */
export function loadMovementModel(serialized: SerializedMovementModel): TrainedMovementModel {
  const orders: Array<Map<string, Map<string, number>>> = Array.from(
    { length: serialized.maxOrder + 1 },
    () => new Map<string, Map<string, number>>(),
  );
  for (const { order, entries } of serialized.orders) {
    const table = orders[order];
    if (!table) {
      continue;
    }
    for (const { key, counts } of entries) {
      table.set(key, new Map(counts));
    }
  }
  return new MarkovMovementModel(serialized.backendId, serialized.maxOrder, orders);
}

/** Backend registry so the training pipeline can select a model by name and a
 * real on-device backend can be dropped in without touching call sites. */
const BACKEND_FACTORIES: Record<string, () => MovementModelBackend> = {
  "deterministic-markov": () => new DeterministicMarkovBackend(),
};

export type MovementModelBackendKind = keyof typeof BACKEND_FACTORIES;

export function createMovementModelBackend(kind: MovementModelBackendKind = "deterministic-markov"): MovementModelBackend {
  const factory = BACKEND_FACTORIES[kind];
  if (!factory) {
    throw new Error(`unknown movement-model backend: ${kind}`);
  }
  return factory();
}

// --- evaluation -----------------------------------------------------------

export type MovementEvalResult = {
  /** Number of held-out examples scored. */
  total: number;
  /** Examples whose predicted token matched the recorded next token. */
  correct: number;
  /** `correct / total` (0 when `total` is 0). */
  accuracy: number;
  /** Correct predictions that required backing off below the full context —
   * evidence of generalization to unseen exact contexts. */
  generalizedCorrect: number;
  /** `generalizedCorrect / correct` (0 when `correct` is 0). */
  generalizationRate: number;
};

/** Next-token top-1 accuracy over held-out examples, plus a generalization rate
 * (share of correct predictions that came from a backed-off context). */
export function evaluateMovementModel(model: TrainedMovementModel, heldOut: MovementExample[]): MovementEvalResult {
  let correct = 0;
  let generalizedCorrect = 0;
  for (const example of heldOut) {
    const prediction = model.predict(example.context);
    if (prediction.token === example.next) {
      correct += 1;
      if (prediction.backedOff) {
        generalizedCorrect += 1;
      }
    }
  }
  const total = heldOut.length;
  return {
    total,
    correct,
    accuracy: total === 0 ? 0 : correct / total,
    generalizedCorrect,
    generalizationRate: correct === 0 ? 0 : generalizedCorrect / correct,
  };
}

// --- internals ------------------------------------------------------------

function contextKey(tokens: string[]): string {
  return tokens.join(CONTEXT_SEP);
}

function recordTransition(table: Map<string, Map<string, number>>, key: string, token: string): void {
  let counts = table.get(key);
  if (!counts) {
    counts = new Map<string, number>();
    table.set(key, counts);
  }
  counts.set(token, (counts.get(token) ?? 0) + 1);
}

/** Rank continuations by count desc, ties broken by first-observed order (Map
 * iteration order), yielding fully deterministic predictions. */
function rankCandidates(counts: Map<string, number>): MovementPredictionCandidate[] {
  const total = [...counts.values()].reduce((sum, count) => sum + count, 0);
  const entries = [...counts.entries()];
  const ranked = entries
    .map(([token, count], index) => ({ token, count, index }))
    .sort((a, b) => (b.count - a.count) || (a.index - b.index));
  return ranked.map(({ token, count }) => ({
    token,
    count,
    probability: total === 0 ? 0 : count / total,
  }));
}
