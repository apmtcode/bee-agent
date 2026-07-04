// Pluggable local movement-model backend.
//
// Standing objective #2 (local-movement learning subsystem) requires bee-agent
// to (c) post-train a local model on recorded movements to repeat them and
// (d) generalize to new but related movements. The *actual* on-device training
// (mlx / axolotl) is described by `runner.ts`, which emits a launch plan that
// runs on the user's Apple-Silicon machine. That heavy backend cannot run in
// the cloud, so this module defines the pluggable seam plus a lightweight,
// fully-deterministic backend that CI (and every self-evolution run) can train
// and evaluate end-to-end without any OS access.
//
// A "movement" is modelled as an ordered stream of action *tokens* (see
// `movement-dataset.ts`). Learning to repeat/generalize movements is therefore
// next-action prediction over that token stream — the same shape whether the
// backend is a Markov table (here) or a real fine-tuned small model.

export type MovementToken = string;

export type MovementExample = {
  /** Prior action tokens, oldest first; the immediate predecessor is last. */
  context: MovementToken[];
  /** The action token that followed `context`. */
  next: MovementToken;
};

export type MovementDataset = {
  version: 1;
  /** Sorted, de-duplicated set of every token seen across all examples. */
  vocabulary: MovementToken[];
  examples: MovementExample[];
};

export type MovementPrediction = {
  token: MovementToken;
  /** Probability mass assigned to `token` at the order used, in `[0, 1]`. */
  confidence: number;
  /** How many context tokens were actually matched (0 = unigram fallback). */
  order: number;
  /** Other candidate tokens, most likely first (excludes the chosen token). */
  alternatives: Array<{ token: MovementToken; confidence: number }>;
};

export type MovementModelArtifact = {
  backend: string;
  version: 1;
  vocabulary: MovementToken[];
  trainedExampleCount: number;
  /** Backend-specific, JSON-serialisable weights. */
  payload: unknown;
};

/**
 * The contract every movement-model backend implements. A backend is trained
 * on a {@link MovementDataset} into a serialisable {@link MovementModelArtifact}
 * and answers next-action queries. Keep it JSON-round-trippable so artifacts can
 * be persisted, shipped between the cloud planner and a local trainer, and
 * reloaded for inference without the original backend instance.
 */
export interface MovementModelBackend {
  readonly name: string;
  train(dataset: MovementDataset): Promise<MovementModelArtifact>;
  predict(model: MovementModelArtifact, context: MovementToken[]): MovementPrediction;
}

const CONTEXT_SEPARATOR = "␟";

type OrderCounts = Record<string, Record<MovementToken, number>>;

type MarkovPayload = {
  maxOrder: number;
  /** counts[k] maps a k-token context key to observed next-token counts. */
  counts: OrderCounts[];
};

export type MarkovMovementBackendOptions = {
  /** Longest context suffix the model conditions on (>= 0). Default 3. */
  maxOrder?: number;
};

/**
 * Deterministic back-off n-gram backend — the reference/mock implementation.
 *
 * `train` tallies, for every order `k` in `0..maxOrder`, how often each next
 * token follows each length-`k` context suffix. `predict` starts from the
 * longest matchable suffix and backs off toward the unigram distribution, so
 * an unseen full context still yields a sensible prediction from a shorter,
 * seen suffix — this is what lets it *generalize* to new-but-related movements
 * rather than only echoing exact recorded prefixes.
 *
 * Ties (equal counts) break by token string ascending, so results are stable
 * across runs and machines — a hard requirement for reproducible evals.
 */
export class MarkovMovementBackend implements MovementModelBackend {
  readonly name = "markov-backoff";
  private readonly maxOrder: number;

  constructor(options: MarkovMovementBackendOptions = {}) {
    this.maxOrder = Math.max(0, Math.trunc(options.maxOrder ?? 3));
  }

  async train(dataset: MovementDataset): Promise<MovementModelArtifact> {
    const counts: OrderCounts[] = Array.from({ length: this.maxOrder + 1 }, () => ({}));
    for (const example of dataset.examples) {
      for (let order = 0; order <= this.maxOrder; order += 1) {
        if (order > example.context.length) {
          break;
        }
        const suffix = order === 0 ? [] : example.context.slice(example.context.length - order);
        const key = suffix.join(CONTEXT_SEPARATOR);
        const table = counts[order];
        const bucket = (table[key] ??= {});
        bucket[example.next] = (bucket[example.next] ?? 0) + 1;
      }
    }
    const payload: MarkovPayload = { maxOrder: this.maxOrder, counts };
    return {
      backend: this.name,
      version: 1,
      vocabulary: [...dataset.vocabulary],
      trainedExampleCount: dataset.examples.length,
      payload,
    };
  }

  predict(model: MovementModelArtifact, context: MovementToken[]): MovementPrediction {
    const payload = model.payload as MarkovPayload;
    const maxOrder = Math.min(payload.maxOrder, context.length);
    for (let order = maxOrder; order >= 0; order -= 1) {
      const suffix = order === 0 ? [] : context.slice(context.length - order);
      const key = suffix.join(CONTEXT_SEPARATOR);
      const bucket = payload.counts[order]?.[key];
      if (!bucket) {
        continue;
      }
      const ranked = rankBucket(bucket);
      if (ranked.length === 0) {
        continue;
      }
      const [best, ...rest] = ranked;
      return {
        token: best.token,
        confidence: best.confidence,
        order,
        alternatives: rest,
      };
    }
    // Nothing was learned at any order (empty model) — surface a defined,
    // zero-confidence miss rather than throwing, so callers can chain safely.
    return { token: "", confidence: 0, order: -1, alternatives: [] };
  }
}

function rankBucket(bucket: Record<MovementToken, number>): Array<{ token: MovementToken; confidence: number }> {
  const total = Object.values(bucket).reduce((sum, count) => sum + count, 0);
  if (total === 0) {
    return [];
  }
  return Object.entries(bucket)
    .map(([token, count]) => ({ token, confidence: count / total }))
    .sort((a, b) => (b.confidence !== a.confidence ? b.confidence - a.confidence : a.token < b.token ? -1 : 1));
}

export type MovementEvalResult = {
  /** Total held-out (context, next) pairs scored. */
  total: number;
  /** Pairs where the top prediction equalled the recorded next token. */
  correct: number;
  /** `correct / total`, or 0 when `total` is 0. */
  accuracy: number;
  /** Mean confidence the model assigned to its own top prediction. */
  meanConfidence: number;
  /** Correct pairs that required backing off below full context (generalized). */
  generalizedCorrect: number;
};

/**
 * Measure next-action fidelity on held-out examples — the generalization eval
 * harness. `generalizedCorrect` counts hits where the model matched on a
 * *shorter* suffix than the full context, i.e. it had to generalize rather than
 * replay an exactly-seen prefix. Feed it examples built from held-out
 * trajectories (see `splitMovementDataset`) to measure true generalization.
 */
export function evaluateMovementModel(
  backend: MovementModelBackend,
  model: MovementModelArtifact,
  heldOut: MovementExample[],
): MovementEvalResult {
  let correct = 0;
  let generalizedCorrect = 0;
  let confidenceSum = 0;
  for (const example of heldOut) {
    const prediction = backend.predict(model, example.context);
    confidenceSum += prediction.confidence;
    if (prediction.token === example.next) {
      correct += 1;
      if (prediction.order < example.context.length) {
        generalizedCorrect += 1;
      }
    }
  }
  const total = heldOut.length;
  return {
    total,
    correct,
    accuracy: total === 0 ? 0 : correct / total,
    meanConfidence: total === 0 ? 0 : confidenceSum / total,
    generalizedCorrect,
  };
}
