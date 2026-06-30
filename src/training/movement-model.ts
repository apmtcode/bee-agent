// Pluggable local movement-model backend.
//
// Objective #2(d): bee-agent must be able to post-train a local model on the
// recorded movement dataset to (c) repeat recorded movements and (d) generalize
// to new-but-related movements. The on-device training plan/launch script lives
// in `runner.ts`/`execution-service.ts` and runs MLX/Axolotl on the user's
// machine. This module provides the *in-process, deterministic* counterpart:
//
//   - a stable movement token + dataset shape derived from reviewed replays,
//   - a pluggable `MovementModelBackend` seam (swap in a real on-device model),
//   - a deterministic Markov back-off backend that actually trains + infers in
//     the cloud/CI with no OS access, so the pipeline is testable end-to-end,
//   - a replay-fidelity / generalization eval harness.
//
// The Markov backend is intentionally simple and dependency-free: a
// variable-order back-off n-gram over movement tokens. Back-off is what gives
// it generalization — when the exact preceding n-gram was never recorded, it
// falls back to a shorter, related context and still predicts a sensible next
// movement.

export type MovementToken = string;

/** A single recorded trajectory rendered as an ordered list of movement tokens. */
export type MovementSequence = {
  trajectoryId: string;
  sessionId: string;
  tokens: MovementToken[];
};

export type MovementDataset = {
  version: 1;
  sequences: MovementSequence[];
  /** Sorted, de-duplicated set of every token seen across all sequences. */
  vocabulary: MovementToken[];
};

export type MovementPrediction = {
  /** Most-likely next movement, or `null` when the model has no continuation. */
  token: MovementToken | null;
  /** Probability of `token` within the (post-back-off) context distribution. */
  probability: number;
  /** Context order actually used after back-off (<= requested order). */
  contextOrder: number;
  /** True when the requested context was unseen and the model backed off. */
  backedOff: boolean;
  /** Full ranked candidate list (deterministic: count desc, then token asc). */
  ranked: Array<{ token: MovementToken; probability: number }>;
};

export type MovementTrainOptions = {
  /** Maximum n-gram context length. Defaults to 3. */
  order?: number;
};

export interface MovementModel {
  readonly backendId: string;
}

/**
 * The pluggable seam. A real on-device small model can implement this same
 * interface (e.g. wrapping an MLX adapter) and be dropped into the registry
 * without touching callers.
 */
export interface MovementModelBackend<TModel extends MovementModel = MovementModel> {
  readonly id: string;
  train(dataset: MovementDataset, options?: MovementTrainOptions): TModel;
  predictNext(model: TModel, context: MovementToken[]): MovementPrediction;
  generate(model: TModel, seed: MovementToken[], steps: number): MovementToken[];
  serialize(model: TModel): string;
  deserialize(serialized: string): TModel;
}

// ---------------------------------------------------------------------------
// Dataset construction from reviewed replays
// ---------------------------------------------------------------------------

export type MovementActionEvent = {
  kind: "action";
  ts: number;
  trajectoryId: string;
  tool: string;
  summary: string;
};

type ReplayLike = {
  sessionId: string;
  events: Array<{ kind: string; ts: number; trajectoryId?: string; tool?: string; summary?: string }>;
};

export type MovementTokenizer = (event: MovementActionEvent) => MovementToken;

/** Default tokenizer: a movement is identified by `<tool>::<summary>`. */
export const defaultMovementTokenizer: MovementTokenizer = (event) => `${event.tool}::${event.summary}`;

/**
 * Build a training dataset from reviewed replay manifests. Each trajectory
 * becomes one sequence of its action movements, ordered by timestamp (ties
 * broken deterministically by tool then summary).
 */
export function buildMovementDataset(
  replays: ReplayLike[],
  tokenize: MovementTokenizer = defaultMovementTokenizer,
): MovementDataset {
  const byTrajectory = new Map<string, { sessionId: string; events: MovementActionEvent[] }>();

  for (const replay of replays) {
    for (const event of replay.events) {
      if (event.kind !== "action" || event.trajectoryId === undefined) {
        continue;
      }
      const action: MovementActionEvent = {
        kind: "action",
        ts: event.ts,
        trajectoryId: event.trajectoryId,
        tool: event.tool ?? "",
        summary: event.summary ?? "",
      };
      const bucket = byTrajectory.get(action.trajectoryId);
      if (bucket) {
        bucket.events.push(action);
      } else {
        byTrajectory.set(action.trajectoryId, { sessionId: replay.sessionId, events: [action] });
      }
    }
  }

  const sequences: MovementSequence[] = [];
  const vocabulary = new Set<MovementToken>();

  for (const [trajectoryId, bucket] of byTrajectory) {
    const ordered = [...bucket.events].sort(compareActionEvents);
    const tokens = ordered.map((event) => {
      const token = tokenize(event);
      vocabulary.add(token);
      return token;
    });
    sequences.push({ trajectoryId, sessionId: bucket.sessionId, tokens });
  }

  // Deterministic ordering so serialized datasets are stable across runs.
  sequences.sort((a, b) => (a.trajectoryId < b.trajectoryId ? -1 : a.trajectoryId > b.trajectoryId ? 1 : 0));

  return {
    version: 1,
    sequences,
    vocabulary: [...vocabulary].sort(),
  };
}

function compareActionEvents(a: MovementActionEvent, b: MovementActionEvent): number {
  if (a.ts !== b.ts) {
    return a.ts - b.ts;
  }
  if (a.tool !== b.tool) {
    return a.tool < b.tool ? -1 : 1;
  }
  return a.summary < b.summary ? -1 : a.summary > b.summary ? 1 : 0;
}

// ---------------------------------------------------------------------------
// Deterministic Markov back-off backend
// ---------------------------------------------------------------------------

export const MARKOV_BACKEND_ID = "markov-backoff" as const;

const BOS = "bos";
const EOS = "eos";
const SEP = "";
const CONTROL_TOKENS = new Set<MovementToken>([BOS, EOS]);

export type MarkovMovementModel = {
  backendId: typeof MARKOV_BACKEND_ID;
  order: number;
  vocabulary: MovementToken[];
  /** context-key (`SEP`-joined suffix) -> { nextToken -> count }. */
  transitions: Record<string, Record<MovementToken, number>>;
};

export class MarkovMovementBackend implements MovementModelBackend<MarkovMovementModel> {
  readonly id = MARKOV_BACKEND_ID;

  train(dataset: MovementDataset, options: MovementTrainOptions = {}): MarkovMovementModel {
    const order = Math.max(1, Math.floor(options.order ?? 3));
    const transitions: Record<string, Record<MovementToken, number>> = {};

    for (const sequence of dataset.sequences) {
      if (sequence.tokens.length === 0) {
        continue;
      }
      const augmented = [BOS, ...sequence.tokens, EOS];
      for (let i = 1; i < augmented.length; i++) {
        const next = augmented[i]!;
        const maxContext = Math.min(order, i);
        for (let c = 0; c <= maxContext; c++) {
          const key = augmented.slice(i - c, i).join(SEP);
          const bucket = (transitions[key] ??= {});
          bucket[next] = (bucket[next] ?? 0) + 1;
        }
      }
    }

    return {
      backendId: MARKOV_BACKEND_ID,
      order,
      vocabulary: [...dataset.vocabulary],
      transitions,
    };
  }

  predictNext(model: MarkovMovementModel, context: MovementToken[]): MovementPrediction {
    const requestedOrder = Math.min(model.order, context.length);
    const core = this.coreLookup(model, context, requestedOrder);
    if (!core) {
      return { token: null, probability: 0, contextOrder: 0, backedOff: requestedOrder > 0, ranked: [] };
    }

    // Public prediction excludes control tokens (BOS can never be next; EOS is
    // an internal end marker surfaced via `generate`, not as a movement).
    const entries = Object.entries(core.counts).filter(([token]) => !CONTROL_TOKENS.has(token));
    const total = entries.reduce((sum, [, count]) => sum + count, 0);
    if (total === 0) {
      return {
        token: null,
        probability: 0,
        contextOrder: core.order,
        backedOff: core.order < requestedOrder,
        ranked: [],
      };
    }

    const ranked = entries
      .map(([token, count]) => ({ token, probability: count / total }))
      .sort(rankPredictions);

    return {
      token: ranked[0]!.token,
      probability: ranked[0]!.probability,
      contextOrder: core.order,
      backedOff: core.order < requestedOrder,
      ranked,
    };
  }

  generate(model: MarkovMovementModel, seed: MovementToken[], steps: number): MovementToken[] {
    const internal: MovementToken[] = [BOS, ...seed];
    const generated: MovementToken[] = [];
    for (let step = 0; step < steps; step++) {
      const requestedOrder = Math.min(model.order, internal.length);
      const core = this.coreLookup(model, internal, requestedOrder);
      if (!core) {
        break;
      }
      const best = pickBest(core.counts);
      if (!best || best === EOS) {
        break;
      }
      generated.push(best);
      internal.push(best);
    }
    return generated;
  }

  serialize(model: MarkovMovementModel): string {
    return JSON.stringify(model);
  }

  deserialize(serialized: string): MarkovMovementModel {
    const parsed = JSON.parse(serialized) as MarkovMovementModel;
    if (parsed.backendId !== MARKOV_BACKEND_ID) {
      throw new Error(`Unexpected movement model backend: ${String(parsed.backendId)}`);
    }
    return parsed;
  }

  /** Longest-suffix-first back-off lookup. Returns the deepest non-empty context. */
  private coreLookup(
    model: MarkovMovementModel,
    context: MovementToken[],
    requestedOrder: number,
  ): { order: number; counts: Record<MovementToken, number> } | undefined {
    for (let c = requestedOrder; c >= 0; c--) {
      const key = context.slice(context.length - c, context.length).join(SEP);
      const counts = model.transitions[key];
      if (counts && Object.keys(counts).length > 0) {
        return { order: c, counts };
      }
    }
    return undefined;
  }
}

function rankPredictions(
  a: { token: MovementToken; probability: number },
  b: { token: MovementToken; probability: number },
): number {
  if (a.probability !== b.probability) {
    return b.probability - a.probability;
  }
  return a.token < b.token ? -1 : a.token > b.token ? 1 : 0;
}

function pickBest(counts: Record<MovementToken, number>): MovementToken | undefined {
  let bestToken: MovementToken | undefined;
  let bestCount = -1;
  for (const [token, count] of Object.entries(counts)) {
    if (token === BOS) {
      continue;
    }
    if (count > bestCount || (count === bestCount && bestToken !== undefined && token < bestToken)) {
      bestToken = token;
      bestCount = count;
    }
  }
  return bestToken;
}

// ---------------------------------------------------------------------------
// Evaluation harness (replay fidelity + generalization)
// ---------------------------------------------------------------------------

export type ReplayFidelityResult = {
  totalPredictions: number;
  correct: number;
  accuracy: number;
  perSequence: Array<{
    trajectoryId: string;
    predictions: number;
    correct: number;
    accuracy: number;
  }>;
};

/**
 * Next-movement accuracy: for every interior position the backend predicts the
 * next token from the preceding context and we score against ground truth.
 * Evaluating on held-out (but related) sequences measures generalization;
 * evaluating on the training sequences measures replay fidelity.
 */
export function evaluateReplayFidelity<TModel extends MovementModel>(
  backend: MovementModelBackend<TModel>,
  model: TModel,
  sequences: MovementSequence[],
): ReplayFidelityResult {
  let totalPredictions = 0;
  let totalCorrect = 0;
  const perSequence: ReplayFidelityResult["perSequence"] = [];

  for (const sequence of sequences) {
    let predictions = 0;
    let correct = 0;
    for (let i = 1; i < sequence.tokens.length; i++) {
      const prediction = backend.predictNext(model, sequence.tokens.slice(0, i));
      predictions += 1;
      if (prediction.token === sequence.tokens[i]) {
        correct += 1;
      }
    }
    totalPredictions += predictions;
    totalCorrect += correct;
    perSequence.push({
      trajectoryId: sequence.trajectoryId,
      predictions,
      correct,
      accuracy: predictions === 0 ? 1 : correct / predictions,
    });
  }

  return {
    totalPredictions,
    correct: totalCorrect,
    accuracy: totalPredictions === 0 ? 1 : totalCorrect / totalPredictions,
    perSequence,
  };
}

// ---------------------------------------------------------------------------
// Pluggable backend registry
// ---------------------------------------------------------------------------

export class MovementModelRegistry {
  private readonly backends = new Map<string, MovementModelBackend>();

  register(backend: MovementModelBackend): this {
    this.backends.set(backend.id, backend);
    return this;
  }

  get(id: string): MovementModelBackend | undefined {
    return this.backends.get(id);
  }

  require(id: string): MovementModelBackend {
    const backend = this.backends.get(id);
    if (!backend) {
      throw new Error(`No movement-model backend registered for id: ${id}`);
    }
    return backend;
  }

  list(): string[] {
    return [...this.backends.keys()].sort();
  }
}

/** Registry seeded with the deterministic in-process Markov backend. */
export function createDefaultMovementModelRegistry(): MovementModelRegistry {
  return new MovementModelRegistry().register(new MarkovMovementBackend());
}
