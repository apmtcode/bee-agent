import type { TrajectoryAction, TrajectorySpan } from "../capture/trajectory.js";

/**
 * Local-movement learning model (standing objective #2, parts c & d).
 *
 * This module turns the recorded/replayable movement dataset into a small,
 * *local* model that can (c) repeat the recorded movements and (d) generalize
 * to related-but-new movements. It is fully in-process and deterministic so it
 * runs and is testable in the cloud with synthetic event streams; the actual
 * on-device recording feeds the same dataset shape when bee-agent runs locally.
 *
 * The model backend is pluggable ({@link MovementModelBackend}). The default
 * {@link NGramMovementBackend} is a variable-order Markov model with Katz-style
 * backoff plus a target-agnostic "abstract" layer, which is what gives it
 * generalization to movements it never saw verbatim. A heavier real backend
 * (e.g. an on-device small transformer) can implement the same interface.
 */

/** A single atomic movement — the token the model learns over. */
export type MovementToken = {
  /** Originating tool/surface, e.g. "device", "browser", "os". */
  tool: string;
  /** The movement verb: gesture kind ("tap", "swipe", "type", …) or tool verb. */
  action: string;
  /** UI target/element the movement acted on, if known. */
  target?: string;
};

/** One ordered movement sequence, typically derived from one trajectory span. */
export type MovementSequence = {
  trajectoryId: string;
  tokens: MovementToken[];
};

/** The replayable training dataset the model learns from. */
export type MovementDataset = {
  version: 1;
  sequences: MovementSequence[];
};

export type MovementTrainOptions = {
  /** Maximum Markov context order (default 3). Clamped to >= 1. */
  order?: number;
};

/** Serializable trained model. Deterministic given the same dataset + order. */
export type MovementModel = {
  version: 1;
  backend: string;
  order: number;
  /** Full-token transitions: context-key -> { nextTokenKey: count }. */
  transitions: Record<string, Record<string, number>>;
  /**
   * Generalization layer: abstract (target-agnostic) context-key -> concrete
   * { nextTokenKey: count }. This lets the model predict a real next movement
   * from a context whose targets it has never seen verbatim — e.g. "menu, then
   * any item -> confirm" — which is what part (d) of the objective asks for.
   */
  abstractTransitions: Record<string, Record<string, number>>;
  /** Overall next-token counts (unigram fallback). */
  unigram: Record<string, number>;
  /** Lookup from a token key back to its structured token. */
  tokenByKey: Record<string, MovementToken>;
  trainedSequences: number;
  trainedTokens: number;
};

/** Which backoff layer produced a prediction (observability + eval breakdown). */
export type MovementPredictionLayer =
  | "exact"
  | "backoff"
  | "abstract"
  | "unigram"
  | "none";

export type MovementCandidate = {
  token: MovementToken;
  probability: number;
};

export type MovementPrediction = {
  token: MovementToken | undefined;
  confidence: number;
  layer: MovementPredictionLayer;
  /** The context order that actually matched (0 when none/unigram). */
  matchedOrder: number;
  candidates: MovementCandidate[];
};

/** Pluggable local-model backend. Async so heavy real backends fit the seam. */
export interface MovementModelBackend {
  readonly id: string;
  train(dataset: MovementDataset, options?: MovementTrainOptions): Promise<MovementModel>;
  predict(model: MovementModel, context: MovementToken[]): MovementPrediction;
}

// --- Token helpers ----------------------------------------------------------

const SEP = ""; // control char: safe field separator, never in real text
const CTX = ""; // context join separator

/** Stable key for a full token (tool|action|target). */
export function movementTokenKey(token: MovementToken): string {
  return `${token.tool}${SEP}${token.action}${SEP}${token.target ?? ""}`;
}

/** Stable key for the target-agnostic ("abstract") view of a token. */
export function abstractTokenKey(token: MovementToken): string {
  return `${token.tool}${SEP}${token.action}`;
}

function contextKey(tokens: MovementToken[], abstract: boolean): string {
  const keyer = abstract ? abstractTokenKey : movementTokenKey;
  return tokens.map(keyer).join(CTX);
}

/** Extract the movement tokens from one trajectory span's action list. */
export function extractMovementTokens(trajectory: TrajectorySpan): MovementToken[] {
  return trajectory.actions.map(actionToToken);
}

function actionToToken(action: TrajectoryAction): MovementToken {
  const metadata = action.metadata ?? {};
  const gesture = typeof metadata.gesture === "string" ? metadata.gesture : undefined;
  const target =
    typeof metadata.target === "string"
      ? metadata.target
      : typeof metadata.direction === "string"
        ? metadata.direction
        : undefined;
  return {
    tool: action.tool,
    action: gesture ?? action.tool,
    ...(target ? { target } : {}),
  };
}

/** Build a movement dataset from a set of trajectory spans. */
export function buildMovementDataset(trajectories: TrajectorySpan[]): MovementDataset {
  return {
    version: 1,
    sequences: trajectories
      .map((trajectory) => ({
        trajectoryId: trajectory.id,
        tokens: extractMovementTokens(trajectory),
      }))
      .filter((sequence) => sequence.tokens.length > 0),
  };
}

// --- Default deterministic backend -----------------------------------------

/**
 * Variable-order Markov model with Katz-style backoff and a target-agnostic
 * abstract layer. Deterministic: ties are broken by descending count then
 * ascending token key, so the same dataset always yields the same predictions.
 */
export class NGramMovementBackend implements MovementModelBackend {
  readonly id = "ngram-markov";

  async train(dataset: MovementDataset, options?: MovementTrainOptions): Promise<MovementModel> {
    const order = Math.max(1, Math.floor(options?.order ?? 3));
    const transitions: Record<string, Record<string, number>> = {};
    const abstractTransitions: Record<string, Record<string, number>> = {};
    const unigram: Record<string, number> = {};
    const tokenByKey: Record<string, MovementToken> = {};
    let trainedTokens = 0;

    for (const sequence of dataset.sequences) {
      const tokens = sequence.tokens;
      for (let i = 0; i < tokens.length; i += 1) {
        const next = tokens[i]!;
        const nextKey = movementTokenKey(next);
        tokenByKey[nextKey] = next;
        bump(unigram, nextKey);
        trainedTokens += 1;

        // Full-token contexts of every order 1..order ending just before i.
        // Abstract-context -> concrete-next of the same orders (generalization).
        for (let k = 1; k <= order && i - k >= 0; k += 1) {
          const context = tokens.slice(i - k, i);
          bump2(transitions, contextKey(context, false), nextKey);
          bump2(abstractTransitions, contextKey(context, true), nextKey);
        }
      }
    }

    return {
      version: 1,
      backend: this.id,
      order,
      transitions,
      abstractTransitions,
      unigram,
      tokenByKey,
      trainedSequences: dataset.sequences.length,
      trainedTokens,
    };
  }

  predict(model: MovementModel, context: MovementToken[]): MovementPrediction {
    // 1. Full-token backoff: longest matching context wins.
    const maxOrder = Math.min(model.order, context.length);
    for (let k = maxOrder; k >= 1; k -= 1) {
      const key = contextKey(context.slice(context.length - k), false);
      const counts = model.transitions[key];
      if (counts) {
        const ranked = rankCounts(counts);
        const candidates = ranked.map(([tokenKey, probability]) => ({
          token: model.tokenByKey[tokenKey] ?? decodeTokenKey(tokenKey),
          probability,
        }));
        const best = candidates[0]!;
        return {
          token: best.token,
          confidence: best.probability,
          // "exact" = matched using the full context available; else a shorter
          // context matched after backing off.
          layer: k === maxOrder ? "exact" : "backoff",
          matchedOrder: k,
          candidates,
        };
      }
    }

    // 2. Abstract-context backoff: match the context by movement *shape*
    //    (targets ignored) but still predict a concrete next movement. This is
    //    what generalizes to related-but-unseen movements — e.g. after "open
    //    menu -> tap <new item>" it still predicts the recurring "confirm".
    for (let k = maxOrder; k >= 1; k -= 1) {
      const key = contextKey(context.slice(context.length - k), true);
      const counts = model.abstractTransitions[key];
      if (counts) {
        const ranked = rankCounts(counts);
        const candidates = ranked.map(([tokenKey, probability]) => ({
          token: model.tokenByKey[tokenKey] ?? decodeTokenKey(tokenKey),
          probability,
        }));
        const best = candidates[0]!;
        return {
          token: best.token,
          confidence: best.probability,
          layer: "abstract",
          matchedOrder: k,
          candidates,
        };
      }
    }

    // 3. Unigram fallback: overall most frequent movement.
    const ranked = rankCounts(model.unigram);
    if (ranked.length > 0) {
      const candidates = ranked.map(([tokenKey, probability]) => ({
        token: model.tokenByKey[tokenKey] ?? decodeTokenKey(tokenKey),
        probability,
      }));
      const best = candidates[0]!;
      return {
        token: best.token,
        confidence: best.probability,
        layer: "unigram",
        matchedOrder: 0,
        candidates,
      };
    }

    return { token: undefined, confidence: 0, layer: "none", matchedOrder: 0, candidates: [] };
  }
}

/** Convenience: train the default backend in one call. */
export async function trainMovementModel(
  dataset: MovementDataset,
  options?: MovementTrainOptions,
): Promise<MovementModel> {
  return new NGramMovementBackend().train(dataset, options);
}

export type MovementGenerateOptions = {
  backend?: MovementModelBackend;
  /**
   * When false (default), generation stops as soon as the model can only
   * continue via the generalization/unigram layers — this yields a *faithful
   * replay* of recorded movements that terminates cleanly. When true, it keeps
   * going through the abstract layer to explore related movements (which may
   * not self-terminate, so bound it with `maxSteps`).
   */
  allowGeneralization?: boolean;
};

/**
 * Autoregressively generate a movement sequence from a seed context — this is
 * how the model "repeats" (and optionally generalizes) recorded movements at
 * replay time. Stops at `maxSteps`, at layer "none", or (by default) at the
 * first non-full-token prediction so faithful replays terminate.
 */
export function generateMovementSequence(
  model: MovementModel,
  seed: MovementToken[],
  maxSteps: number,
  options: MovementGenerateOptions = {},
): MovementToken[] {
  const backend = options.backend ?? new NGramMovementBackend();
  const allowGeneralization = options.allowGeneralization ?? false;
  const generated: MovementToken[] = [];
  const context = [...seed];
  for (let step = 0; step < Math.max(0, maxSteps); step += 1) {
    const prediction = backend.predict(model, context);
    if (!prediction.token || prediction.layer === "none") {
      break;
    }
    const faithful = prediction.layer === "exact" || prediction.layer === "backoff";
    if (!allowGeneralization && !faithful) {
      break;
    }
    generated.push(prediction.token);
    context.push(prediction.token);
  }
  return generated;
}

// --- Generalization eval harness -------------------------------------------

export type MovementEvalResult = {
  /** Positions evaluated (every position after the first in each sequence). */
  predictions: number;
  /** Exact next-token matches (tool+action+target). */
  correct: number;
  accuracy: number;
  /** Softer accuracy: tool+action match, target ignored. */
  actionAccuracy: number;
  /** How many predictions each backoff layer produced. */
  layerCounts: Record<MovementPredictionLayer, number>;
};

/**
 * Teacher-forced next-movement accuracy over held-out sequences — the
 * generalization harness the roadmap asks for. For each held-out sequence we
 * predict every position from its true prefix and compare to the actual next
 * movement, so it measures fidelity on related-but-unseen trajectories.
 */
export function evaluateMovementModel(
  model: MovementModel,
  heldOut: MovementSequence[],
  backend: MovementModelBackend = new NGramMovementBackend(),
): MovementEvalResult {
  let predictions = 0;
  let correct = 0;
  let actionCorrect = 0;
  const layerCounts: Record<MovementPredictionLayer, number> = {
    exact: 0,
    backoff: 0,
    abstract: 0,
    unigram: 0,
    none: 0,
  };

  for (const sequence of heldOut) {
    for (let i = 1; i < sequence.tokens.length; i += 1) {
      const prefix = sequence.tokens.slice(0, i);
      const actual = sequence.tokens[i]!;
      const prediction = backend.predict(model, prefix);
      predictions += 1;
      layerCounts[prediction.layer] += 1;
      if (prediction.token && movementTokenKey(prediction.token) === movementTokenKey(actual)) {
        correct += 1;
      }
      if (prediction.token && abstractTokenKey(prediction.token) === abstractTokenKey(actual)) {
        actionCorrect += 1;
      }
    }
  }

  return {
    predictions,
    correct,
    accuracy: predictions === 0 ? 0 : correct / predictions,
    actionAccuracy: predictions === 0 ? 0 : actionCorrect / predictions,
    layerCounts,
  };
}

/**
 * Deterministic dataset split for train/held-out eval. Every `holdOutEvery`-th
 * sequence (1-indexed) goes to the held-out set — no RNG, so it is stable in
 * tests and reproducible across runs.
 */
export function splitMovementDataset(
  dataset: MovementDataset,
  holdOutEvery = 3,
): { train: MovementDataset; heldOut: MovementSequence[] } {
  const step = Math.max(2, Math.floor(holdOutEvery));
  const trainSequences: MovementSequence[] = [];
  const heldOut: MovementSequence[] = [];
  dataset.sequences.forEach((sequence, index) => {
    if ((index + 1) % step === 0) {
      heldOut.push(sequence);
    } else {
      trainSequences.push(sequence);
    }
  });
  return { train: { version: 1, sequences: trainSequences }, heldOut };
}

// --- internals --------------------------------------------------------------

function bump(map: Record<string, number>, key: string): void {
  map[key] = (map[key] ?? 0) + 1;
}

function bump2(map: Record<string, Record<string, number>>, context: string, next: string): void {
  const inner = (map[context] ??= {});
  inner[next] = (inner[next] ?? 0) + 1;
}

/** Rank next-token counts into [key, probability] pairs, deterministically. */
function rankCounts(counts: Record<string, number>): [string, number][] {
  const entries = Object.entries(counts);
  const total = entries.reduce((sum, [, count]) => sum + count, 0);
  return entries
    .sort((a, b) => (b[1] - a[1]) || (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0))
    .map(([key, count]) => [key, total === 0 ? 0 : count / total]);
}

function decodeTokenKey(key: string): MovementToken {
  const [tool = "", action = "", target = ""] = key.split(SEP);
  return { tool, action, ...(target ? { target } : {}) };
}
