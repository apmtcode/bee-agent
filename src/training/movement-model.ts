/**
 * Movement-model backend: the train -> infer loop for the local-movement
 * learning subsystem (standing objective #2, parts c and d).
 *
 * The existing training runner ({@link ../training/runner.ts}) only emits shell
 * scripts that launch real on-device training (mlx / axolotl). Nothing actually
 * *learns* from a movement dataset or *predicts* new movements — so the
 * capture -> dataset -> replay pipeline could never be validated end-to-end in
 * the cloud, where there is no GPU and no real OS input.
 *
 * This module closes that loop with a pluggable backend interface and a
 * deterministic, dependency-free reference backend (an order-k Markov model with
 * Laplace smoothing and stupid-backoff). It can:
 *   - train on recorded movement sequences,
 *   - reproduce ("repeat") recorded movements via greedy generation, and
 *   - generalize to novel-but-related movements via backoff over shorter
 *     contexts.
 *
 * The backend is pluggable ({@link MovementModelBackend}) so a real on-device
 * small model (mlx/gguf) can implement the same interface later; the Markov
 * backend is the mock that keeps cloud/CI tests green and validates the schema,
 * dataset format, and eval harness with synthetic input.
 */

import type { ReplayManifest } from "../capture/replay.js";
import type { TrajectorySpan } from "../capture/trajectory.js";

/** A single movement primitive, e.g. `act:mouse.move`. */
export type MovementToken = string;

/** Sentinel context token marking the start of a sequence (never predicted). */
export const MOVEMENT_START: MovementToken = "⟨start⟩";
/** Sentinel token marking the end of a sequence (a valid prediction target). */
export const MOVEMENT_END: MovementToken = "⟨end⟩";

/** Separator used to key multi-token contexts; unlikely to appear in a token. */
const CONTEXT_SEP = "␟";

/** A representative action for a token, so a prediction can map back to a replayable action. */
export type MovementExemplar = {
  token: MovementToken;
  tool: string;
  summary: string;
  count: number;
};

export type MovementSequence = {
  id: string;
  tokens: MovementToken[];
};

export type MovementDataset = {
  version: 1;
  vocab: MovementToken[];
  sequences: MovementSequence[];
  exemplars: Record<MovementToken, MovementExemplar>;
};

export type MovementTrainingConfig = {
  /** Context length (n-gram order). Must be >= 1. */
  order: number;
  /** Laplace/additive smoothing alpha (>= 0). */
  smoothing: number;
};

export const DEFAULT_MOVEMENT_TRAINING_CONFIG: MovementTrainingConfig = {
  order: 2,
  smoothing: 0.1,
};

export type MovementModel = {
  backend: string;
  order: number;
  smoothing: number;
  vocab: MovementToken[];
  /** context-key -> nextToken -> count */
  transitions: Record<string, Record<MovementToken, number>>;
  /** context-key -> total observed transitions from that context */
  contextTotals: Record<string, number>;
  exemplars: Record<MovementToken, MovementExemplar>;
  sequenceCount: number;
  observedTransitions: number;
};

export type MovementPredictionCandidate = {
  token: MovementToken;
  probability: number;
  /** Length of the context that produced this prediction (backoff level). */
  contextOrder: number;
};

export type MovementPrediction = {
  token: MovementToken;
  candidates: MovementPredictionCandidate[];
  /** True when the model fell back to a uniform guess (no context matched). */
  fallback: boolean;
};

/**
 * Pluggable movement-model backend. A real on-device model would implement
 * this same shape; {@link MarkovMovementBackend} is the deterministic mock.
 */
export interface MovementModelBackend {
  readonly id: string;
  train(dataset: MovementDataset, config?: MovementTrainingConfig): MovementModel;
}

// ---------------------------------------------------------------------------
// Tokenization
// ---------------------------------------------------------------------------

function normalizeToolName(tool: string): string {
  return tool.trim().toLowerCase().replace(/\s+/g, "_");
}

/** Map an action-like record to its movement token. */
export function tokenizeMovement(action: { tool: string }): MovementToken {
  return `act:${normalizeToolName(action.tool)}`;
}

function recordExemplar(
  exemplars: Record<MovementToken, MovementExemplar>,
  token: MovementToken,
  tool: string,
  summary: string,
): void {
  const existing = exemplars[token];
  if (existing) {
    existing.count += 1;
    return;
  }
  exemplars[token] = { token, tool, summary, count: 1 };
}

function finalizeDataset(sequences: MovementSequence[], exemplars: Record<MovementToken, MovementExemplar>): MovementDataset {
  const vocab = new Set<MovementToken>();
  for (const sequence of sequences) {
    for (const token of sequence.tokens) {
      vocab.add(token);
    }
  }
  return {
    version: 1,
    vocab: [...vocab].sort(),
    sequences,
    exemplars,
  };
}

/** Build a movement dataset from trajectory spans (one sequence per trajectory). */
export function buildMovementDatasetFromTrajectories(trajectories: TrajectorySpan[]): MovementDataset {
  const exemplars: Record<MovementToken, MovementExemplar> = {};
  const sequences: MovementSequence[] = trajectories.map((trajectory) => {
    const ordered = [...trajectory.actions].sort((a, b) => a.ts - b.ts);
    const tokens = ordered.map((action) => {
      const token = tokenizeMovement(action);
      recordExemplar(exemplars, token, action.tool, action.summary);
      return token;
    });
    return { id: trajectory.id, tokens };
  });
  return finalizeDataset(sequences, exemplars);
}

/** Build a movement dataset from replay manifests (action events only, in timeline order). */
export function buildMovementDatasetFromReplays(replays: ReplayManifest[]): MovementDataset {
  const exemplars: Record<MovementToken, MovementExemplar> = {};
  const sequences: MovementSequence[] = replays.map((replay, index) => {
    const tokens = replay.events
      .filter((event): event is Extract<ReplayManifest["events"][number], { kind: "action" }> => event.kind === "action")
      .map((event) => {
        const token = tokenizeMovement(event);
        recordExemplar(exemplars, token, event.tool, event.summary);
        return token;
      });
    return { id: `${replay.sessionId}:${index}`, tokens };
  });
  return finalizeDataset(sequences, exemplars);
}

// ---------------------------------------------------------------------------
// Markov backend (deterministic reference / mock)
// ---------------------------------------------------------------------------

function contextKey(context: MovementToken[]): string {
  return context.join(CONTEXT_SEP);
}

export class MarkovMovementBackend implements MovementModelBackend {
  readonly id = "markov";

  train(dataset: MovementDataset, config: MovementTrainingConfig = DEFAULT_MOVEMENT_TRAINING_CONFIG): MovementModel {
    const order = Math.max(1, Math.floor(config.order));
    const smoothing = Math.max(0, config.smoothing);
    const transitions: Record<string, Record<MovementToken, number>> = {};
    const contextTotals: Record<string, number> = {};
    let observedTransitions = 0;

    const bump = (context: MovementToken[], next: MovementToken): void => {
      const key = contextKey(context);
      const row = (transitions[key] ??= {});
      row[next] = (row[next] ?? 0) + 1;
      contextTotals[key] = (contextTotals[key] ?? 0) + 1;
      observedTransitions += 1;
    };

    for (const sequence of dataset.sequences) {
      // Pad the front with `order` START tokens so first-move-from-start is learnable,
      // and append END so sequence termination is a predictable event.
      const padded = [
        ...Array.from({ length: order }, () => MOVEMENT_START),
        ...sequence.tokens,
        MOVEMENT_END,
      ];
      for (let i = order; i < padded.length; i += 1) {
        const next = padded[i]!;
        // Count this transition at every context length 0..order for backoff.
        for (let k = 0; k <= order; k += 1) {
          bump(padded.slice(i - k, i), next);
        }
      }
    }

    return {
      backend: this.id,
      order,
      smoothing,
      vocab: [...dataset.vocab],
      transitions,
      contextTotals,
      exemplars: { ...dataset.exemplars },
      sequenceCount: dataset.sequences.length,
      observedTransitions,
    };
  }
}

/** Candidate vocabulary for prediction: real vocab + END, never START. */
function predictionVocab(model: MovementModel): MovementToken[] {
  const vocab = new Set<MovementToken>(model.vocab);
  vocab.add(MOVEMENT_END);
  vocab.delete(MOVEMENT_START);
  return [...vocab];
}

function rankRow(
  row: Record<MovementToken, number>,
  total: number,
  smoothing: number,
  vocab: MovementToken[],
  contextOrder: number,
): MovementPredictionCandidate[] {
  const denom = total + smoothing * vocab.length;
  const candidates: MovementPredictionCandidate[] = vocab.map((token) => ({
    token,
    probability: denom > 0 ? ((row[token] ?? 0) + smoothing) / denom : 0,
    contextOrder,
  }));
  // Deterministic ordering: probability desc, then token asc for stable tie-breaks.
  candidates.sort((a, b) => (b.probability - a.probability) || (a.token < b.token ? -1 : a.token > b.token ? 1 : 0));
  return candidates;
}

/**
 * Predict the next movement given a context, using stupid-backoff from the
 * model's full order down to the unigram distribution, and a uniform fallback
 * if nothing matched (e.g. an empty model).
 */
export function predictNextMovement(model: MovementModel, context: MovementToken[]): MovementPrediction {
  const vocab = predictionVocab(model);
  const maxK = Math.min(model.order, context.length);
  for (let k = maxK; k >= 0; k -= 1) {
    const ctx = context.slice(context.length - k);
    const key = contextKey(ctx);
    const row = model.transitions[key];
    const total = model.contextTotals[key] ?? 0;
    if (row && total > 0) {
      const candidates = rankRow(row, total, model.smoothing, vocab, k);
      return { token: candidates[0]?.token ?? MOVEMENT_END, candidates, fallback: false };
    }
  }
  const uniform = vocab.length > 0 ? 1 / vocab.length : 0;
  const candidates = [...vocab]
    .sort()
    .map<MovementPredictionCandidate>((token) => ({ token, probability: uniform, contextOrder: 0 }));
  return { token: candidates[0]?.token ?? MOVEMENT_END, candidates, fallback: true };
}

export type GenerateMovementOptions = {
  /** Seed context (real tokens). Defaults to an all-START context. */
  seed?: MovementToken[];
  /** Hard cap on generated tokens (excluding the terminating END). */
  maxLength?: number;
};

/**
 * Greedily roll out a movement sequence. With a seen prefix this reproduces
 * recorded movements; from novel contexts it generalizes via backoff. Fully
 * deterministic (argmax, stable tie-breaks) so it is safe in cloud/CI.
 */
export function generateMovementSequence(model: MovementModel, options: GenerateMovementOptions = {}): MovementToken[] {
  const maxLength = options.maxLength ?? 64;
  const startPad = Array.from({ length: model.order }, () => MOVEMENT_START);
  const context: MovementToken[] = [...startPad, ...(options.seed ?? [])];
  const produced: MovementToken[] = [];
  while (produced.length < maxLength) {
    const prediction = predictNextMovement(model, context);
    if (prediction.token === MOVEMENT_END) {
      break;
    }
    if (prediction.token === MOVEMENT_START) {
      break; // defensive; START is filtered from candidates but never emit it
    }
    produced.push(prediction.token);
    context.push(prediction.token);
  }
  return produced;
}

/** Map generated tokens back to representative (replayable) actions via exemplars. */
export function detokenizeMovements(
  model: MovementModel,
  tokens: MovementToken[],
): Array<{ token: MovementToken; tool: string; summary: string }> {
  return tokens.map((token) => {
    const exemplar = model.exemplars[token];
    return {
      token,
      tool: exemplar?.tool ?? token,
      summary: exemplar?.summary ?? "",
    };
  });
}

// ---------------------------------------------------------------------------
// Generalization eval harness
// ---------------------------------------------------------------------------

export type MovementEvalReport = {
  sequences: number;
  predictions: number;
  correct: number;
  /** Top-1 next-token accuracy under teacher forcing. */
  accuracy: number;
  /** Mean per-token log2 loss (>= 0); lower is better. */
  averageLogLoss: number;
  /** 2 ** averageLogLoss; a smoothing-independent difficulty summary. */
  perplexity: number;
};

/**
 * Measure how well the model predicts held-out (but related) movement
 * sequences. Uses teacher forcing: at each step the true preceding tokens are
 * the context, and we score whether the model's top-1 prediction matches the
 * true next token, plus the log-loss of the true token's probability.
 */
export function evaluateMovementModel(model: MovementModel, heldOut: MovementSequence[]): MovementEvalReport {
  let predictions = 0;
  let correct = 0;
  let logLossSum = 0;
  const startPad = Array.from({ length: model.order }, () => MOVEMENT_START);

  for (const sequence of heldOut) {
    const context: MovementToken[] = [...startPad];
    const targets = [...sequence.tokens, MOVEMENT_END];
    for (const target of targets) {
      const prediction = predictNextMovement(model, context);
      predictions += 1;
      if (prediction.token === target) {
        correct += 1;
      }
      const targetProb = prediction.candidates.find((candidate) => candidate.token === target)?.probability ?? 0;
      const safeProb = targetProb > 0 ? targetProb : 1e-9;
      logLossSum += -Math.log2(safeProb);
      if (target !== MOVEMENT_END) {
        context.push(target);
      }
    }
  }

  const averageLogLoss = predictions > 0 ? logLossSum / predictions : 0;
  return {
    sequences: heldOut.length,
    predictions,
    correct,
    accuracy: predictions > 0 ? correct / predictions : 0,
    averageLogLoss,
    perplexity: 2 ** averageLogLoss,
  };
}

// ---------------------------------------------------------------------------
// Synthetic event-stream generator (deterministic; no Math.random / Date)
// ---------------------------------------------------------------------------

/** Deterministic LCG so synthetic data is reproducible across runs and CI. */
function lcg(seed: number): () => number {
  let state = (seed >>> 0) || 1;
  return () => {
    // Numerical Recipes constants.
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 0x100000000;
  };
}

export type SyntheticMovementGrammar = Record<MovementToken, MovementToken[]>;

/**
 * A small default "movement grammar": a directed graph of plausible next
 * movements. It has real structure (open before focus, focus before type,
 * save/close as sinks) so a trained model can both reproduce and generalize.
 */
export const DEFAULT_MOVEMENT_GRAMMAR: SyntheticMovementGrammar = {
  [MOVEMENT_START]: ["act:window.open", "act:app.focus"],
  "act:window.open": ["act:app.focus", "act:mouse.move"],
  "act:app.focus": ["act:mouse.move", "act:key.type"],
  "act:mouse.move": ["act:mouse.click", "act:mouse.move"],
  "act:mouse.click": ["act:key.type", "act:menu.open"],
  "act:menu.open": ["act:menu.select"],
  "act:menu.select": ["act:file.save", "act:mouse.move"],
  "act:key.type": ["act:key.type", "act:file.save"],
  "act:file.save": ["act:window.close", MOVEMENT_END],
  "act:window.close": [MOVEMENT_END],
};

export type SyntheticMovementDatasetParams = {
  sequenceCount: number;
  seed: number;
  grammar?: SyntheticMovementGrammar;
  maxSequenceLength?: number;
  idPrefix?: string;
};

/**
 * Generate a deterministic synthetic movement dataset by random-walking a
 * grammar. Used to validate the capture -> dataset -> train -> infer pipeline
 * without any real OS input, and to produce held-out sequences for eval.
 */
export function createSyntheticMovementDataset(params: SyntheticMovementDatasetParams): MovementDataset {
  const grammar = params.grammar ?? DEFAULT_MOVEMENT_GRAMMAR;
  const maxLength = params.maxSequenceLength ?? 32;
  const prefix = params.idPrefix ?? "synthetic";
  const next = lcg(params.seed);
  const exemplars: Record<MovementToken, MovementExemplar> = {};
  const sequences: MovementSequence[] = [];

  for (let s = 0; s < params.sequenceCount; s += 1) {
    const tokens: MovementToken[] = [];
    let current: MovementToken = MOVEMENT_START;
    for (let step = 0; step < maxLength; step += 1) {
      const options = grammar[current] ?? [MOVEMENT_END];
      const pick = options[Math.floor(next() * options.length)] ?? MOVEMENT_END;
      if (pick === MOVEMENT_END) {
        break;
      }
      tokens.push(pick);
      const tool = pick.startsWith("act:") ? pick.slice(4) : pick;
      recordExemplar(exemplars, pick, tool, `synthetic ${tool}`);
      current = pick;
    }
    sequences.push({ id: `${prefix}-${s}`, tokens });
  }

  return finalizeDataset(sequences, exemplars);
}
