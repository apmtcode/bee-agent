/**
 * Movement model subsystem — the on-device "learn to repeat and generalize
 * recorded movements" pipeline (standing objective #2, parts c & d).
 *
 * This module is deliberately backend-pluggable. The interface
 * ({@link MovementModelBackend}) is what a real on-device small model would
 * implement; {@link MarkovMovementBackend} is a fully deterministic,
 * dependency-free reference backend that trains and infers in-process so the
 * whole capture → dataset → train → infer → generalize loop can be validated in
 * the cloud/CI without any OS input capture or GPU training.
 *
 * The Markov backend is not a toy: an order-k n-gram with stupid-backoff plus a
 * class-level backoff table gives it two distinct capabilities the objective
 * asks for — (c) *repeat* recorded movements exactly (high-order exact match)
 * and (d) *generalize* to new-but-related movements (shorter-context and
 * class-context backoff let a novel prefix that shares structure with training
 * data still predict a sensible next movement).
 */

import type { TrajectoryAction, TrajectoryObservation, TrajectorySpan } from "../capture/trajectory.js";
import type { ReplayTimelineEvent } from "../capture/replay.js";

/** Sentinel appended to the end of every training sequence. */
export const MOVEMENT_END_SYMBOL = "<end>";

/** Sentinel prepended to every sequence so the first movement is learnable. */
export const MOVEMENT_START_SYMBOL = "<start>";

export type MovementTokenKind = "observation" | "action";

/**
 * A single discrete movement symbol. `symbol` is the fine-grained vocabulary
 * entry (includes the concrete target); `klass` is the coarser category used
 * for generalization backoff (drops the concrete target, keeping the
 * tool/gesture/direction shape).
 */
export type MovementToken = {
  kind: MovementTokenKind;
  symbol: string;
  klass: string;
  label?: string;
};

export type MovementSequence = {
  trajectoryId: string;
  tokens: MovementToken[];
};

export type MovementDataset = {
  version: 1;
  sequences: MovementSequence[];
  /** Sorted unique fine-grained symbols observed across all sequences. */
  vocabulary: string[];
};

export type MovementModelArtifact = {
  backendId: string;
  version: 1;
  order: number;
  vocabulary: string[];
  /** Backend-private parameter blob (JSON-serialisable). */
  parameters: unknown;
  trainedAt?: string;
  stats: { sequenceCount: number; tokenCount: number };
};

export type MovementContext = {
  /** Recent tokens leading up to the prediction point, oldest first. */
  history: MovementToken[];
};

export type MovementPredictionCandidate = {
  symbol: string;
  probability: number;
};

export type MovementPrediction = {
  symbol: string;
  klass: string;
  kind: MovementTokenKind | "end";
  probability: number;
  candidates: MovementPredictionCandidate[];
  /** How many context tokens matched (0 == unigram / fully generalised). */
  matchedOrder: number;
  /** Whether the prediction required backing off below the model order. */
  backedOff: boolean;
  /** Whether the match came from the class-level (generalisation) table. */
  generalized: boolean;
};

export type MovementTrainOptions = {
  /** Maximum n-gram order (context length). Default 3. */
  order?: number;
  /** Injectable clock for `trainedAt` so tests stay hermetic. */
  now?: () => string;
};

export interface MovementModelBackend {
  readonly id: string;
  train(dataset: MovementDataset, options?: MovementTrainOptions): Promise<MovementModelArtifact>;
  predict(model: MovementModelArtifact, context: MovementContext): MovementPrediction;
}

// ---------------------------------------------------------------------------
// Tokenisation
// ---------------------------------------------------------------------------

function actionToken(action: TrajectoryAction): MovementToken {
  const gesture = readMetaString(action.metadata, "gesture");
  const direction = readMetaString(action.metadata, "direction");
  const target = readMetaString(action.metadata, "target");
  const shape = ["act", action.tool];
  if (gesture) {
    shape.push(gesture);
  }
  if (direction) {
    shape.push(direction);
  }
  const klass = shape.join(":");
  const symbol = target ? `${klass}#${target}` : klass;
  return { kind: "action", symbol, klass, label: action.summary };
}

function observationToken(observation: TrajectoryObservation): MovementToken {
  const symbol = `obs:${observation.source}`;
  return { kind: "observation", symbol, klass: symbol, label: observation.summary };
}

function readMetaString(metadata: Record<string, unknown> | undefined, key: string): string | undefined {
  const value = metadata?.[key];
  return typeof value === "string" && value.length > 0 ? value : undefined;
}

/**
 * Reconstruct a minimal token from a fine-grained symbol (used when feeding a
 * prediction back into the context during a rollout). Class is the symbol with
 * any concrete-target suffix removed.
 */
export function symbolToToken(symbol: string): MovementToken {
  if (symbol === MOVEMENT_END_SYMBOL || symbol === MOVEMENT_START_SYMBOL) {
    return { kind: "action", symbol, klass: symbol };
  }
  const klass = symbol.split("#")[0] ?? symbol;
  const kind: MovementTokenKind = symbol.startsWith("obs:") ? "observation" : "action";
  return { kind, symbol, klass };
}

/** Merge a trajectory's observations and actions into one ts-ordered sequence. */
export function tokenizeTrajectory(trajectory: TrajectorySpan): MovementSequence {
  const entries: Array<{ ts: number; order: number; token: MovementToken }> = [
    ...trajectory.observations.map((observation) => ({
      ts: observation.ts,
      order: 0,
      token: observationToken(observation),
    })),
    ...trajectory.actions.map((action) => ({
      ts: action.ts,
      order: 1,
      token: actionToken(action),
    })),
  ];
  entries.sort((a, b) => (a.ts !== b.ts ? a.ts - b.ts : a.order - b.order));
  return { trajectoryId: trajectory.id, tokens: entries.map((entry) => entry.token) };
}

/** Tokenise a replay timeline (the shape stored in reviewed export manifests). */
export function tokenizeReplayEvents(trajectoryId: string, events: ReplayTimelineEvent[]): MovementSequence {
  const tokens: MovementToken[] = [];
  for (const event of events) {
    if (event.kind === "observation") {
      tokens.push({ kind: "observation", symbol: `obs:${event.source}`, klass: `obs:${event.source}`, label: event.summary });
    } else if (event.kind === "action") {
      const klass = `act:${event.tool}`;
      tokens.push({ kind: "action", symbol: klass, klass, label: event.summary });
    }
    // transcript events carry no movement signal — skipped.
  }
  return { trajectoryId, tokens };
}

export function buildMovementDataset(sequences: MovementSequence[]): MovementDataset {
  const vocabulary = new Set<string>();
  for (const sequence of sequences) {
    for (const token of sequence.tokens) {
      vocabulary.add(token.symbol);
    }
  }
  return {
    version: 1,
    sequences: sequences.filter((sequence) => sequence.tokens.length > 0),
    vocabulary: [...vocabulary].sort(),
  };
}

export function buildDatasetFromTrajectories(trajectories: TrajectorySpan[]): MovementDataset {
  return buildMovementDataset(trajectories.map(tokenizeTrajectory));
}

// ---------------------------------------------------------------------------
// Markov reference backend
// ---------------------------------------------------------------------------

const CONTEXT_SEPARATOR = "";

type CountTable = Record<string, Record<string, number>>;

type MarkovParameters = {
  order: number;
  /** symbol-context (n=1..order) -> { nextSymbol -> count } */
  symbolTransitions: CountTable;
  /** class-context (n=1..order) -> { nextSymbol -> count } — the generalisation table */
  classTransitions: CountTable;
  /** nextSymbol -> count (fallback prior) */
  unigram: Record<string, number>;
};

function increment(table: CountTable, contextKey: string, symbol: string): void {
  const bucket = (table[contextKey] ??= {});
  bucket[symbol] = (bucket[symbol] ?? 0) + 1;
}

function contextKey(values: string[]): string {
  return values.join(CONTEXT_SEPARATOR);
}

/**
 * Deterministic argmax: highest count, ties broken by lexicographically
 * smallest symbol so the model is reproducible run-to-run.
 */
function argmax(bucket: Record<string, number>): { symbol: string; count: number; total: number } | undefined {
  let bestSymbol: string | undefined;
  let bestCount = -1;
  let total = 0;
  for (const [symbol, count] of Object.entries(bucket)) {
    total += count;
    if (count > bestCount || (count === bestCount && (bestSymbol === undefined || symbol < bestSymbol))) {
      bestSymbol = symbol;
      bestCount = count;
    }
  }
  if (bestSymbol === undefined) {
    return undefined;
  }
  return { symbol: bestSymbol, count: bestCount, total };
}

function rankCandidates(bucket: Record<string, number>, total: number, limit = 5): MovementPredictionCandidate[] {
  return Object.entries(bucket)
    .map(([symbol, count]) => ({ symbol, probability: total > 0 ? count / total : 0 }))
    .sort((a, b) => (b.probability !== a.probability ? b.probability - a.probability : a.symbol.localeCompare(b.symbol)))
    .slice(0, limit);
}

export class MarkovMovementBackend implements MovementModelBackend {
  readonly id = "markov-ngram";

  async train(dataset: MovementDataset, options: MovementTrainOptions = {}): Promise<MovementModelArtifact> {
    const order = Math.max(1, Math.floor(options.order ?? 3));
    const symbolTransitions: CountTable = {};
    const classTransitions: CountTable = {};
    const unigram: Record<string, number> = {};
    let tokenCount = 0;

    for (const sequence of dataset.sequences) {
      const symbols = [MOVEMENT_START_SYMBOL, ...sequence.tokens.map((token) => token.symbol), MOVEMENT_END_SYMBOL];
      const classes = [MOVEMENT_START_SYMBOL, ...sequence.tokens.map((token) => token.klass), MOVEMENT_END_SYMBOL];
      tokenCount += sequence.tokens.length;

      // index starts at 1: position 0 is the START sentinel, never predicted.
      for (let index = 1; index < symbols.length; index += 1) {
        const nextSymbol = symbols[index]!;
        unigram[nextSymbol] = (unigram[nextSymbol] ?? 0) + 1;
        for (let n = 1; n <= order; n += 1) {
          if (index - n < 0) {
            break;
          }
          const symbolCtx = symbols.slice(index - n, index);
          const classCtx = classes.slice(index - n, index);
          increment(symbolTransitions, contextKey(symbolCtx), nextSymbol);
          increment(classTransitions, contextKey(classCtx), nextSymbol);
        }
      }
    }

    const parameters: MarkovParameters = { order, symbolTransitions, classTransitions, unigram };
    return {
      backendId: this.id,
      version: 1,
      order,
      vocabulary: [...dataset.vocabulary],
      parameters,
      ...(options.now ? { trainedAt: options.now() } : {}),
      stats: { sequenceCount: dataset.sequences.length, tokenCount },
    };
  }

  predict(model: MovementModelArtifact, context: MovementContext): MovementPrediction {
    const params = model.parameters as MarkovParameters;
    const order = params.order;
    // Prepend the START sentinel so an empty (or short) history can match the
    // learned sequence-start context and generation-from-scratch works.
    const historySymbols = [MOVEMENT_START_SYMBOL, ...context.history.map((token) => token.symbol)];
    const historyClasses = [MOVEMENT_START_SYMBOL, ...context.history.map((token) => token.klass)];

    // 1. Exact symbol-context backoff (repeat recorded movements).
    for (let n = Math.min(order, historySymbols.length); n >= 1; n -= 1) {
      const bucket = params.symbolTransitions[contextKey(historySymbols.slice(-n))];
      const best = bucket && argmax(bucket);
      if (best) {
        return finalize(best, bucket, n, n < order, false);
      }
    }

    // 2. Class-context backoff (generalise to new-but-related movements).
    for (let n = Math.min(order, historyClasses.length); n >= 1; n -= 1) {
      const bucket = params.classTransitions[contextKey(historyClasses.slice(-n))];
      const best = bucket && argmax(bucket);
      if (best) {
        return finalize(best, bucket, n, true, true);
      }
    }

    // 3. Unigram prior.
    const best = argmax(params.unigram);
    if (best) {
      return finalize(best, params.unigram, 0, true, false);
    }

    return {
      symbol: MOVEMENT_END_SYMBOL,
      klass: MOVEMENT_END_SYMBOL,
      kind: "end",
      probability: 1,
      candidates: [{ symbol: MOVEMENT_END_SYMBOL, probability: 1 }],
      matchedOrder: 0,
      backedOff: true,
      generalized: false,
    };
  }
}

function finalize(
  best: { symbol: string; count: number; total: number },
  bucket: Record<string, number>,
  matchedOrder: number,
  backedOff: boolean,
  generalized: boolean,
): MovementPrediction {
  const token = symbolToToken(best.symbol);
  return {
    symbol: best.symbol,
    klass: token.klass,
    kind: best.symbol === MOVEMENT_END_SYMBOL ? "end" : token.kind,
    probability: best.total > 0 ? best.count / best.total : 0,
    candidates: rankCandidates(bucket, best.total),
    matchedOrder,
    backedOff,
    generalized,
  };
}

// ---------------------------------------------------------------------------
// Rollout (multi-step generation) and evaluation
// ---------------------------------------------------------------------------

export type MovementRolloutStep = {
  token: MovementToken;
  prediction: MovementPrediction;
};

export type MovementRollout = {
  tokens: MovementToken[];
  steps: MovementRolloutStep[];
  stopped: "end" | "max-steps";
};

/** Greedily roll the model forward from a seed context until END or maxSteps. */
export function generateMovements(
  backend: MovementModelBackend,
  model: MovementModelArtifact,
  seed: MovementToken[],
  maxSteps = 32,
): MovementRollout {
  const history = [...seed];
  const steps: MovementRolloutStep[] = [];
  const generated: MovementToken[] = [];
  for (let step = 0; step < maxSteps; step += 1) {
    const prediction = backend.predict(model, { history });
    if (prediction.symbol === MOVEMENT_END_SYMBOL) {
      return { tokens: generated, steps, stopped: "end" };
    }
    const token = symbolToToken(prediction.symbol);
    steps.push({ token, prediction });
    generated.push(token);
    history.push(token);
  }
  return { tokens: generated, steps, stopped: "max-steps" };
}

export type MovementEvalStep = {
  expected: string;
  predicted: string;
  hit: boolean;
  matchedOrder: number;
  generalized: boolean;
};

export type MovementEvalResult = {
  trajectoryId: string;
  steps: number;
  correct: number;
  accuracy: number;
  /** Fraction of correct predictions that required class-level generalisation. */
  generalizedShare: number;
  perStep: MovementEvalStep[];
};

/**
 * Teacher-forced next-token fidelity: walk the true sequence and, at each
 * position, predict from the true prefix and compare to the true continuation
 * (including the terminal END). This is the replay-fidelity metric used by the
 * generalisation eval harness.
 */
export function evaluateSequence(
  backend: MovementModelBackend,
  model: MovementModelArtifact,
  sequence: MovementSequence,
): MovementEvalResult {
  const symbols = [...sequence.tokens.map((token) => token.symbol), MOVEMENT_END_SYMBOL];
  const perStep: MovementEvalStep[] = [];
  let correct = 0;
  let generalizedHits = 0;
  for (let index = 0; index < symbols.length; index += 1) {
    const expected = symbols[index]!;
    const prediction = backend.predict(model, { history: sequence.tokens.slice(0, index) });
    const hit = prediction.symbol === expected;
    if (hit) {
      correct += 1;
      if (prediction.generalized) {
        generalizedHits += 1;
      }
    }
    perStep.push({
      expected,
      predicted: prediction.symbol,
      hit,
      matchedOrder: prediction.matchedOrder,
      generalized: prediction.generalized,
    });
  }
  const steps = symbols.length;
  return {
    trajectoryId: sequence.trajectoryId,
    steps,
    correct,
    accuracy: steps > 0 ? correct / steps : 0,
    generalizedShare: correct > 0 ? generalizedHits / correct : 0,
    perStep,
  };
}

export type MovementDatasetEval = {
  sequenceCount: number;
  meanAccuracy: number;
  results: MovementEvalResult[];
};

export function evaluateDataset(
  backend: MovementModelBackend,
  model: MovementModelArtifact,
  sequences: MovementSequence[],
): MovementDatasetEval {
  const results = sequences.map((sequence) => evaluateSequence(backend, model, sequence));
  const meanAccuracy = results.length > 0 ? results.reduce((sum, result) => sum + result.accuracy, 0) / results.length : 0;
  return { sequenceCount: results.length, meanAccuracy, results };
}

// ---------------------------------------------------------------------------
// Synthetic event-stream generator (validates the loop without real OS input)
// ---------------------------------------------------------------------------

/** Deterministic 32-bit LCG so synthetic streams are reproducible from a seed. */
function makeLcg(seed: number): () => number {
  let state = (seed >>> 0) || 1;
  return () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 0x100000000;
  };
}

export type SyntheticMovementOptions = {
  id: string;
  sessionId?: string;
  seed: number;
  /** Pool of concrete targets the workflow can act on; varies per trajectory. */
  targets?: string[];
  app?: string;
  steps?: number;
};

const DEFAULT_TARGETS = ["search", "inbox", "compose", "settings", "profile", "results", "confirm", "cancel"];
const GESTURES: Array<{ kind: string; direction?: "up" | "down" | "left" | "right" }> = [
  { kind: "tap" },
  { kind: "swipe", direction: "up" },
  { kind: "swipe", direction: "down" },
  { kind: "scroll", direction: "down" },
  { kind: "type" },
];

/**
 * Produce a synthetic device trajectory following a fixed "workflow grammar"
 * (activate app → repeated observe/act gesture pairs), with per-seed variation
 * in the concrete targets. Trajectories sharing a seed family have the same
 * gesture *shape* but different targets — exactly the setup needed to test
 * generalisation (train on some targets, hold out related ones).
 */
export function synthesizeMovementTrajectory(options: SyntheticMovementOptions): TrajectorySpan {
  const rng = makeLcg(options.seed);
  const app = options.app ?? "workflow-app";
  const targets = options.targets ?? DEFAULT_TARGETS;
  const steps = Math.max(1, options.steps ?? 4);
  const observations: TrajectoryObservation[] = [];
  const actions: TrajectoryAction[] = [];
  let ts = 1_000;

  observations.push({
    kind: "observation",
    source: "device",
    summary: `${app} active on device`,
    ts,
    metadata: { platform: "macos", appName: app },
  });

  for (let step = 0; step < steps; step += 1) {
    ts += 100;
    const target = targets[Math.floor(rng() * targets.length)] ?? targets[0]!;
    observations.push({
      kind: "observation",
      source: "device",
      summary: `${app} showing ${target}`,
      ts,
      metadata: { platform: "macos", appName: app, screenTitle: target },
    });
    ts += 50;
    const gesture = GESTURES[Math.floor(rng() * GESTURES.length)] ?? GESTURES[0]!;
    actions.push({
      kind: "action",
      tool: "device",
      summary: `${gesture.kind} ${target}`,
      ts,
      metadata: {
        gesture: gesture.kind,
        target,
        ...(gesture.direction ? { direction: gesture.direction } : {}),
      },
    });
  }

  return {
    id: options.id,
    sessionId: options.sessionId ?? `synthetic-${options.id}`,
    createdAt: "1970-01-01T00:00:00.000Z",
    captureTier: "app",
    observations,
    actions,
    outcome: { status: "success", summary: `completed ${steps}-step ${app} workflow`, reward: 1 },
  };
}
