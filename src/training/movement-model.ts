import type { ReplayTimelineEvent } from "../capture/replay.js";

/**
 * Pluggable local-movement model surface.
 *
 * This module defines the backend-agnostic contract the movement-learning
 * subsystem trains and infers against, plus the pure helpers that turn recorded
 * replay timelines into a training dataset, generate movement sequences from a
 * trained model, and measure replay fidelity on held-out trajectories.
 *
 * The concrete learning algorithm lives behind {@link MovementTrainingBackend}
 * so it can be swapped: the in-repo {@link MockMarkovMovementBackend} is a
 * deterministic, dependency-free backend that runs in the cloud/CI, while a real
 * on-device small model can implement the same interface later without changing
 * any call site.
 */

/**
 * Canonical movement token for a recorded timeline event. Tokens abstract away
 * free-text detail (which varies run to run) into the stable "shape" of a
 * movement — the unit the model actually learns transitions over.
 */
export function tokenizeMovementEvent(event: ReplayTimelineEvent): string {
  switch (event.kind) {
    case "transcript":
      return `transcript:${event.role}`;
    case "observation":
      return `observation:${event.source}`;
    case "action":
      return `action:${event.tool}`;
  }
}

export type MovementSequence = {
  /** Stable id of the source trajectory/session this sequence came from. */
  id: string;
  tokens: string[];
};

export type MovementDataset = {
  version: 1;
  /** Sorted, de-duplicated set of every token that appears in the dataset. */
  vocabulary: string[];
  sequences: MovementSequence[];
};

/** Minimal structural shape of a recorded replay (matches ReplayManifest and the exporter's replay entries). */
export type MovementReplaySource = {
  sessionId: string;
  events: ReplayTimelineEvent[];
};

/**
 * Turn recorded replay timelines into a movement dataset. Events are ordered by
 * timestamp (defensively re-sorted) and tokenized; empty replays are dropped.
 */
export function buildMovementDataset(sources: readonly MovementReplaySource[]): MovementDataset {
  const vocabulary = new Set<string>();
  const sequences: MovementSequence[] = [];

  for (const source of sources) {
    const ordered = [...source.events].sort((a, b) => a.ts - b.ts);
    const tokens = ordered.map(tokenizeMovementEvent);
    if (tokens.length === 0) {
      continue;
    }
    for (const token of tokens) {
      vocabulary.add(token);
    }
    sequences.push({ id: source.sessionId, tokens });
  }

  return {
    version: 1,
    vocabulary: [...vocabulary].sort(),
    sequences,
  };
}

export type MovementTrainingConfig = {
  /** N-gram context length. `order: 2` conditions each prediction on the two preceding tokens. */
  order: number;
};

export type MovementModelMetadata = {
  backend: string;
  order: number;
  sequenceCount: number;
  /** Total tokens the model was trained on. */
  tokenCount: number;
  vocabulary: string[];
};

/** A trained movement model. Concrete backends extend this with their learned parameters; the object stays JSON-serializable so it can be persisted as a training artifact. */
export type MovementModel = {
  metadata: MovementModelMetadata;
};

export type MovementDistributionEntry = {
  token: string;
  probability: number;
};

export type MovementPrediction = {
  /** Most likely next token, or `null` when the model has no continuation for the context. */
  token: string | null;
  probability: number;
  /** Full next-token distribution, sorted by probability desc then token asc. */
  distribution: MovementDistributionEntry[];
  /** Backoff order actually used for this prediction (<= config order). */
  contextOrderUsed: number;
};

export type MovementTrainingRequest = {
  dataset: MovementDataset;
  config: MovementTrainingConfig;
};

/**
 * The pluggable contract every movement-model backend implements. `train`
 * produces a serializable model from a dataset; `predictNext` scores the next
 * movement given a context window. Both must be deterministic so replay and
 * evaluation are reproducible in CI.
 */
export interface MovementTrainingBackend<TModel extends MovementModel = MovementModel> {
  readonly id: string;
  train(request: MovementTrainingRequest): Promise<TModel>;
  predictNext(model: TModel, context: readonly string[]): MovementPrediction;
}

export type GenerateMovementSequenceOptions = {
  /** Priming tokens the generated sequence starts from (retained in the output). */
  seed: readonly string[];
  /** Maximum total length of the returned sequence, including the seed. */
  maxLength: number;
  /** Optional token that halts generation once produced (the token is included). */
  stopToken?: string;
};

/**
 * Deterministically generate a movement sequence by repeatedly taking the
 * model's most-likely next token (greedy decode). This is how a trained model
 * "repeats" a recorded movement from a seed and how it generalizes to new but
 * related movements via the backend's backoff behaviour.
 */
export function generateMovementSequence<TModel extends MovementModel>(
  backend: MovementTrainingBackend<TModel>,
  model: TModel,
  options: GenerateMovementSequenceOptions,
): string[] {
  const sequence = [...options.seed];
  const maxLength = Math.max(options.seed.length, options.maxLength);

  while (sequence.length < maxLength) {
    const prediction = backend.predictNext(model, sequence);
    if (prediction.token === null) {
      break;
    }
    sequence.push(prediction.token);
    if (options.stopToken !== undefined && prediction.token === options.stopToken) {
      break;
    }
  }

  return sequence;
}

export type ReplayFidelityReport = {
  /** Number of positions evaluated (positions with at least one preceding token). */
  predictions: number;
  correct: number;
  /** Fraction of positions where the model's greedy next-token matched the held-out truth (0 when no positions). */
  accuracy: number;
  /** Positions where the model produced no prediction at all (unseen context, no backoff hit). */
  unpredicted: number;
};

/**
 * Teacher-forced replay-fidelity eval: walk a held-out token sequence and, at
 * each position, compare the model's greedy next-token against the ground truth
 * given the true preceding context. Measures how well a trained model reproduces
 * / generalizes to a related trajectory it may not have been trained on.
 */
export function evaluateReplayFidelity<TModel extends MovementModel>(
  backend: MovementTrainingBackend<TModel>,
  model: TModel,
  sequence: readonly string[],
): ReplayFidelityReport {
  let correct = 0;
  let unpredicted = 0;
  let predictions = 0;

  for (let index = 1; index < sequence.length; index += 1) {
    predictions += 1;
    const context = sequence.slice(0, index);
    const prediction = backend.predictNext(model, context);
    if (prediction.token === null) {
      unpredicted += 1;
      continue;
    }
    if (prediction.token === sequence[index]) {
      correct += 1;
    }
  }

  return {
    predictions,
    correct,
    accuracy: predictions === 0 ? 0 : correct / predictions,
    unpredicted,
  };
}

/**
 * Deterministic seeded PRNG (mulberry32) — used only by the synthetic generator
 * so cloud/CI validation never depends on real OS input or wall-clock entropy.
 */
function mulberry32(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export type SyntheticMovementState = {
  /** Token emitted when the workflow is in this state. */
  token: string;
  /** Weighted transitions to the next state by name; weights need not sum to 1. */
  next: Array<{ state: string; weight: number }>;
};

export type SynthesizeMovementDatasetOptions = {
  sequenceCount: number;
  minLength: number;
  maxLength: number;
  seed: number;
  /** Optional custom state machine. Defaults to a small desktop-workflow grammar. */
  grammar?: Record<string, SyntheticMovementState>;
  /** State the walk starts from. Defaults to `"focus"` (or the first grammar key). */
  startState?: string;
};

/** A small default desktop-movement grammar: observe screen → move → click/type → maybe switch app. */
export function defaultMovementGrammar(): Record<string, SyntheticMovementState> {
  return {
    focus: {
      token: "observation:screen",
      next: [
        { state: "move", weight: 3 },
        { state: "switch", weight: 1 },
      ],
    },
    move: {
      token: "action:mouse-move",
      next: [
        { state: "click", weight: 3 },
        { state: "type", weight: 2 },
      ],
    },
    click: {
      token: "action:mouse-click",
      next: [
        { state: "focus", weight: 2 },
        { state: "type", weight: 2 },
        { state: "switch", weight: 1 },
      ],
    },
    type: {
      token: "action:key-press",
      next: [
        { state: "type", weight: 2 },
        { state: "click", weight: 1 },
        { state: "focus", weight: 1 },
      ],
    },
    switch: {
      token: "action:app-switch",
      next: [{ state: "focus", weight: 1 }],
    },
  };
}

/**
 * Generate a deterministic synthetic movement dataset from a weighted state
 * machine. Lets the capture→dataset→train→replay pipeline be validated end to
 * end without any real machine input.
 */
export function synthesizeMovementDataset(options: SynthesizeMovementDatasetOptions): MovementDataset {
  const grammar = options.grammar ?? defaultMovementGrammar();
  const stateNames = Object.keys(grammar);
  if (stateNames.length === 0) {
    throw new Error("synthesizeMovementDataset requires a non-empty grammar");
  }
  const startState = options.startState ?? (grammar.focus ? "focus" : stateNames[0]);
  if (!grammar[startState]) {
    throw new Error(`Unknown start state: ${startState}`);
  }
  const minLength = Math.max(1, options.minLength);
  const maxLength = Math.max(minLength, options.maxLength);
  const random = mulberry32(options.seed);

  const sources: MovementReplaySource[] = [];
  for (let seqIndex = 0; seqIndex < options.sequenceCount; seqIndex += 1) {
    const length = minLength + Math.floor(random() * (maxLength - minLength + 1));
    const events: ReplayTimelineEvent[] = [];
    let stateName = startState;
    for (let step = 0; step < length; step += 1) {
      const state = grammar[stateName];
      events.push(toSyntheticEvent(state.token, `${seqIndex}-${step}`, step));
      stateName = pickNextState(state, random);
    }
    sources.push({ sessionId: `synthetic-${seqIndex}`, events });
  }

  return buildMovementDataset(sources);
}

function pickNextState(state: SyntheticMovementState, random: () => number): string {
  if (state.next.length === 0) {
    return "focus";
  }
  const total = state.next.reduce((sum, transition) => sum + Math.max(0, transition.weight), 0);
  if (total <= 0) {
    return state.next[0].state;
  }
  let threshold = random() * total;
  for (const transition of state.next) {
    threshold -= Math.max(0, transition.weight);
    if (threshold < 0) {
      return transition.state;
    }
  }
  return state.next[state.next.length - 1].state;
}

/** Reconstruct a minimal replay event that tokenizes back to the given token. */
function toSyntheticEvent(token: string, trajectoryId: string, ts: number): ReplayTimelineEvent {
  const separatorIndex = token.indexOf(":");
  const kind = token.slice(0, separatorIndex);
  const value = token.slice(separatorIndex + 1);
  if (kind === "observation") {
    return { kind: "observation", ts, trajectoryId, source: value, summary: value };
  }
  if (kind === "transcript") {
    const role = value === "system" || value === "user" || value === "assistant" || value === "tool" ? value : "assistant";
    return { kind: "transcript", ts, messageId: `${trajectoryId}`, role, content: value };
  }
  return { kind: "action", ts, trajectoryId, tool: value, summary: value };
}
