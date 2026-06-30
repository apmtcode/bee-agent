/**
 * Pluggable local movement-model backend.
 *
 * Standing objective #2 (parts c & d): post-train a local model on recorded
 * movement data so the agent can (c) repeat recorded movements and (d)
 * generalize to new-but-related movements.
 *
 * The existing `runner.ts` only emits *shell commands* that drive an external
 * Apple-Silicon trainer (MLX / axolotl) -- there is no in-process model the
 * agent can train and query. This module fills that gap with a small,
 * dependency-free, fully deterministic model so the whole capture -> dataset ->
 * train -> infer loop can be exercised in the cloud (no real OS, no GPU).
 *
 * The backend is pluggable: `MovementModelBackend` is the seam. The bundled
 * `MarkovMovementBackend` is an order-k Markov model with stupid-backoff -- it
 * genuinely learns sequential structure and generalizes via backoff to shorter
 * contexts. A real on-device small model can implement the same interface and
 * be swapped in without touching callers.
 */

import type { TrajectoryAction, TrajectorySpan } from "../capture/trajectory.js";

/** A single recorded movement/action (mouse, key, window, tool, ...). */
export type MovementToken = {
  /** Action channel, e.g. `mouse.move`, `key.press`, `window.focus`. */
  tool: string;
  /** Human-readable label for the action. */
  summary: string;
  /** Optional structured payload (coordinates, key codes, app ids, ...). */
  metadata?: Record<string, unknown>;
};

/** One demonstration: an ordered list of movement tokens. */
export type MovementExample = {
  id?: string;
  actions: MovementToken[];
};

export type MovementTrainingDataset = {
  examples: MovementExample[];
};

/**
 * Maps a token to the key the model groups by. The default keys on
 * `tool` + `summary` (exact replay). `toolMovementTokenizer` keys on `tool`
 * alone, which makes the model generalize across summaries/coordinates --
 * useful for "new but related" movements. These two built-ins are the only
 * tokenizers `predictNext` can resolve unknown tokens against; a custom
 * tokenizer still trains, but inference falls back to backoff for tokens it
 * cannot key.
 */
export type MovementTokenizer = (token: MovementToken) => string;

export type MovementTrainingOptions = {
  /** Markov order (context length). Default 2. */
  order?: number;
  tokenizer?: MovementTokenizer;
};

export type MovementPrediction = {
  key: string;
  /** Representative token for this key (first one seen in training). */
  token: MovementToken;
  probability: number;
  count: number;
  /** True when this prediction is the end-of-sequence boundary. */
  isEnd: boolean;
};

export type MovementModelSnapshot = {
  version: 1;
  backendId: string;
  order: number;
  /** contextKey -> nextKey -> count */
  transitions: Array<[string, Array<[string, number]>]>;
  /** key -> representative token */
  representatives: Array<[string, MovementToken]>;
};

export interface TrainedMovementModel {
  readonly backendId: string;
  readonly order: number;
  /**
   * Rank likely next tokens for a context (the trailing tokens of a
   * sequence). Uses stupid-backoff: the longest context with observed
   * transitions wins; otherwise the model backs off to shorter contexts down
   * to the unigram distribution -- this is what lets it generalize.
   */
  predictNext(context: MovementToken[]): MovementPrediction[];
  /**
   * Greedily roll out up to `steps` tokens from a seed, stopping early at the
   * end-of-sequence boundary. Deterministic (top prediction each step).
   */
  generate(seed: MovementToken[], steps: number): MovementToken[];
  serialize(): MovementModelSnapshot;
}

export interface MovementModelBackend {
  readonly id: string;
  train(
    dataset: MovementTrainingDataset,
    options?: MovementTrainingOptions,
  ): Promise<TrainedMovementModel>;
}

// Boundary sentinels. The leading space keeps them out of realistic tool
// namespaces; CONTEXT_SEP separates joined keys so distinct boundaries cannot
// collide (e.g. ["ab","c"] vs ["a","bc"]).
const START_KEY = " START";
const END_KEY = " END";
const UNKNOWN_KEY = " UNK";
const CONTEXT_SEP = "";
const END_TOKEN: MovementToken = { tool: END_KEY, summary: "" };

export function isMovementBoundary(token: MovementToken): boolean {
  return token.tool === START_KEY || token.tool === END_KEY;
}

export function defaultMovementTokenizer(token: MovementToken): string {
  return `${token.tool}${CONTEXT_SEP}${token.summary}`;
}

/** Tokenizer that groups by channel only -- generalizes across labels. */
export function toolMovementTokenizer(token: MovementToken): string {
  return token.tool;
}

/**
 * Deterministic order-k Markov backend with stupid-backoff. No randomness, no
 * external deps -- identical input always yields an identical model.
 */
export class MarkovMovementBackend implements MovementModelBackend {
  readonly id = "markov-backoff";

  async train(
    dataset: MovementTrainingDataset,
    options: MovementTrainingOptions = {},
  ): Promise<TrainedMovementModel> {
    const order = Math.max(1, Math.floor(options.order ?? 2));
    const tokenizer = options.tokenizer ?? defaultMovementTokenizer;

    const transitions = new Map<string, Map<string, number>>();
    const representatives = new Map<string, MovementToken>();
    representatives.set(END_KEY, END_TOKEN);

    const keyOf = (token: MovementToken): string => {
      if (token.tool === START_KEY) return START_KEY;
      if (token.tool === END_KEY) return END_KEY;
      const key = tokenizer(token);
      if (!representatives.has(key)) {
        representatives.set(key, token);
      }
      return key;
    };

    const bump = (contextKeys: string[], nextKey: string): void => {
      const contextKey = contextKeys.join(CONTEXT_SEP);
      let row = transitions.get(contextKey);
      if (!row) {
        row = new Map<string, number>();
        transitions.set(contextKey, row);
      }
      row.set(nextKey, (row.get(nextKey) ?? 0) + 1);
    };

    for (const example of dataset.examples) {
      const keys = [
        ...Array.from({ length: order }, () => START_KEY),
        ...example.actions.map(keyOf),
        END_KEY,
      ];
      // Record every backoff order (0..order) so shorter contexts -- down to
      // the empty-context unigram marginal (k=0) -- are available even when a
      // longer one was never observed. The k=0 level guarantees a wholly
      // unfamiliar context still yields predictions instead of nothing.
      for (let i = order; i < keys.length; i += 1) {
        const nextKey = keys[i] as string;
        for (let k = 0; k <= order; k += 1) {
          bump(keys.slice(i - k, i), nextKey);
        }
      }
    }

    return new MarkovMovementModel(this.id, order, transitions, representatives);
  }
}

class MarkovMovementModel implements TrainedMovementModel {
  constructor(
    readonly backendId: string,
    readonly order: number,
    private readonly transitions: Map<string, Map<string, number>>,
    private readonly representatives: Map<string, MovementToken>,
  ) {}

  predictNext(context: MovementToken[]): MovementPrediction[] {
    const contextKeys = this.contextKeysFor(context);
    for (let k = this.order; k >= 0; k -= 1) {
      const slice = contextKeys.slice(contextKeys.length - k);
      const row = this.transitions.get(slice.join(CONTEXT_SEP));
      if (row && row.size > 0) {
        return this.rankRow(row);
      }
    }
    return [];
  }

  generate(seed: MovementToken[], steps: number): MovementToken[] {
    const out: MovementToken[] = [];
    const working = [...seed];
    for (let i = 0; i < Math.max(0, steps); i += 1) {
      const ranked = this.predictNext(working);
      const top = ranked[0];
      if (!top || top.isEnd) {
        break;
      }
      out.push(top.token);
      working.push(top.token);
    }
    return out;
  }

  serialize(): MovementModelSnapshot {
    return {
      version: 1,
      backendId: this.backendId,
      order: this.order,
      transitions: Array.from(this.transitions.entries()).map(([context, row]) => [
        context,
        Array.from(row.entries()),
      ]),
      representatives: Array.from(this.representatives.entries()),
    };
  }

  private contextKeysFor(context: MovementToken[]): string[] {
    const tail = context.slice(Math.max(0, context.length - this.order));
    const keys = tail.map((token) => this.keyOfExisting(token));
    while (keys.length < this.order) {
      keys.unshift(START_KEY);
    }
    return keys;
  }

  /**
   * Resolve a token to a known key. Unknown tokens collapse to a sentinel that
   * matches nothing, so an unfamiliar context simply backs off -- generalizing
   * rather than throwing.
   */
  private keyOfExisting(token: MovementToken): string {
    if (token.tool === START_KEY) return START_KEY;
    const exact = `${token.tool}${CONTEXT_SEP}${token.summary}`;
    if (this.representatives.has(exact)) return exact;
    if (this.representatives.has(token.tool)) return token.tool;
    return UNKNOWN_KEY;
  }

  private rankRow(row: Map<string, number>): MovementPrediction[] {
    let total = 0;
    for (const count of row.values()) total += count;
    return Array.from(row.entries())
      .map(([key, count]) => ({
        key,
        token: this.representatives.get(key) ?? { tool: key, summary: "" },
        probability: count / total,
        count,
        isEnd: key === END_KEY,
      }))
      .sort((a, b) => {
        if (b.probability !== a.probability) return b.probability - a.probability;
        return a.key < b.key ? -1 : a.key > b.key ? 1 : 0;
      });
  }
}

export function restoreMovementModel(snapshot: MovementModelSnapshot): TrainedMovementModel {
  const transitions = new Map<string, Map<string, number>>(
    snapshot.transitions.map(([context, row]) => [context, new Map(row)]),
  );
  const representatives = new Map<string, MovementToken>(snapshot.representatives);
  return new MarkovMovementModel(snapshot.backendId, snapshot.order, transitions, representatives);
}

// --- Dataset helpers -------------------------------------------------------

export function trajectoryActionsToMovementTokens(
  actions: TrajectoryAction[],
): MovementToken[] {
  return actions.map((action) => ({
    tool: action.tool,
    summary: action.summary,
    ...(action.metadata ? { metadata: action.metadata } : {}),
  }));
}

export function buildMovementDatasetFromTrajectories(
  trajectories: TrajectorySpan[],
): MovementTrainingDataset {
  return {
    examples: trajectories
      .map((trajectory) => ({
        id: trajectory.id,
        actions: trajectoryActionsToMovementTokens(
          [...trajectory.actions].sort((a, b) => a.ts - b.ts),
        ),
      }))
      .filter((example) => example.actions.length > 0),
  };
}

// --- Evaluation harness ----------------------------------------------------

export type MovementEvalResult = {
  /** Examples scored (non-empty). */
  exampleCount: number;
  /** Total next-token decisions evaluated (incl. the end-of-sequence step). */
  decisionCount: number;
  /** Decisions where the model's top prediction matched the held-out next token. */
  topHits: number;
  topAccuracy: number;
  /** Decisions where the actual next token was within the top-k predictions. */
  topKHits: number;
  topKAccuracy: number;
};

/**
 * Teacher-forced next-token accuracy on held-out demonstrations. For each
 * prefix of an example we ask the model for its ranked next tokens and check
 * whether the true continuation is the top pick (and within top-k). The final
 * decision per example is the end-of-sequence boundary, so "knowing when to
 * stop" is scored too.
 */
export function evaluateMovementModel(
  model: TrainedMovementModel,
  heldOut: MovementExample[],
  options: { topK?: number; tokenizer?: MovementTokenizer } = {},
): MovementEvalResult {
  const topK = Math.max(1, Math.floor(options.topK ?? 3));
  const tokenizer = options.tokenizer ?? defaultMovementTokenizer;
  const keyOf = (token: MovementToken): string =>
    token.tool === END_KEY ? END_KEY : tokenizer(token);

  let exampleCount = 0;
  let decisionCount = 0;
  let topHits = 0;
  let topKHits = 0;

  for (const example of heldOut) {
    if (example.actions.length === 0) continue;
    exampleCount += 1;
    const sequence = [...example.actions, END_TOKEN];
    for (let i = 0; i < sequence.length; i += 1) {
      const expectedKey = keyOf(sequence[i] as MovementToken);
      const ranked = model.predictNext(example.actions.slice(0, i));
      decisionCount += 1;
      if (ranked[0] && ranked[0].key === expectedKey) topHits += 1;
      if (ranked.slice(0, topK).some((p) => p.key === expectedKey)) topKHits += 1;
    }
  }

  return {
    exampleCount,
    decisionCount,
    topHits,
    topAccuracy: decisionCount === 0 ? 0 : topHits / decisionCount,
    topKHits,
    topKAccuracy: decisionCount === 0 ? 0 : topKHits / decisionCount,
  };
}

// --- Synthetic event-stream generator --------------------------------------

/**
 * Deterministic LCG. We deliberately avoid `Math.random`/`Date.now` so every
 * synthetic dataset is reproducible from its seed -- required for stable tests
 * and for the engine's no-`Math.random` constraint.
 */
function lcg(seed: number): () => number {
  let state = (seed >>> 0) || 1;
  return () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 0x100000000;
  };
}

export type SyntheticMovementSkill = {
  name: string;
  /** Canonical token sequence for the skill. */
  steps: MovementToken[];
};

/** A small library of canonical movement "skills" to synthesize variations of. */
export function defaultSyntheticSkills(): SyntheticMovementSkill[] {
  return [
    {
      name: "open-app",
      steps: [
        { tool: "key.press", summary: "cmd+space" },
        { tool: "key.type", summary: "type-app-name" },
        { tool: "key.press", summary: "enter" },
        { tool: "window.focus", summary: "focus-app" },
      ],
    },
    {
      name: "drag-file",
      steps: [
        { tool: "mouse.move", summary: "to-file" },
        { tool: "mouse.down", summary: "grab" },
        { tool: "mouse.move", summary: "to-target" },
        { tool: "mouse.up", summary: "drop" },
      ],
    },
    {
      name: "save-document",
      steps: [
        { tool: "window.focus", summary: "focus-editor" },
        { tool: "key.press", summary: "cmd+s" },
        { tool: "key.press", summary: "enter" },
      ],
    },
  ];
}

export type SyntheticMovementOptions = {
  seed: number;
  count: number;
  skills?: SyntheticMovementSkill[];
  /** Probability of injecting a benign noise action between steps. */
  noiseRate?: number;
};

/**
 * Produce a reproducible dataset of skill demonstrations with mild, structured
 * variation (coordinate jitter in metadata + occasional noise actions). Used to
 * validate the capture -> dataset -> train -> infer loop and to measure
 * generalization on held-out variations -- all without real OS input.
 */
export function generateSyntheticMovementExamples(
  options: SyntheticMovementOptions,
): MovementExample[] {
  const rng = lcg(options.seed);
  const skills = options.skills ?? defaultSyntheticSkills();
  const noiseRate = options.noiseRate ?? 0;
  const examples: MovementExample[] = [];

  for (let i = 0; i < Math.max(0, options.count); i += 1) {
    const index = Math.floor(rng() * skills.length) % skills.length;
    const skill = skills[index] as SyntheticMovementSkill;
    const actions: MovementToken[] = [];
    for (const step of skill.steps) {
      if (noiseRate > 0 && rng() < noiseRate) {
        actions.push({ tool: "mouse.move", summary: "idle-jitter" });
      }
      actions.push({
        tool: step.tool,
        summary: step.summary,
        metadata: { x: Math.floor(rng() * 1000), y: Math.floor(rng() * 1000) },
      });
    }
    examples.push({ id: `${skill.name}-${i}`, actions });
  }

  return examples;
}
