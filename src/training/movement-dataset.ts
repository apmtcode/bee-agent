// Movement dataset construction + synthetic event streams.
//
// Turns recorded `TrajectorySpan`s (from `src/capture`) into the token-stream
// `MovementDataset` the movement-model backend trains on, and provides a
// seeded synthetic generator so the whole capture -> dataset -> train -> eval
// round-trip can be validated in the cloud with no real OS input.

import { buildTrajectorySpan, type TrajectoryAction, type TrajectorySpan } from "../capture/trajectory.js";
import type { MovementDataset, MovementExample, MovementToken } from "./movement-model.js";

/**
 * Canonical token for a recorded action. Prefers a stable structured target
 * from `metadata.target`; otherwise falls back to a slugged summary so free-text
 * actions still collapse to a repeatable token. The tool always leads so the
 * vocabulary stays grouped by action kind.
 */
export function tokenizeAction(action: TrajectoryAction): MovementToken {
  const target = typeof action.metadata?.target === "string" ? action.metadata.target : undefined;
  const detail = target ?? slug(action.summary);
  return detail ? `${action.tool}:${detail}` : action.tool;
}

function slug(value: string): string {
  return value
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
}

/** Ordered token stream for one trajectory (actions sorted by timestamp). */
export function trajectoryToTokens(
  trajectory: TrajectorySpan,
  tokenize: (action: TrajectoryAction) => MovementToken = tokenizeAction,
): MovementToken[] {
  return [...trajectory.actions].sort((a, b) => a.ts - b.ts).map(tokenize);
}

/**
 * Sliding-window (context, next) pairs over a token stream. `context` holds up
 * to `order` preceding tokens; the first token yields an empty-context example
 * so the model also learns movement *starts*.
 */
export function tokensToExamples(tokens: MovementToken[], order: number): MovementExample[] {
  const examples: MovementExample[] = [];
  for (let i = 0; i < tokens.length; i += 1) {
    const start = Math.max(0, i - order);
    examples.push({ context: tokens.slice(start, i), next: tokens[i]! });
  }
  return examples;
}

export type BuildMovementDatasetOptions = {
  /** Longest context window to emit; should match the backend's maxOrder. Default 3. */
  order?: number;
  tokenize?: (action: TrajectoryAction) => MovementToken;
};

export function buildMovementDataset(
  trajectories: TrajectorySpan[],
  options: BuildMovementDatasetOptions = {},
): MovementDataset {
  const order = Math.max(0, Math.trunc(options.order ?? 3));
  const tokenize = options.tokenize ?? tokenizeAction;
  const examples: MovementExample[] = [];
  const vocabulary = new Set<MovementToken>();
  for (const trajectory of trajectories) {
    const tokens = trajectoryToTokens(trajectory, tokenize);
    for (const token of tokens) {
      vocabulary.add(token);
    }
    examples.push(...tokensToExamples(tokens, order));
  }
  return {
    version: 1,
    vocabulary: [...vocabulary].sort(),
    examples,
  };
}

/**
 * Split trajectories into train/held-out groups deterministically (every
 * `1/holdoutRatio`-th trajectory is held out). Splitting at the trajectory
 * level — not the example level — keeps whole movements out of training so the
 * eval measures real generalization, not memorized prefixes.
 */
export function splitTrajectories(
  trajectories: TrajectorySpan[],
  holdoutRatio = 0.25,
): { train: TrajectorySpan[]; heldOut: TrajectorySpan[] } {
  const stride = holdoutRatio > 0 ? Math.max(1, Math.round(1 / holdoutRatio)) : Number.POSITIVE_INFINITY;
  const train: TrajectorySpan[] = [];
  const heldOut: TrajectorySpan[] = [];
  trajectories.forEach((trajectory, index) => {
    if (Number.isFinite(stride) && index % stride === stride - 1) {
      heldOut.push(trajectory);
    } else {
      train.push(trajectory);
    }
  });
  return { train, heldOut };
}

// ---------------------------------------------------------------------------
// Synthetic event-stream generator
// ---------------------------------------------------------------------------

/** Small deterministic PRNG (mulberry32) — no Date/Math.random dependency. */
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

/**
 * A movement "grammar": a graph of UI states, each offering weighted actions
 * that lead to a next state. Sampling walks produce related-but-varied
 * trajectories — shared substructure the model can generalize over, plus
 * enough branching that held-out walks aren't identical to training ones.
 */
export type MovementGrammarState = {
  id: string;
  actions: Array<{ tool: string; target: string; weight?: number; to: string }>;
};

export type MovementGrammar = {
  start: string;
  terminal: string;
  states: MovementGrammarState[];
};

/** A canonical, non-trivial default grammar: a "compose + send message" flow. */
export function defaultMovementGrammar(): MovementGrammar {
  return {
    start: "inbox",
    terminal: "sent",
    states: [
      {
        id: "inbox",
        actions: [
          { tool: "click", target: "compose-button", to: "composer", weight: 3 },
          { tool: "click", target: "search-box", to: "search", weight: 1 },
        ],
      },
      {
        id: "search",
        actions: [
          { tool: "type", target: "search-query", to: "inbox", weight: 1 },
          { tool: "click", target: "compose-button", to: "composer", weight: 2 },
        ],
      },
      {
        id: "composer",
        actions: [
          { tool: "type", target: "recipient-field", to: "composer-body", weight: 4 },
        ],
      },
      {
        id: "composer-body",
        actions: [
          { tool: "type", target: "subject-field", to: "composer-ready", weight: 2 },
          { tool: "type", target: "message-body", to: "composer-ready", weight: 3 },
        ],
      },
      {
        id: "composer-ready",
        actions: [
          { tool: "click", target: "attach-button", to: "composer-body", weight: 1 },
          { tool: "click", target: "send-button", to: "sent", weight: 4 },
        ],
      },
    ],
  };
}

export type GenerateSyntheticTrajectoriesOptions = {
  count: number;
  seed?: number;
  grammar?: MovementGrammar;
  sessionPrefix?: string;
  /** Safety bound on walk length to guard against cyclic grammars. Default 24. */
  maxSteps?: number;
  /** Base epoch-ms for action timestamps (kept explicit — no wall clock). Default 0. */
  baseTs?: number;
};

/**
 * Generate `count` deterministic synthetic trajectories by sampling weighted
 * walks through `grammar`. Same seed + options => byte-identical output, so
 * tests and evals are fully reproducible.
 */
export function generateSyntheticTrajectories(
  options: GenerateSyntheticTrajectoriesOptions,
): TrajectorySpan[] {
  const grammar = options.grammar ?? defaultMovementGrammar();
  const stateById = new Map(grammar.states.map((state) => [state.id, state] as const));
  const rng = mulberry32(options.seed ?? 1);
  const prefix = options.sessionPrefix ?? "synthetic";
  const maxSteps = Math.max(1, options.maxSteps ?? 24);
  const baseTs = options.baseTs ?? 0;
  const trajectories: TrajectorySpan[] = [];

  for (let i = 0; i < options.count; i += 1) {
    const actions: TrajectoryAction[] = [];
    let current = grammar.start;
    let ts = baseTs + i * 1_000_000;
    for (let step = 0; step < maxSteps && current !== grammar.terminal; step += 1) {
      const state = stateById.get(current);
      if (!state || state.actions.length === 0) {
        break;
      }
      const choice = weightedPick(state.actions, rng);
      ts += 1000 + Math.floor(rng() * 500);
      actions.push({
        kind: "action",
        tool: choice.tool,
        summary: `${choice.tool} ${choice.target}`,
        ts,
        metadata: { target: choice.target, fromState: current, toState: choice.to },
      });
      current = choice.to;
    }
    trajectories.push(
      buildTrajectorySpan({
        id: `${prefix}-${i}`,
        sessionId: `${prefix}-session-${i}`,
        captureTier: "full",
        actions,
        outcome:
          current === grammar.terminal
            ? { status: "success", summary: "reached terminal state", reward: 1 }
            : { status: "aborted", summary: "walk exceeded step budget" },
      }),
    );
  }
  return trajectories;
}

function weightedPick<T extends { weight?: number }>(items: T[], rng: () => number): T {
  const total = items.reduce((sum, item) => sum + (item.weight ?? 1), 0);
  let threshold = rng() * total;
  for (const item of items) {
    threshold -= item.weight ?? 1;
    if (threshold < 0) {
      return item;
    }
  }
  return items[items.length - 1]!;
}
