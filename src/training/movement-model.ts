import type { TrajectoryAction, TrajectoryObservation, TrajectorySpan } from "../capture/trajectory.js";

// ---------------------------------------------------------------------------
// Movement model — a pluggable, in-process learner for the local-movement
// subsystem (standing objective #2, parts c & d).
//
// The training *runner* (runner.ts) emits shell plans that drive real on-device
// MLX/axolotl jobs; that path cannot execute in the cloud. This module provides
// the complementary in-process piece: a backend interface plus a deterministic,
// fully local backend that actually LEARNS from a reviewed movement dataset,
// can REPEAT recorded movements exactly, and GENERALIZES to new-but-related
// movements via n-gram backoff. It is deterministic (no clocks, no RNG in the
// hot path) so the whole capture -> dataset -> train -> infer -> eval loop is
// testable with synthetic data. The backend is pluggable so a real on-device
// small model can implement the same contract later.
// ---------------------------------------------------------------------------

/** A single vocabulary token, e.g. `obs:screen` or `act:mouse.click`. */
export type MovementToken = string;

export const OBSERVATION_TOKEN_PREFIX = "obs:";
export const ACTION_TOKEN_PREFIX = "act:";

const CONTEXT_KEY_SEPARATOR = "";

export function observationToken(observation: Pick<TrajectoryObservation, "source">): MovementToken {
  return `${OBSERVATION_TOKEN_PREFIX}${observation.source}`;
}

export function actionToken(action: Pick<TrajectoryAction, "tool">): MovementToken {
  return `${ACTION_TOKEN_PREFIX}${action.tool}`;
}

export function isActionToken(token: MovementToken): boolean {
  return token.startsWith(ACTION_TOKEN_PREFIX);
}

export type MovementStep =
  | { kind: "observation"; token: MovementToken; ts: number; summary: string }
  | { kind: "action"; token: MovementToken; ts: number; summary: string };

export type MovementSequence = {
  trajectoryId: string;
  sessionId: string;
  steps: MovementStep[];
};

/** One supervised example: predict `action` given the preceding `context`. */
export type MovementExample = {
  trajectoryId: string;
  context: MovementToken[];
  action: MovementToken;
};

export type MovementDataset = {
  version: 1;
  sequences: MovementSequence[];
  examples: MovementExample[];
  vocabulary: { observations: string[]; actions: string[] };
};

/**
 * Interleave a trajectory's observations and actions into a single time-ordered
 * step sequence. Uses the reviewed (redacted) view when present, matching the
 * exporter, so training never sees anything the reviewer redacted.
 */
export function buildMovementSequence(trajectory: TrajectorySpan): MovementSequence {
  const observations = trajectory.review?.redactedObservations
    ? trajectory.review.redactedObservations.map((observation) => ({
        source: observation.source,
        summary: observation.summary,
        ts: observation.ts,
      }))
    : trajectory.observations.map((observation) => ({
        source: observation.source,
        summary: observation.summary,
        ts: observation.ts,
      }));
  const actions = trajectory.review?.redactedActions
    ? trajectory.review.redactedActions.map((action) => ({
        tool: action.tool,
        summary: action.summary,
        ts: action.ts,
      }))
    : trajectory.actions.map((action) => ({ tool: action.tool, summary: action.summary, ts: action.ts }));

  const steps: MovementStep[] = [
    ...observations.map<MovementStep>((observation) => ({
      kind: "observation",
      token: observationToken(observation),
      ts: observation.ts,
      summary: observation.summary,
    })),
    ...actions.map<MovementStep>((action) => ({
      kind: "action",
      token: actionToken(action),
      ts: action.ts,
      summary: action.summary,
    })),
  ].sort((a, b) => {
    if (a.ts !== b.ts) {
      return a.ts - b.ts;
    }
    // Stable, deterministic tie-break: observations precede actions at equal ts
    // (you observe, then act), matching replay.ts kind ordering.
    return (a.kind === "observation" ? 0 : 1) - (b.kind === "observation" ? 0 : 1);
  });

  return { trajectoryId: trajectory.id, sessionId: trajectory.sessionId, steps };
}

/** Build a full training dataset (sequences + supervised examples + vocab). */
export function buildMovementDataset(trajectories: TrajectorySpan[]): MovementDataset {
  const sequences = trajectories.map(buildMovementSequence);
  const examples: MovementExample[] = [];
  const observations = new Set<string>();
  const actions = new Set<string>();

  for (const sequence of sequences) {
    const context: MovementToken[] = [];
    for (const step of sequence.steps) {
      if (step.kind === "observation") {
        observations.add(step.token);
      } else {
        actions.add(step.token);
        examples.push({ trajectoryId: sequence.trajectoryId, context: [...context], action: step.token });
      }
      context.push(step.token);
    }
  }

  return {
    version: 1,
    sequences,
    examples,
    vocabulary: {
      observations: [...observations].sort(),
      actions: [...actions].sort(),
    },
  };
}

// ---------------------------------------------------------------------------
// Backend contract
// ---------------------------------------------------------------------------

export type MovementModelConfig = {
  /** n-gram context window (number of trailing tokens conditioned on). */
  order: number;
};

export const DEFAULT_MOVEMENT_MODEL_CONFIG: MovementModelConfig = { order: 3 };

export type MovementPrediction = {
  action: MovementToken;
  /** Count-based probability of the chosen action within the matched context. */
  confidence: number;
  /** Context length that actually matched (order..0); -1 if nothing matched. */
  backoffOrder: number;
  /** True when the full-order context was memorized during training. */
  fromMemory: boolean;
};

/** Serializable trained artifact — round-trips to disk like the rest of the repo. */
export type MovementModelArtifact = {
  version: 1;
  backend: string;
  order: number;
  /** tables[n] maps a context-key (last n tokens joined) -> action -> count. */
  tables: Array<Record<string, Record<string, number>>>;
  fallbackAction: string | null;
};

export interface MovementPolicy {
  readonly backendName: string;
  readonly config: MovementModelConfig;
  predict(context: MovementToken[]): MovementPrediction;
  serialize(): MovementModelArtifact;
}

export interface MovementModelBackend {
  readonly name: string;
  train(dataset: MovementDataset, config?: Partial<MovementModelConfig>): MovementPolicy;
  restore(artifact: MovementModelArtifact): MovementPolicy;
}

function contextKey(context: MovementToken[], n: number): string {
  return n === 0 ? "" : context.slice(-n).join(CONTEXT_KEY_SEPARATOR);
}

/** Deterministic pick: highest count, then lexicographically smallest token. */
function pickBest(counts: Record<string, number>): { action: string; count: number; total: number } | undefined {
  let best: { action: string; count: number } | undefined;
  let total = 0;
  for (const [action, count] of Object.entries(counts)) {
    total += count;
    if (!best || count > best.count || (count === best.count && action < best.action)) {
      best = { action, count };
    }
  }
  return best ? { ...best, total } : undefined;
}

class NGramMovementPolicy implements MovementPolicy {
  constructor(
    readonly backendName: string,
    readonly config: MovementModelConfig,
    private readonly tables: Array<Map<string, Record<string, number>>>,
    private readonly fallbackAction: string | null,
  ) {}

  predict(context: MovementToken[]): MovementPrediction {
    for (let n = this.config.order; n >= 0; n -= 1) {
      const table = this.tables[n];
      if (!table) {
        continue;
      }
      const counts = table.get(contextKey(context, n));
      if (!counts) {
        continue;
      }
      const best = pickBest(counts);
      if (best) {
        return {
          action: best.action,
          confidence: best.total > 0 ? best.count / best.total : 0,
          backoffOrder: n,
          fromMemory: n === this.config.order,
        };
      }
    }
    return { action: this.fallbackAction ?? "", confidence: 0, backoffOrder: -1, fromMemory: false };
  }

  serialize(): MovementModelArtifact {
    return {
      version: 1,
      backend: this.backendName,
      order: this.config.order,
      tables: this.tables.map((table) => Object.fromEntries(table)),
      fallbackAction: this.fallbackAction,
    };
  }
}

/**
 * A deterministic, dependency-free local backend. Learns backoff n-gram counts
 * over movement tokens: it memorizes recorded sequences exactly (full-order
 * context match) and generalizes to unseen contexts by backing off to shorter
 * contexts, all the way down to the marginal action distribution. Serves as the
 * default backend and as the mock for cloud/CI while a real on-device small
 * model implements the same {@link MovementModelBackend} contract.
 */
export class DeterministicNGramBackend implements MovementModelBackend {
  readonly name = "deterministic-ngram";

  train(dataset: MovementDataset, config?: Partial<MovementModelConfig>): MovementPolicy {
    const order = Math.max(0, config?.order ?? DEFAULT_MOVEMENT_MODEL_CONFIG.order);
    const tables: Array<Map<string, Record<string, number>>> = Array.from({ length: order + 1 }, () => new Map());
    const marginal: Record<string, number> = {};

    for (const example of dataset.examples) {
      marginal[example.action] = (marginal[example.action] ?? 0) + 1;
      for (let n = order; n >= 0; n -= 1) {
        const key = contextKey(example.context, n);
        const table = tables[n]!;
        const counts = table.get(key) ?? {};
        counts[example.action] = (counts[example.action] ?? 0) + 1;
        table.set(key, counts);
      }
    }

    const fallback = pickBest(marginal)?.action ?? null;
    return new NGramMovementPolicy(this.name, { order }, tables, fallback);
  }

  restore(artifact: MovementModelArtifact): MovementPolicy {
    const tables = artifact.tables.map((table) => new Map(Object.entries(table)));
    return new NGramMovementPolicy(artifact.backend, { order: artifact.order }, tables, artifact.fallbackAction);
  }
}

// ---------------------------------------------------------------------------
// Generalization eval harness
// ---------------------------------------------------------------------------

export type MovementEvalResult = {
  totalPredictions: number;
  correct: number;
  accuracy: number;
  /** Correct predictions where the full-order context was memorized. */
  exactMemoryHits: number;
  /** Correct predictions produced by backing off to a shorter context. */
  generalizedHits: number;
  /** Per-backoff-order breakdown, keyed by the matched context length. */
  byBackoffOrder: Record<number, { total: number; correct: number }>;
};

/**
 * Replay-fidelity eval: for every action step in the held-out trajectories,
 * predict the next action from the preceding (ground-truth) context and compare
 * to what actually happened. Reports overall accuracy plus how much of it came
 * from exact memorization vs. generalization (backoff), which is the signal for
 * objective #2(d).
 */
export function evaluateMovementPolicy(policy: MovementPolicy, testTrajectories: TrajectorySpan[]): MovementEvalResult {
  const byBackoffOrder: Record<number, { total: number; correct: number }> = {};
  let totalPredictions = 0;
  let correct = 0;
  let exactMemoryHits = 0;
  let generalizedHits = 0;

  for (const trajectory of testTrajectories) {
    const sequence = buildMovementSequence(trajectory);
    const context: MovementToken[] = [];
    for (const step of sequence.steps) {
      if (step.kind === "action") {
        const prediction = policy.predict(context);
        const isCorrect = prediction.action === step.token;
        totalPredictions += 1;
        const bucket = (byBackoffOrder[prediction.backoffOrder] ??= { total: 0, correct: 0 });
        bucket.total += 1;
        if (isCorrect) {
          correct += 1;
          bucket.correct += 1;
          if (prediction.fromMemory) {
            exactMemoryHits += 1;
          } else {
            generalizedHits += 1;
          }
        }
      }
      context.push(step.token);
    }
  }

  return {
    totalPredictions,
    correct,
    accuracy: totalPredictions > 0 ? correct / totalPredictions : 0,
    exactMemoryHits,
    generalizedHits,
    byBackoffOrder,
  };
}

// ---------------------------------------------------------------------------
// Synthetic movement generator (for cloud/CI validation without real OS input)
// ---------------------------------------------------------------------------

/** Deterministic 32-bit LCG — reproducible synthetic streams, no Math.random. */
function createLcg(seed: number): () => number {
  let state = (seed >>> 0) || 0x9e3779b9;
  return () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 0x100000000;
  };
}

export type SyntheticMovementParams = {
  count: number;
  seed: number;
  sessionId?: string;
  /** 0..1 probability of perturbing a step (drop/duplicate) to force generalization. */
  variability?: number;
};

/**
 * A small "workflow grammar": each template is a plausible UI movement pattern
 * expressed as (observation source, action tool) beats. Concrete synthetic
 * trajectories are drawn from these with seeded perturbation so held-out
 * trajectories share local structure but differ globally — exactly the
 * "new but related movement" regime objective #2(d) targets.
 */
const WORKFLOW_TEMPLATES: Array<Array<{ source: string; tool: string }>> = [
  [
    { source: "screen", tool: "window.focus" },
    { source: "menu", tool: "mouse.move" },
    { source: "menu", tool: "mouse.click" },
    { source: "editor", tool: "key.type" },
    { source: "editor", tool: "key.press:enter" },
  ],
  [
    { source: "browser", tool: "window.focus" },
    { source: "address-bar", tool: "mouse.click" },
    { source: "address-bar", tool: "key.type" },
    { source: "address-bar", tool: "key.press:enter" },
    { source: "page", tool: "scroll.down" },
  ],
  [
    { source: "files", tool: "window.focus" },
    { source: "files", tool: "mouse.move" },
    { source: "files", tool: "mouse.double-click" },
    { source: "preview", tool: "scroll.down" },
    { source: "preview", tool: "window.close" },
  ],
];

export function generateSyntheticMovementTrajectories(params: SyntheticMovementParams): TrajectorySpan[] {
  const random = createLcg(params.seed);
  const variability = Math.min(1, Math.max(0, params.variability ?? 0));
  const trajectories: TrajectorySpan[] = [];

  for (let i = 0; i < params.count; i += 1) {
    const template = WORKFLOW_TEMPLATES[Math.floor(random() * WORKFLOW_TEMPLATES.length)]!;
    const observations: TrajectoryObservation[] = [];
    const actions: TrajectoryAction[] = [];
    let ts = 0;

    for (const beat of template) {
      // Perturb: occasionally drop a beat to shift the global sequence order.
      if (random() < variability) {
        continue;
      }
      observations.push({ kind: "observation", source: beat.source, summary: `see ${beat.source}`, ts });
      ts += 1;
      actions.push({ kind: "action", tool: beat.tool, summary: `do ${beat.tool}`, ts });
      ts += 1;
      // Perturb: occasionally duplicate the action (double-tap style).
      if (random() < variability) {
        actions.push({ kind: "action", tool: beat.tool, summary: `repeat ${beat.tool}`, ts });
        ts += 1;
      }
    }

    trajectories.push({
      id: `synthetic-${params.seed}-${i}`,
      sessionId: params.sessionId ?? `synthetic-session-${params.seed}`,
      createdAt: new Date(0).toISOString(),
      captureTier: "full",
      observations,
      actions,
      outcome: { status: "success", summary: "synthetic movement", reward: 1 },
    });
  }

  return trajectories;
}
