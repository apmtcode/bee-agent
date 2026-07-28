import type { TrajectoryAction, TrajectorySpan } from "../capture/trajectory.js";

/**
 * Local-movement learning model.
 *
 * This module closes objectives 2(c) and 2(d) of the self-evolution charter in a
 * form that is fully runnable and testable in the cloud (no OS access required):
 *  - (c) post-train a local model on recorded movements so it can *repeat* them, and
 *  - (d) generalize to perform new-but-related movements.
 *
 * The learner is deterministic (no wall-clock / RNG) so training and inference are
 * reproducible in tests and CI. The heavy on-device training path stays in
 * `runner.ts` (it emits mlx/axolotl launch scripts); this is the in-process,
 * always-available counterpart. The `MovementModelBackend` interface is the seam:
 * a real small on-device model can implement it without changing call sites.
 */

/** A normalized, learnable unit of movement extracted from a recorded action. */
export type MovementToken = {
  /** Application the movement happened in (context, not part of the motion identity). */
  app: string;
  /** Recording tool that produced the action (e.g. "device", "browser", "os"). */
  tool: string;
  /** Gesture kind (tap/swipe/scroll/type/shortcut/...) or "none". */
  gesture: string;
  /** Normalized UI target (button/field label) or "". */
  target: string;
  /** Directional hint (up/down/left/right) or "". */
  direction: string;
};

/** An ordered sequence of movements captured within a single trajectory. */
export type MovementSequence = {
  trajectoryId: string;
  app: string;
  tokens: MovementToken[];
};

/** Training corpus: a set of movement sequences. */
export type MovementDataset = {
  sequences: MovementSequence[];
};

/** Context handed to a trained model when asking for the next movement. */
export type MovementContext = {
  app: string;
  history: MovementToken[];
};

/** How a prediction was resolved — the basis for distinguishing recall from generalization. */
export type MovementPredictionSource = "exact" | "generalized" | "prior" | "empty";

export type MovementPrediction = {
  /** Predicted next movement, or undefined when the model has learned nothing applicable. */
  token: MovementToken | undefined;
  /** Confidence in [0, 1] = share of probability mass on the chosen token within its tier. */
  confidence: number;
  source: MovementPredictionSource;
  /** Ranked alternatives (including the chosen token), most-likely first. */
  alternatives: Array<{ token: MovementToken; weight: number }>;
};

/** A model produced by training a backend on a {@link MovementDataset}. */
export interface TrainedMovementModel {
  readonly backendId: string;
  readonly trainedSequenceCount: number;
  readonly trainedTokenCount: number;
  predict(context: MovementContext): MovementPrediction;
}

/** Pluggable learner seam. Swap the deterministic backend for a real on-device model. */
export interface MovementModelBackend {
  readonly id: string;
  train(dataset: MovementDataset): Promise<TrainedMovementModel>;
}

/** Identity of the *motion* (app-independent) — this is what generalizes across apps. */
export function movementTokenKey(token: MovementToken): string {
  return `${token.tool}|${token.gesture}|${token.target}|${token.direction}`;
}

export function extractMovementToken(action: TrajectoryAction, app: string): MovementToken {
  const metadata = action.metadata ?? {};
  return {
    app,
    tool: action.tool,
    gesture: readString(metadata.gesture) || "none",
    target: normalizeTarget(readString(metadata.target)),
    direction: readString(metadata.direction),
  };
}

/** Build a training corpus from recorded trajectory spans. */
export function buildMovementDataset(trajectories: TrajectorySpan[]): MovementDataset {
  const sequences = trajectories
    .map<MovementSequence>((trajectory) => {
      const app = resolveTrajectoryApp(trajectory);
      const actions = [...trajectory.actions].sort((a, b) => a.ts - b.ts);
      return {
        trajectoryId: trajectory.id,
        app,
        tokens: actions.map((action) => extractMovementToken(action, app)),
      };
    })
    .filter((sequence) => sequence.tokens.length > 0);
  return { sequences };
}

type TokenCounts = Map<string, { token: MovementToken; count: number }>;

function bump(table: Map<string, TokenCounts>, key: string, token: MovementToken): void {
  let counts = table.get(key);
  if (!counts) {
    counts = new Map();
    table.set(key, counts);
  }
  const tokenKey = movementTokenKey(token);
  const existing = counts.get(tokenKey);
  if (existing) {
    existing.count += 1;
  } else {
    counts.set(tokenKey, { token, count: 1 });
  }
}

function pick(counts: TokenCounts, source: MovementPredictionSource): MovementPrediction {
  const entries = [...counts.values()].sort((a, b) => {
    if (a.count !== b.count) {
      return b.count - a.count;
    }
    return movementTokenKey(a.token).localeCompare(movementTokenKey(b.token));
  });
  const total = entries.reduce((sum, entry) => sum + entry.count, 0);
  const best = entries[0];
  return {
    token: best.token,
    confidence: total > 0 ? best.count / total : 0,
    source,
    alternatives: entries.slice(0, 5).map((entry) => ({
      token: entry.token,
      weight: total > 0 ? entry.count / total : 0,
    })),
  };
}

class DeterministicTrainedMovementModel implements TrainedMovementModel {
  constructor(
    readonly backendId: string,
    readonly trainedSequenceCount: number,
    readonly trainedTokenCount: number,
    private readonly byAppPrev: Map<string, TokenCounts>,
    private readonly byPrev: Map<string, TokenCounts>,
    private readonly byAppPrevGesture: Map<string, TokenCounts>,
    private readonly byPrevGesture: Map<string, TokenCounts>,
    private readonly startByApp: Map<string, TokenCounts>,
    private readonly startGlobal: TokenCounts,
    private readonly priorByApp: Map<string, TokenCounts>,
    private readonly priorGlobal: TokenCounts,
  ) {}

  predict(context: MovementContext): MovementPrediction {
    const previous = context.history[context.history.length - 1];
    const app = context.app;

    // Back-off tiers, most specific first. An exact hit means the model has *seen*
    // this app+context (recall); a hit on a broader tier means it is transferring a
    // learned motion to a new-but-related context (generalization).
    const tiers: Array<{ counts: TokenCounts | undefined; source: MovementPredictionSource }> = previous
      ? [
          { counts: this.byAppPrev.get(`${app}::${movementTokenKey(previous)}`), source: "exact" },
          { counts: this.byPrev.get(movementTokenKey(previous)), source: "generalized" },
          { counts: this.byAppPrevGesture.get(`${app}::${previous.gesture}`), source: "generalized" },
          { counts: this.byPrevGesture.get(previous.gesture), source: "generalized" },
        ]
      : [
          { counts: this.startByApp.get(app), source: "exact" },
          { counts: this.startGlobal.size ? this.startGlobal : undefined, source: "generalized" },
        ];

    for (const tier of tiers) {
      if (tier.counts && tier.counts.size > 0) {
        return pick(tier.counts, tier.source);
      }
    }

    const appPrior = this.priorByApp.get(app);
    if (appPrior && appPrior.size > 0) {
      return pick(appPrior, "prior");
    }
    if (this.priorGlobal.size > 0) {
      return pick(this.priorGlobal, "prior");
    }
    return { token: undefined, confidence: 0, source: "empty", alternatives: [] };
  }
}

/**
 * Deterministic back-off Markov backend. Learns next-movement distributions
 * conditioned on (app, previous token), with progressively looser fallbacks so it
 * can transfer a learned motion to an unseen app or an unseen target. Serves as the
 * always-available reference backend and the CI-safe stand-in for a real model.
 */
export class DeterministicMovementBackend implements MovementModelBackend {
  readonly id = "deterministic-backoff-markov@1";

  async train(dataset: MovementDataset): Promise<TrainedMovementModel> {
    const byAppPrev = new Map<string, TokenCounts>();
    const byPrev = new Map<string, TokenCounts>();
    const byAppPrevGesture = new Map<string, TokenCounts>();
    const byPrevGesture = new Map<string, TokenCounts>();
    const startByApp = new Map<string, TokenCounts>();
    const startGlobal: TokenCounts = new Map();
    const priorByApp = new Map<string, TokenCounts>();
    const priorGlobal: TokenCounts = new Map();

    let tokenCount = 0;
    for (const sequence of dataset.sequences) {
      const { tokens, app } = sequence;
      if (tokens.length === 0) {
        continue;
      }
      bump(startByApp, app, tokens[0]);
      bumpCounts(startGlobal, tokens[0]);
      for (let i = 0; i < tokens.length; i += 1) {
        const token = tokens[i];
        tokenCount += 1;
        bump(priorByApp, app, token);
        bumpCounts(priorGlobal, token);
        if (i > 0) {
          const previous = tokens[i - 1];
          const previousKey = movementTokenKey(previous);
          bump(byAppPrev, `${app}::${previousKey}`, token);
          bump(byPrev, previousKey, token);
          bump(byAppPrevGesture, `${app}::${previous.gesture}`, token);
          bump(byPrevGesture, previous.gesture, token);
        }
      }
    }

    return new DeterministicTrainedMovementModel(
      this.id,
      dataset.sequences.length,
      tokenCount,
      byAppPrev,
      byPrev,
      byAppPrevGesture,
      byPrevGesture,
      startByApp,
      startGlobal,
      priorByApp,
      priorGlobal,
    );
  }
}

function bumpCounts(counts: TokenCounts, token: MovementToken): void {
  const tokenKey = movementTokenKey(token);
  const existing = counts.get(tokenKey);
  if (existing) {
    existing.count += 1;
  } else {
    counts.set(tokenKey, { token, count: 1 });
  }
}

/** Convenience: train with the default deterministic backend (or a supplied one). */
export async function trainMovementModel(
  dataset: MovementDataset,
  backend: MovementModelBackend = new DeterministicMovementBackend(),
): Promise<TrainedMovementModel> {
  return await backend.train(dataset);
}

/** Greedy replay of a single sequence: seed with its first token, roll the model forward. */
export type MovementRolloutResult = {
  trajectoryId: string;
  expected: MovementToken[];
  predicted: MovementToken[];
  matchedSteps: number;
  totalSteps: number;
  fidelity: number;
};

export type MovementGeneralizationReport = {
  sequenceCount: number;
  /** Teacher-forced top-1 next-movement accuracy across every held-out step. */
  nextStepAccuracy: number;
  /** Accuracy restricted to predictions the model resolved via recall ("exact"). */
  exactNextStepAccuracy: number;
  /** Accuracy restricted to predictions resolved by transfer ("generalized"/"prior"). */
  generalizedNextStepAccuracy: number;
  /** Mean per-sequence greedy-rollout fidelity (position-wise match). */
  rolloutFidelity: number;
  rollouts: MovementRolloutResult[];
  bySource: Record<MovementPredictionSource, { total: number; correct: number }>;
};

/**
 * Measure how well a trained model reproduces held-out sequences. Feeding sequences
 * the model was NOT trained on quantifies generalization (objective 2d); feeding
 * training sequences quantifies recall (objective 2c).
 */
export function evaluateMovementGeneralization(
  model: TrainedMovementModel,
  heldOut: MovementDataset,
): MovementGeneralizationReport {
  const bySource: MovementGeneralizationReport["bySource"] = {
    exact: { total: 0, correct: 0 },
    generalized: { total: 0, correct: 0 },
    prior: { total: 0, correct: 0 },
    empty: { total: 0, correct: 0 },
  };
  let totalSteps = 0;
  let correctSteps = 0;
  const rollouts: MovementRolloutResult[] = [];

  for (const sequence of heldOut.sequences) {
    const { tokens, app } = sequence;

    // Teacher-forced next-step scoring.
    for (let i = 1; i < tokens.length; i += 1) {
      const prediction = model.predict({ app, history: tokens.slice(0, i) });
      const expected = tokens[i];
      const correct = prediction.token !== undefined && movementTokenKey(prediction.token) === movementTokenKey(expected);
      totalSteps += 1;
      if (correct) {
        correctSteps += 1;
      }
      bySource[prediction.source].total += 1;
      if (correct) {
        bySource[prediction.source].correct += 1;
      }
    }

    // Greedy rollout from the seed movement.
    const predicted: MovementToken[] = [];
    let matched = 0;
    if (tokens.length > 0) {
      const history: MovementToken[] = [tokens[0]];
      for (let i = 1; i < tokens.length; i += 1) {
        const prediction = model.predict({ app, history: [...history] });
        if (!prediction.token) {
          break;
        }
        predicted.push(prediction.token);
        if (movementTokenKey(prediction.token) === movementTokenKey(tokens[i])) {
          matched += 1;
        }
        history.push(prediction.token);
      }
    }
    const rolloutSteps = Math.max(0, tokens.length - 1);
    rollouts.push({
      trajectoryId: sequence.trajectoryId,
      expected: tokens,
      predicted,
      matchedSteps: matched,
      totalSteps: rolloutSteps,
      fidelity: rolloutSteps > 0 ? matched / rolloutSteps : 1,
    });
  }

  const generalizedTotal = bySource.generalized.total + bySource.prior.total;
  const generalizedCorrect = bySource.generalized.correct + bySource.prior.correct;
  const rolloutFidelity =
    rollouts.length > 0 ? rollouts.reduce((sum, rollout) => sum + rollout.fidelity, 0) / rollouts.length : 0;

  return {
    sequenceCount: heldOut.sequences.length,
    nextStepAccuracy: totalSteps > 0 ? correctSteps / totalSteps : 0,
    exactNextStepAccuracy: bySource.exact.total > 0 ? bySource.exact.correct / bySource.exact.total : 0,
    generalizedNextStepAccuracy: generalizedTotal > 0 ? generalizedCorrect / generalizedTotal : 0,
    rolloutFidelity,
    rollouts,
    bySource,
  };
}

function readString(value: unknown): string {
  return typeof value === "string" ? value : "";
}

function normalizeTarget(value: string): string {
  return value.trim().toLowerCase();
}

function resolveTrajectoryApp(trajectory: TrajectorySpan): string {
  for (const observation of trajectory.observations) {
    const metadata = observation.metadata ?? {};
    const appName = readString(metadata.appName) || readString(metadata.appId);
    if (appName) {
      return appName.trim().toLowerCase();
    }
  }
  return "unknown";
}
