import {
  tokenizeMovementText,
  type MovementAction,
  type MovementContext,
  type MovementDataset,
  type MovementExample,
} from "./movement-dataset.js";

/**
 * Pluggable local-movement model backend.
 *
 * The real on-device path (MLX / axolotl, see `src/training/runner.ts`) trains
 * a small local model from a reviewed export. That path cannot run in the cloud
 * or CI, so the model layer is defined behind this interface and ships with a
 * deterministic in-process implementation ({@link MockMovementModelBackend})
 * that learns from a {@link MovementDataset} and serves predictions without any
 * native dependency. Swap in a real backend (same interface) for on-device use.
 */

export type MovementQuery = {
  app?: string;
  screen?: string;
  source?: string;
  summary: string;
};

export type MovementPrediction = {
  /** Predicted next movement. */
  action: MovementAction;
  /** Confidence in [0, 1]. */
  confidence: number;
  /**
   * True when the prediction was produced by generalizing from a related-but-
   * non-identical context rather than recalling an (almost) exact match.
   */
  generalized: boolean;
  /** Trajectory the winning example came from, for traceability. */
  matchedTrajectoryId?: string;
  /** Short explanation of how the prediction was derived. */
  rationale: string;
};

export type MovementTrainOptions = {
  /** Optional fixed timestamp so trained models are reproducible in tests. */
  trainedAt?: string;
};

export type TrainedMovementModelStats = {
  backendId: string;
  exampleCount: number;
  appCount: number;
  targetCount: number;
  trainedAt: string;
};

export interface TrainedMovementModel {
  readonly backendId: string;
  readonly stats: TrainedMovementModelStats;
  /** Predict the next movement for a fresh context. */
  predict(query: MovementQuery): MovementPrediction | undefined;
}

export interface MovementModelBackend {
  readonly id: string;
  train(dataset: MovementDataset, options?: MovementTrainOptions): Promise<TrainedMovementModel>;
}

type ScoredExample = {
  example: MovementExample;
  score: number;
};

function jaccard(a: string[], b: string[]): number {
  if (a.length === 0 && b.length === 0) {
    return 1;
  }
  const setB = new Set(b);
  let intersection = 0;
  for (const token of new Set(a)) {
    if (setB.has(token)) {
      intersection += 1;
    }
  }
  const union = new Set([...a, ...b]).size;
  return union === 0 ? 0 : intersection / union;
}

function queryTokens(query: MovementQuery): string[] {
  return tokenizeMovementText(query.summary, query.screen, query.app);
}

function contextTokens(context: MovementContext): string[] {
  return tokenizeMovementText(context.summary, context.screen, context.app);
}

/**
 * Deterministic, dependency-free movement model. It indexes training examples
 * and, at inference time, scores every example against the query using:
 *   - token Jaccard similarity over summary/screen/app text, plus
 *   - an app-match bonus (same application is a strong prior), plus
 *   - a screen-match bonus.
 *
 * The best-scoring example's action is returned. When no example shares the
 * query's exact context (best score below {@link exactThreshold}) the model
 * *generalizes*: it keeps the closest action's tool/gesture/direction but, if
 * the query names a target the model saw during training, substitutes that
 * target into the action — e.g. learning "tap Send" lets it predict "tap Reply"
 * for a related screen that mentions a known "Reply" element. Confidence is the
 * winning score, scaled down for generalized predictions.
 *
 * Everything is order-stable (ties break on trajectoryId then index) so the
 * same dataset + query always yields the same prediction.
 */
export class MockMovementModelBackend implements MovementModelBackend {
  readonly id = "mock-knn";

  constructor(
    private readonly appMatchBonus = 0.5,
    private readonly screenMatchBonus = 0.25,
    /**
     * A prediction is treated as "recalled" (not generalized) when its score
     * normalized against the best-possible score is at or above this fraction.
     * Below it, the context is only loosely related and we generalize.
     */
    private readonly recallThreshold = 0.6,
  ) {}

  async train(dataset: MovementDataset, options: MovementTrainOptions = {}): Promise<TrainedMovementModel> {
    const trainedAt = options.trainedAt ?? new Date().toISOString();
    const examples = [...dataset.examples].sort((a, b) => {
      if (a.trajectoryId !== b.trajectoryId) {
        return a.trajectoryId < b.trajectoryId ? -1 : 1;
      }
      return a.index - b.index;
    });
    const targets = [...dataset.targets];
    const stats: TrainedMovementModelStats = {
      backendId: this.id,
      exampleCount: examples.length,
      appCount: dataset.apps.length,
      targetCount: targets.length,
      trainedAt,
    };

    const score = (example: MovementExample, query: MovementQuery, qTokens: string[]): number => {
      let value = jaccard(qTokens, contextTokens(example.context));
      if (query.app && example.context.app && query.app.toLowerCase() === example.context.app.toLowerCase()) {
        value += this.appMatchBonus;
      }
      if (query.screen && example.context.screen && query.screen.toLowerCase() === example.context.screen.toLowerCase()) {
        value += this.screenMatchBonus;
      }
      return value;
    };

    const recallThreshold = this.recallThreshold;
    const maxScore = 1 + this.appMatchBonus + this.screenMatchBonus;

    return {
      backendId: this.id,
      stats,
      predict(query: MovementQuery): MovementPrediction | undefined {
        if (examples.length === 0) {
          return undefined;
        }
        const qTokens = queryTokens(query);
        let best: ScoredExample | undefined;
        for (const example of examples) {
          const value = score(example, query, qTokens);
          if (!best || value > best.score) {
            best = { example, score: value };
          }
        }
        if (!best) {
          return undefined;
        }

        const normalized = Math.min(1, best.score / maxScore);
        const generalized = normalized < recallThreshold;

        let action = best.example.action;
        let rationale = generalized
          ? `generalized from trajectory ${best.example.trajectoryId} (score ${best.score.toFixed(2)})`
          : `recalled trajectory ${best.example.trajectoryId} (score ${best.score.toFixed(2)})`;

        if (generalized && action.target) {
          // Slot transfer: if the query mentions a known target the model never
          // paired with this exact context, retarget the closest action to it.
          const mentioned = targets.find((target) => {
            if (target.toLowerCase() === action.target?.toLowerCase()) {
              return false;
            }
            const targetTokens = tokenizeMovementText(target);
            return targetTokens.length > 0 && targetTokens.every((token) => qTokens.includes(token));
          });
          if (mentioned) {
            action = {
              ...action,
              target: mentioned,
              summary: action.summary.replace(action.target, mentioned),
            };
            rationale += `; retargeted to "${mentioned}"`;
          }
        }

        return {
          action,
          confidence: Number((generalized ? normalized * 0.6 : normalized).toFixed(4)),
          generalized,
          matchedTrajectoryId: best.example.trajectoryId,
          rationale,
        };
      },
    };
  }
}
