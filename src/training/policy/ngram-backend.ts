/**
 * Default in-process movement-policy backend: a deterministic n-gram model with a
 * token-overlap nearest-neighbour fallback for generalization.
 *
 * - Exact match: a context seen in training returns its most-frequent next action.
 * - Generalized: an unseen-but-related context is scored by how strongly each of
 *   its feature tokens predicted each action in training (idf-weighted), so the
 *   model performs new-but-related movements it never saw verbatim.
 * - Prior: with no token signal, falls back to the globally most-frequent action.
 *
 * Fully deterministic (no RNG, no clock) so cloud/CI runs are reproducible.
 */

import {
  actionKey,
  contextKey,
  parseActionKey,
  type MovementActionLabel,
  type MovementContext,
  type MovementDataset,
  type MovementModelBackend,
  type MovementPrediction,
  type MovementScoredAction,
  type MovementTrainOptions,
  type TrainedMovementModel,
} from "./model.js";

export const NGRAM_BACKEND_ID = "ngram-nn@1";

export type NgramModelState = {
  /** contextKey -> actionKey -> summed weight. */
  contextCounts: Record<string, Record<string, number>>;
  /** feature token -> actionKey -> summed weight (generalization signal). */
  tokenCounts: Record<string, Record<string, number>>;
  /** feature token -> total weight across all actions (for idf-style damping). */
  tokenTotals: Record<string, number>;
  /** actionKey -> total weight (prior distribution). */
  actionTotals: Record<string, number>;
  /** actionKey -> reconstructable label. */
  labels: Record<string, MovementActionLabel>;
  totalWeight: number;
};

function addWeight(table: Record<string, Record<string, number>>, key: string, actionK: string, weight: number): void {
  const row = (table[key] ??= {});
  row[actionK] = (row[actionK] ?? 0) + weight;
}

function argmax(row: Record<string, number>): MovementScoredAction[] {
  const total = Object.values(row).reduce((sum, value) => sum + value, 0);
  if (total <= 0) {
    return [];
  }
  return Object.entries(row)
    .map(([key, value]) => ({ key, confidence: value / total }))
    // Highest confidence first; deterministic lexical tie-break on the action key.
    .sort((a, b) => (b.confidence !== a.confidence ? b.confidence - a.confidence : a.key < b.key ? -1 : 1))
    .map(({ key, confidence }) => ({ action: parseActionKey(key), confidence }));
}

export class NgramMovementBackend implements MovementModelBackend<NgramModelState> {
  readonly id = NGRAM_BACKEND_ID;

  train(dataset: MovementDataset, options?: MovementTrainOptions): TrainedMovementModel<NgramModelState> {
    const state: NgramModelState = {
      contextCounts: {},
      tokenCounts: {},
      tokenTotals: {},
      actionTotals: {},
      labels: {},
      totalWeight: 0,
    };

    for (const example of dataset.examples) {
      const weight = example.weight ?? 1;
      if (!(weight > 0)) {
        continue;
      }
      const actionK = actionKey(example.action);
      state.labels[actionK] = example.action;
      state.actionTotals[actionK] = (state.actionTotals[actionK] ?? 0) + weight;
      state.totalWeight += weight;

      addWeight(state.contextCounts, contextKey(example.context), actionK, weight);

      for (const token of new Set(example.context.tokens)) {
        addWeight(state.tokenCounts, token, actionK, weight);
        state.tokenTotals[token] = (state.tokenTotals[token] ?? 0) + weight;
      }
    }

    return {
      version: 1,
      backend: this.id,
      trainedAt: options?.trainedAt ?? null,
      exampleCount: dataset.examples.length,
      actionVocabulary: [...dataset.actionVocabulary],
      state,
    };
  }

  predict(model: TrainedMovementModel<NgramModelState>, context: MovementContext): MovementPrediction {
    const { state } = model;

    const exactRow = state.contextCounts[contextKey(context)];
    if (exactRow) {
      const ranked = argmax(exactRow);
      if (ranked.length > 0) {
        return { action: ranked[0].action, confidence: ranked[0].confidence, method: "exact", alternatives: ranked };
      }
    }

    // Generalization: idf-weighted token-overlap vote. A token that predicts few
    // actions (low total weight) is more discriminative than a ubiquitous one.
    const scores: Record<string, number> = {};
    for (const token of new Set(context.tokens)) {
      const row = state.tokenCounts[token];
      const tokenTotal = state.tokenTotals[token];
      if (!row || !tokenTotal) {
        continue;
      }
      const idf = 1 / Math.log2(1 + tokenTotal);
      for (const [actionK, weight] of Object.entries(row)) {
        scores[actionK] = (scores[actionK] ?? 0) + (weight / tokenTotal) * idf;
      }
    }
    const generalized = argmax(scores);
    if (generalized.length > 0) {
      return {
        action: generalized[0].action,
        confidence: generalized[0].confidence,
        method: "generalized",
        alternatives: generalized,
      };
    }

    const prior = argmax(state.actionTotals);
    if (prior.length > 0) {
      return { action: prior[0].action, confidence: prior[0].confidence, method: "prior", alternatives: prior };
    }

    return { action: undefined, confidence: 0, method: "none", alternatives: [] };
  }

  serialize(model: TrainedMovementModel<NgramModelState>): string {
    return JSON.stringify(model);
  }

  deserialize(serialized: string): TrainedMovementModel<NgramModelState> {
    const parsed = JSON.parse(serialized) as TrainedMovementModel<NgramModelState>;
    if (parsed.backend !== this.id) {
      throw new Error(`model backend mismatch: expected ${this.id}, got ${parsed.backend}`);
    }
    return parsed;
  }
}

export type MovementRolloutStep = {
  step: number;
  context: MovementContext;
  prediction: MovementPrediction;
};

export type MovementRolloutOptions = {
  maxSteps: number;
  /** Stop once predicted confidence drops below this. Defaults to 0 (never). */
  minConfidence?: number;
  /**
   * Supply the next situation's base tokens given the last predicted action. When
   * omitted, the initial context's non-`prev:` tokens are reused (a static-scene
   * rollout), and only the `prev:` token advances — enough to replay a learned
   * action chain in tests without a live environment.
   */
  advance?: (previous: MovementActionLabel, step: number) => MovementContext | undefined;
};

/**
 * Autoregressively replay a learned movement chain: predict, feed the action back
 * as the next context's `prev:` token, repeat. Demonstrates piece (c/d) — repeating
 * and generalizing recorded movements — without touching a real machine.
 */
export function rolloutMovements(
  backend: MovementModelBackend<NgramModelState>,
  model: TrainedMovementModel<NgramModelState>,
  initialContext: MovementContext,
  options: MovementRolloutOptions,
): MovementRolloutStep[] {
  const minConfidence = options.minConfidence ?? 0;
  const baseTokens = initialContext.tokens.filter((token) => !token.startsWith("prev:"));
  const steps: MovementRolloutStep[] = [];
  let context = initialContext;

  for (let step = 0; step < options.maxSteps; step += 1) {
    const prediction = backend.predict(model, context);
    steps.push({ step, context, prediction });
    if (!prediction.action || prediction.confidence < minConfidence) {
      break;
    }

    const advanced = options.advance?.(prediction.action, step);
    if (advanced) {
      context = advanced;
    } else {
      context = { tokens: [...baseTokens, `prev:${actionKey(prediction.action)}`] };
    }
  }

  return steps;
}
