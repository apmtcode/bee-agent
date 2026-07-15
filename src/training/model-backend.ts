import type { ReplayManifest, ReplayTimelineEvent } from "../capture/replay.js";
import type { TrajectorySpan } from "../capture/trajectory.js";

/**
 * In-process movement-model backends.
 *
 * The {@link LocalAppleSiliconTrainingRunner} emits launch scripts for real
 * on-device training (mlx / axolotl). Those cannot run in the cloud/CI, so this
 * module provides a *pluggable* backend seam plus a deterministic reference
 * backend that genuinely learns from recorded movements and predicts/generalises
 * next movements without any external dependency or real OS input.
 *
 * This is standing objective #2 (local-movement learning) parts (c) and (d):
 * post-train a model to repeat recorded movements and generalise to related
 * ones. Swap {@link MarkovMovementBackend} for a real small-model backend behind
 * the same interface when running locally.
 */

/** The atomic unit a movement model learns over. */
export type MovementToken = {
  /** Whether this step was something observed or an action the agent took. */
  kind: "observation" | "action";
  /** Discriminating label: the tool for actions, the source for observations. */
  label: string;
};

/** One recorded movement sequence (a single trajectory / replay). */
export type MovementSequence = {
  id: string;
  tokens: MovementToken[];
};

/** A dataset of recorded movement sequences to train on. */
export type MovementDataset = {
  version: 1;
  sequences: MovementSequence[];
};

export type MovementTrainingOptions = {
  /**
   * Markov context length. The model conditions the next token on up to this
   * many preceding tokens, backing off to shorter contexts when the full-order
   * context was never seen — which is what lets it generalise to new-but-related
   * sequences. Defaults to 2.
   */
  order?: number;
};

export type MovementPredictionCandidate = {
  token: MovementToken;
  probability: number;
};

export type MovementPrediction = {
  /** The most likely next token, or `undefined` if the context is unknown. */
  token: MovementToken | undefined;
  /** Probability mass on {@link token} (0..1). */
  confidence: number;
  /** The context length actually used after back-off (0 = unconditional). */
  backoffOrder: number;
  /** All candidates, most-probable first. */
  candidates: MovementPredictionCandidate[];
};

/** Serialised model state — persist and {@link MovementModelBackend.restore}. */
export type MovementModelSnapshot = {
  version: 1;
  backendId: string;
  order: number;
  /** context-key -> (token-key -> count). */
  transitions: Record<string, Record<string, number>>;
  /** token-key -> token, so snapshots are self-describing. */
  vocabulary: Record<string, MovementToken>;
};

/** A trained model: predict, roll out continuations, and serialise. */
export type TrainedMovementModel = {
  readonly backendId: string;
  readonly order: number;
  /** Predict the token that most likely follows `context`. */
  predictNext(context: MovementToken[]): MovementPrediction;
  /**
   * Greedily roll out a continuation from `seed` — the replay/generalisation
   * path. Stops at `maxSteps` or when the model has no confident next token.
   */
  generate(seed: MovementToken[], maxSteps: number): MovementToken[];
  serialize(): MovementModelSnapshot;
};

/** Pluggable backend seam. Real on-device backends implement the same shape. */
export type MovementModelBackend = {
  readonly id: string;
  train(dataset: MovementDataset, options?: MovementTrainingOptions): TrainedMovementModel;
  restore(snapshot: MovementModelSnapshot): TrainedMovementModel;
};

const DEFAULT_ORDER = 2;

export function tokenKey(token: MovementToken): string {
  return `${token.kind}:${token.label}`;
}

function contextKey(context: MovementToken[]): string {
  return context.map(tokenKey).join(">");
}

/**
 * Deterministic reference backend: an order-k Markov next-movement model with
 * stupid-backoff. Ties are broken by token key so predictions are stable across
 * runs — safe for cloud/CI. Genuinely learns transitions from recorded data and
 * generalises via back-off to shorter, previously-seen contexts.
 */
export class MarkovMovementBackend implements MovementModelBackend {
  readonly id = "markov-movement";

  train(dataset: MovementDataset, options: MovementTrainingOptions = {}): TrainedMovementModel {
    const order = normalizeOrder(options.order);
    const transitions: Map<string, Map<string, number>> = new Map();
    const vocabulary: Map<string, MovementToken> = new Map();

    for (const sequence of dataset.sequences) {
      const tokens = sequence.tokens;
      for (let index = 0; index < tokens.length; index += 1) {
        const next = tokens[index];
        vocabulary.set(tokenKey(next), next);
        // Record the transition at every context length 0..order (back-off table).
        for (let ctxLen = 0; ctxLen <= order; ctxLen += 1) {
          if (index - ctxLen < 0) {
            break;
          }
          const context = tokens.slice(index - ctxLen, index);
          const key = contextKey(context);
          const row = transitions.get(key) ?? new Map<string, number>();
          row.set(tokenKey(next), (row.get(tokenKey(next)) ?? 0) + 1);
          transitions.set(key, row);
        }
      }
    }

    return new MarkovTrainedModel(this.id, order, transitions, vocabulary);
  }

  restore(snapshot: MovementModelSnapshot): TrainedMovementModel {
    const transitions = new Map<string, Map<string, number>>();
    for (const [ctx, row] of Object.entries(snapshot.transitions)) {
      transitions.set(ctx, new Map(Object.entries(row)));
    }
    const vocabulary = new Map<string, MovementToken>(Object.entries(snapshot.vocabulary));
    return new MarkovTrainedModel(snapshot.backendId, snapshot.order, transitions, vocabulary);
  }
}

class MarkovTrainedModel implements TrainedMovementModel {
  constructor(
    readonly backendId: string,
    readonly order: number,
    private readonly transitions: Map<string, Map<string, number>>,
    private readonly vocabulary: Map<string, MovementToken>,
  ) {}

  predictNext(context: MovementToken[]): MovementPrediction {
    // Back off from the longest usable context down to the unconditional prior.
    for (let ctxLen = Math.min(this.order, context.length); ctxLen >= 0; ctxLen -= 1) {
      const key = contextKey(context.slice(context.length - ctxLen));
      const row = this.transitions.get(key);
      if (!row || row.size === 0) {
        continue;
      }
      const total = [...row.values()].reduce((sum, count) => sum + count, 0);
      const candidates = [...row.entries()]
        .map(([tk, count]) => ({ token: this.vocabulary.get(tk) as MovementToken, probability: count / total }))
        .sort((a, b) => {
          if (b.probability !== a.probability) {
            return b.probability - a.probability;
          }
          return tokenKey(a.token).localeCompare(tokenKey(b.token));
        });
      const best = candidates[0];
      return {
        token: best?.token,
        confidence: best?.probability ?? 0,
        backoffOrder: ctxLen,
        candidates,
      };
    }
    return { token: undefined, confidence: 0, backoffOrder: 0, candidates: [] };
  }

  generate(seed: MovementToken[], maxSteps: number): MovementToken[] {
    const generated: MovementToken[] = [];
    let context = [...seed];
    for (let step = 0; step < maxSteps; step += 1) {
      const prediction = this.predictNext(context);
      // No token, or the model had to fall back to the unconditional prior
      // (no contextual signal) — that is the end of a faithful continuation.
      if (!prediction.token || (prediction.backoffOrder === 0 && context.length > 0)) {
        break;
      }
      generated.push(prediction.token);
      context = [...context, prediction.token];
    }
    return generated;
  }

  serialize(): MovementModelSnapshot {
    const transitions: Record<string, Record<string, number>> = {};
    for (const [ctx, row] of this.transitions.entries()) {
      transitions[ctx] = Object.fromEntries(row.entries());
    }
    return {
      version: 1,
      backendId: this.backendId,
      order: this.order,
      transitions,
      vocabulary: Object.fromEntries(this.vocabulary.entries()),
    };
  }
}

function normalizeOrder(order: number | undefined): number {
  if (order === undefined || !Number.isFinite(order)) {
    return DEFAULT_ORDER;
  }
  return Math.max(0, Math.floor(order));
}

/** Registry of the in-process backends available in every environment. */
export function createDefaultMovementBackends(): Record<string, MovementModelBackend> {
  const markov = new MarkovMovementBackend();
  return { [markov.id]: markov };
}

// --- Dataset extraction from recorded movements -----------------------------

/** Derive an ordered movement sequence from a replay manifest timeline. */
export function movementSequenceFromReplay(manifest: ReplayManifest): MovementSequence {
  const tokens = manifest.events.flatMap<MovementToken>((event) => replayEventToTokens(event));
  return { id: manifest.sessionId, tokens };
}

function replayEventToTokens(event: ReplayTimelineEvent): MovementToken[] {
  switch (event.kind) {
    case "observation":
      return [{ kind: "observation", label: event.source }];
    case "action":
      return [{ kind: "action", label: event.tool }];
    case "transcript":
      // Transcript turns aren't movements; excluded from the movement model.
      return [];
  }
}

/** Derive a movement sequence from a trajectory span (obs + actions by time). */
export function movementSequenceFromTrajectory(span: TrajectorySpan): MovementSequence {
  const timed: Array<{ ts: number; token: MovementToken }> = [
    ...span.observations.map((observation) => ({
      ts: observation.ts,
      token: { kind: "observation" as const, label: observation.source },
    })),
    ...span.actions.map((action) => ({
      ts: action.ts,
      token: { kind: "action" as const, label: action.tool },
    })),
  ].sort((a, b) => a.ts - b.ts);
  return { id: span.id, tokens: timed.map((entry) => entry.token) };
}

/** Assemble a dataset from any mix of movement sequences. */
export function buildMovementDataset(sequences: MovementSequence[]): MovementDataset {
  return { version: 1, sequences };
}
