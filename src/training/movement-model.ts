/**
 * Local-movement learning model — objective #2 (c) train + (d) generalize.
 *
 * This module turns captured movement trajectories into a small, deterministic,
 * on-device-trainable model that can (a) repeat recorded movement sequences
 * exactly (replay fidelity) and (b) generalize to new-but-related movements via
 * gesture-class backoff. The model is a count-based Markov backoff n-gram — no
 * GPU, no external service, fully deterministic — so it trains and runs in the
 * cloud/CI as well as on a user's machine.
 *
 * The backend is pluggable: {@link MovementModelBackend} is the seam. The
 * shipped {@link MarkovMovementBackend} is the default (and the only one that
 * runs without native deps). A real neural on-device backend (e.g. an MLX LoRA
 * adapter produced by the {@link LocalAppleSiliconTrainingRunner}) can implement
 * the same interface and be swapped in without touching callers.
 */

import type { TrajectoryAction, TrajectorySpan } from "../capture/trajectory.js";
import type { ExportedReplayManifest } from "./export-manifest.js";

/** A canonical, single-token representation of one movement/action. */
export type MovementToken = string;

/** Sentinel emitted at the end of a recorded movement sequence. */
export const MOVEMENT_END_TOKEN: MovementToken = "<end>";

/**
 * One training example: predict {@link token} given the preceding
 * {@link context} (most-recent token last). Carries the originating trajectory
 * for provenance and held-out splitting.
 */
export type MovementStep = {
  trajectoryId: string;
  context: MovementToken[];
  token: MovementToken;
};

export type MovementDataset = {
  version: 1;
  steps: MovementStep[];
  /** token -> gesture-class token, learned from the source actions. */
  classOf: Record<MovementToken, MovementToken>;
  vocabulary: MovementToken[];
};

export type MovementPredictionSource = "specific" | "generalized" | "prior";

export type MovementPrediction = {
  token: MovementToken;
  /** Estimated probability of {@link token} under the matched context. */
  confidence: number;
  /** How the prediction was reached — exact match, class backoff, or prior. */
  source: MovementPredictionSource;
  /** Context order (length) that produced the prediction. */
  order: number;
  alternatives: Array<{ token: MovementToken; confidence: number }>;
};

export type MovementTrainingConfig = {
  /** Maximum context length for the highest-order n-gram. */
  order: number;
};

export const DEFAULT_MOVEMENT_TRAINING_CONFIG: MovementTrainingConfig = { order: 3 };

/** Serializable trained artifact — write to disk, ship to device, reload. */
export type MovementModelArtifact = {
  version: 1;
  backend: string;
  order: number;
  vocabulary: MovementToken[];
  classOf: Record<MovementToken, MovementToken>;
  /** specific-context -> next specific-token counts. */
  specific: SerializedNgrams;
  /** class-mapped-context -> next specific-token counts (generalization). */
  classContext: SerializedNgrams;
  stats: {
    stepCount: number;
    trajectoryCount: number;
  };
};

type SerializedNgrams = Array<{ ctx: string; next: Array<[MovementToken, number]> }>;

/** A loaded, queryable movement model. */
export interface MovementModel {
  readonly artifact: MovementModelArtifact;
  /** Predict the next movement token given a recent context. */
  predictNext(context: MovementToken[]): MovementPrediction | undefined;
  /** Greedily roll out a movement sequence from a seed context. */
  generate(seed: MovementToken[], options?: { maxSteps?: number }): MovementToken[];
  /** Gesture-class of a token (or the token itself if unknown). */
  classOf(token: MovementToken): MovementToken;
}

/** Pluggable training/inference backend. Implement to swap learners. */
export interface MovementModelBackend {
  readonly id: string;
  train(dataset: MovementDataset, config?: MovementTrainingConfig): MovementModelArtifact;
  load(artifact: MovementModelArtifact): MovementModel;
}

// ---------------------------------------------------------------------------
// Tokenization
// ---------------------------------------------------------------------------

/**
 * Derive a gesture-class token purely from a token's shape, so that *unseen*
 * targets (not in the trained vocabulary) can still be generalized. Mirrors the
 * `tool/kind/slot` and `tool/slot` shapes produced by {@link tokenizeAction}.
 */
export function structuralClass(token: MovementToken): MovementToken {
  if (token === MOVEMENT_END_TOKEN) {
    return token;
  }
  const parts = token.split("/");
  if (parts.length >= 3) {
    return `${parts[0]}/${parts[1]}/*`;
  }
  if (parts.length === 2) {
    return `${parts[0]}/*`;
  }
  return token;
}

function normalizeSlot(value: string | undefined): string {
  if (!value) {
    return "";
  }
  return value
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
}

/** Read a string field from an action's metadata, if present. */
function metaString(action: TrajectoryAction, key: string): string | undefined {
  const value = action.metadata?.[key];
  return typeof value === "string" ? value : undefined;
}

/**
 * Canonical token + gesture-class for a trajectory action.
 *
 * - `specific` distinguishes the exact target, e.g. `device/tap/submit-button`.
 * - `class` collapses the target slot, e.g. `device/tap/*`, so that tapping a
 *   *different* button is recognized as the same kind of movement. The class is
 *   what enables generalization to new-but-related actions.
 */
export function tokenizeAction(action: TrajectoryAction): { specific: MovementToken; class: MovementToken } {
  const tool = normalizeSlot(action.tool) || "tool";
  const gesture = metaString(action, "gesture");
  if (gesture) {
    const kind = normalizeSlot(gesture) || "gesture";
    const slot =
      normalizeSlot(metaString(action, "target")) ||
      normalizeSlot(metaString(action, "direction")) ||
      normalizeSlot(metaString(action, "valueSummary"));
    const classToken = `${tool}/${kind}/*`;
    return {
      specific: slot ? `${tool}/${kind}/${slot}` : classToken,
      class: classToken,
    };
  }
  const slot = normalizeSlot(action.summary) || "action";
  return { specific: `${tool}/${slot}`, class: `${tool}/*` };
}

// ---------------------------------------------------------------------------
// Dataset construction
// ---------------------------------------------------------------------------

type TokenizedSequence = {
  trajectoryId: string;
  specifics: MovementToken[];
  classes: MovementToken[];
};

function tokenizeTrajectory(trajectory: TrajectorySpan): TokenizedSequence {
  const ordered = [...trajectory.actions].sort((a, b) => a.ts - b.ts);
  const specifics: MovementToken[] = [];
  const classes: MovementToken[] = [];
  for (const action of ordered) {
    const { specific, class: classToken } = tokenizeAction(action);
    specifics.push(specific);
    classes.push(classToken);
  }
  return { trajectoryId: trajectory.id, specifics, classes };
}

function sequencesToDataset(sequences: TokenizedSequence[], order: number): MovementDataset {
  const steps: MovementStep[] = [];
  const classOf: Record<MovementToken, MovementToken> = {};
  const vocabulary = new Set<MovementToken>();

  for (const sequence of sequences) {
    const tokens = [...sequence.specifics, MOVEMENT_END_TOKEN];
    const classTokens = [...sequence.classes, MOVEMENT_END_TOKEN];
    for (let i = 0; i < tokens.length; i += 1) {
      const token = tokens[i]!;
      classOf[token] = classTokens[i]!;
      vocabulary.add(token);
      const start = Math.max(0, i - order);
      steps.push({
        trajectoryId: sequence.trajectoryId,
        context: tokens.slice(start, i),
        token,
      });
    }
  }
  classOf[MOVEMENT_END_TOKEN] = MOVEMENT_END_TOKEN;
  vocabulary.add(MOVEMENT_END_TOKEN);

  return { version: 1, steps, classOf, vocabulary: [...vocabulary].sort() };
}

/** Build a movement dataset from reviewed/captured trajectory spans. */
export function buildMovementDataset(
  trajectories: TrajectorySpan[],
  config: MovementTrainingConfig = DEFAULT_MOVEMENT_TRAINING_CONFIG,
): MovementDataset {
  const sequences = trajectories
    .map(tokenizeTrajectory)
    .filter((sequence) => sequence.specifics.length > 0);
  return sequencesToDataset(sequences, config.order);
}

/**
 * Build a dataset from exported replay manifests (the privacy-reviewed format
 * the trainer actually ships). Action events become movement tokens; the
 * gesture class is recovered from the summary verb.
 */
export function buildMovementDatasetFromReplays(
  replays: ExportedReplayManifest[],
  config: MovementTrainingConfig = DEFAULT_MOVEMENT_TRAINING_CONFIG,
): MovementDataset {
  const byTrajectory = new Map<string, { ts: number; specific: MovementToken; class: MovementToken }[]>();
  for (const replay of replays) {
    for (const event of replay.events) {
      if (event.kind !== "action") {
        continue;
      }
      const tool = normalizeSlot(event.tool) || "tool";
      const verb = normalizeSlot(event.summary.split(/\s+/)[0]);
      const slot = normalizeSlot(event.summary.split(/\s+/).slice(1).join("-"));
      const classToken = verb ? `${tool}/${verb}/*` : `${tool}/*`;
      const specific = slot ? `${tool}/${verb}/${slot}` : classToken;
      const list = byTrajectory.get(event.trajectoryId) ?? [];
      list.push({ ts: event.ts, specific, class: classToken });
      byTrajectory.set(event.trajectoryId, list);
    }
  }
  const sequences: TokenizedSequence[] = [...byTrajectory.entries()].map(([trajectoryId, events]) => {
    const ordered = events.sort((a, b) => a.ts - b.ts);
    return {
      trajectoryId,
      specifics: ordered.map((event) => event.specific),
      classes: ordered.map((event) => event.class),
    };
  });
  return sequencesToDataset(sequences, config.order);
}

// ---------------------------------------------------------------------------
// Markov backoff backend
// ---------------------------------------------------------------------------

type CountTable = Map<string, Map<MovementToken, number>>;

function bump(table: CountTable, ctxKey: string, token: MovementToken): void {
  const row = table.get(ctxKey) ?? new Map<MovementToken, number>();
  row.set(token, (row.get(token) ?? 0) + 1);
  table.set(ctxKey, row);
}

function contextKey(order: number, tokens: MovementToken[]): string {
  return `${order} ${tokens.join("")}`;
}

function serialize(table: CountTable): SerializedNgrams {
  return [...table.entries()]
    .map(([ctx, row]) => ({
      ctx,
      next: [...row.entries()].sort((a, b) => (b[1] - a[1]) || a[0].localeCompare(b[0])),
    }))
    .sort((a, b) => a.ctx.localeCompare(b.ctx));
}

function deserialize(ngrams: SerializedNgrams): CountTable {
  const table: CountTable = new Map();
  for (const entry of ngrams) {
    table.set(entry.ctx, new Map(entry.next));
  }
  return table;
}

/** Default backend: deterministic count-based Markov model with class backoff. */
export class MarkovMovementBackend implements MovementModelBackend {
  readonly id = "markov";

  train(dataset: MovementDataset, config: MovementTrainingConfig = DEFAULT_MOVEMENT_TRAINING_CONFIG): MovementModelArtifact {
    const order = Math.max(0, config.order);
    const specific: CountTable = new Map();
    const classContext: CountTable = new Map();
    const trajectories = new Set<string>();
    const classOf = { ...dataset.classOf };

    const mapClass = (token: MovementToken): MovementToken => classOf[token] ?? structuralClass(token);

    for (const step of dataset.steps) {
      trajectories.add(step.trajectoryId);
      const ctx = step.context;
      for (let k = Math.min(order, ctx.length); k >= 0; k -= 1) {
        const tail = ctx.slice(ctx.length - k);
        bump(specific, contextKey(k, tail), step.token);
        bump(classContext, contextKey(k, tail.map(mapClass)), step.token);
      }
    }

    return {
      version: 1,
      backend: this.id,
      order,
      vocabulary: [...dataset.vocabulary],
      classOf,
      specific: serialize(specific),
      classContext: serialize(classContext),
      stats: { stepCount: dataset.steps.length, trajectoryCount: trajectories.size },
    };
  }

  load(artifact: MovementModelArtifact): MovementModel {
    return new MarkovMovementModel(artifact);
  }
}

function argmax(row: Map<MovementToken, number> | undefined): {
  token: MovementToken;
  confidence: number;
  alternatives: Array<{ token: MovementToken; confidence: number }>;
} | undefined {
  if (!row || row.size === 0) {
    return undefined;
  }
  let total = 0;
  for (const count of row.values()) {
    total += count;
  }
  const ranked = [...row.entries()]
    .map(([token, count]) => ({ token, confidence: count / total }))
    .sort((a, b) => (b.confidence - a.confidence) || a.token.localeCompare(b.token));
  const best = ranked[0]!;
  return { token: best.token, confidence: best.confidence, alternatives: ranked.slice(1, 4) };
}

class MarkovMovementModel implements MovementModel {
  private readonly specific: CountTable;
  private readonly classContext: CountTable;

  constructor(public readonly artifact: MovementModelArtifact) {
    this.specific = deserialize(artifact.specific);
    this.classContext = deserialize(artifact.classContext);
  }

  classOf(token: MovementToken): MovementToken {
    return this.artifact.classOf[token] ?? structuralClass(token);
  }

  predictNext(context: MovementToken[]): MovementPrediction | undefined {
    const order = this.artifact.order;
    // 1) Exact-context match — highest fidelity, reproduces recorded movements.
    for (let k = Math.min(order, context.length); k >= 1; k -= 1) {
      const tail = context.slice(context.length - k);
      const best = argmax(this.specific.get(contextKey(k, tail)));
      if (best) {
        return { ...best, source: "specific", order: k };
      }
    }
    // 2) Class-mapped context — generalize to new-but-related movements.
    for (let k = Math.min(order, context.length); k >= 1; k -= 1) {
      const tail = context.slice(context.length - k).map((token) => this.classOf(token));
      const best = argmax(this.classContext.get(contextKey(k, tail)));
      if (best) {
        return { ...best, source: "generalized", order: k };
      }
    }
    // 3) Unigram prior — last resort.
    const prior = argmax(this.specific.get(contextKey(0, [])));
    if (prior) {
      return { ...prior, source: "prior", order: 0 };
    }
    return undefined;
  }

  generate(seed: MovementToken[], options: { maxSteps?: number } = {}): MovementToken[] {
    const maxSteps = options.maxSteps ?? 64;
    const produced: MovementToken[] = [];
    let context = [...seed];
    for (let i = 0; i < maxSteps; i += 1) {
      const prediction = this.predictNext(context);
      if (!prediction || prediction.token === MOVEMENT_END_TOKEN) {
        break;
      }
      produced.push(prediction.token);
      context = [...context, prediction.token];
    }
    return produced;
  }
}

// ---------------------------------------------------------------------------
// Convenience + evaluation
// ---------------------------------------------------------------------------

const DEFAULT_BACKEND = new MarkovMovementBackend();

/** Train a movement model from trajectory spans using the default backend. */
export function trainMovementModel(
  trajectories: TrajectorySpan[],
  config: MovementTrainingConfig = DEFAULT_MOVEMENT_TRAINING_CONFIG,
  backend: MovementModelBackend = DEFAULT_BACKEND,
): MovementModel {
  const dataset = buildMovementDataset(trajectories, config);
  return backend.load(backend.train(dataset, config));
}

export type MovementEvaluation = {
  total: number;
  /** Exact next-token match rate (replay fidelity). */
  top1Accuracy: number;
  /** Gesture-class match rate (generalization quality). */
  classAccuracy: number;
  /** Share of predictions that came from class backoff rather than exact match. */
  generalizationRate: number;
};

/**
 * Evaluate a model against held-out steps. `top1Accuracy` measures faithful
 * replay; `classAccuracy` measures whether the model at least picked the right
 * *kind* of movement — the signal for generalization to unseen targets.
 */
export function evaluateMovementModel(model: MovementModel, heldOut: MovementStep[]): MovementEvaluation {
  let exact = 0;
  let classHit = 0;
  let generalized = 0;
  let scored = 0;
  for (const step of heldOut) {
    const prediction = model.predictNext(step.context);
    if (!prediction) {
      continue;
    }
    scored += 1;
    if (prediction.token === step.token) {
      exact += 1;
    }
    if (model.classOf(prediction.token) === model.classOf(step.token)) {
      classHit += 1;
    }
    if (prediction.source === "generalized") {
      generalized += 1;
    }
  }
  return {
    total: scored,
    top1Accuracy: scored === 0 ? 0 : exact / scored,
    classAccuracy: scored === 0 ? 0 : classHit / scored,
    generalizationRate: scored === 0 ? 0 : generalized / scored,
  };
}
