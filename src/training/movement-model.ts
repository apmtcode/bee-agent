import type { TrajectorySpan } from "../capture/trajectory.js";

/**
 * Pluggable local movement-model subsystem (standing objective #2, parts c & d).
 *
 * This module turns reviewed trajectories into a trainable {@link MovementDataset}
 * and defines a backend-agnostic seam for post-training a *local* model that can
 * (a) repeat the recorded movements and (b) generalize to new-but-related ones.
 *
 * The real on-device backend (a small MLX/axolotl model — see
 * {@link LocalAppleSiliconTrainingRunner}) executes only when the user runs
 * bee-agent locally. So that the cloud/CI build can validate the *pipeline*
 * end-to-end without any OS access or heavy dependency, this file ships a fully
 * deterministic reference backend, {@link NGramMovementBackend}: a Markov policy
 * with stupid-back-off that learns movement transitions in-process. Same dataset
 * in -> byte-identical model out, so tests are hermetic.
 *
 * Everything here is pure (no clock, no randomness, no I/O) and additive.
 */

/** A single recorded movement/action, reduced to its trainable essentials. */
export type MovementToken = {
  /** The tool / input channel that produced the movement (e.g. "mouse", "keyboard"). */
  tool: string;
  /** A concrete, replayable description of the movement. */
  summary: string;
};

/** An ordered movement sequence derived from one trajectory. */
export type MovementSequence = {
  trajectoryId: string;
  tokens: MovementToken[];
};

/** The trainable dataset: a versioned collection of movement sequences. */
export type MovementDataset = {
  version: 1;
  sequences: MovementSequence[];
};

/** A predicted next movement, with the back-off level and confidence used. */
export type MovementPrediction = {
  tool: string;
  summary: string;
  /** Fraction of the matched context bucket that chose this tool (0..1). */
  confidence: number;
  /** N-gram order actually used after back-off (0 = unigram/prior). */
  order: number;
};

export type MovementTrainingOptions = {
  /** Maximum context length the model conditions on. Defaults to 2. */
  order?: number;
};

/** Serialized form of a trained model — deterministic and JSON-safe. */
export type SerializedMovementModel = {
  version: 1;
  backend: string;
  order: number;
  /** One row per observed (context -> next) transition, sorted deterministically. */
  transitions: SerializedTransition[];
};

export type SerializedTransition = {
  context: string[];
  next: string;
  count: number;
  /** Representative replayable summary for this transition. */
  summary: string;
};

/** A trained, queryable movement model. */
export interface TrainedMovementModel {
  readonly backend: string;
  readonly order: number;
  /** Predict the next movement given a context prefix, or undefined at a natural stop. */
  predict(context: MovementToken[]): MovementPrediction | undefined;
  /** Autoregressively continue from a seed for up to `steps` movements. */
  rollout(seed: MovementToken[], steps: number): MovementToken[];
  /** Serialize to a deterministic, reloadable representation. */
  serialize(): SerializedMovementModel;
}

/** A pluggable training backend. Swap this for a real on-device model. */
export interface MovementModelBackend {
  readonly name: string;
  train(dataset: MovementDataset, options?: MovementTrainingOptions): TrainedMovementModel;
  load(serialized: SerializedMovementModel): TrainedMovementModel;
}

/** Sentinel emitted after the last token of a sequence so the model learns to stop. */
const END_TOKEN = "\u0000END";
const DEFAULT_ORDER = 2;

/**
 * Build a trainable dataset from reviewed trajectories.
 *
 * Uses redacted (reviewer-approved) actions when present, else the raw actions,
 * ordered by timestamp. Trajectories with no actions are dropped.
 */
export function buildMovementDataset(trajectories: TrajectorySpan[]): MovementDataset {
  const sequences: MovementSequence[] = [];
  for (const trajectory of trajectories) {
    const reviewedActions = trajectory.review?.redactedActions;
    const rawActions = reviewedActions
      ? reviewedActions.map((action) => ({ tool: action.tool, summary: action.summary, ts: action.ts }))
      : trajectory.actions.map((action) => ({ tool: action.tool, summary: action.summary, ts: action.ts }));
    const tokens = [...rawActions]
      .sort((a, b) => a.ts - b.ts)
      .map<MovementToken>((action) => ({ tool: action.tool, summary: action.summary }));
    if (tokens.length > 0) {
      sequences.push({ trajectoryId: trajectory.id, tokens });
    }
  }
  return { version: 1, sequences };
}

type TransitionBucket = Map<string, { count: number; summaries: Map<string, number> }>;

/**
 * Deterministic n-gram movement backend with stupid back-off.
 *
 * Learns, for every context of length 0..order, the frequency of each next tool
 * and its most representative summary. Prediction tries the longest matching
 * context and backs off to shorter ones, so an unseen-but-related prefix still
 * yields a plausible movement (generalization). All tie-breaks are lexical, so
 * training is a pure function of the dataset.
 */
export class NGramMovementBackend implements MovementModelBackend {
  readonly name = "ngram";

  train(dataset: MovementDataset, options?: MovementTrainingOptions): TrainedMovementModel {
    const order = Math.max(0, Math.floor(options?.order ?? DEFAULT_ORDER));
    const transitions: SerializedTransition[] = [];
    const buckets = new Map<string, TransitionBucket>();

    const record = (context: string[], next: string, summary: string): void => {
      const key = contextKey(context);
      let bucket = buckets.get(key);
      if (!bucket) {
        bucket = new Map();
        buckets.set(key, bucket);
      }
      let entry = bucket.get(next);
      if (!entry) {
        entry = { count: 0, summaries: new Map() };
        bucket.set(next, entry);
      }
      entry.count += 1;
      entry.summaries.set(summary, (entry.summaries.get(summary) ?? 0) + 1);
    };

    for (const sequence of dataset.sequences) {
      const tools = sequence.tokens.map((token) => token.tool);
      const summaries = sequence.tokens.map((token) => token.summary);
      // A trailing END lets the model learn where sequences terminate.
      const extendedTools = [...tools, END_TOKEN];
      const extendedSummaries = [...summaries, ""];
      for (let position = 0; position < extendedTools.length; position += 1) {
        const next = extendedTools[position] ?? END_TOKEN;
        const summary = extendedSummaries[position] ?? "";
        for (let contextLength = 0; contextLength <= order; contextLength += 1) {
          const start = position - contextLength;
          if (start < 0) {
            break;
          }
          const context = extendedTools.slice(start, position);
          record(context, next, summary);
        }
      }
    }

    for (const [key, bucket] of buckets) {
      const context = decodeContextKey(key);
      for (const [next, entry] of bucket) {
        transitions.push({ context, next, count: entry.count, summary: pickSummary(entry.summaries) });
      }
    }
    transitions.sort(compareTransitions);

    return new NGramMovementModel(order, transitions);
  }

  load(serialized: SerializedMovementModel): TrainedMovementModel {
    const transitions = [...serialized.transitions].sort(compareTransitions);
    return new NGramMovementModel(serialized.order, transitions);
  }
}

class NGramMovementModel implements TrainedMovementModel {
  readonly backend = "ngram";
  private readonly index: Map<string, SerializedTransition[]>;

  constructor(
    readonly order: number,
    private readonly transitions: SerializedTransition[],
  ) {
    this.index = new Map();
    for (const transition of transitions) {
      const key = contextKey(transition.context);
      const list = this.index.get(key);
      if (list) {
        list.push(transition);
      } else {
        this.index.set(key, [transition]);
      }
    }
  }

  predict(context: MovementToken[]): MovementPrediction | undefined {
    const tools = context.map((token) => token.tool);
    for (let contextLength = Math.min(this.order, tools.length); contextLength >= 0; contextLength -= 1) {
      const suffix = tools.slice(tools.length - contextLength);
      const candidates = this.index.get(contextKey(suffix));
      if (!candidates || candidates.length === 0) {
        continue;
      }
      const total = candidates.reduce((sum, candidate) => sum + candidate.count, 0);
      const realCandidates = candidates.filter((candidate) => candidate.next !== END_TOKEN);
      if (realCandidates.length === 0) {
        // Only termination was ever seen for this context.
        return undefined;
      }
      const best = pickBestTransition(realCandidates);
      const endCount = candidates.find((candidate) => candidate.next === END_TOKEN)?.count ?? 0;
      if (endCount > best.count) {
        // Termination strictly dominates the best real continuation: stop.
        return undefined;
      }
      return {
        tool: best.next,
        summary: best.summary,
        confidence: total > 0 ? best.count / total : 0,
        order: contextLength,
      };
    }
    return undefined;
  }

  rollout(seed: MovementToken[], steps: number): MovementToken[] {
    const produced: MovementToken[] = [];
    let context = [...seed];
    for (let step = 0; step < steps; step += 1) {
      const prediction = this.predict(context);
      if (!prediction) {
        break;
      }
      const token: MovementToken = { tool: prediction.tool, summary: prediction.summary };
      produced.push(token);
      context = [...context, token];
    }
    return produced;
  }

  serialize(): SerializedMovementModel {
    return {
      version: 1,
      backend: this.backend,
      order: this.order,
      transitions: this.transitions.map((transition) => ({
        context: [...transition.context],
        next: transition.next,
        count: transition.count,
        summary: transition.summary,
      })),
    };
  }
}

/**
 * Measure how well a trained model reproduces / generalizes to held-out
 * movement sequences (the generalization eval harness). For each held-out
 * sequence it feeds every growing prefix and compares the predicted next tool
 * against the ground truth, producing token-level accuracy plus exact-sequence
 * reproduction via rollout from the first token.
 */
export function evaluateMovementModel(
  model: TrainedMovementModel,
  heldOut: MovementSequence[],
): MovementEvaluationReport {
  let totalPredictions = 0;
  let correctPredictions = 0;
  let exactSequences = 0;
  let evaluableSequences = 0;

  for (const sequence of heldOut) {
    if (sequence.tokens.length < 2) {
      continue;
    }
    evaluableSequences += 1;
    for (let position = 1; position < sequence.tokens.length; position += 1) {
      const prefix = sequence.tokens.slice(0, position);
      const expected = sequence.tokens[position];
      const prediction = model.predict(prefix);
      totalPredictions += 1;
      if (prediction && expected && prediction.tool === expected.tool) {
        correctPredictions += 1;
      }
    }
    const firstToken = sequence.tokens[0];
    if (firstToken) {
      const rolled = model.rollout([firstToken], sequence.tokens.length - 1);
      const expectedTail = sequence.tokens.slice(1).map((token) => token.tool);
      if (rolled.length === expectedTail.length && rolled.every((token, i) => token.tool === expectedTail[i])) {
        exactSequences += 1;
      }
    }
  }

  return {
    sequences: evaluableSequences,
    totalPredictions,
    correctPredictions,
    tokenAccuracy: totalPredictions > 0 ? correctPredictions / totalPredictions : 0,
    exactSequences,
    exactMatchRate: evaluableSequences > 0 ? exactSequences / evaluableSequences : 0,
  };
}

export type MovementEvaluationReport = {
  sequences: number;
  totalPredictions: number;
  correctPredictions: number;
  tokenAccuracy: number;
  exactSequences: number;
  exactMatchRate: number;
};

const CONTEXT_SEPARATOR = "\u0001";

function contextKey(context: string[]): string {
  return context.join(CONTEXT_SEPARATOR);
}

function decodeContextKey(key: string): string[] {
  return key === "" ? [] : key.split(CONTEXT_SEPARATOR);
}

function pickSummary(summaries: Map<string, number>): string {
  let bestSummary = "";
  let bestCount = -1;
  for (const [summary, count] of summaries) {
    if (count > bestCount || (count === bestCount && summary < bestSummary)) {
      bestSummary = summary;
      bestCount = count;
    }
  }
  return bestSummary;
}

function pickBestTransition(candidates: SerializedTransition[]): SerializedTransition {
  let best = candidates[0]!;
  for (const candidate of candidates) {
    if (candidate.count > best.count || (candidate.count === best.count && candidate.next < best.next)) {
      best = candidate;
    }
  }
  return best;
}

function compareTransitions(a: SerializedTransition, b: SerializedTransition): number {
  const contextA = contextKey(a.context);
  const contextB = contextKey(b.context);
  if (contextA !== contextB) {
    return contextA < contextB ? -1 : 1;
  }
  if (a.next !== b.next) {
    return a.next < b.next ? -1 : 1;
  }
  return 0;
}
