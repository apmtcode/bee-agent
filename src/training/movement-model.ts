import type { TrajectoryAction, TrajectorySpan } from "../capture/trajectory.js";

/**
 * In-process, cloud-testable movement-learning subsystem.
 *
 * Objective #2 parts (c) and (d): post-train a *local* model on a recorded
 * movement dataset so it can (c) repeat recorded movements and (d) generalize
 * to new-but-related movements. The real on-device training/inference will run
 * the {@link LocalAppleSiliconTrainingRunner} launch scripts; this module
 * provides the pluggable model abstraction plus a deterministic in-process
 * backend so the learning loop itself can be validated without real OS input or
 * GPU training.
 *
 * The model operates over *movement tokens* — canonical, target-independent
 * labels derived from captured actions (e.g. `device:tap`, `device:swipe:down`).
 * Learning the pattern independent of the concrete target is what lets a model
 * trained on one flow generalize to a related one.
 */

/** A single learnable movement, derived from a captured trajectory action. */
export type MovementStep = {
  /** Canonical, learnable token, e.g. `device:tap` or `device:swipe:down`. */
  token: string;
  /** Originating tool (`device`, `browser`, `keyboard`, ...). */
  tool: string;
  /** Human-readable label retained for replay. */
  summary: string;
  /** Event timestamp (ms epoch); steps are ordered by this. */
  ts: number;
  metadata?: Record<string, unknown>;
};

export type MovementSequence = {
  id: string;
  sessionId?: string;
  steps: MovementStep[];
};

export type MovementDataset = {
  version: 1;
  sequences: MovementSequence[];
};

/** A model prediction for the token that follows a given context. */
export type MovementPrediction = {
  /** Predicted next token, or `undefined` when the model is empty. */
  token: string | undefined;
  /** Probability assigned to the chosen token within its matched context. */
  confidence: number;
  /** Context order actually matched (0 = fell back to the unigram prior). */
  matchedOrder: number;
  /**
   * How many context tokens had to be dropped to find a known context.
   * `0` means the full context was seen verbatim during training; `> 0` means
   * the prediction generalized across a context the model never saw exactly.
   */
  backoff: number;
};

export type MovementModelArtifact = {
  version: 1;
  backend: string;
  order: number;
  vocabulary: string[];
  sequenceCount: number;
  stepCount: number;
  /** Joined-context -> token -> count, for every order from 1..order. */
  contexts: Record<string, Record<string, number>>;
  /** Distribution of the first token of a sequence. */
  start: Record<string, number>;
  /** Global token frequencies (the order-0 backoff prior). */
  unigram: Record<string, number>;
};

export type TrainMovementModelOptions = {
  /** Maximum context order (n-gram size − 1). Defaults to 3. */
  order?: number;
};

/**
 * Pluggable model backend. The deterministic {@link MarkovMovementBackend} ships
 * for cloud/CI; a real on-device small model can implement the same surface and
 * be swapped into {@link MovementModelTrainer} without touching callers.
 */
export interface MovementModelBackend {
  readonly name: string;
  train(dataset: MovementDataset, options?: TrainMovementModelOptions): MovementModelArtifact;
  predict(artifact: MovementModelArtifact, context: readonly string[]): MovementPrediction;
}

const CONTEXT_SEPARATOR = "";

/**
 * Variable-order Markov backend with stupid-backoff smoothing. It is a genuine
 * learned model (transition counts over observed movement contexts) yet fully
 * deterministic: ties are broken by count then lexical order, so the same
 * dataset always yields the same artifact and the same predictions. Backoff to
 * shorter contexts is what produces generalization to unseen sequences.
 */
export class MarkovMovementBackend implements MovementModelBackend {
  readonly name = "markov-backoff";

  train(dataset: MovementDataset, options: TrainMovementModelOptions = {}): MovementModelArtifact {
    const order = Math.max(1, Math.floor(options.order ?? 3));
    const contexts: Record<string, Record<string, number>> = {};
    const start: Record<string, number> = {};
    const unigram: Record<string, number> = {};
    const vocabulary = new Set<string>();
    let stepCount = 0;

    for (const sequence of dataset.sequences) {
      const tokens = sequence.steps.map((step) => step.token);
      if (tokens.length > 0) {
        start[tokens[0]!] = (start[tokens[0]!] ?? 0) + 1;
      }
      for (let i = 0; i < tokens.length; i += 1) {
        const token = tokens[i]!;
        vocabulary.add(token);
        unigram[token] = (unigram[token] ?? 0) + 1;
        stepCount += 1;
        for (let o = 1; o <= order && o <= i; o += 1) {
          const key = tokens.slice(i - o, i).join(CONTEXT_SEPARATOR);
          const bucket = (contexts[key] ??= {});
          bucket[token] = (bucket[token] ?? 0) + 1;
        }
      }
    }

    return {
      version: 1,
      backend: this.name,
      order,
      vocabulary: [...vocabulary].sort(),
      sequenceCount: dataset.sequences.length,
      stepCount,
      contexts,
      start,
      unigram,
    };
  }

  predict(artifact: MovementModelArtifact, context: readonly string[]): MovementPrediction {
    const requestedOrder = Math.min(artifact.order, context.length);

    for (let o = requestedOrder; o >= 1; o -= 1) {
      const key = context.slice(context.length - o).join(CONTEXT_SEPARATOR);
      const bucket = artifact.contexts[key];
      if (bucket) {
        const choice = argmax(bucket);
        if (choice) {
          return {
            token: choice.token,
            confidence: choice.count / choice.total,
            matchedOrder: o,
            backoff: requestedOrder - o,
          };
        }
      }
    }

    // Order-0 prior: the start distribution for an empty context, else the
    // global unigram. This is the maximum-generalization fallback.
    const prior = context.length === 0 && Object.keys(artifact.start).length > 0 ? artifact.start : artifact.unigram;
    const choice = argmax(prior);
    return {
      token: choice?.token,
      confidence: choice ? choice.count / choice.total : 0,
      matchedOrder: 0,
      backoff: requestedOrder,
    };
  }
}

function argmax(distribution: Record<string, number>): { token: string; count: number; total: number } | undefined {
  let best: { token: string; count: number } | undefined;
  let total = 0;
  for (const [token, count] of Object.entries(distribution)) {
    total += count;
    if (!best || count > best.count || (count === best.count && token < best.token)) {
      best = { token, count };
    }
  }
  return best ? { token: best.token, count: best.count, total } : undefined;
}

export type GeneratedMovement = {
  token: string;
  confidence: number;
  backoff: number;
};

export type MovementEvalReport = {
  sequenceCount: number;
  predictionCount: number;
  correct: number;
  /** Exact next-token accuracy over the held-out set. */
  accuracy: number;
  /** Correct predictions whose full context was seen verbatim in training. */
  exactContextHits: number;
  /** Correct predictions that required backoff — i.e. generalized to a context never seen exactly. */
  generalizedHits: number;
  /**
   * Share of correct predictions that came from generalization (backoff > 0).
   * High values mean the model is succeeding on contexts it never memorized.
   */
  generalizationRate: number;
};

/**
 * Ties a {@link MovementModelBackend} to the train -> repeat -> generalize ->
 * evaluate loop. Generation reproduces recorded movements from a seed context;
 * evaluation measures both raw fidelity and how much of that fidelity is
 * genuine generalization rather than memorization.
 */
export class MovementModelTrainer {
  constructor(private readonly backend: MovementModelBackend = new MarkovMovementBackend()) {}

  get backendName(): string {
    return this.backend.name;
  }

  train(dataset: MovementDataset, options?: TrainMovementModelOptions): MovementModelArtifact {
    return this.backend.train(dataset, options);
  }

  /** Greedily roll out the model from a seed context to produce a movement plan. */
  generate(
    artifact: MovementModelArtifact,
    seed: readonly string[],
    length: number,
  ): GeneratedMovement[] {
    const context: string[] = [...seed];
    const output: GeneratedMovement[] = [];
    for (let i = 0; i < length; i += 1) {
      const prediction = this.backend.predict(artifact, context);
      if (!prediction.token) {
        break;
      }
      output.push({ token: prediction.token, confidence: prediction.confidence, backoff: prediction.backoff });
      context.push(prediction.token);
      if (context.length > artifact.order) {
        context.shift();
      }
    }
    return output;
  }

  /**
   * Next-token evaluation over held-out sequences. For each position the prior
   * tokens form the context and the model predicts the following token; we then
   * compare against ground truth and attribute each hit to memorization vs
   * generalization via the prediction's backoff.
   */
  evaluate(artifact: MovementModelArtifact, heldOut: MovementDataset): MovementEvalReport {
    let predictionCount = 0;
    let correct = 0;
    let exactContextHits = 0;
    let generalizedHits = 0;

    for (const sequence of heldOut.sequences) {
      const tokens = sequence.steps.map((step) => step.token);
      for (let i = 0; i < tokens.length; i += 1) {
        const context = tokens.slice(0, i);
        const prediction = this.backend.predict(artifact, context);
        predictionCount += 1;
        if (prediction.token === tokens[i]) {
          correct += 1;
          if (prediction.backoff === 0 && prediction.matchedOrder > 0) {
            exactContextHits += 1;
          } else {
            generalizedHits += 1;
          }
        }
      }
    }

    return {
      sequenceCount: heldOut.sequences.length,
      predictionCount,
      correct,
      accuracy: predictionCount === 0 ? 0 : correct / predictionCount,
      exactContextHits,
      generalizedHits,
      generalizationRate: correct === 0 ? 0 : generalizedHits / correct,
    };
  }
}

/**
 * Derive the canonical learnable token for a captured action. The token is
 * intentionally target-independent (it omits the concrete UI target) so the
 * model learns the *movement pattern*; the specific target is preserved on the
 * step's `summary`/`metadata` for replay.
 */
export function tokenizeTrajectoryAction(action: TrajectoryAction): MovementStep {
  const gesture = typeof action.metadata?.gesture === "string" ? action.metadata.gesture : undefined;
  const direction = typeof action.metadata?.direction === "string" ? action.metadata.direction : undefined;
  const verb = gesture ?? firstWord(action.summary) ?? "act";
  const token = direction ? `${action.tool}:${verb}:${direction}` : `${action.tool}:${verb}`;
  return {
    token,
    tool: action.tool,
    summary: action.summary,
    ts: action.ts,
    ...(action.metadata ? { metadata: action.metadata } : {}),
  };
}

/** Build a {@link MovementDataset} from captured trajectory spans. */
export function buildMovementDataset(trajectories: readonly TrajectorySpan[]): MovementDataset {
  const sequences: MovementSequence[] = trajectories
    .map((trajectory) => ({
      id: trajectory.id,
      sessionId: trajectory.sessionId,
      steps: [...trajectory.actions].sort((a, b) => a.ts - b.ts).map(tokenizeTrajectoryAction),
    }))
    .filter((sequence) => sequence.steps.length > 0);
  return { version: 1, sequences };
}

function firstWord(value: string): string | undefined {
  const match = value.trim().toLowerCase().match(/[a-z0-9]+/);
  return match ? match[0] : undefined;
}
