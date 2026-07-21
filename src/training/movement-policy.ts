import type { ReplayManifest, ReplayTimelineEvent } from "../capture/replay.js";
import type { TrajectorySpan } from "../capture/trajectory.js";

/**
 * In-process, on-device movement-policy learning.
 *
 * The training {@link LocalAppleSiliconTrainingRunner} builds *external* launch
 * plans (mlx / axolotl) that only execute on the user's real machine. This
 * module fills the complementary gap needed by standing objective #2(c)+(d):
 * a small, pluggable, fully in-process model that can be *trained on the
 * recorded movement dataset to repeat the movements*, and *generalize to new
 * but related movements*. It runs deterministically in the cloud with no OS
 * access, so the capture → dataset → train → infer loop can be validated end to
 * end with synthetic event streams and unit tests.
 *
 * The backend is pluggable ({@link MovementPolicyBackend}): the default
 * {@link MarkovMovementBackend} is a deterministic order-k Markov model with
 * stupid-backoff, and a real on-device small model can be dropped in behind the
 * same interface without touching callers.
 */

/** A single discrete movement/action, tokenized for sequence modelling. */
export type MovementToken = string;

export type MovementSequence = {
  id: string;
  tokens: MovementToken[];
};

export type MovementDataset = {
  sequences: MovementSequence[];
};

export type MovementPrediction = {
  token: MovementToken;
  /** Probability mass assigned to {@link token} at the backoff order used. */
  confidence: number;
  /** Context order actually used to make the prediction (after backoff). */
  backoffOrder: number;
};

/**
 * Pluggable local-model seam. Real on-device backends (a distilled small model,
 * an MLX policy head, etc.) implement the same surface; the deterministic
 * Markov backend below is the reference/mock used for cloud tests.
 */
export type MovementPolicyBackend = {
  readonly name: string;
  /** (Re)train the policy from a movement dataset. Replaces prior state. */
  train(dataset: MovementDataset): void;
  /** Predict the most likely next token given a context prefix. */
  predictNext(context: MovementToken[]): MovementPrediction | undefined;
  /** Roll out a sequence of length `length` starting from `seed`. */
  generate(seed: MovementToken[], length: number): MovementToken[];
};

const TOKEN_STOP = "<stop>";

/**
 * Deterministic order-k Markov model with stupid-backoff.
 *
 * Training records, for every context of length 0..order, the distribution of
 * the token that followed it. Prediction uses the longest context seen during
 * training, backing off to shorter contexts (and finally the unigram
 * distribution) when a context is unseen — this is what lets the model
 * *generalize* to novel prefixes built from familiar local transitions.
 *
 * All tie-breaks are lexicographic so results are reproducible without any
 * randomness (no Date/Math.random), which the cloud sandbox forbids anyway.
 */
export class MarkovMovementBackend implements MovementPolicyBackend {
  readonly name = "markov";

  private readonly order: number;
  private readonly backoffFactor: number;
  /** context-string -> (nextToken -> count) */
  private readonly transitions = new Map<string, Map<MovementToken, number>>();

  constructor(options: { order?: number; backoffFactor?: number } = {}) {
    this.order = Math.max(1, Math.floor(options.order ?? 2));
    this.backoffFactor = options.backoffFactor ?? 0.4;
  }

  train(dataset: MovementDataset): void {
    this.transitions.clear();
    for (const sequence of dataset.sequences) {
      // A trailing stop lets the model learn where sequences end so rollouts
      // reproduce recorded movements exactly instead of running forever.
      const tokens = [...sequence.tokens, TOKEN_STOP];
      for (let i = 0; i < tokens.length; i += 1) {
        const next = tokens[i]!;
        const maxContext = Math.min(this.order, i);
        for (let k = 0; k <= maxContext; k += 1) {
          const context = tokens.slice(i - k, i);
          this.record(contextKey(context), next);
        }
      }
    }
  }

  predictNext(context: MovementToken[]): MovementPrediction | undefined {
    const scored = this.scoreNext(context);
    if (scored.size === 0) {
      return undefined;
    }
    let bestToken: MovementToken | undefined;
    let bestScore = -Infinity;
    for (const [token, score] of [...scored.entries()].sort((a, b) => (a[0] < b[0] ? -1 : 1))) {
      if (score > bestScore) {
        bestScore = score;
        bestToken = token;
      }
    }
    if (bestToken === undefined) {
      return undefined;
    }
    const { order } = this.longestContext(context);
    return { token: bestToken, confidence: bestScore, backoffOrder: order };
  }

  generate(seed: MovementToken[], length: number): MovementToken[] {
    const output: MovementToken[] = [];
    const context = [...seed];
    for (let i = 0; i < length; i += 1) {
      const prediction = this.predictNext(context);
      if (!prediction || prediction.token === TOKEN_STOP) {
        break;
      }
      output.push(prediction.token);
      context.push(prediction.token);
    }
    return output;
  }

  private record(key: string, token: MovementToken): void {
    let counts = this.transitions.get(key);
    if (!counts) {
      counts = new Map();
      this.transitions.set(key, counts);
    }
    counts.set(token, (counts.get(token) ?? 0) + 1);
  }

  /** Blend backoff orders into a single score map (stupid-backoff). */
  private scoreNext(context: MovementToken[]): Map<MovementToken, number> {
    const scores = new Map<MovementToken, number>();
    const maxOrder = Math.min(this.order, context.length);
    for (let k = maxOrder; k >= 0; k -= 1) {
      const suffix = context.slice(context.length - k, context.length);
      const counts = this.transitions.get(contextKey(suffix));
      if (!counts) {
        continue;
      }
      const total = [...counts.values()].reduce((sum, value) => sum + value, 0);
      if (total === 0) {
        continue;
      }
      const weight = Math.pow(this.backoffFactor, maxOrder - k);
      for (const [token, count] of counts) {
        scores.set(token, (scores.get(token) ?? 0) + weight * (count / total));
      }
    }
    return scores;
  }

  private longestContext(context: MovementToken[]): { order: number } {
    const maxOrder = Math.min(this.order, context.length);
    for (let k = maxOrder; k >= 0; k -= 1) {
      const suffix = context.slice(context.length - k, context.length);
      if (this.transitions.has(contextKey(suffix))) {
        return { order: k };
      }
    }
    return { order: 0 };
  }
}

/** No-op backend used to prove the seam is pluggable and honoured by callers. */
export class ConstantMovementBackend implements MovementPolicyBackend {
  readonly name = "constant";
  constructor(private readonly token: MovementToken) {}
  train(): void {
    /* stateless */
  }
  predictNext(): MovementPrediction {
    return { token: this.token, confidence: 1, backoffOrder: 0 };
  }
  generate(_seed: MovementToken[], length: number): MovementToken[] {
    return Array.from({ length }, () => this.token);
  }
}

function contextKey(context: MovementToken[]): string {
  return context.join("␟");
}

// ---------------------------------------------------------------------------
// Tokenization — turn recorded capture artifacts into movement token streams.
// ---------------------------------------------------------------------------

/** Compact, replay-stable token for one recorded action event. */
export function tokenizeAction(action: { tool: string; summary: string; metadata?: Record<string, unknown> }): MovementToken {
  const gesture = typeof action.metadata?.["gesture"] === "string" ? (action.metadata["gesture"] as string) : undefined;
  const direction = typeof action.metadata?.["direction"] === "string" ? (action.metadata["direction"] as string) : undefined;
  const target = typeof action.metadata?.["target"] === "string" ? (action.metadata["target"] as string) : undefined;
  const parts = [action.tool, gesture, direction, target].filter((part): part is string => Boolean(part));
  if (parts.length > 1) {
    return parts.map(slug).join(":");
  }
  return `${slug(action.tool)}:${slug(action.summary)}`;
}

/** Extract the ordered movement-token stream from a captured trajectory span. */
export function tokenizeTrajectory(span: TrajectorySpan): MovementSequence {
  const actions = [...span.actions].sort((a, b) => a.ts - b.ts);
  return { id: span.id, tokens: actions.map((action) => tokenizeAction(action)) };
}

/** Extract the movement-token stream from a replay manifest's action events. */
export function tokenizeReplayManifest(manifest: ReplayManifest): MovementSequence {
  const actions = manifest.events
    .filter((event): event is Extract<ReplayTimelineEvent, { kind: "action" }> => event.kind === "action")
    .sort((a, b) => a.ts - b.ts);
  const id = manifest.trajectoryIds[0] ?? manifest.sessionId;
  return { id, tokens: actions.map((action) => tokenizeAction(action)) };
}

/** Build a dataset from many captured spans. */
export function datasetFromTrajectories(spans: TrajectorySpan[]): MovementDataset {
  return { sequences: spans.map((span) => tokenizeTrajectory(span)).filter((sequence) => sequence.tokens.length > 0) };
}

function slug(value: string): string {
  return value
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "") || "x";
}

// ---------------------------------------------------------------------------
// Evaluation — replay fidelity + generalization on held-out sequences.
// ---------------------------------------------------------------------------

export type MovementEvalResult = {
  sequenceCount: number;
  totalPredictions: number;
  correct: number;
  /** Next-token accuracy across all held-out positions (0..1). */
  accuracy: number;
  perSequence: { id: string; predictions: number; correct: number; accuracy: number }[];
};

/**
 * Teacher-forced next-token accuracy over held-out sequences. On sequences the
 * model trained on this measures *replay fidelity*; on unseen-but-related
 * sequences it measures *generalization* (objective #2(d)).
 */
export function evaluateMovementPolicy(
  backend: MovementPolicyBackend,
  heldOut: MovementDataset,
  options: { contextOrder?: number } = {},
): MovementEvalResult {
  const order = Math.max(1, Math.floor(options.contextOrder ?? 2));
  let totalPredictions = 0;
  let totalCorrect = 0;
  const perSequence: MovementEvalResult["perSequence"] = [];
  for (const sequence of heldOut.sequences) {
    let predictions = 0;
    let correct = 0;
    for (let i = 1; i < sequence.tokens.length; i += 1) {
      const context = sequence.tokens.slice(Math.max(0, i - order), i);
      const prediction = backend.predictNext(context);
      predictions += 1;
      if (prediction?.token === sequence.tokens[i]) {
        correct += 1;
      }
    }
    totalPredictions += predictions;
    totalCorrect += correct;
    perSequence.push({
      id: sequence.id,
      predictions,
      correct,
      accuracy: predictions === 0 ? 0 : correct / predictions,
    });
  }
  return {
    sequenceCount: heldOut.sequences.length,
    totalPredictions,
    correct: totalCorrect,
    accuracy: totalPredictions === 0 ? 0 : totalCorrect / totalPredictions,
    perSequence,
  };
}

/** Exact-match replay fidelity: does a rollout from the seed reproduce it? */
export function measureReplayFidelity(
  backend: MovementPolicyBackend,
  sequence: MovementSequence,
  options: { seedLength?: number } = {},
): { expected: MovementToken[]; produced: MovementToken[]; matched: number; fidelity: number } {
  const seedLength = Math.min(options.seedLength ?? 1, sequence.tokens.length);
  const seed = sequence.tokens.slice(0, seedLength);
  const expected = sequence.tokens.slice(seedLength);
  const produced = backend.generate(seed, expected.length);
  let matched = 0;
  for (let i = 0; i < expected.length; i += 1) {
    if (produced[i] === expected[i]) {
      matched += 1;
    }
  }
  return { expected, produced, matched, fidelity: expected.length === 0 ? 1 : matched / expected.length };
}
