import type { ReplayManifest } from "../capture/replay.js";
import type { TrajectoryAction, TrajectorySpan } from "../capture/trajectory.js";

/**
 * In-repo, pluggable local-movement model.
 *
 * Standing objective #2 asks bee-agent to (c) post-train a *local* model on a
 * recorded movement dataset so it can repeat the recorded movements, and (d)
 * generalize to new-but-related movements. The existing `runner.ts` only emits
 * launch scripts for external Apple-Silicon runtimes (mlx / axolotl); nothing in
 * the repo could actually train + infer a movement policy or be validated in the
 * cloud.
 *
 * This module fills that gap with:
 *   - a canonical movement-token schema (encode/parse) over capture actions,
 *   - a `MovementModelBackend` seam so the learning backend is swappable
 *     (a real on-device small model can implement the same interface later),
 *   - a deterministic `BackoffMarkovMovementBackend` (default local backend)
 *     that reproduces recorded sequences and backs off — by context order and by
 *     token *similarity* — to generalize to related-but-unseen movements,
 *   - a trivial `FrequencyMovementBackend` baseline to make the seam concrete,
 *   - a replay-fidelity evaluator for held-out generalization measurement.
 *
 * Everything here is deterministic (no wall-clock, no Math.random) so it can be
 * exercised with synthetic event streams in CI; the actual on-device recording
 * and training run when the user runs bee-agent locally.
 */

export const MOVEMENT_TOKEN_BOS = "^";
const CONTEXT_SEPARATOR = "␟";
const FIELD_SEPARATOR = "|";

/** A single normalized movement, independent of its capture source. */
export type MovementAction = {
  tool: string;
  gesture?: string;
  target?: string;
  direction?: string;
};

/** Canonical, order-stable string encoding of a movement (the model's token). */
export type MovementToken = string;

export type MovementDataset = {
  version: 1;
  sequences: MovementToken[][];
};

export type MovementTrainOptions = {
  /** Maximum Markov context order to model (>= 1). Default 2. */
  order?: number;
};

export type MovementPrediction = {
  token: MovementToken;
  probability: number;
  /** Context order actually used after backoff (0 = unigram). */
  order: number;
  /** True when token-similarity backoff was needed (generalization path). */
  generalized: boolean;
};

export type SerializedMovementModel = {
  version: 1;
  backend: string;
  order: number;
  vocabulary: MovementToken[];
  /** contextKey -> { token -> count }, one entry per backoff order 0..order. */
  transitions: Record<string, Record<MovementToken, number>>;
};

export interface MovementInferenceSession {
  readonly backendId: string;
  /** Best next token for a context, or undefined for an empty model. */
  predictNext(context: MovementToken[]): MovementPrediction | undefined;
  /** Greedily unroll up to `maxSteps` tokens from `seed` (BOS-seeded if empty). */
  generate(seed: MovementToken[], maxSteps: number): MovementToken[];
}

export interface MovementModelBackend {
  readonly id: string;
  train(dataset: MovementDataset, options?: MovementTrainOptions): SerializedMovementModel;
  load(model: SerializedMovementModel): MovementInferenceSession;
}

// ---------------------------------------------------------------------------
// Token encoding
// ---------------------------------------------------------------------------

/** Encode a normalized movement into a canonical, parseable token. */
export function encodeMovementToken(action: MovementAction): MovementToken {
  const parts: string[] = [`tool=${sanitizeField(action.tool)}`];
  if (action.gesture) {
    parts.push(`gesture=${sanitizeField(action.gesture)}`);
  }
  if (action.direction) {
    parts.push(`dir=${sanitizeField(action.direction)}`);
  }
  if (action.target) {
    parts.push(`target=${sanitizeField(action.target)}`);
  }
  return parts.join(FIELD_SEPARATOR);
}

/** Inverse of {@link encodeMovementToken} for similarity comparisons. */
export function parseMovementToken(token: MovementToken): MovementAction {
  const action: MovementAction = { tool: "" };
  for (const part of token.split(FIELD_SEPARATOR)) {
    const eq = part.indexOf("=");
    if (eq < 0) {
      continue;
    }
    const key = part.slice(0, eq);
    const value = part.slice(eq + 1);
    if (key === "tool") {
      action.tool = value;
    } else if (key === "gesture") {
      action.gesture = value;
    } else if (key === "dir") {
      action.direction = value;
    } else if (key === "target") {
      action.target = value;
    }
  }
  return action;
}

function sanitizeField(value: string): string {
  return value.replaceAll(FIELD_SEPARATOR, "/").replaceAll(CONTEXT_SEPARATOR, "/").trim();
}

/** Lift a captured trajectory action into a normalized movement. */
export function movementActionFromTrajectoryAction(action: TrajectoryAction): MovementAction {
  const metadata = action.metadata ?? {};
  const result: MovementAction = { tool: action.tool };
  const gesture = readString(metadata.gesture);
  const target = readString(metadata.target);
  const direction = readString(metadata.direction);
  if (gesture) {
    result.gesture = gesture;
  }
  if (target) {
    result.target = target;
  }
  if (direction === "up" || direction === "down" || direction === "left" || direction === "right") {
    result.direction = direction;
  }
  return result;
}

function readString(value: unknown): string | undefined {
  return typeof value === "string" && value.length > 0 ? value : undefined;
}

// ---------------------------------------------------------------------------
// Dataset construction
// ---------------------------------------------------------------------------

/** Ordered movement token sequence for one trajectory span. */
export function movementSequenceFromTrajectory(span: TrajectorySpan): MovementToken[] {
  return [...span.actions]
    .sort((a, b) => a.ts - b.ts)
    .map((action) => encodeMovementToken(movementActionFromTrajectoryAction(action)));
}

/** Build a training dataset from reviewed trajectory spans. */
export function buildMovementDatasetFromTrajectories(spans: TrajectorySpan[]): MovementDataset {
  const sequences = spans.map((span) => movementSequenceFromTrajectory(span)).filter((seq) => seq.length > 0);
  return { version: 1, sequences };
}

/**
 * Build a dataset from replay manifests. Replay action events carry only
 * `tool`/`summary` (no gesture metadata), so tokens are coarser than the
 * trajectory path — still useful for sequence structure. Events are grouped by
 * their originating trajectory to preserve per-trajectory ordering.
 */
export function buildMovementDatasetFromReplays(manifests: ReplayManifest[]): MovementDataset {
  const sequences: MovementToken[][] = [];
  for (const manifest of manifests) {
    const byTrajectory = new Map<string, MovementToken[]>();
    for (const event of manifest.events) {
      if (event.kind !== "action") {
        continue;
      }
      const token = encodeMovementToken({ tool: event.tool, gesture: event.summary });
      const existing = byTrajectory.get(event.trajectoryId);
      if (existing) {
        existing.push(token);
      } else {
        byTrajectory.set(event.trajectoryId, [token]);
      }
    }
    for (const seq of byTrajectory.values()) {
      if (seq.length > 0) {
        sequences.push(seq);
      }
    }
  }
  return { version: 1, sequences };
}

// ---------------------------------------------------------------------------
// Backoff Markov backend (default local backend)
// ---------------------------------------------------------------------------

export const BACKOFF_MARKOV_BACKEND_ID = "backoff-markov";

export class BackoffMarkovMovementBackend implements MovementModelBackend {
  readonly id = BACKOFF_MARKOV_BACKEND_ID;

  train(dataset: MovementDataset, options?: MovementTrainOptions): SerializedMovementModel {
    const order = Math.max(1, Math.floor(options?.order ?? 2));
    const transitions: Record<string, Record<MovementToken, number>> = {};
    const vocabulary = new Set<MovementToken>();

    for (const rawSequence of dataset.sequences) {
      const sequence = [MOVEMENT_TOKEN_BOS, ...rawSequence];
      for (const token of rawSequence) {
        vocabulary.add(token);
      }
      for (let i = 1; i < sequence.length; i++) {
        const next = sequence[i]!;
        const maxLevel = Math.min(order, i);
        for (let level = 0; level <= maxLevel; level++) {
          const contextTokens = sequence.slice(i - level, i);
          const key = contextKey(contextTokens);
          const bucket = (transitions[key] ??= {});
          bucket[next] = (bucket[next] ?? 0) + 1;
        }
      }
    }

    return {
      version: 1,
      backend: this.id,
      order,
      vocabulary: [...vocabulary].sort(),
      transitions,
    };
  }

  load(model: SerializedMovementModel): MovementInferenceSession {
    return new BackoffMarkovSession(model);
  }
}

class BackoffMarkovSession implements MovementInferenceSession {
  readonly backendId: string;
  private readonly vocabRank: Map<MovementToken, number>;

  constructor(private readonly model: SerializedMovementModel) {
    this.backendId = model.backend;
    this.vocabRank = new Map(model.vocabulary.map((token, index) => [token, index]));
  }

  predictNext(context: MovementToken[]): MovementPrediction | undefined {
    const withBos = context.length === 0 ? [MOVEMENT_TOKEN_BOS] : context;
    // 1. Exact context match at order >= 1 (the unigram fallback is deferred so
    //    it cannot pre-empt the generalization path below).
    const direct = this.predictFromContext(withBos, false, 1);
    if (direct) {
      return direct;
    }
    // 2. Generalization: the trailing context token is unseen (new but related)
    //    — map it to the most similar known movement and retry from there.
    if (context.length > 0) {
      const mapped = this.mapToSimilarToken(context[context.length - 1]!);
      if (mapped !== undefined) {
        const rewritten = [...context.slice(0, -1), mapped];
        const viaSimilar = this.predictFromContext(rewritten, true, 1);
        if (viaSimilar) {
          return viaSimilar;
        }
      }
    }
    // 3. Unigram backoff — always available for a non-empty model.
    return this.predictFromContext([MOVEMENT_TOKEN_BOS], context.length > 0, 0);
  }

  generate(seed: MovementToken[], maxSteps: number): MovementToken[] {
    const output: MovementToken[] = [];
    let context = seed.length === 0 ? [MOVEMENT_TOKEN_BOS] : [...seed];
    for (let step = 0; step < maxSteps; step++) {
      const prediction = this.predictNext(sansBos(context));
      if (!prediction) {
        break;
      }
      output.push(prediction.token);
      context = [...context, prediction.token].slice(-(this.model.order + 1));
    }
    return output;
  }

  private predictFromContext(
    context: MovementToken[],
    generalized: boolean,
    minLevel: number,
  ): MovementPrediction | undefined {
    const maxLevel = Math.min(this.model.order, context.length);
    for (let level = maxLevel; level >= minLevel; level--) {
      const key = contextKey(context.slice(context.length - level));
      const bucket = this.model.transitions[key];
      if (!bucket) {
        continue;
      }
      const best = this.argmax(bucket);
      if (!best) {
        continue;
      }
      const total = Object.values(bucket).reduce((sum, count) => sum + count, 0);
      return {
        token: best,
        probability: total > 0 ? bucket[best]! / total : 0,
        order: level,
        generalized: generalized || level < maxLevel,
      };
    }
    return undefined;
  }

  /** Deterministic argmax: highest count, ties broken by vocabulary order. */
  private argmax(bucket: Record<MovementToken, number>): MovementToken | undefined {
    let best: MovementToken | undefined;
    let bestCount = -1;
    for (const [token, count] of Object.entries(bucket)) {
      if (count > bestCount || (count === bestCount && this.rank(token) < this.rank(best))) {
        best = token;
        bestCount = count;
      }
    }
    return best;
  }

  private mapToSimilarToken(token: MovementToken): MovementToken | undefined {
    if (this.vocabRank.has(token)) {
      return token;
    }
    const query = parseMovementToken(token);
    let best: MovementToken | undefined;
    let bestScore = 0;
    for (const candidate of this.model.vocabulary) {
      const score = movementSimilarity(query, parseMovementToken(candidate));
      if (score > bestScore || (score === bestScore && best !== undefined && this.rank(candidate) < this.rank(best))) {
        best = candidate;
        bestScore = score;
      }
    }
    return bestScore > 0 ? best : undefined;
  }

  private rank(token: MovementToken | undefined): number {
    if (token === undefined) {
      return Number.MAX_SAFE_INTEGER;
    }
    return this.vocabRank.get(token) ?? Number.MAX_SAFE_INTEGER;
  }
}

/** Field-weighted movement similarity in [0, 6]; higher is more alike. */
export function movementSimilarity(a: MovementAction, b: MovementAction): number {
  let score = 0;
  if (a.tool && a.tool === b.tool) {
    score += 2;
  }
  if (a.gesture && a.gesture === b.gesture) {
    score += 2;
  }
  if (a.direction && a.direction === b.direction) {
    score += 1;
  }
  if (a.target && a.target === b.target) {
    score += 1;
  }
  return score;
}

// ---------------------------------------------------------------------------
// Frequency baseline backend (concrete second backend proving the seam)
// ---------------------------------------------------------------------------

export const FREQUENCY_BACKEND_ID = "frequency-baseline";

/** Ignores context; always predicts the globally most frequent movement. */
export class FrequencyMovementBackend implements MovementModelBackend {
  readonly id = FREQUENCY_BACKEND_ID;

  train(dataset: MovementDataset): SerializedMovementModel {
    const counts: Record<MovementToken, number> = {};
    const vocabulary = new Set<MovementToken>();
    for (const sequence of dataset.sequences) {
      for (const token of sequence) {
        vocabulary.add(token);
        counts[token] = (counts[token] ?? 0) + 1;
      }
    }
    return {
      version: 1,
      backend: this.id,
      order: 0,
      vocabulary: [...vocabulary].sort(),
      transitions: { "": counts },
    };
  }

  load(model: SerializedMovementModel): MovementInferenceSession {
    // A frequency model is a degenerate order-0 Markov model.
    return new BackoffMarkovSession({ ...model, order: 0 });
  }
}

// ---------------------------------------------------------------------------
// Evaluation
// ---------------------------------------------------------------------------

export type ReplayFidelityReport = {
  sequences: number;
  transitions: number;
  matched: number;
  generalizedMatches: number;
  /** matched / transitions in [0, 1]. */
  fidelity: number;
};

/**
 * Measure how faithfully a trained session reproduces a set of held-out
 * sequences: for every (context -> actual-next) transition, does the model's
 * greedy prediction match the recorded next movement? Run against the training
 * sequences it measures replay fidelity; against held-out related sequences it
 * measures generalization.
 */
export function evaluateReplayFidelity(
  session: MovementInferenceSession,
  sequences: MovementToken[][],
): ReplayFidelityReport {
  let transitions = 0;
  let matched = 0;
  let generalizedMatches = 0;
  for (const sequence of sequences) {
    for (let i = 0; i < sequence.length; i++) {
      const context = sequence.slice(0, i);
      const prediction = session.predictNext(context);
      transitions += 1;
      if (prediction && prediction.token === sequence[i]) {
        matched += 1;
        if (prediction.generalized) {
          generalizedMatches += 1;
        }
      }
    }
  }
  return {
    sequences: sequences.length,
    transitions,
    matched,
    generalizedMatches,
    fidelity: transitions > 0 ? matched / transitions : 0,
  };
}

// ---------------------------------------------------------------------------
// Synthetic movement stream generator (cloud-side validation without real OS)
// ---------------------------------------------------------------------------

export type SyntheticMovementSpec = {
  seed: number;
  sequenceCount: number;
  /** Ordered gesture templates; one movement per template, jittered by target. */
  templates: MovementAction[];
  /** Candidate targets substituted into templates lacking an explicit target. */
  targets?: string[];
  /** Probability [0,1] a sequence is truncated early, to vary length. */
  truncationRate?: number;
};

/**
 * Deterministically synthesize movement sequences from a template grammar using
 * a self-contained LCG (Math.random is unavailable in this environment and would
 * break reproducibility). Two calls with the same spec yield identical output.
 */
export function generateSyntheticMovementSequences(spec: SyntheticMovementSpec): MovementToken[][] {
  const rng = createLcg(spec.seed);
  const targets = spec.targets ?? [];
  const truncationRate = spec.truncationRate ?? 0;
  const sequences: MovementToken[][] = [];
  for (let s = 0; s < spec.sequenceCount; s++) {
    const sequence: MovementToken[] = [];
    for (const template of spec.templates) {
      if (sequence.length > 0 && truncationRate > 0 && rng() < truncationRate) {
        break;
      }
      const action: MovementAction = { ...template };
      if (!action.target && targets.length > 0) {
        action.target = targets[Math.floor(rng() * targets.length)];
      }
      sequence.push(encodeMovementToken(action));
    }
    if (sequence.length > 0) {
      sequences.push(sequence);
    }
  }
  return sequences;
}

/** Minimal deterministic PRNG (numeric-recipes LCG) returning [0, 1). */
function createLcg(seed: number): () => number {
  let state = (Math.floor(seed) % 2147483647 + 2147483647) % 2147483647;
  if (state === 0) {
    state = 1;
  }
  return () => {
    state = (state * 48271) % 2147483647;
    return (state - 1) / 2147483646;
  };
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

function contextKey(tokens: MovementToken[]): string {
  return tokens.join(CONTEXT_SEPARATOR);
}

function sansBos(context: MovementToken[]): MovementToken[] {
  return context.filter((token) => token !== MOVEMENT_TOKEN_BOS);
}

/** The default local movement backend used when none is specified. */
export function defaultMovementBackend(): MovementModelBackend {
  return new BackoffMarkovMovementBackend();
}
