import type { ReplayManifest, ReplayTimelineEvent } from "../capture/replay.js";
import type { TrajectorySpan } from "../capture/trajectory.js";

/**
 * Movement policy model - the on-device learning core of bee-agent's
 * local-movement subsystem (standing objective #2, parts c and d).
 *
 * The pipeline before this module already turns recorded mouse / keyboard /
 * window / UI activity into TrajectorySpans and ReplayManifests. This module
 * closes the loop: it tokenizes those recordings into ordered movement
 * sequences, trains a model that can (c) repeat the recorded movements and (d)
 * generalize to new-but-related movements, and evaluates how faithfully it
 * does so.
 *
 * The model backend is pluggable (MovementModelBackend). The bundled
 * DeterministicMarkovMovementBackend is a dependency-free, fully deterministic
 * implementation so the whole train -> replay -> generalize loop runs and is
 * testable in the cloud with synthetic event streams. A real on-device small
 * model (e.g. an MLX/GGUF policy) can implement the same interface and slot in
 * unchanged - see MovementModelTrainer.
 */

/** Sentinel prepended to every training sequence so an empty prefix can still
 * predict the most likely opening movement. */
export const MOVEMENT_START_TOKEN = "<movement:start>";
/** Sentinel appended to every training sequence so the model can learn when a
 * movement sequence naturally terminates. */
export const MOVEMENT_END_TOKEN = "<movement:end>";

/** A single normalized movement step. Tokens are canonical strings so distinct
 * backends and serialized artifacts agree on the vocabulary. */
export type MovementToken = string;

/** An ordered movement sequence derived from one recording. */
export type MovementSequence = {
  id: string;
  tokens: MovementToken[];
};

export type MovementTrainOptions = {
  /** Markov context length. Higher orders reproduce recordings more exactly;
   * lower orders generalize more aggressively. Clamped to >= 1. Default 2. */
  order?: number;
};

export type MovementTransition = {
  context: MovementToken[];
  next: Array<{ token: MovementToken; count: number }>;
};

/** A trained, serializable movement model. Plain JSON so it can be persisted
 * next to the reviewed export and reloaded by any backend of the same name. */
export type MovementModelArtifact = {
  version: 1;
  backend: string;
  order: number;
  vocabulary: MovementToken[];
  transitions: MovementTransition[];
  exampleCount: number;
  tokenCount: number;
};

export type MovementPredictionCandidate = {
  token: MovementToken;
  score: number;
};

export type MovementPrediction = {
  /** Most likely next movement, or undefined for an empty/unseen model. */
  token: MovementToken | undefined;
  /** Estimated probability of the chosen token in [0, 1]. */
  score: number;
  /** Context length actually used after backoff (order..0), or -1 if the model
   * had nothing to go on. Lets callers see when a prediction is a
   * generalization (shorter context than trained) vs. an exact recall. */
  contextOrder: number;
  /** All candidate next movements for the resolved context, best first. */
  candidates: MovementPredictionCandidate[];
};

export type MovementGenerateOptions = {
  /** Hard cap on generated steps (excludes the seed prefix). Default 256. */
  maxSteps?: number;
  /** Stop once this token is produced (in addition to the learned end token). */
  stopToken?: MovementToken;
};

/** Pluggable movement-model backend. Implement this to swap the deterministic
 * mock for a real on-device model without touching callers. */
export interface MovementModelBackend {
  readonly name: string;
  train(examples: MovementSequence[], options?: MovementTrainOptions): MovementModelArtifact;
  predictNext(artifact: MovementModelArtifact, prefix: MovementToken[]): MovementPrediction;
  generate(
    artifact: MovementModelArtifact,
    prefix: MovementToken[],
    options?: MovementGenerateOptions,
  ): MovementToken[];
}

/**
 * Deterministic order-k Markov backend with stupid-backoff smoothing.
 *
 * Training counts, for every context length 0..order, how often each token
 * follows a given context. Prediction takes the longest context that was seen
 * and picks its most frequent successor, backing off to shorter contexts when
 * the full context is novel - this is what lets it generalize to related but
 * unseen movement prefixes. Argmax ties break on token order, so every result
 * is fully reproducible: identical inputs always yield identical models and
 * predictions (no clocks, no RNG).
 */
export class DeterministicMarkovMovementBackend implements MovementModelBackend {
  readonly name = "deterministic-markov";

  train(examples: MovementSequence[], options: MovementTrainOptions = {}): MovementModelArtifact {
    const order = Math.max(1, Math.floor(options.order ?? 2));
    // context key -> (token -> count)
    const counts = new Map<string, Map<MovementToken, number>>();
    const vocabulary = new Set<MovementToken>();
    let tokenCount = 0;

    for (const example of examples) {
      const padded = [
        ...Array<MovementToken>(order).fill(MOVEMENT_START_TOKEN),
        ...example.tokens,
        MOVEMENT_END_TOKEN,
      ];
      for (const token of example.tokens) {
        vocabulary.add(token);
      }
      tokenCount += example.tokens.length;

      for (let index = order; index < padded.length; index += 1) {
        const target = padded[index];
        for (let k = 0; k <= order; k += 1) {
          const context = padded.slice(index - k, index);
          const key = contextKey(context);
          let bucket = counts.get(key);
          if (!bucket) {
            bucket = new Map<MovementToken, number>();
            counts.set(key, bucket);
          }
          bucket.set(target, (bucket.get(target) ?? 0) + 1);
        }
      }
    }

    const transitions: MovementTransition[] = [...counts.entries()]
      .map(([key, bucket]) => ({
        context: decodeContextKey(key),
        next: [...bucket.entries()]
          .map(([token, count]) => ({ token, count }))
          .sort((a, b) => (b.count - a.count) || compareToken(a.token, b.token)),
      }))
      .sort((a, b) => contextKey(a.context).localeCompare(contextKey(b.context)));

    return {
      version: 1,
      backend: this.name,
      order,
      vocabulary: [...vocabulary].sort(compareToken),
      transitions,
      exampleCount: examples.length,
      tokenCount,
    };
  }

  predictNext(artifact: MovementModelArtifact, prefix: MovementToken[]): MovementPrediction {
    const index = indexTransitions(artifact);
    const padded = [...Array<MovementToken>(artifact.order).fill(MOVEMENT_START_TOKEN), ...prefix];

    for (let k = artifact.order; k >= 0; k -= 1) {
      const context = padded.slice(padded.length - k, padded.length);
      const entry = index.get(contextKey(context));
      if (!entry || entry.next.length === 0) {
        continue;
      }
      const total = entry.next.reduce((sum, candidate) => sum + candidate.count, 0);
      const candidates = entry.next.map((candidate) => ({
        token: candidate.token,
        score: total > 0 ? candidate.count / total : 0,
      }));
      return {
        token: candidates[0].token,
        score: candidates[0].score,
        contextOrder: k,
        candidates,
      };
    }

    return { token: undefined, score: 0, contextOrder: -1, candidates: [] };
  }

  generate(
    artifact: MovementModelArtifact,
    prefix: MovementToken[],
    options: MovementGenerateOptions = {},
  ): MovementToken[] {
    const maxSteps = Math.max(0, Math.floor(options.maxSteps ?? 256));
    const working = [...prefix];
    const generated: MovementToken[] = [];

    for (let step = 0; step < maxSteps; step += 1) {
      const prediction = this.predictNext(artifact, working);
      const token = prediction.token;
      if (token === undefined || token === MOVEMENT_END_TOKEN) {
        break;
      }
      generated.push(token);
      working.push(token);
      if (options.stopToken !== undefined && token === options.stopToken) {
        break;
      }
    }

    return generated;
  }
}

export type MovementReplayFidelity = {
  sequences: number;
  tokens: number;
  correct: number;
  /** Teacher-forced next-token accuracy in [0, 1] across all held-out tokens. */
  accuracy: number;
  perSequence: Array<{ id: string; tokens: number; correct: number; accuracy: number }>;
};

export type MovementModelTrainerOptions = {
  backend?: MovementModelBackend;
};

/**
 * High-level facade tying the movement pipeline to a backend. Converts existing
 * recordings (TrajectorySpan / ReplayManifest) into movement sequences, trains,
 * replays, and measures generalization fidelity on held-out sequences. Swap
 * MovementModelTrainerOptions.backend to train a real on-device model with the
 * same calls.
 */
export class MovementModelTrainer {
  private readonly backend: MovementModelBackend;

  constructor(options: MovementModelTrainerOptions = {}) {
    this.backend = options.backend ?? new DeterministicMarkovMovementBackend();
  }

  get backendName(): string {
    return this.backend.name;
  }

  train(sequences: MovementSequence[], options?: MovementTrainOptions): MovementModelArtifact {
    return this.backend.train(sequences, options);
  }

  predictNext(artifact: MovementModelArtifact, prefix: MovementToken[]): MovementPrediction {
    return this.backend.predictNext(artifact, prefix);
  }

  /** Reproduce a movement sequence from an optional seed prefix - objective
   * #2(c). With no seed the model replays its most likely full trajectory. */
  replay(
    artifact: MovementModelArtifact,
    seed: MovementToken[] = [],
    options?: MovementGenerateOptions,
  ): MovementToken[] {
    return [...seed, ...this.backend.generate(artifact, seed, options)];
  }

  /** Measure how well the model predicts held-out (but related) sequences -
   * the generalization eval harness for objective #2(d). */
  evaluate(artifact: MovementModelArtifact, heldOut: MovementSequence[]): MovementReplayFidelity {
    let totalTokens = 0;
    let totalCorrect = 0;
    const perSequence = heldOut.map((sequence) => {
      let correct = 0;
      for (let index = 0; index < sequence.tokens.length; index += 1) {
        const prefix = sequence.tokens.slice(0, index);
        const prediction = this.backend.predictNext(artifact, prefix);
        if (prediction.token === sequence.tokens[index]) {
          correct += 1;
        }
      }
      totalTokens += sequence.tokens.length;
      totalCorrect += correct;
      return {
        id: sequence.id,
        tokens: sequence.tokens.length,
        correct,
        accuracy: sequence.tokens.length > 0 ? correct / sequence.tokens.length : 0,
      };
    });

    return {
      sequences: heldOut.length,
      tokens: totalTokens,
      correct: totalCorrect,
      accuracy: totalTokens > 0 ? totalCorrect / totalTokens : 0,
      perSequence,
    };
  }
}

/** Canonical token for a single replay-timeline movement event. Transcript
 * (chat) events are not movements and yield undefined. */
export function movementTokenFromReplayEvent(event: ReplayTimelineEvent): MovementToken | undefined {
  switch (event.kind) {
    case "action":
      return `action:${normalizeSegment(event.tool)}:${normalizeSegment(event.summary)}`;
    case "observation":
      return `obs:${normalizeSegment(event.source)}:${normalizeSegment(event.summary)}`;
    case "transcript":
      return undefined;
  }
}

/** Build a movement sequence from a trajectory's observations + actions, in
 * timestamp order (ties keep observations before actions). */
export function movementSequenceFromTrajectory(trajectory: TrajectorySpan): MovementSequence {
  const steps: Array<{ ts: number; order: number; token: MovementToken }> = [
    ...trajectory.observations.map((observation) => ({
      ts: observation.ts,
      order: 0,
      token: `obs:${normalizeSegment(observation.source)}:${normalizeSegment(observation.summary)}`,
    })),
    ...trajectory.actions.map((action) => ({
      ts: action.ts,
      order: 1,
      token: `action:${normalizeSegment(action.tool)}:${normalizeSegment(action.summary)}`,
    })),
  ];
  steps.sort((a, b) => (a.ts - b.ts) || (a.order - b.order));
  return { id: trajectory.id, tokens: steps.map((step) => step.token) };
}

/** Build a movement sequence from a replay manifest, dropping transcript events
 * and preserving the manifest's already-sorted timeline. */
export function movementSequenceFromReplayManifest(manifest: ReplayManifest): MovementSequence {
  const tokens = manifest.events
    .map((event) => movementTokenFromReplayEvent(event))
    .filter((token): token is MovementToken => token !== undefined);
  return { id: manifest.sessionId, tokens };
}

function normalizeSegment(value: string): string {
  return value.trim().replace(/\s+/g, " ").toLowerCase();
}

function contextKey(context: MovementToken[]): string {
  // JSON encoding is injective over string arrays, so distinct contexts never
  // collide regardless of what characters a token summary contains.
  return JSON.stringify(context);
}

function decodeContextKey(key: string): MovementToken[] {
  return JSON.parse(key) as MovementToken[];
}

function compareToken(a: MovementToken, b: MovementToken): number {
  return a < b ? -1 : a > b ? 1 : 0;
}

function indexTransitions(artifact: MovementModelArtifact): Map<string, MovementTransition> {
  const index = new Map<string, MovementTransition>();
  for (const transition of artifact.transitions) {
    index.set(contextKey(transition.context), transition);
  }
  return index;
}
