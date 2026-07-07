// Movement-model backend: the in-process, cloud-runnable half of the
// local-movement learning subsystem (standing objective #2, pieces c + d).
//
// The training *runner* (runner.ts) emits shell plans for real on-device
// runtimes (mlx / axolotl) that only execute on the user's machine. That gives
// us no way to validate the "post-train a model to repeat recorded movements
// and generalize to related movements" loop in the cloud or in CI.
//
// This module fills that gap with a *pluggable* backend seam
// (`MovementModelBackend`) and a deterministic, dependency-free reference
// implementation (`MarkovMovementBackend`) that:
//   - trains an order-N Markov model over movement tokens derived from reviewed
//     trajectories / replay manifests,
//   - repeats recorded movement sequences exactly, and
//   - generalizes to unseen-but-related prefixes via context back-off.
//
// A real on-device small model plugs in by implementing `MovementModelBackend`
// (see the interface docs). Everything here is synchronous-friendly and
// JSON-serializable so a trained artifact can be persisted next to a training
// job and reloaded for inference.

import type { ReplayManifest } from "../capture/replay.js";
import type { TrajectorySpan } from "../capture/trajectory.js";

/** A single movement, canonicalized to a stable string token. */
export type MovementToken = string;

/** Sentinels used to model sequence start / end during training + generation. */
export const MOVEMENT_START_TOKEN = "START" as const;
export const MOVEMENT_END_TOKEN = "END" as const;

export type MovementSequence = {
  /** Source trajectory id when the sequence came from a captured trajectory. */
  trajectoryId?: string;
  tokens: MovementToken[];
};

export type MovementDataset = {
  version: 1;
  sequences: MovementSequence[];
};

/** The shape every trained artifact shares. Backends extend it. */
export type MovementModelArtifact = {
  backendId: string;
  version: number;
  vocabulary: MovementToken[];
  sequenceCount: number;
  tokenCount: number;
};

export type MovementPrediction = {
  token: MovementToken;
  /** Context order actually used after back-off (0 = unigram prior). */
  order: number;
  /** P(token | context) within the backed-off context. */
  confidence: number;
};

export type MovementTrainOptions = {
  /** Max context length (n-gram order). Default 2. */
  order?: number;
};

export type MovementGenerationOptions = {
  /** Priming context (recorded prefix). Defaults to empty (predict from start). */
  prompt?: MovementToken[];
  /** Hard cap on generated tokens to guarantee termination. Default 64. */
  maxSteps?: number;
};

export type MovementGenerationResult = {
  tokens: MovementToken[];
  /** True when generation stopped because the model emitted END. */
  completed: boolean;
  steps: MovementPrediction[];
};

/**
 * Pluggable backend seam. A real on-device small model implements this by:
 *   - `train`: fine-tune / fit on the dataset and return a (serializable-ish)
 *     handle or artifact; may be async (spawns a local process).
 *   - `predictNext`: run inference for the next movement given a context. Must
 *     be deterministic given the same artifact + context so replay + eval are
 *     reproducible.
 */
export interface MovementModelBackend<A extends MovementModelArtifact = MovementModelArtifact> {
  readonly id: string;
  train(dataset: MovementDataset, options?: MovementTrainOptions): Promise<A>;
  predictNext(artifact: A, context: MovementToken[]): MovementPrediction | undefined;
}

// --- Tokenization ---------------------------------------------------------

/** Lowercase, collapse whitespace, strip to a stable slug for token identity. */
export function slugifyMovementSummary(summary: string): string {
  return summary
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
}

/** Canonical token for a captured action: `tool:slug(summary)`. */
export function tokenizeMovementAction(action: { tool: string; summary: string }): MovementToken {
  const tool = action.tool.trim().toLowerCase() || "unknown";
  const slug = slugifyMovementSummary(action.summary);
  return slug ? `${tool}:${slug}` : tool;
}

// --- Dataset builders -----------------------------------------------------

/** Build a movement dataset from trajectory spans (actions ordered by ts). */
export function buildMovementDatasetFromTrajectories(
  trajectories: readonly TrajectorySpan[],
): MovementDataset {
  const sequences: MovementSequence[] = [];
  for (const trajectory of trajectories) {
    const tokens = [...trajectory.actions]
      .sort((a, b) => a.ts - b.ts)
      .map((action) => tokenizeMovementAction(action));
    if (tokens.length > 0) {
      sequences.push({ trajectoryId: trajectory.id, tokens });
    }
  }
  return { version: 1, sequences };
}

/** Build a movement dataset from replay manifests (action events, per trajectory). */
export function buildMovementDatasetFromReplays(
  replays: readonly Pick<ReplayManifest, "events">[],
): MovementDataset {
  const byTrajectory = new Map<string, { ts: number; token: MovementToken }[]>();
  for (const replay of replays) {
    for (const event of replay.events) {
      if (event.kind !== "action") {
        continue;
      }
      const list = byTrajectory.get(event.trajectoryId) ?? [];
      list.push({ ts: event.ts, token: tokenizeMovementAction(event) });
      byTrajectory.set(event.trajectoryId, list);
    }
  }
  const sequences: MovementSequence[] = [];
  // Deterministic ordering by trajectory id.
  for (const trajectoryId of [...byTrajectory.keys()].sort()) {
    const entries = byTrajectory.get(trajectoryId)!;
    entries.sort((a, b) => a.ts - b.ts);
    sequences.push({ trajectoryId, tokens: entries.map((entry) => entry.token) });
  }
  return { version: 1, sequences };
}

// --- Markov reference backend --------------------------------------------

export type MarkovMovementArtifact = MovementModelArtifact & {
  backendId: "markov";
  version: 1;
  order: number;
  /** contextKey -> (token -> count). Includes all back-off orders 0..order. */
  transitions: Record<string, Record<string, number>>;
};

const CONTEXT_SEPARATOR = "\u0001";

function contextKey(context: readonly MovementToken[]): string {
  return context.length === 0 ? "" : context.join(CONTEXT_SEPARATOR);
}

/**
 * Deterministic order-N Markov backend with stupid-backoff generalization.
 *
 * Generalization: when the exact order-N context was never seen, prediction
 * backs off to shorter contexts (N-1, ..., unigram), so a novel-but-related
 * prefix still yields a sensible next movement instead of failing.
 */
export class MarkovMovementBackend implements MovementModelBackend<MarkovMovementArtifact> {
  readonly id = "markov";

  async train(dataset: MovementDataset, options: MovementTrainOptions = {}): Promise<MarkovMovementArtifact> {
    const order = Math.max(1, Math.floor(options.order ?? 2));
    const transitions: Record<string, Record<string, number>> = {};
    const vocabulary = new Set<MovementToken>();
    let tokenCount = 0;

    for (const sequence of dataset.sequences) {
      const augmented = [MOVEMENT_START_TOKEN, ...sequence.tokens, MOVEMENT_END_TOKEN];
      for (const token of sequence.tokens) {
        vocabulary.add(token);
      }
      tokenCount += sequence.tokens.length;

      for (let i = 1; i < augmented.length; i += 1) {
        const target = augmented[i]!;
        // Record every back-off order 0..order for position i.
        for (let o = 0; o <= order; o += 1) {
          if (i - o < 0) {
            break;
          }
          const key = contextKey(augmented.slice(i - o, i));
          const bucket = (transitions[key] ??= {});
          bucket[target] = (bucket[target] ?? 0) + 1;
        }
      }
    }

    return {
      backendId: "markov",
      version: 1,
      order,
      vocabulary: [...vocabulary].sort(),
      sequenceCount: dataset.sequences.length,
      tokenCount,
      transitions,
    };
  }

  predictNext(artifact: MarkovMovementArtifact, context: MovementToken[]): MovementPrediction | undefined {
    const primed = context[0] === MOVEMENT_START_TOKEN ? context : [MOVEMENT_START_TOKEN, ...context];
    const maxOrder = Math.min(artifact.order, primed.length);
    for (let o = maxOrder; o >= 0; o -= 1) {
      const key = contextKey(primed.slice(primed.length - o, primed.length));
      const bucket = artifact.transitions[key];
      if (!bucket) {
        continue;
      }
      const best = argmaxToken(bucket);
      if (best) {
        return { token: best.token, order: o, confidence: best.count / best.total };
      }
    }
    return undefined;
  }
}

function argmaxToken(bucket: Record<string, number>): { token: string; count: number; total: number } | undefined {
  let total = 0;
  let bestToken: string | undefined;
  let bestCount = -1;
  // Deterministic: highest count wins; ties broken by lexicographic token order.
  for (const token of Object.keys(bucket).sort()) {
    const count = bucket[token]!;
    total += count;
    if (count > bestCount) {
      bestCount = count;
      bestToken = token;
    }
  }
  if (bestToken === undefined) {
    return undefined;
  }
  return { token: bestToken, count: bestCount, total };
}

// --- Generation + evaluation ---------------------------------------------

/**
 * Roll out a full movement sequence from a backend + artifact. Repeats recorded
 * movements when primed with a seen prefix; generalizes when primed with a
 * novel-but-related prefix. Terminates at END or `maxSteps`.
 */
export async function generateMovementSequence<A extends MovementModelArtifact>(
  backend: MovementModelBackend<A>,
  artifact: A,
  options: MovementGenerationOptions = {},
): Promise<MovementGenerationResult> {
  const maxSteps = Math.max(1, Math.floor(options.maxSteps ?? 64));
  const context: MovementToken[] = [...(options.prompt ?? [])];
  const tokens: MovementToken[] = [];
  const steps: MovementPrediction[] = [];
  let completed = false;

  for (let step = 0; step < maxSteps; step += 1) {
    const prediction = backend.predictNext(artifact, context);
    if (!prediction) {
      break;
    }
    steps.push(prediction);
    if (prediction.token === MOVEMENT_END_TOKEN) {
      completed = true;
      break;
    }
    tokens.push(prediction.token);
    context.push(prediction.token);
  }

  return { tokens, completed, steps };
}

export type MovementEvalReport = {
  sequenceCount: number;
  tokenCount: number;
  /** Teacher-forced next-token accuracy across all held-out positions. */
  nextTokenAccuracy: number;
  /** Fraction of held-out sequences reproduced exactly by free generation. */
  exactSequenceMatch: number;
  /** Avg fraction of a held-out sequence's leading tokens generated correctly. */
  meanPrefixMatch: number;
};

/**
 * Generalization eval harness: measure how well a trained model reproduces /
 * predicts *held-out* sequences it was not trained on. Deterministic.
 */
export async function evaluateMovementModel<A extends MovementModelArtifact>(
  backend: MovementModelBackend<A>,
  artifact: A,
  heldOut: MovementDataset,
): Promise<MovementEvalReport> {
  let correctNext = 0;
  let totalNext = 0;
  let exactMatches = 0;
  let prefixMatchSum = 0;
  let evaluatedSequences = 0;

  for (const sequence of heldOut.sequences) {
    if (sequence.tokens.length === 0) {
      continue;
    }
    evaluatedSequences += 1;

    // Teacher-forced next-token accuracy over the true prefixes.
    const truth = [...sequence.tokens, MOVEMENT_END_TOKEN];
    for (let i = 0; i < truth.length; i += 1) {
      const prediction = backend.predictNext(artifact, sequence.tokens.slice(0, i));
      totalNext += 1;
      if (prediction?.token === truth[i]) {
        correctNext += 1;
      }
    }

    // Free-running generation from an empty prompt.
    const generated = await generateMovementSequence(backend, artifact, {
      maxSteps: sequence.tokens.length + 4,
    });
    if (arraysEqual(generated.tokens, sequence.tokens)) {
      exactMatches += 1;
    }
    prefixMatchSum += leadingMatchFraction(generated.tokens, sequence.tokens);
  }

  return {
    sequenceCount: evaluatedSequences,
    tokenCount: totalNext,
    nextTokenAccuracy: totalNext === 0 ? 0 : correctNext / totalNext,
    exactSequenceMatch: evaluatedSequences === 0 ? 0 : exactMatches / evaluatedSequences,
    meanPrefixMatch: evaluatedSequences === 0 ? 0 : prefixMatchSum / evaluatedSequences,
  };
}

function arraysEqual(a: readonly string[], b: readonly string[]): boolean {
  return a.length === b.length && a.every((value, index) => value === b[index]);
}

function leadingMatchFraction(generated: readonly string[], truth: readonly string[]): number {
  if (truth.length === 0) {
    return 1;
  }
  let matched = 0;
  for (let i = 0; i < truth.length && i < generated.length; i += 1) {
    if (generated[i] !== truth[i]) {
      break;
    }
    matched += 1;
  }
  return matched / truth.length;
}

// --- Synthetic event-stream generator ------------------------------------
//
// Validates the capture -> dataset -> train -> generalize loop without any real
// OS input. Uses a seeded LCG so runs are byte-for-byte reproducible in CI.

export type SyntheticMovementOptions = {
  seed?: number;
  sequenceCount?: number;
  /** Tools/gestures the synthetic operator draws from. */
  tools?: string[];
  minLength?: number;
  maxLength?: number;
};

class Lcg {
  private state: number;
  constructor(seed: number) {
    this.state = (seed >>> 0) || 1;
  }
  next(): number {
    // Numerical Recipes LCG constants.
    this.state = (Math.imul(this.state, 1664525) + 1013904223) >>> 0;
    return this.state / 0x100000000;
  }
  pick<T>(items: readonly T[]): T {
    return items[Math.floor(this.next() * items.length) % items.length]!;
  }
  int(min: number, max: number): number {
    return min + Math.floor(this.next() * (max - min + 1));
  }
}

/**
 * Generate a deterministic synthetic movement dataset by random-walking a
 * canonical UI workflow chain (open -> act -> type -> submit -> confirm ->
 * close). Each step follows the chain with high probability and occasionally
 * branches, so sequences share strongly-learnable sub-paths — a Markov model
 * trained on some walks predicts *held-out* walks well, which is exactly the
 * "generalize to related movements" behaviour the subsystem must demonstrate.
 */
export function generateSyntheticMovementDataset(options: SyntheticMovementOptions = {}): MovementDataset {
  const rng = new Lcg(options.seed ?? 1);
  const sequenceCount = Math.max(1, options.sequenceCount ?? 12);
  const tools = options.tools ?? ["browser", "keyboard", "device", "window"];
  const minLength = Math.max(1, options.minLength ?? 3);

  // Canonical workflow: each stage is one movement token. The i-th stage is
  // performed with a stable tool so the token identity is consistent across
  // sequences (the property the model learns).
  const verbs = [
    "open-dashboard",
    "click-primary",
    "type-input",
    "click-submit",
    "confirm-action",
    "close-dialog",
  ];
  const chain = verbs.map((verb, i) => `${tools[i % tools.length]}:${verb}`);
  const maxLength = Math.min(chain.length, Math.max(minLength, options.maxLength ?? chain.length));

  const sequences: MovementSequence[] = [];
  for (let i = 0; i < sequenceCount; i += 1) {
    const targetLength = rng.int(minLength, maxLength);
    const tokens: MovementToken[] = [];
    // Most walks start at stage 0; a few start one stage in (shared sub-paths).
    let idx = rng.next() < 0.2 ? 1 : 0;
    while (idx < chain.length && tokens.length < targetLength) {
      tokens.push(chain[idx]!);
      // 85% advance one stage; 15% skip ahead one extra (a related variation).
      idx += rng.next() < 0.85 ? 1 : 2;
    }
    sequences.push({ trajectoryId: `synthetic-${i}`, tokens });
  }
  return { version: 1, sequences };
}

/** Split a dataset deterministically into train / held-out folds. */
export function splitMovementDataset(
  dataset: MovementDataset,
  holdOutEvery = 4,
): { train: MovementDataset; heldOut: MovementDataset } {
  const step = Math.max(2, Math.floor(holdOutEvery));
  const train: MovementSequence[] = [];
  const heldOut: MovementSequence[] = [];
  dataset.sequences.forEach((sequence, index) => {
    if ((index + 1) % step === 0) {
      heldOut.push(sequence);
    } else {
      train.push(sequence);
    }
  });
  return {
    train: { version: 1, sequences: train },
    heldOut: { version: 1, sequences: heldOut },
  };
}
