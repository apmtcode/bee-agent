import type { ReplayManifest, ReplayTimelineEvent } from "../capture/replay.js";

/**
 * In-process movement-learning core for the local-movement subsystem.
 *
 * The rest of the pipeline (capture → schema → dataset → replay → training
 * plan) already exists; what was missing was an actual model that can *learn*
 * from a reviewed movement dataset, *repeat* the recorded movements, and
 * *generalize* to related-but-unseen movement contexts. The real on-device
 * training shells out to MLX/axolotl (see `runner.ts`) and cannot run in the
 * cloud, so this module provides a pluggable backend seam plus a fully
 * deterministic, dependency-free reference backend that runs anywhere — in the
 * cloud for tests/CI and on-device as an instant baseline before a heavier
 * model is trained.
 */

/** A discrete movement token — one normalized recorded action. */
export type MovementToken = string;

/** Sentinel appended to every training sequence so the model learns to stop. */
export const MOVEMENT_END: MovementToken = "<end>";
/** Sentinel used to left-pad short contexts so boundary n-grams are learnable. */
export const MOVEMENT_START: MovementToken = "<start>";

/** An ordered episode of movement tokens (typically one trajectory). */
export type MovementSequence = {
  id: string;
  tokens: MovementToken[];
};

/**
 * Collapse the volatile detail out of an action summary so that *related*
 * movements share a token. Digits (coordinates, indices, counts) become `#`,
 * whitespace is normalized, and case is folded. This is what lets the model
 * generalize: "click button at (120, 340)" and "click button at (17, 8)" both
 * reduce to `click button at (#, #)`.
 */
export function normalizeMovementSummary(summary: string): string {
  return summary
    .toLowerCase()
    .replace(/\d+/g, "#")
    .replace(/\s+/g, " ")
    .trim();
}

/** Derive a movement token from a recorded action's tool + summary. */
export function actionToMovementToken(tool: string, summary: string): MovementToken {
  const base = tool.trim().toLowerCase();
  const detail = normalizeMovementSummary(summary);
  return detail ? `${base}|${detail}` : base;
}

type ActionEvent = Extract<ReplayTimelineEvent, { kind: "action" }>;

/**
 * Extract one movement sequence per trajectory from a replay manifest. Action
 * events are grouped by trajectory and ordered by timestamp so each sequence is
 * a coherent movement episode rather than an interleaving of concurrent spans.
 */
export function extractMovementSequences(manifest: ReplayManifest): MovementSequence[] {
  const byTrajectory = new Map<string, ActionEvent[]>();
  for (const event of manifest.events) {
    if (event.kind !== "action") {
      continue;
    }
    const bucket = byTrajectory.get(event.trajectoryId);
    if (bucket) {
      bucket.push(event);
    } else {
      byTrajectory.set(event.trajectoryId, [event]);
    }
  }

  const sequences: MovementSequence[] = [];
  for (const [trajectoryId, events] of byTrajectory) {
    const ordered = [...events].sort((a, b) => a.ts - b.ts);
    sequences.push({
      id: trajectoryId,
      tokens: ordered.map((event) => actionToMovementToken(event.tool, event.summary)),
    });
  }
  return sequences;
}

/** Build a movement dataset from many reviewed replay manifests. */
export function buildMovementDataset(manifests: ReplayManifest[]): MovementSequence[] {
  return manifests.flatMap((manifest) => extractMovementSequences(manifest));
}

export type MovementCandidate = {
  token: MovementToken;
  probability: number;
};

export type MovementPrediction = {
  /** The most likely next token, or `null` when the model predicts the episode ends. */
  token: MovementToken | null;
  /** Probability mass on the chosen token within its backoff level (0..1). */
  confidence: number;
  /**
   * The n-gram order actually used. A value below the model's configured order
   * means the exact context was unseen and the model *generalized* by backing
   * off to shorter, more general context — the mechanism behind objective 2(d).
   */
  order: number;
  /** All candidates for the chosen backoff level, most probable first. */
  candidates: MovementCandidate[];
};

/** A trained movement model: repeats and generalizes recorded movements. */
export interface MovementModel {
  readonly backend: string;
  readonly order: number;
  readonly vocabulary: readonly MovementToken[];
  /** Predict the next movement given a context (most-recent-last). */
  predictNext(context: MovementToken[]): MovementPrediction;
  /** Roll out movements from a context until `<end>` or `maxSteps`. */
  generate(context: MovementToken[], maxSteps: number): MovementToken[];
}

/** Pluggable training backend. Swap the deterministic mock for a real one. */
export interface MovementModelBackend {
  readonly name: string;
  train(dataset: MovementSequence[]): MovementModel;
}

export type MarkovMovementBackendOptions = {
  /** Highest n-gram order (context length) to learn. Default 2. */
  order?: number;
};

type CountTable = Map<string, Map<MovementToken, number>>;

/**
 * Deterministic, dependency-free reference backend: a back-off n-gram (Markov)
 * model. Training counts token transitions at every order from `order` down to
 * a unigram. Prediction tries the longest matching context first and falls back
 * to shorter contexts when the exact history was never observed — so an unseen
 * but *related* prefix still yields the globally-consistent next movement.
 *
 * Fully deterministic: ties break by descending count then lexical token, so
 * the same dataset always yields the same model — important for reproducible
 * training and stable tests/CI.
 */
export class MarkovMovementBackend implements MovementModelBackend {
  readonly name = "deterministic-markov";

  constructor(private readonly options: MarkovMovementBackendOptions = {}) {}

  train(dataset: MovementSequence[]): MovementModel {
    const order = Math.max(1, Math.floor(this.options.order ?? 2));
    // tables[k] maps a k-token context key -> next-token counts.
    const tables: CountTable[] = Array.from({ length: order + 1 }, () => new Map());
    const vocabulary = new Set<MovementToken>();

    for (const sequence of dataset) {
      const padded = [
        ...Array.from({ length: order }, () => MOVEMENT_START),
        ...sequence.tokens,
        MOVEMENT_END,
      ];
      for (const token of sequence.tokens) {
        vocabulary.add(token);
      }
      for (let i = order; i < padded.length; i += 1) {
        const next = padded[i]!;
        for (let k = 0; k <= order; k += 1) {
          const context = padded.slice(i - k, i);
          const key = contextKey(context);
          const table = tables[k]!;
          const counts = table.get(key) ?? new Map<MovementToken, number>();
          counts.set(next, (counts.get(next) ?? 0) + 1);
          table.set(key, counts);
        }
      }
    }

    return new MarkovMovementModel(order, tables, [...vocabulary].sort());
  }
}

class MarkovMovementModel implements MovementModel {
  readonly backend = "deterministic-markov";

  constructor(
    readonly order: number,
    private readonly tables: CountTable[],
    readonly vocabulary: readonly MovementToken[],
  ) {}

  predictNext(context: MovementToken[]): MovementPrediction {
    const trimmed = context.slice(-this.order);
    // Start at the full order and back off. Short contexts (e.g. the start of an
    // episode) are left-padded with MOVEMENT_START exactly as training padded
    // them, so the highest-order start n-grams stay reachable at inference.
    for (let k = this.order; k >= 0; k -= 1) {
      const key = contextKey(padContext(trimmed, k));
      const counts = this.tables[k]?.get(key);
      if (!counts || counts.size === 0) {
        continue;
      }
      const candidates = rankCandidates(counts);
      const best = candidates[0]!;
      return {
        token: best.token === MOVEMENT_END ? null : best.token,
        confidence: best.probability,
        order: k,
        candidates,
      };
    }
    return { token: null, confidence: 0, order: 0, candidates: [] };
  }

  generate(context: MovementToken[], maxSteps: number): MovementToken[] {
    const produced: MovementToken[] = [];
    const running = [...context];
    for (let step = 0; step < maxSteps; step += 1) {
      const prediction = this.predictNext(running);
      if (prediction.token === null) {
        break;
      }
      produced.push(prediction.token);
      running.push(prediction.token);
    }
    return produced;
  }
}

function padContext(context: MovementToken[], order: number): MovementToken[] {
  if (context.length >= order) {
    return context.slice(context.length - order);
  }
  return [
    ...Array.from({ length: order - context.length }, () => MOVEMENT_START),
    ...context,
  ];
}

function contextKey(context: MovementToken[]): string {
  return context.join("");
}

function rankCandidates(counts: Map<MovementToken, number>): MovementCandidate[] {
  const total = [...counts.values()].reduce((sum, value) => sum + value, 0);
  return [...counts.entries()]
    .map(([token, count]) => ({ token, count }))
    .sort((a, b) => (b.count !== a.count ? b.count - a.count : a.token < b.token ? -1 : 1))
    .map(({ token, count }) => ({ token, probability: total === 0 ? 0 : count / total }));
}

/**
 * Deterministic synthetic movement generator — the cloud-safe stand-in for real
 * OS input capture. Given a small "grammar" of movement classes it emits
 * repeatable, structured sequences (seeded, no `Math.random`), so the whole
 * capture→dataset→train→eval loop can be validated without a real machine.
 */
export type SyntheticMovementSpec = {
  /** Movement-class vocabulary, e.g. ["focus", "click", "type", "submit"]. */
  vocabulary: MovementToken[];
  /** Number of sequences to emit. */
  sequences: number;
  /** Inclusive min/max length of each sequence. */
  minLength: number;
  maxLength: number;
  /** Deterministic seed. */
  seed: number;
};

export function synthesizeMovementSequences(spec: SyntheticMovementSpec): MovementSequence[] {
  if (spec.vocabulary.length === 0) {
    return [];
  }
  const rng = createSeededRng(spec.seed);
  const out: MovementSequence[] = [];
  for (let s = 0; s < spec.sequences; s += 1) {
    const span = Math.max(1, spec.maxLength - spec.minLength);
    const length = spec.minLength + Math.floor(rng() * (span + 1));
    const tokens: MovementToken[] = [];
    // A first-order Markov walk over the vocabulary index with a bias toward
    // advancing to the next class — yields coherent, learnable structure.
    let index = Math.floor(rng() * spec.vocabulary.length);
    for (let i = 0; i < length; i += 1) {
      tokens.push(spec.vocabulary[index % spec.vocabulary.length]!);
      const advance = rng() < 0.7 ? 1 : 1 + Math.floor(rng() * spec.vocabulary.length);
      index = (index + advance) % spec.vocabulary.length;
    }
    out.push({ id: `synthetic-${spec.seed}-${s}`, tokens });
  }
  return out;
}

/** Small, self-contained LCG so synthetic data is reproducible and offline. */
function createSeededRng(seed: number): () => number {
  let state = (seed >>> 0) || 1;
  return () => {
    // Numerical Recipes LCG constants.
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 0x1_0000_0000;
  };
}
