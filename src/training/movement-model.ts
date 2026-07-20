import type { ReviewedExportManifest } from "./export-manifest.js";

/**
 * In-process movement-model subsystem.
 *
 * The training runner (`runner.ts`) emits Apple-Silicon shell plans for real
 * on-device training. That path cannot execute in the cloud, so it can never
 * validate the core objective: learn from recorded movements, *repeat* them,
 * and *generalize* to new-but-related movements.
 *
 * This module closes that loop entirely in-process with a deterministic,
 * dependency-free backend so the pipeline can be exercised and tested in CI.
 * The {@link MovementModelBackend} interface is the pluggable seam: swap in a
 * real on-device small-model backend without touching callers, dataset
 * builders, or the eval harness.
 */

export type MovementStepKind = "observation" | "action";

/**
 * A single discrete movement step. `token` is the modelled unit (e.g.
 * `action:click`, `observation:screen`); `label` is the human-facing summary.
 */
export type MovementStep = {
  kind: MovementStepKind;
  token: string;
  label: string;
  ts: number;
};

export type MovementSequence = {
  id: string;
  steps: MovementStep[];
};

export type MovementDataset = {
  sequences: MovementSequence[];
};

export type MovementTrainingOptions = {
  /** Markov context length (n-gram order). Defaults to 2. */
  order?: number;
};

export type MovementCandidate = {
  token: string;
  probability: number;
};

export type MovementPrediction = {
  /** Candidate next tokens, highest probability first (ties broken by token). */
  candidates: MovementCandidate[];
  /** The single most likely next token, or undefined when nothing is known. */
  top?: string;
  /** Markov context length actually used after backoff (-1 when no context matched). */
  contextOrderUsed: number;
};

export type MovementGenerateOptions = {
  maxSteps?: number;
  /** PRNG seed for reproducible sampling. Defaults to 1. */
  seed?: number;
  /** Timestamp of the last seed step; generated steps advance from here. */
  startTs?: number;
  /** Fallback inter-step gap (ms) when the timing model has no sample. */
  defaultGapMs?: number;
};

export type SerializedMovementModel = {
  version: 1;
  backendId: string;
  order: number;
  vocabulary: string[];
  transitions: Array<{ context: string[]; next: Array<{ token: string; count: number }> }>;
  timing: Array<{ token: string; meanGapMs: number; sampleCount: number }>;
  labels: Array<{ token: string; label: string }>;
};

/** A trained model: repeat recorded movements and generalize to new ones. */
export interface TrainedMovementModel {
  readonly backendId: string;
  readonly order: number;
  readonly vocabulary: readonly string[];
  /** Rank likely next tokens given a context of prior steps. */
  predictNext(context: MovementStep[]): MovementPrediction;
  /** Roll out a plausible continuation from the seed steps (never verbatim-bound). */
  generate(seed: MovementStep[], options?: MovementGenerateOptions): MovementStep[];
  serialize(): SerializedMovementModel;
}

/** Pluggable training backend. The mock and a future on-device model share this. */
export interface MovementModelBackend {
  readonly id: string;
  train(dataset: MovementDataset, options?: MovementTrainingOptions): Promise<TrainedMovementModel>;
}

const CONTEXT_SEPARATOR = "";
const DEFAULT_ORDER = 2;
const DEFAULT_GAP_MS = 120;

function contextKey(tokens: string[]): string {
  return tokens.join(CONTEXT_SEPARATOR);
}

/** Deterministic PRNG (mulberry32) — reproducible rollouts across runs. */
function createPrng(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Canonicalize a tool/source name into a stable token segment. */
function normalizeSegment(value: string): string {
  const cleaned = value
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/(^-+)|(-+$)/g, "");
  return cleaned.length > 0 ? cleaned : "unknown";
}

export function movementTokenForAction(tool: string): string {
  return `action:${normalizeSegment(tool)}`;
}

export function movementTokenForObservation(source: string): string {
  return `observation:${normalizeSegment(source)}`;
}

/**
 * Build a training dataset from a reviewed export. Only `observation` and
 * `action` timeline events are movements; transcript events are dropped.
 * Steps within each replay are kept in timeline order.
 */
export function datasetFromReviewedExport(manifest: ReviewedExportManifest): MovementDataset {
  const sequences: MovementSequence[] = [];
  for (const replay of manifest.replays) {
    const steps: MovementStep[] = [];
    for (const event of replay.events) {
      if (event.kind === "action") {
        steps.push({ kind: "action", token: movementTokenForAction(event.tool), label: event.summary, ts: event.ts });
      } else if (event.kind === "observation") {
        steps.push({
          kind: "observation",
          token: movementTokenForObservation(event.source),
          label: event.summary,
          ts: event.ts,
        });
      }
    }
    if (steps.length > 0) {
      sequences.push({ id: `${replay.sessionId}:${replay.trajectoryIds.join(",")}`, steps });
    }
  }
  return { sequences };
}

type TransitionCounts = Map<string, Map<string, number>>;

class MarkovTrainedModel implements TrainedMovementModel {
  constructor(
    readonly backendId: string,
    readonly order: number,
    private readonly transitions: TransitionCounts,
    private readonly timing: Map<string, { totalGap: number; count: number }>,
    private readonly labels: Map<string, string>,
    readonly vocabulary: readonly string[],
  ) {}

  predictNext(context: MovementStep[]): MovementPrediction {
    const tokens = context.map((step) => step.token);
    // Back off from the longest matching n-gram down to a single prior token.
    // We deliberately stop at order 1 (never the empty/unigram context) so an
    // unrelated context yields no prediction — a clean generation terminator.
    for (let used = Math.min(this.order, tokens.length); used >= 1; used -= 1) {
      const key = contextKey(tokens.slice(tokens.length - used));
      const next = this.transitions.get(key);
      if (next && next.size > 0) {
        return { ...this.rank(next), contextOrderUsed: used };
      }
    }
    return { candidates: [], contextOrderUsed: -1 };
  }

  generate(seed: MovementStep[], options: MovementGenerateOptions = {}): MovementStep[] {
    const maxSteps = options.maxSteps ?? 8;
    const defaultGap = options.defaultGapMs ?? DEFAULT_GAP_MS;
    const prng = createPrng(options.seed ?? 1);
    const context = [...seed];
    const generated: MovementStep[] = [];
    let ts = options.startTs ?? seed.at(-1)?.ts ?? 0;

    for (let i = 0; i < maxSteps; i += 1) {
      const prediction = this.predictNext(context);
      if (prediction.candidates.length === 0) {
        break;
      }
      const token = this.sample(prediction.candidates, prng);
      const gap = this.meanGap(token) ?? defaultGap;
      ts += gap;
      const step: MovementStep = {
        kind: token.startsWith("action:") ? "action" : "observation",
        token,
        label: this.labels.get(token) ?? token,
        ts,
      };
      generated.push(step);
      context.push(step);
    }
    return generated;
  }

  serialize(): SerializedMovementModel {
    return {
      version: 1,
      backendId: this.backendId,
      order: this.order,
      vocabulary: [...this.vocabulary],
      transitions: [...this.transitions.entries()].map(([key, next]) => ({
        context: key.length > 0 ? key.split(CONTEXT_SEPARATOR) : [],
        next: [...next.entries()].map(([token, count]) => ({ token, count })),
      })),
      timing: [...this.timing.entries()].map(([token, { totalGap, count }]) => ({
        token,
        meanGapMs: count > 0 ? totalGap / count : 0,
        sampleCount: count,
      })),
      labels: [...this.labels.entries()].map(([token, label]) => ({ token, label })),
    };
  }

  private rank(next: Map<string, number>): { candidates: MovementCandidate[]; top?: string } {
    const total = [...next.values()].reduce((sum, count) => sum + count, 0);
    const candidates = [...next.entries()]
      .map(([token, count]) => ({ token, probability: total > 0 ? count / total : 0 }))
      .sort((a, b) => (b.probability !== a.probability ? b.probability - a.probability : a.token.localeCompare(b.token)));
    return { candidates, top: candidates[0]?.token };
  }

  private sample(candidates: MovementCandidate[], prng: () => number): string {
    const roll = prng();
    let cumulative = 0;
    for (const candidate of candidates) {
      cumulative += candidate.probability;
      if (roll < cumulative) {
        return candidate.token;
      }
    }
    return candidates[candidates.length - 1]?.token ?? candidates[0]!.token;
  }

  private meanGap(token: string): number | undefined {
    const entry = this.timing.get(token);
    if (!entry || entry.count === 0) {
      return undefined;
    }
    return entry.totalGap / entry.count;
  }
}

/**
 * Deterministic, dependency-free backend. Learns an order-k Markov transition
 * model over movement tokens (with stupid-backoff to shorter contexts) plus a
 * per-token timing model. Generalizes because it composes learned transitions
 * into sequences never observed verbatim. Stands in for a real on-device model
 * and keeps CI green without hardware.
 */
export class MarkovMovementBackend implements MovementModelBackend {
  readonly id = "mock-markov";

  async train(dataset: MovementDataset, options: MovementTrainingOptions = {}): Promise<TrainedMovementModel> {
    const order = Math.max(1, options.order ?? DEFAULT_ORDER);
    const transitions: TransitionCounts = new Map();
    const timing = new Map<string, { totalGap: number; count: number }>();
    const labels = new Map<string, string>();
    const labelCounts = new Map<string, Map<string, number>>();
    const vocabulary = new Set<string>();

    for (const sequence of dataset.sequences) {
      const steps = sequence.steps;
      for (let i = 0; i < steps.length; i += 1) {
        const step = steps[i]!;
        vocabulary.add(step.token);
        recordLabel(labelCounts, step.token, step.label);

        if (i > 0) {
          const gap = Math.max(0, step.ts - steps[i - 1]!.ts);
          const entry = timing.get(step.token) ?? { totalGap: 0, count: 0 };
          entry.totalGap += gap;
          entry.count += 1;
          timing.set(step.token, entry);
        }

        // Record transitions for every context length 1..order (backoff support).
        for (let ctxLen = 1; ctxLen <= order; ctxLen += 1) {
          if (i - ctxLen < 0) {
            break;
          }
          const context = steps.slice(i - ctxLen, i).map((s) => s.token);
          const key = contextKey(context);
          const next = transitions.get(key) ?? new Map<string, number>();
          next.set(step.token, (next.get(step.token) ?? 0) + 1);
          transitions.set(key, next);
        }
      }
    }

    for (const [token, counts] of labelCounts) {
      labels.set(token, mostCommon(counts));
    }

    return new MarkovTrainedModel(
      this.id,
      order,
      transitions,
      timing,
      labels,
      [...vocabulary].sort((a, b) => a.localeCompare(b)),
    );
  }
}

function recordLabel(labelCounts: Map<string, Map<string, number>>, token: string, label: string): void {
  const counts = labelCounts.get(token) ?? new Map<string, number>();
  counts.set(label, (counts.get(label) ?? 0) + 1);
  labelCounts.set(token, counts);
}

function mostCommon(counts: Map<string, number>): string {
  let best: string | undefined;
  let bestCount = -1;
  for (const [value, count] of counts) {
    if (count > bestCount || (count === bestCount && best !== undefined && value.localeCompare(best) < 0)) {
      best = value;
      bestCount = count;
    }
  }
  return best ?? "";
}

/** Reload a persisted model for inference without retraining. */
export function deserializeMovementModel(serialized: SerializedMovementModel): TrainedMovementModel {
  const transitions: TransitionCounts = new Map();
  for (const entry of serialized.transitions) {
    const next = new Map<string, number>();
    for (const { token, count } of entry.next) {
      next.set(token, count);
    }
    transitions.set(contextKey(entry.context), next);
  }
  const timing = new Map<string, { totalGap: number; count: number }>();
  for (const entry of serialized.timing) {
    timing.set(entry.token, { totalGap: entry.meanGapMs * entry.sampleCount, count: entry.sampleCount });
  }
  const labels = new Map<string, string>();
  for (const entry of serialized.labels) {
    labels.set(entry.token, entry.label);
  }
  return new MarkovTrainedModel(
    serialized.backendId,
    serialized.order,
    transitions,
    timing,
    labels,
    [...serialized.vocabulary],
  );
}

/**
 * Deterministic synthetic movement generator. Produces related sequences drawn
 * from a shared token grammar so capture→dataset→train→infer round-trips can be
 * validated without real OS input. Same seed → identical dataset.
 */
export function synthesizeMovementSequences(params: {
  count: number;
  grammar: string[][];
  seed?: number;
  gapMs?: number;
}): MovementSequence[] {
  const prng = createPrng(params.seed ?? 7);
  const gap = params.gapMs ?? 100;
  const sequences: MovementSequence[] = [];
  for (let s = 0; s < params.count; s += 1) {
    const pattern = params.grammar[Math.floor(prng() * params.grammar.length)] ?? params.grammar[0] ?? [];
    const steps: MovementStep[] = pattern.map((token, index) => ({
      kind: token.startsWith("action:") ? "action" : "observation",
      token,
      label: `${token}#${index}`,
      ts: index * gap,
    }));
    sequences.push({ id: `synthetic-${s}`, steps });
  }
  return sequences;
}

export type GeneralizationEvalResult = {
  sequencesEvaluated: number;
  transitionsEvaluated: number;
  /** Fraction of held-out transitions where the top prediction matched. */
  top1Accuracy: number;
  /** Fraction where the actual next token appeared in the top-k predictions. */
  topKAccuracy: number;
  k: number;
};

/**
 * Leave-one-out generalization harness. For each sequence, train on the others
 * and measure how well the model predicts the held-out sequence's transitions —
 * i.e. its ability to perform new-but-related movements it never trained on.
 */
export async function evaluateGeneralization(
  backend: MovementModelBackend,
  dataset: MovementDataset,
  options: { k?: number; trainingOptions?: MovementTrainingOptions } = {},
): Promise<GeneralizationEvalResult> {
  const k = options.k ?? 3;
  const evaluable = dataset.sequences.filter((sequence) => sequence.steps.length >= 2);
  let transitions = 0;
  let top1Hits = 0;
  let topKHits = 0;

  for (let held = 0; held < evaluable.length; held += 1) {
    const heldOut = evaluable[held]!;
    const trainingSequences = evaluable.filter((_, index) => index !== held);
    const model = await backend.train({ sequences: trainingSequences }, options.trainingOptions);

    for (let i = 1; i < heldOut.steps.length; i += 1) {
      const context = heldOut.steps.slice(0, i);
      const actual = heldOut.steps[i]!.token;
      const prediction = model.predictNext(context);
      if (prediction.candidates.length === 0) {
        transitions += 1;
        continue;
      }
      transitions += 1;
      if (prediction.top === actual) {
        top1Hits += 1;
      }
      if (prediction.candidates.slice(0, k).some((candidate) => candidate.token === actual)) {
        topKHits += 1;
      }
    }
  }

  return {
    sequencesEvaluated: evaluable.length,
    transitionsEvaluated: transitions,
    top1Accuracy: transitions > 0 ? top1Hits / transitions : 0,
    topKAccuracy: transitions > 0 ? topKHits / transitions : 0,
    k,
  };
}
