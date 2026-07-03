import type { TrajectoryAction, TrajectorySpan } from "../capture/trajectory.js";
import type { ReplayManifest } from "../capture/replay.js";

/**
 * In-process, pluggable movement-prediction model for the local-movement
 * learning subsystem.
 *
 * The capture pipeline records user movements as {@link TrajectorySpan}s /
 * {@link ReplayManifest}s; the training runner (`runner.ts`) knows how to hand a
 * reviewed dataset to an external on-device trainer (mlx / axolotl). What was
 * missing is a backend that can actually *learn* from a movement dataset and
 * *predict / regenerate* movements entirely in-process — so the learn→repeat→
 * generalize loop (standing objective 2c/2d) can be exercised and validated in
 * the cloud with synthetic data, and a real on-device small model can be dropped
 * in behind the same {@link MovementModelBackend} seam later.
 *
 * The reference backend here is a variable-order Markov model with stupid
 * backoff. It is fully deterministic (argmax with a stable lexical tie-break, no
 * RNG at inference), serializable, and small — a faithful mock that still
 * demonstrates repeat-recorded-movements and generalize-to-related-movements.
 */

/** A single learnable movement, e.g. `device:tap:composeButton`. */
export type MovementToken = string;

/** One recorded movement sequence (the ordered actions of a trajectory/session). */
export type MovementSequence = {
  sessionId: string;
  trajectoryId?: string;
  tokens: MovementToken[];
};

export type MovementDataset = {
  version: 1;
  sequences: MovementSequence[];
};

export type MovementPrediction = {
  token: MovementToken;
  /** Conditional probability of `token` at the matched backoff context. */
  probability: number;
  /** Length of the context that produced the prediction (0 = unigram). */
  contextOrder: number;
};

export type MovementGenerateOptions = {
  /** Seed prefix to continue from; defaults to the sequence-start sentinel. */
  prefix?: MovementToken[];
  maxLength?: number;
};

export interface MovementModel {
  readonly backendId: string;
  /** Most-likely next token given a prefix, or undefined if nothing is known. */
  predictNext(prefix: MovementToken[]): MovementPrediction | undefined;
  /** Deterministically regenerate a full movement sequence. */
  generate(options?: MovementGenerateOptions): MovementToken[];
  /** Serialize the trained model (for persistence / the on-device seam). */
  toJSON(): SerializedMovementModel;
}

export type MovementTrainConfig = {
  /** Maximum context length. Higher = more faithful repeat, less generalization. */
  order?: number;
};

export interface MovementModelBackend {
  readonly id: string;
  train(dataset: MovementDataset, config?: MovementTrainConfig): MovementModel;
  restore(serialized: SerializedMovementModel): MovementModel;
}

export type SerializedMovementModel = {
  version: 1;
  backendId: string;
  order: number;
  /** contexts[n] maps a context key of n tokens to token→count. */
  contexts: Record<string, Record<MovementToken, number>>[];
};

/** Internal sentinels — never collide with real tokens (contain a control char). */
const START_TOKEN = "<start>";
const END_TOKEN = "<end>";
const CONTEXT_DELIMITER = "";
const DEFAULT_ORDER = 3;
const DEFAULT_MAX_GENERATE = 256;

/** Turn a recorded action into a stable, learnable movement token. */
export function tokenizeAction(action: TrajectoryAction): MovementToken {
  const tool = slug(action.tool) || "action";
  const meta = action.metadata ?? {};
  const gesture = typeof meta.gesture === "string" ? slug(meta.gesture) : undefined;
  const target =
    typeof meta.target === "string"
      ? slug(meta.target)
      : typeof meta.direction === "string"
        ? slug(meta.direction)
        : undefined;
  if (gesture) {
    return target ? `${tool}:${gesture}:${target}` : `${tool}:${gesture}`;
  }
  const summarySlug = slug(action.summary).split("-").slice(0, 3).join("-");
  return summarySlug ? `${tool}:${summarySlug}` : tool;
}

/** Build a movement dataset from recorded trajectory spans. */
export function buildMovementDatasetFromTrajectories(spans: TrajectorySpan[]): MovementDataset {
  const sequences = spans
    .map<MovementSequence>((span) => ({
      sessionId: span.sessionId,
      trajectoryId: span.id,
      tokens: [...span.actions]
        .sort((a, b) => a.ts - b.ts)
        .map((action) => tokenizeAction(action)),
    }))
    .filter((sequence) => sequence.tokens.length > 0);
  return { version: 1, sequences };
}

/** Build a movement dataset from a replay manifest (uses its `action` events). */
export function buildMovementDatasetFromReplay(manifest: ReplayManifest): MovementDataset {
  const byTrajectory = new Map<string, MovementToken[]>();
  for (const event of manifest.events) {
    if (event.kind !== "action") {
      continue;
    }
    const tokens = byTrajectory.get(event.trajectoryId) ?? [];
    tokens.push(tokenizeAction({ kind: "action", tool: event.tool, summary: event.summary, ts: event.ts }));
    byTrajectory.set(event.trajectoryId, tokens);
  }
  const sequences = [...byTrajectory.entries()].map<MovementSequence>(([trajectoryId, tokens]) => ({
    sessionId: manifest.sessionId,
    trajectoryId,
    tokens,
  }));
  return { version: 1, sequences };
}

class MarkovMovementModel implements MovementModel {
  constructor(
    readonly backendId: string,
    private readonly order: number,
    private readonly contexts: Map<string, Map<MovementToken, number>>[],
  ) {}

  predictNext(prefix: MovementToken[]): MovementPrediction | undefined {
    const padded = [START_TOKEN, ...prefix];
    for (let n = Math.min(this.order, padded.length); n >= 1; n -= 1) {
      const contextTokens = padded.slice(padded.length - n);
      const table = this.contexts[n]?.get(contextKey(contextTokens));
      if (!table || table.size === 0) {
        continue;
      }
      const best = argmax(table);
      if (best) {
        return { token: best.token, probability: best.count / best.total, contextOrder: n };
      }
    }
    return undefined;
  }

  generate(options: MovementGenerateOptions = {}): MovementToken[] {
    const maxLength = options.maxLength ?? DEFAULT_MAX_GENERATE;
    const generated = [...(options.prefix ?? [])];
    while (generated.length < maxLength) {
      const next = this.predictNext(generated);
      if (!next || next.token === END_TOKEN) {
        break;
      }
      generated.push(next.token);
    }
    return generated;
  }

  toJSON(): SerializedMovementModel {
    return {
      version: 1,
      backendId: this.backendId,
      order: this.order,
      contexts: this.contexts.map((table) => {
        const out: Record<string, Record<MovementToken, number>> = {};
        for (const [key, counts] of table.entries()) {
          out[key] = Object.fromEntries(counts.entries());
        }
        return out;
      }),
    };
  }
}

/**
 * Reference deterministic movement backend: a variable-order Markov model with
 * stupid backoff. Serves as the mock the cloud/CI tests train against, and the
 * documented seam a real on-device model implements.
 */
export class MarkovMovementBackend implements MovementModelBackend {
  readonly id = "markov";

  constructor(private readonly defaultOrder: number = DEFAULT_ORDER) {}

  train(dataset: MovementDataset, config: MovementTrainConfig = {}): MovementModel {
    const order = Math.max(1, config.order ?? this.defaultOrder);
    // contexts[n] holds n-token contexts; index 0 is unused.
    const contexts: Map<string, Map<MovementToken, number>>[] = Array.from(
      { length: order + 1 },
      () => new Map<string, Map<MovementToken, number>>(),
    );

    for (const sequence of dataset.sequences) {
      const tokens = [START_TOKEN, ...sequence.tokens, END_TOKEN];
      for (let i = 1; i < tokens.length; i += 1) {
        const next = tokens[i];
        for (let n = 1; n <= order; n += 1) {
          if (i - n < 0) {
            break;
          }
          const context = tokens.slice(i - n, i);
          if (context.length !== n) {
            break;
          }
          increment(contexts[n], contextKey(context), next);
        }
      }
    }

    return new MarkovMovementModel(this.id, order, contexts);
  }

  restore(serialized: SerializedMovementModel): MovementModel {
    const contexts = serialized.contexts.map((table) => {
      const map = new Map<string, Map<MovementToken, number>>();
      for (const [key, counts] of Object.entries(table)) {
        map.set(key, new Map(Object.entries(counts)));
      }
      return map;
    });
    return new MarkovMovementModel(serialized.backendId, serialized.order, contexts);
  }
}

// --- Generalization evaluation ------------------------------------------------

export type MovementEvalResult = {
  /** Held-out sequences scored. */
  sequenceCount: number;
  /** Next-token predictions attempted (includes the terminal END prediction). */
  predictions: number;
  correct: number;
  /** Top-1 next-token accuracy on held-out data ∈ [0, 1]. */
  accuracy: number;
  /** Fraction of held-out sequences reproduced exactly by generate(). */
  exactSequenceMatch: number;
};

/**
 * Measure how well a model generalizes: next-token top-1 accuracy on held-out
 * sequences the model was NOT trained on, plus how many it reproduces exactly.
 * With backoff, a model trained on related movements predicts the correct next
 * token in held-out-but-related sequences well above chance.
 */
export function evaluateMovementModel(model: MovementModel, heldOut: MovementSequence[]): MovementEvalResult {
  let predictions = 0;
  let correct = 0;
  let exact = 0;
  for (const sequence of heldOut) {
    const targets = [...sequence.tokens, END_TOKEN];
    for (let i = 0; i < targets.length; i += 1) {
      const prediction = model.predictNext(sequence.tokens.slice(0, i));
      predictions += 1;
      if (prediction?.token === targets[i]) {
        correct += 1;
      }
    }
    const regenerated = model.generate();
    if (arraysEqual(regenerated, sequence.tokens)) {
      exact += 1;
    }
  }
  return {
    sequenceCount: heldOut.length,
    predictions,
    correct,
    accuracy: predictions === 0 ? 0 : correct / predictions,
    exactSequenceMatch: heldOut.length === 0 ? 0 : exact / heldOut.length,
  };
}

// --- Synthetic movement generator (deterministic, seeded) ---------------------

export type SyntheticMovementOptions = {
  seed: number;
  /**
   * Named "skills" as ordered movement templates. Sessions are sampled from
   * these, with occasional recombination so held-out sessions are related but
   * novel — exactly the generalization target.
   */
  skills: { name: string; steps: MovementToken[] }[];
  sessionCount: number;
  /** Probability [0,1] of splicing two skills into one recombined session. */
  recombineRate?: number;
};

/**
 * Produce a deterministic, reproducible set of synthetic movement sequences to
 * validate the capture→dataset→train→replay loop without any real OS input.
 * Uses a seeded PRNG (no Math.random) so runs are byte-identical.
 */
export function generateSyntheticMovementSessions(options: SyntheticMovementOptions): MovementSequence[] {
  const rng = mulberry32(options.seed >>> 0);
  const recombineRate = options.recombineRate ?? 0;
  const skills = options.skills;
  if (skills.length === 0) {
    return [];
  }
  const sequences: MovementSequence[] = [];
  for (let index = 0; index < options.sessionCount; index += 1) {
    const primary = skills[Math.floor(rng() * skills.length) % skills.length];
    let tokens = [...primary.steps];
    let label = primary.name;
    if (rng() < recombineRate && skills.length > 1) {
      const secondary = skills[Math.floor(rng() * skills.length) % skills.length];
      const cut = 1 + Math.floor(rng() * Math.max(1, tokens.length - 1));
      tokens = [...tokens.slice(0, cut), ...secondary.steps];
      label = `${primary.name}+${secondary.name}`;
    }
    sequences.push({ sessionId: `synthetic-${label}-${index}`, tokens });
  }
  return sequences;
}

// --- helpers ------------------------------------------------------------------

function increment(table: Map<string, Map<MovementToken, number>>, key: string, token: MovementToken): void {
  const counts = table.get(key) ?? new Map<MovementToken, number>();
  counts.set(token, (counts.get(token) ?? 0) + 1);
  table.set(key, counts);
}

function argmax(table: Map<MovementToken, number>): { token: MovementToken; count: number; total: number } | undefined {
  let bestToken: MovementToken | undefined;
  let bestCount = -1;
  let total = 0;
  for (const [token, count] of table.entries()) {
    total += count;
    // Deterministic tie-break: higher count wins; equal counts break by token order.
    if (count > bestCount || (count === bestCount && bestToken !== undefined && token < bestToken)) {
      bestToken = token;
      bestCount = count;
    }
  }
  return bestToken === undefined ? undefined : { token: bestToken, count: bestCount, total };
}

function contextKey(tokens: MovementToken[]): string {
  return tokens.join(CONTEXT_DELIMITER);
}

function slug(value: string): string {
  return value
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
}

function arraysEqual(a: MovementToken[], b: MovementToken[]): boolean {
  return a.length === b.length && a.every((value, index) => value === b[index]);
}

/** Small seeded PRNG (mulberry32) — deterministic, no global RNG state. */
function mulberry32(seed: number): () => number {
  let state = seed;
  return () => {
    state |= 0;
    state = (state + 0x6d2b79f5) | 0;
    let t = Math.imul(state ^ (state >>> 15), 1 | state);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
