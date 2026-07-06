import type { TrajectoryAction, TrajectorySpan } from "../capture/trajectory.js";

/**
 * Local-movement learning model.
 *
 * This is the in-process, deterministic learning core for the local-movement
 * subsystem (standing objective #2c/#2d). The capture pipeline records device
 * gestures / OS events into {@link TrajectorySpan}s; this module distils those
 * into a normalized movement-token dataset, learns a policy over them, and can
 * (c) *replay* recorded movements and (d) *generalize* to new-but-related
 * movements it never saw verbatim.
 *
 * The learning backend is pluggable via {@link MovementModelBackend} so the
 * default deterministic in-process model can be swapped for a real on-device
 * small model when bee-agent runs locally. The default
 * {@link MarkovMovementBackend} needs no OS access and no randomness, so it is
 * validated entirely with synthetic event streams in the cloud.
 */

/** A single normalized movement — the atomic unit the model learns over. */
export type MovementToken = {
  /** The tool/surface the movement happened on, e.g. "device" or "browser". */
  tool: string;
  /** The gesture class, e.g. "tap" | "swipe" | "scroll" | "type" | "shortcut". */
  gesture: string;
  /** Optional target the movement acted on (a button, field, element…). */
  target?: string;
  /** Optional direction for directional gestures. */
  direction?: string;
};

/** An ordered run of movements distilled from one trajectory. */
export type MovementSequence = {
  trajectoryId: string;
  sessionId: string;
  tokens: MovementToken[];
};

/** The replayable dataset the model trains on. */
export type MovementDataset = {
  version: 1;
  sequences: MovementSequence[];
  /** Distinct encoded tokens observed, sorted — the model's vocabulary. */
  vocabulary: string[];
};

export type MovementPrediction = {
  token: MovementToken;
  /** Normalized score in (0, 1] among the candidates at the winning context. */
  probability: number;
  /** How many context tokens were dropped to find a prediction (0 = exact). */
  backoffDistance: number;
  /**
   * True when the prediction came from generalizing across targets (the
   * concrete target-specific context was unseen but the gesture class was
   * known). This is the "new but related movement" path.
   */
  generalized: boolean;
};

/** A trained, queryable movement policy. */
export interface MovementPolicy {
  readonly backendId: string;
  /** Predict the next movement given a (possibly empty) recent context. */
  predictNext(context: MovementToken[]): MovementPrediction | undefined;
  /** Greedily roll out a movement sequence from a seed context. */
  generate(seed: MovementToken[], options?: { maxSteps?: number }): MovementToken[];
}

/** Pluggable learning backend — swap for a real on-device model locally. */
export interface MovementModelBackend {
  readonly id: string;
  train(dataset: MovementDataset, options?: MovementTrainOptions): MovementPolicy;
}

export type MovementTrainOptions = {
  /** Max context length the model conditions on (n-gram order). Default 3. */
  order?: number;
};

const TOKEN_FIELD_SEP = "";

/** Stable, reversible string encoding of a movement token. */
export function encodeMovementToken(token: MovementToken): string {
  return [token.tool, token.gesture, token.target ?? "", token.direction ?? ""].join(TOKEN_FIELD_SEP);
}

export function decodeMovementToken(encoded: string): MovementToken {
  const [tool = "", gesture = "", target = "", direction = ""] = encoded.split(TOKEN_FIELD_SEP);
  return {
    tool,
    gesture,
    ...(target ? { target } : {}),
    ...(direction ? { direction } : {}),
  };
}

/**
 * The class-level encoding drops the concrete target so movements that differ
 * only in *what* they acted on collapse together. This is the generalization
 * axis: "tap Save" and "tap Send" share a class and can substitute for each
 * other when the exact target was never observed.
 */
export function encodeMovementClass(token: MovementToken): string {
  return [token.tool, token.gesture, token.direction ?? ""].join(TOKEN_FIELD_SEP);
}

/** Distil a single trajectory's actions into an ordered movement sequence. */
export function extractMovementTokens(trajectory: TrajectorySpan): MovementToken[] {
  const source = trajectory.review?.redactedActions
    ? trajectory.review.redactedActions.map((action) => ({
        kind: "action" as const,
        tool: action.tool,
        summary: action.summary,
        ts: action.ts,
      }))
    : trajectory.actions;

  return [...source]
    .sort((a, b) => a.ts - b.ts)
    .map((action) => toMovementToken(action));
}

function toMovementToken(action: Pick<TrajectoryAction, "tool" | "summary" | "metadata">): MovementToken {
  const metadata = action.metadata ?? {};
  const gesture = typeof metadata.gesture === "string" ? metadata.gesture : "action";
  const target = typeof metadata.target === "string" ? metadata.target : undefined;
  const direction = typeof metadata.direction === "string" ? metadata.direction : undefined;
  return {
    tool: action.tool,
    gesture,
    ...(target ? { target } : {}),
    ...(direction ? { direction } : {}),
  };
}

/** Build the replayable movement dataset from recorded trajectories. */
export function buildMovementDataset(trajectories: TrajectorySpan[]): MovementDataset {
  const sequences: MovementSequence[] = trajectories
    .map((trajectory) => ({
      trajectoryId: trajectory.id,
      sessionId: trajectory.sessionId,
      tokens: extractMovementTokens(trajectory),
    }))
    .filter((sequence) => sequence.tokens.length > 0);

  const vocabulary = [
    ...new Set(sequences.flatMap((sequence) => sequence.tokens.map(encodeMovementToken))),
  ].sort();

  return { version: 1, sequences, vocabulary };
}

type CountTable = Map<string, Map<string, number>>;

/**
 * Deterministic order-N Markov backend with two independent back-off ladders:
 *
 *  1. **Context back-off** — if the full recent context was never seen, drop
 *     the oldest context token and retry, down to the unigram distribution.
 *     This lets the model respond to novel *situations*.
 *  2. **Class back-off** — the transition tables are keyed twice: once on exact
 *     tokens and once on the target-agnostic movement *class*. When the exact
 *     next token is ambiguous or the context only matches at the class level,
 *     the model still produces a concrete, plausible movement — generalizing
 *     across targets to perform new-but-related movements.
 *
 * No randomness is used: ties break on the encoded token string, so training
 * and inference are fully reproducible in cloud tests.
 */
export class MarkovMovementBackend implements MovementModelBackend {
  readonly id = "markov";

  train(dataset: MovementDataset, options: MovementTrainOptions = {}): MovementPolicy {
    const order = Math.max(1, options.order ?? 3);
    const exact: CountTable[] = Array.from({ length: order + 1 }, () => new Map());
    const byClass: CountTable[] = Array.from({ length: order + 1 }, () => new Map());

    for (const sequence of dataset.sequences) {
      const tokens = sequence.tokens;
      for (let i = 0; i < tokens.length; i += 1) {
        const next = tokens[i];
        const nextEncoded = encodeMovementToken(next);
        for (let k = 0; k <= order; k += 1) {
          if (i - k < 0) {
            break;
          }
          const contextTokens = tokens.slice(i - k, i);
          increment(exact[k], contextKey(contextTokens.map(encodeMovementToken)), nextEncoded);
          increment(byClass[k], contextKey(contextTokens.map(encodeMovementClass)), nextEncoded);
        }
      }
    }

    return new MarkovMovementPolicy(this.id, order, exact, byClass);
  }
}

class MarkovMovementPolicy implements MovementPolicy {
  constructor(
    readonly backendId: string,
    private readonly order: number,
    private readonly exact: CountTable[],
    private readonly byClass: CountTable[],
  ) {}

  predictNext(context: MovementToken[]): MovementPrediction | undefined {
    const maxK = Math.min(this.order, context.length);
    for (let k = maxK; k >= 0; k -= 1) {
      const recent = context.slice(context.length - k);

      const exactRow = this.exact[k].get(contextKey(recent.map(encodeMovementToken)));
      if (exactRow && exactRow.size > 0) {
        return this.pick(exactRow, context.length - k, false);
      }

      const classRow = this.byClass[k].get(contextKey(recent.map(encodeMovementClass)));
      if (classRow && classRow.size > 0) {
        return this.pick(classRow, context.length - k, k > 0);
      }
    }
    return undefined;
  }

  generate(seed: MovementToken[], options: { maxSteps?: number } = {}): MovementToken[] {
    const maxSteps = Math.max(0, options.maxSteps ?? 16);
    const out: MovementToken[] = [];
    let context = [...seed];
    for (let step = 0; step < maxSteps; step += 1) {
      const prediction = this.predictNext(context);
      if (!prediction) {
        break;
      }
      out.push(prediction.token);
      context = [...context, prediction.token];
    }
    return out;
  }

  private pick(row: Map<string, number>, backoffDistance: number, generalized: boolean): MovementPrediction {
    let total = 0;
    let best: { encoded: string; count: number } | undefined;
    for (const [encoded, count] of row) {
      total += count;
      if (!best || count > best.count || (count === best.count && encoded < best.encoded)) {
        best = { encoded, count };
      }
    }
    // `best` is defined: callers only invoke pick() on non-empty rows.
    const winner = best as { encoded: string; count: number };
    return {
      token: decodeMovementToken(winner.encoded),
      probability: winner.count / total,
      backoffDistance,
      generalized,
    };
  }
}

function contextKey(encodedTokens: string[]): string {
  return encodedTokens.join("");
}

function increment(table: CountTable, context: string, next: string): void {
  let row = table.get(context);
  if (!row) {
    row = new Map();
    table.set(context, row);
  }
  row.set(next, (row.get(next) ?? 0) + 1);
}

export type MovementEvalReport = {
  /** Held-out next-movement decisions scored. */
  steps: number;
  /** Predictions whose full token (tool+gesture+target+direction) matched. */
  exactMatches: number;
  /** Predictions whose movement *class* matched (target may differ). */
  classMatches: number;
  exactAccuracy: number;
  classAccuracy: number;
  /** Fraction of scored steps whose prediction came via class generalization. */
  generalizationRate: number;
  /** Fraction of scored steps where the policy produced no prediction. */
  abstentionRate: number;
};

/**
 * Generalization eval: for each held-out sequence, walk its prefixes and ask
 * the policy to predict the true next movement. `exactAccuracy` measures verbatim
 * replay fidelity; `classAccuracy` measures whether the model captured the
 * *shape* of the movement even when it substituted a related target — the
 * signal that it generalizes rather than merely memorizes.
 */
export function evaluateMovementPolicy(
  policy: MovementPolicy,
  heldOut: MovementSequence[],
): MovementEvalReport {
  let steps = 0;
  let exactMatches = 0;
  let classMatches = 0;
  let generalized = 0;
  let abstentions = 0;

  for (const sequence of heldOut) {
    for (let i = 0; i < sequence.tokens.length; i += 1) {
      steps += 1;
      const expected = sequence.tokens[i];
      const prediction = policy.predictNext(sequence.tokens.slice(0, i));
      if (!prediction) {
        abstentions += 1;
        continue;
      }
      if (prediction.generalized) {
        generalized += 1;
      }
      if (encodeMovementToken(prediction.token) === encodeMovementToken(expected)) {
        exactMatches += 1;
      }
      if (encodeMovementClass(prediction.token) === encodeMovementClass(expected)) {
        classMatches += 1;
      }
    }
  }

  return {
    steps,
    exactMatches,
    classMatches,
    exactAccuracy: steps === 0 ? 0 : exactMatches / steps,
    classAccuracy: steps === 0 ? 0 : classMatches / steps,
    generalizationRate: steps === 0 ? 0 : generalized / steps,
    abstentionRate: steps === 0 ? 0 : abstentions / steps,
  };
}

/**
 * Convenience: train the default deterministic backend on recorded
 * trajectories and return the policy. Real deployments pass a custom
 * {@link MovementModelBackend} for on-device training.
 */
export function trainMovementPolicy(
  trajectories: TrajectorySpan[],
  options: MovementTrainOptions & { backend?: MovementModelBackend } = {},
): { dataset: MovementDataset; policy: MovementPolicy } {
  const backend = options.backend ?? new MarkovMovementBackend();
  const dataset = buildMovementDataset(trajectories);
  const policy = backend.train(dataset, options);
  return { dataset, policy };
}
