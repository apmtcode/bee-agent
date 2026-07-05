import type { ReplayTimelineEvent } from "../capture/replay.js";
import type { TrajectoryAction } from "../capture/trajectory.js";

/**
 * Local-movement learning: pluggable model backend.
 *
 * This module gives bee-agent the "post-train a local model on recorded
 * movements + generalize" half of the local-movement subsystem (standing
 * objective #2c/#2d). Recording, dataset export and replay already live in
 * `src/capture` and `src/training`; what was missing was an in-process,
 * *trainable* policy that the cloud engine can validate with synthetic
 * streams and that a real on-device backend (MLX/axolotl) can later drop
 * into via the same {@link MovementModelBackend} interface.
 *
 * The reference backend here is a deterministic order-k Markov policy with
 * stupid-backoff. It has two properties the objective calls for:
 *   - **Reproduce recorded movements** — given a prefix seen in training it
 *     replays the exact recorded continuation (argmax, deterministic ties),
 *     terminating on a learned end-of-sequence sentinel.
 *   - **Generalize to related-but-new movements** — for a novel higher-order
 *     context it backs off to the shortest matching context (down to the
 *     global next-token distribution), so learned local transitions transfer
 *     to sequences never seen verbatim. Predictions flag `novel`/`backoffDepth`
 *     so callers can see generalization happening.
 *
 * Everything is deterministic (no RNG, no clock) so it is safe to unit-test
 * in the cloud with simulated event streams.
 */

/** A single normalized movement/action step (mouse/keyboard/gesture/tool). */
export interface MovementToken {
  /** Originating tool/surface, e.g. "device", "browser", "keyboard". */
  tool: string;
  /** Normalized verb for the movement, e.g. "tap", "swipe:down", "type". */
  action: string;
  /** Optional target the movement acted on, e.g. "submit-button". */
  target?: string;
}

/** A training corpus: an ordered set of movement sequences. */
export interface MovementDataset {
  sequences: MovementToken[][];
  sampleCount: number;
  tokenCount: number;
}

export interface MovementTrainConfig {
  /** Markov context length. Higher = more faithful replay, less generalization. */
  order?: number;
}

/** One scored continuation candidate. */
export interface MovementCandidate {
  token: MovementToken;
  probability: number;
}

export interface MovementPrediction {
  /** Best next movement, or `undefined` if the model is empty. */
  token: MovementToken | undefined;
  /** Conditional probability of `token` within the backed-off context. */
  probability: number;
  /** How many context tokens were dropped to find a match (0 = exact order). */
  backoffDepth: number;
  /** True when the prediction generalized beyond the exact recorded context. */
  novel: boolean;
  /** All candidates for the matched context, best first. */
  candidates: MovementCandidate[];
}

/** Serialized, JSON-safe form of a trained model. */
export interface SerializedMovementModel {
  version: 1;
  backendId: string;
  order: number;
  /** contextKey -> (tokenKey -> count). Empty contextKey is the unigram. */
  transitions: Record<string, Record<string, number>>;
  /** tokenKey -> token, so tokens can be reconstructed on load. */
  vocabulary: Record<string, MovementToken>;
}

export interface MovementModel {
  readonly backendId: string;
  readonly order: number;
  /** Predict the next movement given the trailing context (may be empty). */
  predictNext(context: MovementToken[]): MovementPrediction;
  /**
   * Autoregressively roll out movements from a seed until the learned
   * end-of-sequence sentinel is reached or `maxSteps` is hit.
   */
  generate(seed: MovementToken[], options?: { maxSteps?: number }): MovementToken[];
  serialize(): SerializedMovementModel;
}

/**
 * A pluggable local-model backend. The Markov reference is provided; a real
 * on-device backend (MLX/axolotl) implements the same shape — `train` is async
 * so heavyweight backends can shell out, while inference stays synchronous.
 */
export interface MovementModelBackend {
  readonly id: string;
  train(dataset: MovementDataset, config?: MovementTrainConfig): Promise<MovementModel>;
  load(model: SerializedMovementModel): MovementModel;
}

const FIELD_SEPARATOR = "\u0001";
const CONTEXT_SEPARATOR = "\u0002";
/** Sentinel appended to every training sequence so the model learns to stop. */
const END_TOKEN_KEY = "\u0000END";

export const DEFAULT_MOVEMENT_ORDER = 2;

/** Stable string key for a movement token (used for counts + vocab). */
export function movementTokenKey(token: MovementToken): string {
  return `${token.tool}${FIELD_SEPARATOR}${token.action}${FIELD_SEPARATOR}${token.target ?? ""}`;
}

function contextKey(tokens: MovementToken[]): string {
  return tokens.map(movementTokenKey).join(CONTEXT_SEPARATOR);
}

/** Build a dataset from raw movement sequences (filters out empty ones). */
export function buildMovementDataset(sequences: MovementToken[][]): MovementDataset {
  const nonEmpty = sequences.filter((sequence) => sequence.length > 0);
  return {
    sequences: nonEmpty.map((sequence) => sequence.map((token) => ({ ...token }))),
    sampleCount: nonEmpty.length,
    tokenCount: nonEmpty.reduce((total, sequence) => total + sequence.length, 0),
  };
}

/** Normalize a captured action into a movement token. */
export function movementTokenFromAction(
  action: Pick<TrajectoryAction, "tool" | "summary"> & { metadata?: Record<string, unknown> },
): MovementToken {
  const metadata = action.metadata ?? {};
  const gesture = typeof metadata.gesture === "string" ? metadata.gesture : undefined;
  const direction = typeof metadata.direction === "string" ? metadata.direction : undefined;
  const target =
    (typeof metadata.target === "string" ? metadata.target : undefined) ??
    (typeof metadata.valueSummary === "string" ? metadata.valueSummary : undefined);

  const verb = gesture ?? firstWord(action.summary);
  const composed = direction ? `${verb}:${direction}` : verb;
  return {
    tool: action.tool,
    action: composed || "act",
    ...(target ? { target } : {}),
  };
}

/** Extract per-trajectory movement sequences from replay manifests. */
export function buildMovementDatasetFromReplays(
  replays: Array<{ events: ReplayTimelineEvent[] }>,
): MovementDataset {
  type ReplayActionEvent = Extract<ReplayTimelineEvent, { kind: "action" }>;
  const sequences: MovementToken[][] = [];
  for (const replay of replays) {
    const byTrajectory = new Map<string, ReplayActionEvent[]>();
    for (const event of replay.events) {
      if (event.kind !== "action") {
        continue;
      }
      const bucket = byTrajectory.get(event.trajectoryId) ?? [];
      bucket.push(event);
      byTrajectory.set(event.trajectoryId, bucket);
    }
    for (const events of byTrajectory.values()) {
      const ordered = [...events].sort((a, b) => a.ts - b.ts);
      sequences.push(ordered.map((event) => movementTokenFromAction({ tool: event.tool, summary: event.summary })));
    }
  }
  return buildMovementDataset(sequences);
}

function firstWord(summary: string): string {
  const trimmed = summary.trim().toLowerCase();
  const space = trimmed.indexOf(" ");
  return space === -1 ? trimmed : trimmed.slice(0, space);
}

class MarkovMovementModel implements MovementModel {
  constructor(
    readonly backendId: string,
    readonly order: number,
    private readonly transitions: Map<string, Map<string, number>>,
    private readonly vocabulary: Map<string, MovementToken>,
  ) {}

  predictNext(context: MovementToken[]): MovementPrediction {
    const requestedOrder = Math.min(this.order, context.length);
    for (let used = requestedOrder; used >= 0; used--) {
      const key = contextKey(context.slice(context.length - used));
      const counts = this.transitions.get(key);
      if (!counts) {
        continue;
      }
      const candidates = this.rankCandidates(counts);
      const best = candidates[0];
      const backoffDepth = requestedOrder - used;
      return {
        token: best?.token,
        probability: best?.probability ?? 0,
        backoffDepth,
        novel: backoffDepth > 0,
        candidates,
      };
    }
    return { token: undefined, probability: 0, backoffDepth: requestedOrder, novel: true, candidates: [] };
  }

  generate(seed: MovementToken[], options: { maxSteps?: number } = {}): MovementToken[] {
    const maxSteps = options.maxSteps ?? 64;
    const produced: MovementToken[] = [];
    const context = [...seed];
    for (let step = 0; step < maxSteps; step++) {
      const nextKey = this.pickNextKey(context);
      if (nextKey === undefined || nextKey === END_TOKEN_KEY) {
        break;
      }
      const token = this.vocabulary.get(nextKey);
      if (!token) {
        break;
      }
      produced.push({ ...token });
      context.push(token);
    }
    return produced;
  }

  serialize(): SerializedMovementModel {
    const transitions: Record<string, Record<string, number>> = {};
    for (const [key, counts] of this.transitions) {
      transitions[key] = Object.fromEntries(counts);
    }
    const vocabulary: Record<string, MovementToken> = {};
    for (const [key, token] of this.vocabulary) {
      vocabulary[key] = { ...token };
    }
    return { version: 1, backendId: this.backendId, order: this.order, transitions, vocabulary };
  }

  /** Raw next-key pick (includes the END sentinel) used by generate(). */
  private pickNextKey(context: MovementToken[]): string | undefined {
    const requestedOrder = Math.min(this.order, context.length);
    for (let used = requestedOrder; used >= 0; used--) {
      const key = contextKey(context.slice(context.length - used));
      const counts = this.transitions.get(key);
      if (!counts) {
        continue;
      }
      return argmaxKey(counts);
    }
    return undefined;
  }

  /** Rank real (non-sentinel) continuations by probability, deterministic ties. */
  private rankCandidates(counts: Map<string, number>): MovementCandidate[] {
    const total = [...counts.values()].reduce((sum, count) => sum + count, 0);
    return [...counts.entries()]
      .filter(([key]) => key !== END_TOKEN_KEY)
      .map(([key, count]) => ({ key, count }))
      .sort((a, b) => (b.count !== a.count ? b.count - a.count : a.key < b.key ? -1 : 1))
      .map(({ key, count }) => ({
        token: this.vocabulary.get(key) as MovementToken,
        probability: total === 0 ? 0 : count / total,
      }));
  }
}

/** Deterministic argmax over a count map; ties broken by smallest key. */
function argmaxKey(counts: Map<string, number>): string | undefined {
  let bestKey: string | undefined;
  let bestCount = -Infinity;
  for (const [key, count] of counts) {
    if (count > bestCount || (count === bestCount && bestKey !== undefined && key < bestKey)) {
      bestKey = key;
      bestCount = count;
    }
  }
  return bestKey;
}

/**
 * Reference deterministic backend: an order-k Markov policy with stupid-backoff.
 * Trains purely in-process, so the cloud engine can validate the full
 * train → reproduce → generalize loop with synthetic movement streams.
 */
export class MarkovMovementBackend implements MovementModelBackend {
  readonly id = "markov-backoff";

  async train(dataset: MovementDataset, config: MovementTrainConfig = {}): Promise<MovementModel> {
    const order = Math.max(1, config.order ?? DEFAULT_MOVEMENT_ORDER);
    const transitions = new Map<string, Map<string, number>>();
    const vocabulary = new Map<string, MovementToken>();

    for (const sequence of dataset.sequences) {
      // Append the end sentinel so the model learns where sequences stop.
      const tokenKeys = sequence.map((token) => {
        const key = movementTokenKey(token);
        if (!vocabulary.has(key)) {
          vocabulary.set(key, { ...token });
        }
        return key;
      });
      tokenKeys.push(END_TOKEN_KEY);

      for (let i = 0; i < tokenKeys.length; i++) {
        const nextKey = tokenKeys[i];
        const maxContext = Math.min(order, i);
        for (let used = 0; used <= maxContext; used++) {
          const key = tokenKeys.slice(i - used, i).join(CONTEXT_SEPARATOR);
          increment(transitions, key, nextKey);
        }
      }
    }

    return new MarkovMovementModel(this.id, order, transitions, vocabulary);
  }

  load(model: SerializedMovementModel): MovementModel {
    const transitions = new Map<string, Map<string, number>>();
    for (const [key, counts] of Object.entries(model.transitions)) {
      transitions.set(key, new Map(Object.entries(counts)));
    }
    const vocabulary = new Map<string, MovementToken>();
    for (const [key, token] of Object.entries(model.vocabulary)) {
      vocabulary.set(key, { ...token });
    }
    return new MarkovMovementModel(model.backendId, model.order, transitions, vocabulary);
  }
}

function increment(transitions: Map<string, Map<string, number>>, contextKey: string, tokenKey: string): void {
  const counts = transitions.get(contextKey) ?? new Map<string, number>();
  counts.set(tokenKey, (counts.get(tokenKey) ?? 0) + 1);
  transitions.set(contextKey, counts);
}
