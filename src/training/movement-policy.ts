import type { TrajectoryAction, TrajectorySpan } from "../capture/trajectory.js";
import type { ReplayManifest, ReplayTimelineEvent } from "../capture/replay.js";

/**
 * Local-movement learning: in-process policy model.
 *
 * Objective #2 (c)+(d): post-train a *local* model on recorded movements so it
 * can (c) repeat the recorded movements and (d) generalize to new-but-related
 * movements. bee-agent runs in the cloud with no access to the user's machine,
 * so this module is the trainable/inferable *code and schema*; the real
 * on-device backend plugs in behind {@link MovementPolicyBackend}. The default
 * backend is a deterministic backoff n-gram model — no native deps, no
 * randomness — so it trains and predicts identically in cloud CI and on-device.
 */

/** A single canonical movement, derived from a captured trajectory action. */
export type MovementToken = {
  /** Tool/surface that produced the movement (e.g. "device", "browser"). */
  tool: string;
  /** Gesture/verb (e.g. "tap", "swipe", "scroll", "type", "shortcut"). */
  gesture: string;
  /** Optional UI target the movement acted on. */
  target?: string;
  /** Optional spatial direction for swipes/scrolls. */
  direction?: string;
};

/** An ordered movement sequence extracted from one trajectory/session. */
export type MovementSequence = {
  /** Source identifier (trajectory or session id) — for provenance only. */
  source: string;
  tokens: MovementToken[];
};

export type MovementDataset = {
  sequences: MovementSequence[];
};

export type MovementTrainingOptions = {
  /**
   * Highest context order (number of preceding tokens conditioned on). The
   * model backs off from this order down to 0 (the marginal distribution).
   */
  order?: number;
};

export const DEFAULT_MOVEMENT_ORDER = 3;

export type MovementCandidate = {
  token: MovementToken;
  probability: number;
  count: number;
};

export type MovementPrediction = {
  token: MovementToken;
  probability: number;
  /** Context length actually used (equals full context when not backed off). */
  contextOrder: number;
  /** True when the full context was unseen and a shorter suffix was used. */
  backoff: boolean;
  /** Ranked alternatives from the same context order (best first). */
  candidates: MovementCandidate[];
};

/** Serialized model state — portable across processes/backends. */
export type MovementModelSnapshot = {
  version: 1;
  backendId: string;
  order: number;
  /** contextKey -> (tokenKey -> count). Empty context key = order-0 marginal. */
  counts: Record<string, Record<string, number>>;
  /** tokenKey -> token, so predictions can be rehydrated to structured tokens. */
  vocabulary: Record<string, MovementToken>;
  observedSequences: number;
  observedTokens: number;
};

/** Pluggable local-model backend seam. Real on-device backends implement this. */
export interface MovementPolicyBackend {
  readonly id: string;
  train(dataset: MovementDataset, options?: MovementTrainingOptions): MovementPolicyModel;
  restore(snapshot: MovementModelSnapshot): MovementPolicyModel;
}

export interface MovementPolicyModel {
  readonly backendId: string;
  readonly order: number;
  /** Predict the single most-likely next movement for a context, or undefined. */
  predictNext(context: MovementToken[]): MovementPrediction | undefined;
  /** Roll out `steps` movements from a seed context (greedy, deterministic). */
  generate(seed: MovementToken[], steps: number): MovementToken[];
  serialize(): MovementModelSnapshot;
}

const FIELD_SEP = "";
const CONTEXT_SEP = "";

export function movementTokenKey(token: MovementToken): string {
  return [token.tool, token.gesture, token.target ?? "", token.direction ?? ""].join(FIELD_SEP);
}

function contextKey(context: MovementToken[]): string {
  return context.map(movementTokenKey).join(CONTEXT_SEP);
}

/** Derive a canonical movement token from a captured trajectory action. */
export function deriveMovementToken(action: TrajectoryAction): MovementToken {
  const metadata = action.metadata ?? {};
  const gesture = pickString(metadata.gesture) ?? pickString(metadata.action) ?? action.tool;
  const target = pickString(metadata.target);
  const direction = pickString(metadata.direction);
  return {
    tool: action.tool,
    gesture,
    ...(target ? { target } : {}),
    ...(direction ? { direction } : {}),
  };
}

function pickString(value: unknown): string | undefined {
  return typeof value === "string" && value.length > 0 ? value : undefined;
}

/** Build a movement dataset from captured trajectory spans. */
export function buildMovementDataset(trajectories: TrajectorySpan[]): MovementDataset {
  const sequences = trajectories.map((trajectory) => ({
    source: trajectory.id,
    tokens: [...trajectory.actions]
      .sort((a, b) => a.ts - b.ts)
      .map((action) => deriveMovementToken(action)),
  }));
  return { sequences: sequences.filter((sequence) => sequence.tokens.length > 0) };
}

/** Build a movement dataset from a reviewed replay manifest (action events). */
export function buildMovementDatasetFromReplays(replays: ReplayManifest[]): MovementDataset {
  const sequences = replays.map((replay) => ({
    source: replay.sessionId,
    tokens: replay.events
      .filter((event): event is Extract<ReplayTimelineEvent, { kind: "action" }> => event.kind === "action")
      .map((event) => ({ tool: event.tool, gesture: parseGestureFromSummary(event.summary, event.tool) })),
  }));
  return { sequences: sequences.filter((sequence) => sequence.tokens.length > 0) };
}

function parseGestureFromSummary(summary: string, fallback: string): string {
  const verb = summary.trim().split(/\s+/)[0];
  return verb ? verb.toLowerCase() : fallback;
}

class BackoffNgramMovementModel implements MovementPolicyModel {
  constructor(
    readonly backendId: string,
    readonly order: number,
    private readonly counts: Map<string, Map<string, number>>,
    private readonly vocabulary: Map<string, MovementToken>,
    private readonly observedSequences: number,
    private readonly observedTokens: number,
  ) {}

  predictNext(context: MovementToken[]): MovementPrediction | undefined {
    for (let used = Math.min(context.length, this.order); used >= 0; used -= 1) {
      const suffix = used === 0 ? [] : context.slice(context.length - used);
      const distribution = this.counts.get(contextKey(suffix));
      if (!distribution || distribution.size === 0) {
        continue;
      }
      const candidates = this.rankCandidates(distribution);
      const best = candidates[0];
      if (!best) {
        continue;
      }
      return {
        token: best.token,
        probability: best.probability,
        contextOrder: used,
        backoff: used < Math.min(context.length, this.order),
        candidates,
      };
    }
    return undefined;
  }

  generate(seed: MovementToken[], steps: number): MovementToken[] {
    const produced: MovementToken[] = [];
    let context = [...seed];
    for (let step = 0; step < steps; step += 1) {
      const prediction = this.predictNext(context);
      if (!prediction) {
        break;
      }
      produced.push(prediction.token);
      context = [...context, prediction.token].slice(-this.order);
    }
    return produced;
  }

  serialize(): MovementModelSnapshot {
    const counts: Record<string, Record<string, number>> = {};
    for (const [context, distribution] of this.counts) {
      counts[context] = Object.fromEntries(distribution);
    }
    const vocabulary: Record<string, MovementToken> = {};
    for (const [key, token] of this.vocabulary) {
      vocabulary[key] = token;
    }
    return {
      version: 1,
      backendId: this.backendId,
      order: this.order,
      counts,
      vocabulary,
      observedSequences: this.observedSequences,
      observedTokens: this.observedTokens,
    };
  }

  private rankCandidates(distribution: Map<string, number>): MovementCandidate[] {
    const total = [...distribution.values()].reduce((sum, count) => sum + count, 0);
    return [...distribution.entries()]
      .map(([tokenKey, count]) => ({
        token: this.vocabulary.get(tokenKey) ?? parseTokenKey(tokenKey),
        count,
        probability: total > 0 ? count / total : 0,
      }))
      // Deterministic ordering: higher count first, then stable key sort for ties.
      .sort((a, b) => (b.count - a.count) || movementTokenKey(a.token).localeCompare(movementTokenKey(b.token)));
  }
}

function parseTokenKey(key: string): MovementToken {
  const [tool = "", gesture = "", target = "", direction = ""] = key.split(FIELD_SEP);
  return {
    tool,
    gesture,
    ...(target ? { target } : {}),
    ...(direction ? { direction } : {}),
  };
}

/**
 * Deterministic backoff n-gram movement backend. The reference/default local
 * backend: no native deps, no randomness — identical results in cloud CI and
 * on-device. Swap in a neural backend by implementing {@link MovementPolicyBackend}.
 */
export class NgramMovementPolicyBackend implements MovementPolicyBackend {
  readonly id = "ngram-backoff";

  train(dataset: MovementDataset, options: MovementTrainingOptions = {}): MovementPolicyModel {
    const order = Math.max(1, options.order ?? DEFAULT_MOVEMENT_ORDER);
    const counts = new Map<string, Map<string, number>>();
    const vocabulary = new Map<string, MovementToken>();
    let observedTokens = 0;

    for (const sequence of dataset.sequences) {
      for (let index = 0; index < sequence.tokens.length; index += 1) {
        const next = sequence.tokens[index]!;
        const nextKey = movementTokenKey(next);
        vocabulary.set(nextKey, next);
        observedTokens += 1;
        // Record next given every context suffix from order down to 0.
        for (let contextLength = 0; contextLength <= order; contextLength += 1) {
          if (contextLength > index) {
            break;
          }
          const context = contextLength === 0 ? [] : sequence.tokens.slice(index - contextLength, index);
          const key = contextKey(context);
          let distribution = counts.get(key);
          if (!distribution) {
            distribution = new Map<string, number>();
            counts.set(key, distribution);
          }
          distribution.set(nextKey, (distribution.get(nextKey) ?? 0) + 1);
        }
      }
    }

    return new BackoffNgramMovementModel(
      this.id,
      order,
      counts,
      vocabulary,
      dataset.sequences.length,
      observedTokens,
    );
  }

  restore(snapshot: MovementModelSnapshot): MovementPolicyModel {
    const counts = new Map<string, Map<string, number>>();
    for (const [context, distribution] of Object.entries(snapshot.counts)) {
      counts.set(context, new Map(Object.entries(distribution)));
    }
    const vocabulary = new Map<string, MovementToken>(Object.entries(snapshot.vocabulary));
    return new BackoffNgramMovementModel(
      snapshot.backendId,
      snapshot.order,
      counts,
      vocabulary,
      snapshot.observedSequences,
      snapshot.observedTokens,
    );
  }
}

const BACKEND_REGISTRY = new Map<string, () => MovementPolicyBackend>([
  ["ngram-backoff", () => new NgramMovementPolicyBackend()],
]);

/** Register a pluggable movement-policy backend (e.g. a real on-device model). */
export function registerMovementPolicyBackend(id: string, factory: () => MovementPolicyBackend): void {
  BACKEND_REGISTRY.set(id, factory);
}

/** Resolve a movement-policy backend by id. Defaults to the n-gram backend. */
export function createMovementPolicyBackend(id = "ngram-backoff"): MovementPolicyBackend {
  const factory = BACKEND_REGISTRY.get(id);
  if (!factory) {
    throw new Error(`Unknown movement-policy backend: ${id}`);
  }
  return factory();
}

export type MovementGeneralizationCase = {
  context: MovementToken[];
  expected: MovementToken;
};

export type MovementGeneralizationReport = {
  total: number;
  matched: number;
  backoffMatched: number;
  accuracy: number;
  results: {
    expected: MovementToken;
    predicted?: MovementToken;
    matched: boolean;
    backoff: boolean;
    contextOrder: number;
  }[];
};

/**
 * Generalization eval harness (objective #2 d): measure how well a trained
 * model predicts held-out but related movements, tracking how many correct
 * predictions required backoff (i.e. were genuine generalizations, not memory).
 */
export function evaluateMovementGeneralization(
  model: MovementPolicyModel,
  cases: MovementGeneralizationCase[],
): MovementGeneralizationReport {
  const results = cases.map((testCase) => {
    const prediction = model.predictNext(testCase.context);
    const matched =
      prediction !== undefined && movementTokenKey(prediction.token) === movementTokenKey(testCase.expected);
    return {
      expected: testCase.expected,
      predicted: prediction?.token,
      matched,
      backoff: prediction?.backoff ?? false,
      contextOrder: prediction?.contextOrder ?? 0,
    };
  });
  const matched = results.filter((result) => result.matched).length;
  const backoffMatched = results.filter((result) => result.matched && result.backoff).length;
  return {
    total: cases.length,
    matched,
    backoffMatched,
    accuracy: cases.length > 0 ? matched / cases.length : 0,
    results,
  };
}
