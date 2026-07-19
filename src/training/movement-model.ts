import type { ReplayManifest } from "../capture/replay.js";
import type { TrajectorySpan } from "../capture/trajectory.js";

/**
 * In-process, on-device movement model.
 *
 * Standing objective #2 requires bee-agent to (c) train a *local* model on the
 * recorded movement dataset so it can repeat the recorded movements, and (d)
 * generalize to new-but-related movements. The existing training runner only
 * emits a plan that shells out to Apple-Silicon Python tooling (mlx/axolotl),
 * which cannot run in the cloud or in CI. This module provides a fully
 * deterministic, dependency-free backend that trains and infers *in-process*,
 * so the pipeline is exercisable everywhere, and keeps the model backend
 * pluggable so a real on-device small model can be swapped in behind the same
 * interface.
 */

/** A single recorded movement, decoupled from any capture surface. */
export interface MovementToken {
  /** Gesture / tool family, e.g. "tap", "swipe", "type", "shortcut". */
  kind: string;
  /** UI target the movement acted on, when known. */
  target?: string;
  /** Directional component for swipes/scrolls, when known. */
  direction?: string;
}

export interface MovementSequence {
  id: string;
  tokens: MovementToken[];
}

export interface MovementDataset {
  version: 1;
  sequences: MovementSequence[];
}

export interface MovementTrainingOptions {
  /** Maximum context length the model conditions on. Defaults to 3. */
  maxOrder?: number;
}

export type MovementPredictionSource = "exact" | "backoff" | "generalized";

export interface MovementPrediction {
  token: MovementToken;
  /** Estimated probability of this continuation, in [0, 1]. */
  confidence: number;
  /** How the prediction was derived — exact match, shorter-context backoff, or kind-level generalization. */
  source: MovementPredictionSource;
  /** Length of the context the prediction conditioned on. */
  order: number;
}

/** A trained, queryable movement model. */
export interface MovementModel {
  readonly backendId: string;
  /** Predict the next movement given a (possibly empty) recent context. */
  predictNext(context: MovementToken[]): MovementPrediction | undefined;
  /** Roll out `steps` movements starting from `seed`, greedily and deterministically. */
  generate(seed: MovementToken[], steps: number): MovementToken[];
  /** Serialize to a plain JSON-safe value for persistence. */
  toJSON(): MovementModelSnapshot;
}

/** Pluggable seam: a training backend produces a `MovementModel` from a dataset. */
export interface MovementModelBackend {
  readonly id: string;
  train(dataset: MovementDataset, options?: MovementTrainingOptions): MovementModel;
}

// --- Token helpers ------------------------------------------------------------

export function serializeMovementToken(token: MovementToken): string {
  let key = token.kind;
  if (token.target) {
    key += `@${token.target}`;
  }
  if (token.direction) {
    key += `^${token.direction}`;
  }
  return key;
}

function contextKey(tokens: MovementToken[]): string {
  return tokens.map(serializeMovementToken).join(" ");
}

function kindContextKey(tokens: MovementToken[]): string {
  return tokens.map((token) => token.kind).join(" ");
}

// --- Dataset construction -----------------------------------------------------

/**
 * Derive a movement token from a captured trajectory action. Prefers the
 * structured gesture metadata written by the device adapter, and falls back to
 * the action tool when no gesture metadata is present.
 */
export function movementTokenFromAction(action: TrajectorySpan["actions"][number]): MovementToken {
  const metadata = action.metadata ?? {};
  const gesture = typeof metadata.gesture === "string" ? metadata.gesture : undefined;
  const target = typeof metadata.target === "string" ? metadata.target : undefined;
  const direction = typeof metadata.direction === "string" ? metadata.direction : undefined;
  const token: MovementToken = { kind: gesture ?? action.tool };
  if (target) {
    token.target = target;
  }
  if (direction) {
    token.direction = direction;
  }
  return token;
}

/** Build a dataset from captured trajectory spans (one sequence per trajectory). */
export function buildMovementDatasetFromTrajectories(trajectories: TrajectorySpan[]): MovementDataset {
  const sequences: MovementSequence[] = trajectories
    .map((trajectory) => ({
      id: trajectory.id,
      tokens: [...trajectory.actions]
        .sort((a, b) => a.ts - b.ts)
        .map((action) => movementTokenFromAction(action)),
    }))
    .filter((sequence) => sequence.tokens.length > 0);
  return { version: 1, sequences };
}

/** Build a dataset from a replay manifest, grouping action events per trajectory. */
export function buildMovementDatasetFromReplay(manifest: ReplayManifest): MovementDataset {
  const byTrajectory = new Map<string, { ts: number; token: MovementToken }[]>();
  for (const event of manifest.events) {
    if (event.kind !== "action") {
      continue;
    }
    const list = byTrajectory.get(event.trajectoryId) ?? [];
    list.push({ ts: event.ts, token: parseMovementSummary(event.tool, event.summary) });
    byTrajectory.set(event.trajectoryId, list);
  }
  const sequences: MovementSequence[] = [...byTrajectory.entries()].map(([id, entries]) => ({
    id,
    tokens: entries.sort((a, b) => a.ts - b.ts).map((entry) => entry.token),
  }));
  return { version: 1, sequences };
}

const SUMMARY_PATTERNS: { kind: string; re: RegExp }[] = [
  { kind: "tap", re: /^tapped (.+)$/ },
  { kind: "type", re: /^typed into (.+)$/ },
  { kind: "shortcut", re: /^triggered (.+)$/ },
];
const DIRECTION_PATTERNS: { kind: string; re: RegExp }[] = [
  { kind: "swipe", re: /^swiped (up|down|left|right)$/ },
  { kind: "scroll", re: /^scrolled (up|down|left|right)$/ },
];

/** Best-effort parse of a replay action's tool+summary back into a movement token. */
function parseMovementSummary(tool: string, summary: string): MovementToken {
  for (const pattern of DIRECTION_PATTERNS) {
    const match = pattern.re.exec(summary);
    if (match) {
      return { kind: pattern.kind, direction: match[1] };
    }
  }
  for (const pattern of SUMMARY_PATTERNS) {
    const match = pattern.re.exec(summary);
    if (match) {
      return { kind: pattern.kind, target: match[1] };
    }
  }
  return { kind: tool };
}

// --- Serialized model shape ---------------------------------------------------

export interface MovementModelSnapshot {
  version: 1;
  backendId: "markov";
  maxOrder: number;
  vocab: Record<string, MovementToken>;
  /** order -> contextKey -> nextTokenKey -> count */
  transitions: Record<string, Record<string, Record<string, number>>>;
  /** order -> kindContextKey -> nextKind -> count */
  kindTransitions: Record<string, Record<string, Record<string, number>>>;
  /** nextKind -> tokenKey -> count (for generalization synthesis) */
  kindTokens: Record<string, Record<string, number>>;
}

// --- Deterministic n-gram backend --------------------------------------------

type CountMap = Map<string, Map<string, number>>;

function increment(map: CountMap, context: string, next: string): void {
  const inner = map.get(context) ?? new Map<string, number>();
  inner.set(next, (inner.get(next) ?? 0) + 1);
  map.set(context, inner);
}

/** Deterministic argmax: highest count, ties broken by lexical key order. */
function argmax(counts: Map<string, number>): { key: string; count: number; total: number } | undefined {
  let bestKey: string | undefined;
  let bestCount = -1;
  let total = 0;
  for (const [key, count] of counts) {
    total += count;
    if (count > bestCount || (count === bestCount && bestKey !== undefined && key < bestKey)) {
      bestKey = key;
      bestCount = count;
    }
  }
  if (bestKey === undefined) {
    return undefined;
  }
  return { key: bestKey, count: bestCount, total };
}

const GENERALIZATION_PENALTY = 0.5;

/**
 * A variable-order Markov model over movement tokens with Katz-style backoff
 * and a kind-level generalization fallback.
 *
 * - Exact/backoff replay reproduces recorded movement sequences (objective 2c).
 * - When no token-level context matches, the model backs off to a model over
 *   gesture *kinds* and synthesizes a concrete movement of the predicted kind,
 *   letting it perform a new-but-related movement (objective 2d).
 */
export class MarkovMovementBackend implements MovementModelBackend {
  readonly id = "markov";

  train(dataset: MovementDataset, options?: MovementTrainingOptions): MovementModel {
    const maxOrder = Math.max(1, options?.maxOrder ?? 3);
    const vocab = new Map<string, MovementToken>();
    const transitions = new Map<number, CountMap>();
    const kindTransitions = new Map<number, CountMap>();
    const kindTokens: CountMap = new Map();
    for (let order = 1; order <= maxOrder; order += 1) {
      transitions.set(order, new Map());
      kindTransitions.set(order, new Map());
    }

    for (const sequence of dataset.sequences) {
      const tokens = sequence.tokens;
      for (let i = 0; i < tokens.length; i += 1) {
        const next = tokens[i];
        const nextKey = serializeMovementToken(next);
        vocab.set(nextKey, next);
        increment(kindTokens, next.kind, nextKey);
        for (let order = 1; order <= maxOrder; order += 1) {
          if (i - order < 0) {
            continue;
          }
          const context = tokens.slice(i - order, i);
          increment(transitions.get(order)!, contextKey(context), nextKey);
          increment(kindTransitions.get(order)!, kindContextKey(context), next.kind);
        }
      }
    }

    return new MarkovMovementModel(maxOrder, vocab, transitions, kindTransitions, kindTokens);
  }

  static fromJSON(snapshot: MovementModelSnapshot): MovementModel {
    const vocab = new Map<string, MovementToken>(Object.entries(snapshot.vocab));
    const transitions = deserializeOrderMaps(snapshot.transitions);
    const kindTransitions = deserializeOrderMaps(snapshot.kindTransitions);
    const kindTokens: CountMap = new Map();
    for (const [kind, tokens] of Object.entries(snapshot.kindTokens)) {
      kindTokens.set(kind, new Map(Object.entries(tokens)));
    }
    return new MarkovMovementModel(snapshot.maxOrder, vocab, transitions, kindTransitions, kindTokens);
  }
}

function deserializeOrderMaps(source: Record<string, Record<string, Record<string, number>>>): Map<number, CountMap> {
  const result = new Map<number, CountMap>();
  for (const [order, contexts] of Object.entries(source)) {
    const countMap: CountMap = new Map();
    for (const [context, nexts] of Object.entries(contexts)) {
      countMap.set(context, new Map(Object.entries(nexts)));
    }
    result.set(Number(order), countMap);
  }
  return result;
}

class MarkovMovementModel implements MovementModel {
  readonly backendId = "markov";

  constructor(
    private readonly maxOrder: number,
    private readonly vocab: Map<string, MovementToken>,
    private readonly transitions: Map<number, CountMap>,
    private readonly kindTransitions: Map<number, CountMap>,
    private readonly kindTokens: CountMap,
  ) {}

  predictNext(context: MovementToken[]): MovementPrediction | undefined {
    const maxUsable = Math.min(this.maxOrder, context.length);

    // 1) Exact / backoff over the full token model, longest context first.
    for (let order = maxUsable; order >= 1; order -= 1) {
      const slice = context.slice(context.length - order);
      const counts = this.transitions.get(order)?.get(contextKey(slice));
      const best = counts && argmax(counts);
      if (best) {
        const token = this.vocab.get(best.key);
        if (token) {
          return {
            token: { ...token },
            confidence: best.count / best.total,
            source: order === maxUsable ? "exact" : "backoff",
            order,
          };
        }
      }
    }

    // 2) Generalize: predict the next *kind* from the kind model, then synthesize
    //    the most representative concrete movement of that kind.
    for (let order = maxUsable; order >= 1; order -= 1) {
      const slice = context.slice(context.length - order);
      const counts = this.kindTransitions.get(order)?.get(kindContextKey(slice));
      const bestKind = counts && argmax(counts);
      if (bestKind) {
        const token = this.representativeTokenForKind(bestKind.key);
        if (token) {
          return {
            token,
            confidence: (bestKind.count / bestKind.total) * GENERALIZATION_PENALTY,
            source: "generalized",
            order,
          };
        }
      }
    }

    // 3) Cold start with empty context: fall back to the single most frequent movement.
    if (context.length === 0) {
      return this.mostFrequentMovement();
    }
    return undefined;
  }

  generate(seed: MovementToken[], steps: number): MovementToken[] {
    const produced: MovementToken[] = [];
    let context = [...seed];
    for (let i = 0; i < steps; i += 1) {
      const prediction = this.predictNext(context);
      if (!prediction) {
        break;
      }
      produced.push(prediction.token);
      context = [...context, prediction.token].slice(-this.maxOrder);
    }
    return produced;
  }

  toJSON(): MovementModelSnapshot {
    return {
      version: 1,
      backendId: "markov",
      maxOrder: this.maxOrder,
      vocab: Object.fromEntries([...this.vocab.entries()].map(([key, token]) => [key, token])),
      transitions: serializeOrderMaps(this.transitions),
      kindTransitions: serializeOrderMaps(this.kindTransitions),
      kindTokens: Object.fromEntries(
        [...this.kindTokens.entries()].map(([kind, counts]) => [kind, Object.fromEntries(counts)]),
      ),
    };
  }

  private representativeTokenForKind(kind: string): MovementToken | undefined {
    const counts = this.kindTokens.get(kind);
    const best = counts && argmax(counts);
    if (best) {
      const token = this.vocab.get(best.key);
      if (token) {
        return { ...token };
      }
    }
    return { kind };
  }

  private mostFrequentMovement(): MovementPrediction | undefined {
    let bestKey: string | undefined;
    let bestCount = -1;
    let total = 0;
    for (const counts of this.kindTokens.values()) {
      for (const [key, count] of counts) {
        total += count;
        if (count > bestCount || (count === bestCount && bestKey !== undefined && key < bestKey)) {
          bestKey = key;
          bestCount = count;
        }
      }
    }
    if (bestKey === undefined) {
      return undefined;
    }
    const token = this.vocab.get(bestKey);
    if (!token) {
      return undefined;
    }
    return { token: { ...token }, confidence: bestCount / total, source: "generalized", order: 0 };
  }
}

function serializeOrderMaps(source: Map<number, CountMap>): Record<string, Record<string, Record<string, number>>> {
  const result: Record<string, Record<string, Record<string, number>>> = {};
  for (const [order, contexts] of source) {
    const contextRecord: Record<string, Record<string, number>> = {};
    for (const [context, counts] of contexts) {
      contextRecord[context] = Object.fromEntries(counts);
    }
    result[String(order)] = contextRecord;
  }
  return result;
}

// --- Backend registry (pluggable seam) ---------------------------------------

export type MovementModelBackendId = "markov";

/**
 * Resolve a movement-model backend by id. `markov` is the deterministic
 * in-process backend that runs everywhere. Real on-device backends (e.g. an
 * mlx-trained small model) can register here behind the same interface without
 * changing any caller.
 */
export function createMovementModelBackend(id: MovementModelBackendId = "markov"): MovementModelBackend {
  switch (id) {
    case "markov":
      return new MarkovMovementBackend();
    default: {
      const exhaustive: never = id;
      throw new Error(`unknown movement-model backend: ${String(exhaustive)}`);
    }
  }
}
