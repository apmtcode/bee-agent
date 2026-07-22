import type { ReplayManifest, ReplayTimelineEvent } from "../capture/replay.js";

/**
 * Local-movement learning: an in-process, cloud-testable train→infer→generalize
 * pipeline for the recorded-movement subsystem (standing objective #2d).
 *
 * The real on-device training path lives in {@link ./runner.js} (it emits
 * mlx/axolotl launch scripts for Apple-silicon). That path cannot run in the
 * cloud, so it also cannot be validated here. This module provides a *pluggable*
 * model backend with a deterministic reference implementation that:
 *   - trains on ordered movement-action sequences extracted from replay manifests,
 *   - predicts the next movement for a given context (repeat recorded movements),
 *   - generalizes to new-but-related movements via variable-order backoff plus a
 *     nearest-known-token fallback keyed on movement features.
 *
 * The backend is an interface so a real local/open model (e.g. a small on-device
 * transformer) can be swapped in behind the same seam; the Markov backend keeps
 * the whole loop exercised and green in CI without OS access.
 */

/** A single ordered movement sequence — the tokens of one recorded trajectory. */
export type MovementSample = {
  /** Optional provenance label (session/trajectory id) for eval reporting. */
  readonly label?: string;
  /** Ordered movement tokens (see {@link movementToken}). */
  readonly tokens: string[];
};

/** Feature descriptor for a movement token, used for nearest-token generalization. */
export type MovementTokenFeatures = {
  readonly tool: string;
  readonly keywords: string[];
};

/** A trained, serializable movement model artifact (plain JSON, backend-agnostic). */
export type MovementModel = {
  readonly version: 1;
  readonly backend: string;
  readonly maxOrder: number;
  readonly vocabulary: string[];
  /** context-key (tokens joined by CONTEXT_SEP) -> next-token -> count. */
  readonly transitions: Record<string, Record<string, number>>;
  readonly tokenFeatures: Record<string, MovementTokenFeatures>;
  readonly trainedSamples: number;
  readonly trainedTransitions: number;
};

export type MovementBackendConfig = {
  /** Highest context order to learn. Order 0 (unigram) is always included. */
  readonly maxOrder: number;
};

export const DEFAULT_MOVEMENT_BACKEND_CONFIG: MovementBackendConfig = {
  maxOrder: 3,
};

export type MovementPredictionMethod = "exact" | "backoff" | "nearest" | "none";

export type MovementCandidate = {
  readonly token: string;
  readonly probability: number;
};

export type MovementPrediction = {
  /** Best next token, or undefined when the model is empty. */
  readonly token: string | undefined;
  readonly probability: number;
  /** Context order actually used after backoff (<= maxOrder). */
  readonly order: number;
  readonly method: MovementPredictionMethod;
  /** Ranked candidates (highest probability first). */
  readonly candidates: MovementCandidate[];
};

/**
 * Pluggable movement-model backend. Implementations are deterministic given the
 * same inputs so cloud/CI tests are stable.
 */
export interface MovementModelBackend {
  readonly name: string;
  train(samples: readonly MovementSample[], config?: Partial<MovementBackendConfig>): MovementModel;
  predict(model: MovementModel, context: readonly string[]): MovementPrediction;
  /** Roll the model forward `steps` movements from a seed context. */
  generate(model: MovementModel, seed: readonly string[], steps: number): string[];
}

const CONTEXT_SEP = "␟";
const EMPTY_CONTEXT = "";

/** Canonical movement token for an action event: `tool::slug(summary)`. */
export function movementToken(tool: string, summary: string): string {
  return `${slug(tool)}::${slug(summary)}`;
}

/** Extract ordered movement samples (action sequences) from replay manifests. */
export function extractMovementSamples(manifests: readonly ReplayManifest[]): MovementSample[] {
  return manifests.map((manifest) => ({
    label: manifest.sessionId,
    tokens: manifest.events
      .filter((event): event is Extract<ReplayTimelineEvent, { kind: "action" }> => event.kind === "action")
      .map((event) => movementToken(event.tool, event.summary)),
  }));
}

/** Split a movement token back into its features (tool + keywords). */
function featuresForToken(token: string): MovementTokenFeatures {
  const separatorIndex = token.indexOf("::");
  const tool = separatorIndex >= 0 ? token.slice(0, separatorIndex) : "";
  const rest = separatorIndex >= 0 ? token.slice(separatorIndex + 2) : token;
  const keywords = rest.split("-").filter((word) => word.length > 0);
  return { tool, keywords };
}

/**
 * Variable-order Markov backend with Katz-style backoff and a nearest-token
 * fallback for generalization. Fully deterministic.
 */
export class MarkovMovementBackend implements MovementModelBackend {
  readonly name = "markov";

  train(samples: readonly MovementSample[], config?: Partial<MovementBackendConfig>): MovementModel {
    const maxOrder = Math.max(0, config?.maxOrder ?? DEFAULT_MOVEMENT_BACKEND_CONFIG.maxOrder);
    const transitions: Record<string, Record<string, number>> = {};
    const vocabulary = new Set<string>();
    const tokenFeatures: Record<string, MovementTokenFeatures> = {};
    let trainedTransitions = 0;

    const record = (contextKey: string, token: string): void => {
      const row = (transitions[contextKey] ??= {});
      row[token] = (row[token] ?? 0) + 1;
      trainedTransitions += 1;
    };

    for (const sample of samples) {
      const tokens = sample.tokens;
      for (let index = 0; index < tokens.length; index += 1) {
        const token = tokens[index]!;
        vocabulary.add(token);
        tokenFeatures[token] ??= featuresForToken(token);
        // order 0..maxOrder: context is the preceding `order` tokens.
        for (let order = 0; order <= maxOrder; order += 1) {
          if (order > index) {
            break;
          }
          const context = tokens.slice(index - order, index);
          record(contextKeyFor(context), token);
        }
      }
    }

    return {
      version: 1,
      backend: this.name,
      maxOrder,
      vocabulary: [...vocabulary].sort(),
      transitions,
      tokenFeatures,
      trainedSamples: samples.length,
      trainedTransitions,
    };
  }

  predict(model: MovementModel, context: readonly string[]): MovementPrediction {
    const maxUsableOrder = Math.min(model.maxOrder, context.length);
    // Try the specific context first, backing off toward (but not yet reaching)
    // the context-free unigram.
    for (let order = maxUsableOrder; order >= 1; order -= 1) {
      const contextTokens = context.slice(context.length - order);
      const row = model.transitions[contextKeyFor(contextTokens)];
      if (row && hasCounts(row)) {
        const method: MovementPredictionMethod = order === maxUsableOrder ? "exact" : "backoff";
        return distributionToPrediction(row, order, method);
      }
    }

    // Generalization: no order>=1 context matched (the seed movement is unseen).
    // Map it to the nearest known token by feature similarity and reuse that
    // token's continuation — a more informed guess than the global unigram.
    const nearest = this.nearestKnownToken(model, context[context.length - 1]);
    if (nearest) {
      const row = model.transitions[contextKeyFor([nearest])];
      if (row && hasCounts(row)) {
        return distributionToPrediction(row, 1, "nearest");
      }
    }

    // Last resort: the context-free unigram distribution.
    const unigram = model.transitions[EMPTY_CONTEXT];
    if (unigram && hasCounts(unigram)) {
      return distributionToPrediction(unigram, 0, "backoff");
    }

    return { token: undefined, probability: 0, order: 0, method: "none", candidates: [] };
  }

  generate(model: MovementModel, seed: readonly string[], steps: number): string[] {
    const produced: string[] = [];
    const context = [...seed];
    for (let step = 0; step < steps; step += 1) {
      const prediction = this.predict(model, context);
      if (prediction.token === undefined) {
        break;
      }
      produced.push(prediction.token);
      context.push(prediction.token);
      if (context.length > model.maxOrder) {
        context.shift();
      }
    }
    return produced;
  }

  /** Nearest known vocabulary token to `token` by Jaccard(keywords) + tool bonus. */
  private nearestKnownToken(model: MovementModel, token: string | undefined): string | undefined {
    if (token === undefined) {
      return undefined;
    }
    const target = model.tokenFeatures[token] ?? featuresForToken(token);
    let best: string | undefined;
    let bestScore = 0;
    for (const candidate of model.vocabulary) {
      if (candidate === token) {
        return candidate;
      }
      const score = featureSimilarity(target, model.tokenFeatures[candidate] ?? featuresForToken(candidate));
      // Deterministic tie-break: keep the lexicographically smaller candidate.
      if (score > bestScore || (score === bestScore && best !== undefined && candidate < best)) {
        bestScore = score;
        best = candidate;
      }
    }
    return bestScore > 0 ? best : undefined;
  }
}

export type MovementModelEvaluation = {
  /** Total next-token predictions attempted across all held-out samples. */
  readonly predictions: number;
  /** Predictions whose top-1 token matched the recorded next movement. */
  readonly correct: number;
  /** correct / predictions (0 when no predictions were attempted). */
  readonly accuracy: number;
  /** How often each prediction method was used. */
  readonly methodBreakdown: Record<MovementPredictionMethod, number>;
};

/**
 * Generalization eval harness: next-token top-1 accuracy on held-out sequences.
 * For each sample, replay its prefixes and check whether the model predicts the
 * true next movement. Sequences the model never trained on exercise the backoff
 * and nearest-token paths, measuring generalization rather than memorization.
 */
export function evaluateMovementModel(
  backend: MovementModelBackend,
  model: MovementModel,
  heldOut: readonly MovementSample[],
): MovementModelEvaluation {
  let predictions = 0;
  let correct = 0;
  const methodBreakdown: Record<MovementPredictionMethod, number> = {
    exact: 0,
    backoff: 0,
    nearest: 0,
    none: 0,
  };

  for (const sample of heldOut) {
    for (let index = 1; index < sample.tokens.length; index += 1) {
      const context = sample.tokens.slice(0, index);
      const expected = sample.tokens[index]!;
      const prediction = backend.predict(model, context);
      predictions += 1;
      methodBreakdown[prediction.method] += 1;
      if (prediction.token === expected) {
        correct += 1;
      }
    }
  }

  return {
    predictions,
    correct,
    accuracy: predictions === 0 ? 0 : correct / predictions,
    methodBreakdown,
  };
}

const MOVEMENT_BACKENDS = new Map<string, () => MovementModelBackend>([
  ["markov", () => new MarkovMovementBackend()],
]);

/**
 * Resolve a movement backend by name. New backends (e.g. a real on-device model)
 * register here without changing call sites. Defaults to the Markov reference.
 */
export function createMovementBackend(name = "markov"): MovementModelBackend {
  const factory = MOVEMENT_BACKENDS.get(name);
  if (!factory) {
    throw new Error(`unknown movement backend: ${name}`);
  }
  return factory();
}

/** Register a custom movement backend factory (pluggable-backend seam). */
export function registerMovementBackend(name: string, factory: () => MovementModelBackend): void {
  MOVEMENT_BACKENDS.set(name, factory);
}

export function listMovementBackends(): string[] {
  return [...MOVEMENT_BACKENDS.keys()].sort();
}

// --- synthetic event-stream generator (cloud validation without OS input) -----

export type SyntheticMovementOptions = {
  /** Deterministic seed. */
  readonly seed: number;
  /** Number of sequences to generate. */
  readonly sampleCount: number;
  /** Length of each generated sequence. */
  readonly sequenceLength: number;
  /**
   * Vocabulary of (tool, action) movement primitives to draw from. Defaults to a
   * small UI-navigation grammar that has learnable structure (tool affinity).
   */
  readonly primitives?: readonly { tool: string; action: string }[];
};

const DEFAULT_MOVEMENT_PRIMITIVES: readonly { tool: string; action: string }[] = [
  { tool: "mouse", action: "move to toolbar" },
  { tool: "mouse", action: "click menu" },
  { tool: "mouse", action: "drag selection" },
  { tool: "keyboard", action: "type command" },
  { tool: "keyboard", action: "press enter" },
  { tool: "keyboard", action: "shortcut save" },
  { tool: "window", action: "focus editor" },
  { tool: "window", action: "switch app" },
];

/**
 * Generate deterministic synthetic movement samples with learnable structure:
 * the next primitive is biased toward sharing the current primitive's tool, so a
 * trained model should beat a uniform baseline. Uses a seeded LCG (no Math.random)
 * so runs are reproducible in CI.
 */
export function generateSyntheticMovementSamples(options: SyntheticMovementOptions): MovementSample[] {
  const primitives = options.primitives ?? DEFAULT_MOVEMENT_PRIMITIVES;
  if (primitives.length === 0) {
    return [];
  }
  const rng = createLcg(options.seed);
  const samples: MovementSample[] = [];

  for (let sampleIndex = 0; sampleIndex < options.sampleCount; sampleIndex += 1) {
    const tokens: string[] = [];
    let current = primitives[Math.floor(rng() * primitives.length)]!;
    for (let step = 0; step < options.sequenceLength; step += 1) {
      tokens.push(movementToken(current.tool, current.action));
      // 70% of the time continue within the same tool (learnable affinity).
      const sameTool = primitives.filter((primitive) => primitive.tool === current.tool);
      const pool = rng() < 0.7 && sameTool.length > 0 ? sameTool : primitives;
      current = pool[Math.floor(rng() * pool.length)]!;
    }
    samples.push({ label: `synthetic-${sampleIndex}`, tokens });
  }

  return samples;
}

// --- helpers ------------------------------------------------------------------

function contextKeyFor(context: readonly string[]): string {
  return context.length === 0 ? EMPTY_CONTEXT : context.join(CONTEXT_SEP);
}

function hasCounts(row: Record<string, number>): boolean {
  for (const key in row) {
    if (row[key]! > 0) {
      return true;
    }
  }
  return false;
}

function distributionToPrediction(
  row: Record<string, number>,
  order: number,
  method: MovementPredictionMethod,
): MovementPrediction {
  let total = 0;
  for (const key in row) {
    total += row[key]!;
  }
  const candidates: MovementCandidate[] = Object.keys(row)
    .map((token) => ({ token, probability: total === 0 ? 0 : row[token]! / total }))
    .sort((a, b) => {
      if (b.probability !== a.probability) {
        return b.probability - a.probability;
      }
      return a.token < b.token ? -1 : a.token > b.token ? 1 : 0;
    });
  const best = candidates[0];
  return {
    token: best?.token,
    probability: best?.probability ?? 0,
    order,
    method,
    candidates,
  };
}

function featureSimilarity(a: MovementTokenFeatures, b: MovementTokenFeatures): number {
  const setA = new Set(a.keywords);
  const setB = new Set(b.keywords);
  let intersection = 0;
  for (const keyword of setA) {
    if (setB.has(keyword)) {
      intersection += 1;
    }
  }
  const union = setA.size + setB.size - intersection;
  const jaccard = union === 0 ? 0 : intersection / union;
  const toolBonus = a.tool !== "" && a.tool === b.tool ? 0.5 : 0;
  return jaccard + toolBonus;
}

function slug(value: string): string {
  return value
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
}

/** Deterministic 32-bit LCG (numerical recipes constants), values in [0, 1). */
function createLcg(seed: number): () => number {
  let state = (seed >>> 0) || 1;
  return () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 0x100000000;
  };
}
