import type { ReplayManifest, ReplayTimelineEvent } from "../capture/replay.js";
import type { ExportedReplayManifest } from "./export-manifest.js";

/**
 * Local-movement learning subsystem — in-process model backend.
 *
 * The capture/replay/export pipeline turns recorded local movements into a
 * reviewable, replayable dataset. This module closes the loop objective #2
 * asks for: post-train a *local* model on that dataset so it can (c) repeat the
 * recorded movements and (d) generalize to new-but-related movements.
 *
 * Everything here is deterministic and runs in-process, so it validates in the
 * cloud/CI with synthetic event streams and needs no real OS access. The
 * `MovementModelBackend` interface is the pluggable seam: the shipped
 * `NGramMovementBackend` is an on-device-friendly reference/mock model; a real
 * small local model (MLX/torch) can implement the same interface later without
 * touching call sites.
 */

/** A single canonical movement, e.g. `tap` on `submit-button` via the `device` tool. */
export type MovementToken = {
  /** Capturing tool/surface: `device`, `browser`, `keyboard`, `os`, … */
  tool: string;
  /** Normalized verb: `tap`, `swipe`, `scroll`, `type`, `click`, `shortcut`, … */
  action: string;
  /** Optional UI target/element the movement acted on. */
  target?: string;
  /** Optional spatial direction for swipes/scrolls. */
  direction?: "up" | "down" | "left" | "right";
};

/** An ordered movement sequence — the replayable unit the model learns from. */
export type MovementSequence = {
  id: string;
  tokens: MovementToken[];
};

/** A dataset of movement sequences, the training input for a backend. */
export type MovementDataset = {
  sequences: MovementSequence[];
};

export type MovementPredictionSource =
  | "ngram"
  | "backoff"
  | "generalized"
  | "prior"
  | "none";

export type MovementPredictionCandidate = {
  token: MovementToken;
  score: number;
};

export type MovementPrediction = {
  /** Best next movement, or `undefined` when the model has learned nothing. */
  token: MovementToken | undefined;
  /** 0..1 confidence in `token`. */
  confidence: number;
  /** How the prediction was derived (exact n-gram vs. generalized, etc.). */
  source: MovementPredictionSource;
  /** Ranked alternatives (includes `token`), most likely first. */
  candidates: MovementPredictionCandidate[];
};

export type TrainMovementModelOptions = {
  /** Max n-gram order (context length + 1). Default 3. */
  order?: number;
};

/** A trained, queryable movement model. */
export interface TrainedMovementModel {
  readonly backend: string;
  readonly order: number;
  readonly vocabularySize: number;
  /** Predict the most likely movement following `context`. */
  predictNext(context: MovementToken[]): MovementPrediction;
  /** Autoregressively extend `seed` by up to `steps` movements. */
  generate(seed: MovementToken[], steps: number): MovementToken[];
  /** Serialize to a plain snapshot for persistence/transport. */
  serialize(): MovementModelSnapshot;
}

/** Pluggable backend seam: swap the reference n-gram model for a real local model. */
export interface MovementModelBackend {
  readonly name: string;
  train(dataset: MovementDataset, options?: TrainMovementModelOptions): TrainedMovementModel;
}

export type MovementModelSnapshot = {
  version: 1;
  backend: string;
  order: number;
  vocabulary: MovementToken[];
  /** context-id → { nextTokenId → count } for every observed n-gram order. */
  transitions: Record<string, Record<string, number>>;
  /** tokenId → total occurrences, the unigram prior. */
  priors: Record<string, number>;
};

const FIELD_SEPARATOR = "";
const CONTEXT_SEPARATOR = "";

/** Canonical, collision-free id for a movement token. */
export function movementTokenId(token: MovementToken): string {
  return [token.tool, token.action, token.target ?? "", token.direction ?? ""].join(FIELD_SEPARATOR);
}

function contextId(tokens: MovementToken[]): string {
  return tokens.map(movementTokenId).join(CONTEXT_SEPARATOR);
}

/** Feature bag used for generalization to unseen-but-related movements. */
function tokenFeatures(token: MovementToken): string[] {
  const features = [`tool:${token.tool}`, `action:${token.action}`];
  if (token.target) {
    features.push(`target:${token.target}`);
  }
  if (token.direction) {
    features.push(`direction:${token.direction}`);
  }
  return features;
}

function jaccardSimilarity(a: string[], b: string[]): number {
  if (a.length === 0 && b.length === 0) {
    return 0;
  }
  const setB = new Set(b);
  let intersection = 0;
  for (const feature of a) {
    if (setB.has(feature)) {
      intersection += 1;
    }
  }
  const union = new Set([...a, ...b]).size;
  return union === 0 ? 0 : intersection / union;
}

/**
 * Deterministic n-gram movement model with feature-similarity generalization.
 *
 * - Learns transition counts for every context length up to `order - 1`.
 * - `predictNext` uses the longest matching context, backing off to shorter
 *   contexts, then to a feature-similar seen movement (generalization), then to
 *   the global prior. Ties break lexicographically on token id, so predictions
 *   are fully reproducible — no randomness, safe for CI.
 */
export class NGramMovementBackend implements MovementModelBackend {
  readonly name = "ngram";

  train(dataset: MovementDataset, options: TrainMovementModelOptions = {}): TrainedMovementModel {
    const order = Math.max(2, Math.floor(options.order ?? 3));
    const vocabulary = new Map<string, MovementToken>();
    const transitions = new Map<string, Map<string, number>>();
    const priors = new Map<string, number>();

    const register = (token: MovementToken): string => {
      const id = movementTokenId(token);
      if (!vocabulary.has(id)) {
        vocabulary.set(id, token);
      }
      return id;
    };

    for (const sequence of dataset.sequences) {
      const tokens = sequence.tokens;
      for (let index = 0; index < tokens.length; index += 1) {
        const nextId = register(tokens[index]);
        priors.set(nextId, (priors.get(nextId) ?? 0) + 1);
        // Record every context length from 1..order-1 ending just before `index`.
        for (let contextLength = 1; contextLength <= order - 1; contextLength += 1) {
          const start = index - contextLength;
          if (start < 0) {
            break;
          }
          const context = tokens.slice(start, index);
          const key = contextId(context);
          let row = transitions.get(key);
          if (!row) {
            row = new Map<string, number>();
            transitions.set(key, row);
          }
          row.set(nextId, (row.get(nextId) ?? 0) + 1);
        }
      }
    }

    return new NGramMovementModel(this.name, order, vocabulary, transitions, priors);
  }
}

class NGramMovementModel implements TrainedMovementModel {
  constructor(
    readonly backend: string,
    readonly order: number,
    private readonly vocabulary: Map<string, MovementToken>,
    private readonly transitions: Map<string, Map<string, number>>,
    private readonly priors: Map<string, number>,
  ) {}

  get vocabularySize(): number {
    return this.vocabulary.size;
  }

  predictNext(context: MovementToken[]): MovementPrediction {
    if (this.vocabulary.size === 0) {
      return { token: undefined, confidence: 0, source: "none", candidates: [] };
    }

    // 1) Longest exact context match, backing off to shorter contexts.
    const maxContext = Math.min(context.length, this.order - 1);
    for (let length = maxContext; length >= 1; length -= 1) {
      const key = contextId(context.slice(context.length - length));
      const row = this.transitions.get(key);
      if (row && row.size > 0) {
        const candidates = this.rankRow(row);
        return {
          token: candidates[0].token,
          confidence: candidates[0].score,
          source: length === maxContext && length === this.order - 1 ? "ngram" : "backoff",
          candidates,
        };
      }
    }

    // 2) Generalize: no context seen, but the last movement may resemble a known
    //    one. Borrow the transitions of the most feature-similar seen token.
    const lastToken = context[context.length - 1];
    if (lastToken) {
      const generalized = this.generalizeFrom(lastToken);
      if (generalized) {
        return generalized;
      }
    }

    // 3) Fall back to the global prior (most frequent movement overall).
    const priorCandidates = this.rankRow(this.priors);
    return {
      token: priorCandidates[0].token,
      confidence: priorCandidates[0].score,
      source: "prior",
      candidates: priorCandidates,
    };
  }

  generate(seed: MovementToken[], steps: number): MovementToken[] {
    const generated: MovementToken[] = [];
    const context = [...seed];
    for (let step = 0; step < steps; step += 1) {
      const prediction = this.predictNext(context);
      if (!prediction.token) {
        break;
      }
      generated.push(prediction.token);
      context.push(prediction.token);
    }
    return generated;
  }

  serialize(): MovementModelSnapshot {
    const transitions: Record<string, Record<string, number>> = {};
    for (const [key, row] of this.transitions) {
      transitions[key] = Object.fromEntries(row);
    }
    return {
      version: 1,
      backend: this.backend,
      order: this.order,
      vocabulary: [...this.vocabulary.values()],
      transitions,
      priors: Object.fromEntries(this.priors),
    };
  }

  private generalizeFrom(lastToken: MovementToken): MovementPrediction | undefined {
    const lastFeatures = tokenFeatures(lastToken);
    let bestId: string | undefined;
    let bestSimilarity = 0;
    for (const [id, token] of this.vocabulary) {
      const similarity = jaccardSimilarity(lastFeatures, tokenFeatures(token));
      // Strict `>` with lexicographic tie-break keeps this deterministic.
      if (similarity > bestSimilarity || (similarity === bestSimilarity && bestId !== undefined && id < bestId)) {
        if (similarity > 0) {
          bestSimilarity = similarity;
          bestId = id;
        }
      }
    }
    if (bestId === undefined || bestSimilarity <= 0) {
      return undefined;
    }
    const row = this.transitions.get(contextId([this.vocabulary.get(bestId)!]));
    if (!row || row.size === 0) {
      return undefined;
    }
    const candidates = this.rankRow(row).map((candidate) => ({
      token: candidate.token,
      score: candidate.score * bestSimilarity,
    }));
    return {
      token: candidates[0].token,
      confidence: candidates[0].score,
      source: "generalized",
      candidates,
    };
  }

  private rankRow(row: Map<string, number>): MovementPredictionCandidate[] {
    let total = 0;
    for (const count of row.values()) {
      total += count;
    }
    const candidates: MovementPredictionCandidate[] = [];
    for (const [id, count] of row) {
      const token = this.vocabulary.get(id);
      if (token) {
        candidates.push({ token, score: total === 0 ? 0 : count / total });
      }
    }
    candidates.sort((a, b) => {
      if (a.score !== b.score) {
        return b.score - a.score;
      }
      return movementTokenId(a.token) < movementTokenId(b.token) ? -1 : 1;
    });
    return candidates;
  }
}

/** Convenience: train with the default deterministic backend. */
export function trainMovementModel(
  dataset: MovementDataset,
  options?: TrainMovementModelOptions,
  backend: MovementModelBackend = new NGramMovementBackend(),
): TrainedMovementModel {
  return backend.train(dataset, options);
}

/** Rehydrate a model from a serialized snapshot (no retraining). */
export function loadMovementModel(snapshot: MovementModelSnapshot): TrainedMovementModel {
  const vocabulary = new Map<string, MovementToken>();
  for (const token of snapshot.vocabulary) {
    vocabulary.set(movementTokenId(token), token);
  }
  const transitions = new Map<string, Map<string, number>>();
  for (const [key, row] of Object.entries(snapshot.transitions)) {
    transitions.set(key, new Map(Object.entries(row)));
  }
  const priors = new Map(Object.entries(snapshot.priors));
  return new NGramMovementModel(snapshot.backend, snapshot.order, vocabulary, transitions, priors);
}

/**
 * Parse a replay action event's `tool` + `summary` into a structured movement
 * token. The capture adapters phrase summaries as "tapped X" / "swiped up" /
 * "typed into Y", so this is a tolerant inverse of that phrasing; unknown
 * shapes fall back to using the first word as the action.
 */
export function parseMovementToken(tool: string, summary: string): MovementToken {
  const words = summary.trim().toLowerCase().split(/\s+/).filter(Boolean);
  const verb = words[0] ?? "";
  const action = normalizeAction(verb);
  const direction = words.find(isDirection);
  const target = extractTarget(words, direction);
  const token: MovementToken = { tool, action };
  if (target) {
    token.target = target;
  }
  if (direction) {
    token.direction = direction;
  }
  return token;
}

function normalizeAction(verb: string): string {
  switch (verb) {
    case "tapped":
      return "tap";
    case "swiped":
      return "swipe";
    case "scrolled":
      return "scroll";
    case "typed":
      return "type";
    case "clicked":
      return "click";
    case "triggered":
      return "shortcut";
    default:
      return verb || "unknown";
  }
}

function isDirection(word: string): word is "up" | "down" | "left" | "right" {
  return word === "up" || word === "down" || word === "left" || word === "right";
}

function extractTarget(words: string[], direction: string | undefined): string | undefined {
  const rest = words.slice(1).filter((word) => word !== "into" && word !== "on" && word !== direction);
  return rest.length > 0 ? rest.join("-") : undefined;
}

type AnyReplayManifest = ReplayManifest | ExportedReplayManifest;

/**
 * Build a training dataset from replay manifests (as produced by the capture
 * exporter). Each manifest's ordered `action` events become one sequence, so
 * the model learns the movement grammar of a recorded session.
 */
export function buildMovementDataset(replays: AnyReplayManifest[]): MovementDataset {
  const sequences: MovementSequence[] = [];
  for (const replay of replays) {
    const tokens = replay.events
      .filter((event): event is Extract<ReplayTimelineEvent, { kind: "action" }> => event.kind === "action")
      .map((event) => parseMovementToken(event.tool, event.summary));
    if (tokens.length > 0) {
      sequences.push({ id: replay.sessionId, tokens });
    }
  }
  return { sequences };
}

export type MovementEvalResult = {
  /** Held-out next-movement predictions attempted. */
  total: number;
  /** Predictions whose top-1 token exactly matched the held-out movement. */
  correct: number;
  /** correct / total, or 0 when nothing was evaluated. */
  accuracy: number;
  /** Of the correct predictions, how many came via generalization. */
  generalizedCorrect: number;
};

/**
 * Generalization eval harness: for each prefix of each held-out sequence, ask
 * the model to predict the next movement and score top-1 accuracy. Because the
 * held-out sequences are *related but unseen*, a non-trivial score demonstrates
 * generalization, not memorization.
 */
export function evaluateMovementModel(
  model: TrainedMovementModel,
  heldOut: MovementSequence[],
): MovementEvalResult {
  let total = 0;
  let correct = 0;
  let generalizedCorrect = 0;
  for (const sequence of heldOut) {
    for (let index = 1; index < sequence.tokens.length; index += 1) {
      const context = sequence.tokens.slice(0, index);
      const expected = sequence.tokens[index];
      const prediction = model.predictNext(context);
      total += 1;
      if (prediction.token && movementTokenId(prediction.token) === movementTokenId(expected)) {
        correct += 1;
        if (prediction.source === "generalized") {
          generalizedCorrect += 1;
        }
      }
    }
  }
  return {
    total,
    correct,
    accuracy: total === 0 ? 0 : correct / total,
    generalizedCorrect,
  };
}

/**
 * Deterministic synthetic movement-stream generator. Validates the full
 * capture→dataset→train→infer loop in the cloud without real OS input: it
 * emits reproducible sequences drawn from a small movement grammar, with a
 * seeded LCG so runs are byte-stable.
 */
export function generateSyntheticMovementSequences(params: {
  count: number;
  lengthEach: number;
  seed?: number;
  vocabulary?: MovementToken[];
}): MovementSequence[] {
  const vocabulary = params.vocabulary ?? DEFAULT_SYNTHETIC_VOCABULARY;
  if (vocabulary.length === 0) {
    return [];
  }
  let state = (params.seed ?? 1) >>> 0 || 1;
  const nextRandom = (): number => {
    // Numerical Recipes LCG — deterministic, no Math.random (which is banned in
    // some run contexts and would break reproducibility anyway).
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 0x100000000;
  };
  const sequences: MovementSequence[] = [];
  for (let index = 0; index < params.count; index += 1) {
    const tokens: MovementToken[] = [];
    let cursor = Math.floor(nextRandom() * vocabulary.length);
    for (let step = 0; step < params.lengthEach; step += 1) {
      tokens.push(vocabulary[cursor]);
      // Movement grammar: usually advance to the neighbouring movement, so the
      // sequences have learnable local structure rather than being uniform noise.
      const advance = nextRandom() < 0.8 ? 1 : Math.floor(nextRandom() * vocabulary.length);
      cursor = (cursor + advance) % vocabulary.length;
    }
    sequences.push({ id: `synthetic-${index}`, tokens });
  }
  return sequences;
}

const DEFAULT_SYNTHETIC_VOCABULARY: MovementToken[] = [
  { tool: "device", action: "tap", target: "search-field" },
  { tool: "device", action: "type", target: "search-field" },
  { tool: "device", action: "tap", target: "result-row" },
  { tool: "device", action: "scroll", direction: "down" },
  { tool: "device", action: "tap", target: "detail-action" },
  { tool: "device", action: "swipe", direction: "left" },
];
