import type { TrajectoryAction, TrajectorySpan } from "../capture/trajectory.js";

/**
 * In-process, pluggable movement-model backend.
 *
 * The training runner (`LocalAppleSiliconTrainingRunner`) emits a *plan* + shell
 * script that runs MLX / axolotl on the user's Apple-Silicon machine. That path
 * cannot execute — nor be validated — in the cloud, so this module provides a
 * pluggable model seam plus a deterministic, dependency-free backend that can be
 * trained and evaluated entirely in-process (and in CI).
 *
 * The deterministic backend is an n-gram (variable-order Markov) model with
 * Katz-style back-off over *movement tokens*. It satisfies both objectives of
 * the local-movement learning subsystem:
 *   - (c) repeat recorded movements: greedy generation from a seen prefix
 *     reproduces the recorded continuation, because the highest-order context
 *     matches exactly and its most-frequent successor is the recorded one.
 *   - (d) generalize to new-but-related movements: an unseen prefix that shares
 *     a shorter suffix with training data still predicts, because back-off
 *     consults progressively shorter contexts until one is known.
 *
 * A real on-device small model (e.g. an MLX LoRA adapter) is a drop-in
 * alternative: implement {@link MovementModelBackend} and hand it to the same
 * dataset. The token vocabulary produced here is exactly what such a model would
 * consume, so the two backends are interchangeable behind this interface.
 */

/** A single discrete unit of movement the model learns to predict. */
export type MovementToken = {
  /** The tool/effector that produced the movement (e.g. "device", "mouse"). */
  tool: string;
  /**
   * The canonical gesture/verb (e.g. "tap", "swipe", "type"). Derived from
   * action metadata when present, else inferred from the summary.
   */
  gesture: string;
  /** Optional direction for directional gestures (up/down/left/right). */
  direction?: string;
  /** Optional target the movement acted on (a button, field, element id). */
  target?: string;
};

/** Canonical string form of a token — the vocabulary symbol. */
export function encodeMovementToken(token: MovementToken): string {
  const parts = [token.tool, token.gesture];
  if (token.direction) {
    parts.push(`>${token.direction}`);
  }
  if (token.target) {
    parts.push(`@${token.target}`);
  }
  return parts.join(":");
}

/** A single training example: an ordered sequence of movement tokens. */
export type MovementSequence = {
  /** Trajectory this sequence was derived from, for traceability. */
  trajectoryId: string;
  tokens: MovementToken[];
};

/** The dataset the backend trains on. */
export type MovementDataset = {
  sequences: MovementSequence[];
};

/** A next-movement prediction with a normalized confidence in [0, 1]. */
export type MovementPrediction = {
  token: MovementToken;
  confidence: number;
  /** How many context tokens actually matched (the back-off order used). */
  matchedOrder: number;
};

/** A trained model: predicts the next movement given a context. */
export interface TrainedMovementModel {
  /** Predict the single most likely next movement, or undefined if the model is empty. */
  predictNext(context: MovementToken[]): MovementPrediction | undefined;
  /** Greedily roll out up to `steps` movements from a seed context. */
  generate(seed: MovementToken[], steps: number): MovementToken[];
  /** Serialize to a plain JSON-safe object (a persistable model artifact). */
  toJSON(): SerializedMovementModel;
}

/** The pluggable backend contract. Swap this for a real on-device model. */
export interface MovementModelBackend {
  readonly id: string;
  train(dataset: MovementDataset): TrainedMovementModel;
}

// --- token derivation ------------------------------------------------------

const DIRECTION_WORDS: Record<string, string> = {
  up: "up",
  down: "down",
  left: "left",
  right: "right",
};

/**
 * Derive a movement token from a captured action. Prefers structured metadata
 * (as emitted by the device adapter) and falls back to parsing the human
 * summary so hand-authored or legacy trajectories still tokenize.
 */
export function actionToMovementToken(action: TrajectoryAction): MovementToken {
  const metadata = action.metadata ?? {};
  const gesture = asString(metadata.gesture) ?? inferGesture(action.summary);
  const direction = asString(metadata.direction) ?? inferDirection(action.summary);
  const target = asString(metadata.target) ?? asString(metadata.valueSummary) ?? inferTarget(action.summary);
  return {
    tool: action.tool,
    gesture,
    ...(direction ? { direction } : {}),
    ...(target ? { target } : {}),
  };
}

function asString(value: unknown): string | undefined {
  return typeof value === "string" && value.length > 0 ? value : undefined;
}

function inferGesture(summary: string): string {
  const first = summary.trim().split(/\s+/)[0]?.toLowerCase() ?? "act";
  // Normalize common past-tense verbs to their canonical gesture.
  const map: Record<string, string> = {
    tapped: "tap",
    swiped: "swipe",
    scrolled: "scroll",
    typed: "type",
    triggered: "shortcut",
    clicked: "tap",
    pressed: "tap",
  };
  return map[first] ?? first;
}

function inferDirection(summary: string): string | undefined {
  for (const word of summary.toLowerCase().split(/\s+/)) {
    if (word in DIRECTION_WORDS) {
      return DIRECTION_WORDS[word];
    }
  }
  return undefined;
}

function inferTarget(summary: string): string | undefined {
  // Grab the token after a preposition ("into X", "on X") if present.
  const match = summary.match(/\b(?:into|on|to)\s+([\w.-]+)/i);
  return match?.[1];
}

/** Turn approved/reviewed trajectories into a movement dataset. */
export function buildMovementDataset(trajectories: TrajectorySpan[]): MovementDataset {
  const sequences: MovementSequence[] = [];
  for (const trajectory of trajectories) {
    const actions = trajectory.review?.redactedActions
      ? trajectory.review.redactedActions.map((action) => ({
          kind: "action" as const,
          tool: action.tool,
          summary: action.summary,
          ts: action.ts,
        }))
      : trajectory.actions;
    const ordered = [...actions].sort((a, b) => a.ts - b.ts);
    const tokens = ordered.map((action) => actionToMovementToken(action));
    if (tokens.length > 0) {
      sequences.push({ trajectoryId: trajectory.id, tokens });
    }
  }
  return { sequences };
}

// --- serialized artifact ---------------------------------------------------

export type SerializedMovementModel = {
  version: 1;
  backend: string;
  order: number;
  /** context-key -> (next-token-symbol -> count). Context "" is the unigram. */
  transitions: Record<string, Record<string, number>>;
  /** symbol -> canonical token, so generation can emit structured tokens. */
  vocabulary: Record<string, MovementToken>;
};

const CONTEXT_DELIMITER = String.fromCharCode(1);

function contextKey(tokens: string[]): string {
  return tokens.join(CONTEXT_DELIMITER);
}

// --- Markov backend --------------------------------------------------------

export type MarkovMovementBackendOptions = {
  /** Maximum context length (n-gram order minus one). Default 3. */
  order?: number;
};

/**
 * Deterministic variable-order Markov backend with back-off. Zero external
 * dependencies, fully reproducible (argmax with a stable lexicographic
 * tie-break — no randomness), so it trains and evaluates identically in the
 * cloud and on-device.
 */
export class MarkovMovementBackend implements MovementModelBackend {
  readonly id = "markov-backoff";
  private readonly order: number;

  constructor(options: MarkovMovementBackendOptions = {}) {
    this.order = Math.max(1, options.order ?? 3);
  }

  train(dataset: MovementDataset): TrainedMovementModel {
    const transitions: Record<string, Record<string, number>> = {};
    const vocabulary: Record<string, MovementToken> = {};

    const bump = (key: string, symbol: string) => {
      const row = (transitions[key] ??= {});
      row[symbol] = (row[symbol] ?? 0) + 1;
    };

    for (const sequence of dataset.sequences) {
      const symbols = sequence.tokens.map((token) => {
        const symbol = encodeMovementToken(token);
        vocabulary[symbol] ??= token;
        return symbol;
      });
      for (let i = 0; i < symbols.length; i += 1) {
        const next = symbols[i];
        // Record contexts of every order from 0..order ending just before i.
        for (let ctxLen = 0; ctxLen <= this.order; ctxLen += 1) {
          if (i - ctxLen < 0) {
            break;
          }
          const context = symbols.slice(i - ctxLen, i);
          bump(contextKey(context), next);
        }
      }
    }

    return new MarkovMovementModel({
      version: 1,
      backend: this.id,
      order: this.order,
      transitions,
      vocabulary,
    });
  }
}

export class MarkovMovementModel implements TrainedMovementModel {
  constructor(private readonly model: SerializedMovementModel) {}

  static fromJSON(model: SerializedMovementModel): MarkovMovementModel {
    return new MarkovMovementModel(model);
  }

  predictNext(context: MovementToken[]): MovementPrediction | undefined {
    const symbols = context.map((token) => encodeMovementToken(token));
    // Back off from the longest usable context to the unigram.
    const maxLen = Math.min(this.model.order, symbols.length);
    for (let ctxLen = maxLen; ctxLen >= 0; ctxLen -= 1) {
      const key = contextKey(symbols.slice(symbols.length - ctxLen));
      const row = this.model.transitions[key];
      if (!row) {
        continue;
      }
      const best = argmax(row);
      if (!best) {
        continue;
      }
      const total = Object.values(row).reduce((sum, count) => sum + count, 0);
      const token = this.model.vocabulary[best.symbol];
      if (!token) {
        continue;
      }
      return {
        token,
        confidence: total > 0 ? best.count / total : 0,
        matchedOrder: ctxLen,
      };
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
      context = [...context, prediction.token].slice(-this.model.order);
    }
    return produced;
  }

  toJSON(): SerializedMovementModel {
    return this.model;
  }
}

/** Argmax over a count row with a stable lexicographic tie-break. */
function argmax(row: Record<string, number>): { symbol: string; count: number } | undefined {
  let best: { symbol: string; count: number } | undefined;
  for (const symbol of Object.keys(row).sort()) {
    const count = row[symbol];
    if (!best || count > best.count) {
      best = { symbol, count };
    }
  }
  return best;
}

/** Convenience: train the default deterministic backend on trajectories. */
export function trainMovementModel(
  trajectories: TrajectorySpan[],
  options: MarkovMovementBackendOptions = {},
): TrainedMovementModel {
  const dataset = buildMovementDataset(trajectories);
  return new MarkovMovementBackend(options).train(dataset);
}
