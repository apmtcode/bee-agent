/**
 * Local-movement model backend.
 *
 * This module closes the "post-train a local model to repeat recorded
 * movements, then generalize to new but related movements" piece of the
 * local-movement learning subsystem. It sits downstream of the capture →
 * replay → export pipeline: the recorder produces {@link ReplayTimelineEvent}
 * timelines, the exporter reviews and packages them, and this module turns
 * those timelines into a trainable movement dataset and learns a model that
 * can predict / regenerate movement sequences.
 *
 * The model backend is intentionally *pluggable*: {@link MovementModelBackend}
 * is a narrow interface, and {@link NGramMovementBackend} is a fully
 * deterministic, dependency-free reference implementation so the pipeline can
 * be trained, evaluated, serialized and replayed in the cloud/CI with no OS
 * access and no native ML runtime. A real on-device small model (e.g. an
 * MLX-backed sequence model, matching {@link LocalAppleSiliconTrainingRunner})
 * can implement the same interface and be swapped in when bee-agent runs
 * locally — see the seam notes on {@link MovementModelBackend}.
 */

import type { ReplayTimelineEvent } from "../capture/replay.js";
import type { ReviewedExportManifest } from "./export-manifest.js";

/** Whether a movement token came from an observation of, or an action on, the machine. */
export type MovementTokenType = "action" | "observation";

/**
 * One atomic movement step. `label` is the tool (for actions, e.g.
 * `pointer.move`, `key.press`, `window.focus`) or the observation source (for
 * observations, e.g. `os.window`, `browser.dom`). Kept coarse on purpose:
 * the model learns the *shape* of a movement sequence, not raw coordinates.
 */
export type MovementToken = {
  type: MovementTokenType;
  label: string;
};

/** Sentinel key marking the end of a movement sequence in the transition tables. */
export const MOVEMENT_END_TOKEN = "<end>";

/** Sentinel key anchoring the start of a movement sequence (start distribution). */
export const MOVEMENT_BEGIN_TOKEN = "<begin>";

/** A single recorded (or synthesized) movement sequence. */
export type MovementSequence = {
  id: string;
  tokens: MovementToken[];
};

/** A trainable collection of movement sequences. */
export type MovementDataset = {
  version: 1;
  sequences: MovementSequence[];
};

/** Result of predicting the single next movement token from a context. */
export type MovementPrediction = {
  /** The predicted next token, or `null` when the model predicts end-of-sequence. */
  token: MovementToken | null;
  /** Serialized key of the predicted token (equals {@link MOVEMENT_END_TOKEN} at end). */
  tokenKey: string;
  /** Conditional probability of the prediction under the chosen backoff context. */
  probability: number;
  /** How many tokens of context were actually used (0 = unconditional prior). */
  backoffOrder: number;
};

export type GenerateOptions = {
  /** Hard cap on generated tokens (excluding the prompt). Defaults to 64. */
  maxSteps?: number;
};

export type TrainOptions = {
  /**
   * n-gram order = maximum context length + 1. `order: 3` learns up to
   * 2-token contexts with backoff to 1-token and unconditional. Defaults to 3.
   */
  order?: number;
};

/**
 * Pluggable local-movement model backend.
 *
 * Contract for implementers (so alternative backends stay interchangeable):
 * - `train` is pure w.r.t. its inputs and MUST be deterministic — no wall
 *   clock, no RNG — so cloud/CI runs are reproducible. On-device backends that
 *   are inherently stochastic should accept a seed via {@link TrainOptions}.
 * - `predictNext` MUST break ties deterministically (the reference backend
 *   uses first-seen order) so replays are stable.
 * - `serialize`/`deserialize` MUST round-trip a model to/from a UTF-8 string
 *   so trained models can be persisted next to the export/job artifacts.
 */
export interface MovementModelBackend<TModel = unknown> {
  /** Stable identifier, recorded alongside serialized models. */
  readonly id: string;
  train(dataset: MovementDataset, options?: TrainOptions): TModel;
  predictNext(model: TModel, context: MovementToken[]): MovementPrediction;
  /** Autoregressively extend `prompt`; returns only the newly generated tokens. */
  generate(model: TModel, prompt: MovementToken[], options?: GenerateOptions): MovementToken[];
  serialize(model: TModel): string;
  deserialize(data: string): TModel;
}

/** Serialize a token to its transition-table key (`type:label`). */
export function serializeMovementToken(token: MovementToken): string {
  return `${token.type}:${token.label}`;
}

/** Parse a transition-table key back into a token, or `null` for the end sentinel. */
export function parseMovementTokenKey(key: string): MovementToken | null {
  if (key === MOVEMENT_END_TOKEN) {
    return null;
  }
  const separator = key.indexOf(":");
  if (separator === -1) {
    return { type: "action", label: key };
  }
  const type = key.slice(0, separator);
  const label = key.slice(separator + 1);
  return {
    type: type === "observation" ? "observation" : "action",
    label,
  };
}

/**
 * Convert a replay timeline into movement tokens. Transcript (dialogue) events
 * are movement-irrelevant and dropped by default; actions and observations
 * become tokens in timeline order.
 */
export function replayEventsToMovementTokens(
  events: ReplayTimelineEvent[],
  options: { includeObservations?: boolean } = {},
): MovementToken[] {
  const includeObservations = options.includeObservations ?? true;
  const tokens: MovementToken[] = [];
  for (const event of events) {
    if (event.kind === "action") {
      tokens.push({ type: "action", label: event.tool });
    } else if (event.kind === "observation" && includeObservations) {
      tokens.push({ type: "observation", label: event.source });
    }
  }
  return tokens;
}

/** Build a training dataset from a reviewed export manifest's replay timelines. */
export function datasetFromReviewedExport(
  manifest: ReviewedExportManifest,
  options: { includeObservations?: boolean } = {},
): MovementDataset {
  const sequences: MovementSequence[] = manifest.replays.map((replay, index) => ({
    id: replay.trajectoryIds[0] ?? `${replay.sessionId}#${index}`,
    tokens: replayEventsToMovementTokens(replay.events as ReplayTimelineEvent[], options),
  }));
  return { version: 1, sequences: sequences.filter((sequence) => sequence.tokens.length > 0) };
}

// ---------------------------------------------------------------------------
// Reference backend: deterministic n-gram model with Katz-style backoff.
// ---------------------------------------------------------------------------

/**
 * A serialized n-gram movement model. `transitions` maps a context key (the
 * ``-joined last L token keys, `""` for the unconditional prior) to an
 * insertion-ordered list of `[nextTokenKey, count]` pairs.
 */
export type NGramMovementModel = {
  version: 1;
  backend: "ngram";
  order: number;
  vocabulary: string[];
  transitions: Record<string, Array<[string, number]>>;
};

const CONTEXT_SEPARATOR = "";

function contextKey(tokenKeys: string[]): string {
  return tokenKeys.join(CONTEXT_SEPARATOR);
}

export class NGramMovementBackend implements MovementModelBackend<NGramMovementModel> {
  readonly id = "ngram";

  train(dataset: MovementDataset, options: TrainOptions = {}): NGramMovementModel {
    const order = Math.max(1, Math.floor(options.order ?? 3));
    // context key -> (next token key -> count), Maps preserve first-seen order.
    const counts = new Map<string, Map<string, number>>();
    const vocabulary = new Set<string>();

    for (const sequence of dataset.sequences) {
      const keys = sequence.tokens.map(serializeMovementToken);
      keys.forEach((key) => vocabulary.add(key));
      // Anchor with BEGIN so the start-of-sequence distribution is distinct
      // from the unconditional (empty-context) unigram backoff.
      const withMarkers = [MOVEMENT_BEGIN_TOKEN, ...keys, MOVEMENT_END_TOKEN];
      for (let i = 1; i < withMarkers.length; i += 1) {
        const target = withMarkers[i];
        const maxContext = Math.min(order - 1, i);
        for (let length = 0; length <= maxContext; length += 1) {
          const context = contextKey(withMarkers.slice(i - length, i));
          let row = counts.get(context);
          if (!row) {
            row = new Map<string, number>();
            counts.set(context, row);
          }
          row.set(target, (row.get(target) ?? 0) + 1);
        }
      }
    }

    const transitions: Record<string, Array<[string, number]>> = {};
    for (const [context, row] of counts) {
      transitions[context] = [...row.entries()];
    }

    return {
      version: 1,
      backend: "ngram",
      order,
      vocabulary: [...vocabulary],
      transitions,
    };
  }

  predictNext(model: NGramMovementModel, context: MovementToken[]): MovementPrediction {
    // Prepend the BEGIN anchor so an empty context resolves to the learned
    // start distribution, and richer contexts can key on "start + first move".
    const keys = [MOVEMENT_BEGIN_TOKEN, ...context.map(serializeMovementToken)];
    const maxContext = Math.min(model.order - 1, keys.length);
    for (let length = maxContext; length >= 0; length -= 1) {
      const key = contextKey(keys.slice(keys.length - length));
      const row = model.transitions[key];
      if (!row || row.length === 0) {
        continue;
      }
      let best = row[0];
      let total = 0;
      for (const entry of row) {
        total += entry[1];
        if (entry[1] > best[1]) {
          best = entry;
        }
      }
      return {
        token: parseMovementTokenKey(best[0]),
        tokenKey: best[0],
        probability: total > 0 ? best[1] / total : 0,
        backoffOrder: length,
      };
    }
    return { token: null, tokenKey: MOVEMENT_END_TOKEN, probability: 0, backoffOrder: 0 };
  }

  generate(
    model: NGramMovementModel,
    prompt: MovementToken[],
    options: GenerateOptions = {},
  ): MovementToken[] {
    const maxSteps = Math.max(0, Math.floor(options.maxSteps ?? 64));
    const context = [...prompt];
    const generated: MovementToken[] = [];
    for (let step = 0; step < maxSteps; step += 1) {
      const prediction = this.predictNext(model, context);
      if (!prediction.token) {
        break;
      }
      generated.push(prediction.token);
      context.push(prediction.token);
    }
    return generated;
  }

  serialize(model: NGramMovementModel): string {
    return JSON.stringify(model);
  }

  deserialize(data: string): NGramMovementModel {
    const parsed = JSON.parse(data) as NGramMovementModel;
    if (parsed.backend !== "ngram") {
      throw new Error(`unexpected movement model backend: ${String(parsed.backend)}`);
    }
    return parsed;
  }
}

// ---------------------------------------------------------------------------
// Evaluation: replay fidelity + generalization to held-out related sequences.
// ---------------------------------------------------------------------------

export type MovementEvaluation = {
  /** Number of next-token predictions scored. */
  total: number;
  /** How many teacher-forced next-token predictions matched the ground truth. */
  correct: number;
  /** `correct / total`, or 1 when there was nothing to score. */
  accuracy: number;
};

/**
 * Teacher-forced next-token accuracy over `sequences` — i.e. replay fidelity.
 * For each position the model is given the true prefix and must predict the
 * true next token (end-of-sequence included). Run on the training sequences it
 * measures reproduction; run on held-out-but-related sequences it measures
 * generalization.
 */
export function evaluateMovementModel<TModel>(
  backend: MovementModelBackend<TModel>,
  model: TModel,
  sequences: MovementSequence[],
): MovementEvaluation {
  let total = 0;
  let correct = 0;
  for (const sequence of sequences) {
    const keys = sequence.tokens.map(serializeMovementToken);
    const withEnd = [...keys, MOVEMENT_END_TOKEN];
    for (let i = 0; i < withEnd.length; i += 1) {
      const prefix = sequence.tokens.slice(0, i);
      const prediction = backend.predictNext(model, prefix);
      total += 1;
      if (prediction.tokenKey === withEnd[i]) {
        correct += 1;
      }
    }
  }
  return { total, correct, accuracy: total === 0 ? 1 : correct / total };
}

// ---------------------------------------------------------------------------
// Synthetic movement generator — validates the pipeline without real OS input.
// ---------------------------------------------------------------------------

/** A named movement "program": a reusable ordered list of movement steps. */
export type MovementProgram = {
  name: string;
  tokens: MovementToken[];
};

/**
 * Materialize a deterministic dataset from movement programs. Each program is
 * emitted `repeats` times with a stable id so capture → dataset → replay
 * round-trips can be exercised in tests with no RNG and no real input capture.
 */
export function buildSyntheticMovementDataset(
  programs: MovementProgram[],
  options: { repeats?: number } = {},
): MovementDataset {
  const repeats = Math.max(1, Math.floor(options.repeats ?? 1));
  const sequences: MovementSequence[] = [];
  for (const program of programs) {
    for (let index = 0; index < repeats; index += 1) {
      sequences.push({
        id: `${program.name}#${index}`,
        tokens: program.tokens.map((token) => ({ ...token })),
      });
    }
  }
  return { version: 1, sequences };
}

/** Convenience: `action` movement token. */
export function actionToken(label: string): MovementToken {
  return { type: "action", label };
}

/** Convenience: `observation` movement token. */
export function observationToken(label: string): MovementToken {
  return { type: "observation", label };
}
