import type { ExportedReplayManifest, ReviewedExportManifest } from "./export-manifest.js";
import type { ReplayTimelineEvent } from "../capture/replay.js";

/**
 * Pluggable local-model backend for the movement-learning subsystem.
 *
 * The standing objective is for bee-agent to post-train a small *local* model on
 * recorded movement/action sequences so it can (a) faithfully repeat a recorded
 * movement and (b) generalise to new-but-related movements. The real on-device
 * path emits an mlx/axolotl launch plan (see `runner.ts`) that only executes on
 * the user's machine. That path cannot be validated in the cloud, so this module
 * adds a backend *seam* plus a deterministic, dependency-free reference backend
 * that actually trains and infers in-process — letting every layer of the
 * pipeline (dataset -> train -> infer -> eval) be exercised by tests here.
 *
 * A real backend (a quantised gguf policy loaded via mlx, say) implements the
 * same `MovementModelBackend` interface and is swapped in via the registry; no
 * caller downstream of `train`/`generate` needs to change.
 */

/** A single movement token — the smallest replayable unit of a movement. */
export type MovementToken = string;

/** An ordered movement (e.g. the actions of one trajectory). */
export type MovementSequence = MovementToken[];

/** Sentinel appended to every training sequence so a model learns where a
 * movement ends, allowing `generate` to stop at the natural length. */
export const MOVEMENT_EOS = "<eos>";

/** Separator used to build an n-gram context key. Chosen to never collide with a
 * realistic tool name or summary. */
const CONTEXT_SEP = "";

export type MovementExample = {
  /** Stable id of the source (typically a trajectory id). */
  id: string;
  tokens: MovementSequence;
};

export type MovementDataset = {
  version: 1;
  examples: MovementExample[];
};

/**
 * A trained movement model. Deliberately a plain JSON-serialisable object so it
 * can be persisted as a training artifact and reloaded for inference without the
 * backend that produced it being resident.
 */
export type MovementModel = {
  backend: string;
  version: 1;
  /** Maximum n-gram context length used for prediction. */
  order: number;
  /** Sorted unique tokens observed during training (excludes EOS). */
  vocabulary: string[];
  /**
   * Backoff transition table: `contextKey -> { nextToken: count }`. Contexts of
   * every length from 0 (the empty global unigram key) up to `order` are stored,
   * enabling Katz-style backoff: an exact high-order match reproduces a recorded
   * movement; a partial suffix match generalises to a related movement.
   */
  transitions: Record<string, Record<MovementToken, number>>;
  /** First-token counts, used to seed generation when no seed is supplied. */
  starts: Record<MovementToken, number>;
  trainedExampleCount: number;
  /** Optional, caller-supplied timestamp (kept out of the model math so training
   * is deterministic). */
  trainedAt?: string;
};

export type TrainMovementModelOptions = {
  /** n-gram order (context window). Defaults to 4. */
  order?: number;
  /** Optional timestamp recorded on the model for provenance. */
  trainedAt?: string;
};

export type GenerateMovementOptions = {
  /** Leading tokens to condition on. May be empty. */
  seed?: MovementSequence;
  /** Hard cap on generated length (excludes the seed). Defaults to 256. */
  maxLength?: number;
  /** Include the seed tokens in the returned sequence. Defaults to true. */
  includeSeed?: boolean;
};

/**
 * The backend seam. Training may be asynchronous (real on-device training is);
 * inference is synchronous against an already-loaded `MovementModel`.
 */
export interface MovementModelBackend {
  readonly name: string;
  train(dataset: MovementDataset, options?: TrainMovementModelOptions): Promise<MovementModel>;
  /** Predict the single most likely next token given a context, or `undefined`
   * if nothing is known (empty model). Never returns the EOS sentinel. */
  predictNext(model: MovementModel, context: MovementSequence): MovementToken | undefined;
  /** Roll out a full movement, stopping at EOS or `maxLength`. */
  generate(model: MovementModel, options?: GenerateMovementOptions): MovementSequence;
}

function contextKey(context: MovementSequence): string {
  return context.join(CONTEXT_SEP);
}

/** Deterministic argmax: highest count wins, ties broken by lexical token order.
 * This makes every train/infer run reproducible — essential for cloud tests. */
function argmax(distribution: Record<MovementToken, number>): MovementToken | undefined {
  let best: MovementToken | undefined;
  let bestCount = -Infinity;
  for (const token of Object.keys(distribution).sort()) {
    const count = distribution[token];
    if (count > bestCount) {
      bestCount = count;
      best = token;
    }
  }
  return best;
}

/**
 * Deterministic n-gram (Markov) movement policy. This is a genuine — if small —
 * local model: it learns a backoff transition table from movement sequences,
 * reproduces a recorded movement exactly when the full-order context is known,
 * and generalises to related movements by backing off to shorter shared
 * suffixes. Zero external dependencies, fully deterministic.
 */
export class MarkovMovementBackend implements MovementModelBackend {
  readonly name = "markov-mock";

  async train(dataset: MovementDataset, options: TrainMovementModelOptions = {}): Promise<MovementModel> {
    const order = Math.max(1, options.order ?? 4);
    const transitions: Record<string, Record<MovementToken, number>> = {};
    const starts: Record<MovementToken, number> = {};
    const vocabulary = new Set<MovementToken>();

    let exampleCount = 0;
    for (const example of dataset.examples) {
      if (example.tokens.length === 0) {
        continue;
      }
      exampleCount += 1;
      // Terminate the sequence so the model learns its natural end.
      const tokens = [...example.tokens, MOVEMENT_EOS];
      starts[tokens[0]] = (starts[tokens[0]] ?? 0) + 1;

      for (let i = 0; i < tokens.length; i += 1) {
        const next = tokens[i];
        if (next !== MOVEMENT_EOS) {
          vocabulary.add(next);
        }
        // Record every context length 0..order that precedes this token.
        for (let k = 0; k <= order; k += 1) {
          if (i - k < 0) {
            break;
          }
          const key = contextKey(tokens.slice(i - k, i));
          const dist = (transitions[key] ??= {});
          dist[next] = (dist[next] ?? 0) + 1;
        }
      }
    }

    return {
      backend: this.name,
      version: 1,
      order,
      vocabulary: [...vocabulary].sort(),
      transitions,
      starts,
      trainedExampleCount: exampleCount,
      ...(options.trainedAt ? { trainedAt: options.trainedAt } : {}),
    };
  }

  /** Internal: predict next token, optionally allowing EOS (used by generate). */
  private predictWithEos(model: MovementModel, context: MovementSequence): MovementToken | undefined {
    const clamped = context.slice(Math.max(0, context.length - model.order));
    for (let k = clamped.length; k >= 0; k -= 1) {
      const key = contextKey(clamped.slice(clamped.length - k));
      const dist = model.transitions[key];
      if (dist && Object.keys(dist).length > 0) {
        return argmax(dist);
      }
    }
    return undefined;
  }

  predictNext(model: MovementModel, context: MovementSequence): MovementToken | undefined {
    const predicted = this.predictWithEos(model, context);
    if (predicted === undefined || predicted === MOVEMENT_EOS) {
      // Fall back to the most likely non-terminal token so callers asking for a
      // concrete movement never receive the sentinel.
      const clamped = context.slice(Math.max(0, context.length - model.order));
      for (let k = clamped.length; k >= 0; k -= 1) {
        const key = contextKey(clamped.slice(clamped.length - k));
        const dist = model.transitions[key];
        if (!dist) {
          continue;
        }
        const filtered: Record<MovementToken, number> = {};
        for (const [token, count] of Object.entries(dist)) {
          if (token !== MOVEMENT_EOS) {
            filtered[token] = count;
          }
        }
        const best = argmax(filtered);
        if (best !== undefined) {
          return best;
        }
      }
      return undefined;
    }
    return predicted;
  }

  generate(model: MovementModel, options: GenerateMovementOptions = {}): MovementSequence {
    const maxLength = options.maxLength ?? 256;
    const includeSeed = options.includeSeed ?? true;
    const seed = options.seed ? [...options.seed] : [];

    const context: MovementSequence = [...seed];
    if (context.length === 0) {
      const first = argmax(model.starts);
      if (first === undefined) {
        return [];
      }
      context.push(first);
    }

    const generatedFromSeed: MovementSequence = [];
    // If we synthesised a start token above (no seed), it belongs to the output.
    if (seed.length === 0) {
      generatedFromSeed.push(context[0]);
    }

    for (let step = 0; step < maxLength; step += 1) {
      const next = this.predictWithEos(model, context);
      if (next === undefined || next === MOVEMENT_EOS) {
        break;
      }
      generatedFromSeed.push(next);
      context.push(next);
    }

    return includeSeed ? [...seed, ...generatedFromSeed] : generatedFromSeed;
  }
}

/** Registry that makes the movement-model backend pluggable. */
export class MovementModelBackendRegistry {
  private readonly backends = new Map<string, MovementModelBackend>();

  register(backend: MovementModelBackend): this {
    this.backends.set(backend.name, backend);
    return this;
  }

  get(name: string): MovementModelBackend | undefined {
    return this.backends.get(name);
  }

  require(name: string): MovementModelBackend {
    const backend = this.backends.get(name);
    if (!backend) {
      throw new Error(`Unknown movement-model backend: ${name}`);
    }
    return backend;
  }

  list(): string[] {
    return [...this.backends.keys()].sort();
  }
}

/**
 * Registry pre-loaded with the deterministic reference backend. A real on-device
 * backend (mlx/gguf) is added with `.register(...)` at startup without changing
 * any downstream caller.
 */
export function createDefaultMovementBackendRegistry(): MovementModelBackendRegistry {
  return new MovementModelBackendRegistry().register(new MarkovMovementBackend());
}

/** How a replay event is turned into a movement token. Pluggable so callers can
 * encode coordinates/targets when a richer movement vocabulary is desired. */
export type MovementTokenizer = (event: Extract<ReplayTimelineEvent, { kind: "action" }>) => MovementToken;

/** Default tokenizer: the action's tool name is the movement primitive. */
export const defaultMovementTokenizer: MovementTokenizer = (event) => event.tool;

/**
 * Build a movement dataset from reviewed-export replay manifests, grouping the
 * `action` events of each trajectory into one ordered movement sequence.
 */
export function createMovementDatasetFromReplays(
  replays: ExportedReplayManifest[],
  tokenizer: MovementTokenizer = defaultMovementTokenizer,
): MovementDataset {
  const byTrajectory = new Map<string, Array<{ ts: number; token: MovementToken }>>();

  for (const replay of replays) {
    for (const event of replay.events) {
      if (event.kind !== "action") {
        continue;
      }
      const bucket = byTrajectory.get(event.trajectoryId) ?? [];
      bucket.push({ ts: event.ts, token: tokenizer(event) });
      byTrajectory.set(event.trajectoryId, bucket);
    }
  }

  const examples: MovementExample[] = [];
  for (const [id, entries] of [...byTrajectory.entries()].sort((a, b) => a[0].localeCompare(b[0]))) {
    const tokens = entries.sort((a, b) => a.ts - b.ts).map((entry) => entry.token);
    if (tokens.length > 0) {
      examples.push({ id, tokens });
    }
  }

  return { version: 1, examples };
}

/** Convenience: derive a dataset straight from a reviewed export manifest. */
export function createMovementDatasetFromExport(
  manifest: ReviewedExportManifest,
  tokenizer: MovementTokenizer = defaultMovementTokenizer,
): MovementDataset {
  return createMovementDatasetFromReplays(manifest.replays, tokenizer);
}

export type MovementReplayFidelity = {
  id: string;
  expected: MovementSequence;
  reproduced: MovementSequence;
  /** Fraction of expected tokens reproduced in-position (0..1). */
  accuracy: number;
  exactMatch: boolean;
};

export type MovementEvaluation = {
  backend: string;
  exampleCount: number;
  exactMatchCount: number;
  /** Mean per-example accuracy across all examples (0..1). */
  meanAccuracy: number;
  perExample: MovementReplayFidelity[];
};

/**
 * Generalisation/fidelity eval harness. For each example we regenerate the
 * movement from its first token and measure how faithfully the model reproduces
 * it. Run against training data it measures memorisation; run against held-out
 * related sequences it measures generalisation.
 */
export function evaluateMovementModel(
  backend: MovementModelBackend,
  model: MovementModel,
  dataset: MovementDataset,
): MovementEvaluation {
  const perExample: MovementReplayFidelity[] = [];
  let exactMatchCount = 0;
  let accuracySum = 0;

  for (const example of dataset.examples) {
    const expected = example.tokens;
    const reproduced =
      expected.length === 0
        ? []
        : backend.generate(model, {
            seed: [expected[0]],
            maxLength: expected.length, // allow generating the remaining tokens
            includeSeed: true,
          });

    let matched = 0;
    for (let i = 0; i < expected.length; i += 1) {
      if (reproduced[i] === expected[i]) {
        matched += 1;
      }
    }
    const accuracy = expected.length === 0 ? 1 : matched / expected.length;
    const exactMatch = accuracy === 1 && reproduced.length === expected.length;
    if (exactMatch) {
      exactMatchCount += 1;
    }
    accuracySum += accuracy;
    perExample.push({ id: example.id, expected, reproduced, accuracy, exactMatch });
  }

  const exampleCount = dataset.examples.length;
  return {
    backend: backend.name,
    exampleCount,
    exactMatchCount,
    meanAccuracy: exampleCount === 0 ? 1 : accuracySum / exampleCount,
    perExample,
  };
}
