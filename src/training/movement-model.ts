import type { ReplayManifest, ReplayTimelineEvent } from "../capture/replay.js";
import type { TrajectorySpan } from "../capture/trajectory.js";

/**
 * Local-movement learning subsystem — model layer (objective #2, parts c & d).
 *
 * This module turns reviewed movement recordings into a token dataset and
 * post-trains a *local* model that can (c) repeat recorded movements and
 * (d) generalize to new-but-related movements. The model backend is pluggable:
 * the cloud/CI ships a deterministic in-process n-gram backend (no OS access,
 * no external process, fully testable), and a real on-device small-model
 * backend can be registered under the same interface when bee-agent runs
 * locally. Nothing here spawns a process or touches the real machine.
 */

/** A canonical movement token, e.g. `device::tapped submit`. */
export type MovementToken = string;

/** Terminal token appended to every training sequence so the model can learn where a movement ends. */
export const MOVEMENT_END_TOKEN: MovementToken = "<end>";

export type MovementSequence = {
  /** Stable id of the source (trajectory id, replay session id, etc.). */
  id: string;
  tokens: MovementToken[];
};

export type MovementDataset = {
  version: 1;
  sequences: MovementSequence[];
  /** Sorted, de-duplicated set of every token seen across all sequences (excludes the end token). */
  vocabulary: MovementToken[];
};

/** Canonicalize a single action into a stable, comparable movement token. */
export function movementTokenFromAction(tool: string, summary: string): MovementToken {
  const normalizedTool = tool.trim().toLowerCase() || "unknown";
  const normalizedSummary = summary.trim().replace(/\s+/g, " ").toLowerCase();
  return `${normalizedTool}::${normalizedSummary}`;
}

function buildDataset(sequences: MovementSequence[]): MovementDataset {
  const nonEmpty = sequences.filter((sequence) => sequence.tokens.length > 0);
  const vocabulary = [
    ...new Set(nonEmpty.flatMap((sequence) => sequence.tokens).filter((token) => token !== MOVEMENT_END_TOKEN)),
  ].sort();
  return { version: 1, sequences: nonEmpty, vocabulary };
}

/**
 * Build a movement dataset from trajectory spans (uses each span's action list,
 * in recorded order). An `<end>` token is appended per span so the model learns
 * completion boundaries.
 */
export function buildMovementDatasetFromTrajectories(trajectories: TrajectorySpan[]): MovementDataset {
  const sequences = trajectories.map<MovementSequence>((trajectory) => {
    const tokens = [...trajectory.actions]
      .sort((a, b) => a.ts - b.ts)
      .map((action) => movementTokenFromAction(action.tool, action.summary));
    return { id: trajectory.id, tokens: tokens.length > 0 ? [...tokens, MOVEMENT_END_TOKEN] : [] };
  });
  return buildDataset(sequences);
}

/**
 * Build a movement dataset from replay manifests. Only `action` timeline events
 * are treated as movements; transcript/observation events are context, not
 * motor actions to be replayed.
 */
export function buildMovementDatasetFromReplays(replays: ReplayManifest[]): MovementDataset {
  const sequences = replays.map<MovementSequence>((replay) => {
    const tokens = replay.events
      .filter((event): event is Extract<ReplayTimelineEvent, { kind: "action" }> => event.kind === "action")
      .slice()
      .sort((a, b) => a.ts - b.ts)
      .map((event) => movementTokenFromAction(event.tool, event.summary));
    return { id: replay.sessionId, tokens: tokens.length > 0 ? [...tokens, MOVEMENT_END_TOKEN] : [] };
  });
  return buildDataset(sequences);
}

export type MovementPrediction = {
  token: MovementToken;
  /** Empirical probability of this token within the matched context bucket. */
  confidence: number;
  /**
   * Context length actually used to make the prediction. Equals the requested
   * context length for an exact recall; a smaller value means the model backed
   * off — i.e. it *generalized* from a shorter shared suffix.
   */
  matchedOrder: number;
  /** True when the prediction required backing off below the full context (generalization). */
  generalized: boolean;
};

export type GenerateOptions = {
  /** Hard cap on generated tokens (excludes the terminal token). Defaults to 64. */
  maxSteps?: number;
  /** Stop as soon as the model emits the end token. Defaults to true. */
  stopAtEnd?: boolean;
};

export type SerializedMovementModel = {
  backendId: string;
  order: number;
  vocabulary: MovementToken[];
  /** context-key -> (nextToken -> count). The empty string key holds unigram counts. */
  transitions: Record<string, Record<MovementToken, number>>;
};

/** A trained, in-memory movement model. Serializable so a post-trained artifact can be persisted and reloaded. */
export interface TrainedMovementModel {
  readonly backendId: string;
  readonly order: number;
  /** Predict the most likely next movement given a prior context, with deterministic tie-breaking. */
  predictNext(context: MovementToken[]): MovementPrediction | undefined;
  /** Autoregressively roll out a movement continuation from a seed context. */
  generate(seed: MovementToken[], options?: GenerateOptions): MovementToken[];
  serialize(): SerializedMovementModel;
}

export type TrainOptions = {
  /** Maximum Markov context order. Higher = more literal recall, less generalization. Defaults to 3. */
  order?: number;
};

/** A pluggable movement-model backend. Register a real on-device backend under the same interface. */
export interface MovementModelBackend {
  readonly id: string;
  train(dataset: MovementDataset, options?: TrainOptions): TrainedMovementModel;
  restore(serialized: SerializedMovementModel): TrainedMovementModel;
}

function contextKey(context: MovementToken[]): string {
  return context.join("␟");
}

/** Deterministic argmax over a count bucket: highest count wins, lexicographic token order breaks ties. */
function argmax(counts: Record<MovementToken, number>): { token: MovementToken; count: number; total: number } | undefined {
  let best: { token: MovementToken; count: number } | undefined;
  let total = 0;
  for (const [token, count] of Object.entries(counts)) {
    total += count;
    if (!best || count > best.count || (count === best.count && token < best.token)) {
      best = { token, count };
    }
  }
  return best ? { ...best, total } : undefined;
}

class NgramMovementModel implements TrainedMovementModel {
  constructor(
    readonly backendId: string,
    readonly order: number,
    private readonly vocabulary: MovementToken[],
    private readonly transitions: Map<string, Record<MovementToken, number>>,
  ) {}

  predictNext(context: MovementToken[]): MovementPrediction | undefined {
    // Try the longest suffix of the context first, then back off. Backing off
    // below the full context length is exactly how the model generalizes to a
    // new-but-related movement it never saw verbatim.
    const maxLen = Math.min(context.length, this.order);
    for (let len = maxLen; len >= 0; len -= 1) {
      const suffix = len === 0 ? [] : context.slice(context.length - len);
      const bucket = this.transitions.get(contextKey(suffix));
      if (!bucket) {
        continue;
      }
      const best = argmax(bucket);
      if (!best) {
        continue;
      }
      return {
        token: best.token,
        confidence: best.total > 0 ? best.count / best.total : 0,
        matchedOrder: len,
        generalized: len < Math.min(context.length, this.order),
      };
    }
    return undefined;
  }

  generate(seed: MovementToken[], options: GenerateOptions = {}): MovementToken[] {
    const maxSteps = options.maxSteps ?? 64;
    const stopAtEnd = options.stopAtEnd ?? true;
    const output: MovementToken[] = [];
    const context = [...seed];
    for (let step = 0; step < maxSteps; step += 1) {
      const prediction = this.predictNext(context);
      if (!prediction) {
        break;
      }
      if (prediction.token === MOVEMENT_END_TOKEN) {
        if (stopAtEnd) {
          break;
        }
        // Not stopping at end: keep the boundary in context but don't emit it.
        context.push(prediction.token);
        continue;
      }
      output.push(prediction.token);
      context.push(prediction.token);
    }
    return output;
  }

  serialize(): SerializedMovementModel {
    const transitions: Record<string, Record<MovementToken, number>> = {};
    for (const [key, counts] of this.transitions.entries()) {
      transitions[key] = { ...counts };
    }
    return {
      backendId: this.backendId,
      order: this.order,
      vocabulary: [...this.vocabulary],
      transitions,
    };
  }
}

/**
 * Deterministic, dependency-free n-gram movement backend with Katz-style
 * backoff. It "post-trains" by counting context→next-token transitions across
 * the reviewed dataset and predicts by deterministic argmax over the longest
 * matching context. This is the default cloud/CI backend; it needs no OS access
 * and no external process, so the whole train→infer→generalize loop is testable.
 */
export class NgramMovementBackend implements MovementModelBackend {
  readonly id = "ngram-local";

  train(dataset: MovementDataset, options: TrainOptions = {}): TrainedMovementModel {
    const order = Math.max(1, Math.floor(options.order ?? 3));
    const transitions = new Map<string, Record<MovementToken, number>>();

    const bump = (context: MovementToken[], next: MovementToken): void => {
      const key = contextKey(context);
      const bucket = transitions.get(key) ?? {};
      bucket[next] = (bucket[next] ?? 0) + 1;
      transitions.set(key, bucket);
    };

    for (const sequence of dataset.sequences) {
      const tokens = sequence.tokens;
      for (let i = 0; i < tokens.length; i += 1) {
        const next = tokens[i]!;
        // Record every context length from 0 (unigram) up to `order`, so
        // prediction can back off smoothly at inference time.
        for (let len = 0; len <= order; len += 1) {
          if (i - len < 0) {
            break;
          }
          bump(tokens.slice(i - len, i), next);
        }
      }
    }

    return new NgramMovementModel(this.id, order, [...dataset.vocabulary], transitions);
  }

  restore(serialized: SerializedMovementModel): TrainedMovementModel {
    const transitions = new Map<string, Record<MovementToken, number>>(
      Object.entries(serialized.transitions).map(([key, counts]) => [key, { ...counts }]),
    );
    return new NgramMovementModel(serialized.backendId, serialized.order, [...serialized.vocabulary], transitions);
  }
}

/** Registry of pluggable movement-model backends. Seeded with the deterministic n-gram backend. */
export class MovementModelBackendRegistry {
  private readonly backends = new Map<string, MovementModelBackend>();

  constructor(backends: MovementModelBackend[] = [new NgramMovementBackend()]) {
    for (const backend of backends) {
      this.register(backend);
    }
  }

  register(backend: MovementModelBackend): void {
    this.backends.set(backend.id, backend);
  }

  has(id: string): boolean {
    return this.backends.has(id);
  }

  get(id: string): MovementModelBackend {
    const backend = this.backends.get(id);
    if (!backend) {
      throw new Error(`Unknown movement-model backend: ${id}`);
    }
    return backend;
  }

  list(): string[] {
    return [...this.backends.keys()].sort();
  }

  restore(serialized: SerializedMovementModel): TrainedMovementModel {
    return this.get(serialized.backendId).restore(serialized);
  }
}

export type MovementModelEvaluation = {
  /** Number of (context, expected-next) prediction points scored. */
  samples: number;
  /** Fraction of prediction points where the model's top token matched the held-out next token. */
  accuracy: number;
  /** Of the correct predictions, the fraction that required backing off (i.e. were genuine generalizations). */
  generalizationRate: number;
  correct: number;
  generalizedCorrect: number;
};

/**
 * Generalization eval harness: score a trained model on held-out sequences it
 * was not trained on. For each position we ask the model to predict the next
 * token from the true prefix and compare to ground truth. `generalizationRate`
 * isolates the correct predictions that came from backoff — evidence the model
 * transfers to new-but-related movements rather than only recalling verbatim.
 */
export function evaluateMovementModel(
  model: TrainedMovementModel,
  heldOut: MovementSequence[],
): MovementModelEvaluation {
  let samples = 0;
  let correct = 0;
  let generalizedCorrect = 0;

  for (const sequence of heldOut) {
    for (let i = 0; i < sequence.tokens.length; i += 1) {
      const expected = sequence.tokens[i]!;
      const context = sequence.tokens.slice(0, i);
      const prediction = model.predictNext(context);
      samples += 1;
      if (prediction && prediction.token === expected) {
        correct += 1;
        if (prediction.generalized) {
          generalizedCorrect += 1;
        }
      }
    }
  }

  return {
    samples,
    correct,
    generalizedCorrect,
    accuracy: samples > 0 ? correct / samples : 0,
    generalizationRate: correct > 0 ? generalizedCorrect / correct : 0,
  };
}

/** Convenience: train a movement model with a chosen (or default) backend. */
export function trainMovementModel(
  dataset: MovementDataset,
  options: TrainOptions & { backend?: MovementModelBackend } = {},
): TrainedMovementModel {
  const backend = options.backend ?? new NgramMovementBackend();
  return backend.train(dataset, options);
}
