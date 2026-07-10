/**
 * Local-movement learning model.
 *
 * This module closes the "post-train a local model on the recorded dataset and
 * generalize to new but related movements" pieces of the movement-learning
 * subsystem (standing objective 2c + 2d). It is deliberately runnable *in
 * process* — no Python, MLX, or GPU — so the capture → dataset → train → infer
 * loop can be validated in the cloud/CI with synthetic event streams.
 *
 * The design is a **pluggable backend**: {@link MovementModelBackend} is the
 * training seam and {@link TrainedMovementModel} is the inference seam. The
 * deterministic {@link MarkovMovementBackend} shipped here learns a
 * variable-order Markov (n-gram) model with stupid-backoff over movement
 * tokens. High-order context reproduces recorded movements exactly ("repeat the
 * recorded movements"); when a prefix was never seen at full order the model
 * backs off to shorter contexts, which is how it generalizes to new-but-related
 * movements. A real on-device small model (e.g. an MLX/GGUF adapter produced by
 * `LocalAppleSiliconTrainingRunner`) can implement the same two interfaces and
 * be dropped in without touching callers.
 */

import type { ReplayManifest } from "../capture/replay.js";
import type { TrajectoryAction, TrajectoryObservation, TrajectorySpan } from "../capture/trajectory.js";

/** A single discrete movement/observation symbol the model reasons over. */
export type MovementToken = string;

/** One recorded (or synthetic) movement sequence: an ordered list of tokens. */
export type MovementSequence = {
  id: string;
  tokens: MovementToken[];
};

/** The replayable, model-ready dataset produced from trajectories/replays. */
export type MovementDataset = {
  version: 1;
  sequences: MovementSequence[];
};

/** Sentinel tokens used internally for sequence boundaries. */
export const MOVEMENT_BOS: MovementToken = "bos";
export const MOVEMENT_EOS: MovementToken = "eos";

export type MovementTokenizerOptions = {
  /** Include observation events as context tokens (state → movement). */
  includeObservations?: boolean;
  /** Include a gesture direction in device tokens when present. */
  includeDirection?: boolean;
};

const DEFAULT_TOKENIZER_OPTIONS: Required<MovementTokenizerOptions> = {
  includeObservations: true,
  includeDirection: true,
};

/**
 * Canonicalize a trajectory action into a movement token. Device gestures
 * (mouse/keyboard/UI events) become structured `tool:gesture[:direction]`
 * tokens so sequences of *kinds* — the transferable structure — are learned
 * rather than one-off free-text targets.
 */
export function tokenizeAction(
  action: Pick<TrajectoryAction, "tool" | "summary" | "metadata">,
  options: MovementTokenizerOptions = {},
): MovementToken {
  const opts = { ...DEFAULT_TOKENIZER_OPTIONS, ...options };
  const metadata = action.metadata ?? {};
  const gesture = typeof metadata.gesture === "string" ? metadata.gesture : undefined;
  if (gesture) {
    const direction = opts.includeDirection && typeof metadata.direction === "string" ? metadata.direction : undefined;
    return direction ? `${action.tool}:${gesture}:${direction}` : `${action.tool}:${gesture}`;
  }
  return `act:${action.tool}`;
}

/** Canonicalize an observation into a lightweight context token. */
export function tokenizeObservation(
  observation: Pick<TrajectoryObservation, "source" | "summary" | "metadata">,
): MovementToken {
  return `obs:${observation.source}`;
}

/**
 * Build a model-ready dataset from trajectory spans. Each span becomes one
 * sequence, interleaving observation and action tokens in timestamp order.
 * Redacted (reviewed) events are preferred when present so training only ever
 * sees consent-approved data.
 */
export function buildMovementDatasetFromTrajectories(
  trajectories: TrajectorySpan[],
  options: MovementTokenizerOptions = {},
): MovementDataset {
  const opts = { ...DEFAULT_TOKENIZER_OPTIONS, ...options };
  const sequences: MovementSequence[] = [];
  for (const trajectory of trajectories) {
    const events: Array<{ ts: number; token: MovementToken }> = [];
    const actions = trajectory.review?.redactedActions ?? trajectory.actions;
    for (const action of actions) {
      events.push({ ts: action.ts, token: tokenizeAction(action as TrajectoryAction, opts) });
    }
    if (opts.includeObservations) {
      const observations = trajectory.review?.redactedObservations ?? trajectory.observations;
      for (const observation of observations) {
        events.push({ ts: observation.ts, token: tokenizeObservation(observation as TrajectoryObservation) });
      }
    }
    events.sort((a, b) => a.ts - b.ts);
    if (events.length > 0) {
      sequences.push({ id: trajectory.id, tokens: events.map((event) => event.token) });
    }
  }
  return { version: 1, sequences };
}

/** Build a dataset from replay manifests (the reviewed-export representation). */
export function buildMovementDatasetFromReplays(
  replays: ReplayManifest[],
  options: MovementTokenizerOptions = {},
): MovementDataset {
  const opts = { ...DEFAULT_TOKENIZER_OPTIONS, ...options };
  const sequences: MovementSequence[] = [];
  for (const replay of replays) {
    const tokens: MovementToken[] = [];
    for (const event of replay.events) {
      if (event.kind === "action") {
        tokens.push(tokenizeAction({ tool: event.tool, summary: event.summary, metadata: undefined }, opts));
      } else if (event.kind === "observation" && opts.includeObservations) {
        tokens.push(tokenizeObservation({ source: event.source, summary: event.summary, metadata: undefined }));
      }
    }
    if (tokens.length > 0) {
      sequences.push({ id: replay.sessionId, tokens });
    }
  }
  return { version: 1, sequences };
}

export type MovementTrainingOptions = {
  /** Maximum n-gram context order (default 3). */
  order?: number;
};

/** A ranked next-movement prediction with backoff transparency. */
export type MovementPrediction = {
  token: MovementToken;
  probability: number;
  /** Context order actually used to produce this prediction (backoff depth). */
  order: number;
};

export type SerializedMovementModel = {
  version: 1;
  backend: string;
  order: number;
  /** Per-order transition counts. Index = context length. */
  levels: Array<Array<{ context: MovementToken[]; next: Array<{ token: MovementToken; count: number }> }>>;
};

/** Inference seam. Any backend (Markov, MLX adapter, …) returns one of these. */
export interface TrainedMovementModel {
  readonly backend: string;
  readonly order: number;
  /** Best next movement for a context, or undefined if the model is empty. */
  predictNext(context: MovementToken[]): MovementPrediction | undefined;
  /** All candidate next movements at the chosen backoff order, best first. */
  rankNext(context: MovementToken[]): MovementPrediction[];
  /** Continue a movement sequence from a seed until EOS or maxSteps. */
  generate(seed: MovementToken[], maxSteps: number): MovementToken[];
  serialize(): SerializedMovementModel;
}

/** Training seam. Real on-device backends implement this same interface. */
export interface MovementModelBackend {
  readonly name: string;
  train(dataset: MovementDataset, options?: MovementTrainingOptions): Promise<TrainedMovementModel>;
}

const CONTEXT_SEPARATOR = "|";

type CountMap = Map<string, Map<MovementToken, number>>;

/**
 * Deterministic variable-order Markov model with stupid-backoff. Greedy,
 * tie-broken lexicographically, so training and inference are fully
 * reproducible (no RNG) — a requirement for testable self-evolution.
 */
export class MarkovMovementModel implements TrainedMovementModel {
  readonly backend = "markov";

  private constructor(
    readonly order: number,
    private readonly levels: CountMap[],
  ) {}

  static train(dataset: MovementDataset, options: MovementTrainingOptions = {}): MarkovMovementModel {
    const order = Math.max(0, Math.floor(options.order ?? 3));
    const levels: CountMap[] = Array.from({ length: order + 1 }, () => new Map());
    for (const sequence of dataset.sequences) {
      const padded = [
        ...Array.from({ length: order }, () => MOVEMENT_BOS),
        ...sequence.tokens,
        MOVEMENT_EOS,
      ];
      for (let i = order; i < padded.length; i += 1) {
        const next = padded[i]!;
        for (let o = 0; o <= order; o += 1) {
          const context = padded.slice(i - o, i);
          record(levels[o]!, context, next);
        }
      }
    }
    return new MarkovMovementModel(order, levels);
  }

  static fromSerialized(serialized: SerializedMovementModel): MarkovMovementModel {
    const levels: CountMap[] = Array.from({ length: serialized.order + 1 }, () => new Map());
    serialized.levels.forEach((entries, o) => {
      const level = levels[o];
      if (!level) {
        return;
      }
      for (const entry of entries) {
        const counts = new Map<MovementToken, number>();
        for (const { token, count } of entry.next) {
          counts.set(token, count);
        }
        level.set(entry.context.join(CONTEXT_SEPARATOR), counts);
      }
    });
    return new MarkovMovementModel(serialized.order, levels);
  }

  rankNext(context: MovementToken[]): MovementPrediction[] {
    for (let o = Math.min(this.order, context.length); o >= 0; o -= 1) {
      const suffix = context.slice(context.length - o);
      const counts = this.levels[o]?.get(suffix.join(CONTEXT_SEPARATOR));
      if (!counts || counts.size === 0) {
        continue;
      }
      const total = [...counts.values()].reduce((sum, value) => sum + value, 0);
      return [...counts.entries()]
        .map(([token, count]) => ({ token, probability: count / total, order: o }))
        .sort((a, b) => (b.probability - a.probability) || (a.token < b.token ? -1 : a.token > b.token ? 1 : 0));
    }
    return [];
  }

  predictNext(context: MovementToken[]): MovementPrediction | undefined {
    return this.rankNext(context)[0];
  }

  generate(seed: MovementToken[], maxSteps: number): MovementToken[] {
    const context = [...seed];
    const generated: MovementToken[] = [];
    for (let step = 0; step < maxSteps; step += 1) {
      const prediction = this.predictNext(context);
      if (!prediction || prediction.token === MOVEMENT_EOS) {
        break;
      }
      generated.push(prediction.token);
      context.push(prediction.token);
    }
    return generated;
  }

  serialize(): SerializedMovementModel {
    return {
      version: 1,
      backend: this.backend,
      order: this.order,
      levels: this.levels.map((level) =>
        [...level.entries()].map(([contextKey, counts]) => ({
          context: contextKey.length === 0 ? [] : contextKey.split(CONTEXT_SEPARATOR),
          next: [...counts.entries()].map(([token, count]) => ({ token, count })),
        })),
      ),
    };
  }
}

/** Default pluggable backend: the in-process deterministic Markov model. */
export class MarkovMovementBackend implements MovementModelBackend {
  readonly name = "markov";

  // eslint-disable-next-line @typescript-eslint/require-await
  async train(dataset: MovementDataset, options: MovementTrainingOptions = {}): Promise<TrainedMovementModel> {
    return MarkovMovementModel.train(dataset, options);
  }
}

export type MovementEvalReport = {
  sequences: number;
  predictions: number;
  correct: number;
  /** Teacher-forced top-1 next-token accuracy over held-out sequences. */
  accuracy: number;
  /** Fraction of held-out sequences reproduced exactly from their first token. */
  exactReplayRate: number;
};

/**
 * Generalization eval harness. Measures how well a trained model predicts
 * held-out (unseen) but related movement sequences: both step-by-step
 * next-token accuracy (teacher-forced) and whole-sequence exact replay from a
 * one-token seed.
 */
export function evaluateMovementModel(
  model: TrainedMovementModel,
  heldOut: MovementSequence[],
): MovementEvalReport {
  let predictions = 0;
  let correct = 0;
  let exactReplays = 0;
  let scorableSequences = 0;
  for (const sequence of heldOut) {
    if (sequence.tokens.length < 2) {
      continue;
    }
    scorableSequences += 1;
    for (let i = 1; i < sequence.tokens.length; i += 1) {
      const prediction = model.predictNext(sequence.tokens.slice(0, i));
      predictions += 1;
      if (prediction?.token === sequence.tokens[i]) {
        correct += 1;
      }
    }
    const seed = sequence.tokens.slice(0, 1);
    const generated = model.generate(seed, sequence.tokens.length);
    const reproduced = [...seed, ...generated];
    if (reproduced.length === sequence.tokens.length && reproduced.every((token, index) => token === sequence.tokens[index])) {
      exactReplays += 1;
    }
  }
  return {
    sequences: scorableSequences,
    predictions,
    correct,
    accuracy: predictions === 0 ? 0 : correct / predictions,
    exactReplayRate: scorableSequences === 0 ? 0 : exactReplays / scorableSequences,
  };
}

/**
 * Deterministic synthetic movement-stream generator. Expands a small grammar of
 * related task flows so the capture → dataset → train → infer loop can be
 * validated without real OS input. No RNG — variation comes from the `seed`
 * index — so runs are reproducible.
 */
export function generateSyntheticMovementSequences(count: number, seed = 0): MovementSequence[] {
  const flows: MovementToken[][] = [
    ["obs:device", "device:tap", "device:type", "device:tap:down", "device:shortcut"],
    ["obs:device", "device:tap", "device:swipe:up", "device:tap", "device:type"],
    ["obs:browser", "act:navigate", "act:click", "act:type", "act:submit"],
    ["obs:device", "device:scroll:down", "device:tap", "device:type", "device:shortcut"],
  ];
  const sequences: MovementSequence[] = [];
  for (let i = 0; i < count; i += 1) {
    const flow = flows[(i + seed) % flows.length]!;
    // Deterministically vary length so held-out prefixes differ from training.
    const length = 3 + ((i + seed) % (flow.length - 2));
    sequences.push({ id: `synthetic-${seed}-${i}`, tokens: flow.slice(0, length) });
  }
  return sequences;
}

function record(level: CountMap, context: MovementToken[], next: MovementToken): void {
  const key = context.join(CONTEXT_SEPARATOR);
  let counts = level.get(key);
  if (!counts) {
    counts = new Map();
    level.set(key, counts);
  }
  counts.set(next, (counts.get(next) ?? 0) + 1);
}
