import type { ExportedReplayManifest, ReviewedExportManifest } from "./export-manifest.js";

/**
 * Pluggable local-model backend for the movement-learning subsystem.
 *
 * Objective #2 pieces (d) + (e): post-train a local model to *repeat* recorded
 * movements, and *generalize* to new-but-related movements. On the user's
 * machine this is served by an on-device runtime (MLX / axolotl — see
 * `runner.ts`). In the cloud we cannot run those, so this module provides:
 *
 *   1. A backend-agnostic {@link MovementModelBackend} interface.
 *   2. A deterministic {@link DeterministicMarkovBackend} that trains a
 *      context-conditioned transition model with backoff smoothing — small,
 *      dependency-free, fully reproducible, so capture→dataset→train→infer
 *      round-trips are validated in CI without real OS input or GPUs.
 *   3. Dataset adapters from the reviewed-export replay stream, an
 *      autoregressive replay/rollout helper, and a generalization eval harness.
 *
 * A real on-device backend implements the same interface and serialises to the
 * same {@link SerializedMovementModel} schema, so training location is a swap,
 * not a rewrite.
 */

export type MovementEventKind = "observation" | "action";

/** A single normalised movement event (mouse/keyboard/UI observation or tool action). */
export type MovementEvent = {
  kind: MovementEventKind;
  /** action tool for actions, observation source for observations. */
  channel: string;
  summary: string;
  ts: number;
};

export type MovementSequence = {
  id: string;
  events: MovementEvent[];
};

export type MovementDataset = {
  sequences: MovementSequence[];
};

/** Recent events (most-recent last) leading up to the action we want to predict. */
export type MovementContext = {
  recent: ReadonlyArray<MovementEvent>;
};

export type PredictionSource =
  | "exact-context"
  | "backoff-action"
  | "backoff-observation"
  | "unigram"
  | "empty";

export type MovementPrediction = {
  /** predicted next action tool (empty string when the model has nothing to offer). */
  tool: string;
  summary: string;
  /** 0..1 — max/sum of the distribution the prediction was drawn from. */
  confidence: number;
  /** true when produced by a backoff level, i.e. a novel-but-related context. */
  generalized: boolean;
  source: PredictionSource;
};

export type SerializedMovementModel = {
  version: 1;
  backend: string;
  /** `${lastAction}»${lastObservation}` -> nextTool -> count */
  contextTransitions: Record<string, Record<string, number>>;
  /** lastAction -> nextTool -> count */
  actionTransitions: Record<string, Record<string, number>>;
  /** lastObservation -> nextTool -> count */
  observationTransitions: Record<string, Record<string, number>>;
  /** nextTool -> count */
  unigram: Record<string, number>;
  /** tool -> summary -> count */
  summaries: Record<string, Record<string, number>>;
  sequenceCount: number;
  eventCount: number;
};

export interface TrainedMovementModel {
  readonly backend: string;
  predict(context: MovementContext): MovementPrediction;
  serialize(): SerializedMovementModel;
}

export type TrainOptions = {
  /**
   * How many preceding actions are folded into the composite context key.
   * Default 1 (last action + last observation). Larger windows capture longer
   * motifs at the cost of sparser statistics (mitigated by backoff).
   */
  contextWindow?: number;
};

export interface MovementModelBackend {
  readonly name: string;
  train(dataset: MovementDataset, options?: TrainOptions): Promise<TrainedMovementModel>;
}

const NONE = "∅"; // ∅ — sentinel for "no preceding event of this kind".
const CONTEXT_SEP = "»"; // »

// --- dataset adapters ---------------------------------------------------------

/** Build a movement dataset from reviewed-export replay manifests. */
export function datasetFromReplays(replays: ReadonlyArray<ExportedReplayManifest>): MovementDataset {
  const sequences = replays.map<MovementSequence>((replay) => {
    const events = replay.events.flatMap<MovementEvent>((event) => {
      if (event.kind === "action") {
        return [{ kind: "action", channel: event.tool, summary: event.summary, ts: event.ts }];
      }
      if (event.kind === "observation") {
        return [{ kind: "observation", channel: event.source, summary: event.summary, ts: event.ts }];
      }
      return [];
    });
    events.sort((a, b) => a.ts - b.ts);
    return { id: replay.sessionId, events };
  });
  return { sequences };
}

/** Build a movement dataset directly from a reviewed training export. */
export function datasetFromExport(manifest: ReviewedExportManifest): MovementDataset {
  return datasetFromReplays(manifest.replays);
}

// --- deterministic mock backend ----------------------------------------------

function increment(table: Record<string, Record<string, number>>, key: string, value: string): void {
  const row = (table[key] ??= {});
  row[value] = (row[value] ?? 0) + 1;
}

/** Deterministic argmax over a count distribution: highest count, ties broken lexicographically. */
function argmax(distribution: Record<string, number> | undefined): { value: string; confidence: number } | undefined {
  if (!distribution) {
    return undefined;
  }
  const entries = Object.entries(distribution);
  if (entries.length === 0) {
    return undefined;
  }
  let total = 0;
  for (const [, count] of entries) {
    total += count;
  }
  entries.sort((a, b) => (b[1] - a[1]) || (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0));
  const [value, count] = entries[0]!;
  return { value, confidence: total === 0 ? 0 : count / total };
}

function lastChannel(recent: ReadonlyArray<MovementEvent>, kind: MovementEventKind): string {
  for (let i = recent.length - 1; i >= 0; i--) {
    const event = recent[i]!;
    if (event.kind === kind) {
      return event.channel;
    }
  }
  return NONE;
}

function contextKey(recent: ReadonlyArray<MovementEvent>, contextWindow: number): string {
  const actions: string[] = [];
  for (let i = recent.length - 1; i >= 0 && actions.length < contextWindow; i--) {
    if (recent[i]!.kind === "action") {
      actions.unshift(recent[i]!.channel);
    }
  }
  while (actions.length < contextWindow) {
    actions.unshift(NONE);
  }
  return `${actions.join(">")}${CONTEXT_SEP}${lastChannel(recent, "observation")}`;
}

class MarkovMovementModel implements TrainedMovementModel {
  constructor(
    readonly backend: string,
    private readonly model: SerializedMovementModel,
    private readonly contextWindow: number,
  ) {}

  predict(context: MovementContext): MovementPrediction {
    const recent = context.recent;
    const key = contextKey(recent, this.contextWindow);
    const lastAction = lastChannel(recent, "action");
    const lastObservation = lastChannel(recent, "observation");

    const exact = argmax(this.model.contextTransitions[key]);
    if (exact) {
      return this.decorate(exact, false, "exact-context");
    }
    const byAction = argmax(this.model.actionTransitions[lastAction]);
    if (byAction) {
      return this.decorate(byAction, true, "backoff-action");
    }
    const byObservation = argmax(this.model.observationTransitions[lastObservation]);
    if (byObservation) {
      return this.decorate(byObservation, true, "backoff-observation");
    }
    const unigram = argmax(this.model.unigram);
    if (unigram) {
      return this.decorate(unigram, true, "unigram");
    }
    return { tool: "", summary: "", confidence: 0, generalized: true, source: "empty" };
  }

  serialize(): SerializedMovementModel {
    return structuredCloneModel(this.model);
  }

  private decorate(
    pick: { value: string; confidence: number },
    generalized: boolean,
    source: PredictionSource,
  ): MovementPrediction {
    const summary = argmax(this.model.summaries[pick.value])?.value ?? "";
    return { tool: pick.value, summary, confidence: pick.confidence, generalized, source };
  }
}

function structuredCloneModel(model: SerializedMovementModel): SerializedMovementModel {
  return JSON.parse(JSON.stringify(model)) as SerializedMovementModel;
}

/**
 * Deterministic, dependency-free movement backend. Learns a context-conditioned
 * next-action transition model (last N actions × last observation) with three
 * backoff levels (action-only → observation-only → unigram). Same dataset +
 * options always yields byte-identical weights and predictions.
 */
export class DeterministicMarkovBackend implements MovementModelBackend {
  readonly name = "deterministic-markov";

  async train(dataset: MovementDataset, options: TrainOptions = {}): Promise<TrainedMovementModel> {
    const contextWindow = Math.max(1, options.contextWindow ?? 1);
    const model: SerializedMovementModel = {
      version: 1,
      backend: this.name,
      contextTransitions: {},
      actionTransitions: {},
      observationTransitions: {},
      unigram: {},
      summaries: {},
      sequenceCount: dataset.sequences.length,
      eventCount: 0,
    };

    for (const sequence of dataset.sequences) {
      const history: MovementEvent[] = [];
      for (const event of sequence.events) {
        model.eventCount += 1;
        if (event.kind === "action") {
          const key = contextKey(history, contextWindow);
          const lastAction = lastChannel(history, "action");
          const lastObservation = lastChannel(history, "observation");
          increment(model.contextTransitions, key, event.channel);
          increment(model.actionTransitions, lastAction, event.channel);
          increment(model.observationTransitions, lastObservation, event.channel);
          model.unigram[event.channel] = (model.unigram[event.channel] ?? 0) + 1;
          increment(model.summaries, event.channel, event.summary);
        }
        history.push(event);
      }
    }

    return new MarkovMovementModel(this.name, model, contextWindow);
  }
}

/** Rehydrate a persisted model (from any backend that serialises to the shared schema) for inference. */
export function loadMovementModel(serialized: SerializedMovementModel): TrainedMovementModel {
  const contextWindow = Math.max(
    1,
    ...Object.keys(serialized.contextTransitions).map((key) => key.split(CONTEXT_SEP)[0]!.split(">").length),
  );
  return new MarkovMovementModel(serialized.backend, structuredCloneModel(serialized), contextWindow);
}

// --- replay / rollout ---------------------------------------------------------

/**
 * Autoregressively roll the model forward from a seed context, feeding each
 * predicted action back in. This is "repeat the recorded movements" (and, when
 * the seed is novel-but-related, generalise them). Stops early if the model has
 * nothing to predict.
 */
export function rolloutMovements(
  model: TrainedMovementModel,
  seed: MovementContext,
  steps: number,
): MovementPrediction[] {
  const recent: MovementEvent[] = [...seed.recent];
  const predictions: MovementPrediction[] = [];
  let ts = recent.length > 0 ? recent[recent.length - 1]!.ts : 0;
  for (let i = 0; i < steps; i++) {
    const prediction = model.predict({ recent });
    if (prediction.source === "empty") {
      break;
    }
    predictions.push(prediction);
    ts += 1;
    recent.push({ kind: "action", channel: prediction.tool, summary: prediction.summary, ts });
  }
  return predictions;
}

// --- evaluation ---------------------------------------------------------------

export type MovementEvalResult = {
  /** number of action events scored. */
  total: number;
  correct: number;
  /** predictions that required backoff (generalisation). */
  generalized: number;
  generalizedCorrect: number;
  /** correct / total. */
  accuracy: number;
  /** generalizedCorrect / generalized (0 when nothing was generalised). */
  generalizedAccuracy: number;
};

/**
 * Next-action prediction accuracy on held-out sequences. Overall accuracy
 * measures replay fidelity; generalizedAccuracy isolates how well the model
 * handles novel-but-related contexts (objective #2 piece (e)).
 */
export function evaluateNextActionAccuracy(
  model: TrainedMovementModel,
  heldOut: MovementDataset,
): MovementEvalResult {
  let total = 0;
  let correct = 0;
  let generalized = 0;
  let generalizedCorrect = 0;

  for (const sequence of heldOut.sequences) {
    const history: MovementEvent[] = [];
    for (const event of sequence.events) {
      if (event.kind === "action") {
        total += 1;
        const prediction = model.predict({ recent: history });
        const hit = prediction.tool === event.channel;
        if (hit) {
          correct += 1;
        }
        if (prediction.generalized) {
          generalized += 1;
          if (hit) {
            generalizedCorrect += 1;
          }
        }
      }
      history.push(event);
    }
  }

  return {
    total,
    correct,
    generalized,
    generalizedCorrect,
    accuracy: total === 0 ? 0 : correct / total,
    generalizedAccuracy: generalized === 0 ? 0 : generalizedCorrect / generalized,
  };
}

/** Convenience: train a model straight from a reviewed export using the given backend. */
export async function trainMovementModelFromExport(
  manifest: ReviewedExportManifest,
  backend: MovementModelBackend,
  options?: TrainOptions,
): Promise<{ dataset: MovementDataset; model: TrainedMovementModel }> {
  const dataset = datasetFromExport(manifest);
  const model = await backend.train(dataset, options);
  return { dataset, model };
}

// --- synthetic event-stream generator ----------------------------------------

export type SyntheticProgram = {
  name: string;
  /** repeating cycle of (observation source, action tool) pairs. */
  steps: Array<{ observation: string; action: string }>;
};

export type SyntheticMovementSpec = {
  seed: number;
  sessions: number;
  stepsPerSession: number;
  programs: SyntheticProgram[];
};

/** Tiny deterministic LCG so synthetic streams are reproducible without `Math.random`. */
function lcg(seed: number): () => number {
  let state = (seed >>> 0) || 1;
  return () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 0x100000000;
  };
}

/**
 * Deterministically synthesise reviewed-export replay manifests that follow a
 * learnable structure — each session samples one program and walks its cycle,
 * emitting an observation then an action per step. Used to validate the
 * capture→dataset→train→infer→replay pipeline without real OS input.
 */
export function generateSyntheticReplays(spec: SyntheticMovementSpec): ExportedReplayManifest[] {
  if (spec.programs.length === 0) {
    return [];
  }
  const random = lcg(spec.seed);
  const replays: ExportedReplayManifest[] = [];
  for (let s = 0; s < spec.sessions; s++) {
    const program = spec.programs[Math.floor(random() * spec.programs.length)] ?? spec.programs[0]!;
    const sessionId = `synthetic-${program.name}-${s}`;
    const trajectoryId = `${sessionId}-traj`;
    const events: ExportedReplayManifest["events"] = [];
    let ts = 0;
    for (let step = 0; step < spec.stepsPerSession; step++) {
      const cycle = program.steps[step % program.steps.length]!;
      events.push({ kind: "observation", ts: ts++, trajectoryId, source: cycle.observation, summary: `observe ${cycle.observation}` });
      events.push({ kind: "action", ts: ts++, trajectoryId, tool: cycle.action, summary: `perform ${cycle.action}` });
    }
    replays.push({ sessionId, trajectoryIds: [trajectoryId], eventCount: events.length, events });
  }
  return replays;
}
