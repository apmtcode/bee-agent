import { readJsonFile, writeJsonAtomic } from "../shared/fs.js";
import type { ReplayManifest } from "../capture/replay.js";
import type { TrajectorySpan } from "../capture/trajectory.js";

/**
 * In-cloud movement-prediction model (objective #2, pieces c+d).
 *
 * The training runner ({@link LocalAppleSiliconTrainingRunner}) only emits
 * *plans* for real on-device runtimes (mlx / axolotl) — nothing in bee-agent
 * could actually learn from recorded movements and predict the next one. This
 * module closes that loop with a small, deterministic, fully in-cloud model so
 * the capture -> dataset -> replay -> train/infer pipeline can be validated end
 * to end without real OS input or GPU training.
 *
 * The model backend is pluggable ({@link MovementModelBackend}) so the
 * deterministic n-gram backend used in cloud/CI can be swapped for a real
 * on-device small model later without touching call sites.
 */

/** A single recorded movement step — one captured action (the unit we predict). */
export interface MovementStep {
  /** The tool / movement kind, e.g. "mouse.click", "keyboard.type", "window.focus". */
  tool: string;
  /** Human-/machine-readable detail for the step (target, text, coordinates…). */
  summary: string;
}

/** An ordered sequence of movement steps (one recorded trajectory). */
export type MovementSequence = MovementStep[];

/** Distribution entry over candidate next tools. */
export interface MovementDistributionEntry {
  tool: string;
  probability: number;
}

/** A single next-step prediction from a backend. */
export interface MovementPrediction {
  /** Predicted next step, or null when the model has no data to predict from. */
  step: MovementStep | null;
  /** Confidence in [0,1] — probability mass assigned to the predicted tool. */
  confidence: number;
  /** Full sorted distribution over candidate next tools. */
  distribution: MovementDistributionEntry[];
  /**
   * Context length (number of preceding steps) that actually produced the
   * prediction after back-off. A value < the requested order means the model
   * generalized from a shorter, related context rather than an exact match.
   */
  backoffOrder: number;
}

/** Serializable base state shared by every backend. */
export interface MovementModelState {
  readonly backend: string;
  readonly trainedSequences: number;
  readonly trainedSteps: number;
  readonly vocabulary: string[];
}

/**
 * Pluggable model backend. The deterministic n-gram backend below is used for
 * cloud/CI; a real on-device small-model backend can implement this same seam.
 */
export interface MovementModelBackend<TState extends MovementModelState = MovementModelState> {
  readonly name: string;
  /** Fit the model to a set of recorded sequences. Pure: no side effects. */
  train(sequences: MovementSequence[]): TState;
  /** Predict the next step given the preceding context. */
  predict(context: MovementStep[], state: TState): MovementPrediction;
  /** Convert state to a plain JSON-safe object for persistence. */
  serialize(state: TState): unknown;
  /** Reconstruct state from a previously serialized object. */
  deserialize(data: unknown): TState;
}

const START_TOKEN = "START";
const END_TOKEN = "END";

/** Persisted shape of the n-gram backend. */
export interface NGramMovementState extends MovementModelState {
  readonly backend: "ngram";
  readonly order: number;
  /** context-key -> next-tool -> count (all back-off orders 0..order). */
  readonly transitions: Record<string, Record<string, number>>;
  /** tool -> summary -> count, used to pick a representative replayable summary. */
  readonly summaries: Record<string, Record<string, number>>;
}

/**
 * Deterministic n-gram movement model with "stupid back-off".
 *
 * Learns tool-to-tool transition frequencies at every context length from
 * `order` down to 0 (unigram). At prediction time it uses the longest matching
 * context and, when that exact context was never observed, backs off to a
 * shorter — but still related — context. That back-off is precisely what lets
 * the model perform *new but related* movements rather than only exact replays.
 */
export class NGramMovementBackend implements MovementModelBackend<NGramMovementState> {
  readonly name = "ngram";

  constructor(private readonly order = 2) {
    if (!Number.isInteger(order) || order < 1) {
      throw new Error(`NGramMovementBackend order must be a positive integer, got ${order}`);
    }
  }

  train(sequences: MovementSequence[]): NGramMovementState {
    const transitions: Record<string, Record<string, number>> = {};
    const summaries: Record<string, Record<string, number>> = {};
    const vocabulary = new Set<string>();
    let trainedSteps = 0;

    for (const sequence of sequences) {
      const tokens = [START_TOKEN, ...sequence.map((step) => step.tool), END_TOKEN];
      for (const step of sequence) {
        vocabulary.add(step.tool);
        const bucket = (summaries[step.tool] ??= {});
        bucket[step.summary] = (bucket[step.summary] ?? 0) + 1;
        trainedSteps += 1;
      }
      // Record transitions for every back-off order 0..order.
      for (let i = 1; i < tokens.length; i += 1) {
        const next = tokens[i] as string;
        for (let k = 0; k <= this.order; k += 1) {
          if (i - k < 0) {
            break;
          }
          const context = tokens.slice(i - k, i);
          const key = contextKey(context);
          const row = (transitions[key] ??= {});
          row[next] = (row[next] ?? 0) + 1;
        }
      }
    }

    return {
      backend: "ngram",
      order: this.order,
      trainedSequences: sequences.length,
      trainedSteps,
      vocabulary: [...vocabulary].sort(),
      transitions,
      summaries,
    };
  }

  predict(context: MovementStep[], state: NGramMovementState): MovementPrediction {
    const tokens = [START_TOKEN, ...context.map((step) => step.tool)];
    const maxOrder = Math.min(state.order, tokens.length);

    for (let k = maxOrder; k >= 0; k -= 1) {
      const key = contextKey(tokens.slice(tokens.length - k));
      const row = state.transitions[key];
      if (!row) {
        continue;
      }
      const distribution = toDistribution(row);
      if (distribution.length === 0) {
        continue;
      }
      const top = distribution[0] as MovementDistributionEntry;
      if (top.tool === END_TOKEN) {
        // The most likely continuation is "stop" — surface that as a null step
        // but still report the distribution and confidence.
        return { step: null, confidence: top.probability, distribution, backoffOrder: k };
      }
      return {
        step: { tool: top.tool, summary: representativeSummary(state, top.tool) },
        confidence: top.probability,
        distribution,
        backoffOrder: k,
      };
    }

    return { step: null, confidence: 0, distribution: [], backoffOrder: 0 };
  }

  serialize(state: NGramMovementState): unknown {
    return state;
  }

  deserialize(data: unknown): NGramMovementState {
    const state = data as Partial<NGramMovementState> | null;
    if (!state || state.backend !== "ngram" || typeof state.order !== "number") {
      throw new Error("invalid serialized NGramMovementState");
    }
    return {
      backend: "ngram",
      order: state.order,
      trainedSequences: state.trainedSequences ?? 0,
      trainedSteps: state.trainedSteps ?? 0,
      vocabulary: state.vocabulary ? [...state.vocabulary] : [],
      transitions: state.transitions ?? {},
      summaries: state.summaries ?? {},
    };
  }
}

function contextKey(tokens: string[]): string {
  return tokens.join("");
}

function toDistribution(row: Record<string, number>): MovementDistributionEntry[] {
  const total = Object.values(row).reduce((sum, count) => sum + count, 0);
  if (total === 0) {
    return [];
  }
  return Object.entries(row)
    .map(([tool, count]) => ({ tool, probability: count / total }))
    .sort((a, b) => (b.probability !== a.probability ? b.probability - a.probability : a.tool.localeCompare(b.tool)));
}

function representativeSummary(state: NGramMovementState, tool: string): string {
  const bucket = state.summaries[tool];
  if (!bucket) {
    return "";
  }
  let bestSummary = "";
  let bestCount = -1;
  for (const [summary, count] of Object.entries(bucket)) {
    if (count > bestCount || (count === bestCount && summary.localeCompare(bestSummary) < 0)) {
      bestSummary = summary;
      bestCount = count;
    }
  }
  return bestSummary;
}

/** Result of generating (rolling out) a movement sequence from a trained model. */
export interface MovementRollout {
  steps: MovementStep[];
  /** True if generation stopped because the model predicted END, not the cap. */
  stoppedNaturally: boolean;
  /** Average confidence across the generated steps (0 if none). */
  meanConfidence: number;
}

/** Per-sequence and aggregate next-step accuracy for a held-out eval set. */
export interface MovementEvalResult {
  sequences: number;
  predictions: number;
  correct: number;
  /** Top-1 next-tool accuracy in [0,1]. */
  accuracy: number;
  /**
   * Mean back-off context length used across *correct* predictions. A lower
   * value means the model leaned on shorter, shared structure to get the answer
   * right — i.e. it generalized rather than matched a long memorized context.
   * NaN when there are no correct predictions.
   */
  meanBackoffOrder: number;
  perSequence: Array<{ index: number; predictions: number; correct: number; accuracy: number }>;
}

export interface MovementModelOptions<TState extends MovementModelState = MovementModelState> {
  backend?: MovementModelBackend<TState>;
}

/**
 * Orchestrates the pluggable backend: builds sequences from recorded
 * trajectories / replay manifests, trains, predicts, rolls out full movement
 * sequences, evaluates generalization, and persists the model.
 */
export class MovementModel<TState extends MovementModelState = MovementModelState> {
  private readonly backend: MovementModelBackend<TState>;
  private state: TState | undefined;

  constructor(options: MovementModelOptions<TState> = {}) {
    this.backend =
      options.backend ?? (new NGramMovementBackend() as unknown as MovementModelBackend<TState>);
  }

  get backendName(): string {
    return this.backend.name;
  }

  get trained(): boolean {
    return this.state !== undefined;
  }

  getState(): TState {
    if (!this.state) {
      throw new Error("MovementModel has not been trained yet");
    }
    return this.state;
  }

  train(sequences: MovementSequence[]): TState {
    this.state = this.backend.train(sequences);
    return this.state;
  }

  predictNext(context: MovementStep[]): MovementPrediction {
    return this.backend.predict(context, this.getState());
  }

  /**
   * Greedily generate a movement sequence from an optional seed context. Used
   * to "repeat the recorded movements" and to perform new-but-related ones when
   * seeded with an unseen-but-similar prefix (back-off kicks in).
   */
  generate(options: { seed?: MovementStep[]; maxSteps?: number } = {}): MovementRollout {
    const maxSteps = options.maxSteps ?? 64;
    const context: MovementStep[] = [...(options.seed ?? [])];
    const steps: MovementStep[] = [];
    let confidenceSum = 0;
    let stoppedNaturally = false;

    for (let i = 0; i < maxSteps; i += 1) {
      const prediction = this.predictNext(context);
      if (!prediction.step) {
        stoppedNaturally = prediction.distribution.length > 0;
        break;
      }
      steps.push(prediction.step);
      confidenceSum += prediction.confidence;
      context.push(prediction.step);
    }

    return {
      steps,
      stoppedNaturally,
      meanConfidence: steps.length > 0 ? confidenceSum / steps.length : 0,
    };
  }

  /** Measure top-1 next-tool accuracy on held-out sequences (generalization). */
  evaluate(heldOut: MovementSequence[]): MovementEvalResult {
    const state = this.getState();
    let predictions = 0;
    let correct = 0;
    let backoffSum = 0;
    const perSequence: MovementEvalResult["perSequence"] = [];

    heldOut.forEach((sequence, index) => {
      let seqPredictions = 0;
      let seqCorrect = 0;
      for (let i = 0; i < sequence.length; i += 1) {
        const context = sequence.slice(0, i);
        const prediction = this.backend.predict(context, state);
        seqPredictions += 1;
        predictions += 1;
        const expected = sequence[i] as MovementStep;
        if (prediction.step?.tool === expected.tool) {
          seqCorrect += 1;
          correct += 1;
          backoffSum += prediction.backoffOrder;
        }
      }
      perSequence.push({
        index,
        predictions: seqPredictions,
        correct: seqCorrect,
        accuracy: seqPredictions > 0 ? seqCorrect / seqPredictions : 0,
      });
    });

    return {
      sequences: heldOut.length,
      predictions,
      correct,
      accuracy: predictions > 0 ? correct / predictions : 0,
      meanBackoffOrder: correct > 0 ? backoffSum / correct : Number.NaN,
      perSequence,
    };
  }

  async save(filePath: string): Promise<void> {
    await writeJsonAtomic(filePath, {
      version: 1,
      backend: this.backend.name,
      state: this.backend.serialize(this.getState()),
    });
  }

  async load(filePath: string): Promise<TState> {
    const raw = await readJsonFile<{ backend?: string; state?: unknown } | undefined>(filePath, undefined);
    if (!raw || raw.backend !== this.backend.name) {
      throw new Error(`no saved ${this.backend.name} movement model at ${filePath}`);
    }
    this.state = this.backend.deserialize(raw.state);
    return this.state;
  }
}

/** Build an ordered movement sequence from a recorded trajectory's actions. */
export function sequenceFromTrajectory(trajectory: TrajectorySpan): MovementSequence {
  return [...trajectory.actions]
    .sort((a, b) => a.ts - b.ts)
    .map((action) => ({ tool: action.tool, summary: action.summary }));
}

/** Build movement sequences from many trajectories (drops empty sequences). */
export function sequencesFromTrajectories(trajectories: TrajectorySpan[]): MovementSequence[] {
  return trajectories.map(sequenceFromTrajectory).filter((sequence) => sequence.length > 0);
}

/** Build a movement sequence from a replay manifest's action timeline events. */
export function sequenceFromReplayManifest(manifest: ReplayManifest): MovementSequence {
  return manifest.events
    .filter((event): event is Extract<ReplayManifest["events"][number], { kind: "action" }> => event.kind === "action")
    .sort((a, b) => a.ts - b.ts)
    .map((event) => ({ tool: event.tool, summary: event.summary }));
}
