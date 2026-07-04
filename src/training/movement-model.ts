import type { ReplayManifest, ReplayTimelineEvent } from "../capture/replay.js";
import type { ExportedReplayManifest, ReviewedExportManifest } from "./export-manifest.js";
import type { TrajectorySpan } from "../capture/trajectory.js";

/**
 * Local-movement learning: a pluggable, on-device model backend that learns to
 * repeat recorded movement sequences and generalize to new-but-related ones.
 *
 * This module is the in-process, cloud-runnable counterpart to
 * {@link ../training/runner.ts LocalAppleSiliconTrainingRunner}: the runner
 * emits a shell plan to launch a heavy external trainer (mlx/axolotl) on the
 * user's real machine, whereas this module trains and infers deterministically
 * inside the agent so the pipeline can be validated with synthetic event
 * streams in CI. The backend is pluggable so a real small on-device model can
 * be dropped in behind the same interface.
 */

/** A single normalized movement — one action or observation on the timeline. */
export type MovementEvent = {
  kind: "action" | "observation";
  /** Coarse channel: the tool for actions, the source for observations. */
  channel: string;
  summary: string;
  ts: number;
};

/** An ordered movement sequence for one trajectory/session. */
export type MovementSequence = {
  id: string;
  tokens: string[];
  events: MovementEvent[];
};

/** A replayable dataset of movement sequences with a shared vocabulary. */
export type MovementDataset = {
  version: 1;
  sequences: MovementSequence[];
  vocabulary: string[];
};

/**
 * Framing sentinels. Real tokens are always `${kind}:${channel}` (see
 * {@link movementToken}), so a leading "#" cannot collide with any real token.
 */
export const MOVEMENT_START_TOKEN = "#start";
export const MOVEMENT_STOP_TOKEN = "#stop";

/** Frame a raw token list with the start/stop sentinels used during training. */
function frameSequence(tokens: readonly string[]): string[] {
  return [MOVEMENT_START_TOKEN, ...tokens, MOVEMENT_STOP_TOKEN];
}

/**
 * Derive the discrete token for a movement event. Coarse by design (channel,
 * not free-text summary) so the model generalizes across paraphrased summaries
 * and unseen-but-related sequences rather than memorizing exact strings.
 */
export function movementToken(event: MovementEvent): string {
  return `${event.kind}:${event.channel}`;
}

function toMovementEvent(event: ReplayTimelineEvent): MovementEvent | undefined {
  switch (event.kind) {
    case "action":
      return { kind: "action", channel: event.tool, summary: event.summary, ts: event.ts };
    case "observation":
      return { kind: "observation", channel: event.source, summary: event.summary, ts: event.ts };
    case "transcript":
      // Transcript turns are context, not movements; excluded from the dataset.
      return undefined;
  }
}

/** Build one movement sequence from an ordered list of replay timeline events. */
export function buildMovementSequence(id: string, events: readonly ReplayTimelineEvent[]): MovementSequence {
  const movements = events
    .map(toMovementEvent)
    .filter((event): event is MovementEvent => event !== undefined)
    .sort((a, b) => a.ts - b.ts);
  return {
    id,
    events: movements,
    tokens: movements.map(movementToken),
  };
}

function assembleDataset(sequences: MovementSequence[]): MovementDataset {
  const vocabulary = new Set<string>();
  for (const sequence of sequences) {
    for (const token of sequence.tokens) {
      vocabulary.add(token);
    }
  }
  return {
    version: 1,
    sequences,
    vocabulary: [...vocabulary].sort(),
  };
}

/** Build a dataset from replay manifests (in-memory or exported form). */
export function buildMovementDatasetFromReplays(
  replays: ReadonlyArray<ReplayManifest | ExportedReplayManifest>,
): MovementDataset {
  const sequences = replays.map((replay, index) =>
    buildMovementSequence(replay.sessionId || `replay-${index}`, replay.events),
  );
  return assembleDataset(sequences.filter((sequence) => sequence.tokens.length > 0));
}

/** Build a dataset directly from trajectory spans (observations + actions). */
export function buildMovementDatasetFromTrajectories(
  trajectories: ReadonlyArray<TrajectorySpan>,
): MovementDataset {
  const sequences = trajectories.map((trajectory) => {
    const events: MovementEvent[] = [
      ...trajectory.observations.map<MovementEvent>((observation) => ({
        kind: "observation",
        channel: observation.source,
        summary: observation.summary,
        ts: observation.ts,
      })),
      ...trajectory.actions.map<MovementEvent>((action) => ({
        kind: "action",
        channel: action.tool,
        summary: action.summary,
        ts: action.ts,
      })),
    ].sort((a, b) => a.ts - b.ts);
    return {
      id: trajectory.id,
      events,
      tokens: events.map(movementToken),
    };
  });
  return assembleDataset(sequences.filter((sequence) => sequence.tokens.length > 0));
}

/** Build a dataset from a reviewed training export manifest. */
export function buildMovementDatasetFromExport(manifest: ReviewedExportManifest): MovementDataset {
  return buildMovementDatasetFromReplays(manifest.replays);
}

export type MovementTrainingConfig = {
  /** Highest Markov context order to learn. Higher = more literal replay. */
  maxOrder?: number;
};

export type MovementCandidate = {
  token: string;
  probability: number;
  count: number;
};

export type MovementPrediction = {
  /** Argmax next token, or undefined when the model has seen nothing. */
  token: string | undefined;
  probability: number;
  /** Context order actually used after backoff (0 = unigram). */
  order: number;
  candidates: MovementCandidate[];
};

/**
 * Pluggable backend contract. Swap {@link DeterministicMarkovMovementBackend}
 * for a real on-device small-model backend (e.g. an MLX policy) without
 * touching dataset/rollout/eval code.
 */
export interface MovementModelBackend<Model = unknown> {
  readonly name: string;
  train(dataset: MovementDataset, config?: MovementTrainingConfig): Model;
  predictNext(model: Model, context: readonly string[]): MovementPrediction;
  serialize(model: Model): string;
  deserialize(data: string): Model;
}

const DEFAULT_MAX_ORDER = 3;

/** Unambiguous context key — JSON encodes any token contents without collisions. */
function contextKey(context: readonly string[]): string {
  return JSON.stringify(context);
}

export type MarkovMovementModel = {
  version: 1;
  backend: "deterministic-markov";
  maxOrder: number;
  vocabulary: string[];
  sequenceCount: number;
  tokenCount: number;
  /** JSON-encoded context -> token -> count. Order 0 key is "[]". */
  transitions: Record<string, Record<string, number>>;
};

/**
 * A deterministic variable-order Markov chain with stupid-backoff. It memorizes
 * transitions up to `maxOrder`, so it replays recorded movements faithfully;
 * when an exact high-order context is unseen it backs off to shorter contexts,
 * which is what lets it generalize to new-but-related sequences. Fully
 * deterministic (no randomness), so training and inference are reproducible in
 * CI and the model round-trips through JSON.
 */
export class DeterministicMarkovMovementBackend implements MovementModelBackend<MarkovMovementModel> {
  readonly name = "deterministic-markov";

  train(dataset: MovementDataset, config?: MovementTrainingConfig): MarkovMovementModel {
    const maxOrder = Math.max(0, config?.maxOrder ?? DEFAULT_MAX_ORDER);
    const transitions: Record<string, Record<string, number>> = {};
    let tokenCount = 0;

    const record = (context: string[], token: string): void => {
      const key = contextKey(context);
      const bucket = (transitions[key] ??= {});
      bucket[token] = (bucket[token] ?? 0) + 1;
    };

    for (const sequence of dataset.sequences) {
      const framed = frameSequence(sequence.tokens);
      tokenCount += sequence.tokens.length;
      // Start at 1: the START sentinel is only ever context, never a target.
      for (let i = 1; i < framed.length; i += 1) {
        const token = framed[i]!;
        for (let order = 0; order <= maxOrder && order <= i; order += 1) {
          record(framed.slice(i - order, i), token);
        }
      }
    }

    return {
      version: 1,
      backend: "deterministic-markov",
      maxOrder,
      vocabulary: [...dataset.vocabulary],
      sequenceCount: dataset.sequences.length,
      tokenCount,
      transitions,
    };
  }

  predictNext(model: MarkovMovementModel, context: readonly string[]): MovementPrediction {
    for (let order = Math.min(model.maxOrder, context.length); order >= 0; order -= 1) {
      const key = contextKey(context.slice(context.length - order));
      const bucket = model.transitions[key];
      if (!bucket) {
        continue;
      }
      const candidates = rankCandidates(bucket);
      if (candidates.length === 0) {
        continue;
      }
      return {
        token: candidates[0]!.token,
        probability: candidates[0]!.probability,
        order,
        candidates,
      };
    }
    return { token: undefined, probability: 0, order: 0, candidates: [] };
  }

  serialize(model: MarkovMovementModel): string {
    return JSON.stringify(model);
  }

  deserialize(data: string): MarkovMovementModel {
    const parsed = JSON.parse(data) as MarkovMovementModel;
    if (parsed.backend !== "deterministic-markov") {
      throw new Error(`unexpected movement model backend: ${String(parsed.backend)}`);
    }
    return parsed;
  }
}

function rankCandidates(bucket: Record<string, number>): MovementCandidate[] {
  const total = Object.values(bucket).reduce((sum, count) => sum + count, 0);
  if (total === 0) {
    return [];
  }
  return Object.entries(bucket)
    .map(([token, count]) => ({ token, count, probability: count / total }))
    // Deterministic tie-break: higher count first, then lexicographic token.
    .sort((a, b) => (b.count !== a.count ? b.count - a.count : a.token < b.token ? -1 : 1));
}

export type MovementRolloutStep = {
  token: string;
  probability: number;
  order: number;
};

export type MovementRolloutParams = {
  /**
   * Seed prefix to continue from. Defaults to the start sentinel, so an empty
   * call generates a full sequence from scratch.
   */
  seed?: readonly string[];
  /** Hard cap on generated tokens (excluding the seed). */
  maxSteps: number;
  /** Stop when this token is produced. Defaults to the training stop sentinel. */
  stopToken?: string;
};

/**
 * Replay engine: deterministically roll out a movement sequence from the model.
 * With a seed drawn from a recorded trajectory it repeats that movement; with a
 * partial/related seed it generalizes via backoff. The framing sentinels are
 * never included in the returned tokens.
 */
export function rolloutMovements<Model>(
  backend: MovementModelBackend<Model>,
  model: Model,
  params: MovementRolloutParams,
): MovementRolloutStep[] {
  const stopToken = params.stopToken ?? MOVEMENT_STOP_TOKEN;
  const context = params.seed ? [...params.seed] : [MOVEMENT_START_TOKEN];
  const steps: MovementRolloutStep[] = [];
  for (let i = 0; i < params.maxSteps; i += 1) {
    const prediction = backend.predictNext(model, context);
    if (
      prediction.token === undefined ||
      prediction.token === stopToken ||
      prediction.token === MOVEMENT_START_TOKEN
    ) {
      break;
    }
    steps.push({ token: prediction.token, probability: prediction.probability, order: prediction.order });
    context.push(prediction.token);
  }
  return steps;
}

export type MovementSequenceEval = {
  id: string;
  predictions: number;
  correct: number;
  accuracy: number;
};

export type MovementEvalReport = {
  sequenceCount: number;
  predictions: number;
  correct: number;
  /** Next-token accuracy across every position in every held-out sequence. */
  accuracy: number;
  perSequence: MovementSequenceEval[];
};

/**
 * Teacher-forced next-token accuracy on held-out sequences — the generalization
 * metric. Each real token (and the terminal stop) is scored against the model's
 * prediction given the true framed prefix. The START sentinel is the given seed
 * and is not itself scored.
 */
export function evaluateMovementModel<Model>(
  backend: MovementModelBackend<Model>,
  model: Model,
  heldOut: ReadonlyArray<MovementSequence>,
): MovementEvalReport {
  const perSequence: MovementSequenceEval[] = [];
  let totalPredictions = 0;
  let totalCorrect = 0;

  for (const sequence of heldOut) {
    const framed = frameSequence(sequence.tokens);
    let predictions = 0;
    let correct = 0;
    for (let i = 1; i < framed.length; i += 1) {
      const prediction = backend.predictNext(model, framed.slice(0, i));
      predictions += 1;
      if (prediction.token === framed[i]) {
        correct += 1;
      }
    }
    totalPredictions += predictions;
    totalCorrect += correct;
    perSequence.push({
      id: sequence.id,
      predictions,
      correct,
      accuracy: predictions === 0 ? 0 : correct / predictions,
    });
  }

  return {
    sequenceCount: heldOut.length,
    predictions: totalPredictions,
    correct: totalCorrect,
    accuracy: totalPredictions === 0 ? 0 : totalCorrect / totalPredictions,
    perSequence,
  };
}

export type MovementTrainingReport = {
  backend: string;
  trainedAt?: string;
  sequenceCount: number;
  tokenCount: number;
  vocabularySize: number;
  maxOrder: number;
  /** In-sample fidelity: accuracy replaying the training sequences. */
  trainFidelity: MovementEvalReport;
  /** Optional held-out generalization accuracy. */
  generalization?: MovementEvalReport;
};

/**
 * End-to-end in-process pipeline: train a movement model on a dataset, measure
 * replay fidelity (and optional held-out generalization), and expose the model
 * plus a serialized snapshot. Backend is injectable; defaults to the
 * deterministic Markov backend so it runs in the cloud with no ML deps.
 */
export class MovementModelTrainingService<Model = MarkovMovementModel> {
  private readonly backend: MovementModelBackend<Model>;

  constructor(backend?: MovementModelBackend<Model>) {
    this.backend = backend ?? (new DeterministicMarkovMovementBackend() as unknown as MovementModelBackend<Model>);
  }

  trainAndEvaluate(params: {
    dataset: MovementDataset;
    heldOut?: ReadonlyArray<MovementSequence>;
    config?: MovementTrainingConfig;
    trainedAt?: string;
  }): { model: Model; serialized: string; report: MovementTrainingReport } {
    const model = this.backend.train(params.dataset, params.config);
    const trainFidelity = evaluateMovementModel(this.backend, model, params.dataset.sequences);
    const generalization = params.heldOut
      ? evaluateMovementModel(this.backend, model, params.heldOut)
      : undefined;
    const report: MovementTrainingReport = {
      backend: this.backend.name,
      ...(params.trainedAt ? { trainedAt: params.trainedAt } : {}),
      sequenceCount: params.dataset.sequences.length,
      tokenCount: params.dataset.sequences.reduce((sum, sequence) => sum + sequence.tokens.length, 0),
      vocabularySize: params.dataset.vocabulary.length,
      maxOrder: params.config?.maxOrder ?? DEFAULT_MAX_ORDER,
      trainFidelity,
      ...(generalization ? { generalization } : {}),
    };
    return { model, serialized: this.backend.serialize(model), report };
  }

  rollout(model: Model, params: MovementRolloutParams): MovementRolloutStep[] {
    return rolloutMovements(this.backend, model, params);
  }

  predictNext(model: Model, context: readonly string[]): MovementPrediction {
    return this.backend.predictNext(model, context);
  }
}
