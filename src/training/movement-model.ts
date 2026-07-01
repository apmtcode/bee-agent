import type { TrajectoryAction, TrajectorySpan } from "../capture/trajectory.js";
import type { ExportedReplayManifest } from "./export-manifest.js";

/**
 * Movement-model subsystem (standing objective #2, parts c + d).
 *
 * Turns reviewed movement trajectories into a token dataset, trains a local,
 * in-process model that repeats the recorded movements, and — via context
 * back-off — generalizes to new-but-related movements. The real on-device
 * training (mlx / axolotl) is emitted by `LocalAppleSiliconTrainingRunner`;
 * this module provides a *pluggable backend interface* plus a deterministic
 * backend that trains and infers entirely in-process so the pipeline can be
 * exercised and tested in the cloud with no OS input or GPU.
 */

/** Sentinel tokens framing every training sequence. Real tokens contain "::". */
export const MOVEMENT_START = "<start>";
export const MOVEMENT_END = "<end>";

/** Separator used to join a context window into a single map key. */
const CONTEXT_SEP = "␟";

/** A single normalized movement in a sequence. */
export type MovementStep = {
  /** Stable, normalized token (e.g. `device::tapped submit`). */
  token: string;
  /** Original tool/source that produced the movement. */
  tool: string;
  /** Human-readable summary retained for replay reconstruction. */
  summary: string;
  ts: number;
};

/** An ordered set of movements captured within a single trajectory. */
export type MovementSequence = {
  trajectoryId: string;
  sessionId: string;
  steps: MovementStep[];
};

/** A dataset of movement sequences plus the derived token vocabulary. */
export type MovementDataset = {
  sequences: MovementSequence[];
  vocabulary: string[];
};

export type MovementCandidate = {
  token: string;
  probability: number;
};

export type MovementPrediction = {
  /** Most likely next token, or undefined when the context is unseen. */
  token: string | undefined;
  confidence: number;
  candidates: MovementCandidate[];
  /** Length of the context suffix that actually matched (back-off depth). */
  matchedContextLength: number;
};

export type MovementRolloutOptions = {
  maxSteps: number;
  /** Token that halts the rollout when predicted. Defaults to MOVEMENT_END. */
  stopToken?: string;
};

export type SerializedMovementModel = {
  version: 1;
  backendId: string;
  order: number;
  vocabulary: string[];
  /** context-key -> [token, count][] */
  transitions: Array<[string, Array<[string, number]>]>;
};

/** A trained, queryable movement model. */
export interface TrainedMovementModel {
  readonly backendId: string;
  readonly order: number;
  readonly vocabulary: string[];
  predictNext(context: string[], options?: { topK?: number }): MovementPrediction;
  rollout(seed: string[], options: MovementRolloutOptions): string[];
  serialize(): SerializedMovementModel;
}

export type MovementTrainingOptions = {
  /** Context length (Markov order). Defaults to the backend default. */
  order?: number;
};

/**
 * Pluggable local-model backend. The deterministic backend below trains in
 * process; a real backend would shell out to on-device training and return a
 * model that loads the produced weights. Backends share this contract so the
 * training runner and eval harness are backend-agnostic.
 */
export interface MovementModelBackend {
  readonly id: string;
  train(dataset: MovementDataset, options?: MovementTrainingOptions): Promise<TrainedMovementModel>;
}

/** Normalize a free-form movement summary into a stable token fragment. */
export function slugifyMovementSummary(summary: string): string {
  return summary
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, " ")
    .trim()
    .replace(/\s+/g, " ")
    .slice(0, 64);
}

/** Build a deterministic movement token from an action's tool + summary. */
export function movementToken(tool: string, summary: string): string {
  const slug = slugifyMovementSummary(summary);
  return slug ? `${tool}::${slug}` : `${tool}::`;
}

function toMovementStep(action: Pick<TrajectoryAction, "tool" | "summary" | "ts">): MovementStep {
  return {
    token: movementToken(action.tool, action.summary),
    tool: action.tool,
    summary: action.summary,
    ts: action.ts,
  };
}

/** Build movement sequences from reviewed trajectory spans (action events). */
export function buildMovementSequencesFromTrajectories(trajectories: TrajectorySpan[]): MovementSequence[] {
  return trajectories
    .map((trajectory) => ({
      trajectoryId: trajectory.id,
      sessionId: trajectory.sessionId,
      steps: [...trajectory.actions]
        .sort((a, b) => a.ts - b.ts)
        .map((action) => toMovementStep(action)),
    }))
    .filter((sequence) => sequence.steps.length > 0);
}

/**
 * Build movement sequences from exported replay manifests. Each manifest can
 * span multiple trajectories, so action events are grouped by `trajectoryId`
 * and ordered by timestamp.
 */
export function buildMovementSequencesFromReplays(replays: ExportedReplayManifest[]): MovementSequence[] {
  const sequences: MovementSequence[] = [];
  for (const replay of replays) {
    const byTrajectory = new Map<string, MovementStep[]>();
    for (const event of replay.events) {
      if (event.kind !== "action") {
        continue;
      }
      const steps = byTrajectory.get(event.trajectoryId) ?? [];
      steps.push(toMovementStep(event));
      byTrajectory.set(event.trajectoryId, steps);
    }
    for (const [trajectoryId, steps] of byTrajectory) {
      const ordered = [...steps].sort((a, b) => a.ts - b.ts);
      if (ordered.length > 0) {
        sequences.push({ trajectoryId, sessionId: replay.sessionId, steps: ordered });
      }
    }
  }
  return sequences;
}

/** Assemble a dataset (sequences + sorted, de-duplicated vocabulary). */
export function buildMovementDataset(sequences: MovementSequence[]): MovementDataset {
  const vocabulary = new Set<string>();
  for (const sequence of sequences) {
    for (const step of sequence.steps) {
      vocabulary.add(step.token);
    }
  }
  return { sequences, vocabulary: [...vocabulary].sort() };
}

function joinContext(context: string[]): string {
  return context.join(CONTEXT_SEP);
}

/** Framed token list for a sequence: [START, ...tokens, END]. */
function framedTokens(sequence: MovementSequence): string[] {
  return [MOVEMENT_START, ...sequence.steps.map((step) => step.token), MOVEMENT_END];
}

class MarkovMovementModel implements TrainedMovementModel {
  constructor(
    readonly backendId: string,
    readonly order: number,
    readonly vocabulary: string[],
    private readonly transitions: Map<string, Map<string, number>>,
  ) {}

  predictNext(context: string[], options?: { topK?: number }): MovementPrediction {
    const topK = Math.max(1, options?.topK ?? 3);
    const maxLen = Math.min(this.order, context.length);
    for (let len = maxLen; len >= 0; len -= 1) {
      const suffix = context.slice(context.length - len);
      const distribution = this.transitions.get(joinContext(suffix));
      if (!distribution || distribution.size === 0) {
        continue;
      }
      let total = 0;
      for (const count of distribution.values()) {
        total += count;
      }
      const candidates = [...distribution.entries()]
        .map(([token, count]) => ({ token, probability: count / total }))
        .sort((a, b) => (b.probability !== a.probability ? b.probability - a.probability : a.token < b.token ? -1 : 1));
      const top = candidates[0];
      return {
        token: top?.token,
        confidence: top?.probability ?? 0,
        candidates: candidates.slice(0, topK),
        matchedContextLength: len,
      };
    }
    return { token: undefined, confidence: 0, candidates: [], matchedContextLength: -1 };
  }

  rollout(seed: string[], options: MovementRolloutOptions): string[] {
    const stopToken = options.stopToken ?? MOVEMENT_END;
    const context = seed.length > 0 ? [...seed] : [MOVEMENT_START];
    const produced: string[] = [];
    for (let step = 0; step < options.maxSteps; step += 1) {
      const prediction = this.predictNext(context);
      if (prediction.token === undefined || prediction.token === stopToken) {
        break;
      }
      produced.push(prediction.token);
      context.push(prediction.token);
    }
    return produced;
  }

  serialize(): SerializedMovementModel {
    return {
      version: 1,
      backendId: this.backendId,
      order: this.order,
      vocabulary: [...this.vocabulary],
      transitions: [...this.transitions.entries()]
        .map(([key, distribution]) => [key, [...distribution.entries()]] as [string, Array<[string, number]>])
        .sort((a, b) => (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0)),
    };
  }
}

/** Rehydrate a model previously produced by `serialize()`. */
export function loadMovementModel(serialized: SerializedMovementModel): TrainedMovementModel {
  const transitions = new Map<string, Map<string, number>>();
  for (const [key, entries] of serialized.transitions) {
    transitions.set(key, new Map(entries));
  }
  return new MarkovMovementModel(serialized.backendId, serialized.order, [...serialized.vocabulary], transitions);
}

export const DEFAULT_MOVEMENT_ORDER = 2;

/**
 * Deterministic, in-process backend. Trains a back-off Markov model over the
 * movement tokens: for every position it records the next token against each
 * context suffix from length 0 (unigram) up to `order`. Prediction uses the
 * longest matching suffix and backs off to shorter (shared) contexts, which is
 * what lets it generalize to new-but-related movement prefixes it never saw in
 * full. Fully deterministic — no randomness, no OS or GPU dependency.
 */
export class DeterministicMarkovBackend implements MovementModelBackend {
  readonly id = "deterministic-markov";

  constructor(private readonly defaultOrder: number = DEFAULT_MOVEMENT_ORDER) {}

  async train(dataset: MovementDataset, options?: MovementTrainingOptions): Promise<TrainedMovementModel> {
    const order = Math.max(1, options?.order ?? this.defaultOrder);
    const transitions = new Map<string, Map<string, number>>();
    for (const sequence of dataset.sequences) {
      const tokens = framedTokens(sequence);
      for (let i = 1; i < tokens.length; i += 1) {
        const target = tokens[i];
        if (target === undefined) {
          continue;
        }
        const maxLen = Math.min(order, i);
        for (let len = 0; len <= maxLen; len += 1) {
          const suffix = tokens.slice(i - len, i);
          const key = joinContext(suffix);
          const distribution = transitions.get(key) ?? new Map<string, number>();
          distribution.set(target, (distribution.get(target) ?? 0) + 1);
          transitions.set(key, distribution);
        }
      }
    }
    return new MarkovMovementModel(this.id, order, [...dataset.vocabulary], transitions);
  }
}
