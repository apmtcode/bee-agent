import type { ReplayTimelineEvent } from "../capture/replay.js";
import type { TrajectoryAction, TrajectorySpan } from "../capture/trajectory.js";

/**
 * A single normalized movement/action on the local computer, distilled from a
 * captured {@link TrajectoryAction}. This is the atomic unit the movement model
 * learns to repeat and generalize over (mouse/keyboard/gesture/window events).
 */
export type MovementStep = {
  tool: string;
  gesture?: string;
  target?: string;
  direction?: string;
  summary: string;
};

/** An ordered run of movements belonging to one reviewed trajectory. */
export type MovementSequence = {
  trajectoryId: string;
  steps: MovementStep[];
};

/** The replayable dataset a backend trains on. */
export type MovementDataset = {
  version: 1;
  sequences: MovementSequence[];
};

/** Options controlling how a backend fits a model to the dataset. */
export type TrainMovementModelOptions = {
  /** Markov context length (backoff orders 0..order are learned). Default 2. */
  order?: number;
};

/** The runtime input for inference — the movements observed so far. */
export type MovementContext = {
  prefix: MovementStep[];
  /** Hard cap on generated steps (safety bound). Default 32. */
  maxSteps?: number;
};

export type PredictedMovementStep = MovementStep & {
  /** Empirical probability of this step given the matched context. */
  confidence: number;
  /**
   * `recorded` when the full-order context was seen verbatim during training
   * (i.e. we are repeating a learned movement); `generalized` when the model
   * backed off to a shorter context to produce a novel-but-related step.
   */
  source: "recorded" | "generalized";
  /** Context length that actually matched (order used after backoff). */
  matchedOrder: number;
};

export type MovementPrediction = {
  steps: PredictedMovementStep[];
  stopped: "end" | "max-steps" | "no-continuation";
};

type TransitionCount = { token: string; count: number };

/**
 * A fully serializable (plain-JSON) trained model. Deliberately backend-neutral
 * in shape so a real on-device backend can persist its own weights under the
 * same envelope and be swapped in transparently.
 */
export type TrainedMovementModel = {
  version: 1;
  backendId: string;
  order: number;
  /** Backoff n-gram transitions keyed by `"<k>:<ctx>"`, each list sorted desc. */
  transitions: Record<string, TransitionCount[]>;
  /** Canonical token -> representative step, for decoding predictions. */
  stepByToken: Record<string, MovementStep>;
  trainedSequences: number;
  trainedSteps: number;
};

/**
 * The pluggable backend seam. The deterministic implementation below runs fully
 * in-process (cloud/CI safe); a real on-device small-model backend can implement
 * the same interface and be registered without touching call sites.
 */
export interface LocalMovementModelBackend {
  readonly id: string;
  train(dataset: MovementDataset, options?: TrainMovementModelOptions): Promise<TrainedMovementModel>;
  predict(model: TrainedMovementModel, context: MovementContext): Promise<MovementPrediction>;
}

const START = "START";
const END = "END";
const DEFAULT_ORDER = 2;
const DEFAULT_MAX_STEPS = 32;

function normalize(value: string | undefined): string {
  return (value ?? "").trim().toLowerCase().replace(/\s+/g, " ");
}

/** Canonical, separator-safe token for a movement step (never contains ">"). */
export function movementStepToken(step: MovementStep): string {
  const parts = [step.tool, step.gesture, step.direction, step.target].map(normalize).filter((part) => part.length > 0);
  if (parts.length <= 1) {
    parts.push(normalize(step.summary));
  }
  return parts.join("|");
}

export function movementStepFromAction(action: TrajectoryAction): MovementStep {
  const metadata = action.metadata ?? {};
  const pick = (key: string): string | undefined =>
    typeof metadata[key] === "string" ? (metadata[key] as string) : undefined;
  return {
    tool: action.tool,
    gesture: pick("gesture"),
    target: pick("target"),
    direction: pick("direction"),
    summary: action.summary,
  };
}

/** Build a dataset from captured trajectory spans (skips those with no actions). */
export function buildMovementDataset(trajectories: TrajectorySpan[]): MovementDataset {
  const sequences = trajectories
    .map((trajectory) => ({
      trajectoryId: trajectory.id,
      steps: [...trajectory.actions]
        .sort((a, b) => a.ts - b.ts)
        .map((action) => movementStepFromAction(action)),
    }))
    .filter((sequence) => sequence.steps.length > 0);
  return { version: 1, sequences };
}

/** Build a dataset from replay manifests (the reviewed-export event format). */
export function movementDatasetFromReplays(
  replays: { trajectoryIds: string[]; events: ReplayTimelineEvent[] }[],
): MovementDataset {
  const sequences: MovementSequence[] = [];
  for (const replay of replays) {
    const byTrajectory = new Map<string, MovementStep[]>();
    for (const event of [...replay.events].sort((a, b) => a.ts - b.ts)) {
      if (event.kind !== "action") {
        continue;
      }
      const steps = byTrajectory.get(event.trajectoryId) ?? [];
      steps.push({ tool: event.tool, summary: event.summary });
      byTrajectory.set(event.trajectoryId, steps);
    }
    for (const [trajectoryId, steps] of byTrajectory) {
      if (steps.length > 0) {
        sequences.push({ trajectoryId, steps });
      }
    }
  }
  return { version: 1, sequences };
}

function contextKey(order: number, tokens: string[]): string {
  return `${order}:${tokens.join(">")}`;
}

function bump(transitions: Record<string, TransitionCount[]>, key: string, token: string): void {
  const entries = (transitions[key] ??= []);
  const existing = entries.find((entry) => entry.token === token);
  if (existing) {
    existing.count += 1;
  } else {
    entries.push({ token, count: 1 });
  }
}

/**
 * Deterministic backoff n-gram ("Markov") backend. No randomness: ties break by
 * token string so training + prediction are byte-for-byte reproducible, which is
 * exactly what the cloud/CI verification gate needs. It repeats recorded
 * movements when the full context matches and generalizes via backoff when it
 * sees a novel-but-related prefix.
 */
export class DeterministicMovementBackend implements LocalMovementModelBackend {
  readonly id = "deterministic-markov";

  async train(dataset: MovementDataset, options: TrainMovementModelOptions = {}): Promise<TrainedMovementModel> {
    const order = Math.max(1, Math.floor(options.order ?? DEFAULT_ORDER));
    const transitions: Record<string, TransitionCount[]> = {};
    const stepByToken: Record<string, MovementStep> = {};
    let trainedSteps = 0;

    for (const sequence of dataset.sequences) {
      const tokens = sequence.steps.map((step) => {
        const token = movementStepToken(step);
        if (!(token in stepByToken)) {
          stepByToken[token] = { ...step };
        }
        return token;
      });
      trainedSteps += tokens.length;

      const padded = [...Array<string>(order).fill(START), ...tokens, END];
      for (let i = order; i < padded.length; i += 1) {
        const next = padded[i]!;
        for (let k = 0; k <= order; k += 1) {
          bump(transitions, contextKey(k, padded.slice(i - k, i)), next);
        }
      }
    }

    for (const entries of Object.values(transitions)) {
      entries.sort((a, b) => (b.count - a.count) || a.token.localeCompare(b.token));
    }

    return {
      version: 1,
      backendId: this.id,
      order,
      transitions,
      stepByToken,
      trainedSequences: dataset.sequences.length,
      trainedSteps,
    };
  }

  async predict(model: TrainedMovementModel, context: MovementContext): Promise<MovementPrediction> {
    const order = model.order;
    const maxSteps = Math.max(0, context.maxSteps ?? DEFAULT_MAX_STEPS);
    const tokens = [...Array<string>(order).fill(START), ...context.prefix.map((step) => movementStepToken(step))];
    const steps: PredictedMovementStep[] = [];

    while (steps.length < maxSteps) {
      let chosen: { entries: TransitionCount[]; matchedOrder: number } | undefined;
      for (let k = order; k >= 0; k -= 1) {
        const entries = model.transitions[contextKey(k, tokens.slice(tokens.length - k))];
        if (entries && entries.length > 0) {
          chosen = { entries, matchedOrder: k };
          break;
        }
      }
      if (!chosen) {
        return { steps, stopped: "no-continuation" };
      }

      const best = chosen.entries[0]!;
      if (best.token === END) {
        return { steps, stopped: "end" };
      }
      const decoded = model.stepByToken[best.token];
      if (!decoded) {
        return { steps, stopped: "no-continuation" };
      }
      const total = chosen.entries.reduce((sum, entry) => sum + entry.count, 0);
      steps.push({
        ...decoded,
        confidence: total > 0 ? best.count / total : 0,
        source: chosen.matchedOrder === order ? "recorded" : "generalized",
        matchedOrder: chosen.matchedOrder,
      });
      tokens.push(best.token);
    }

    return { steps, stopped: "max-steps" };
  }
}

/**
 * Registry so callers select a backend by id rather than hard-wiring one. The
 * deterministic backend is registered by default; a real on-device backend
 * registers under its own id and becomes selectable with no call-site changes.
 */
export class MovementBackendRegistry {
  private readonly backends = new Map<string, LocalMovementModelBackend>();

  constructor(backends: LocalMovementModelBackend[] = [new DeterministicMovementBackend()]) {
    for (const backend of backends) {
      this.register(backend);
    }
  }

  register(backend: LocalMovementModelBackend): void {
    this.backends.set(backend.id, backend);
  }

  get(id: string): LocalMovementModelBackend {
    const backend = this.backends.get(id);
    if (!backend) {
      throw new Error(`Unknown movement model backend: ${id}`);
    }
    return backend;
  }

  ids(): string[] {
    return [...this.backends.keys()];
  }
}
