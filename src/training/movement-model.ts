import type { ReplayTimelineEvent } from "../capture/replay.js";

/**
 * In-process movement-model layer for the local-movement learning subsystem.
 *
 * The training {@link LocalAppleSiliconTrainingRunner} emits a launch script that
 * shells out to a real on-device backend (mlx / axolotl). That path only runs on
 * the user's machine. This module provides the *pluggable backend seam* plus a
 * deterministic, dependency-free reference backend so the full
 * capture -> dataset -> train -> infer -> generalize loop can be exercised and
 * regression-tested in the cloud where no real OS input or GPU is available.
 *
 * A real local model implements {@link MovementModelBackend} the same way the
 * mock backend does; call sites depend only on the interface, never on the
 * n-gram implementation.
 */

/** A single recorded movement/observation event. Reuses the replay schema so the
 * model consumes exactly what the reviewed-export pipeline already produces. */
export type MovementEvent = ReplayTimelineEvent;

/** An ordered sequence of movement events belonging to one recorded episode. */
export type MovementTrajectory = {
  id: string;
  events: MovementEvent[];
};

/** A dataset is a bag of independent trajectories. */
export type MovementDataset = {
  trajectories: MovementTrajectory[];
};

/** A discrete token an event is mapped to for sequence modelling. */
export type MovementToken = string;

/** Emitted after the final event of a trajectory so a rollout knows to stop. */
export const MOVEMENT_END_TOKEN = "<end>";

const CONTEXT_SEPARATOR = "";

/**
 * Map an event to a stable, comparable token. Actions and observations keep
 * their identifying fields (tool/source + normalized summary) so distinct
 * movements stay distinct; transcript turns collapse to their role since their
 * free text is not a movement to reproduce.
 */
export function tokenizeMovementEvent(event: MovementEvent): MovementToken {
  switch (event.kind) {
    case "action":
      return `action:${event.tool}:${normalizeSummary(event.summary)}`;
    case "observation":
      return `obs:${event.source}:${normalizeSummary(event.summary)}`;
    case "transcript":
      return `msg:${event.role}`;
  }
}

/** Tokenize a whole trajectory and append the end marker. */
export function tokenizeTrajectory(trajectory: MovementTrajectory): MovementToken[] {
  return [...trajectory.events.map(tokenizeMovementEvent), MOVEMENT_END_TOKEN];
}

function normalizeSummary(summary: string): string {
  return summary.trim().toLowerCase().replace(/\s+/g, " ");
}

/**
 * A trained model artifact. Fully serializable (plain JSON) so it can be
 * persisted next to a job's other artifacts and reloaded for inference.
 */
export type MovementModelArtifact = {
  version: 1;
  backend: string;
  /** Maximum context length (n-1 for an n-gram). */
  order: number;
  /** Distinct real tokens observed during training (excludes the end marker). */
  vocabulary: MovementToken[];
  /**
   * context-key -> next-token -> count. Keys hold contexts of every length from
   * 0 (empty, the base rate) up to `order`, enabling stupid-backoff at
   * inference time so unseen full contexts still yield a prediction.
   */
  transitions: Record<string, Record<MovementToken, number>>;
  trajectoryCount: number;
  tokenCount: number;
};

export type MovementTrainingOptions = {
  /** n-gram context length. Defaults to 3. Clamped to >= 0. */
  order?: number;
};

export type MovementInferenceOptions = {
  /** Cap the context length actually used; defaults to the model's order. */
  maxOrder?: number;
};

export type MovementPrediction = {
  token: MovementToken;
  probability: number;
};

export type MovementInferenceResult = {
  /** Argmax next token, or undefined when the model is empty. */
  prediction: MovementToken | undefined;
  probability: number;
  /** Context length actually used after backoff (0 = base rate). */
  backoffOrder: number;
  /** Full next-token distribution, highest probability first. */
  distribution: MovementPrediction[];
  /** True when the predicted next token is the end marker. */
  isEnd: boolean;
};

/**
 * Pluggable training/inference backend. The mock n-gram backend below is the
 * cloud-testable default; a real on-device small-model backend implements the
 * same three members and is swapped in via {@link registerMovementBackend}.
 */
export interface MovementModelBackend {
  readonly name: string;
  train(dataset: MovementDataset, options?: MovementTrainingOptions): Promise<MovementModelArtifact>;
  infer(
    model: MovementModelArtifact,
    context: MovementToken[],
    options?: MovementInferenceOptions,
  ): MovementInferenceResult;
}

/**
 * Deterministic n-gram backend with stupid-backoff. No native deps, no
 * randomness — identical dataset in, identical model out — so it is safe to
 * assert on in CI. Generalizes to unseen full contexts by backing off to the
 * longest seen suffix.
 */
export class NgramMovementBackend implements MovementModelBackend {
  readonly name = "ngram-mock";

  async train(dataset: MovementDataset, options: MovementTrainingOptions = {}): Promise<MovementModelArtifact> {
    const order = Math.max(0, Math.trunc(options.order ?? 3));
    const transitions: Record<string, Record<MovementToken, number>> = {};
    const vocabulary = new Set<MovementToken>();
    let tokenCount = 0;

    for (const trajectory of dataset.trajectories) {
      const tokens = tokenizeTrajectory(trajectory);
      for (let index = 0; index < tokens.length; index += 1) {
        const target = tokens[index]!;
        tokenCount += 1;
        if (target !== MOVEMENT_END_TOKEN) {
          vocabulary.add(target);
        }
        const maxContext = Math.min(order, index);
        for (let contextLength = 0; contextLength <= maxContext; contextLength += 1) {
          const key = contextKey(tokens.slice(index - contextLength, index));
          const counts = (transitions[key] ??= {});
          counts[target] = (counts[target] ?? 0) + 1;
        }
      }
    }

    return {
      version: 1,
      backend: this.name,
      order,
      vocabulary: [...vocabulary].sort(),
      transitions,
      trajectoryCount: dataset.trajectories.length,
      tokenCount,
    };
  }

  infer(
    model: MovementModelArtifact,
    context: MovementToken[],
    options: MovementInferenceOptions = {},
  ): MovementInferenceResult {
    const maxOrder = Math.max(0, Math.min(model.order, options.maxOrder ?? model.order));
    for (let contextLength = Math.min(maxOrder, context.length); contextLength >= 0; contextLength -= 1) {
      const key = contextKey(context.slice(context.length - contextLength));
      const counts = model.transitions[key];
      if (!counts) {
        continue;
      }
      const distribution = toDistribution(counts);
      if (distribution.length === 0) {
        continue;
      }
      const top = distribution[0]!;
      return {
        prediction: top.token,
        probability: top.probability,
        backoffOrder: contextLength,
        distribution,
        isEnd: top.token === MOVEMENT_END_TOKEN,
      };
    }
    return { prediction: undefined, probability: 0, backoffOrder: 0, distribution: [], isEnd: false };
  }
}

function contextKey(tokens: MovementToken[]): string {
  return tokens.join(CONTEXT_SEPARATOR);
}

function toDistribution(counts: Record<MovementToken, number>): MovementPrediction[] {
  const total = Object.values(counts).reduce((sum, count) => sum + count, 0);
  if (total === 0) {
    return [];
  }
  return Object.entries(counts)
    .map(([token, count]) => ({ token, probability: count / total }))
    // Deterministic ordering: probability desc, then token asc as a stable tie-break.
    .sort((a, b) => (b.probability - a.probability) || (a.token < b.token ? -1 : a.token > b.token ? 1 : 0));
}

/**
 * Roll the model forward from a seed context, emitting predicted next tokens
 * until the end marker, an empty prediction, or `maxSteps` is hit. This is how a
 * recorded movement is *repeated*, and — because inference backs off — how the
 * model *generalizes* to a related-but-unseen starting context.
 */
export function generateMovementSequence(
  backend: MovementModelBackend,
  model: MovementModelArtifact,
  seed: MovementToken[],
  options: { maxSteps?: number; maxOrder?: number } = {},
): MovementToken[] {
  const maxSteps = Math.max(0, options.maxSteps ?? 64);
  const context = [...seed];
  const generated: MovementToken[] = [];
  for (let step = 0; step < maxSteps; step += 1) {
    const result = backend.infer(model, context, { maxOrder: options.maxOrder });
    if (result.prediction === undefined || result.isEnd) {
      break;
    }
    generated.push(result.prediction);
    context.push(result.prediction);
  }
  return generated;
}

export type MovementEvalResult = {
  trajectoryId: string;
  /** Number of next-token predictions scored. */
  steps: number;
  correct: number;
  /** correct / steps, or 1 for an empty trajectory. */
  accuracy: number;
};

/**
 * Teacher-forced next-token accuracy over a held-out trajectory: from the first
 * token onward, feed the true prefix and check the argmax matches the recorded
 * next token (the end marker included). The opening token is treated as the
 * given seed — reproducing a movement means "given where you are, what's next",
 * so the unconditioned first token is not scored. The generalization eval
 * measures this on trajectories the model was not trained on.
 */
export function evaluateNextTokenAccuracy(
  backend: MovementModelBackend,
  model: MovementModelArtifact,
  trajectory: MovementTrajectory,
): MovementEvalResult {
  const tokens = tokenizeTrajectory(trajectory);
  let correct = 0;
  let steps = 0;
  for (let index = 1; index < tokens.length; index += 1) {
    steps += 1;
    const result = backend.infer(model, tokens.slice(0, index));
    if (result.prediction === tokens[index]) {
      correct += 1;
    }
  }
  return {
    trajectoryId: trajectory.id,
    steps,
    correct,
    accuracy: steps === 0 ? 1 : correct / steps,
  };
}

/** Mean next-token accuracy across a held-out set — the top-line generalization score. */
export function evaluateDatasetAccuracy(
  backend: MovementModelBackend,
  model: MovementModelArtifact,
  dataset: MovementDataset,
): { perTrajectory: MovementEvalResult[]; meanAccuracy: number } {
  const perTrajectory = dataset.trajectories.map((trajectory) =>
    evaluateNextTokenAccuracy(backend, model, trajectory),
  );
  const meanAccuracy =
    perTrajectory.length === 0
      ? 1
      : perTrajectory.reduce((sum, result) => sum + result.accuracy, 0) / perTrajectory.length;
  return { perTrajectory, meanAccuracy };
}

/** Build a training dataset from reviewed-export replay manifests. */
export function movementDatasetFromReplays(
  replays: Array<{ trajectoryIds?: string[]; events: MovementEvent[] }>,
): MovementDataset {
  return {
    trajectories: replays.map((replay, index) => ({
      id: replay.trajectoryIds?.join("+") || `replay-${index}`,
      events: replay.events,
    })),
  };
}

const backendFactories = new Map<string, () => MovementModelBackend>();

/** Register (or override) a backend factory under a name. */
export function registerMovementBackend(name: string, factory: () => MovementModelBackend): void {
  backendFactories.set(name, factory);
}

/** Instantiate a registered backend; defaults to the deterministic mock. */
export function createMovementBackend(name = "ngram-mock"): MovementModelBackend {
  const factory = backendFactories.get(name);
  if (!factory) {
    throw new Error(`unknown movement backend: ${name} (available: ${listMovementBackends().join(", ")})`);
  }
  return factory();
}

/** Names of all registered backends, sorted. */
export function listMovementBackends(): string[] {
  return [...backendFactories.keys()].sort();
}

registerMovementBackend("ngram-mock", () => new NgramMovementBackend());
