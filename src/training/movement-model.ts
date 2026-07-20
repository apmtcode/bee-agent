import type { TrajectorySpan } from "../capture/trajectory.js";

/**
 * Movement-model subsystem (standing objective #2, parts c & d).
 *
 * This module turns reviewed movement/action trajectories into a training
 * dataset of ordered *action tokens*, and defines a **pluggable** backend
 * interface that a local model can implement to (c) repeat recorded movements
 * and (d) generalize to new-but-related movements.
 *
 * The concrete on-device backends (MLX / axolotl, driven by
 * `LocalAppleSiliconTrainingRunner`) run only when the user runs bee-agent
 * locally. To keep the pipeline exercisable in the cloud/CI — where there is no
 * real OS input and no GPU — the seam here is deliberately backend-agnostic and
 * ships with a deterministic in-process backend (see
 * `markov-movement-backend.ts`) so the full capture -> dataset -> train ->
 * infer loop can be validated with synthetic data.
 */

/** Sentinel marking the start of a movement sequence. */
export const MOVEMENT_BOS = "<bos>";
/** Sentinel marking the end of a movement sequence. */
export const MOVEMENT_EOS = "<eos>";

/** One recorded movement sequence, tokenized into ordered action tokens. */
export type MovementSample = {
  /** Source trajectory id, for traceability back to the reviewed capture. */
  id: string;
  /** Ordered action tokens (does not include BOS/EOS sentinels). */
  tokens: string[];
  /** Relative importance of this sample during training (>= 0). */
  weight: number;
  /** Optional human-readable label (e.g. the trajectory outcome summary). */
  label?: string;
};

/** A backend-agnostic training dataset built from reviewed trajectories. */
export type MovementDataset = {
  version: 1;
  /** Sorted, de-duplicated set of every token that appears (incl. sentinels). */
  vocabulary: string[];
  samples: MovementSample[];
};

/** The prediction returned for a single next-action step. */
export type MovementPrediction = {
  /** Most likely next action token (or {@link MOVEMENT_EOS} to stop). */
  token: string;
  /** Probability of {@link token} within the matched context (0..1). */
  probability: number;
  /**
   * Length of the context the prediction was actually conditioned on. A value
   * below the requested context length means the backend *backed off* to a
   * shorter context — the mechanism that lets it generalize to unseen prefixes.
   */
  backoffOrder: number;
  /** Other candidate continuations, most probable first. */
  alternatives: Array<{ token: string; probability: number }>;
};

/** Backend-portable serialized form of a trained model. */
export type SerializedMovementModel = {
  version: 1;
  backendId: string;
  order: number;
  vocabulary: string[];
  /** context-key -> (next token -> weighted count). "" is the unigram base. */
  transitions: Record<string, Record<string, number>>;
};

/** A trained model that can predict/continue movement sequences. */
export interface MovementModel {
  readonly backendId: string;
  readonly vocabulary: readonly string[];
  /** Predict the next action given a context of prior action tokens. */
  predictNext(context: readonly string[]): MovementPrediction;
  /**
   * Continue a seed sequence for up to `steps` actions, stopping early if the
   * model predicts {@link MOVEMENT_EOS}. Returns only the generated tokens.
   */
  generate(seed: readonly string[], steps: number): string[];
  /** Serialize to a backend-portable JSON-safe structure. */
  serialize(): SerializedMovementModel;
}

export type MovementTrainOptions = {
  /** Max context length the backend should learn (implementation may clamp). */
  order?: number;
};

/**
 * Pluggable local-model backend. A real on-device small model implements this
 * same interface; the mock/deterministic backend used in tests implements it
 * too, so call sites are backend-agnostic.
 */
export interface MovementModelBackend {
  readonly id: string;
  train(dataset: MovementDataset, options?: MovementTrainOptions): Promise<MovementModel>;
  /** Rehydrate a model previously produced by this backend's `serialize()`. */
  load(serialized: SerializedMovementModel): MovementModel;
}

/** Registry so callers can select a backend by id (config-driven pluggability). */
export class MovementBackendRegistry {
  private readonly backends = new Map<string, MovementModelBackend>();

  register(backend: MovementModelBackend): this {
    this.backends.set(backend.id, backend);
    return this;
  }

  get(id: string): MovementModelBackend {
    const backend = this.backends.get(id);
    if (!backend) {
      throw new Error(`unknown movement backend: ${id} (available: ${this.list().join(", ") || "none"})`);
    }
    return backend;
  }

  has(id: string): boolean {
    return this.backends.has(id);
  }

  list(): string[] {
    return [...this.backends.keys()].sort();
  }
}

/** Lower-case, hyphenate and bound a free-text value into a stable token piece. */
export function slugifyMovement(value: string): string {
  const slug = value
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
  if (slug.length <= 48) {
    return slug;
  }
  return slug.slice(0, 48).replace(/-+$/g, "");
}

/** Build a deterministic action token from a tool name and action summary. */
export function movementActionToken(tool: string, summary: string): string {
  const toolSlug = slugifyMovement(tool) || "action";
  const summarySlug = slugifyMovement(summary);
  return summarySlug ? `${toolSlug}:${summarySlug}` : toolSlug;
}

/**
 * Tokenize a trajectory into ordered action tokens. Reviewed/redacted actions
 * are preferred over raw ones (only reviewed data is ever exported for
 * training); actions are ordered by timestamp for a stable sequence.
 */
export function tokenizeTrajectory(span: TrajectorySpan): string[] {
  const reviewed = span.review?.redactedActions;
  if (reviewed && reviewed.length > 0) {
    return [...reviewed]
      .sort((a, b) => a.ts - b.ts)
      .map((action) => movementActionToken(action.tool, action.summary));
  }
  return [...span.actions]
    .sort((a, b) => a.ts - b.ts)
    .map((action) => movementActionToken(action.tool, action.summary));
}

/**
 * Assemble a {@link MovementDataset} from trajectory spans. Trajectories with
 * fewer than `minTokens` actions are skipped. A positive outcome reward is used
 * as the sample weight so successful demonstrations count more heavily.
 */
export function buildMovementDataset(
  spans: readonly TrajectorySpan[],
  options?: { minTokens?: number },
): MovementDataset {
  const minTokens = Math.max(1, options?.minTokens ?? 1);
  const vocabulary = new Set<string>([MOVEMENT_BOS, MOVEMENT_EOS]);
  const samples: MovementSample[] = [];

  for (const span of spans) {
    const tokens = tokenizeTrajectory(span);
    if (tokens.length < minTokens) {
      continue;
    }
    for (const token of tokens) {
      vocabulary.add(token);
    }
    const reward = span.outcome?.reward;
    const weight = typeof reward === "number" && reward > 0 ? reward : 1;
    samples.push({
      id: span.id,
      tokens,
      weight,
      ...(span.outcome?.summary ? { label: span.outcome.summary } : {}),
    });
  }

  return {
    version: 1,
    vocabulary: [...vocabulary].sort(),
    samples,
  };
}
