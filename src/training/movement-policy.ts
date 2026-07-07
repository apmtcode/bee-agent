import type { TrajectoryAction, TrajectorySpan } from "../capture/trajectory.js";

/**
 * Movement policy learning (standing objective #2c/#2d).
 *
 * This module is the in-process, cloud-runnable half of the local-movement
 * learning subsystem: it turns a reviewed movement dataset into a small model
 * that can (a) *repeat* recorded movement sequences and (b) *generalize* to new
 * but related sequences it never saw as a whole.
 *
 * The heavy on-device training path (`runner.ts`) shells out to MLX/axolotl and
 * cannot run in the cloud. To keep the pipeline testable and to give bee-agent a
 * real, deterministic policy it can execute anywhere, the model backend is a
 * pluggable interface (`MovementModelBackend`) with a default deterministic
 * n-gram back-off backend (`NgramMovementBackend`). A real on-device small model
 * can be dropped in behind the same interface — see `movementModelRegistry`.
 */

/** A single discrete movement/gesture the policy can predict or emit. */
export type MovementAction = {
  tool: string;
  gesture?: string;
  target?: string;
  direction?: string;
  summary: string;
};

/** An ordered run of movements, optionally tagged with the goal/app context. */
export type MovementSequence = {
  id: string;
  goal?: string;
  appId?: string;
  actions: MovementAction[];
};

export type MovementDataset = {
  version: 1;
  sequences: MovementSequence[];
};

/** Sentinel token marking the end of a sequence, so generation can terminate. */
export const MOVEMENT_END_TOKEN = "<end>";

/**
 * Deterministic token for a movement. Same movement → same token, so the policy
 * can count transitions. Lower-cased and normalized; missing fields collapse to
 * `-` so related movements that differ only in an absent field still align.
 */
export function movementToken(action: MovementAction): string {
  const part = (value: string | undefined): string => {
    const normalized = value?.trim().toLowerCase();
    return normalized && normalized.length > 0 ? normalized.replaceAll(/\s+/g, "_") : "-";
  };
  return [part(action.tool), part(action.gesture), part(action.target), part(action.direction)].join(":");
}

/** Extract the movement view of a recorded trajectory action. */
export function movementFromTrajectoryAction(action: TrajectoryAction): MovementAction {
  const metadata = action.metadata ?? {};
  const asString = (value: unknown): string | undefined =>
    typeof value === "string" && value.trim().length > 0 ? value : undefined;
  return {
    tool: action.tool,
    gesture: asString(metadata.gesture),
    target: asString(metadata.target),
    direction: asString(metadata.direction),
    summary: action.summary,
  };
}

/**
 * Build a movement dataset from reviewed trajectory spans. Only trajectories
 * that are not explicitly rejected contribute; redacted actions (from review)
 * take precedence over raw actions, mirroring the exporter's redaction policy.
 * Sequences are ordered by action timestamp for a stable, replayable dataset.
 */
export function buildMovementDataset(trajectories: TrajectorySpan[]): MovementDataset {
  const sequences: MovementSequence[] = [];
  for (const trajectory of trajectories) {
    if (trajectory.review?.status === "rejected") {
      continue;
    }
    const redacted = trajectory.review?.redactedActions;
    const rawActions: MovementAction[] = redacted
      ? redacted
          .map((action) => ({ ...action, ts: action.ts }))
          .sort((a, b) => a.ts - b.ts)
          .map((action) => ({ tool: action.tool, summary: action.summary }))
      : [...trajectory.actions]
          .sort((a, b) => a.ts - b.ts)
          .map(movementFromTrajectoryAction);
    if (rawActions.length === 0) {
      continue;
    }
    sequences.push({
      id: trajectory.id,
      ...(trajectory.outcome?.summary ? { goal: trajectory.outcome.summary } : {}),
      actions: rawActions,
    });
  }
  return { version: 1, sequences };
}

export type MovementPrediction = {
  token: string;
  action: MovementAction | undefined;
  /** Estimated conditional probability of this token given the context. */
  probability: number;
  /** How many context tokens actually matched (n-gram order used). */
  contextOrder: number;
  /** True when the prediction is the end-of-sequence sentinel. */
  end: boolean;
};

export type MovementGenerateOptions = {
  maxSteps?: number;
  /** Optional seed movements establishing the starting context. */
  seed?: MovementAction[];
};

export type MovementModelSnapshot = {
  version: 1;
  backend: string;
  order: number;
  vocabulary: Record<string, MovementAction>;
  transitions: Array<{ context: string; next: Array<{ token: string; count: number }> }>;
};

/** A trained, executable movement policy. */
export interface MovementPolicyModel {
  readonly backend: string;
  readonly order: number;
  /** Predict the next movement token given a context of prior tokens. */
  predictNext(contextTokens: string[]): MovementPrediction | undefined;
  /** Generate a movement sequence, optionally seeded with a starting context. */
  generate(options?: MovementGenerateOptions): MovementAction[];
  /** Resolve a token back to a concrete movement, if known. */
  actionForToken(token: string): MovementAction | undefined;
  serialize(): MovementModelSnapshot;
}

/** Pluggable model backend. Swap the default for a real on-device model here. */
export interface MovementModelBackend {
  readonly id: string;
  train(dataset: MovementDataset): MovementPolicyModel;
}

export type NgramMovementBackendOptions = {
  /** Maximum context length (n-gram order − 1). Defaults to 2 (trigram). */
  order?: number;
};

type TransitionCounts = Map<string, Map<string, number>>;

/**
 * Deterministic n-gram back-off policy. Memorizes movement transitions at every
 * context length from `order` down to 0. On prediction it uses the longest
 * matching context (exact recall of recorded runs) and backs off to shorter
 * contexts for novel prefixes (generalization to new-but-related sequences).
 * Fully deterministic: argmax by count, ties broken by token order — no RNG.
 */
export class NgramMovementBackend implements MovementModelBackend {
  readonly id = "ngram-backoff";
  private readonly order: number;

  constructor(options: NgramMovementBackendOptions = {}) {
    this.order = Math.max(1, Math.floor(options.order ?? 2));
  }

  train(dataset: MovementDataset): MovementPolicyModel {
    const transitions: TransitionCounts = new Map();
    const vocabulary = new Map<string, MovementAction>();

    for (const sequence of dataset.sequences) {
      const tokens: string[] = [];
      for (const action of sequence.actions) {
        const token = movementToken(action);
        if (!vocabulary.has(token)) {
          vocabulary.set(token, action);
        }
        tokens.push(token);
      }
      tokens.push(MOVEMENT_END_TOKEN);

      for (let index = 0; index < tokens.length; index += 1) {
        const next = tokens[index];
        for (let ctxLen = 0; ctxLen <= this.order; ctxLen += 1) {
          if (ctxLen > index) {
            break;
          }
          const context = tokens.slice(index - ctxLen, index).join("");
          addCount(transitions, contextKey(ctxLen, context), next);
        }
      }
    }

    return new NgramMovementModel(this.id, this.order, transitions, vocabulary);
  }
}

class NgramMovementModel implements MovementPolicyModel {
  constructor(
    readonly backend: string,
    readonly order: number,
    private readonly transitions: TransitionCounts,
    private readonly vocabulary: Map<string, MovementAction>,
  ) {}

  predictNext(contextTokens: string[]): MovementPrediction | undefined {
    for (let ctxLen = Math.min(this.order, contextTokens.length); ctxLen >= 0; ctxLen -= 1) {
      const context = contextTokens.slice(contextTokens.length - ctxLen).join("");
      const counts = this.transitions.get(contextKey(ctxLen, context));
      if (!counts || counts.size === 0) {
        continue;
      }
      const { token, count, total } = argmax(counts);
      return {
        token,
        action: this.vocabulary.get(token),
        probability: total > 0 ? count / total : 0,
        contextOrder: ctxLen,
        end: token === MOVEMENT_END_TOKEN,
      };
    }
    return undefined;
  }

  generate(options: MovementGenerateOptions = {}): MovementAction[] {
    const maxSteps = Math.max(0, Math.floor(options.maxSteps ?? 32));
    const context = (options.seed ?? []).map(movementToken);
    const emitted: MovementAction[] = [];
    for (let step = 0; step < maxSteps; step += 1) {
      const prediction = this.predictNext(context);
      if (!prediction || prediction.end || !prediction.action) {
        break;
      }
      emitted.push(prediction.action);
      context.push(prediction.token);
    }
    return emitted;
  }

  actionForToken(token: string): MovementAction | undefined {
    return this.vocabulary.get(token);
  }

  serialize(): MovementModelSnapshot {
    const transitions = [...this.transitions.entries()]
      .map(([context, counts]) => ({
        context,
        next: [...counts.entries()]
          .map(([token, count]) => ({ token, count }))
          .sort((a, b) => (a.token < b.token ? -1 : a.token > b.token ? 1 : 0)),
      }))
      .sort((a, b) => (a.context < b.context ? -1 : a.context > b.context ? 1 : 0));
    return {
      version: 1,
      backend: this.backend,
      order: this.order,
      vocabulary: Object.fromEntries(this.vocabulary),
      transitions,
    };
  }
}

/**
 * Registry of pluggable movement backends. Register a real on-device small-model
 * backend under a new id and select it via `getMovementBackend(id)`.
 */
export class MovementModelRegistry {
  private readonly backends = new Map<string, MovementModelBackend>();

  register(backend: MovementModelBackend): this {
    this.backends.set(backend.id, backend);
    return this;
  }

  get(id: string): MovementModelBackend | undefined {
    return this.backends.get(id);
  }

  ids(): string[] {
    return [...this.backends.keys()].sort();
  }

  train(id: string, dataset: MovementDataset): MovementPolicyModel {
    const backend = this.backends.get(id);
    if (!backend) {
      throw new Error(`Unknown movement backend: ${id}`);
    }
    return backend.train(dataset);
  }
}

export const DEFAULT_MOVEMENT_BACKEND_ID = "ngram-backoff";

/** A registry pre-populated with the deterministic default backend. */
export function createDefaultMovementRegistry(
  options: NgramMovementBackendOptions = {},
): MovementModelRegistry {
  return new MovementModelRegistry().register(new NgramMovementBackend(options));
}

export type MovementEvaluation = {
  /** Number of next-movement predictions scored across held-out sequences. */
  predictions: number;
  /** Predictions whose top token matched the held-out ground truth. */
  correct: number;
  /** correct / predictions (0 when there is nothing to score). */
  fidelity: number;
  /** Predictions that required backing off below full context (generalization). */
  generalized: number;
};

/**
 * Generalization eval harness (roadmap: "measure replay fidelity on held-out but
 * related synthetic trajectories"). Walks each held-out sequence, asks the model
 * to predict the next movement from the true prefix, and scores top-1 accuracy.
 */
export function evaluateMovementPolicy(
  model: MovementPolicyModel,
  heldOut: MovementSequence[],
): MovementEvaluation {
  let predictions = 0;
  let correct = 0;
  let generalized = 0;
  for (const sequence of heldOut) {
    const tokens = sequence.actions.map(movementToken);
    for (let index = 0; index < tokens.length; index += 1) {
      const prediction = model.predictNext(tokens.slice(0, index));
      predictions += 1;
      if (prediction?.token === tokens[index]) {
        correct += 1;
      }
      if (prediction && prediction.contextOrder < Math.min(model.order, index)) {
        generalized += 1;
      }
    }
  }
  return {
    predictions,
    correct,
    fidelity: predictions > 0 ? correct / predictions : 0,
    generalized,
  };
}

function contextKey(length: number, context: string): string {
  return `${length} ${context}`;
}

function addCount(transitions: TransitionCounts, key: string, token: string): void {
  let counts = transitions.get(key);
  if (!counts) {
    counts = new Map();
    transitions.set(key, counts);
  }
  counts.set(token, (counts.get(token) ?? 0) + 1);
}

function argmax(counts: Map<string, number>): { token: string; count: number; total: number } {
  let bestToken = "";
  let bestCount = -1;
  let total = 0;
  for (const [token, count] of counts) {
    total += count;
    if (count > bestCount || (count === bestCount && token < bestToken)) {
      bestToken = token;
      bestCount = count;
    }
  }
  return { token: bestToken, count: bestCount, total };
}
