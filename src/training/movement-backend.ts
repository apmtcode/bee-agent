import type { MovementActionLabel, MovementContext, MovementDataset } from "./movement-dataset.js";

/**
 * Pluggable local-model backend for the movement-learning subsystem.
 *
 * A backend post-trains a {@link MovementPolicy} on a {@link MovementDataset}
 * (context -> action pairs derived from recorded trajectories). The policy can
 * then be asked to predict the action for a given context — reproducing
 * recorded movements exactly and generalizing to new-but-related ones.
 *
 * The interface is deliberately backend-agnostic so a real on-device small
 * model (e.g. an MLX/GGUF policy trained by {@link LocalAppleSiliconTrainingRunner})
 * can be dropped in later. Cloud/CI runs use {@link DeterministicMovementPolicyBackend},
 * an n-gram-style frequency model that trains and infers fully in-process with
 * no randomness, so the whole capture -> dataset -> train -> infer -> replay
 * loop is testable without a real machine or GPU.
 */
export interface MovementPolicyBackend {
  readonly id: string;
  train(dataset: MovementDataset, options?: MovementTrainingOptions): Promise<MovementPolicy>;
  load(serialized: SerializedMovementPolicy): MovementPolicy;
}

export type MovementTrainingOptions = {
  /** Reserved for real backends (learning rate, epochs…); ignored by the mock. */
  hyperparameters?: Record<string, number | string | boolean>;
};

export type MovementPredictionSource = "exact" | "generalized" | "fallback" | "empty";

export type MovementPrediction = {
  /** Predicted action, or undefined when the policy has no training signal. */
  action: MovementActionLabel | undefined;
  /** 0..1 heuristic confidence; higher for exact reproductions. */
  confidence: number;
  source: MovementPredictionSource;
  /** Context features that drove the match (for inspection / eval harnesses). */
  matchedFeatures: string[];
};

export interface MovementPolicy {
  readonly backendId: string;
  readonly stepCount: number;
  predict(context: MovementContext): MovementPrediction;
  serialize(): SerializedMovementPolicy;
}

// ---------------------------------------------------------------------------
// Deterministic in-process backend
// ---------------------------------------------------------------------------

type ScoredAction = { action: MovementActionLabel; count: number; total: number };

export type SerializedMovementPolicy = {
  version: 1;
  backendId: string;
  stepCount: number;
  exact: Record<string, ScoredAction>;
  levels: Array<{ features: string[]; table: Record<string, ScoredAction> }>;
  fallback: ScoredAction | null;
};

/** Backoff feature sets, most specific first. Empty entry = exact match. */
const BACKOFF_LEVELS: string[][] = [
  ["appName", "observationSource", "lastActionTool"],
  ["observationSource", "lastActionTool"],
  ["appName", "lastActionTool"],
  ["lastActionTool"],
  ["appName", "observationSource"],
  ["observationSource"],
  ["appName"],
];

const EXACT_FEATURES: Array<keyof MovementContext> = [
  "appName",
  "observationSource",
  "observationSummary",
  "lastActionTool",
  "lastActionSummary",
  "stepIndex",
];

export class DeterministicMovementPolicyBackend implements MovementPolicyBackend {
  readonly id = "deterministic-frequency";

  async train(dataset: MovementDataset): Promise<MovementPolicy> {
    const exact = new BucketTable();
    const levelTables = BACKOFF_LEVELS.map(() => new BucketTable());
    const fallback = new Bucket();

    for (const example of dataset.examples) {
      exact.add(encodeExact(example.context), example.action);
      BACKOFF_LEVELS.forEach((features, index) => {
        const signature = encodeSubset(example.context, features);
        if (signature !== null) {
          levelTables[index].add(signature, example.action);
        }
      });
      fallback.add(example.action);
    }

    return new DeterministicMovementPolicy(this.id, dataset.exampleCount, {
      version: 1,
      backendId: this.id,
      stepCount: dataset.exampleCount,
      exact: exact.finalize(),
      levels: BACKOFF_LEVELS.map((features, index) => ({
        features,
        table: levelTables[index].finalize(),
      })),
      fallback: fallback.finalize(),
    });
  }

  load(serialized: SerializedMovementPolicy): MovementPolicy {
    return new DeterministicMovementPolicy(serialized.backendId, serialized.stepCount, serialized);
  }
}

class DeterministicMovementPolicy implements MovementPolicy {
  constructor(
    readonly backendId: string,
    readonly stepCount: number,
    private readonly model: SerializedMovementPolicy,
  ) {}

  predict(context: MovementContext): MovementPrediction {
    const exactHit = this.model.exact[encodeExact(context)];
    if (exactHit) {
      return {
        action: exactHit.action,
        confidence: clamp(0.5 + 0.5 * share(exactHit)),
        source: "exact",
        matchedFeatures: ["exact"],
      };
    }

    for (const level of this.model.levels) {
      const signature = encodeSubset(context, level.features);
      if (signature === null) {
        continue;
      }
      const hit = level.table[signature];
      if (hit) {
        return {
          action: hit.action,
          confidence: clamp(0.5 * share(hit)),
          source: "generalized",
          matchedFeatures: [...level.features],
        };
      }
    }

    if (this.model.fallback) {
      return {
        action: this.model.fallback.action,
        confidence: clamp(0.25 * share(this.model.fallback)),
        source: "fallback",
        matchedFeatures: [],
      };
    }

    return { action: undefined, confidence: 0, source: "empty", matchedFeatures: [] };
  }

  serialize(): SerializedMovementPolicy {
    return this.model;
  }
}

/** Run a policy over a sequence of contexts (e.g. held-out synthetic steps). */
export function rolloutMovementPolicy(policy: MovementPolicy, contexts: MovementContext[]): MovementPrediction[] {
  return contexts.map((context) => policy.predict(context));
}

// ---------------------------------------------------------------------------
// Registry — the seam where real on-device backends are plugged in
// ---------------------------------------------------------------------------

export class MovementBackendRegistry {
  private readonly backends = new Map<string, MovementPolicyBackend>();

  register(backend: MovementPolicyBackend): void {
    this.backends.set(backend.id, backend);
  }

  get(id: string): MovementPolicyBackend | undefined {
    return this.backends.get(id);
  }

  list(): MovementPolicyBackend[] {
    return [...this.backends.values()].sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0));
  }
}

export function createDefaultMovementBackendRegistry(): MovementBackendRegistry {
  const registry = new MovementBackendRegistry();
  registry.register(new DeterministicMovementPolicyBackend());
  return registry;
}

// ---------------------------------------------------------------------------
// Internal frequency bookkeeping (deterministic, order-independent)
// ---------------------------------------------------------------------------

class Bucket {
  private readonly counts = new Map<string, { action: MovementActionLabel; count: number }>();

  add(action: MovementActionLabel): void {
    const key = actionKey(action);
    const existing = this.counts.get(key);
    if (existing) {
      existing.count += 1;
    } else {
      this.counts.set(key, { action, count: 1 });
    }
  }

  finalize(): ScoredAction | null {
    let best: { action: MovementActionLabel; count: number } | undefined;
    let total = 0;
    // Sort by key so ties resolve deterministically regardless of insertion order.
    for (const key of [...this.counts.keys()].sort()) {
      const entry = this.counts.get(key)!;
      total += entry.count;
      if (!best || entry.count > best.count) {
        best = entry;
      }
    }
    return best ? { action: best.action, count: best.count, total } : null;
  }
}

class BucketTable {
  private readonly buckets = new Map<string, Bucket>();

  add(signature: string, action: MovementActionLabel): void {
    let bucket = this.buckets.get(signature);
    if (!bucket) {
      bucket = new Bucket();
      this.buckets.set(signature, bucket);
    }
    bucket.add(action);
  }

  finalize(): Record<string, ScoredAction> {
    const table: Record<string, ScoredAction> = {};
    for (const signature of [...this.buckets.keys()].sort()) {
      const scored = this.buckets.get(signature)!.finalize();
      if (scored) {
        table[signature] = scored;
      }
    }
    return table;
  }
}

function encodeExact(context: MovementContext): string {
  return JSON.stringify(EXACT_FEATURES.map((feature) => encodeValue(context[feature])));
}

function encodeSubset(context: MovementContext, features: string[]): string | null {
  const values: string[] = [];
  for (const feature of features) {
    const value = context[feature as keyof MovementContext];
    if (value === undefined) {
      return null;
    }
    values.push(`${feature}=${encodeValue(value)}`);
  }
  return values.join("|");
}

function encodeValue(value: string | number | undefined): string {
  return value === undefined ? "∅" : String(value);
}

function actionKey(action: MovementActionLabel): string {
  return JSON.stringify([action.tool, action.summary, action.gesture, action.target, action.direction]);
}

function share(scored: ScoredAction): number {
  return scored.total > 0 ? scored.count / scored.total : 0;
}

function clamp(value: number): number {
  return Math.max(0, Math.min(1, value));
}
