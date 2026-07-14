import type { ExportedReplayManifest } from "./export-manifest.js";

/**
 * Movement-model subsystem (standing objective #2c/#2d).
 *
 * This module turns reviewed replay trajectories into a trainable sequence
 * dataset, defines a *pluggable* local-model backend interface, and ships a
 * deterministic backend that can run in the cloud with no OS/GPU access. The
 * deterministic backend learns a first-order Markov model over movement tokens
 * so it can (a) *repeat* recorded movements exactly and (b) *generalize* to new
 * but related movements via source/tool back-off. A real on-device small model
 * (e.g. an MLX/axolotl policy) can implement the same `MovementModelBackend`
 * seam without changing any call site.
 */

/** A single step in a recorded movement trajectory. */
export type MovementToken =
  | { kind: "observation"; source: string; summary: string }
  | { kind: "action"; tool: string; summary: string };

/** An ordered token sequence for one trajectory. */
export type MovementSequence = {
  trajectoryId: string;
  tokens: MovementToken[];
};

export type MovementTrainingDataset = {
  sequences: MovementSequence[];
};

/** A predicted next movement plus the evidence behind it. */
export type MovementPrediction = {
  action: { tool: string; summary: string };
  /** Share of observed transitions from this state that chose this action (0..1). */
  confidence: number;
  /** How many observed transitions back this prediction. */
  support: number;
  /**
   * `false` when the prediction came from an exact state match (a faithful
   * replay of a recorded movement); `true` when it was reached by back-off to a
   * coarser state (a generalization to a new-but-related movement).
   */
  generalized: boolean;
  /** Which resolution tier produced the prediction. */
  via: "exact" | "coarse" | "global";
};

export type MovementModelMetrics = {
  sequenceCount: number;
  transitionCount: number;
  distinctStates: number;
  distinctActions: number;
};

type ActionEntry = { tool: string; summary: string; count: number };
type StateTable = Record<string, Record<string, ActionEntry>>;

export type MovementModelSnapshot = {
  version: 1;
  backend: string;
  exact: StateTable;
  coarse: StateTable;
  global: Record<string, ActionEntry>;
  metrics: MovementModelMetrics;
};

export interface TrainedMovementModel {
  readonly backend: string;
  /** Predict the next movement given the context so far (uses the last token). */
  predictNext(context: MovementToken[]): MovementPrediction | undefined;
  metrics(): MovementModelMetrics;
  serialize(): MovementModelSnapshot;
}

export interface MovementModelBackend {
  readonly name: string;
  train(dataset: MovementTrainingDataset): Promise<TrainedMovementModel>;
  restore(snapshot: MovementModelSnapshot): TrainedMovementModel;
}

const KEY_SEP = "\u0000";

function actionKey(tool: string, summary: string): string {
  return `${tool}${KEY_SEP}${summary}`;
}

/** Full-fidelity state key — distinguishes summaries, so exact replays match. */
function stateKeyExact(token: MovementToken): string {
  return token.kind === "observation"
    ? `o${KEY_SEP}${token.source}${KEY_SEP}${token.summary}`
    : `a${KEY_SEP}${token.tool}${KEY_SEP}${token.summary}`;
}

/** Coarse state key — drops the summary so related-but-new states still match. */
function stateKeyCoarse(token: MovementToken): string {
  return token.kind === "observation" ? `o${KEY_SEP}${token.source}` : `a${KEY_SEP}${token.tool}`;
}

function bump(table: Record<string, ActionEntry>, action: { tool: string; summary: string }): void {
  const key = actionKey(action.tool, action.summary);
  const existing = table[key];
  if (existing) {
    existing.count += 1;
  } else {
    table[key] = { tool: action.tool, summary: action.summary, count: 1 };
  }
}

/** Pick the highest-count action, breaking ties by action key for determinism. */
function topAction(table: Record<string, ActionEntry> | undefined): { entry: ActionEntry; total: number } | undefined {
  if (!table) {
    return undefined;
  }
  const entries = Object.entries(table);
  if (entries.length === 0) {
    return undefined;
  }
  let best: [string, ActionEntry] | undefined;
  let total = 0;
  for (const entry of entries) {
    total += entry[1].count;
    if (!best || entry[1].count > best[1].count || (entry[1].count === best[1].count && entry[0] < best[0])) {
      best = entry;
    }
  }
  return best ? { entry: best[1], total } : undefined;
}

class DeterministicMovementModel implements TrainedMovementModel {
  constructor(
    readonly backend: string,
    private readonly snapshot: MovementModelSnapshot,
  ) {}

  predictNext(context: MovementToken[]): MovementPrediction | undefined {
    const last = context.at(-1);
    if (last) {
      const exact = topAction(this.snapshot.exact[stateKeyExact(last)]);
      if (exact) {
        return toPrediction(exact, false, "exact");
      }
      const coarse = topAction(this.snapshot.coarse[stateKeyCoarse(last)]);
      if (coarse) {
        return toPrediction(coarse, true, "coarse");
      }
    }
    const global = topAction(this.snapshot.global);
    return global ? toPrediction(global, true, "global") : undefined;
  }

  metrics(): MovementModelMetrics {
    return { ...this.snapshot.metrics };
  }

  serialize(): MovementModelSnapshot {
    return structuredCloneSnapshot(this.snapshot);
  }
}

function toPrediction(
  hit: { entry: ActionEntry; total: number },
  generalized: boolean,
  via: MovementPrediction["via"],
): MovementPrediction {
  return {
    action: { tool: hit.entry.tool, summary: hit.entry.summary },
    confidence: hit.total > 0 ? hit.entry.count / hit.total : 0,
    support: hit.entry.count,
    generalized,
    via,
  };
}

/**
 * Deterministic, dependency-free backend. Trains a first-order Markov model
 * over movement tokens: for every adjacent `(state → action)` pair it records a
 * frequency at both exact and coarse resolution, plus a global action prior.
 * No randomness, so training and inference are fully reproducible in CI.
 */
export class DeterministicMovementBackend implements MovementModelBackend {
  readonly name = "deterministic-markov";

  async train(dataset: MovementTrainingDataset): Promise<TrainedMovementModel> {
    const exact: StateTable = {};
    const coarse: StateTable = {};
    const global: Record<string, ActionEntry> = {};
    let transitionCount = 0;

    for (const sequence of dataset.sequences) {
      for (let i = 1; i < sequence.tokens.length; i += 1) {
        const next = sequence.tokens[i];
        if (next.kind !== "action") {
          continue;
        }
        const prev = sequence.tokens[i - 1];
        const action = { tool: next.tool, summary: next.summary };
        (exact[stateKeyExact(prev)] ??= {});
        (coarse[stateKeyCoarse(prev)] ??= {});
        bump(exact[stateKeyExact(prev)], action);
        bump(coarse[stateKeyCoarse(prev)], action);
        bump(global, action);
        transitionCount += 1;
      }
    }

    const snapshot: MovementModelSnapshot = {
      version: 1,
      backend: this.name,
      exact,
      coarse,
      global,
      metrics: {
        sequenceCount: dataset.sequences.length,
        transitionCount,
        distinctStates: Object.keys(exact).length,
        distinctActions: Object.keys(global).length,
      },
    };
    return new DeterministicMovementModel(this.name, snapshot);
  }

  restore(snapshot: MovementModelSnapshot): TrainedMovementModel {
    return new DeterministicMovementModel(snapshot.backend, structuredCloneSnapshot(snapshot));
  }
}

/**
 * Registry making the model backend pluggable (objective #2: "make the model
 * backend pluggable"). Register a real on-device backend under its own name and
 * select it by name at train/restore time without touching call sites.
 */
export class MovementModelRegistry {
  private readonly backends = new Map<string, MovementModelBackend>();

  constructor(backends: MovementModelBackend[] = [new DeterministicMovementBackend()]) {
    for (const backend of backends) {
      this.register(backend);
    }
  }

  register(backend: MovementModelBackend): void {
    this.backends.set(backend.name, backend);
  }

  get(name: string): MovementModelBackend {
    const backend = this.backends.get(name);
    if (!backend) {
      throw new Error(`Unknown movement-model backend: ${name}`);
    }
    return backend;
  }

  has(name: string): boolean {
    return this.backends.has(name);
  }

  list(): string[] {
    return [...this.backends.keys()].sort();
  }

  restore(snapshot: MovementModelSnapshot): TrainedMovementModel {
    return this.get(snapshot.backend).restore(snapshot);
  }
}

/**
 * Derive a movement-training dataset from reviewed replay manifests. Only
 * observation/action events become tokens (transcript chatter is dropped);
 * tokens are grouped per trajectory and ordered by timestamp so each sequence
 * reflects the real order the movements happened in.
 */
export function datasetFromReplays(replays: ExportedReplayManifest[]): MovementTrainingDataset {
  const byTrajectory = new Map<string, Array<{ ts: number; order: number; token: MovementToken }>>();
  let order = 0;
  for (const replay of replays) {
    for (const event of replay.events) {
      if (event.kind === "transcript") {
        continue;
      }
      const token: MovementToken =
        event.kind === "observation"
          ? { kind: "observation", source: event.source, summary: event.summary }
          : { kind: "action", tool: event.tool, summary: event.summary };
      const bucket = byTrajectory.get(event.trajectoryId) ?? [];
      bucket.push({ ts: event.ts, order: order++, token });
      byTrajectory.set(event.trajectoryId, bucket);
    }
  }

  const sequences: MovementSequence[] = [];
  for (const [trajectoryId, entries] of byTrajectory) {
    entries.sort((a, b) => (a.ts !== b.ts ? a.ts - b.ts : a.order - b.order));
    sequences.push({ trajectoryId, tokens: entries.map((entry) => entry.token) });
  }
  sequences.sort((a, b) => (a.trajectoryId < b.trajectoryId ? -1 : a.trajectoryId > b.trajectoryId ? 1 : 0));
  return { sequences };
}

export type MovementEvaluation = {
  /** Transitions where the model's top action exactly matched the recorded one. */
  correct: number;
  /** Total action transitions evaluated. */
  total: number;
  /** correct / total (0..1); 1 means the model reproduces every held-out move. */
  accuracy: number;
  /** How many correct predictions required generalization (coarse/global back-off). */
  generalizedHits: number;
};

/**
 * Generalization eval harness (roadmap item): replay each held-out sequence and
 * measure whether the model predicts the actual next action at every step. When
 * the held-out set contains new-but-related movements, `generalizedHits`
 * captures how much of the accuracy came from back-off rather than exact recall.
 */
export function evaluateMovementModel(
  model: TrainedMovementModel,
  heldOut: MovementSequence[],
): MovementEvaluation {
  let correct = 0;
  let total = 0;
  let generalizedHits = 0;
  for (const sequence of heldOut) {
    for (let i = 1; i < sequence.tokens.length; i += 1) {
      const expected = sequence.tokens[i];
      if (expected.kind !== "action") {
        continue;
      }
      total += 1;
      const prediction = model.predictNext(sequence.tokens.slice(0, i));
      if (prediction && prediction.action.tool === expected.tool && prediction.action.summary === expected.summary) {
        correct += 1;
        if (prediction.generalized) {
          generalizedHits += 1;
        }
      }
    }
  }
  return { correct, total, accuracy: total > 0 ? correct / total : 0, generalizedHits };
}

function structuredCloneSnapshot(snapshot: MovementModelSnapshot): MovementModelSnapshot {
  const cloneTable = (table: StateTable): StateTable => {
    const out: StateTable = {};
    for (const [state, actions] of Object.entries(table)) {
      out[state] = {};
      for (const [key, entry] of Object.entries(actions)) {
        out[state][key] = { ...entry };
      }
    }
    return out;
  };
  const cloneActions = (table: Record<string, ActionEntry>): Record<string, ActionEntry> => {
    const out: Record<string, ActionEntry> = {};
    for (const [key, entry] of Object.entries(table)) {
      out[key] = { ...entry };
    }
    return out;
  };
  return {
    version: snapshot.version,
    backend: snapshot.backend,
    exact: cloneTable(snapshot.exact),
    coarse: cloneTable(snapshot.coarse),
    global: cloneActions(snapshot.global),
    metrics: { ...snapshot.metrics },
  };
}
