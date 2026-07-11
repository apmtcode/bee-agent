import type { ReplayManifest } from "../capture/replay.js";

/**
 * Movement-learning model backend.
 *
 * This module implements the "post-train a local model on the recorded dataset
 * to repeat (and generalize) the recorded movements" objective of bee-agent's
 * local-movement learning subsystem. The heavy on-device trainers
 * (`LocalAppleSiliconTrainingRunner`) shell out to external Python runtimes and
 * cannot run in the cloud. This module instead provides a *pluggable*,
 * in-process model backend seam plus a deterministic reference implementation
 * (an n-gram / Markov policy over action tokens) so that the full
 * capture → dataset → train → infer → evaluate loop can be exercised and tested
 * without any real OS input or GPU.
 *
 * A real on-device small model can implement {@link MovementModelBackend}
 * without touching call sites; the Markov backend documents the seam.
 */

/** A single recorded movement/action reduced to its learnable fields. */
export type MovementActionToken = {
  tool: string;
  summary: string;
};

/**
 * One trajectory's worth of movements: the ordered actions plus the observation
 * sources that were in view (kept as lightweight context for generalization).
 */
export type MovementSequence = {
  trajectoryId: string;
  contextSources: string[];
  actions: MovementActionToken[];
};

/** The replayable dataset a backend trains on. */
export type MovementDataset = {
  version: 1;
  sequences: MovementSequence[];
  /** Sorted, de-duplicated action-token keys (`toolsummary`). */
  vocabulary: string[];
};

export type MovementPrediction =
  | { kind: "action"; tool: string; summary: string; probability: number; backoffOrder: number }
  | { kind: "end"; probability: number; backoffOrder: number };

export type MovementGenerateOptions = {
  /** Hard cap on generated actions, guarding against non-terminating policies. */
  maxSteps?: number;
};

export type MovementGenerationResult = {
  actions: MovementActionToken[];
  /** True when the model itself emitted the end-of-sequence sentinel. */
  terminatedByModel: boolean;
  steps: number;
};

export type MovementFidelityReport = {
  referenceLength: number;
  matchedSteps: number;
  /** matchedSteps / referenceLength, in [0, 1]; 1 for an empty reference. */
  fidelity: number;
  mismatches: Array<{
    index: number;
    expected: MovementActionToken;
    predicted: MovementPrediction | undefined;
  }>;
};

export type SerializedMovementTransition = {
  order: number;
  context: string;
  next: string;
  count: number;
};

export type SerializedMovementModel = {
  version: 1;
  backendId: string;
  order: number;
  vocabulary: string[];
  transitions: SerializedMovementTransition[];
};

/** A trained, in-process movement policy. */
export interface TrainedMovementModel {
  readonly backendId: string;
  readonly order: number;
  readonly vocabulary: string[];
  /**
   * Predict the next action that follows `context` (the actions produced so
   * far, empty for the first step). Undefined when the model has learned
   * nothing that applies even after backing off to the unigram.
   */
  predictNext(context: MovementActionToken[]): MovementPrediction | undefined;
  /** Roll out a full movement sequence from an empty seed. */
  generate(options?: MovementGenerateOptions): MovementGenerationResult;
  /**
   * Teacher-forced replay fidelity: for each position, feed the true prefix and
   * check whether the model's top prediction matches the recorded action.
   */
  scoreFidelity(reference: MovementActionToken[]): MovementFidelityReport;
  toJSON(): SerializedMovementModel;
}

export type MovementTrainOptions = {
  /** Markov context length (number of preceding actions conditioned on). */
  order?: number;
};

/** The pluggable seam: swap in a real on-device model without changing callers. */
export interface MovementModelBackend {
  readonly id: string;
  train(dataset: MovementDataset, options?: MovementTrainOptions): Promise<TrainedMovementModel>;
}

const START = "START";
const END = "END";
const CTX_SEP = "";
const TOK_SEP = "";

function tokenKey(action: MovementActionToken): string {
  return `${action.tool}${TOK_SEP}${action.summary}`;
}

function parseTokenKey(key: string): MovementActionToken {
  const separator = key.indexOf(TOK_SEP);
  if (separator === -1) {
    return { tool: key, summary: "" };
  }
  return { tool: key.slice(0, separator), summary: key.slice(separator + 1) };
}

/**
 * Reduce one or more replay manifests into a movement dataset: actions grouped
 * by trajectory in timestamp order, with observation sources retained as
 * context. Transcript events are ignored — they are not movements.
 */
export function buildMovementDataset(manifests: ReplayManifest[]): MovementDataset {
  type Draft = {
    sources: Array<{ ts: number; source: string }>;
    actions: Array<{ ts: number; tool: string; summary: string }>;
  };
  const drafts = new Map<string, Draft>();
  const order: string[] = [];

  const draftFor = (trajectoryId: string): Draft => {
    let draft = drafts.get(trajectoryId);
    if (!draft) {
      draft = { sources: [], actions: [] };
      drafts.set(trajectoryId, draft);
      order.push(trajectoryId);
    }
    return draft;
  };

  for (const manifest of manifests) {
    for (const event of manifest.events) {
      if (event.kind === "observation") {
        draftFor(event.trajectoryId).sources.push({ ts: event.ts, source: event.source });
      } else if (event.kind === "action") {
        draftFor(event.trajectoryId).actions.push({ ts: event.ts, tool: event.tool, summary: event.summary });
      }
    }
  }

  const vocabulary = new Set<string>();
  const sequences: MovementSequence[] = order.map((trajectoryId) => {
    const draft = drafts.get(trajectoryId)!;
    const actions = [...draft.actions]
      .sort((a, b) => a.ts - b.ts)
      .map(({ tool, summary }) => {
        const action = { tool, summary };
        vocabulary.add(tokenKey(action));
        return action;
      });
    const contextSources = [...draft.sources].sort((a, b) => a.ts - b.ts).map((entry) => entry.source);
    return { trajectoryId, contextSources, actions };
  });

  return {
    version: 1,
    sequences,
    vocabulary: [...vocabulary].sort(),
  };
}

type TransitionTable = Map<number, Map<string, Map<string, number>>>;

class InMemoryMarkovModel implements TrainedMovementModel {
  constructor(
    readonly backendId: string,
    readonly order: number,
    readonly vocabulary: string[],
    private readonly transitions: TransitionTable,
  ) {}

  predictNext(context: MovementActionToken[]): MovementPrediction | undefined {
    const tokens = [START, ...context.map(tokenKey)];
    const maxK = Math.min(this.order, tokens.length);
    for (let k = maxK; k >= 0; k -= 1) {
      const contextKey = tokens.slice(tokens.length - k).join(CTX_SEP);
      const table = this.transitions.get(k)?.get(contextKey);
      if (!table || table.size === 0) {
        continue;
      }
      const best = argmaxToken(table);
      const total = totalCount(table);
      const probability = total === 0 ? 0 : best.count / total;
      const backoffOrder = this.order - k;
      if (best.token === END) {
        return { kind: "end", probability, backoffOrder };
      }
      const action = parseTokenKey(best.token);
      return { kind: "action", tool: action.tool, summary: action.summary, probability, backoffOrder };
    }
    return undefined;
  }

  generate(options: MovementGenerateOptions = {}): MovementGenerationResult {
    const maxSteps = options.maxSteps ?? 64;
    const actions: MovementActionToken[] = [];
    let terminatedByModel = false;
    for (let step = 0; step < maxSteps; step += 1) {
      const prediction = this.predictNext(actions);
      if (!prediction) {
        break;
      }
      if (prediction.kind === "end") {
        terminatedByModel = true;
        break;
      }
      actions.push({ tool: prediction.tool, summary: prediction.summary });
    }
    return { actions, terminatedByModel, steps: actions.length };
  }

  scoreFidelity(reference: MovementActionToken[]): MovementFidelityReport {
    const mismatches: MovementFidelityReport["mismatches"] = [];
    let matchedSteps = 0;
    for (let index = 0; index < reference.length; index += 1) {
      const expected = reference[index]!;
      const prediction = this.predictNext(reference.slice(0, index));
      if (
        prediction?.kind === "action" &&
        prediction.tool === expected.tool &&
        prediction.summary === expected.summary
      ) {
        matchedSteps += 1;
      } else {
        mismatches.push({ index, expected, predicted: prediction });
      }
    }
    const fidelity = reference.length === 0 ? 1 : matchedSteps / reference.length;
    return { referenceLength: reference.length, matchedSteps, fidelity, mismatches };
  }

  toJSON(): SerializedMovementModel {
    const transitions: SerializedMovementTransition[] = [];
    for (const [order, contexts] of this.transitions) {
      for (const [context, nextTable] of contexts) {
        for (const [next, count] of nextTable) {
          transitions.push({ order, context, next, count });
        }
      }
    }
    transitions.sort(
      (a, b) => a.order - b.order || a.context.localeCompare(b.context) || a.next.localeCompare(b.next),
    );
    return {
      version: 1,
      backendId: this.backendId,
      order: this.order,
      vocabulary: [...this.vocabulary],
      transitions,
    };
  }
}

function argmaxToken(table: Map<string, number>): { token: string; count: number } {
  let bestToken = "";
  let bestCount = -1;
  for (const [token, count] of table) {
    if (count > bestCount || (count === bestCount && token < bestToken)) {
      bestToken = token;
      bestCount = count;
    }
  }
  return { token: bestToken, count: bestCount };
}

function totalCount(table: Map<string, number>): number {
  let total = 0;
  for (const count of table.values()) {
    total += count;
  }
  return total;
}

/**
 * Deterministic reference backend: an order-N Markov policy over action tokens
 * with stupid-backoff to shorter contexts, so it both reproduces recorded
 * movements exactly and generalizes to novel-but-related contexts by falling
 * back on the most frequent shorter-context continuation.
 */
export class MarkovMovementBackend implements MovementModelBackend {
  readonly id = "markov-movement-v1";

  constructor(private readonly defaultOrder = 2) {}

  async train(dataset: MovementDataset, options: MovementTrainOptions = {}): Promise<TrainedMovementModel> {
    const order = Math.max(0, options.order ?? this.defaultOrder);
    const transitions: TransitionTable = new Map();

    const record = (k: number, contextKey: string, next: string): void => {
      let contexts = transitions.get(k);
      if (!contexts) {
        contexts = new Map();
        transitions.set(k, contexts);
      }
      let nextTable = contexts.get(contextKey);
      if (!nextTable) {
        nextTable = new Map();
        contexts.set(contextKey, nextTable);
      }
      nextTable.set(next, (nextTable.get(next) ?? 0) + 1);
    };

    for (const sequence of dataset.sequences) {
      const tokens = [START, ...sequence.actions.map(tokenKey), END];
      for (let position = 1; position < tokens.length; position += 1) {
        const next = tokens[position]!;
        const maxK = Math.min(order, position);
        for (let k = 0; k <= maxK; k += 1) {
          const contextKey = tokens.slice(position - k, position).join(CTX_SEP);
          record(k, contextKey, next);
        }
      }
    }

    return new InMemoryMarkovModel(this.id, order, [...dataset.vocabulary], transitions);
  }
}

/**
 * Convenience: build a dataset from replay manifests and train a model in one
 * call. The backend is injectable — defaults to the deterministic Markov
 * reference so cloud/CI runs stay hermetic.
 */
export async function trainMovementModelFromReplays(
  manifests: ReplayManifest[],
  backend: MovementModelBackend = new MarkovMovementBackend(),
  options?: MovementTrainOptions,
): Promise<{ dataset: MovementDataset; model: TrainedMovementModel }> {
  const dataset = buildMovementDataset(manifests);
  const model = await backend.train(dataset, options);
  return { dataset, model };
}
