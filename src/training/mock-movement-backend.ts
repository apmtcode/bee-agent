import type {
  MovementBackendCapabilities,
  MovementDataset,
  MovementPrediction,
  MovementSequence,
  MovementStep,
  MovementTrainingBackend,
  SerializedMovementModel,
  TrainMovementOptions,
  TrainedMovementModel,
} from "./movement-backend.js";

/**
 * Deterministic, dependency-free movement backend.
 *
 * It "post-trains" by fitting a first-order transition model (an n-gram over
 * action summaries) plus a context index, so the resulting model can:
 *  - replay recorded trajectories exactly,
 *  - predict the next step from the previous action (transition) or a goal cue
 *    (context), and
 *  - generalize to unseen-but-related goals by retrieving the nearest recorded
 *    sequence via token-set (Jaccard) similarity.
 *
 * Being fully in-process and deterministic, it validates the whole
 * capture -> dataset -> train -> infer pipeline in the cloud/CI with no OS or
 * native-model access. A real on-device backend (e.g. MLX) implements the same
 * {@link MovementTrainingBackend} seam.
 */

export const MOCK_MOVEMENT_BACKEND_ID = "mock-deterministic";

type Successor = { step: MovementStep; count: number };

type MockPayload = {
  sequences: MovementSequence[];
  /** normalized last-summary -> ranked successor steps */
  transitions: Record<string, Successor[]>;
  /** ranked first-steps across all sequences */
  firstSteps: Successor[];
};

const STOPWORDS = new Set([
  "the",
  "a",
  "an",
  "on",
  "in",
  "to",
  "of",
  "and",
  "or",
  "into",
  "with",
  "at",
  "is",
  "active",
  "device",
]);

export class MockMovementBackend implements MovementTrainingBackend {
  readonly id = MOCK_MOVEMENT_BACKEND_ID;
  readonly description = "Deterministic in-process transition + retrieval learner (no OS access required)";
  readonly capabilities: MovementBackendCapabilities = {
    onDevice: false,
    deterministic: true,
    generalizes: true,
  };

  async train(dataset: MovementDataset, _options?: TrainMovementOptions): Promise<TrainedMovementModel> {
    const transitions = new Map<string, Map<string, Successor>>();
    const firstSteps = new Map<string, Successor>();

    for (const sequence of dataset.sequences) {
      const first = sequence.steps[0];
      if (first) {
        bump(firstSteps, normalize(first.summary), first);
      }
      for (let index = 0; index < sequence.steps.length - 1; index += 1) {
        const current = sequence.steps[index];
        const next = sequence.steps[index + 1];
        const key = normalize(current.summary);
        const table = transitions.get(key) ?? new Map<string, Successor>();
        bump(table, normalize(next.summary), next);
        transitions.set(key, table);
      }
    }

    const payload: MockPayload = {
      sequences: dataset.sequences,
      transitions: rankAll(transitions),
      firstSteps: rank(firstSteps),
    };
    return new MockMovementModel(payload);
  }

  load(serialized: SerializedMovementModel): TrainedMovementModel {
    if (serialized.backendId !== this.id) {
      throw new Error(`cannot load model from backend ${serialized.backendId} into ${this.id}`);
    }
    return new MockMovementModel(serialized.payload as MockPayload);
  }
}

class MockMovementModel implements TrainedMovementModel {
  readonly backendId = MOCK_MOVEMENT_BACKEND_ID;
  readonly sequenceCount: number;
  readonly stepCount: number;
  private readonly byTrajectory: Map<string, MovementStep[]>;

  constructor(private readonly payload: MockPayload) {
    this.sequenceCount = payload.sequences.length;
    this.stepCount = payload.sequences.reduce((sum, sequence) => sum + sequence.steps.length, 0);
    this.byTrajectory = new Map(payload.sequences.map((sequence) => [sequence.trajectoryId, sequence.steps]));
  }

  replay(trajectoryId: string): MovementStep[] | undefined {
    const steps = this.byTrajectory.get(trajectoryId);
    return steps ? steps.map((step) => ({ ...step })) : undefined;
  }

  predictNext(input: { lastSummary?: string; context?: string }): MovementPrediction | undefined {
    if (input.lastSummary !== undefined) {
      const successors = this.payload.transitions[normalize(input.lastSummary)];
      if (successors && successors.length > 0) {
        const total = successors.reduce((sum, entry) => sum + entry.count, 0);
        const best = successors[0];
        return { step: { ...best.step }, confidence: best.count / total, source: "transition" };
      }
    }

    if (input.context !== undefined) {
      const match = this.bestSequenceByContext(input.context);
      if (match && match.sequence.steps[0]) {
        return {
          step: { ...match.sequence.steps[0] },
          confidence: match.similarity,
          source: "context",
        };
      }
    }

    if (this.payload.firstSteps.length > 0) {
      const total = this.payload.firstSteps.reduce((sum, entry) => sum + entry.count, 0);
      const best = this.payload.firstSteps[0];
      return { step: { ...best.step }, confidence: best.count / total, source: "fallback" };
    }
    return undefined;
  }

  generate(input: { context: string; maxSteps?: number }): MovementStep[] {
    const match = this.bestSequenceByContext(input.context);
    const sequence = match?.sequence ?? this.payload.sequences[0];
    if (!sequence) {
      return [];
    }
    const limit = input.maxSteps ?? sequence.steps.length;
    return sequence.steps.slice(0, Math.max(0, limit)).map((step) => ({ ...step }));
  }

  serialize(): SerializedMovementModel {
    return {
      version: 1,
      backendId: this.backendId,
      sequenceCount: this.sequenceCount,
      stepCount: this.stepCount,
      payload: this.payload,
    };
  }

  private bestSequenceByContext(
    context: string,
  ): { sequence: MovementSequence; similarity: number } | undefined {
    const queryTokens = tokenize(context);
    if (queryTokens.size === 0) {
      return undefined;
    }
    let best: { sequence: MovementSequence; similarity: number } | undefined;
    for (const sequence of this.payload.sequences) {
      const sequenceTokens = sequenceContextTokens(sequence);
      const similarity = jaccard(queryTokens, sequenceTokens);
      if (
        similarity > 0 &&
        (best === undefined ||
          similarity > best.similarity ||
          // Deterministic tie-break by trajectory id.
          (similarity === best.similarity && sequence.trajectoryId < best.sequence.trajectoryId))
      ) {
        best = { sequence, similarity };
      }
    }
    return best;
  }
}

function bump(table: Map<string, Successor>, key: string, step: MovementStep): void {
  const existing = table.get(key);
  if (existing) {
    existing.count += 1;
  } else {
    table.set(key, { step: { ...step }, count: 1 });
  }
}

function rank(table: Map<string, Successor>): Successor[] {
  return [...table.values()].sort(
    (a, b) => b.count - a.count || normalize(a.step.summary).localeCompare(normalize(b.step.summary)),
  );
}

function rankAll(transitions: Map<string, Map<string, Successor>>): Record<string, Successor[]> {
  const result: Record<string, Successor[]> = {};
  for (const [key, table] of transitions) {
    result[key] = rank(table);
  }
  return result;
}

function sequenceContextTokens(sequence: MovementSequence): Set<string> {
  const tokens = new Set<string>();
  for (const step of sequence.steps) {
    if (step.context) {
      for (const token of tokenize(step.context)) {
        tokens.add(token);
      }
    }
    for (const token of tokenize(step.summary)) {
      tokens.add(token);
    }
  }
  return tokens;
}

function tokenize(text: string): Set<string> {
  const tokens = new Set<string>();
  for (const raw of text.toLowerCase().match(/[a-z0-9]+/g) ?? []) {
    if (raw.length >= 2 && !STOPWORDS.has(raw)) {
      tokens.add(raw);
    }
  }
  return tokens;
}

function jaccard(a: Set<string>, b: Set<string>): number {
  if (a.size === 0 || b.size === 0) {
    return 0;
  }
  let intersection = 0;
  for (const token of a) {
    if (b.has(token)) {
      intersection += 1;
    }
  }
  const union = a.size + b.size - intersection;
  return union === 0 ? 0 : intersection / union;
}

function normalize(summary: string): string {
  return summary.trim().toLowerCase().replace(/\s+/g, " ");
}
