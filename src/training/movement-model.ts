import type { TrajectorySpan } from "../capture/trajectory.js";

/**
 * Pluggable local-model backend for the movement-learning subsystem.
 *
 * The capture pipeline records operator movements as {@link TrajectorySpan}s.
 * To *repeat* recorded movements and *generalize* to new-but-related ones we
 * need a trainable policy over movement sequences. Real on-device training runs
 * via {@link LocalAppleSiliconTrainingRunner} (mlx/axolotl) when the user runs
 * bee-agent locally; in the cloud we cannot train a neural net, so the backend
 * is an interface with a deterministic, dependency-free reference implementation
 * ({@link NgramMovementBackend}) that trains and infers purely in-process. This
 * lets the whole capture -> dataset -> train -> infer loop be validated with
 * synthetic event streams and unit tests, and keeps the model backend swappable
 * for a real small local model behind the same seam.
 */

/** A single recorded movement/action step (mouse, key, window, tool, ...). */
export type MovementStep = {
  /** Action verb, e.g. "mouse.click", "key.press", "window.focus". */
  tool: string;
  /** Human/agent-readable descriptor of the movement. */
  summary: string;
  metadata?: Record<string, unknown>;
};

/** An ordered movement sequence for one trajectory. */
export type MovementSequence = {
  trajectoryId: string;
  steps: MovementStep[];
};

/** The replayable dataset the backend trains on. */
export type MovementDataset = {
  version: 1;
  sequences: MovementSequence[];
};

export type MovementTrainOptions = {
  /**
   * Max context length (in prior steps) the model conditions on. Higher order
   * memorizes recorded movements more exactly; lower order generalizes more.
   * Defaults to 3.
   */
  order?: number;
};

/** A prediction for the next movement given some context. */
export type MovementPrediction = {
  step: MovementStep;
  /** Estimated probability of this step in [0, 1]. */
  probability: number;
  /** How many context steps were actually used (after back-off). */
  contextUsed: number;
  /** True when predicted from the full available context (no back-off needed). */
  exact: boolean;
};

/** A serialized model snapshot — portable across processes/backends. */
export type MovementModelSnapshot = {
  version: 1;
  backend: string;
  order: number;
  /** token key -> full step (vocabulary). */
  vocab: Record<string, MovementStep>;
  /** context key -> { next token key -> count }. */
  transitions: Record<string, Record<string, number>>;
};

/** A trained movement model: repeats recorded movements and generalizes. */
export interface MovementModel {
  readonly backend: string;
  readonly order: number;
  /** Predict the single most-likely next step for a context, or undefined. */
  predictNext(context: MovementStep[]): MovementPrediction | undefined;
  /** Full ranked distribution over next steps for a context. */
  predictDistribution(context: MovementStep[]): MovementPrediction[];
  /** Greedily roll out a movement sequence from a seed context. */
  generate(seed: MovementStep[], maxSteps: number): MovementStep[];
  serialize(): MovementModelSnapshot;
}

/** A pluggable backend that trains a {@link MovementModel} from a dataset. */
export interface MovementModelBackend {
  readonly name: string;
  train(dataset: MovementDataset, options?: MovementTrainOptions): Promise<MovementModel>;
  restore(snapshot: MovementModelSnapshot): MovementModel;
}

const STEP_KEY_SEP = "|";
const CONTEXT_KEY_SEP = "\n";

function stepKey(step: MovementStep): string {
  return `${step.tool}${STEP_KEY_SEP}${step.summary}`;
}

function contextKey(tokens: string[]): string {
  return tokens.join(CONTEXT_KEY_SEP);
}

/**
 * Build a {@link MovementDataset} from recorded trajectory spans. Actions are
 * ordered by timestamp so the sequence reflects real movement order. Only
 * approved trajectories should be passed when export policy requires review;
 * this function does not enforce that (the exporter does).
 */
export function buildMovementDataset(spans: TrajectorySpan[]): MovementDataset {
  const sequences: MovementSequence[] = spans
    .map((span) => ({
      trajectoryId: span.id,
      steps: [...span.actions]
        .sort((a, b) => a.ts - b.ts)
        .map<MovementStep>((action) => ({
          tool: action.tool,
          summary: action.summary,
          ...(action.metadata ? { metadata: action.metadata } : {}),
        })),
    }))
    .filter((sequence) => sequence.steps.length > 0);
  return { version: 1, sequences };
}

/**
 * Deterministic, dependency-free reference backend: a variable-order n-gram
 * policy with stupid-backoff over fixed-length movement windows. It memorizes
 * recorded transitions (enabling exact replay) and falls back to shorter
 * windows for unseen prefixes (enabling generalization to new-but-related
 * movements). Ordering of equal-count candidates is stable (by insertion), so
 * training and inference are fully reproducible — no RNG, safe for cloud tests.
 */
export class NgramMovementBackend implements MovementModelBackend {
  readonly name = "ngram-mock";

  async train(dataset: MovementDataset, options: MovementTrainOptions = {}): Promise<MovementModel> {
    const order = Math.max(1, Math.floor(options.order ?? 3));
    const vocab = new Map<string, MovementStep>();
    // context key -> ordered map of nextKey -> count
    const transitions = new Map<string, Map<string, number>>();

    const record = (contextTokens: string[], nextKey: string): void => {
      const key = contextKey(contextTokens);
      let row = transitions.get(key);
      if (!row) {
        row = new Map<string, number>();
        transitions.set(key, row);
      }
      row.set(nextKey, (row.get(nextKey) ?? 0) + 1);
    };

    for (const sequence of dataset.sequences) {
      const tokens: string[] = [];
      for (const step of sequence.steps) {
        const key = stepKey(step);
        if (!vocab.has(key)) {
          vocab.set(key, step);
        }
        tokens.push(key);
      }
      // For each position, record windows of length 0..min(order, position) so
      // the model can back off from the highest available order to the unigram.
      for (let i = 0; i < tokens.length; i += 1) {
        const nextKey = tokens[i]!;
        const maxCtx = Math.min(order, i);
        for (let ctx = 0; ctx <= maxCtx; ctx += 1) {
          record(tokens.slice(i - ctx, i), nextKey);
        }
      }
    }

    return new NgramMovementModel(this.name, order, vocab, transitions);
  }

  restore(snapshot: MovementModelSnapshot): MovementModel {
    const vocab = new Map<string, MovementStep>(Object.entries(snapshot.vocab));
    const transitions = new Map<string, Map<string, number>>();
    for (const [key, row] of Object.entries(snapshot.transitions)) {
      transitions.set(key, new Map<string, number>(Object.entries(row)));
    }
    return new NgramMovementModel(snapshot.backend, snapshot.order, vocab, transitions);
  }
}

class NgramMovementModel implements MovementModel {
  constructor(
    readonly backend: string,
    readonly order: number,
    private readonly vocab: Map<string, MovementStep>,
    private readonly transitions: Map<string, Map<string, number>>,
  ) {}

  predictNext(context: MovementStep[]): MovementPrediction | undefined {
    return this.predictDistribution(context)[0];
  }

  predictDistribution(context: MovementStep[]): MovementPrediction[] {
    const tokens = context.map(stepKey);
    const maxCtx = Math.min(this.order, tokens.length);
    // Back off from the longest available window to the unigram.
    for (let ctx = maxCtx; ctx >= 0; ctx -= 1) {
      const window = tokens.slice(tokens.length - ctx);
      const row = this.transitions.get(contextKey(window));
      if (!row || row.size === 0) {
        continue;
      }
      const total = [...row.values()].reduce((sum, count) => sum + count, 0);
      const exact = ctx === maxCtx && maxCtx > 0;
      return [...row.entries()]
        .sort((a, b) => b[1] - a[1])
        .map<MovementPrediction>(([nextKey, count]) => ({
          step: this.vocab.get(nextKey)!,
          probability: count / total,
          contextUsed: ctx,
          exact,
        }));
    }
    return [];
  }

  generate(seed: MovementStep[], maxSteps: number): MovementStep[] {
    const out: MovementStep[] = [];
    const context = [...seed];
    for (let i = 0; i < maxSteps; i += 1) {
      const prediction = this.predictNext(context);
      if (!prediction) {
        break;
      }
      out.push(prediction.step);
      context.push(prediction.step);
    }
    return out;
  }

  serialize(): MovementModelSnapshot {
    const vocab: Record<string, MovementStep> = {};
    for (const [key, step] of this.vocab) {
      vocab[key] = step;
    }
    const transitions: Record<string, Record<string, number>> = {};
    for (const [key, row] of this.transitions) {
      const rowObj: Record<string, number> = {};
      for (const [nextKey, count] of row) {
        rowObj[nextKey] = count;
      }
      transitions[key] = rowObj;
    }
    return { version: 1, backend: this.backend, order: this.order, vocab, transitions };
  }
}

export type MovementEvalReport = {
  /** Total next-step predictions attempted across all held-out sequences. */
  predictions: number;
  /** Predictions whose top candidate exactly matched the recorded next step. */
  correct: number;
  /** correct / predictions in [0, 1]; 0 when there were no predictions. */
  accuracy: number;
  /** Fraction of correct predictions that came from the full (exact) context. */
  exactMatchRate: number;
  /** Fraction of correct predictions that required back-off (generalization). */
  generalizedRate: number;
};

/**
 * Measure how faithfully a model reproduces held-out movement sequences by
 * teacher-forced next-step prediction: for each step, feed the true prefix and
 * check whether the model's top prediction matches the recorded next step. The
 * split of correct predictions into exact vs. generalized (backed-off) tells us
 * whether fidelity comes from memorization or from generalizing to related
 * prefixes — the two capabilities objective 2 requires.
 */
export function evaluateMovementModel(
  model: MovementModel,
  heldOut: MovementSequence[],
): MovementEvalReport {
  let predictions = 0;
  let correct = 0;
  let correctExact = 0;
  for (const sequence of heldOut) {
    for (let i = 0; i < sequence.steps.length; i += 1) {
      const prediction = model.predictNext(sequence.steps.slice(0, i));
      predictions += 1;
      if (prediction && stepKey(prediction.step) === stepKey(sequence.steps[i]!)) {
        correct += 1;
        if (prediction.exact) {
          correctExact += 1;
        }
      }
    }
  }
  return {
    predictions,
    correct,
    accuracy: predictions === 0 ? 0 : correct / predictions,
    exactMatchRate: correct === 0 ? 0 : correctExact / correct,
    generalizedRate: correct === 0 ? 0 : (correct - correctExact) / correct,
  };
}
