import type { TrajectorySpan } from "../capture/trajectory.js";

/**
 * Local-movement learning subsystem — in-process model layer.
 *
 * Standing objective #2 (c) "post-train a local model on the recorded dataset to
 * repeat the recorded movements" and (d) "generalize to perform new but related
 * movements".
 *
 * The heavyweight, on-device training path (mlx/axolotl launch scripts) lives in
 * {@link ./runner.ts}. That path can only run on the user's real machine. This
 * module provides the *pluggable backend seam* plus a fully deterministic,
 * dependency-free reference backend so the whole learn → repeat → generalize loop
 * can be exercised and validated in the cloud with synthetic movement streams.
 *
 * Everything here is pure: no filesystem, no clock, no randomness. That keeps the
 * model reproducible (the same dataset always yields the same weights) and makes
 * the eval harness a reliable regression signal.
 */

/** A single tokenized movement/action, the atomic unit the model learns over. */
export type MovementStep = {
  /** Gesture / action class, e.g. "tap", "swipe", "type", "scroll", "shortcut". */
  gesture: string;
  /** UI target the gesture acted on, e.g. "Save button". Optional for gestures like scroll. */
  target?: string;
  /** Spatial direction for swipe/scroll gestures. */
  direction?: "up" | "down" | "left" | "right";
  /** Free-text summary of the value entered (excluded from step identity so text can vary). */
  valueSummary?: string;
};

/** An ordered movement sequence for one trajectory/workflow, the training example. */
export type MovementSequence = {
  id: string;
  /** Optional workflow label, used only for grouping/debugging — never a model input. */
  label?: string;
  steps: MovementStep[];
};

export type MovementDataset = {
  sequences: MovementSequence[];
};

export type MovementBackoff = "exact" | "context" | "class" | "unigram" | "none";

export type MovementPrediction = {
  /** Best predicted next step (undefined only for the "none" backoff on an empty model). */
  step?: MovementStep;
  /** Estimated probability of the predicted step given the context. */
  confidence: number;
  /**
   * Which resolution produced the prediction:
   * - `exact`   full-order specific context matched (repeats a recorded movement),
   * - `context` a shorter specific context matched,
   * - `class`   a gesture-class context matched (generalization to a new target),
   * - `unigram` global most-frequent fallback,
   * - `none`    the model is empty.
   */
  backoff: MovementBackoff;
};

export type MovementTrainOptions = {
  /** Markov context length. Higher = more faithful repetition, less generalization. Default 2. */
  order?: number;
};

/**
 * Pluggable training backend. Real on-device backends (mlx/axolotl, llama.cpp,
 * a small transformer) implement this same shape; {@link MarkovMovementBackend}
 * is the deterministic reference used in tests and cloud runs.
 */
export interface MovementModelBackend {
  readonly name: string;
  train(dataset: MovementDataset, options?: MovementTrainOptions): MovementModel;
}

/** A trained model: repeat recorded movements and generalize to related ones. */
export interface MovementModel {
  readonly backend: string;
  readonly order: number;
  /** Predict the single most-likely next step given a context of prior steps. */
  predictNext(context: MovementStep[]): MovementPrediction;
  /** Autoregressively generate up to `count` steps starting from `context`. */
  generate(context: MovementStep[], count: number): MovementStep[];
  serialize(): SerializedMovementModel;
}

export type SerializedMovementModel = {
  version: 1;
  backend: string;
  order: number;
  /** specific[order-1] : contextKey -> (nextStepKey -> count) */
  specific: Array<Record<string, Record<string, number>>>;
  /** classSpecific[order-1] : classContextKey -> (nextClass -> count) */
  classSpecific: Array<Record<string, Record<string, number>>>;
  /** Reconstruct a full step from its identity key. */
  stepByKey: Record<string, MovementStep>;
  /** Most frequent exemplar step per gesture class (for class backoff reconstruction). */
  exemplarByClass: Record<string, MovementStep>;
  /** Global next-step counts. */
  unigram: Record<string, number>;
};

const CONTEXT_SEP = ">>";

function stepKey(step: MovementStep): string {
  return JSON.stringify({ g: step.gesture, t: step.target ?? null, d: step.direction ?? null });
}

function stepClass(step: MovementStep): string {
  return step.gesture;
}

function contextKey(steps: MovementStep[], order: number): string {
  return steps.slice(-order).map(stepKey).join(CONTEXT_SEP);
}

function classContextKey(steps: MovementStep[], order: number): string {
  return steps.slice(-order).map(stepClass).join(CONTEXT_SEP);
}

/** Deterministic argmax over a count map: highest count, ties broken by smallest key. */
function argmax(counts: Record<string, number>): { key: string; count: number; total: number } | undefined {
  let bestKey: string | undefined;
  let bestCount = -1;
  let total = 0;
  for (const key of Object.keys(counts).sort()) {
    const count = counts[key] ?? 0;
    total += count;
    if (count > bestCount) {
      bestCount = count;
      bestKey = key;
    }
  }
  if (bestKey === undefined) {
    return undefined;
  }
  return { key: bestKey, count: bestCount, total };
}

class MarkovMovementModel implements MovementModel {
  constructor(
    readonly backend: string,
    readonly order: number,
    private readonly specific: Array<Map<string, Map<string, number>>>,
    private readonly classSpecific: Array<Map<string, Map<string, number>>>,
    private readonly stepByKey: Map<string, MovementStep>,
    private readonly exemplarByClass: Map<string, MovementStep>,
    private readonly unigram: Map<string, number>,
  ) {}

  predictNext(context: MovementStep[]): MovementPrediction {
    // 1. Specific context, most specific order first (repeat recorded movements).
    for (let order = this.order; order >= 1; order -= 1) {
      if (context.length < order) {
        continue;
      }
      const table = this.specific[order - 1];
      const counts = table?.get(contextKey(context, order));
      if (counts && counts.size > 0) {
        const best = argmax(Object.fromEntries(counts));
        if (best) {
          const step = this.stepByKey.get(best.key);
          if (step) {
            return {
              step,
              confidence: best.total > 0 ? best.count / best.total : 0,
              backoff: order === this.order ? "exact" : "context",
            };
          }
        }
      }
    }

    // 2. Gesture-class context (generalize: same gesture pattern, new target).
    for (let order = this.order; order >= 1; order -= 1) {
      if (context.length < order) {
        continue;
      }
      const table = this.classSpecific[order - 1];
      const counts = table?.get(classContextKey(context, order));
      if (counts && counts.size > 0) {
        const best = argmax(Object.fromEntries(counts));
        if (best) {
          const step = this.exemplarByClass.get(best.key);
          if (step) {
            return { step, confidence: best.total > 0 ? best.count / best.total : 0, backoff: "class" };
          }
        }
      }
    }

    // 3. Global unigram fallback.
    const uni = argmax(Object.fromEntries(this.unigram));
    if (uni) {
      const step = this.stepByKey.get(uni.key);
      if (step) {
        return { step, confidence: uni.total > 0 ? uni.count / uni.total : 0, backoff: "unigram" };
      }
    }

    return { confidence: 0, backoff: "none" };
  }

  generate(context: MovementStep[], count: number): MovementStep[] {
    const working = [...context];
    const generated: MovementStep[] = [];
    for (let i = 0; i < count; i += 1) {
      const prediction = this.predictNext(working);
      if (!prediction.step) {
        break;
      }
      generated.push(prediction.step);
      working.push(prediction.step);
    }
    return generated;
  }

  serialize(): SerializedMovementModel {
    const dumpTables = (tables: Array<Map<string, Map<string, number>>>) =>
      tables.map((table) => {
        const out: Record<string, Record<string, number>> = {};
        for (const [ctx, counts] of table) {
          out[ctx] = Object.fromEntries(counts);
        }
        return out;
      });
    return {
      version: 1,
      backend: this.backend,
      order: this.order,
      specific: dumpTables(this.specific),
      classSpecific: dumpTables(this.classSpecific),
      stepByKey: Object.fromEntries(this.stepByKey),
      exemplarByClass: Object.fromEntries(this.exemplarByClass),
      unigram: Object.fromEntries(this.unigram),
    };
  }
}

/**
 * Deterministic order-k Markov backend with specific → class → unigram backoff.
 *
 * The backoff ladder is what buys generalization: an exact context reproduces a
 * recorded workflow step-for-step, while an unseen context whose *gesture shape*
 * was seen before still predicts the right next gesture (against a new target).
 */
export class MarkovMovementBackend implements MovementModelBackend {
  readonly name = "markov";

  train(dataset: MovementDataset, options?: MovementTrainOptions): MovementModel {
    const order = Math.max(1, Math.trunc(options?.order ?? 2));
    const specific: Array<Map<string, Map<string, number>>> = Array.from({ length: order }, () => new Map());
    const classSpecific: Array<Map<string, Map<string, number>>> = Array.from({ length: order }, () => new Map());
    const stepByKey = new Map<string, MovementStep>();
    const exemplarCounts = new Map<string, Map<string, number>>();
    const unigram = new Map<string, number>();

    const bump = (table: Map<string, Map<string, number>>, ctx: string, next: string) => {
      let counts = table.get(ctx);
      if (!counts) {
        counts = new Map();
        table.set(ctx, counts);
      }
      counts.set(next, (counts.get(next) ?? 0) + 1);
    };

    for (const sequence of dataset.sequences) {
      const steps = sequence.steps;
      for (let i = 0; i < steps.length; i += 1) {
        const step = steps[i];
        if (!step) {
          continue;
        }
        const key = stepKey(step);
        stepByKey.set(key, cloneStep(step));
        unigram.set(key, (unigram.get(key) ?? 0) + 1);

        const cls = stepClass(step);
        let byExemplar = exemplarCounts.get(cls);
        if (!byExemplar) {
          byExemplar = new Map();
          exemplarCounts.set(cls, byExemplar);
        }
        byExemplar.set(key, (byExemplar.get(key) ?? 0) + 1);

        const prior = steps.slice(0, i);
        for (let order_ = 1; order_ <= order; order_ += 1) {
          if (prior.length < order_) {
            break;
          }
          bump(specific[order_ - 1]!, contextKey(prior, order_), key);
          bump(classSpecific[order_ - 1]!, classContextKey(prior, order_), cls);
        }
      }
    }

    const exemplarByClass = new Map<string, MovementStep>();
    for (const [cls, counts] of exemplarCounts) {
      const best = argmax(Object.fromEntries(counts));
      const step = best ? stepByKey.get(best.key) : undefined;
      if (step) {
        exemplarByClass.set(cls, cloneStep(step));
      }
    }

    return new MarkovMovementModel(this.name, order, specific, classSpecific, stepByKey, exemplarByClass, unigram);
  }
}

/** Rehydrate a model persisted with {@link MovementModel.serialize}. */
export function loadMovementModel(serialized: SerializedMovementModel): MovementModel {
  const toTables = (raw: Array<Record<string, Record<string, number>>>) =>
    raw.map((table) => {
      const map = new Map<string, Map<string, number>>();
      for (const [ctx, counts] of Object.entries(table)) {
        map.set(ctx, new Map(Object.entries(counts)));
      }
      return map;
    });
  return new MarkovMovementModel(
    serialized.backend,
    serialized.order,
    toTables(serialized.specific),
    toTables(serialized.classSpecific),
    new Map(Object.entries(serialized.stepByKey)),
    new Map(Object.entries(serialized.exemplarByClass)),
    new Map(Object.entries(serialized.unigram)),
  );
}

function cloneStep(step: MovementStep): MovementStep {
  return {
    gesture: step.gesture,
    ...(step.target !== undefined ? { target: step.target } : {}),
    ...(step.direction !== undefined ? { direction: step.direction } : {}),
    ...(step.valueSummary !== undefined ? { valueSummary: step.valueSummary } : {}),
  };
}

/**
 * Derive a training example from a recorded trajectory span. Reads the structured
 * gesture metadata written by the device/browser/OS adapters when present, and
 * falls back to the action's tool + summary otherwise.
 */
export function tokenizeTrajectory(trajectory: TrajectorySpan): MovementSequence {
  const steps: MovementStep[] = trajectory.actions.map((action) => {
    const metadata = action.metadata ?? {};
    const gesture = typeof metadata.gesture === "string" ? metadata.gesture : action.tool;
    const target = typeof metadata.target === "string" ? metadata.target : undefined;
    const direction = isDirection(metadata.direction) ? metadata.direction : undefined;
    const valueSummary = typeof metadata.valueSummary === "string" ? metadata.valueSummary : action.summary;
    return {
      gesture,
      ...(target !== undefined ? { target } : {}),
      ...(direction !== undefined ? { direction } : {}),
      ...(valueSummary !== undefined ? { valueSummary } : {}),
    };
  });
  return { id: trajectory.id, steps };
}

function isDirection(value: unknown): value is MovementStep["direction"] {
  return value === "up" || value === "down" || value === "left" || value === "right";
}

// --- Eval harness -----------------------------------------------------------

export type MovementEvalOptions = {
  /** Minimum prior-context length before a position is scored. Default 1. */
  minContext?: number;
};

export type MovementEvalResult = {
  /** Number of next-step positions scored. */
  predictions: number;
  exactMatches: number;
  gestureMatches: number;
  /** Fraction of positions where the full step (gesture+target+direction) matched. */
  exactAccuracy: number;
  /** Fraction where at least the gesture class matched — the generalization signal. */
  gestureAccuracy: number;
  /** Sequences reproduced end-to-end from their first step (repeat fidelity). */
  fullyReproduced: number;
  sequences: number;
  /** How often each backoff level was used across all predictions. */
  backoffBreakdown: Record<MovementBackoff, number>;
};

/**
 * Measure replay fidelity + generalization of a model on a held-out dataset.
 *
 * `exactAccuracy` captures faithful repetition; `gestureAccuracy` captures
 * generalization to new-but-related movements (right gesture, possibly new
 * target). `fullyReproduced` is the strict end-to-end replay score.
 */
export function evaluateMovementModel(
  model: MovementModel,
  dataset: MovementDataset,
  options?: MovementEvalOptions,
): MovementEvalResult {
  const minContext = Math.max(1, Math.trunc(options?.minContext ?? 1));
  const backoffBreakdown: Record<MovementBackoff, number> = {
    exact: 0,
    context: 0,
    class: 0,
    unigram: 0,
    none: 0,
  };
  let predictions = 0;
  let exactMatches = 0;
  let gestureMatches = 0;
  let fullyReproduced = 0;

  for (const sequence of dataset.sequences) {
    const steps = sequence.steps;

    // Next-step scoring across the sequence.
    for (let i = minContext; i < steps.length; i += 1) {
      const actual = steps[i];
      if (!actual) {
        continue;
      }
      const prediction = model.predictNext(steps.slice(0, i));
      predictions += 1;
      backoffBreakdown[prediction.backoff] += 1;
      if (prediction.step) {
        if (stepKey(prediction.step) === stepKey(actual)) {
          exactMatches += 1;
        }
        if (prediction.step.gesture === actual.gesture) {
          gestureMatches += 1;
        }
      }
    }

    // Strict end-to-end replay: seed with the first step, regenerate the rest.
    if (steps.length > 1) {
      const generated = model.generate(steps.slice(0, 1), steps.length - 1);
      const matched =
        generated.length === steps.length - 1 &&
        generated.every((step, index) => stepKey(step) === stepKey(steps[index + 1]!));
      if (matched) {
        fullyReproduced += 1;
      }
    }
  }

  return {
    predictions,
    exactMatches,
    gestureMatches,
    exactAccuracy: predictions > 0 ? exactMatches / predictions : 0,
    gestureAccuracy: predictions > 0 ? gestureMatches / predictions : 0,
    fullyReproduced,
    sequences: dataset.sequences.length,
    backoffBreakdown,
  };
}

// --- Synthetic movement generator -------------------------------------------

/**
 * A workflow template: a fixed gesture skeleton whose `{slot}` targets are filled
 * from a per-variant substitution map. Substituting different targets into the
 * same skeleton yields "new but related" movements — exactly the generalization
 * the model should handle.
 */
export type MovementWorkflowTemplate = {
  label: string;
  steps: Array<Omit<MovementStep, "target"> & { target?: string }>;
  /** Ordered list of target substitution maps, one per variant. */
  variants: Array<Record<string, string>>;
};

export type SyntheticDatasetOptions = {
  templates: MovementWorkflowTemplate[];
};

/**
 * Deterministically expand workflow templates × variants into a dataset. No
 * randomness — the same options always produce byte-identical sequences, so the
 * generator doubles as fixture-builder for tests and the eval harness.
 */
export function generateSyntheticMovementDataset(options: SyntheticDatasetOptions): MovementDataset {
  const sequences: MovementSequence[] = [];
  for (const template of options.templates) {
    template.variants.forEach((substitutions, variantIndex) => {
      const steps: MovementStep[] = template.steps.map((step) => {
        const target = step.target ? substituteSlots(step.target, substitutions) : undefined;
        return {
          gesture: step.gesture,
          ...(target !== undefined ? { target } : {}),
          ...(step.direction !== undefined ? { direction: step.direction } : {}),
          ...(step.valueSummary !== undefined ? { valueSummary: step.valueSummary } : {}),
        };
      });
      sequences.push({ id: `${template.label}-${variantIndex}`, label: template.label, steps });
    });
  }
  return { sequences };
}

function substituteSlots(value: string, substitutions: Record<string, string>): string {
  return value.replace(/\{(\w+)\}/g, (match, slot: string) => substitutions[slot] ?? match);
}

/** A ready-made set of related workflow templates for demos, tests, and evals. */
export function defaultSyntheticWorkflows(): MovementWorkflowTemplate[] {
  return [
    {
      label: "form-submit",
      steps: [
        { gesture: "tap", target: "{field}" },
        { gesture: "type", target: "{field}", valueSummary: "entered value" },
        { gesture: "tap", target: "{submit}" },
      ],
      variants: [
        { field: "Email field", submit: "Sign in button" },
        { field: "Search box", submit: "Search button" },
        { field: "Name field", submit: "Save button" },
      ],
    },
    {
      label: "menu-navigate",
      steps: [
        { gesture: "tap", target: "{menu}" },
        { gesture: "tap", target: "{item}" },
        { gesture: "scroll", direction: "down" },
      ],
      variants: [
        { menu: "File menu", item: "Open" },
        { menu: "Edit menu", item: "Preferences" },
        { menu: "View menu", item: "Zoom In" },
      ],
    },
  ];
}
