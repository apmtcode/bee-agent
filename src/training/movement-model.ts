import type {
  MovementContext,
  MovementDataset,
  MovementExample,
  MovementStep,
} from "./movement-dataset.js";

/**
 * Local-movement learning model: the train + infer half of the movement
 * subsystem (objective #2 items c & d). A {@link MovementModelBackend} learns
 * from a {@link MovementDataset} to (c) repeat recorded movements and (d)
 * generalize to new-but-related movements.
 *
 * The backend is pluggable so a real on-device small model can be dropped in
 * behind the same interface. The bundled {@link DeterministicSequenceModelBackend}
 * is a fully in-process, deterministic implementation with no OS dependency, so
 * capture -> dataset -> train -> infer round-trips can be validated in the cloud
 * with synthetic event streams.
 */

/** Fields a movement step exposes as generalizable slots. */
export const MOVEMENT_SLOT_FIELDS = ["target", "valueSummary", "direction"] as const;
export type MovementSlotField = (typeof MOVEMENT_SLOT_FIELDS)[number];

export type MovementTrainingConfig = {
  /**
   * Similarity at/above which a query is treated as an exact recall of a
   * recorded movement (repeat). Default 0.999.
   */
  repeatThreshold?: number;
  /**
   * Minimum similarity for a query to generalize from the nearest recorded
   * movement. Below this the model returns `mode: "unknown"`. Default 0.34.
   */
  generalizeThreshold?: number;
};

export type MovementPredictionMode = "repeat" | "generalize" | "unknown";

export type MovementPrediction = {
  mode: MovementPredictionMode;
  /** Predicted movement steps to perform. Empty when `mode === "unknown"`. */
  steps: MovementStep[];
  /** 0..1 confidence — 1 for an exact repeat, similarity for a generalization. */
  confidence: number;
  /** Example the prediction was derived from, when any matched. */
  matchedExampleId?: string;
  /** Slot fields the query filled during generalization, by field name. */
  filledSlots: Record<string, string>;
};

/** An induced variable slot: a step field whose value varied across a shape group. */
export type MovementSlot = {
  shapeKey: string;
  stepIndex: number;
  field: MovementSlotField;
};

/** JSON-serializable trained model produced by a backend. */
export type SerializedMovementModel = {
  version: 1;
  backend: string;
  repeatThreshold: number;
  generalizeThreshold: number;
  examples: MovementExample[];
  /** Step fields the model induced as variable (used to guide generalization). */
  slots: MovementSlot[];
};

export interface MovementModelBackend {
  readonly name: string;
  train(dataset: MovementDataset, config?: MovementTrainingConfig): SerializedMovementModel;
  predict(model: SerializedMovementModel, query: MovementContext): MovementPrediction;
}

const DEFAULT_REPEAT_THRESHOLD = 0.999;
const DEFAULT_GENERALIZE_THRESHOLD = 0.34;

/**
 * A deterministic nearest-neighbour movement policy with data-induced slot
 * generalization.
 *
 * Training indexes every example and induces which step fields are *variable*:
 * within a group of movements sharing the same app/platform and gesture shape,
 * any field whose value differs across examples becomes a slot. At inference the
 * model finds the most similar recorded movement and replays its steps, filling
 * induced slots from `query.slots` when provided — so a movement recorded for one
 * target reproduces for a new target it has never seen.
 */
export class DeterministicSequenceModelBackend implements MovementModelBackend {
  readonly name = "deterministic-sequence";

  train(dataset: MovementDataset, config: MovementTrainingConfig = {}): SerializedMovementModel {
    const examples = dataset.examples.map(cloneExample);
    return {
      version: 1,
      backend: this.name,
      repeatThreshold: config.repeatThreshold ?? DEFAULT_REPEAT_THRESHOLD,
      generalizeThreshold: config.generalizeThreshold ?? DEFAULT_GENERALIZE_THRESHOLD,
      examples,
      slots: induceSlots(examples),
    };
  }

  predict(model: SerializedMovementModel, query: MovementContext): MovementPrediction {
    const queryTokens = contextTokens(query);
    let best: { example: MovementExample; similarity: number } | undefined;

    for (const example of model.examples) {
      if (!contextsCompatible(example.context, query)) {
        continue;
      }
      const similarity = jaccard(queryTokens, contextTokens(example.context));
      if (!best || similarity > best.similarity) {
        best = { example, similarity };
      }
    }

    if (!best || best.similarity < model.generalizeThreshold) {
      return { mode: "unknown", steps: [], confidence: 0, filledSlots: {} };
    }

    const providedSlots = query.slots ?? {};
    const hasOverride = MOVEMENT_SLOT_FIELDS.some((field) => typeof providedSlots[field] === "string");

    if (best.similarity >= model.repeatThreshold && !hasOverride) {
      return {
        mode: "repeat",
        steps: best.example.steps.map(cloneStep),
        confidence: 1,
        matchedExampleId: best.example.id,
        filledSlots: {},
      };
    }

    const shapeKey = shapeKeyFor(best.example);
    const variableFieldsByStep = variableFieldIndex(model.slots, shapeKey);
    const filledSlots: Record<string, string> = {};
    const steps = best.example.steps.map((step, index) => {
      const next = cloneStep(step);
      const variableFields = variableFieldsByStep.get(index);
      if (variableFields) {
        for (const field of variableFields) {
          const override = providedSlots[field];
          if (typeof override === "string" && override.length > 0) {
            next[field] = override;
            filledSlots[field] = override;
          }
        }
      }
      return next;
    });

    return {
      mode: "generalize",
      steps,
      confidence: best.similarity,
      matchedExampleId: best.example.id,
      filledSlots,
    };
  }
}

/** Trained, ready-to-query movement policy — pairs a backend with its model. */
export class MovementPolicy {
  constructor(
    private readonly backend: MovementModelBackend,
    private readonly serialized: SerializedMovementModel,
  ) {}

  /** The JSON-serializable trained model (persist / ship this to a device). */
  get model(): SerializedMovementModel {
    return this.serialized;
  }

  /** Predict the movement to perform for a query context. */
  infer(query: MovementContext): MovementPrediction {
    return this.backend.predict(this.serialized, query);
  }

  /** Re-attach the deterministic backend to a previously serialized model. */
  static fromSerialized(model: SerializedMovementModel): MovementPolicy {
    return new MovementPolicy(new DeterministicSequenceModelBackend(), model);
  }
}

/** High-level trainer: dataset -> trained {@link MovementPolicy}. */
export class MovementModelTrainer {
  constructor(private readonly backend: MovementModelBackend = new DeterministicSequenceModelBackend()) {}

  train(dataset: MovementDataset, config?: MovementTrainingConfig): MovementPolicy {
    return new MovementPolicy(this.backend, this.backend.train(dataset, config));
  }
}

function induceSlots(examples: MovementExample[]): MovementSlot[] {
  const groups = new Map<string, MovementExample[]>();
  for (const example of examples) {
    const key = shapeKeyFor(example);
    const group = groups.get(key);
    if (group) {
      group.push(example);
    } else {
      groups.set(key, [example]);
    }
  }

  const slots: MovementSlot[] = [];
  for (const [shapeKey, group] of groups) {
    if (group.length < 2) {
      continue;
    }
    const stepCount = Math.min(...group.map((example) => example.steps.length));
    for (let stepIndex = 0; stepIndex < stepCount; stepIndex += 1) {
      for (const field of MOVEMENT_SLOT_FIELDS) {
        const values = new Set<string>();
        for (const example of group) {
          const value = example.steps[stepIndex]?.[field];
          if (typeof value === "string" && value.length > 0) {
            values.add(value);
          }
        }
        if (values.size >= 2) {
          slots.push({ shapeKey, stepIndex, field });
        }
      }
    }
  }
  return slots;
}

function variableFieldIndex(slots: MovementSlot[], shapeKey: string): Map<number, MovementSlotField[]> {
  const index = new Map<number, MovementSlotField[]>();
  for (const slot of slots) {
    if (slot.shapeKey !== shapeKey) {
      continue;
    }
    const fields = index.get(slot.stepIndex);
    if (fields) {
      fields.push(slot.field);
    } else {
      index.set(slot.stepIndex, [slot.field]);
    }
  }
  return index;
}

/**
 * The invariant "shape" of a movement: its app, platform, and gesture sequence.
 * Two movements with the same shape but different targets are the same skill
 * applied to different objects — that difference is what becomes a slot.
 */
function shapeKeyFor(example: MovementExample): string {
  const app = example.context.appId ?? "";
  const platform = example.context.platform ?? "";
  const gestures = example.steps.map((step) => `${step.tool}:${step.gesture ?? ""}`).join(">");
  return `${app}|${platform}|${gestures}`;
}

function contextsCompatible(recorded: MovementContext, query: MovementContext): boolean {
  if (recorded.appId && query.appId && recorded.appId !== query.appId) {
    return false;
  }
  if (recorded.platform && query.platform && recorded.platform !== query.platform) {
    return false;
  }
  return true;
}

function contextTokens(context: MovementContext): Set<string> {
  const tokens = new Set<string>();
  if (context.appId) {
    tokens.add(`app:${context.appId.toLowerCase()}`);
  }
  if (context.platform) {
    tokens.add(`platform:${context.platform.toLowerCase()}`);
  }
  for (const token of tokenize(context.screenTitle)) {
    tokens.add(token);
  }
  for (const token of tokenize(context.goal)) {
    tokens.add(token);
  }
  return tokens;
}

function tokenize(value: string | undefined): string[] {
  if (!value) {
    return [];
  }
  return value
    .toLowerCase()
    .split(/[^a-z0-9]+/)
    .filter((token) => token.length > 0);
}

function jaccard(a: Set<string>, b: Set<string>): number {
  if (a.size === 0 && b.size === 0) {
    return 1;
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

function cloneExample(example: MovementExample): MovementExample {
  return {
    id: example.id,
    sessionId: example.sessionId,
    context: { ...example.context, ...(example.context.slots ? { slots: { ...example.context.slots } } : {}) },
    steps: example.steps.map(cloneStep),
  };
}

function cloneStep(step: MovementStep): MovementStep {
  return { ...step };
}
