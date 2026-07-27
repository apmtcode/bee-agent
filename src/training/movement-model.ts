/**
 * Local-movement learning subsystem — model layer.
 *
 * Objective 2(c)/2(d): post-train a *local* model on recorded low-level
 * movements so it can (i) repeat a recorded movement and (ii) generalize to
 * new-but-related movements (e.g. the same gesture aimed at a new target).
 *
 * bee-agent runs in Anthropic's cloud with NO access to the user's machine, so
 * the *code, schema, and pipelines* live here and are validated with synthetic
 * movement streams. The heavy on-device backend (mlx / a small local net) is
 * pluggable behind {@link MovementModelBackend}; the {@link InProcessMovementModelBackend}
 * shipped here is a real, deterministic learner that trains and infers entirely
 * in-process so cloud/CI tests exercise the full train→predict→generalize loop.
 */

// ---------------------------------------------------------------------------
// Low-level movement event schema
// ---------------------------------------------------------------------------

export const MOVEMENT_EVENT_TYPES = [
  "mouse-move",
  "mouse-down",
  "mouse-up",
  "key-down",
  "key-up",
  "scroll",
  "wait",
] as const;

export type MovementEventType = (typeof MOVEMENT_EVENT_TYPES)[number];

export type PointerButton = "left" | "right" | "middle";

/**
 * A single low-level input event. Pointer coordinates are normalized to the
 * unit square [0,1] so a model trained on one screen resolution generalizes to
 * another; `t` is a millisecond offset from the start of its sequence.
 */
export type MovementEvent = {
  t: number;
  type: MovementEventType;
  x?: number;
  y?: number;
  button?: PointerButton;
  key?: string;
  dx?: number;
  dy?: number;
};

/**
 * One labeled demonstration — a captured gesture such as "open-menu" or
 * "drag-file-to-folder". `params` describe *what this instance was aimed at*
 * (target coordinates, distance, ...); the model learns how the event stream
 * depends on those params so it can be re-aimed (generalization).
 */
export type MovementSequence = {
  id: string;
  label: string;
  events: MovementEvent[];
  params?: Record<string, number>;
};

export type MovementDataset = {
  version: 1;
  sequences: MovementSequence[];
};

// ---------------------------------------------------------------------------
// Pluggable backend + model interfaces
// ---------------------------------------------------------------------------

export type MovementTrainingOptions = {
  /** Ridge regularization for the per-slot regressions; tiny by default. */
  ridge?: number;
};

export type MovementPredictRequest = {
  label: string;
  /** Target parameters for the movement to synthesize (re-aiming the gesture). */
  params?: Record<string, number>;
  /** Stable id for the produced sequence (defaults to a label-derived id). */
  id?: string;
};

/** Serialized model state — JSON-safe, so a trained model can be persisted. */
export type MovementModelState = {
  version: 1;
  backend: string;
  labels: MovementLabelTemplate[];
};

export interface MovementModel {
  readonly backend: string;
  /** Labels this model can synthesize. */
  labels(): string[];
  /** Synthesize a movement sequence for a label + optional target params. */
  predict(request: MovementPredictRequest): MovementSequence;
  serialize(): MovementModelState;
}

export interface MovementModelBackend {
  readonly name: string;
  train(dataset: MovementDataset, options?: MovementTrainingOptions): MovementModel;
  restore(state: MovementModelState): MovementModel;
}

// ---------------------------------------------------------------------------
// In-process backend — a real deterministic learner
// ---------------------------------------------------------------------------

/** A linear model `value = intercept + Σ weights[k] * params[paramKeys[k]]`. */
type LinearFit = {
  intercept: number;
  weights: number[];
};

type MovementSlotTemplate = {
  type: MovementEventType;
  button?: PointerButton;
  key?: string;
  t: LinearFit;
  x?: LinearFit;
  y?: LinearFit;
  dx?: LinearFit;
  dy?: LinearFit;
};

export type MovementLabelTemplate = {
  label: string;
  /** Ordered param keys the slot regressions are defined over. */
  paramKeys: string[];
  /** Number of demonstrations that matched the canonical structure. */
  support: number;
  slots: MovementSlotTemplate[];
};

const DEFAULT_RIDGE = 1e-8;

export class InProcessMovementModelBackend implements MovementModelBackend {
  readonly name = "in-process-linear";

  train(dataset: MovementDataset, options?: MovementTrainingOptions): MovementModel {
    const ridge = options?.ridge ?? DEFAULT_RIDGE;
    const byLabel = new Map<string, MovementSequence[]>();
    for (const sequence of dataset.sequences) {
      const bucket = byLabel.get(sequence.label);
      if (bucket) {
        bucket.push(sequence);
      } else {
        byLabel.set(sequence.label, [sequence]);
      }
    }

    const templates: MovementLabelTemplate[] = [];
    for (const [label, sequences] of byLabel) {
      const template = trainLabelTemplate(label, sequences, ridge);
      if (template) {
        templates.push(template);
      }
    }

    return new InProcessMovementModel(this.name, templates);
  }

  restore(state: MovementModelState): MovementModel {
    return new InProcessMovementModel(state.backend, state.labels);
  }
}

class InProcessMovementModel implements MovementModel {
  private readonly templates: Map<string, MovementLabelTemplate>;

  constructor(
    readonly backend: string,
    templates: MovementLabelTemplate[],
  ) {
    this.templates = new Map(templates.map((template) => [template.label, template]));
  }

  labels(): string[] {
    return [...this.templates.keys()].sort();
  }

  predict(request: MovementPredictRequest): MovementSequence {
    const template = this.templates.get(request.label);
    if (!template) {
      throw new Error(`movement model has no template for label "${request.label}"`);
    }
    const params = request.params ?? {};
    const vector = template.paramKeys.map((key) => params[key] ?? 0);

    const events: MovementEvent[] = template.slots.map((slot) => {
      const event: MovementEvent = {
        t: Math.max(0, Math.round(applyFit(slot.t, vector))),
        type: slot.type,
      };
      if (slot.x) {
        event.x = clampUnit(applyFit(slot.x, vector));
      }
      if (slot.y) {
        event.y = clampUnit(applyFit(slot.y, vector));
      }
      if (slot.dx) {
        event.dx = applyFit(slot.dx, vector);
      }
      if (slot.dy) {
        event.dy = applyFit(slot.dy, vector);
      }
      if (slot.button) {
        event.button = slot.button;
      }
      if (slot.key) {
        event.key = slot.key;
      }
      return event;
    });

    return {
      id: request.id ?? `${request.label}-prediction`,
      label: request.label,
      events,
      ...(request.params ? { params: { ...request.params } } : {}),
    };
  }

  serialize(): MovementModelState {
    return {
      version: 1,
      backend: this.backend,
      labels: [...this.templates.values()],
    };
  }
}

// ---------------------------------------------------------------------------
// Training internals
// ---------------------------------------------------------------------------

function structureSignature(sequence: MovementSequence): string {
  return sequence.events.map((event) => event.type).join(">");
}

function trainLabelTemplate(
  label: string,
  sequences: MovementSequence[],
  ridge: number,
): MovementLabelTemplate | undefined {
  const usable = sequences.filter((sequence) => sequence.events.length > 0);
  if (usable.length === 0) {
    return undefined;
  }

  // Pick the most common event-type structure so we align slot-by-slot. Ties
  // break deterministically toward the first seen (stable ordering).
  const groups = new Map<string, MovementSequence[]>();
  for (const sequence of usable) {
    const signature = structureSignature(sequence);
    const group = groups.get(signature);
    if (group) {
      group.push(sequence);
    } else {
      groups.set(signature, [sequence]);
    }
  }
  let canonical: MovementSequence[] = [];
  for (const group of groups.values()) {
    if (group.length > canonical.length) {
      canonical = group;
    }
  }

  const paramKeys = collectParamKeys(canonical);
  const design = canonical.map((sequence) => paramKeys.map((key) => sequence.params?.[key] ?? 0));
  const slotCount = canonical[0]!.events.length;
  const slots: MovementSlotTemplate[] = [];

  for (let index = 0; index < slotCount; index += 1) {
    const slotEvents = canonical.map((sequence) => sequence.events[index]!);
    const slot: MovementSlotTemplate = {
      type: slotEvents[0]!.type,
      t: fitColumn(design, slotEvents.map((event) => event.t), ridge),
    };
    const button = modalValue(slotEvents.map((event) => event.button));
    if (button) {
      slot.button = button;
    }
    const key = modalValue(slotEvents.map((event) => event.key));
    if (key) {
      slot.key = key;
    }
    for (const axis of ["x", "y", "dx", "dy"] as const) {
      if (slotEvents.every((event) => typeof event[axis] === "number")) {
        slot[axis] = fitColumn(design, slotEvents.map((event) => event[axis] as number), ridge);
      }
    }
    slots.push(slot);
  }

  return { label, paramKeys, support: canonical.length, slots };
}

function collectParamKeys(sequences: MovementSequence[]): string[] {
  const keys = new Set<string>();
  for (const sequence of sequences) {
    for (const key of Object.keys(sequence.params ?? {})) {
      keys.add(key);
    }
  }
  return [...keys].sort();
}

function modalValue<T extends string>(values: (T | undefined)[]): T | undefined {
  const counts = new Map<T, number>();
  for (const value of values) {
    if (value === undefined) {
      continue;
    }
    counts.set(value, (counts.get(value) ?? 0) + 1);
  }
  let best: T | undefined;
  let bestCount = 0;
  for (const [value, count] of counts) {
    if (count > bestCount) {
      best = value;
      bestCount = count;
    }
  }
  return best;
}

/**
 * Ridge-regularized least-squares fit of `targets ~ [1, design...]`.
 * With distinct param rows the normal matrix is well-conditioned and the tiny
 * ridge keeps it invertible without materially biasing exact linear relations,
 * so an event coordinate that equals a target param is recovered near-exactly.
 */
function fitColumn(design: number[][], targets: number[], ridge: number): LinearFit {
  const rows = design.length;
  const featureCount = (design[0]?.length ?? 0) + 1; // + intercept

  // Augment each row with a leading 1 for the intercept.
  const x = design.map((row) => [1, ...row]);

  // Normal equations: (XᵀX + ridge·I) β = Xᵀy. Regularize only the slopes.
  const xtx: number[][] = Array.from({ length: featureCount }, () => new Array(featureCount).fill(0));
  const xty: number[] = new Array(featureCount).fill(0);
  for (let r = 0; r < rows; r += 1) {
    const row = x[r]!;
    const target = targets[r]!;
    for (let i = 0; i < featureCount; i += 1) {
      xty[i] += row[i]! * target;
      for (let j = 0; j < featureCount; j += 1) {
        xtx[i]![j] += row[i]! * row[j]!;
      }
    }
  }
  for (let i = 1; i < featureCount; i += 1) {
    xtx[i]![i] += ridge;
  }

  const beta = solveLinearSystem(xtx, xty);
  return { intercept: beta[0]!, weights: beta.slice(1) };
}

/** Gaussian elimination with partial pivoting; falls back to zeros if singular. */
function solveLinearSystem(matrix: number[][], vector: number[]): number[] {
  const n = vector.length;
  const a = matrix.map((row, index) => [...row, vector[index]!]);

  for (let col = 0; col < n; col += 1) {
    let pivot = col;
    for (let row = col + 1; row < n; row += 1) {
      if (Math.abs(a[row]![col]!) > Math.abs(a[pivot]![col]!)) {
        pivot = row;
      }
    }
    if (Math.abs(a[pivot]![col]!) < 1e-12) {
      continue; // singular column — leave its coefficient at 0
    }
    [a[col], a[pivot]] = [a[pivot]!, a[col]!];
    const pivotRow = a[col]!;
    const pivotValue = pivotRow[col]!;
    for (let row = 0; row < n; row += 1) {
      if (row === col) {
        continue;
      }
      const factor = a[row]![col]! / pivotValue;
      if (factor === 0) {
        continue;
      }
      for (let k = col; k <= n; k += 1) {
        a[row]![k]! -= factor * pivotRow[k]!;
      }
    }
  }

  const solution = new Array(n).fill(0);
  for (let i = 0; i < n; i += 1) {
    const diagonal = a[i]![i]!;
    solution[i] = Math.abs(diagonal) < 1e-12 ? 0 : a[i]![n]! / diagonal;
  }
  return solution;
}

function applyFit(fit: LinearFit, params: number[]): number {
  let value = fit.intercept;
  for (let i = 0; i < fit.weights.length; i += 1) {
    value += fit.weights[i]! * (params[i] ?? 0);
  }
  return value;
}

function clampUnit(value: number): number {
  if (value < 0) {
    return 0;
  }
  if (value > 1) {
    return 1;
  }
  return value;
}
