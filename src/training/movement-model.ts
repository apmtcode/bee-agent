import type { TrajectoryAction, TrajectorySpan } from "../capture/trajectory.js";

/**
 * Local-movement learning subsystem — in-process model backend.
 *
 * Standing objective #2 requires bee-agent to (c) post-train a local model on a
 * recorded movement dataset to repeat the recorded movements and (d) generalize
 * to new-but-related movements. The `LocalAppleSiliconTrainingRunner` only emits
 * shell plans for real on-device runtimes (mlx/axolotl), which cannot execute in
 * the cloud/CI. This module supplies a **pluggable backend interface** plus a
 * fully deterministic, in-process **mock backend** so the train → predict →
 * generalize loop is exercisable and testable without a real machine or GPU.
 *
 * A real on-device small model implements the same {@link MovementModelBackend}
 * seam; the mock's {@link MovementModel} is plain JSON so it persists/replays
 * exactly like a real checkpoint would.
 */

export type MaybePromise<T> = T | Promise<T>;

export type MovementDirection = "up" | "down" | "left" | "right";

/** A single low-level movement extracted from a captured trajectory action. */
export type MovementStep = {
  tool: string;
  gesture: string;
  target?: string;
  direction?: MovementDirection;
  value?: string;
  coordinate?: { x: number; y: number };
  /** Delta time (ms) since the previous step in the same sequence. */
  dtMs?: number;
};

export type MovementSequence = {
  id: string;
  steps: MovementStep[];
  outcome?: TrajectorySpan["outcome"];
};

export type MovementTrainingDataset = {
  version: 1;
  sequences: MovementSequence[];
};

export type MovementTokenTemplate = {
  tool: string;
  gesture: string;
  direction?: MovementDirection;
  /** Distinct targets seen for this token, most-frequent first (deterministic). */
  targets: string[];
  /** Distinct values seen for this token, most-frequent first. */
  values: string[];
  coordinate?: { minX: number; maxX: number; minY: number; maxY: number };
  /** Mean start coordinate across sequences that begin with this token. */
  startCoordinate?: { x: number; y: number };
  avgDtMs?: number;
};

/** A serializable, first-order-Markov movement model (mock backend artifact). */
export type MovementModel = {
  version: 1;
  backend: string;
  tokens: string[];
  start: Record<string, number>;
  transitions: Record<string, Record<string, number>>;
  terminal: Record<string, number>;
  templates: Record<string, MovementTokenTemplate>;
  sequenceCount: number;
  stepCount: number;
};

export type MovementContext = {
  /** Force the opening token; defaults to the most frequent learned start. */
  startToken?: string;
  /** Retarget every generated step to a new (possibly unseen) UI target. */
  targetOverride?: string;
  /** Interpolate generated coordinates from the learned start toward this goal. */
  goalCoordinate?: { x: number; y: number };
  maxSteps?: number;
};

export type MovementPrediction = {
  steps: MovementStep[];
  tokens: string[];
  terminatedBy: "terminal" | "max-steps" | "no-transition";
};

export type MovementModelTrainConfig = {
  backendId?: string;
};

/**
 * Pluggable local-model backend. The cloud-safe mock is synchronous; a real
 * on-device backend may return promises — hence {@link MaybePromise}.
 */
export interface MovementModelBackend {
  readonly id: string;
  train(dataset: MovementTrainingDataset, config?: MovementModelTrainConfig): MaybePromise<MovementModel>;
  predict(model: MovementModel, context?: MovementContext): MaybePromise<MovementPrediction>;
}

const DEFAULT_MAX_STEPS = 64;

/** Canonical token capturing the *structure* of a movement (not its target). */
export function movementStepToken(step: Pick<MovementStep, "tool" | "gesture" | "direction">): string {
  const base = `${step.tool}/${step.gesture}`;
  return step.direction ? `${base}:${step.direction}` : base;
}

function coordinateFromMetadata(metadata: Record<string, unknown> | undefined): { x: number; y: number } | undefined {
  if (!metadata) {
    return undefined;
  }
  const x = metadata.x ?? (metadata.coordinate as { x?: unknown } | undefined)?.x;
  const y = metadata.y ?? (metadata.coordinate as { y?: unknown } | undefined)?.y;
  if (typeof x === "number" && Number.isFinite(x) && typeof y === "number" && Number.isFinite(y)) {
    return { x, y };
  }
  return undefined;
}

function directionFromMetadata(metadata: Record<string, unknown> | undefined): MovementDirection | undefined {
  const value = metadata?.direction;
  if (value === "up" || value === "down" || value === "left" || value === "right") {
    return value;
  }
  return undefined;
}

function stringFromMetadata(metadata: Record<string, unknown> | undefined, key: string): string | undefined {
  const value = metadata?.[key];
  return typeof value === "string" && value.length > 0 ? value : undefined;
}

/** Extract an ordered movement sequence from a captured trajectory's actions. */
export function deriveMovementSequence(trajectory: TrajectorySpan): MovementSequence {
  const ordered = [...trajectory.actions].sort((a, b) => a.ts - b.ts);
  let previousTs: number | undefined;
  const steps = ordered.map<MovementStep>((action: TrajectoryAction) => {
    const metadata = action.metadata;
    const gesture = stringFromMetadata(metadata, "gesture") ?? action.tool;
    const coordinate = coordinateFromMetadata(metadata);
    const direction = directionFromMetadata(metadata);
    const target = stringFromMetadata(metadata, "target");
    const value = stringFromMetadata(metadata, "valueSummary") ?? stringFromMetadata(metadata, "value");
    const dtMs = previousTs === undefined ? undefined : Math.max(0, action.ts - previousTs);
    previousTs = action.ts;
    return {
      tool: action.tool,
      gesture,
      ...(target ? { target } : {}),
      ...(direction ? { direction } : {}),
      ...(value ? { value } : {}),
      ...(coordinate ? { coordinate } : {}),
      ...(dtMs === undefined ? {} : { dtMs }),
    };
  });
  return {
    id: trajectory.id,
    steps,
    ...(trajectory.outcome ? { outcome: trajectory.outcome } : {}),
  };
}

/** Build a training dataset from captured trajectories (dropping empty ones). */
export function buildMovementDataset(trajectories: TrajectorySpan[]): MovementTrainingDataset {
  return {
    version: 1,
    sequences: trajectories.map(deriveMovementSequence).filter((sequence) => sequence.steps.length > 0),
  };
}

function increment(map: Record<string, number>, key: string): void {
  map[key] = (map[key] ?? 0) + 1;
}

/** Most-frequent-first ordering with a lexical tie-break for determinism. */
function rankByFrequency(counts: Map<string, number>): string[] {
  return [...counts.entries()]
    .sort((a, b) => (b[1] !== a[1] ? b[1] - a[1] : a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0))
    .map(([key]) => key);
}

/** Deterministic argmax over a count map (lexical tie-break). Empty → undefined. */
function argmax(counts: Record<string, number> | undefined): { key: string; count: number } | undefined {
  if (!counts) {
    return undefined;
  }
  let best: { key: string; count: number } | undefined;
  for (const [key, count] of Object.entries(counts)) {
    if (!best || count > best.count || (count === best.count && key < best.key)) {
      best = { key, count };
    }
  }
  return best;
}

/**
 * Deterministic, first-order-Markov mock backend. No RNG, no wall-clock — the
 * same dataset always yields byte-identical models and predictions, so it is a
 * safe stand-in for a real on-device model in cloud/CI tests.
 */
export class MarkovMovementBackend implements MovementModelBackend {
  readonly id: string;

  constructor(id = "markov-mock") {
    this.id = id;
  }

  train(dataset: MovementTrainingDataset, config?: MovementModelTrainConfig): MovementModel {
    const start: Record<string, number> = {};
    const transitions: Record<string, Record<string, number>> = {};
    const terminal: Record<string, number> = {};
    const targetCounts = new Map<string, Map<string, number>>();
    const valueCounts = new Map<string, Map<string, number>>();
    const coordAgg = new Map<string, { minX: number; maxX: number; minY: number; maxY: number }>();
    const startCoordAgg = new Map<string, { sumX: number; sumY: number; count: number }>();
    const dtAgg = new Map<string, { sum: number; count: number }>();
    const templateShape = new Map<string, { tool: string; gesture: string; direction?: MovementDirection }>();
    let stepCount = 0;

    for (const sequence of dataset.sequences) {
      if (sequence.steps.length === 0) {
        continue;
      }
      sequence.steps.forEach((step, index) => {
        const token = movementStepToken(step);
        stepCount += 1;
        if (!templateShape.has(token)) {
          templateShape.set(token, {
            tool: step.tool,
            gesture: step.gesture,
            ...(step.direction ? { direction: step.direction } : {}),
          });
        }
        if (index === 0) {
          increment(start, token);
          if (step.coordinate) {
            const agg = startCoordAgg.get(token) ?? { sumX: 0, sumY: 0, count: 0 };
            agg.sumX += step.coordinate.x;
            agg.sumY += step.coordinate.y;
            agg.count += 1;
            startCoordAgg.set(token, agg);
          }
        }
        if (index === sequence.steps.length - 1) {
          increment(terminal, token);
        } else {
          const nextToken = movementStepToken(sequence.steps[index + 1]);
          const row = (transitions[token] ??= {});
          increment(row, nextToken);
        }
        if (step.target) {
          const counts = targetCounts.get(token) ?? new Map<string, number>();
          counts.set(step.target, (counts.get(step.target) ?? 0) + 1);
          targetCounts.set(token, counts);
        }
        if (step.value) {
          const counts = valueCounts.get(token) ?? new Map<string, number>();
          counts.set(step.value, (counts.get(step.value) ?? 0) + 1);
          valueCounts.set(token, counts);
        }
        if (step.coordinate) {
          const agg = coordAgg.get(token) ?? {
            minX: step.coordinate.x,
            maxX: step.coordinate.x,
            minY: step.coordinate.y,
            maxY: step.coordinate.y,
          };
          agg.minX = Math.min(agg.minX, step.coordinate.x);
          agg.maxX = Math.max(agg.maxX, step.coordinate.x);
          agg.minY = Math.min(agg.minY, step.coordinate.y);
          agg.maxY = Math.max(agg.maxY, step.coordinate.y);
          coordAgg.set(token, agg);
        }
        if (step.dtMs !== undefined) {
          const agg = dtAgg.get(token) ?? { sum: 0, count: 0 };
          agg.sum += step.dtMs;
          agg.count += 1;
          dtAgg.set(token, agg);
        }
      });
    }

    const tokens = [...templateShape.keys()].sort();
    const templates: Record<string, MovementTokenTemplate> = {};
    for (const token of tokens) {
      const shape = templateShape.get(token)!;
      const coordinate = coordAgg.get(token);
      const startCoord = startCoordAgg.get(token);
      const dt = dtAgg.get(token);
      templates[token] = {
        tool: shape.tool,
        gesture: shape.gesture,
        ...(shape.direction ? { direction: shape.direction } : {}),
        targets: rankByFrequency(targetCounts.get(token) ?? new Map()),
        values: rankByFrequency(valueCounts.get(token) ?? new Map()),
        ...(coordinate ? { coordinate } : {}),
        ...(startCoord && startCoord.count > 0
          ? { startCoordinate: { x: startCoord.sumX / startCoord.count, y: startCoord.sumY / startCoord.count } }
          : {}),
        ...(dt && dt.count > 0 ? { avgDtMs: dt.sum / dt.count } : {}),
      };
    }

    return {
      version: 1,
      backend: config?.backendId ?? this.id,
      tokens,
      start,
      transitions,
      terminal,
      templates,
      sequenceCount: dataset.sequences.filter((sequence) => sequence.steps.length > 0).length,
      stepCount,
    };
  }

  predict(model: MovementModel, context: MovementContext = {}): MovementPrediction {
    const maxSteps = context.maxSteps ?? DEFAULT_MAX_STEPS;
    const startToken = context.startToken ?? argmax(model.start)?.key;
    const tokens: string[] = [];
    let terminatedBy: MovementPrediction["terminatedBy"] = "no-transition";

    // Decode by *consuming* transition counts: at each state follow the
    // most-traveled remaining outgoing edge and decrement it. A first-order
    // Markov argmax alone would loop forever on a tie (e.g. move→move vs
    // move→tap both seen twice); consuming counts reproduces the learned
    // multiset of transitions and is what makes move→move→tap decode correctly.
    const remaining: Record<string, Record<string, number>> = {};
    for (const [from, row] of Object.entries(model.transitions)) {
      remaining[from] = { ...row };
    }

    let current = startToken;
    while (current && tokens.length < maxSteps) {
      tokens.push(current);

      const terminalWeight = model.terminal[current] ?? 0;
      const next = argmax(remaining[current]);
      if (!next || next.count <= 0) {
        terminatedBy = terminalWeight > 0 ? "terminal" : "no-transition";
        break;
      }
      // Prefer terminating when ending here is at least as likely as continuing.
      if (terminalWeight >= next.count) {
        terminatedBy = "terminal";
        break;
      }
      remaining[current]![next.key] -= 1;
      current = next.key;
    }
    if (tokens.length >= maxSteps) {
      terminatedBy = "max-steps";
    }

    const steps = this.materializeSteps(model, tokens, context);
    return { steps, tokens, terminatedBy };
  }

  private materializeSteps(model: MovementModel, tokens: string[], context: MovementContext): MovementStep[] {
    const start = tokens.length > 0 ? model.templates[tokens[0]]?.startCoordinate : undefined;
    return tokens.map((token, index) => {
      const template = model.templates[token];
      const tool = template?.tool ?? token.split("/")[0] ?? token;
      const gesture = template?.gesture ?? token.split("/")[1]?.split(":")[0] ?? token;
      const target = context.targetOverride ?? template?.targets[0];
      const value = template?.values[0];
      const direction = template?.direction;
      const coordinate = this.interpolateCoordinate(template, start, context, index, tokens.length);
      return {
        tool,
        gesture,
        ...(target ? { target } : {}),
        ...(direction ? { direction } : {}),
        ...(value ? { value } : {}),
        ...(coordinate ? { coordinate } : {}),
        ...(template?.avgDtMs !== undefined ? { dtMs: Math.round(template.avgDtMs) } : {}),
      };
    });
  }

  /**
   * Generalization: when the caller supplies a goal coordinate, linearly
   * interpolate each generated step from the model's learned start coordinate
   * toward the goal — reproducing the *shape* of learned movements against a new
   * destination the model never saw during training.
   */
  private interpolateCoordinate(
    template: MovementTokenTemplate | undefined,
    startCoordinate: { x: number; y: number } | undefined,
    context: MovementContext,
    index: number,
    total: number,
  ): { x: number; y: number } | undefined {
    if (context.goalCoordinate && startCoordinate) {
      const t = total <= 1 ? 1 : index / (total - 1);
      return {
        x: round2(startCoordinate.x + (context.goalCoordinate.x - startCoordinate.x) * t),
        y: round2(startCoordinate.y + (context.goalCoordinate.y - startCoordinate.y) * t),
      };
    }
    if (template?.startCoordinate) {
      return { x: round2(template.startCoordinate.x), y: round2(template.startCoordinate.y) };
    }
    return undefined;
  }
}

function round2(value: number): number {
  return Math.round(value * 100) / 100;
}

/**
 * Deterministic synthetic movement generator — validates the
 * capture → dataset → train → predict loop without real OS input. Uses a small
 * seeded LCG so generated datasets are reproducible across runs and machines.
 */
export function generateSyntheticMovementDataset(spec: {
  sequenceCount: number;
  seed?: number;
  targets?: string[];
}): MovementTrainingDataset {
  const targets = spec.targets ?? ["menu", "submit", "confirm"];
  let state = (spec.seed ?? 1) >>> 0 || 1;
  const nextUnit = (): number => {
    // Numerical Recipes LCG — deterministic, no Math.random.
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 0xffffffff;
  };

  const sequences: MovementSequence[] = [];
  for (let i = 0; i < spec.sequenceCount; i += 1) {
    const target = targets[Math.floor(nextUnit() * targets.length) % targets.length];
    const originX = Math.round(nextUnit() * 200);
    const originY = Math.round(nextUnit() * 200);
    const destX = originX + 200 + Math.round(nextUnit() * 200);
    const destY = originY + 100 + Math.round(nextUnit() * 100);
    const steps: MovementStep[] = [
      { tool: "mouse", gesture: "move", coordinate: { x: originX, y: originY } },
      { tool: "mouse", gesture: "move", coordinate: { x: destX, y: destY }, dtMs: 120 },
      { tool: "mouse", gesture: "tap", target, coordinate: { x: destX, y: destY }, dtMs: 120 },
    ];
    sequences.push({
      id: `synthetic-${spec.seed ?? 1}-${i}`,
      steps,
      outcome: { status: "success", summary: `reached ${target}`, reward: 1 },
    });
  }
  return { version: 1, sequences };
}
