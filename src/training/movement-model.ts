/**
 * Local-movement learning model (standing objective #2, pieces c + d).
 *
 * This module supplies the trainable, generalizing core of the movement
 * subsystem. The capture pipeline (`src/capture/`) records movements at a
 * gesture/summary abstraction; here we define a low-level, numerically precise,
 * fully replayable movement-event schema plus a *pluggable* model backend that
 * can be trained on a dataset of demonstrations and then produce movements for
 * new-but-related tasks it never saw during training.
 *
 * Everything here is deterministic and OS-free so it runs (and is tested) in the
 * cloud. Real on-device recording feeds the same schema; a real on-device model
 * plugs in behind {@link MovementModelBackend}. The default
 * {@link NearestNeighborMovementBackend} learns motion *style* from
 * demonstrations and retargets it onto new endpoints via an exact 2-D frame
 * transform — a concrete, verifiable form of generalization.
 */

export type PointerButton = "left" | "right" | "middle";

/** A single low-level movement event. Timestamps are monotonic milliseconds. */
export type MovementEvent =
  | { kind: "pointer-move"; ts: number; x: number; y: number }
  | { kind: "pointer-down"; ts: number; x: number; y: number; button: PointerButton }
  | { kind: "pointer-up"; ts: number; x: number; y: number; button: PointerButton }
  | { kind: "scroll"; ts: number; x: number; y: number; dx: number; dy: number }
  | { kind: "key"; ts: number; key: string; down: boolean };

export type MovementGestureKind = "click" | "double-click" | "drag" | "scroll" | "type";

export type MovementPoint = { x: number; y: number };

/** A movement to perform: what a policy is asked to reproduce/generalize. */
export type MovementTask = {
  gesture: MovementGestureKind;
  start: MovementPoint;
  target: MovementPoint;
  /** Text to type (gesture "type"). */
  text?: string;
  /** Scroll delta (gesture "scroll"). */
  scroll?: { dx: number; dy: number };
  label?: string;
};

/** One recorded demonstration: a task paired with the events that achieved it. */
export type MovementDemonstration = {
  id: string;
  task: MovementTask;
  events: MovementEvent[];
};

/** The on-disk / on-wire dataset format for movement demonstrations. */
export type MovementDataset = {
  version: 1;
  demonstrations: MovementDemonstration[];
};

export interface MovementPolicy {
  readonly backendId: string;
  readonly demonstrationCount: number;
  /** Produce a movement-event sequence for a (possibly unseen) task. */
  predict(task: MovementTask): MovementEvent[];
}

/**
 * Pluggable model backend. A real on-device small model implements this to
 * train weights and expose inference; the deterministic backend below stands in
 * for cloud/CI. Kept intentionally tiny so backends are trivial to register.
 */
export interface MovementModelBackend {
  readonly id: string;
  train(dataset: MovementDataset): MovementPolicy;
}

// ---------------------------------------------------------------------------
// Dataset format helpers
// ---------------------------------------------------------------------------

export function createMovementDataset(demonstrations: MovementDemonstration[] = []): MovementDataset {
  return { version: 1, demonstrations: [...demonstrations] };
}

export function serializeMovementDataset(dataset: MovementDataset): string {
  return `${JSON.stringify(dataset, null, 2)}\n`;
}

export function parseMovementDataset(raw: string): MovementDataset {
  const parsed = JSON.parse(raw) as MovementDataset;
  if (parsed.version !== 1 || !Array.isArray(parsed.demonstrations)) {
    throw new Error("invalid movement dataset: expected { version: 1, demonstrations: [] }");
  }
  return parsed;
}

// ---------------------------------------------------------------------------
// Synthetic event-stream generator (no OS input required)
// ---------------------------------------------------------------------------

export type SynthesizeOptions = {
  /** Interpolation steps for the pointer path (>= 1). */
  steps?: number;
  /** Milliseconds between consecutive events. */
  stepMs?: number;
  /**
   * Perpendicular curvature of the approach path, as a fraction of path length.
   * 0 = straight line; positive bows the path so learned "style" is non-trivial.
   */
  curvature?: number;
  /** Timestamp of the first event. */
  startTs?: number;
};

/**
 * Build a plausible, replayable event sequence for a task without any real OS
 * input. Used to seed datasets and eval sets for cloud tests, and mirrors the
 * shape a real recorder would emit.
 */
export function synthesizeMovement(task: MovementTask, options: SynthesizeOptions = {}): MovementEvent[] {
  const steps = Math.max(1, Math.floor(options.steps ?? 8));
  const stepMs = options.stepMs ?? 16;
  const curvature = options.curvature ?? 0;
  let ts = options.startTs ?? 0;
  const events: MovementEvent[] = [];

  const path = interpolatePath(task.start, task.target, steps, curvature);
  for (const point of path) {
    events.push({ kind: "pointer-move", ts, x: round(point.x), y: round(point.y) });
    ts += stepMs;
  }
  const end = task.target;

  switch (task.gesture) {
    case "click":
      events.push({ kind: "pointer-down", ts, x: round(end.x), y: round(end.y), button: "left" });
      ts += stepMs;
      events.push({ kind: "pointer-up", ts, x: round(end.x), y: round(end.y), button: "left" });
      break;
    case "double-click":
      for (let i = 0; i < 2; i += 1) {
        events.push({ kind: "pointer-down", ts, x: round(end.x), y: round(end.y), button: "left" });
        ts += stepMs;
        events.push({ kind: "pointer-up", ts, x: round(end.x), y: round(end.y), button: "left" });
        ts += stepMs;
      }
      break;
    case "drag": {
      // Insert a press at the start, drag along the path, release at the target.
      events.unshift({ kind: "pointer-down", ts: (options.startTs ?? 0) - stepMs, x: round(task.start.x), y: round(task.start.y), button: "left" });
      events.push({ kind: "pointer-up", ts, x: round(end.x), y: round(end.y), button: "left" });
      break;
    }
    case "scroll": {
      const scroll = task.scroll ?? { dx: 0, dy: -120 };
      events.push({ kind: "scroll", ts, x: round(end.x), y: round(end.y), dx: scroll.dx, dy: scroll.dy });
      break;
    }
    case "type": {
      events.push({ kind: "pointer-down", ts, x: round(end.x), y: round(end.y), button: "left" });
      ts += stepMs;
      events.push({ kind: "pointer-up", ts, x: round(end.x), y: round(end.y), button: "left" });
      ts += stepMs;
      for (const key of [...(task.text ?? "")]) {
        events.push({ kind: "key", ts, key, down: true });
        ts += stepMs;
        events.push({ kind: "key", ts, key, down: false });
        ts += stepMs;
      }
      break;
    }
  }
  return events;
}

export function synthesizeDemonstration(
  id: string,
  task: MovementTask,
  options: SynthesizeOptions = {},
): MovementDemonstration {
  return { id, task, events: synthesizeMovement(task, options) };
}

// ---------------------------------------------------------------------------
// Nearest-neighbor generalizing backend (deterministic; cloud-safe)
// ---------------------------------------------------------------------------

/**
 * Learns motion *style* from demonstrations and generalizes to new endpoints.
 *
 * Training just indexes the demonstrations. Inference picks the most similar
 * same-gesture demonstration (by endpoint proximity, then path-length), then
 * retargets its pointer path onto the requested start/target with an exact 2-D
 * frame transform — so the produced path preserves the recorded shape while
 * landing precisely on a target the model never saw. Non-pointer payloads
 * (typed text, scroll deltas) are taken from the requested task.
 */
export class NearestNeighborMovementBackend implements MovementModelBackend {
  readonly id = "nearest-neighbor-mock";

  train(dataset: MovementDataset): MovementPolicy {
    const demonstrations = [...dataset.demonstrations];
    const backendId = this.id;
    return {
      backendId,
      demonstrationCount: demonstrations.length,
      predict(task: MovementTask): MovementEvent[] {
        const reference = selectReference(demonstrations, task);
        if (!reference) {
          // No prior knowledge — fall back to a straight synthesized movement.
          return synthesizeMovement(task);
        }
        return retargetMovement(reference, task);
      },
    };
  }
}

/** Default registry so backends are swappable by id. */
export class MovementBackendRegistry {
  private readonly backends = new Map<string, MovementModelBackend>();

  register(backend: MovementModelBackend): this {
    this.backends.set(backend.id, backend);
    return this;
  }

  get(id: string): MovementModelBackend | undefined {
    return this.backends.get(id);
  }

  list(): string[] {
    return [...this.backends.keys()];
  }
}

export function createDefaultMovementBackendRegistry(): MovementBackendRegistry {
  return new MovementBackendRegistry().register(new NearestNeighborMovementBackend());
}

// ---------------------------------------------------------------------------
// Generalization eval harness
// ---------------------------------------------------------------------------

export type MovementEvalResult = {
  count: number;
  meanEndpointError: number;
  maxEndpointError: number;
  meanShapeError: number;
};

/** Final pointer position of a produced sequence, if any. */
export function movementEndpoint(events: MovementEvent[]): MovementPoint | undefined {
  for (let i = events.length - 1; i >= 0; i -= 1) {
    const event = events[i];
    if (event.kind === "pointer-move" || event.kind === "pointer-down" || event.kind === "pointer-up" || event.kind === "scroll") {
      return { x: event.x, y: event.y };
    }
  }
  return undefined;
}

/** Euclidean distance between a sequence's endpoint and the task target. */
export function movementEndpointError(events: MovementEvent[], task: MovementTask): number {
  const endpoint = movementEndpoint(events);
  if (!endpoint) {
    return Number.POSITIVE_INFINITY;
  }
  return distance(endpoint, task.target);
}

/**
 * Evaluate a trained policy on held-out (but related) tasks. Reports endpoint
 * accuracy plus a shape-fidelity metric (how closely the produced path tracks a
 * straight-line-normalized reference), so regressions in generalization are
 * measurable across runs.
 */
export function evaluateMovementPolicy(policy: MovementPolicy, tasks: MovementTask[]): MovementEvalResult {
  if (tasks.length === 0) {
    return { count: 0, meanEndpointError: 0, maxEndpointError: 0, meanShapeError: 0 };
  }
  let sumEndpoint = 0;
  let maxEndpoint = 0;
  let sumShape = 0;
  for (const task of tasks) {
    const events = policy.predict(task);
    const endpointError = movementEndpointError(events, task);
    sumEndpoint += endpointError;
    maxEndpoint = Math.max(maxEndpoint, endpointError);
    sumShape += pathShapeError(events, task);
  }
  return {
    count: tasks.length,
    meanEndpointError: sumEndpoint / tasks.length,
    maxEndpointError: maxEndpoint,
    meanShapeError: sumShape / tasks.length,
  };
}

// ---------------------------------------------------------------------------
// Internals
// ---------------------------------------------------------------------------

function selectReference(
  demonstrations: MovementDemonstration[],
  task: MovementTask,
): MovementDemonstration | undefined {
  const sameGesture = demonstrations.filter((demo) => demo.task.gesture === task.gesture);
  const candidates = sameGesture.length > 0 ? sameGesture : demonstrations;
  let best: MovementDemonstration | undefined;
  let bestScore = Number.POSITIVE_INFINITY;
  for (const demo of candidates) {
    // Similarity: how close the demo's endpoints are to the requested endpoints.
    const score =
      distance(demo.task.target, task.target) + distance(demo.task.start, task.start);
    if (score < bestScore) {
      bestScore = score;
      best = demo;
    }
  }
  return best;
}

function retargetMovement(reference: MovementDemonstration, task: MovementTask): MovementEvent[] {
  const refStart = reference.task.start;
  const refEnd = reference.task.target;
  const baseTs = reference.events[0]?.ts ?? 0;

  const transformed = reference.events.map((event): MovementEvent => {
    const ts = event.ts - baseTs;
    if (event.kind === "key") {
      return { ...event, ts };
    }
    const mapped = frameTransform({ x: event.x, y: event.y }, refStart, refEnd, task.start, task.target);
    if (event.kind === "scroll") {
      const scroll = task.scroll ?? { dx: event.dx, dy: event.dy };
      return { kind: "scroll", ts, x: round(mapped.x), y: round(mapped.y), dx: scroll.dx, dy: scroll.dy };
    }
    return { ...event, ts, x: round(mapped.x), y: round(mapped.y) };
  });

  // For typing, regenerate key events from the requested text so the policy
  // generalizes to new content, not just new positions.
  if (task.gesture === "type" && task.text !== undefined) {
    const withoutKeys = transformed.filter((event) => event.kind !== "key");
    const lastTs = withoutKeys.at(-1)?.ts ?? 0;
    let ts = lastTs + 16;
    const keys: MovementEvent[] = [];
    for (const key of [...task.text]) {
      keys.push({ kind: "key", ts, key, down: true });
      ts += 16;
      keys.push({ kind: "key", ts, key, down: false });
      ts += 16;
    }
    return [...withoutKeys, ...keys];
  }

  return transformed;
}

/**
 * Map a point from the reference frame (refStart→refEnd) into the target frame
 * (start→target), preserving along-axis progress and perpendicular offset. Maps
 * refStart→start and refEnd→target exactly, so endpoints are reproduced with
 * zero error while the recorded path shape is carried over.
 */
function frameTransform(
  point: MovementPoint,
  refStart: MovementPoint,
  refEnd: MovementPoint,
  start: MovementPoint,
  target: MovementPoint,
): MovementPoint {
  const u = { x: refEnd.x - refStart.x, y: refEnd.y - refStart.y };
  const refLenSq = u.x * u.x + u.y * u.y;
  const rel = { x: point.x - refStart.x, y: point.y - refStart.y };

  if (refLenSq === 0) {
    // Degenerate reference (no travel): translate by the requested start delta.
    return { x: start.x + rel.x, y: start.y + rel.y };
  }

  // Decompose into along-axis fraction (s) and perpendicular distance / refLen.
  const s = (rel.x * u.x + rel.y * u.y) / refLenSq;
  const tPerp = (rel.x * -u.y + rel.y * u.x) / refLenSq;

  const v = { x: target.x - start.x, y: target.y - start.y };
  return {
    x: start.x + s * v.x + tPerp * -v.y,
    y: start.y + s * v.y + tPerp * v.x,
  };
}

function interpolatePath(
  start: MovementPoint,
  target: MovementPoint,
  steps: number,
  curvature: number,
): MovementPoint[] {
  const points: MovementPoint[] = [];
  const perp = { x: -(target.y - start.y), y: target.x - start.x };
  for (let i = 0; i <= steps; i += 1) {
    const s = i / steps;
    const bow = curvature * Math.sin(Math.PI * s);
    points.push({
      x: start.x + s * (target.x - start.x) + bow * perp.x,
      y: start.y + s * (target.y - start.y) + bow * perp.y,
    });
  }
  return points;
}

/**
 * Mean perpendicular deviation of the produced pointer path from the straight
 * start→target line, normalized by path length. A stable, scale-free proxy for
 * how much recorded "style" survived generalization.
 */
function pathShapeError(events: MovementEvent[], task: MovementTask): number {
  const points = events
    .filter((event): event is Extract<MovementEvent, { x: number; y: number }> =>
      event.kind === "pointer-move" || event.kind === "pointer-down" || event.kind === "pointer-up" || event.kind === "scroll",
    )
    .map((event) => ({ x: event.x, y: event.y }));
  if (points.length === 0) {
    return 0;
  }
  const u = { x: task.target.x - task.start.x, y: task.target.y - task.start.y };
  const len = Math.hypot(u.x, u.y);
  if (len === 0) {
    return 0;
  }
  let sum = 0;
  for (const point of points) {
    const rel = { x: point.x - task.start.x, y: point.y - task.start.y };
    const perpDist = Math.abs(rel.x * -u.y + rel.y * u.x) / len;
    sum += perpDist / len;
  }
  return sum / points.length;
}

function distance(a: MovementPoint, b: MovementPoint): number {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function round(value: number): number {
  return Math.round(value * 1000) / 1000;
}
