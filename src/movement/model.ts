/**
 * Pluggable local-movement model backend.
 *
 * Objective #2 (c)+(d): post-train a *local* model that can (c) repeat recorded
 * movements and (d) generalize to new-but-related movements. The real on-device
 * path will be a small open model trained via the runner in `src/training/`
 * (mlx / axolotl). That cannot run in the cloud, so the model surface is an
 * interface ({@link MovementModelBackend}) with a deterministic, dependency-free
 * reference implementation ({@link SimilarityTransformBackend}) that genuinely
 * learns to repeat *and* generalize — exercised end-to-end in CI.
 *
 * The reference backend models a movement as a *shape in a reference frame*.
 * Training indexes demonstrations by intent. Inference selects the closest
 * demonstration for the requested intent and re-targets it to the requested
 * start/end via the exact 2-point similarity transform (translation + rotation
 * + uniform scale) that maps the demo's frame onto the requested frame. When the
 * requested frame equals the demo frame the transform is the identity, so the
 * model reproduces the recording exactly; otherwise it produces a correctly
 * rotated/scaled/translated version — the generalization property.
 */

import {
  datasetIntents,
  euclideanDistance,
  isPositionedEvent,
  type MovementDataset,
  type MovementDemonstration,
  type MovementEvent,
  type Point,
} from "./events.js";

export type MovementContext = {
  intent: string;
  start: Point;
  end: Point;
};

export interface TrainedMovementModel {
  readonly backend: string;
  readonly intents: string[];
  /** Predict the event stream that performs `context`, or `[]` if unknown intent. */
  predict(context: MovementContext): MovementEvent[];
}

export interface MovementModelBackend {
  readonly name: string;
  train(dataset: MovementDataset): Promise<TrainedMovementModel>;
}

/**
 * Exact similarity transform from correspondence (p0→q0, p1→q1), expressed with
 * complex arithmetic: f(p) = q0 + c·(p − p0) where c = (q1 − q0) / (p1 − p0).
 * If the source vector p1−p0 is degenerate (zero length) we fall back to pure
 * translation by q0 − p0 so the transform is always well-defined.
 */
export function similarityTransform(
  fromStart: Point,
  fromEnd: Point,
  toStart: Point,
  toEnd: Point,
): (p: Point) => Point {
  const px = fromEnd.x - fromStart.x;
  const py = fromEnd.y - fromStart.y;
  const denom = px * px + py * py;

  if (denom === 0) {
    const tx = toStart.x - fromStart.x;
    const ty = toStart.y - fromStart.y;
    return (p) => ({ x: p.x + tx, y: p.y + ty });
  }

  const qx = toEnd.x - toStart.x;
  const qy = toEnd.y - toStart.y;
  // c = (q1-q0) / (p1-p0) as complex division.
  const cx = (qx * px + qy * py) / denom;
  const cy = (qy * px - qx * py) / denom;

  return (p) => {
    const dx = p.x - fromStart.x;
    const dy = p.y - fromStart.y;
    return {
      x: toStart.x + (cx * dx - cy * dy),
      y: toStart.y + (cx * dy + cy * dx),
    };
  };
}

/** Apply a point transform to every spatially-positioned event in a stream. */
export function retargetEvents(
  events: MovementEvent[],
  transform: (p: Point) => Point,
): MovementEvent[] {
  return events.map((event) => {
    if (!isPositionedEvent(event)) {
      return event;
    }
    const mapped = transform({ x: event.x, y: event.y });
    return { ...event, x: mapped.x, y: mapped.y };
  });
}

class SimilarityTransformModel implements TrainedMovementModel {
  readonly backend = "similarity-transform";

  constructor(private readonly byIntent: Map<string, MovementDemonstration[]>) {}

  get intents(): string[] {
    return [...this.byIntent.keys()];
  }

  predict(context: MovementContext): MovementEvent[] {
    const candidates = this.byIntent.get(context.intent);
    if (!candidates || candidates.length === 0) {
      return [];
    }
    const demo = this.selectNearest(candidates, context);
    const transform = similarityTransform(
      demo.frame.start,
      demo.frame.end,
      context.start,
      context.end,
    );
    return retargetEvents(demo.events, transform);
  }

  /**
   * Pick the demonstration whose frame is most similar to the request, by the
   * combined endpoint distance after normalizing the demo onto the request's
   * start. This favors demonstrations with a similar direction/scale, which
   * keeps the retargeting transform close to the identity (least distortion).
   */
  private selectNearest(
    candidates: MovementDemonstration[],
    context: MovementContext,
  ): MovementDemonstration {
    let best = candidates[0]!;
    let bestScore = Number.POSITIVE_INFINITY;
    for (const demo of candidates) {
      const demoVec = { x: demo.frame.end.x - demo.frame.start.x, y: demo.frame.end.y - demo.frame.start.y };
      const ctxVec = { x: context.end.x - context.start.x, y: context.end.y - context.start.y };
      const score =
        euclideanDistance(demoVec, ctxVec) +
        Math.abs(Math.hypot(demoVec.x, demoVec.y) - Math.hypot(ctxVec.x, ctxVec.y));
      if (score < bestScore) {
        bestScore = score;
        best = demo;
      }
    }
    return best;
  }
}

/**
 * Deterministic, dependency-free reference backend. Pluggable: swap in an
 * on-device backend (e.g. an mlx-trained policy) by implementing
 * {@link MovementModelBackend} and returning a {@link TrainedMovementModel}.
 */
export class SimilarityTransformBackend implements MovementModelBackend {
  readonly name = "similarity-transform";

  async train(dataset: MovementDataset): Promise<TrainedMovementModel> {
    const byIntent = new Map<string, MovementDemonstration[]>();
    for (const intent of datasetIntents(dataset)) {
      byIntent.set(intent, []);
    }
    for (const demo of dataset.demonstrations) {
      const bucket = byIntent.get(demo.intent);
      if (bucket) {
        bucket.push(demo);
      }
    }
    return new SimilarityTransformModel(byIntent);
  }
}
