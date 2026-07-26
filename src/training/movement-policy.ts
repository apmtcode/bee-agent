import type { DeviceGestureKind, DevicePlatform } from "../capture/device-adapter.js";
import type { TrajectorySpan } from "../capture/trajectory.js";

/**
 * In-process, pluggable movement-policy learning subsystem.
 *
 * This is the local model half of the movement-learning objective: given a
 * dataset of recorded movements (context -> gesture pairs) it can be trained to
 * (a) repeat the recorded movements exactly and (b) generalize to new but
 * related movements (e.g. the same gesture kind against a target it never saw
 * during training).
 *
 * The heavy, platform-specific runtimes (mlx / axolotl) stay behind the
 * {@link LocalAppleSiliconTrainingRunner}. This module provides a deterministic
 * backend that runs anywhere (cloud/CI included) so the pipeline — dataset ->
 * train -> predict -> evaluate — can be validated without real OS input or a
 * GPU. Real on-device backends implement the same {@link MovementPolicyBackend}
 * interface and are registered in the {@link MovementPolicyBackendRegistry}.
 */

export type MovementDirection = "up" | "down" | "left" | "right";

export type MovementGesture = {
  kind: DeviceGestureKind;
  target?: string;
  direction?: MovementDirection;
  valueSummary?: string;
};

/**
 * The observable state that precedes a movement. `targetHint` / `directionHint`
 * describe the element or direction the user *intends* to act on for this step;
 * a well-trained policy learns to fill those slots for contexts it has never
 * seen, which is what "generalize to new but related movements" means here.
 */
export type MovementContext = {
  appId: string;
  platform?: DevicePlatform;
  screenTitle?: string;
  targetHint?: string;
  directionHint?: MovementDirection;
  priorGestureKind?: DeviceGestureKind;
};

export type MovementSample = {
  context: MovementContext;
  gesture: MovementGesture;
};

export type MovementDataset = {
  samples: MovementSample[];
};

export type MovementPredictionSource = "recall" | "generalized" | "empty";

export type MovementPrediction = {
  gesture: MovementGesture;
  confidence: number;
  source: MovementPredictionSource;
  neighborDistance?: number;
};

export type SerializedMovementModel = {
  version: 1;
  backendId: string;
  samples: MovementSample[];
};

export interface MovementPolicyModel {
  readonly backendId: string;
  readonly sampleCount: number;
  predict(context: MovementContext): MovementPrediction;
  toJSON(): SerializedMovementModel;
}

export interface MovementPolicyBackend {
  readonly id: string;
  train(dataset: MovementDataset): MovementPolicyModel;
  load(serialized: SerializedMovementModel): MovementPolicyModel;
}

// --- Feature encoding -------------------------------------------------------

const STRUCTURAL_WEIGHTS: Record<string, number> = {
  appId: 3,
  screenTitle: 2,
  platform: 1,
  priorGestureKind: 1,
};

function normalize(value: string | undefined): string {
  return (value ?? "").trim().toLowerCase();
}

/** Full-context key used for exact recall of a recorded movement. */
function exactKey(context: MovementContext): string {
  return JSON.stringify([
    normalize(context.appId),
    normalize(context.platform),
    normalize(context.screenTitle),
    normalize(context.targetHint),
    normalize(context.directionHint),
    normalize(context.priorGestureKind),
  ]);
}

/** Structural distance used for nearest-neighbor generalization. */
function structuralDistance(a: MovementContext, b: MovementContext): number {
  let distance = 0;
  if (normalize(a.appId) !== normalize(b.appId)) distance += STRUCTURAL_WEIGHTS.appId;
  if (normalize(a.screenTitle) !== normalize(b.screenTitle)) distance += STRUCTURAL_WEIGHTS.screenTitle;
  if (normalize(a.platform) !== normalize(b.platform)) distance += STRUCTURAL_WEIGHTS.platform;
  if (normalize(a.priorGestureKind) !== normalize(b.priorGestureKind)) distance += STRUCTURAL_WEIGHTS.priorGestureKind;
  return distance;
}

function targetIsSlot(sample: MovementSample): boolean {
  return (
    sample.gesture.target !== undefined &&
    sample.context.targetHint !== undefined &&
    normalize(sample.gesture.target) === normalize(sample.context.targetHint)
  );
}

function directionIsSlot(sample: MovementSample): boolean {
  return (
    sample.gesture.direction !== undefined &&
    sample.context.directionHint !== undefined &&
    sample.gesture.direction === sample.context.directionHint
  );
}

function majority<T>(values: T[], keyOf: (value: T) => string): T | undefined {
  const counts = new Map<string, { count: number; value: T }>();
  for (const value of values) {
    const key = keyOf(value);
    const entry = counts.get(key);
    if (entry) {
      entry.count += 1;
    } else {
      counts.set(key, { count: 1, value });
    }
  }
  let best: { count: number; value: T } | undefined;
  for (const entry of counts.values()) {
    if (!best || entry.count > best.count) {
      best = entry;
    }
  }
  return best?.value;
}

// --- Reference backend ------------------------------------------------------

/**
 * Deterministic nearest-neighbor / slot-filling policy. Chosen as the default
 * because it is fully reproducible (no randomness, no external process) yet
 * still exhibits the two behaviours the objective requires: exact replay of
 * recorded movements, and generalization of a learned gesture to a new target
 * or direction within the same app/screen family.
 */
export const NEAREST_NEIGHBOR_BACKEND_ID = "nearest-neighbor-v1";

/**
 * Confidence ceiling for a generalized (inferred) prediction. Full certainty
 * (1.0) is reserved for exact recall of an observed movement, so downstream
 * consumers can threshold on it (e.g. auto-execute recalled moves, confirm
 * generalized ones). An inferred action is never treated as fully certain.
 */
const GENERALIZATION_PRIOR = 0.95;

class NearestNeighborMovementModel implements MovementPolicyModel {
  readonly backendId = NEAREST_NEIGHBOR_BACKEND_ID;
  private readonly exact = new Map<string, MovementGesture[]>();

  constructor(private readonly samples: MovementSample[]) {
    for (const sample of samples) {
      const key = exactKey(sample.context);
      const bucket = this.exact.get(key);
      if (bucket) {
        bucket.push(sample.gesture);
      } else {
        this.exact.set(key, [sample.gesture]);
      }
    }
  }

  get sampleCount(): number {
    return this.samples.length;
  }

  predict(context: MovementContext): MovementPrediction {
    if (this.samples.length === 0) {
      return { gesture: fallbackGesture(context), confidence: 0, source: "empty" };
    }

    const recalled = this.exact.get(exactKey(context));
    if (recalled && recalled.length > 0) {
      const gesture = majority(recalled, gestureKey) ?? recalled[0];
      return { gesture: cloneGesture(gesture), confidence: 1, source: "recall" };
    }

    // Generalize: find the structurally nearest recorded sample(s).
    let nearestDistance = Number.POSITIVE_INFINITY;
    for (const sample of this.samples) {
      const distance = structuralDistance(context, sample.context);
      if (distance < nearestDistance) {
        nearestDistance = distance;
      }
    }
    const nearest = this.samples.filter(
      (sample) => structuralDistance(context, sample.context) === nearestDistance,
    );

    const kind = (majority(nearest, (sample) => sample.gesture.kind) ?? nearest[0]).gesture.kind;
    const kindMatched = nearest.filter((sample) => sample.gesture.kind === kind);

    const gesture: MovementGesture = { kind };

    // Fill the target slot from the query hint when the neighbourhood learned
    // that the target is a slot; otherwise reuse the neighbour's literal target.
    if (context.targetHint !== undefined && kindMatched.some(targetIsSlot)) {
      gesture.target = context.targetHint;
    } else {
      const literal = kindMatched.find((sample) => sample.gesture.target !== undefined);
      if (literal?.gesture.target !== undefined) {
        gesture.target = literal.gesture.target;
      }
    }

    if (context.directionHint !== undefined && kindMatched.some(directionIsSlot)) {
      gesture.direction = context.directionHint;
    } else {
      const literal = kindMatched.find((sample) => sample.gesture.direction !== undefined);
      if (literal?.gesture.direction !== undefined) {
        gesture.direction = literal.gesture.direction;
      }
    }

    const agreementRatio = kindMatched.length / nearest.length;
    return {
      gesture,
      confidence: (GENERALIZATION_PRIOR * agreementRatio) / (1 + nearestDistance),
      source: "generalized",
      neighborDistance: nearestDistance,
    };
  }

  toJSON(): SerializedMovementModel {
    return {
      version: 1,
      backendId: this.backendId,
      samples: this.samples.map((sample) => ({
        context: { ...sample.context },
        gesture: cloneGesture(sample.gesture),
      })),
    };
  }
}

export class NearestNeighborMovementBackend implements MovementPolicyBackend {
  readonly id = NEAREST_NEIGHBOR_BACKEND_ID;

  train(dataset: MovementDataset): MovementPolicyModel {
    return new NearestNeighborMovementModel(
      dataset.samples.map((sample) => ({
        context: { ...sample.context },
        gesture: cloneGesture(sample.gesture),
      })),
    );
  }

  load(serialized: SerializedMovementModel): MovementPolicyModel {
    if (serialized.backendId !== this.id) {
      throw new Error(`Cannot load model for backend "${serialized.backendId}" with backend "${this.id}".`);
    }
    return new NearestNeighborMovementModel(serialized.samples.map((sample) => ({
      context: { ...sample.context },
      gesture: cloneGesture(sample.gesture),
    })));
  }
}

function fallbackGesture(context: MovementContext): MovementGesture {
  const gesture: MovementGesture = { kind: context.priorGestureKind ?? "tap" };
  if (context.targetHint !== undefined) gesture.target = context.targetHint;
  if (context.directionHint !== undefined) gesture.direction = context.directionHint;
  return gesture;
}

function cloneGesture(gesture: MovementGesture): MovementGesture {
  return {
    kind: gesture.kind,
    ...(gesture.target !== undefined ? { target: gesture.target } : {}),
    ...(gesture.direction !== undefined ? { direction: gesture.direction } : {}),
    ...(gesture.valueSummary !== undefined ? { valueSummary: gesture.valueSummary } : {}),
  };
}

function gestureKey(gesture: MovementGesture): string {
  return `${gesture.kind}|${gesture.target ?? ""}|${gesture.direction ?? ""}|${gesture.valueSummary ?? ""}`;
}

// --- Pluggable backend registry --------------------------------------------

export class MovementPolicyBackendRegistry {
  private readonly backends = new Map<string, MovementPolicyBackend>();

  constructor(backends: MovementPolicyBackend[] = [new NearestNeighborMovementBackend()]) {
    for (const backend of backends) {
      this.register(backend);
    }
  }

  register(backend: MovementPolicyBackend): void {
    this.backends.set(backend.id, backend);
  }

  get(id: string): MovementPolicyBackend {
    const backend = this.backends.get(id);
    if (!backend) {
      throw new Error(`Unknown movement-policy backend "${id}". Registered: ${this.list().join(", ") || "(none)"}.`);
    }
    return backend;
  }

  has(id: string): boolean {
    return this.backends.has(id);
  }

  list(): string[] {
    return [...this.backends.keys()];
  }
}

export function createDefaultMovementPolicyRegistry(): MovementPolicyBackendRegistry {
  return new MovementPolicyBackendRegistry();
}

// --- Dataset construction from captured trajectories ------------------------

const GESTURE_KINDS: DeviceGestureKind[] = ["tap", "swipe", "scroll", "type", "shortcut"];
const DIRECTIONS: MovementDirection[] = ["up", "down", "left", "right"];

function asGestureKind(value: unknown): DeviceGestureKind | undefined {
  return typeof value === "string" && (GESTURE_KINDS as string[]).includes(value)
    ? (value as DeviceGestureKind)
    : undefined;
}

function asDirection(value: unknown): MovementDirection | undefined {
  return typeof value === "string" && (DIRECTIONS as string[]).includes(value)
    ? (value as MovementDirection)
    : undefined;
}

function asString(value: unknown): string | undefined {
  return typeof value === "string" && value.length > 0 ? value : undefined;
}

/**
 * Reconstruct movement samples from recorded trajectories. Each device gesture
 * is turned into a training pair whose context is drawn from the most recent
 * device observation (app/screen/platform) and the previous gesture kind. This
 * mirrors exactly what {@link DeviceCaptureAdapter} writes into action/
 * observation metadata, so a captured -> exported dataset round-trips cleanly.
 */
export function buildMovementDataset(trajectories: TrajectorySpan[]): MovementDataset {
  const samples: MovementSample[] = [];

  for (const trajectory of trajectories) {
    const timeline = [...trajectory.observations, ...trajectory.actions].sort((a, b) => a.ts - b.ts);
    let appId: string | undefined;
    let platform: DevicePlatform | undefined;
    let screenTitle: string | undefined;
    let priorGestureKind: DeviceGestureKind | undefined;

    for (const event of timeline) {
      if (event.kind === "observation") {
        const metadata = event.metadata ?? {};
        appId = asString(metadata.appName) ?? asString(metadata.appId) ?? event.source ?? appId;
        platform = (asString(metadata.platform) as DevicePlatform | undefined) ?? platform;
        screenTitle = asString(metadata.screenTitle) ?? screenTitle;
        continue;
      }

      const metadata = event.metadata ?? {};
      const kind = asGestureKind(metadata.gesture);
      if (!kind) {
        continue;
      }
      const target = asString(metadata.target);
      const direction = asDirection(metadata.direction);
      const valueSummary = asString(metadata.valueSummary);

      const context: MovementContext = {
        appId: appId ?? event.tool,
        ...(platform ? { platform } : {}),
        ...(screenTitle ? { screenTitle } : {}),
        ...(target ? { targetHint: target } : {}),
        ...(direction ? { directionHint: direction } : {}),
        ...(priorGestureKind ? { priorGestureKind } : {}),
      };
      const gesture: MovementGesture = {
        kind,
        ...(target ? { target } : {}),
        ...(direction ? { direction } : {}),
        ...(valueSummary ? { valueSummary } : {}),
      };
      samples.push({ context, gesture });
      priorGestureKind = kind;
    }
  }

  return { samples };
}

// --- Generalization eval harness -------------------------------------------

export type MovementEvaluation = {
  total: number;
  kindMatches: number;
  targetMatches: number;
  directionMatches: number;
  exactMatches: number;
  recallCount: number;
  generalizedCount: number;
  kindAccuracy: number;
  targetAccuracy: number;
  exactAccuracy: number;
  meanConfidence: number;
};

/**
 * Measure replay/generalization fidelity of a trained model against a set of
 * held-out (but related) samples. Use with a train/held-out split of a
 * synthetic or captured dataset to quantify how well the policy generalizes.
 */
export function evaluateMovementPolicy(
  model: MovementPolicyModel,
  heldOut: MovementSample[],
): MovementEvaluation {
  let kindMatches = 0;
  let targetMatches = 0;
  let directionMatches = 0;
  let exactMatches = 0;
  let recallCount = 0;
  let generalizedCount = 0;
  let confidenceSum = 0;

  for (const sample of heldOut) {
    const prediction = model.predict(sample.context);
    confidenceSum += prediction.confidence;
    if (prediction.source === "recall") recallCount += 1;
    if (prediction.source === "generalized") generalizedCount += 1;

    const kindMatch = prediction.gesture.kind === sample.gesture.kind;
    const targetMatch = normalize(prediction.gesture.target) === normalize(sample.gesture.target);
    const directionMatch = (prediction.gesture.direction ?? undefined) === (sample.gesture.direction ?? undefined);
    if (kindMatch) kindMatches += 1;
    if (targetMatch) targetMatches += 1;
    if (directionMatch) directionMatches += 1;
    if (kindMatch && targetMatch && directionMatch) exactMatches += 1;
  }

  const total = heldOut.length;
  const ratio = (value: number): number => (total === 0 ? 0 : value / total);

  return {
    total,
    kindMatches,
    targetMatches,
    directionMatches,
    exactMatches,
    recallCount,
    generalizedCount,
    kindAccuracy: ratio(kindMatches),
    targetAccuracy: ratio(targetMatches),
    exactAccuracy: ratio(exactMatches),
    meanConfidence: ratio(confidenceSum),
  };
}
