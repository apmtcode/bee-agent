import type { ReplayManifest, ReplayTimelineEvent } from "../capture/replay.js";

/**
 * Local-movement learning subsystem — model layer (objective #2, parts c & d).
 *
 * This module turns reviewed replay timelines into a supervised
 * "predict the next movement" dataset, trains a small local *movement policy*
 * on it, and measures how well that policy generalizes to new-but-related
 * movements it was not trained on.
 *
 * The model backend is pluggable: `MovementModelBackend` is the seam a real
 * on-device model (e.g. a tiny MLX/GGUF policy) plugs into later. The shipped
 * `MockNgramMovementBackend` is fully deterministic so the whole train → infer
 * → evaluate loop runs — and is testable — in the cloud with no OS access and
 * no randomness. Generalization comes from multi-level context backoff: an
 * unseen (surface, previous-action) pair falls back to surface-only, then
 * previous-action-only, then the global prior, so the policy still emits a
 * plausible movement instead of nothing.
 */

/** A normalized movement the policy observes or predicts. */
export type MovementAction = {
  /** Emitting tool/modality, e.g. "device" | "os" | "browser". */
  tool: string;
  /** Canonical action label, e.g. "device:tapped submit". */
  label: string;
};

/** Features the policy conditions on when predicting the next movement. */
export type MovementContext = {
  /** Active surface (modality + app/window token), e.g. "device/mail". */
  surface: string;
  /** Label of the immediately preceding action, if any. */
  previousAction?: string;
};

/** One supervised (context → next action) training example. */
export type MovementExample = {
  context: MovementContext;
  action: MovementAction;
};

export type MovementDataset = {
  version: 1;
  examples: MovementExample[];
  /** Distinct action labels observed, sorted — the policy's output vocabulary. */
  actionVocabulary: string[];
};

/** A single next-movement prediction. */
export type MovementPrediction = {
  action: MovementAction;
  /** Share of the mass on the chosen action at the matched backoff level. */
  confidence: number;
  /**
   * Which context level produced the prediction: 0 = full (surface + previous
   * action), 1 = surface only, 2 = previous action only, 3 = global prior,
   * -1 = the policy has no knowledge at all (empty training set).
   */
  backoffLevel: number;
};

/** The pluggable local-model backend seam. */
export interface MovementModelBackend {
  readonly id: string;
  train(dataset: MovementDataset): Promise<TrainedMovementPolicy>;
}

/** A trained policy that predicts and can be persisted/reloaded. */
export interface TrainedMovementPolicy {
  readonly backendId: string;
  predict(context: MovementContext): MovementPrediction;
  serialize(): SerializedMovementPolicy;
}

export type SerializedMovementPolicy = {
  version: 1;
  backendId: string;
  /** Per-level feature → (action label → count) tables. */
  levels: Array<Record<string, Record<string, number>>>;
  /** action label → tool, so predictions can be rehydrated to MovementAction. */
  actionTools: Record<string, string>;
};

const BACKOFF_LEVELS = 4;

// ---------------------------------------------------------------------------
// Dataset construction: reviewed replay timeline → supervised examples.
// ---------------------------------------------------------------------------

/**
 * Extract (context → next action) examples from a single replay timeline.
 * Observation events advance the "current surface"; each action event becomes
 * one labelled example conditioned on the surface and the previous action.
 */
export function extractMovementExamples(manifest: ReplayManifest): MovementExample[] {
  const examples: MovementExample[] = [];
  let surface = "unknown";
  let previousAction: string | undefined;

  for (const event of manifest.events) {
    if (event.kind === "observation") {
      surface = normalizeSurface(event);
      continue;
    }
    if (event.kind !== "action") {
      continue;
    }
    const action = normalizeAction(event);
    examples.push({
      context: previousAction === undefined ? { surface } : { surface, previousAction },
      action,
    });
    previousAction = action.label;
  }

  return examples;
}

/** Build a training dataset from many reviewed replay manifests. */
export function buildMovementDataset(manifests: ReplayManifest[]): MovementDataset {
  const examples = manifests.flatMap((manifest) => extractMovementExamples(manifest));
  const vocabulary = new Set<string>();
  for (const example of examples) {
    vocabulary.add(example.action.label);
  }
  return {
    version: 1,
    examples,
    actionVocabulary: [...vocabulary].sort(),
  };
}

function normalizeSurface(event: Extract<ReplayTimelineEvent, { kind: "observation" }>): string {
  const token = firstMeaningfulToken(event.summary);
  return `${event.source}/${token}`;
}

function normalizeAction(event: Extract<ReplayTimelineEvent, { kind: "action" }>): MovementAction {
  const summary = event.summary.trim().toLowerCase().replace(/\s+/g, " ");
  const label = `${event.tool}:${summary || "act"}`;
  return { tool: event.tool, label };
}

function firstMeaningfulToken(summary: string): string {
  const tokens = summary
    .toLowerCase()
    .split(/[^a-z0-9]+/)
    .filter((token) => token.length > 0 && !SURFACE_STOPWORDS.has(token));
  return tokens[0] ?? "unknown";
}

const SURFACE_STOPWORDS = new Set(["on", "in", "the", "a", "active", "device", "of", "at"]);

// ---------------------------------------------------------------------------
// Feature extraction (shared by training and inference so they can't drift).
// ---------------------------------------------------------------------------

/** Ordered feature keys, most-specific first, one per backoff level. */
export function contextFeatures(context: MovementContext): string[] {
  const prev = context.previousAction ?? "∅";
  return [
    `s=${context.surface}&p=${prev}`, // 0: full bigram
    `s=${context.surface}`, // 1: surface only
    `p=${prev}`, // 2: previous action only
    "*", // 3: global prior
  ];
}

// ---------------------------------------------------------------------------
// Mock n-gram backend + policy (deterministic; the default local backend).
// ---------------------------------------------------------------------------

class NgramMovementPolicy implements TrainedMovementPolicy {
  constructor(
    readonly backendId: string,
    private readonly levels: Array<Map<string, Map<string, number>>>,
    private readonly actionTools: Map<string, string>,
  ) {}

  predict(context: MovementContext): MovementPrediction {
    const features = contextFeatures(context);
    for (let level = 0; level < features.length; level += 1) {
      const table = this.levels[level]?.get(features[level] as string);
      if (!table || table.size === 0) {
        continue;
      }
      const { label, count, total } = argmax(table);
      return {
        action: { tool: this.actionTools.get(label) ?? "unknown", label },
        confidence: total === 0 ? 0 : count / total,
        backoffLevel: level,
      };
    }
    return { action: { tool: "unknown", label: "∅" }, confidence: 0, backoffLevel: -1 };
  }

  serialize(): SerializedMovementPolicy {
    return {
      version: 1,
      backendId: this.backendId,
      levels: this.levels.map((table) => {
        const record: Record<string, Record<string, number>> = {};
        for (const [feature, actions] of table) {
          record[feature] = Object.fromEntries(actions);
        }
        return record;
      }),
      actionTools: Object.fromEntries(this.actionTools),
    };
  }

  static deserialize(serialized: SerializedMovementPolicy): NgramMovementPolicy {
    const levels = serialized.levels.map((record) => {
      const table = new Map<string, Map<string, number>>();
      for (const [feature, actions] of Object.entries(record)) {
        table.set(feature, new Map(Object.entries(actions)));
      }
      return table;
    });
    return new NgramMovementPolicy(
      serialized.backendId,
      levels,
      new Map(Object.entries(serialized.actionTools)),
    );
  }
}

/**
 * Deterministic count-based backoff policy. Not a neural net — but it is a real
 * learned model with a genuine generalization mechanism (context backoff), and
 * it stands in for a pluggable on-device model so the pipeline is exercised
 * end-to-end in tests without OS access or training hardware.
 */
export class MockNgramMovementBackend implements MovementModelBackend {
  readonly id = "mock-ngram";

  async train(dataset: MovementDataset): Promise<TrainedMovementPolicy> {
    const levels: Array<Map<string, Map<string, number>>> = Array.from(
      { length: BACKOFF_LEVELS },
      () => new Map<string, Map<string, number>>(),
    );
    const actionTools = new Map<string, string>();

    for (const example of dataset.examples) {
      actionTools.set(example.action.label, example.action.tool);
      const features = contextFeatures(example.context);
      for (let level = 0; level < BACKOFF_LEVELS; level += 1) {
        const feature = features[level] as string;
        const table = levels[level] as Map<string, Map<string, number>>;
        let actions = table.get(feature);
        if (!actions) {
          actions = new Map<string, number>();
          table.set(feature, actions);
        }
        actions.set(example.action.label, (actions.get(example.action.label) ?? 0) + 1);
      }
    }

    return new NgramMovementPolicy(this.id, levels, actionTools);
  }
}

/** Reload a persisted policy (mock backend format). */
export function deserializeMovementPolicy(serialized: SerializedMovementPolicy): TrainedMovementPolicy {
  return NgramMovementPolicy.deserialize(serialized);
}

/**
 * Pluggable backend factory. Only the deterministic mock ships today; real
 * on-device backends register here without changing call sites.
 */
export function createMovementModelBackend(runtime: "mock-ngram" = "mock-ngram"): MovementModelBackend {
  switch (runtime) {
    case "mock-ngram":
      return new MockNgramMovementBackend();
    default:
      throw new Error(`unknown movement model backend: ${runtime as string}`);
  }
}

function argmax(table: Map<string, number>): { label: string; count: number; total: number } {
  let bestLabel = "";
  let bestCount = -1;
  let total = 0;
  // Deterministic: iterate in sorted key order, strict > so ties keep the
  // lexicographically smallest label.
  for (const [label, count] of [...table.entries()].sort((a, b) => (a[0] < b[0] ? -1 : 1))) {
    total += count;
    if (count > bestCount) {
      bestCount = count;
      bestLabel = label;
    }
  }
  return { label: bestLabel, count: Math.max(bestCount, 0), total };
}

// ---------------------------------------------------------------------------
// Generalization eval harness (objective #2, part d).
// ---------------------------------------------------------------------------

export type MovementEvaluation = {
  total: number;
  correct: number;
  /** Top-1 accuracy over held-out examples. */
  accuracy: number;
  /** Mean backoff level used (lower = more specific match = higher fidelity). */
  meanBackoffLevel: number;
  /** How often each backoff level (and -1 = no knowledge) was hit. */
  backoffHistogram: Record<string, number>;
  /**
   * Accuracy restricted to examples whose exact (surface, previous-action)
   * context was NOT present in training — the true generalization signal.
   */
  generalizationAccuracy: number;
  generalizationTotal: number;
};

/**
 * Evaluate a trained policy on held-out examples. Reports overall top-1
 * accuracy plus accuracy on genuinely novel contexts (those absent from the
 * training set), which is where backoff-driven generalization is measured.
 */
export function evaluateMovementGeneralization(
  policy: TrainedMovementPolicy,
  heldOut: MovementExample[],
  trainingContexts?: Iterable<MovementContext>,
): MovementEvaluation {
  const seenFullFeatures = new Set<string>();
  for (const context of trainingContexts ?? []) {
    seenFullFeatures.add(contextFeatures(context)[0] as string);
  }

  let correct = 0;
  let backoffSum = 0;
  let known = 0;
  const histogram: Record<string, number> = {};
  let generalizationCorrect = 0;
  let generalizationTotal = 0;

  for (const example of heldOut) {
    const prediction = policy.predict(example.context);
    const key = String(prediction.backoffLevel);
    histogram[key] = (histogram[key] ?? 0) + 1;
    if (prediction.backoffLevel >= 0) {
      backoffSum += prediction.backoffLevel;
      known += 1;
    }
    const hit = prediction.action.label === example.action.label;
    if (hit) {
      correct += 1;
    }
    const isNovel = !seenFullFeatures.has(contextFeatures(example.context)[0] as string);
    if (isNovel && seenFullFeatures.size > 0) {
      generalizationTotal += 1;
      if (hit) {
        generalizationCorrect += 1;
      }
    }
  }

  const total = heldOut.length;
  return {
    total,
    correct,
    accuracy: total === 0 ? 0 : correct / total,
    meanBackoffLevel: known === 0 ? 0 : backoffSum / known,
    backoffHistogram: histogram,
    generalizationTotal,
    generalizationAccuracy: generalizationTotal === 0 ? 0 : generalizationCorrect / generalizationTotal,
  };
}

// ---------------------------------------------------------------------------
// Synthetic event-stream generator (validate pipeline without real OS input).
// ---------------------------------------------------------------------------

export type SyntheticMovementOptions = {
  /** Deterministic seed — no Math.random is used anywhere. */
  seed: number;
  sessions: number;
  /** Actions per session. */
  stepsPerSession: number;
  /** Base epoch millis for the first event; each step advances by 1000ms. */
  startTs?: number;
};

/**
 * Generate deterministic synthetic replay manifests whose movements follow a
 * learnable structure: each surface has a preferred gesture and each gesture
 * biases the next surface, so a policy trained on some sessions can predict
 * held-out sessions well above chance. Used to validate the full
 * capture → dataset → train → evaluate loop with no OS access.
 */
export function generateSyntheticMovementReplays(options: SyntheticMovementOptions): ReplayManifest[] {
  const rand = mulberry32(options.seed >>> 0);
  const start = options.startTs ?? 1_700_000_000_000;
  const manifests: ReplayManifest[] = [];

  for (let session = 0; session < options.sessions; session += 1) {
    const events: ReplayTimelineEvent[] = [];
    const sessionId = `synthetic-${options.seed}-${session}`;
    let ts = start + session * options.stepsPerSession * 2000;
    let surfaceIndex = Math.floor(rand() * SYNTHETIC_SURFACES.length);

    for (let step = 0; step < options.stepsPerSession; step += 1) {
      const surface = SYNTHETIC_SURFACES[surfaceIndex] as SyntheticSurface;
      events.push({
        kind: "observation",
        ts,
        trajectoryId: sessionId,
        source: surface.source,
        summary: `${surface.app} active`,
      });
      ts += 1000;

      // Structured (learnable) gesture choice: 70% the surface's preferred
      // gesture, otherwise a deterministic alternate.
      const gesture =
        rand() < 0.7
          ? surface.preferredGesture
          : (surface.altGestures[Math.floor(rand() * surface.altGestures.length)] as string);
      events.push({
        kind: "action",
        ts,
        trajectoryId: sessionId,
        tool: surface.tool,
        summary: `${gesture} ${surface.targetToken}`,
      });
      ts += 1000;

      // Gesture biases the next surface, giving the bigram signal something to
      // learn beyond surface-only priors.
      surfaceIndex = gesture === surface.preferredGesture
        ? (surfaceIndex + 1) % SYNTHETIC_SURFACES.length
        : Math.floor(rand() * SYNTHETIC_SURFACES.length);
    }

    manifests.push({
      version: 1,
      sessionId,
      trajectoryIds: [sessionId],
      eventCount: events.length,
      events,
    });
  }

  return manifests;
}

type SyntheticSurface = {
  source: string;
  app: string;
  tool: string;
  targetToken: string;
  preferredGesture: string;
  altGestures: string[];
};

const SYNTHETIC_SURFACES: SyntheticSurface[] = [
  { source: "device", app: "mail", tool: "device", targetToken: "compose", preferredGesture: "tapped", altGestures: ["swiped", "scrolled"] },
  { source: "device", app: "browser", tool: "browser", targetToken: "address-bar", preferredGesture: "typed", altGestures: ["tapped"] },
  { source: "os", app: "editor", tool: "os", targetToken: "buffer", preferredGesture: "focused", altGestures: ["opened"] },
  { source: "device", app: "chat", tool: "device", targetToken: "thread", preferredGesture: "scrolled", altGestures: ["tapped", "typed"] },
];

/** Deterministic PRNG (mulberry32) — avoids Math.random for reproducibility. */
function mulberry32(seed: number): () => number {
  let state = seed;
  return () => {
    state |= 0;
    state = (state + 0x6d2b79f5) | 0;
    let t = Math.imul(state ^ (state >>> 15), 1 | state);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Split manifests into train/test by session index (deterministic). */
export function splitMovementReplays(
  manifests: ReplayManifest[],
  trainFraction: number,
): { train: ReplayManifest[]; test: ReplayManifest[] } {
  const cutoff = Math.max(1, Math.floor(manifests.length * trainFraction));
  return {
    train: manifests.slice(0, cutoff),
    test: manifests.slice(cutoff),
  };
}
