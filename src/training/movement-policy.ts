import type { TrajectorySpan } from "../capture/trajectory.js";

/**
 * Movement-policy inference layer for the local-movement learning subsystem.
 *
 * The capture/replay/export/runner pipeline already turns recorded on-device
 * movements into a reviewed dataset and an on-device training *plan*. What was
 * missing is the inference side: a way to (c) repeat recorded movements and (d)
 * generalize to new-but-related movements once a policy has been trained.
 *
 * This module supplies:
 *   - a compact, replayable movement dataset schema derived from trajectories,
 *   - a *pluggable* `MovementPolicyBackend` seam so a real on-device small model
 *     can be dropped in later, and
 *   - a deterministic, dependency-free reference backend
 *     (`NearestNeighborMovementBackend`) that runs in the cloud/CI so the whole
 *     loop can be validated without a GPU or real OS input,
 *   - a generalization eval harness that measures replay fidelity on held-out
 *     trajectories,
 *   - a deterministic synthetic trajectory generator to exercise the loop
 *     without real mouse/keyboard/window capture.
 *
 * Everything here is pure and deterministic (no wall-clock, no RNG beyond a
 * seeded LCG) so tests are reproducible in the cloud.
 */

// ---------------------------------------------------------------------------
// Dataset schema
// ---------------------------------------------------------------------------

export type MovementContext = {
  /** Application the movement happened in, when known. */
  appId?: string;
  /** Foreground screen/window title, when known. */
  screenTitle?: string;
  /** Ordered observation summaries leading up to the action. */
  observationSummaries: string[];
  /** Tools of the actions already taken in this span before this one. */
  priorActionTools: string[];
  /**
   * Summaries of the actions already taken in this span before this one. This
   * is what makes the context sequence-aware: two consecutive movements on the
   * same screen differ by the trajectory that led to them, so the policy can
   * tell "step 2 after tapping send" from "step 1".
   */
  priorActionSummaries: string[];
};

export type MovementAction = {
  tool: string;
  summary: string;
  gesture?: string;
  target?: string;
  direction?: string;
};

export type MovementExample = {
  context: MovementContext;
  action: MovementAction;
};

export type MovementDataset = {
  version: 1;
  examples: MovementExample[];
};

/**
 * Turn reviewed/captured trajectory spans into a flat (context → next action)
 * dataset. Each action becomes one example whose context is everything the
 * model would legitimately know *before* taking it: the app/screen, the
 * observations up to that point, and the tools of the prior actions.
 */
export function buildMovementDataset(spans: TrajectorySpan[]): MovementDataset {
  const examples: MovementExample[] = [];
  for (const span of spans) {
    const appId = firstMetadataString(span, "appName") ?? firstMetadataString(span, "appId");
    const screenTitle = firstMetadataString(span, "screenTitle");
    span.actions.forEach((action, index) => {
      const observationSummaries = span.observations
        .filter((observation) => observation.ts <= action.ts)
        .map((observation) => observation.summary);
      const context: MovementContext = {
        ...(appId ? { appId } : {}),
        ...(screenTitle ? { screenTitle } : {}),
        observationSummaries,
        priorActionTools: span.actions.slice(0, index).map((prior) => prior.tool),
        priorActionSummaries: span.actions.slice(0, index).map((prior) => prior.summary),
      };
      examples.push({ context, action: toMovementAction(action) });
    });
  }
  return { version: 1, examples };
}

function toMovementAction(action: TrajectorySpan["actions"][number]): MovementAction {
  const metadata = action.metadata ?? {};
  return {
    tool: action.tool,
    summary: action.summary,
    ...(typeof metadata.gesture === "string" ? { gesture: metadata.gesture } : {}),
    ...(typeof metadata.target === "string" ? { target: metadata.target } : {}),
    ...(typeof metadata.direction === "string" ? { direction: metadata.direction } : {}),
  };
}

function firstMetadataString(span: TrajectorySpan, key: string): string | undefined {
  for (const observation of span.observations) {
    const value = observation.metadata?.[key];
    if (typeof value === "string" && value.length > 0) {
      return value;
    }
  }
  return undefined;
}

// ---------------------------------------------------------------------------
// Pluggable backend seam
// ---------------------------------------------------------------------------

export type MovementPrediction = {
  action: MovementAction;
  /** 0..1 similarity between the query context and the matched training case. */
  confidence: number;
  /** Index into the training dataset of the case this prediction came from. */
  matchedExampleIndex: number;
  /** True when the query context matched a training case exactly. */
  exact: boolean;
};

export interface TrainedMovementPolicy {
  readonly backendId: string;
  readonly exampleCount: number;
  /** Predict the next movement for a context, or undefined if nothing learned. */
  predict(context: MovementContext): MovementPrediction | undefined;
}

/**
 * The seam a real on-device model implements. `train` is async so a backend can
 * shell out to the runner / load weights; the reference backend resolves
 * synchronously.
 */
export interface MovementPolicyBackend {
  readonly id: string;
  train(dataset: MovementDataset): Promise<TrainedMovementPolicy>;
}

// ---------------------------------------------------------------------------
// Deterministic reference backend (k-nearest-neighbour over context tokens)
// ---------------------------------------------------------------------------

type IndexedExample = {
  example: MovementExample;
  tokens: Set<string>;
};

/**
 * A dependency-free, fully deterministic backend used as the default and for
 * tests. It memorises training contexts as token bags and, at inference time,
 * returns the action of the most similar context (Jaccard overlap). Identical
 * contexts reproduce the recorded movement exactly (confidence 1); related
 * contexts generalise to the nearest recorded movement with confidence < 1.
 */
export class NearestNeighborMovementBackend implements MovementPolicyBackend {
  readonly id = "nearest-neighbor";

  async train(dataset: MovementDataset): Promise<TrainedMovementPolicy> {
    const indexed: IndexedExample[] = dataset.examples.map((example) => ({
      example,
      tokens: contextTokens(example.context),
    }));
    return new NearestNeighborMovementPolicy(this.id, indexed);
  }
}

class NearestNeighborMovementPolicy implements TrainedMovementPolicy {
  constructor(
    readonly backendId: string,
    private readonly indexed: IndexedExample[],
  ) {}

  get exampleCount(): number {
    return this.indexed.length;
  }

  predict(context: MovementContext): MovementPrediction | undefined {
    if (this.indexed.length === 0) {
      return undefined;
    }
    const queryTokens = contextTokens(context);
    let bestIndex = -1;
    let bestScore = -1;
    for (let index = 0; index < this.indexed.length; index += 1) {
      const score = jaccard(queryTokens, this.indexed[index]!.tokens);
      // Strict `>` keeps the earliest example on ties → deterministic.
      if (score > bestScore) {
        bestScore = score;
        bestIndex = index;
      }
    }
    const match = this.indexed[bestIndex]!;
    return {
      action: match.example.action,
      confidence: round(bestScore),
      matchedExampleIndex: bestIndex,
      exact: bestScore >= 1,
    };
  }
}

export function contextTokens(context: MovementContext): Set<string> {
  const tokens = new Set<string>();
  if (context.appId) {
    for (const token of tokenize(context.appId)) {
      tokens.add(`app:${token}`);
    }
  }
  if (context.screenTitle) {
    for (const token of tokenize(context.screenTitle)) {
      tokens.add(`screen:${token}`);
    }
  }
  for (const summary of context.observationSummaries) {
    for (const token of tokenize(summary)) {
      tokens.add(`obs:${token}`);
    }
  }
  for (const tool of context.priorActionTools) {
    tokens.add(`prior:${tool.toLowerCase()}`);
  }
  context.priorActionSummaries.forEach((summary, step) => {
    for (const token of tokenize(summary)) {
      tokens.add(`step${step}:${token}`);
    }
  });
  return tokens;
}

function tokenize(value: string): string[] {
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

function round(value: number): number {
  return Math.round(value * 1e6) / 1e6;
}

// ---------------------------------------------------------------------------
// Generalization eval harness
// ---------------------------------------------------------------------------

export type MovementPolicyEval = {
  total: number;
  /** Held-out cases for which the policy produced any prediction. */
  predicted: number;
  /** Predictions whose matched context was an exact hit. */
  exactContextMatches: number;
  /** Predictions whose tool matched the held-out action's tool. */
  toolMatches: number;
  /** Predictions whose full summary matched the held-out action. */
  summaryMatches: number;
  toolMatchRate: number;
  summaryMatchRate: number;
  meanConfidence: number;
};

/**
 * Measure how well a trained policy reproduces held-out movements. Feed it
 * examples the policy was NOT trained on (but drawn from related trajectories)
 * to quantify generalization; feed it the training examples to confirm exact
 * replay.
 */
export function evaluateMovementPolicy(
  policy: TrainedMovementPolicy,
  heldOut: MovementExample[],
): MovementPolicyEval {
  let predicted = 0;
  let exactContextMatches = 0;
  let toolMatches = 0;
  let summaryMatches = 0;
  let confidenceSum = 0;

  for (const example of heldOut) {
    const prediction = policy.predict(example.context);
    if (!prediction) {
      continue;
    }
    predicted += 1;
    confidenceSum += prediction.confidence;
    if (prediction.exact) {
      exactContextMatches += 1;
    }
    if (prediction.action.tool === example.action.tool) {
      toolMatches += 1;
    }
    if (prediction.action.summary === example.action.summary) {
      summaryMatches += 1;
    }
  }

  return {
    total: heldOut.length,
    predicted,
    exactContextMatches,
    toolMatches,
    summaryMatches,
    toolMatchRate: predicted === 0 ? 0 : round(toolMatches / predicted),
    summaryMatchRate: predicted === 0 ? 0 : round(summaryMatches / predicted),
    meanConfidence: predicted === 0 ? 0 : round(confidenceSum / predicted),
  };
}

// ---------------------------------------------------------------------------
// Deterministic synthetic trajectory generator
// ---------------------------------------------------------------------------

/** Seeded linear-congruential generator — no Math.random, fully reproducible. */
function makeRng(seed: number): () => number {
  let state = (seed >>> 0) || 1;
  return () => {
    // Numerical Recipes LCG constants.
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 0x100000000;
  };
}

const SYNTHETIC_APPS = [
  { appId: "browser", screens: ["Inbox", "Compose", "Settings"], targets: ["send button", "search box", "reply link"] },
  { appId: "editor", screens: ["main.ts", "README", "diff view"], targets: ["save icon", "run button", "terminal"] },
  { appId: "chat", screens: ["General", "Direct", "Threads"], targets: ["message input", "emoji picker", "call button"] },
] as const;

const SYNTHETIC_GESTURES = ["tap", "type", "scroll", "shortcut"] as const;

/**
 * Produce deterministic synthetic movement trajectories for validating the
 * capture → dataset → train → infer loop without real OS input. Same seed →
 * byte-identical output.
 */
export function generateSyntheticMovementTrajectories(params: {
  seed: number;
  spanCount: number;
  sessionId?: string;
  actionsPerSpan?: number;
  baseTs?: number;
}): TrajectorySpan[] {
  const rng = makeRng(params.seed);
  const sessionId = params.sessionId ?? `synthetic-${params.seed}`;
  const actionsPerSpan = Math.max(1, params.actionsPerSpan ?? 3);
  const baseTs = params.baseTs ?? 1_700_000_000_000;
  const spans: TrajectorySpan[] = [];

  for (let s = 0; s < params.spanCount; s += 1) {
    const app = SYNTHETIC_APPS[Math.floor(rng() * SYNTHETIC_APPS.length)]!;
    const screen = app.screens[Math.floor(rng() * app.screens.length)]!;
    const spanTs = baseTs + s * 60_000;
    const observations: TrajectorySpan["observations"] = [
      {
        kind: "observation",
        source: "device",
        summary: `${app.appId} on ${screen}`,
        ts: spanTs,
        metadata: { appName: app.appId, screenTitle: screen, platform: "macos" },
      },
    ];
    const actions: TrajectorySpan["actions"] = [];
    for (let a = 0; a < actionsPerSpan; a += 1) {
      const gesture = SYNTHETIC_GESTURES[Math.floor(rng() * SYNTHETIC_GESTURES.length)]!;
      const target = app.targets[Math.floor(rng() * app.targets.length)]!;
      const ts = spanTs + (a + 1) * 1_000;
      actions.push({
        kind: "action",
        tool: "device",
        summary: `${gesture} ${target}`,
        ts,
        metadata: { gesture, target },
      });
    }
    spans.push({
      id: `${sessionId}-span-${s}`,
      sessionId,
      createdAt: new Date(spanTs).toISOString(),
      captureTier: "app",
      observations,
      actions,
      outcome: { status: "success", summary: `completed ${app.appId} flow` },
    });
  }

  return spans;
}
