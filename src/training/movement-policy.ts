import type { ExportedReplayManifest, ReviewedExportManifest } from "./export-manifest.js";

/**
 * Movement prediction / inference pipeline (standing objective #2, parts c & d).
 *
 * The capture subsystem records movements, the exporter turns reviewed
 * trajectories into a dataset, and the runner launches on-device training. This
 * module closes the loop: given a *learned* dataset it predicts the next
 * movement for a live context so bee-agent can (c) repeat recorded movements
 * and (d) generalize to new-but-related movements.
 *
 * The model backend is pluggable via {@link MovementModelBackend}. A real
 * on-device small model plugs in here; the bundled
 * {@link NearestNeighborMovementBackend} is a fully deterministic reference
 * implementation so cloud/CI tests exercise the whole pipeline without any OS
 * access or trained weights.
 */

/** A single learned (context -> next action) transition mined from the dataset. */
export type MovementTransition = {
  trajectoryId: string;
  /** Normalized feature tokens describing the situation before the action. */
  contextTokens: string[];
  /** The focus (app / screen / target) the recorded action operated on. */
  focus?: string;
  action: {
    tool: string;
    summary: string;
  };
};

/** The live situation we want a next-movement prediction for. */
export type MovementContext = {
  /** Recent event summaries (observations and actions), most recent last. */
  recentSummaries: string[];
  /** The app / screen / target currently in focus, if known. */
  focus?: string;
  /** Optional free-text goal to bias token matching. */
  goal?: string;
};

export type MovementPredictionSource = "recall" | "generalized" | "fallback";

export type PredictedMovement = {
  tool: string;
  summary: string;
  /** 0..1 similarity-derived confidence. */
  confidence: number;
  source: MovementPredictionSource;
  /** Trajectory the prediction was recalled from, when applicable. */
  basisTrajectoryId?: string;
};

/**
 * Pluggable model backend. Implementations range from the deterministic
 * nearest-neighbor reference below to a real trained on-device policy that
 * loads weights produced by {@link LocalAppleSiliconTrainingRunner}.
 */
export interface MovementModelBackend {
  readonly name: string;
  /** Fit / index the backend on the learned transitions. Idempotent. */
  train(transitions: readonly MovementTransition[]): void;
  /** Predict the next movement for a context. Returns undefined if unable. */
  predict(context: MovementContext): PredictedMovement | undefined;
}

const TOKEN_SPLIT = /[^a-z0-9]+/i;
const STOP_TOKENS = new Set(["the", "a", "an", "on", "of", "to", "in", "and", "with", "for", "device", "active"]);

/** Tokenize free text into lowercased, de-noised feature tokens. */
export function tokenize(text: string): string[] {
  return text
    .toLowerCase()
    .split(TOKEN_SPLIT)
    .map((token) => token.trim())
    .filter((token) => token.length > 1 && !STOP_TOKENS.has(token));
}

/** Build the feature token set for a context (recent summaries + goal + focus). */
export function contextTokens(context: MovementContext): string[] {
  const parts = [...context.recentSummaries];
  if (context.goal) {
    parts.push(context.goal);
  }
  if (context.focus) {
    parts.push(context.focus);
  }
  return dedupe(parts.flatMap(tokenize));
}

/**
 * Mine (context -> next action) transitions from a reviewed export's replays.
 * For each action event, the context is the window of events immediately
 * preceding it within the same replay.
 */
export function extractTransitions(
  manifest: Pick<ReviewedExportManifest, "replays">,
  options: { windowSize?: number } = {},
): MovementTransition[] {
  const windowSize = options.windowSize ?? 4;
  const transitions: MovementTransition[] = [];
  for (const replay of manifest.replays) {
    transitions.push(...extractReplayTransitions(replay, windowSize));
  }
  return transitions;
}

function extractReplayTransitions(replay: ExportedReplayManifest, windowSize: number): MovementTransition[] {
  const ordered = [...replay.events].sort((a, b) => a.ts - b.ts);
  const transitions: MovementTransition[] = [];
  for (let index = 0; index < ordered.length; index += 1) {
    const event = ordered[index];
    if (event.kind !== "action") {
      continue;
    }
    const priorEvents = ordered.slice(Math.max(0, index - windowSize), index);
    const contextText = priorEvents.map(eventSummary);
    const focus = mostRecentFocus(priorEvents);
    transitions.push({
      trajectoryId: event.trajectoryId,
      contextTokens: dedupe(contextText.flatMap(tokenize)),
      ...(focus ? { focus } : {}),
      action: { tool: event.tool, summary: event.summary },
    });
  }
  return transitions;
}

function eventSummary(event: ExportedReplayManifest["events"][number]): string {
  switch (event.kind) {
    case "transcript":
      return event.content;
    case "observation":
      return event.summary;
    case "action":
      return event.summary;
  }
}

function mostRecentFocus(priorEvents: ExportedReplayManifest["events"]): string | undefined {
  for (let index = priorEvents.length - 1; index >= 0; index -= 1) {
    const event = priorEvents[index];
    if (event.kind === "observation") {
      return event.summary;
    }
  }
  return undefined;
}

/** Jaccard similarity over two token sets. Deterministic, 0..1. */
export function jaccardSimilarity(a: readonly string[], b: readonly string[]): number {
  if (a.length === 0 && b.length === 0) {
    return 0;
  }
  const setA = new Set(a);
  const setB = new Set(b);
  let intersection = 0;
  for (const token of setA) {
    if (setB.has(token)) {
      intersection += 1;
    }
  }
  const union = setA.size + setB.size - intersection;
  return union === 0 ? 0 : intersection / union;
}

/**
 * Rewrite a recalled action summary so it applies to a new focus — the core of
 * generalization (objective #2 part d). If the recalled focus contributed a
 * distinctive token to the summary, swap it for the live focus's counterpart.
 */
export function generalizeSummary(summary: string, fromFocus: string | undefined, toFocus: string | undefined): string {
  if (!fromFocus || !toFocus) {
    return summary;
  }
  const toKey = focusKeyword(toFocus);
  if (!toKey) {
    return summary;
  }
  // Swap the first distinctive token of the recalled focus that actually
  // appears in the recalled summary for the live focus's keyword.
  for (const fromKey of tokenize(fromFocus)) {
    if (fromKey === toKey) {
      continue;
    }
    const pattern = new RegExp(`\\b${escapeRegExp(fromKey)}\\b`, "gi");
    if (pattern.test(summary)) {
      return summary.replace(pattern, toKey);
    }
  }
  return summary;
}

/** Distinctive keyword of a focus string (first non-stop token). */
function focusKeyword(focus: string): string | undefined {
  return tokenize(focus)[0];
}

/**
 * Deterministic reference backend: indexes transitions and answers with the
 * nearest recorded context by token similarity, generalizing the recalled
 * action to the live focus. No randomness, no external model — safe for CI.
 */
export class NearestNeighborMovementBackend implements MovementModelBackend {
  readonly name = "nearest-neighbor";

  private transitions: MovementTransition[] = [];

  constructor(private readonly options: { minConfidence?: number } = {}) {}

  train(transitions: readonly MovementTransition[]): void {
    this.transitions = [...transitions];
  }

  predict(context: MovementContext): PredictedMovement | undefined {
    if (this.transitions.length === 0) {
      return undefined;
    }
    const tokens = contextTokens(context);
    let best: { transition: MovementTransition; score: number } | undefined;
    for (const transition of this.transitions) {
      const score = jaccardSimilarity(tokens, transition.contextTokens);
      if (
        !best ||
        score > best.score ||
        // Deterministic tie-break: prefer the lexicographically smaller id.
        (score === best.score && transition.trajectoryId < best.transition.trajectoryId)
      ) {
        best = { transition, score };
      }
    }
    if (!best) {
      return undefined;
    }
    const minConfidence = this.options.minConfidence ?? 0;
    if (best.score < minConfidence) {
      return undefined;
    }
    const generalizedSummary = generalizeSummary(best.transition.action.summary, best.transition.focus, context.focus);
    const generalized = generalizedSummary !== best.transition.action.summary;
    return {
      tool: best.transition.action.tool,
      summary: generalizedSummary,
      confidence: round(best.score),
      source: generalized ? "generalized" : "recall",
      basisTrajectoryId: best.transition.trajectoryId,
    };
  }
}

export type MovementPolicyEngineOptions = {
  backend?: MovementModelBackend;
  windowSize?: number;
  /** Emitted when the backend cannot answer (e.g. empty dataset). */
  fallback?: PredictedMovement;
};

/**
 * High-level movement prediction engine: mines transitions from a reviewed
 * export, trains a (pluggable) backend, and predicts single steps or a rollout.
 */
export class MovementPolicyEngine {
  private readonly backend: MovementModelBackend;
  private readonly windowSize: number;
  private readonly fallback?: PredictedMovement;
  private trained = false;
  private transitionCount = 0;

  constructor(options: MovementPolicyEngineOptions = {}) {
    this.backend = options.backend ?? new NearestNeighborMovementBackend();
    this.windowSize = options.windowSize ?? 4;
    if (options.fallback) {
      this.fallback = options.fallback;
    }
  }

  get backendName(): string {
    return this.backend.name;
  }

  get learnedTransitionCount(): number {
    return this.transitionCount;
  }

  /** Train the backend from a reviewed export manifest. */
  trainFromExport(manifest: Pick<ReviewedExportManifest, "replays">): this {
    const transitions = extractTransitions(manifest, { windowSize: this.windowSize });
    this.backend.train(transitions);
    this.transitionCount = transitions.length;
    this.trained = true;
    return this;
  }

  /** Train directly from pre-mined transitions (e.g. a real model export). */
  trainFromTransitions(transitions: readonly MovementTransition[]): this {
    this.backend.train(transitions);
    this.transitionCount = transitions.length;
    this.trained = true;
    return this;
  }

  predictNext(context: MovementContext): PredictedMovement {
    if (!this.trained) {
      throw new Error("MovementPolicyEngine.predictNext called before training");
    }
    const prediction = this.backend.predict(context);
    if (prediction) {
      return prediction;
    }
    return this.fallback ?? { tool: "noop", summary: "no movement predicted", confidence: 0, source: "fallback" };
  }

  /**
   * Autoregressive rollout: predict a sequence of movements, feeding each
   * prediction back into the context window. Stops at maxSteps or when a
   * prediction falls below stopBelowConfidence.
   */
  predictSequence(
    context: MovementContext,
    options: { maxSteps?: number; stopBelowConfidence?: number } = {},
  ): PredictedMovement[] {
    const maxSteps = options.maxSteps ?? 5;
    const stopBelow = options.stopBelowConfidence ?? 0;
    const predictions: PredictedMovement[] = [];
    const recentSummaries = [...context.recentSummaries];
    for (let step = 0; step < maxSteps; step += 1) {
      const prediction = this.predictNext({ ...context, recentSummaries: [...recentSummaries] });
      if (prediction.source === "fallback" || prediction.confidence < stopBelow) {
        break;
      }
      predictions.push(prediction);
      recentSummaries.push(prediction.summary);
      if (recentSummaries.length > this.windowSize) {
        recentSummaries.shift();
      }
    }
    return predictions;
  }
}

function dedupe(tokens: string[]): string[] {
  return [...new Set(tokens)];
}

function round(value: number): number {
  return Math.round(value * 1000) / 1000;
}

function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}
