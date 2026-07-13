import type { ReviewedExportManifest, ExportedReplayManifest } from "./export-manifest.js";

/**
 * Movement-policy inference (standing objective 2d: "generalize to perform new
 * but related movements").
 *
 * The rest of the movement subsystem produces reviewed datasets and on-device
 * training *plans*. This module closes the loop on the inference side: given a
 * reviewed dataset (or a model artifact, once a real backend is plugged in) and
 * a new goal, it predicts the action sequence to perform.
 *
 * The backend is pluggable. A deterministic retrieval backend ships here so the
 * pipeline is fully testable in the cloud with no model weights; a real
 * on-device small-model backend (e.g. an mlx-served policy) can be registered
 * under the same interface later.
 */

/** A single recorded action within a demonstration, time-normalized so it can be replayed. */
export type MovementAction = {
  tool: string;
  summary: string;
  /** Milliseconds from the first action of the demonstration. */
  relativeTs: number;
};

/** One reviewed demonstration extracted from the dataset: context + the movements taken. */
export type MovementDemonstration = {
  trajectoryId: string;
  sessionId: string;
  /** Free-text describing the goal/context (transcript + observation summaries). */
  goalText: string;
  actions: MovementAction[];
  outcomeStatus?: "success" | "failure" | "aborted";
  reward?: number;
};

/** The request presented to a policy: a new goal, optional live cues, and optional slot overrides. */
export type MovementPolicyContext = {
  /** Natural-language description of the new goal to perform. */
  goal: string;
  /** Optional live observation summaries ("cues") describing current screen/app state. */
  cue?: string[];
  /**
   * Literal substring replacements applied to the predicted action summaries,
   * e.g. `{ reports: "invoices" }`. Always applied when present; forces
   * `generalized: true`.
   */
  overrides?: Record<string, string>;
  /** Predictions scoring below this confidence are returned with `actions: []`. */
  minConfidence?: number;
};

/** A single predicted movement step. */
export type PredictedAction = MovementAction;

/** The policy's answer for one context. */
export type PredictedMovement = {
  backendId: string;
  /** The demonstration the prediction was derived from, if any. */
  sourceTrajectoryId?: string;
  /** 0..1 similarity/confidence score. */
  confidence: number;
  /** True when the predicted actions were adapted (slot-substituted) from the source demo. */
  generalized: boolean;
  actions: PredictedAction[];
};

/** Pluggable inference backend. Deterministic given the same dataset + context. */
export interface MovementPolicyBackend {
  readonly id: string;
  predict(context: MovementPolicyContext): PredictedMovement;
}

export type MovementPolicyBackendOptions = {
  /** Weight applied to demonstrations whose outcome succeeded (default 1.25). */
  successBoost?: number;
  /** Automatically substitute a single differing noun between goal and demo (default true). */
  autoGeneralize?: boolean;
};

export type MovementPolicyBackendFactory = (
  dataset: MovementDemonstration[],
  options?: MovementPolicyBackendOptions,
) => MovementPolicyBackend;

const STOPWORDS = new Set([
  "the",
  "a",
  "an",
  "to",
  "of",
  "in",
  "on",
  "for",
  "and",
  "or",
  "with",
  "then",
  "please",
  "open",
  "click",
  "go",
  "into",
  "my",
  "this",
  "that",
  "it",
  "is",
  "at",
  "by",
  "from",
]);

export function tokenizeMovementText(text: string): string[] {
  return text
    .toLowerCase()
    .split(/[^a-z0-9]+/)
    .filter((token) => token.length >= 2 && !STOPWORDS.has(token));
}

/**
 * Deterministic retrieval policy: scores each demonstration by weighted Jaccard
 * similarity of its goal tokens to the context tokens (goal + cues), picks the
 * best, and optionally generalizes by substituting the single differing noun.
 */
export class RetrievalMovementPolicyBackend implements MovementPolicyBackend {
  readonly id = "retrieval-v1";

  private readonly successBoost: number;
  private readonly autoGeneralize: boolean;
  private readonly index: Array<{ demo: MovementDemonstration; tokens: Set<string> }>;

  constructor(dataset: MovementDemonstration[], options: MovementPolicyBackendOptions = {}) {
    this.successBoost = options.successBoost ?? 1.25;
    this.autoGeneralize = options.autoGeneralize ?? true;
    this.index = dataset.map((demo) => ({
      demo,
      tokens: new Set(tokenizeMovementText(demo.goalText)),
    }));
  }

  predict(context: MovementPolicyContext): PredictedMovement {
    const contextTokens = new Set(
      tokenizeMovementText([context.goal, ...(context.cue ?? [])].join(" ")),
    );

    // Rank on the uncapped score (so the success boost breaks ties between
    // equally-similar demos); report a 0..1-clamped confidence to callers.
    let best: { demo: MovementDemonstration; tokens: Set<string>; score: number } | undefined;
    for (const entry of this.index) {
      const similarity = weightedJaccard(contextTokens, entry.tokens);
      const boost = entry.demo.outcomeStatus === "success" ? this.successBoost : 1;
      const score = similarity * boost;
      if (!best || score > best.score) {
        best = { demo: entry.demo, tokens: entry.tokens, score };
      }
    }

    const confidence = best ? Math.min(1, best.score) : 0;
    const minConfidence = context.minConfidence ?? 0;
    if (!best || best.score <= 0 || confidence < minConfidence) {
      return {
        backendId: this.id,
        sourceTrajectoryId: best?.demo.trajectoryId,
        confidence,
        generalized: false,
        actions: [],
      };
    }

    const { actions, generalized } = adaptActions(best.demo.actions, best.tokens, contextTokens, {
      autoGeneralize: this.autoGeneralize,
      overrides: context.overrides,
    });

    return {
      backendId: this.id,
      sourceTrajectoryId: best.demo.trajectoryId,
      confidence,
      generalized,
      actions,
    };
  }
}

function weightedJaccard(a: Set<string>, b: Set<string>): number {
  if (a.size === 0 || b.size === 0) {
    return 0;
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

/**
 * Adapt a demonstration's actions to a new context. If exactly one salient noun
 * differs between the demo's goal tokens and the context tokens, substitute it
 * throughout the action summaries — a deterministic form of generalization.
 * Explicit overrides are always applied on top.
 */
function adaptActions(
  actions: MovementAction[],
  demoTokens: Set<string>,
  contextTokens: Set<string>,
  options: { autoGeneralize: boolean; overrides?: Record<string, string> },
): { actions: MovementAction[]; generalized: boolean } {
  const substitutions: Array<{ from: string; to: string }> = [];

  if (options.autoGeneralize) {
    const added = [...contextTokens].filter((token) => !demoTokens.has(token));
    const removed = [...demoTokens].filter((token) => !contextTokens.has(token));
    if (added.length === 1 && removed.length === 1) {
      substitutions.push({ from: removed[0]!, to: added[0]! });
    }
  }

  for (const [from, to] of Object.entries(options.overrides ?? {})) {
    substitutions.push({ from, to });
  }

  if (substitutions.length === 0) {
    return { actions: actions.map((action) => ({ ...action })), generalized: false };
  }

  const adapted = actions.map((action) => ({
    ...action,
    summary: applySubstitutions(action.summary, substitutions),
  }));
  const generalized = adapted.some((action, i) => action.summary !== actions[i]!.summary);
  return { actions: adapted, generalized };
}

function applySubstitutions(text: string, substitutions: Array<{ from: string; to: string }>): string {
  let result = text;
  for (const { from, to } of substitutions) {
    if (!from) {
      continue;
    }
    result = result.replace(new RegExp(`\\b${escapeRegExp(from)}\\b`, "gi"), to);
  }
  return result;
}

function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

/** Registry of pluggable backends keyed by a kind string. */
export class MovementPolicyRegistry {
  private readonly factories = new Map<string, MovementPolicyBackendFactory>();

  register(kind: string, factory: MovementPolicyBackendFactory): this {
    this.factories.set(kind, factory);
    return this;
  }

  has(kind: string): boolean {
    return this.factories.has(kind);
  }

  kinds(): string[] {
    return [...this.factories.keys()];
  }

  create(
    kind: string,
    dataset: MovementDemonstration[],
    options?: MovementPolicyBackendOptions,
  ): MovementPolicyBackend {
    const factory = this.factories.get(kind);
    if (!factory) {
      throw new Error(`unknown movement policy backend: ${kind}`);
    }
    return factory(dataset, options);
  }
}

export function defaultMovementPolicyRegistry(): MovementPolicyRegistry {
  return new MovementPolicyRegistry().register(
    "retrieval",
    (dataset, options) => new RetrievalMovementPolicyBackend(dataset, options),
  );
}

/** Build the movement demonstration dataset from reviewed export replays. */
export function buildMovementDataset(manifest: ReviewedExportManifest): MovementDemonstration[] {
  const rewardByTrajectory = new Map(
    manifest.trajectories.map((trajectory) => [
      trajectory.id,
      { status: trajectory.outcomeStatus, reward: trajectory.reward },
    ]),
  );
  return manifest.replays.flatMap((replay) =>
    buildDemonstrationsFromReplay(replay, rewardByTrajectory),
  );
}

function buildDemonstrationsFromReplay(
  replay: ExportedReplayManifest,
  rewardByTrajectory: Map<string, { status?: "success" | "failure" | "aborted"; reward?: number }>,
): MovementDemonstration[] {
  const actions: MovementAction[] = [];
  const goalParts: string[] = [];
  let trajectoryId = replay.trajectoryIds[0] ?? replay.sessionId;
  const rawActions: Array<{ tool: string; summary: string; ts: number }> = [];

  for (const event of replay.events) {
    if (event.kind === "action") {
      rawActions.push({ tool: event.tool, summary: event.summary, ts: event.ts });
      trajectoryId = event.trajectoryId;
    } else if (event.kind === "observation") {
      goalParts.push(event.summary);
    } else if (event.kind === "transcript" && (event.role === "user" || event.role === "assistant")) {
      goalParts.push(event.content);
    }
  }

  if (rawActions.length === 0) {
    return [];
  }

  const firstTs = Math.min(...rawActions.map((action) => action.ts));
  for (const action of rawActions.sort((a, b) => a.ts - b.ts)) {
    actions.push({ tool: action.tool, summary: action.summary, relativeTs: action.ts - firstTs });
  }

  const outcome = rewardByTrajectory.get(trajectoryId);
  return [
    {
      trajectoryId,
      sessionId: replay.sessionId,
      goalText: goalParts.join(" \n "),
      actions,
      ...(outcome?.status ? { outcomeStatus: outcome.status } : {}),
      ...(outcome?.reward !== undefined ? { reward: outcome.reward } : {}),
    },
  ];
}

export type MovementInferenceServiceOptions = MovementPolicyBackendOptions & {
  backendKind?: string;
  registry?: MovementPolicyRegistry;
};

/**
 * High-level entry point: build a policy from a reviewed export and predict
 * movements for new goals. Backend selection is pluggable via the registry.
 */
export class MovementInferenceService {
  private constructor(
    private readonly backend: MovementPolicyBackend,
    readonly datasetSize: number,
  ) {}

  static fromDataset(
    dataset: MovementDemonstration[],
    options: MovementInferenceServiceOptions = {},
  ): MovementInferenceService {
    const registry = options.registry ?? defaultMovementPolicyRegistry();
    const backend = registry.create(options.backendKind ?? "retrieval", dataset, options);
    return new MovementInferenceService(backend, dataset.length);
  }

  static fromExport(
    manifest: ReviewedExportManifest,
    options: MovementInferenceServiceOptions = {},
  ): MovementInferenceService {
    return MovementInferenceService.fromDataset(buildMovementDataset(manifest), options);
  }

  get backendId(): string {
    return this.backend.id;
  }

  predict(context: MovementPolicyContext): PredictedMovement {
    return this.backend.predict(context);
  }
}
