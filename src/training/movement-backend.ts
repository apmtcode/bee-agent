/**
 * Pluggable local-model backend for the movement-learning subsystem.
 *
 * Standing objective #2 asks bee-agent to (c) post-train a local model on
 * recorded movements and (d) generalize to new-but-related movements. The
 * on-device runtimes (mlx / axolotl, see `runner.ts`) only run on the user's
 * Apple-silicon machine and cannot execute in the cloud. This module provides
 * the backend-agnostic *interface* those runtimes will implement, plus a fully
 * deterministic in-process backend that trains and infers with no OS access —
 * so the whole capture → dataset → train → infer → generalize loop is testable
 * in CI. Real backends implement `LocalModelBackend` and register themselves in
 * a `MovementBackendRegistry`; the training runner picks a backend by id.
 */
import type { ReviewedExportManifest } from "./export-manifest.js";

export type MovementActionStep = {
  tool: string;
  summary: string;
  summaryTokens: string[];
};

export type MovementExample = {
  trajectoryId: string;
  sessionId: string;
  contextTokens: string[];
  contextSummary: string;
  actions: MovementActionStep[];
};

export type MovementDataset = {
  version: 1;
  examples: MovementExample[];
};

export type MovementTrainingConfig = {
  /** Policies return no actions when the best structural match scores below this. */
  minConfidence?: number;
};

/** A learned exemplar: structural skeleton, parameter slots, and the action sequence to emit. */
export type MovementExemplar = {
  trajectoryId: string;
  sessionId: string;
  contextTokens: string[];
  /** Context tokens that also flow into an action summary — the parameters. */
  slotTokens: string[];
  /** Context tokens that do not appear in any action — the structural intent. */
  skeletonTokens: string[];
  actions: MovementActionStep[];
};

/** JSON-serializable trained artifact — persists to disk like every other store here. */
export type MovementModelArtifact = {
  backendId: string;
  version: 1;
  config: Required<MovementTrainingConfig>;
  vocabulary: string[];
  exemplars: MovementExemplar[];
};

export type MovementPredictionInput = {
  goal?: string;
  tokens?: string[];
};

export type MovementSubstitution = { from: string; to: string };

export type MovementPrediction = {
  matchedTrajectoryId: string | null;
  confidence: number;
  generalized: boolean;
  substitutions: MovementSubstitution[];
  actions: MovementActionStep[];
};

export interface MovementPolicy {
  readonly backendId: string;
  predict(input: MovementPredictionInput): MovementPrediction;
}

export interface LocalModelBackend {
  readonly id: string;
  train(dataset: MovementDataset, config?: MovementTrainingConfig): Promise<MovementModelArtifact>;
  load(artifact: MovementModelArtifact): MovementPolicy;
}

export const DEFAULT_MOVEMENT_BACKEND_ID = "mock-deterministic";

const TOKEN_PATTERN = /[a-z0-9][a-z0-9._-]*/g;

export function tokenizeMovementText(text: string): string[] {
  return text.toLowerCase().match(TOKEN_PATTERN) ?? [];
}

function uniqueOrdered(tokens: Iterable<string>): string[] {
  const seen = new Set<string>();
  const out: string[] = [];
  for (const token of tokens) {
    if (!seen.has(token)) {
      seen.add(token);
      out.push(token);
    }
  }
  return out;
}

/**
 * Convert a reviewed export manifest's replay timelines into a movement dataset:
 * each replay becomes one example whose context is its observation/transcript
 * tokens and whose action sequence is the ordered `action` events.
 */
export function buildMovementDataset(manifest: ReviewedExportManifest): MovementDataset {
  const examples: MovementExample[] = [];

  for (const replay of manifest.replays) {
    const events = [...replay.events].sort((a, b) => a.ts - b.ts);
    const contextParts: string[] = [];
    const contextTokens: string[] = [];
    const actions: MovementActionStep[] = [];

    for (const event of events) {
      if (event.kind === "action") {
        actions.push({
          tool: event.tool,
          summary: event.summary,
          summaryTokens: tokenizeMovementText(event.summary),
        });
        continue;
      }
      const text = event.kind === "transcript" ? event.content : event.summary;
      contextParts.push(text);
      contextTokens.push(...tokenizeMovementText(text));
    }

    if (actions.length === 0) {
      continue;
    }

    const trajectoryId = replay.trajectoryIds[0] ?? replay.sessionId;
    examples.push({
      trajectoryId,
      sessionId: replay.sessionId,
      contextTokens: uniqueOrdered(contextTokens),
      contextSummary: contextParts.join(" "),
      actions,
    });
  }

  return { version: 1, examples };
}

function jaccard(a: Set<string>, b: Set<string>): number {
  if (a.size === 0 && b.size === 0) {
    return 0;
  }
  let intersection = 0;
  for (const value of a) {
    if (b.has(value)) {
      intersection += 1;
    }
  }
  return intersection / (a.size + b.size - intersection);
}

/**
 * Deterministic, dependency-free reference backend. It learns exemplars by
 * separating each example's context tokens into a structural *skeleton* and
 * parameter *slots* (context tokens that reappear inside an action summary).
 * At inference it retrieves the closest skeleton, then substitutes the query's
 * novel token(s) into the retrieved slots — reproducing recorded movements
 * exactly and generalizing to related ones (e.g. "open report.txt → type
 * report.txt" learned, "open budget.csv" yields "type budget.csv").
 */
export class DeterministicMockBackend implements LocalModelBackend {
  readonly id = DEFAULT_MOVEMENT_BACKEND_ID;

  async train(dataset: MovementDataset, config: MovementTrainingConfig = {}): Promise<MovementModelArtifact> {
    const resolvedConfig: Required<MovementTrainingConfig> = {
      minConfidence: config.minConfidence ?? 0,
    };
    const vocabulary = new Set<string>();
    const exemplars: MovementExemplar[] = [];

    for (const example of dataset.examples) {
      const actionTokens = new Set<string>();
      for (const action of example.actions) {
        for (const token of action.summaryTokens) {
          actionTokens.add(token);
        }
      }
      const slotTokens = example.contextTokens.filter((token) => actionTokens.has(token));
      const slotSet = new Set(slotTokens);
      const skeletonTokens = example.contextTokens.filter((token) => !slotSet.has(token));
      for (const token of example.contextTokens) {
        vocabulary.add(token);
      }
      exemplars.push({
        trajectoryId: example.trajectoryId,
        sessionId: example.sessionId,
        contextTokens: example.contextTokens,
        slotTokens,
        skeletonTokens,
        actions: example.actions,
      });
    }

    return {
      backendId: this.id,
      version: 1,
      config: resolvedConfig,
      vocabulary: [...vocabulary].sort(),
      exemplars,
    };
  }

  load(artifact: MovementModelArtifact): MovementPolicy {
    return new MockMovementPolicy(artifact);
  }
}

class MockMovementPolicy implements MovementPolicy {
  readonly backendId: string;

  constructor(private readonly artifact: MovementModelArtifact) {
    this.backendId = artifact.backendId;
  }

  predict(input: MovementPredictionInput): MovementPrediction {
    const queryTokens = uniqueOrdered(input.tokens ?? tokenizeMovementText(input.goal ?? ""));
    const querySet = new Set(queryTokens);

    let best: MovementExemplar | undefined;
    let bestConfidence = -1;
    for (const exemplar of this.artifact.exemplars) {
      const confidence = skeletonCoverage(querySet, exemplar.skeletonTokens);
      // Deterministic tie-break: higher coverage wins; ties keep earliest exemplar.
      if (confidence > bestConfidence) {
        best = exemplar;
        bestConfidence = confidence;
      }
    }

    const empty: MovementPrediction = {
      matchedTrajectoryId: null,
      confidence: 0,
      generalized: false,
      substitutions: [],
      actions: [],
    };
    if (!best || bestConfidence < this.artifact.config.minConfidence) {
      return empty;
    }

    const substitutions = buildSubstitutions(best, queryTokens);
    const substitutionMap = new Map(substitutions.map((sub) => [sub.from, sub.to] as const));
    const actions = best.actions.map((action) => applySubstitutions(action, substitutionMap));

    return {
      matchedTrajectoryId: best.trajectoryId,
      confidence: bestConfidence,
      generalized: substitutions.length > 0,
      substitutions,
      actions,
    };
  }
}

function skeletonCoverage(querySet: Set<string>, skeletonTokens: string[]): number {
  if (skeletonTokens.length === 0) {
    return 0;
  }
  let covered = 0;
  for (const token of skeletonTokens) {
    if (querySet.has(token)) {
      covered += 1;
    }
  }
  return covered / skeletonTokens.length;
}

/**
 * Infer parameter substitutions by aligning the exemplar's *changed* slots with
 * the query's *novel* tokens. A "missing slot" is a parameter token from the
 * recorded context that the query no longer mentions (the old value); a novel
 * token is one the query introduces that the exemplar never saw (the new value).
 * Slot tokens the query still mentions (e.g. shared structural verbs like
 * "close") are left untouched. Substitution only fires when the counts line up,
 * keeping the mapping unambiguous and deterministic.
 */
function buildSubstitutions(exemplar: MovementExemplar, queryTokens: string[]): MovementSubstitution[] {
  if (exemplar.slotTokens.length === 0) {
    return [];
  }
  const querySet = new Set(queryTokens);
  const contextSet = new Set(exemplar.contextTokens);
  const missingSlots = exemplar.slotTokens.filter((token) => !querySet.has(token));
  if (missingSlots.length === 0) {
    return [];
  }
  const novel = queryTokens.filter((token) => !contextSet.has(token));
  if (novel.length !== missingSlots.length) {
    return [];
  }
  return missingSlots.map((from, index) => ({ from, to: novel[index]! }));
}

function applySubstitutions(action: MovementActionStep, substitutions: Map<string, string>): MovementActionStep {
  if (substitutions.size === 0) {
    return action;
  }
  const summaryTokens = action.summaryTokens.map((token) => substitutions.get(token) ?? token);
  let summary = action.summary;
  for (const [from, to] of substitutions) {
    summary = replaceWholeToken(summary, from, to);
  }
  return { tool: action.tool, summary, summaryTokens };
}

function replaceWholeToken(text: string, from: string, to: string): string {
  const escaped = from.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  return text.replace(new RegExp(`(^|[^a-z0-9._-])${escaped}(?![a-z0-9._-])`, "gi"), (_match, prefix: string) => `${prefix}${to}`);
}

export type MovementEvalResult = {
  count: number;
  exactMatchRate: number;
  meanToolAccuracy: number;
  meanSummaryF1: number;
};

/**
 * Generalization eval harness: replay each held-out example through the policy
 * and score how faithfully the predicted action sequence matches the recorded
 * one (exact sequence match, per-step tool accuracy, summary-token F1).
 */
export function evaluateMovementPolicy(policy: MovementPolicy, heldOut: MovementExample[]): MovementEvalResult {
  if (heldOut.length === 0) {
    return { count: 0, exactMatchRate: 0, meanToolAccuracy: 0, meanSummaryF1: 0 };
  }
  let exact = 0;
  let toolAccuracySum = 0;
  let summaryF1Sum = 0;

  for (const example of heldOut) {
    const prediction = policy.predict({ tokens: example.contextTokens });
    const predicted = prediction.actions;
    const expected = example.actions;

    const stepCount = Math.max(predicted.length, expected.length);
    let toolMatches = 0;
    let f1Sum = 0;
    let sequenceMatch = predicted.length === expected.length;
    for (let index = 0; index < stepCount; index += 1) {
      const predictedStep = predicted[index];
      const expectedStep = expected[index];
      if (predictedStep && expectedStep && predictedStep.tool === expectedStep.tool) {
        toolMatches += 1;
      }
      if (
        !predictedStep ||
        !expectedStep ||
        predictedStep.tool !== expectedStep.tool ||
        predictedStep.summary !== expectedStep.summary
      ) {
        sequenceMatch = false;
      }
      f1Sum += tokenF1(predictedStep?.summaryTokens ?? [], expectedStep?.summaryTokens ?? []);
    }

    if (sequenceMatch) {
      exact += 1;
    }
    toolAccuracySum += stepCount === 0 ? 1 : toolMatches / stepCount;
    summaryF1Sum += stepCount === 0 ? 1 : f1Sum / stepCount;
  }

  return {
    count: heldOut.length,
    exactMatchRate: exact / heldOut.length,
    meanToolAccuracy: toolAccuracySum / heldOut.length,
    meanSummaryF1: summaryF1Sum / heldOut.length,
  };
}

function tokenF1(predicted: string[], expected: string[]): number {
  if (predicted.length === 0 && expected.length === 0) {
    return 1;
  }
  if (predicted.length === 0 || expected.length === 0) {
    return 0;
  }
  const expectedSet = new Set(expected);
  const predictedSet = new Set(predicted);
  let overlap = 0;
  for (const token of predictedSet) {
    if (expectedSet.has(token)) {
      overlap += 1;
    }
  }
  if (overlap === 0) {
    return 0;
  }
  const precision = overlap / predictedSet.size;
  const recall = overlap / expectedSet.size;
  return (2 * precision * recall) / (precision + recall);
}

/**
 * Registry of pluggable movement backends. Real on-device runtimes register a
 * factory under their id; callers resolve a backend by id at train time. The
 * deterministic mock is registered by default so cloud/CI always has a backend.
 */
export class MovementBackendRegistry {
  private readonly factories = new Map<string, () => LocalModelBackend>();

  constructor(seedDefault = true) {
    if (seedDefault) {
      this.register(DEFAULT_MOVEMENT_BACKEND_ID, () => new DeterministicMockBackend());
    }
  }

  register(id: string, factory: () => LocalModelBackend): this {
    this.factories.set(id, factory);
    return this;
  }

  has(id: string): boolean {
    return this.factories.has(id);
  }

  ids(): string[] {
    return [...this.factories.keys()].sort();
  }

  create(id: string = DEFAULT_MOVEMENT_BACKEND_ID): LocalModelBackend {
    const factory = this.factories.get(id);
    if (!factory) {
      throw new Error(`unknown movement backend: ${id} (registered: ${this.ids().join(", ") || "none"})`);
    }
    return factory();
  }
}
