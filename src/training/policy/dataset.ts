/**
 * Turns recorded {@link TrajectorySpan}s (the capture subsystem's output) into a
 * supervised {@link MovementDataset} of (context -> next-action) examples. This is
 * the bridge between the capture pipeline and the training/inference backend.
 */

import type {
  TrajectoryAction,
  TrajectoryObservation,
  TrajectorySpan,
} from "../../capture/trajectory.js";
import {
  actionKey,
  normalizeTokens,
  type MovementActionLabel,
  type MovementContext,
  type MovementDataset,
  type MovementTrainingExample,
} from "./model.js";

const STOPWORDS = new Set([
  "the",
  "a",
  "an",
  "on",
  "in",
  "to",
  "of",
  "into",
  "and",
  "with",
  "for",
  "at",
]);

/** Metadata keys that carry situation signal worth featurizing. */
const OBSERVATION_META_KEYS = [
  "event",
  "platform",
  "appName",
  "windowTitle",
  "filePath",
  "commandSummary",
  "target",
  "screenTitle",
] as const;

export function tokenizeSummary(text: string): string[] {
  return text
    .toLowerCase()
    .split(/[^a-z0-9]+/)
    .filter((token) => token.length > 0 && !STOPWORDS.has(token));
}

function slug(value: string): string {
  return tokenizeSummary(value).join("-") || value.trim().toLowerCase();
}

/** Feature tokens for a single observation. */
export function observationTokens(observation: TrajectoryObservation): string[] {
  const tokens = [`src:${observation.source}`, ...tokenizeSummary(observation.summary)];
  const metadata = observation.metadata ?? {};
  for (const key of OBSERVATION_META_KEYS) {
    const value = metadata[key];
    if (typeof value === "string" && value.trim().length > 0) {
      tokens.push(`${key}:${slug(value)}`);
    }
  }
  return tokens;
}

/** Normalize a recorded action into a stable, replayable label. */
export function labelAction(action: TrajectoryAction): MovementActionLabel {
  const metadata = action.metadata ?? {};
  const parts: string[] = [];
  const gesture = metadata.gesture;
  if (typeof gesture === "string" && gesture.trim().length > 0) {
    parts.push(gesture.trim().toLowerCase());
  }
  const target = metadata.target;
  if (typeof target === "string" && target.trim().length > 0) {
    parts.push(slug(target));
  }
  const direction = metadata.direction;
  if (typeof direction === "string" && direction.trim().length > 0) {
    parts.push(direction.trim().toLowerCase());
  }
  const descriptor = parts.length > 0 ? parts.join(":") : slug(action.summary);
  return { tool: action.tool, descriptor };
}

type TimelineEntry =
  | { ts: number; order: number; kind: "observation"; observation: TrajectoryObservation }
  | { ts: number; order: number; kind: "action"; action: TrajectoryAction };

function buildTimeline(trajectory: TrajectorySpan): TimelineEntry[] {
  const entries: TimelineEntry[] = [
    ...trajectory.observations.map<TimelineEntry>((observation, index) => ({
      ts: observation.ts,
      order: index,
      kind: "observation",
      observation,
    })),
    ...trajectory.actions.map<TimelineEntry>((action, index) => ({
      ts: action.ts,
      order: index,
      kind: "action",
      action,
    })),
  ];
  // Stable chronological order; observations sort before actions at the same ts so
  // the situation that motivated an action is always in its context window.
  return entries.sort((a, b) => {
    if (a.ts !== b.ts) {
      return a.ts - b.ts;
    }
    if (a.kind !== b.kind) {
      return a.kind === "observation" ? -1 : 1;
    }
    return a.order - b.order;
  });
}

function exampleWeight(trajectory: TrajectorySpan): number {
  const reward = trajectory.outcome?.reward;
  if (typeof reward === "number" && Number.isFinite(reward) && reward > 0) {
    return reward;
  }
  return 1;
}

/**
 * Emit one (context -> action) example per recorded action. Context = feature
 * tokens of the observations seen since the previous action, plus a `prev:` token
 * for the previous action (so sequence is captured without a fixed n-gram width).
 */
export function trajectoryToExamples(trajectory: TrajectorySpan): MovementTrainingExample[] {
  const timeline = buildTimeline(trajectory);
  const weight = exampleWeight(trajectory);
  const examples: MovementTrainingExample[] = [];

  let pendingObservationTokens: string[] = [];
  let previousActionKey: string | undefined;

  for (const entry of timeline) {
    if (entry.kind === "observation") {
      pendingObservationTokens.push(...observationTokens(entry.observation));
      continue;
    }

    const contextTokens = [...pendingObservationTokens];
    if (previousActionKey) {
      contextTokens.push(`prev:${previousActionKey}`);
    }
    const context: MovementContext = { tokens: normalizeTokens(contextTokens) };
    const action = labelAction(entry.action);
    examples.push({
      context,
      action,
      weight,
      sourceTrajectoryId: trajectory.id,
    });

    previousActionKey = actionKey(action);
    pendingObservationTokens = [];
  }

  return examples;
}

/** Build a dataset from many trajectories. */
export function buildMovementDataset(trajectories: TrajectorySpan[]): MovementDataset {
  const examples = trajectories.flatMap((trajectory) => trajectoryToExamples(trajectory));
  const actionVocabulary = [...new Set(examples.map((example) => actionKey(example.action)))].sort();
  return { version: 1, examples, actionVocabulary };
}
