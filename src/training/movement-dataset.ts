/**
 * Local-movement learning subsystem — dataset layer.
 *
 * Turns captured movements (reviewed trajectories, or the replay events an
 * export manifest carries) into the structured, replayable
 * {@link MovementDataset} the model backend trains on, and provides a
 * generalization eval harness to measure how well a trained model repeats
 * held-out but related sequences.
 */
import type { ReplayTimelineEvent } from "../capture/replay.js";
import type { TrajectorySpan } from "../capture/trajectory.js";
import {
  MOVEMENT_END_TOKEN,
  MOVEMENT_START_TOKEN,
  type MovementDataset,
  type MovementEvent,
  type MovementSequence,
  type TrainedMovementModel,
} from "./movement-model.js";

/** Normalize a free-text summary into a stable single-verb token component. */
export function normalizeMovementVerb(summary: string): string {
  const first = summary.trim().toLowerCase().split(/\s+/)[0] ?? "";
  const cleaned = first.replace(/[^a-z0-9-]/g, "");
  return cleaned.length > 0 ? cleaned : "act";
}

function movementToken(kind: MovementEvent["kind"], channel: string, verb: string): string {
  const safeChannel = channel.trim().toLowerCase().replace(/[^a-z0-9-]/g, "") || "unknown";
  return `${kind}:${safeChannel}:${verb}`;
}

/** Convert one replay timeline into ordered movement events (drops transcript). */
export function replayEventsToMovements(events: ReplayTimelineEvent[]): MovementEvent[] {
  return [...events]
    .filter((event): event is Extract<ReplayTimelineEvent, { kind: "observation" | "action" }> =>
      event.kind === "observation" || event.kind === "action",
    )
    .sort((a, b) => a.ts - b.ts)
    .map((event) => {
      if (event.kind === "observation") {
        const verb = normalizeMovementVerb(event.summary);
        return {
          ts: event.ts,
          kind: "observation" as const,
          channel: event.source,
          verb,
          target: event.summary,
          token: movementToken("observation", event.source, verb),
        };
      }
      const verb = normalizeMovementVerb(event.summary);
      return {
        ts: event.ts,
        kind: "action" as const,
        channel: event.tool,
        verb,
        target: event.summary,
        token: movementToken("action", event.tool, verb),
      };
    });
}

/** Convert a captured trajectory span into ordered movement events. */
export function trajectoryToMovements(trajectory: TrajectorySpan): MovementEvent[] {
  const observations: MovementEvent[] = trajectory.observations.map((observation) => {
    const verb = normalizeMovementVerb(observation.summary);
    return {
      ts: observation.ts,
      kind: "observation" as const,
      channel: observation.source,
      verb,
      target: observation.summary,
      token: movementToken("observation", observation.source, verb),
    };
  });
  const actions: MovementEvent[] = trajectory.actions.map((action) => {
    const verb = normalizeMovementVerb(action.summary);
    return {
      ts: action.ts,
      kind: "action" as const,
      channel: action.tool,
      verb,
      target: action.summary,
      token: movementToken("action", action.tool, verb),
    };
  });
  return [...observations, ...actions].sort((a, b) => a.ts - b.ts);
}

function assembleDataset(sequences: MovementSequence[], createdAt: string): MovementDataset {
  const vocabulary = new Set<string>([MOVEMENT_START_TOKEN, MOVEMENT_END_TOKEN]);
  let tokenCount = 0;
  for (const sequence of sequences) {
    for (const token of sequence.tokens) {
      vocabulary.add(token);
      tokenCount += 1;
    }
  }
  return {
    version: 1,
    createdAt,
    vocabulary: [...vocabulary].sort(),
    sequences,
    tokenCount,
  };
}

export type BuildMovementDatasetOptions = {
  createdAt?: string;
  /** Drop sequences shorter than this many movements (default 1). */
  minSequenceLength?: number;
};

/** Build a dataset from captured trajectory spans. */
export function buildMovementDatasetFromTrajectories(
  trajectories: TrajectorySpan[],
  options: BuildMovementDatasetOptions = {},
): MovementDataset {
  const minLength = options.minSequenceLength ?? 1;
  const sequences: MovementSequence[] = trajectories
    .map((trajectory) => {
      const events = trajectoryToMovements(trajectory);
      return {
        id: trajectory.id,
        sourceTrajectoryIds: [trajectory.id],
        events,
        tokens: events.map((event) => event.token),
      };
    })
    .filter((sequence) => sequence.tokens.length >= minLength);
  return assembleDataset(sequences, options.createdAt ?? new Date().toISOString());
}

export type ReplaySequenceInput = {
  id: string;
  trajectoryIds?: string[];
  events: ReplayTimelineEvent[];
};

/** Build a dataset from exported replay manifests. */
export function buildMovementDatasetFromReplays(
  replays: ReplaySequenceInput[],
  options: BuildMovementDatasetOptions = {},
): MovementDataset {
  const minLength = options.minSequenceLength ?? 1;
  const sequences: MovementSequence[] = replays
    .map((replay) => {
      const events = replayEventsToMovements(replay.events);
      return {
        id: replay.id,
        sourceTrajectoryIds: replay.trajectoryIds ?? [],
        events,
        tokens: events.map((event) => event.token),
      };
    })
    .filter((sequence) => sequence.tokens.length >= minLength);
  return assembleDataset(sequences, options.createdAt ?? new Date().toISOString());
}

export type MovementGeneralizationReport = {
  sequenceCount: number;
  /** Next-token predictions attempted (one per non-first token). */
  predictions: number;
  correct: number;
  /** Fraction of next tokens predicted correctly on held-out sequences. */
  accuracy: number;
  /** How many correct predictions relied on backoff (context unseen at full order). */
  backoffCorrect: number;
  /** Fraction of full sequences reproduced exactly by greedy generation. */
  exactSequenceMatch: number;
};

/**
 * Measure generalization: for each held-out sequence, walk its true tokens and
 * ask the model to predict each next token from the true prefix (teacher
 * forcing), then separately check whether greedy generation from the START
 * sentinel reproduces the whole sequence. Backoff-driven correct predictions
 * are the signal that the model generalized rather than memorized.
 */
export function evaluateMovementGeneralization(
  model: TrainedMovementModel,
  sequences: MovementSequence[],
): MovementGeneralizationReport {
  let predictions = 0;
  let correct = 0;
  let backoffCorrect = 0;
  let exactSequenceMatch = 0;

  for (const sequence of sequences) {
    const tokens = [MOVEMENT_START_TOKEN, ...sequence.tokens];
    for (let index = 1; index < tokens.length; index += 1) {
      const context = tokens.slice(0, index);
      const expected = tokens[index]!;
      const prediction = model.predictNext(context);
      predictions += 1;
      if (prediction && prediction.token === expected) {
        correct += 1;
        if (prediction.backoffOrder < Math.min(model.order, context.length)) {
          backoffCorrect += 1;
        }
      }
    }

    const generated = model.generate([], { stopToken: MOVEMENT_END_TOKEN });
    if (generated.length === sequence.tokens.length && generated.every((token, i) => token === sequence.tokens[i])) {
      exactSequenceMatch += 1;
    }
  }

  return {
    sequenceCount: sequences.length,
    predictions,
    correct,
    accuracy: predictions === 0 ? 0 : correct / predictions,
    backoffCorrect,
    exactSequenceMatch: sequences.length === 0 ? 0 : exactSequenceMatch / sequences.length,
  };
}
