import type { ReplayManifest, ReplayTimelineEvent } from "../capture/replay.js";
import type { TrajectorySpan } from "../capture/trajectory.js";

/**
 * Movement dataset: the structured, replayable training corpus that a local
 * model learns from. A dataset is a set of token *sequences* (one per
 * trajectory), where each token is a discrete movement drawn from a trajectory
 * observation or action. Sequences are what a sequence model (n-gram, RNN, a
 * small on-device transformer, …) trains on to (a) repeat recorded movements
 * and (b) generalize to new-but-related movements.
 *
 * The schema is intentionally backend-agnostic: it carries a canonical modeling
 * `symbol` (what the model predicts over) plus enough payload (`channel`,
 * `summary`) to render a concrete, replayable movement back out of a prediction.
 */

export type MovementTokenType = "observation" | "action";

export type MovementToken = {
  type: MovementTokenType;
  /** Canonical symbol the model predicts over, e.g. `action:browser:click`. */
  symbol: string;
  /** Tool (for actions) or source (for observations). */
  channel: string;
  /** Human-readable movement summary, kept for replay/inspection. */
  summary: string;
};

export type MovementSequence = {
  trajectoryId: string;
  tokens: MovementToken[];
};

export type MovementDataset = {
  version: 1;
  sequences: MovementSequence[];
  /** Sorted distinct symbols across every sequence (the model vocabulary). */
  vocabulary: string[];
  /** One representative token per symbol, so a predicted symbol can be replayed. */
  representatives: Record<string, MovementToken>;
};

/** Sentinel symbols marking sequence boundaries during training/generation. */
export const MOVEMENT_START = "<s>";
export const MOVEMENT_END = "</s>";

/** Normalize a free-text summary into a short verb symbol component. */
export function movementVerb(summary: string): string {
  const first = summary
    .trim()
    .toLowerCase()
    .split(/\s+/, 1)[0]
    ?.replace(/[^a-z0-9]/g, "");
  return first && first.length > 0 ? first : "act";
}

function channelOf(event: ReplayTimelineEvent): string | undefined {
  if (event.kind === "action") {
    return event.tool;
  }
  if (event.kind === "observation") {
    return event.source;
  }
  return undefined;
}

/** Derive a canonical movement token from a replay timeline event. */
export function tokenFromReplayEvent(event: ReplayTimelineEvent): MovementToken | undefined {
  if (event.kind === "transcript") {
    return undefined;
  }
  const channel = channelOf(event) ?? "unknown";
  const type: MovementTokenType = event.kind === "action" ? "action" : "observation";
  return {
    type,
    channel,
    summary: event.summary,
    symbol: `${type}:${channel.trim().toLowerCase()}:${movementVerb(event.summary)}`,
  };
}

function tokensFromTrajectory(trajectory: TrajectorySpan): MovementToken[] {
  const events = [
    ...trajectory.observations.map((observation) => ({
      ts: observation.ts,
      token: {
        type: "observation" as const,
        channel: observation.source,
        summary: observation.summary,
        symbol: `observation:${observation.source.trim().toLowerCase()}:${movementVerb(observation.summary)}`,
      },
    })),
    ...trajectory.actions.map((action) => ({
      ts: action.ts,
      token: {
        type: "action" as const,
        channel: action.tool,
        summary: action.summary,
        symbol: `action:${action.tool.trim().toLowerCase()}:${movementVerb(action.summary)}`,
      },
    })),
  ];
  return events.sort((a, b) => a.ts - b.ts).map((entry) => entry.token);
}

function assemble(sequences: MovementSequence[]): MovementDataset {
  const representatives: Record<string, MovementToken> = {};
  for (const sequence of sequences) {
    for (const token of sequence.tokens) {
      if (!representatives[token.symbol]) {
        representatives[token.symbol] = token;
      }
    }
  }
  return {
    version: 1,
    sequences,
    vocabulary: Object.keys(representatives).sort(),
    representatives,
  };
}

/** Build a dataset directly from captured trajectory spans. */
export function buildMovementDataset(trajectories: TrajectorySpan[]): MovementDataset {
  const sequences = trajectories
    .map((trajectory) => ({ trajectoryId: trajectory.id, tokens: tokensFromTrajectory(trajectory) }))
    .filter((sequence) => sequence.tokens.length > 0);
  return assemble(sequences);
}

/** Build a dataset from reviewed replay manifests (the training-export path). */
export function buildMovementDatasetFromReplays(replays: Pick<ReplayManifest, "trajectoryIds" | "events">[]): MovementDataset {
  const byTrajectory = new Map<string, MovementToken[]>();
  for (const replay of replays) {
    for (const event of replay.events) {
      if (event.kind === "transcript") {
        continue;
      }
      const token = tokenFromReplayEvent(event);
      if (!token) {
        continue;
      }
      const list = byTrajectory.get(event.trajectoryId) ?? [];
      list.push(token);
      byTrajectory.set(event.trajectoryId, list);
    }
  }
  const sequences: MovementSequence[] = [...byTrajectory.entries()]
    .map(([trajectoryId, tokens]) => ({ trajectoryId, tokens }))
    .filter((sequence) => sequence.tokens.length > 0);
  return assemble(sequences);
}
