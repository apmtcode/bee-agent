/**
 * Movement dataset builder.
 *
 * Bridges the *capture* half of the subsystem (replay timelines / trajectory
 * spans) to the *learning* half ({@link MovementModelBackend}) by tokenizing
 * events into the compact discrete {@link MovementSequence} format the model
 * trains on. Tokenization is intentionally coarse (verb + a couple of summary
 * words) so related-but-distinct movements share tokens and the model can
 * generalize instead of memorizing every unique free-text summary.
 */
import type { ReplayManifest, ReplayTimelineEvent } from "../capture/replay.js";
import type { TrajectorySpan } from "../capture/trajectory.js";
import type { ExportedReplayManifest, ReviewedExportManifest } from "./export-manifest.js";
import {
  NGramMovementModelBackend,
  type MovementModelBackend,
  type MovementSequence,
  type MovementToken,
  type TrainedMovementModel,
} from "./movement-model.js";

export type MovementTokenizerOptions = {
  /** Include `observation` events as `obs:*` tokens (default true). */
  includeObservations?: boolean;
  /** Include `transcript` events as `msg:<role>` tokens (default false). */
  includeTranscript?: boolean;
  /** How many leading summary words to fold into a token (default 2). */
  summaryWords?: number;
};

const DEFAULT_SUMMARY_WORDS = 2;

/** Compact a free-text summary into a stable, low-cardinality slug. */
function summarySlug(summary: string, words: number): string {
  const slug = summary
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, " ")
    .split(/\s+/)
    .filter((word) => word.length > 0)
    .slice(0, Math.max(1, words))
    .join("-");
  return slug.length > 0 ? slug : "unknown";
}

/**
 * Tokenize a single replay timeline event. Returns undefined for events the
 * options exclude (so callers can `flatMap` over a timeline).
 */
export function tokenizeReplayEvent(
  event: ReplayTimelineEvent,
  options: MovementTokenizerOptions = {},
): MovementToken | undefined {
  const words = options.summaryWords ?? DEFAULT_SUMMARY_WORDS;
  switch (event.kind) {
    case "action":
      return `act:${event.tool}:${summarySlug(event.summary, words)}`;
    case "observation":
      if (options.includeObservations === false) {
        return undefined;
      }
      return `obs:${event.source}:${summarySlug(event.summary, words)}`;
    case "transcript":
      if (!options.includeTranscript) {
        return undefined;
      }
      return `msg:${event.role}`;
  }
}

type ReplayLike = Pick<ReplayManifest, "sessionId" | "trajectoryIds"> & {
  events: ReplayTimelineEvent[];
};

function replayToSequence(replay: ReplayLike, options: MovementTokenizerOptions): MovementSequence {
  const tokens = replay.events
    .map((event) => tokenizeReplayEvent(event, options))
    .filter((token): token is MovementToken => token !== undefined);
  const id = replay.trajectoryIds[0] ?? replay.sessionId;
  return { id, tokens };
}

/** Build one {@link MovementSequence} from a replay manifest. */
export function tokenizeReplayManifest(
  manifest: ReplayManifest,
  options: MovementTokenizerOptions = {},
): MovementSequence {
  return replayToSequence(manifest, options);
}

/** Build a dataset (one sequence per manifest) from replay manifests. */
export function buildMovementDatasetFromReplays(
  manifests: ReplayManifest[],
  options: MovementTokenizerOptions = {},
): MovementSequence[] {
  return manifests.map((manifest) => replayToSequence(manifest, options)).filter((sequence) => sequence.tokens.length > 0);
}

/** Build a dataset directly from trajectory spans (observations + actions, time-ordered). */
export function buildMovementDatasetFromTrajectories(
  trajectories: TrajectorySpan[],
  options: MovementTokenizerOptions = {},
): MovementSequence[] {
  return trajectories
    .map((trajectory) => trajectoryToSequence(trajectory, options))
    .filter((sequence) => sequence.tokens.length > 0);
}

function trajectoryToSequence(trajectory: TrajectorySpan, options: MovementTokenizerOptions): MovementSequence {
  const events: ReplayTimelineEvent[] = [
    ...trajectory.observations.map<ReplayTimelineEvent>((observation) => ({
      kind: "observation",
      ts: observation.ts,
      trajectoryId: trajectory.id,
      source: observation.source,
      summary: observation.summary,
    })),
    ...trajectory.actions.map<ReplayTimelineEvent>((action) => ({
      kind: "action",
      ts: action.ts,
      trajectoryId: trajectory.id,
      tool: action.tool,
      summary: action.summary,
    })),
  ].sort((a, b) => a.ts - b.ts);
  const tokens = events
    .map((event) => tokenizeReplayEvent(event, options))
    .filter((token): token is MovementToken => token !== undefined);
  return { id: trajectory.id, tokens };
}

/** Build a dataset from the reviewed-export manifest's embedded replays. */
export function buildMovementDatasetFromExport(
  manifest: Pick<ReviewedExportManifest, "replays">,
  options: MovementTokenizerOptions = {},
): MovementSequence[] {
  return manifest.replays
    .map((replay: ExportedReplayManifest) => replayToSequence(replay, options))
    .filter((sequence) => sequence.tokens.length > 0);
}

export type TrainMovementModelFromExportOptions = MovementTokenizerOptions & {
  backend?: MovementModelBackend;
  order?: number;
};

/**
 * Convenience end-to-end: reviewed export manifest -> tokenized dataset ->
 * trained movement model. Defaults to the deterministic n-gram backend so it
 * runs anywhere; pass a real on-device backend to swap the seam.
 */
export async function trainMovementModelFromExport(
  manifest: Pick<ReviewedExportManifest, "replays">,
  options: TrainMovementModelFromExportOptions = {},
): Promise<{ model: TrainedMovementModel; dataset: MovementSequence[] }> {
  const dataset = buildMovementDatasetFromExport(manifest, options);
  const backend = options.backend ?? new NGramMovementModelBackend();
  const model = await backend.train({ sequences: dataset, order: options.order });
  return { model, dataset };
}
