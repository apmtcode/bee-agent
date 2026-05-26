import { readJsonFile, writeJsonAtomic } from "../shared/fs.js";
import type { RecallMatchReason, TrajectoryRecallHit } from "../memory/types.js";
import type {
  ReviewedTranscriptMessage,
  ReviewedTrajectoryAction,
  ReviewedTrajectoryObservation,
  TrajectoryReviewStatus,
  TrajectorySpan,
} from "./trajectory.js";

export type TrajectoryStoreShape = {
  version: 1;
  trajectories: TrajectorySpan[];
};

export type ReviewTrajectoryParams = {
  status: TrajectoryReviewStatus;
  reviewedBy: string;
  reviewNote?: string;
  redactedObservations?: ReviewedTrajectoryObservation[];
  redactedActions?: ReviewedTrajectoryAction[];
  redactedTranscript?: ReviewedTranscriptMessage[];
};

const EMPTY_STORE: TrajectoryStoreShape = {
  version: 1,
  trajectories: [],
};

function scoreMatch(value: string | undefined, needle: string): number {
  if (!value) {
    return 0;
  }
  const haystack = value.toLowerCase();
  const query = needle.toLowerCase();
  if (haystack === query) {
    return 6;
  }
  if (haystack.startsWith(query)) {
    return 4;
  }
  return haystack.includes(query) ? 2 : 0;
}

function pushReason(reasons: RecallMatchReason[], reason: RecallMatchReason, score: number): number {
  if (score > 0 && !reasons.includes(reason)) {
    reasons.push(reason);
  }
  return score;
}

function getReviewedSummary(trajectory: TrajectorySpan): string {
  const review = trajectory.review;
  if (review?.redactedObservations?.length) {
    return review.redactedObservations.map((item) => item.summary).join(" ");
  }
  if (review?.redactedActions?.length) {
    return review.redactedActions.map((item) => item.summary).join(" ");
  }
  if (review?.redactedTranscript?.length) {
    return review.redactedTranscript.map((item) => item.content).join(" ");
  }
  return trajectory.outcome?.summary ?? trajectory.observations.map((item) => item.summary).join(" ");
}

export class FileTrajectoryStore {
  constructor(private readonly filePath: string) {}

  async load(): Promise<TrajectoryStoreShape> {
    return await readJsonFile(this.filePath, EMPTY_STORE);
  }

  async save(store: TrajectoryStoreShape): Promise<void> {
    await writeJsonAtomic(this.filePath, store);
  }

  async add(trajectory: TrajectorySpan): Promise<TrajectorySpan> {
    const store = await this.load();
    store.trajectories.push(trajectory);
    await this.save(store);
    return trajectory;
  }

  async get(trajectoryId: string): Promise<TrajectorySpan | undefined> {
    const store = await this.load();
    return store.trajectories.find((trajectory) => trajectory.id === trajectoryId);
  }

  async review(trajectoryId: string, params: ReviewTrajectoryParams): Promise<TrajectorySpan | undefined> {
    const store = await this.load();
    const trajectory = store.trajectories.find((entry) => entry.id === trajectoryId);
    if (!trajectory) {
      return undefined;
    }
    trajectory.review = {
      status: params.status,
      reviewedAt: new Date().toISOString(),
      reviewedBy: params.reviewedBy,
      ...(params.reviewNote ? { reviewNote: params.reviewNote } : {}),
      ...(params.redactedObservations ? { redactedObservations: [...params.redactedObservations] } : {}),
      ...(params.redactedActions ? { redactedActions: [...params.redactedActions] } : {}),
      ...(params.redactedTranscript ? { redactedTranscript: [...params.redactedTranscript] } : {}),
    };
    await this.save(store);
    return trajectory;
  }

  async list(): Promise<TrajectorySpan[]> {
    const store = await this.load();
    return [...store.trajectories].sort((a, b) => a.createdAt.localeCompare(b.createdAt));
  }

  async listBySession(sessionId: string): Promise<TrajectorySpan[]> {
    const trajectories = await this.list();
    return trajectories.filter((trajectory) => trajectory.sessionId === sessionId);
  }

  async listReviewed(status: TrajectoryReviewStatus = "approved"): Promise<TrajectorySpan[]> {
    const trajectories = await this.list();
    return trajectories.filter((trajectory) => trajectory.review?.status === status);
  }

  async searchReviewed(query: string): Promise<Array<Omit<TrajectoryRecallHit, "preview">>> {
    const needle = query.trim();
    if (!needle) {
      return [];
    }
    const trajectories = await this.listReviewed("approved");
    return trajectories
      .map((trajectory) => {
        const reasons: RecallMatchReason[] = [];
        let score = 0;
        score += pushReason(reasons, "outcome", scoreMatch(trajectory.outcome?.summary, needle));
        score += pushReason(reasons, "review", scoreMatch(trajectory.review?.reviewNote, needle));
        score += pushReason(reasons, "summary", scoreMatch(getReviewedSummary(trajectory), needle));
        for (const observation of trajectory.review?.redactedObservations ?? trajectory.observations) {
          score += pushReason(reasons, "observation", scoreMatch(observation.summary, needle));
        }
        for (const action of trajectory.review?.redactedActions ?? trajectory.actions) {
          score += pushReason(reasons, "action", scoreMatch(action.summary, needle));
        }
        for (const message of trajectory.review?.redactedTranscript ?? []) {
          score += pushReason(reasons, "transcript", scoreMatch(message.content, needle));
        }
        if (score === 0) {
          return null;
        }
        return {
          kind: "trajectory",
          score,
          reasons,
          trajectoryId: trajectory.id,
          sessionId: trajectory.sessionId,
          createdAt: trajectory.createdAt,
          captureTier: trajectory.captureTier,
          outcomeSummary: trajectory.outcome?.summary,
          outcomeStatus: trajectory.outcome?.status,
          reward: trajectory.outcome?.reward,
          reviewStatus: "approved",
          reviewedAt: trajectory.review?.reviewedAt,
          reviewedBy: trajectory.review?.reviewedBy,
          summary: getReviewedSummary(trajectory),
          sourceObservationCount: (trajectory.review?.redactedObservations ?? trajectory.observations).length,
          sourceActionCount: (trajectory.review?.redactedActions ?? trajectory.actions).length,
        } satisfies Omit<TrajectoryRecallHit, "preview">;
      })
      .filter((hit): hit is Omit<TrajectoryRecallHit, "preview"> => hit !== null)
      .sort((a, b) => b.score - a.score || b.createdAt.localeCompare(a.createdAt));
  }
}
