import { readJsonFile, writeJsonAtomic } from "../shared/fs.js";
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
}
