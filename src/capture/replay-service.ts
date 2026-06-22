import { JsonlTranscriptStore } from "../harness/transcript-store.js";
import type { RecallReplayPreviewEvent } from "../memory/types.js";
import type { FileSessionStore } from "../harness/session-store.js";
import type { FileTrajectoryStore } from "./trajectory-store.js";
import type { TrajectorySpan } from "./trajectory.js";
import { buildReplayManifest, type ReplayManifest } from "./replay.js";

export class ReplayRuntimeService {
  constructor(
    private readonly sessions: FileSessionStore,
    private readonly trajectories: FileTrajectoryStore,
  ) {}

  buildTrajectoryPreview(trajectory: TrajectorySpan, limit = 5): RecallReplayPreviewEvent[] {
    const transcriptPreview = (trajectory.review?.redactedTranscript ?? []).map((message) => ({
      kind: "transcript" as const,
      ts: message.ts,
      summary: message.content,
    }));
    const observationPreview = (trajectory.review?.redactedObservations ?? trajectory.observations).map((observation) => ({
      kind: "observation" as const,
      ts: observation.ts,
      summary: observation.summary,
    }));
    const actionPreview = (trajectory.review?.redactedActions ?? trajectory.actions).map((action) => ({
      kind: "action" as const,
      ts: action.ts,
      summary: action.summary,
    }));

    return [...observationPreview, ...actionPreview, ...transcriptPreview]
      .sort((a, b) => a.ts - b.ts)
      .slice(0, limit);
  }

  async getSessionReplay(sessionId: string): Promise<ReplayManifest | undefined> {
    const session = await this.sessions.get(sessionId);
    if (!session?.transcriptFile) {
      return undefined;
    }

    const transcript = await new JsonlTranscriptStore(session.transcriptFile).read({ includeHeader: true });
    const trajectorySpans = await this.trajectories.listBySession(sessionId);
    if (transcript.length === 0 && trajectorySpans.length === 0) {
      return undefined;
    }

    return buildReplayManifest({
      sessionId,
      transcript,
      trajectories: trajectorySpans,
    });
  }

  async listSessionReplays(): Promise<ReplayManifest[]> {
    const sessions = await this.sessions.list();
    const replays = await Promise.all(sessions.map(async (session) => await this.getSessionReplay(session.id)));
    return replays.filter((replay): replay is ReplayManifest => replay !== undefined && replay.eventCount > 0);
  }
}
