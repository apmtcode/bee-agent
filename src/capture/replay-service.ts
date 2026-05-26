import { JsonlTranscriptStore } from "../harness/transcript-store.js";
import type { FileSessionStore } from "../harness/session-store.js";
import type { FileTrajectoryStore } from "./trajectory-store.js";
import { buildReplayManifest, type ReplayManifest } from "./replay.js";

export class ReplayRuntimeService {
  constructor(
    private readonly sessions: FileSessionStore,
    private readonly trajectories: FileTrajectoryStore,
  ) {}

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
    return replays.filter((replay): replay is ReplayManifest => Boolean(replay) && replay.eventCount > 0);
  }
}
