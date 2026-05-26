import type { TranscriptRecord } from "../harness/transcript-store.js";
import type { TrajectorySpan } from "./trajectory.js";

export type ReplayTimelineEvent =
  | {
      kind: "transcript";
      ts: number;
      messageId: string;
      role: "system" | "user" | "assistant" | "tool";
      content: string;
    }
  | {
      kind: "observation";
      ts: number;
      trajectoryId: string;
      source: string;
      summary: string;
    }
  | {
      kind: "action";
      ts: number;
      trajectoryId: string;
      tool: string;
      summary: string;
    };

export type ReplayManifest = {
  version: 1;
  sessionId: string;
  trajectoryIds: string[];
  eventCount: number;
  events: ReplayTimelineEvent[];
};

export function buildReplayManifest(params: {
  sessionId: string;
  transcript: TranscriptRecord[];
  trajectories: TrajectorySpan[];
}): ReplayManifest {
  const transcriptEvents: ReplayTimelineEvent[] = params.transcript.flatMap((record) => {
    if (!("message" in record)) {
      return [];
    }
    return [
      {
        kind: "transcript",
        ts: record.message.timestamp,
        messageId: record.id,
        role: record.message.role,
        content: record.message.content,
      },
    ];
  });

  const trajectoryEvents: ReplayTimelineEvent[] = params.trajectories.flatMap((trajectory) => [
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
  ]);

  const events = [...transcriptEvents, ...trajectoryEvents].sort((a, b) => {
    if (a.ts !== b.ts) {
      return a.ts - b.ts;
    }
    return kindOrder(a.kind) - kindOrder(b.kind);
  });

  return {
    version: 1,
    sessionId: params.sessionId,
    trajectoryIds: params.trajectories.map((trajectory) => trajectory.id),
    eventCount: events.length,
    events,
  };
}

function kindOrder(kind: ReplayTimelineEvent["kind"]): number {
  switch (kind) {
    case "transcript":
      return 0;
    case "observation":
      return 1;
    case "action":
      return 2;
  }
}
