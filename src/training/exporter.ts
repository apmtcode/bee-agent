import { writeJsonAtomic } from "../shared/fs.js";
import { buildReplayManifest } from "../capture/replay.js";
import { FileTrajectoryStore } from "../capture/trajectory-store.js";
import { FileMemoryStore } from "../memory/memory-store.js";
import { JsonlTranscriptStore, type TranscriptRecord } from "../harness/transcript-store.js";
import type { SessionRecord } from "../harness/types.js";
import type { TrajectorySpan } from "../capture/trajectory.js";
import type { ReviewedExportManifest, TrainingMode } from "./export-manifest.js";

export type LocalTrainingExporterOptions = {
  memory: FileMemoryStore;
  trajectories: FileTrajectoryStore;
  resolveSession?: (sessionId: string) => Promise<SessionRecord | undefined>;
};

export type CreateReviewedExportParams = {
  reviewedBy: string;
  purpose: string;
  outputFile: string;
  modes?: TrainingMode[];
};

export class LocalTrainingExporter {
  constructor(private readonly options: LocalTrainingExporterOptions) {}

  async createReviewedExport(params: CreateReviewedExportParams): Promise<ReviewedExportManifest> {
    const skills = await this.options.memory.listPromotedSkills();
    const memories = await this.options.memory.load();
    const trajectories = await this.options.trajectories.list();
    const approvedTrajectoryIds = new Set(
      trajectories.filter((trajectory) => trajectory.review?.status === "approved").map((trajectory) => trajectory.id),
    );

    const exportedSkills = skills.filter((skill) =>
      skill.sourceTrajectoryIds.length > 0 && skill.sourceTrajectoryIds.every((trajectoryId) => approvedTrajectoryIds.has(trajectoryId)),
    );
    const exportedExecutableSkills = memories.executableSkills.filter((skill) =>
      skill.sourceTrajectoryIds.length > 0 && skill.sourceTrajectoryIds.every((trajectoryId) => approvedTrajectoryIds.has(trajectoryId)),
    );

    const selectedTrajectoryIds = new Set([
      ...exportedSkills.flatMap((skill) => skill.sourceTrajectoryIds),
      ...exportedExecutableSkills.flatMap((skill) => skill.sourceTrajectoryIds),
    ]);
    const reviewedTrajectories = trajectories.filter((trajectory) => selectedTrajectoryIds.has(trajectory.id));
    const reviewedTrajectoryIds = new Set(reviewedTrajectories.map((trajectory) => trajectory.id));

    const exportedTrajectories = reviewedTrajectories.map((trajectory) => ({
      id: trajectory.id,
      sessionId: trajectory.sessionId,
      createdAt: trajectory.createdAt,
      captureTier: trajectory.captureTier,
      observationCount: getReviewedObservationCount(trajectory),
      actionCount: getReviewedActionCount(trajectory),
      outcomeStatus: trajectory.outcome?.status,
      reward: trajectory.outcome?.reward,
      reviewedAt: trajectory.review?.reviewedAt,
      reviewedBy: trajectory.review?.reviewedBy,
    }));

    const replays = this.options.resolveSession
      ? await Promise.all(
          reviewedTrajectories.map(async (trajectory) => {
            const session = await this.options.resolveSession?.(trajectory.sessionId);
            if (!session?.transcriptFile) {
              return undefined;
            }
            const transcript = await new JsonlTranscriptStore(session.transcriptFile).read({ includeHeader: true });
            const replay = buildReplayManifest({
              sessionId: trajectory.sessionId,
              transcript: getReviewedTranscript(transcript, trajectory),
              trajectories: [getReviewedTrajectory(trajectory)],
            });
            return {
              sessionId: replay.sessionId,
              trajectoryIds: replay.trajectoryIds,
              eventCount: replay.eventCount,
              events: replay.events,
            };
          }),
        )
      : [];

    const manifest: ReviewedExportManifest = {
      version: 1,
      createdAt: new Date().toISOString(),
      reviewedBy: params.reviewedBy,
      purpose: params.purpose,
      targetPlatform: "apple-silicon",
      modes: [...(params.modes ?? ["sft"])],
      rawCaptureIncluded: false,
      promotedSkills: exportedSkills.map((skill) => ({
        id: skill.id,
        title: skill.title,
        summary: skill.summary,
        sourceCandidateId: skill.sourceCandidateId,
        sourceTrajectoryIds: [...skill.sourceTrajectoryIds],
        promotedAt: skill.promotedAt,
        version: skill.version,
        usage: skill.usage,
      })),
      executableSkills: exportedExecutableSkills.map((skill) => ({
        id: skill.id,
        promotedSkillId: skill.promotedSkillId,
        title: skill.title,
        summary: skill.summary,
        version: skill.version,
        sourceTrajectoryIds: [...skill.sourceTrajectoryIds],
        usage: { ...skill.usage },
      })),
      executableSkillRuns: memories.executableSkillRuns
        .filter((run) => run.sourceTrajectoryIds.every((trajectoryId) => approvedTrajectoryIds.has(trajectoryId)))
        .map((run) => ({
          id: run.id,
          skillId: run.skillId,
          sessionId: run.sessionId,
          parentRunId: run.parentRunId,
          status: run.status,
          sourceTrajectoryIds: [...run.sourceTrajectoryIds],
          replayPreview: run.replayPreview.map((event) => ({ ...event })),
          stepResults: run.stepResults.map((result) => ({
            stepId: result.stepId,
            title: result.title,
            kind: result.kind,
            output: result.output,
            agentRole: result.agentRole,
            subagentRunId: result.subagentRunId,
            commentOutputs: result.commentOutputs ? [...result.commentOutputs] : undefined,
          })),
        })),
      memories: memories.items
        .filter((item) => !item.sourceTrajectoryId || reviewedTrajectoryIds.has(item.sourceTrajectoryId))
        .map((item) => ({
          id: item.id,
          type: item.type,
          summary: item.summary,
          sourceSessionId: item.sourceSessionId,
          sourceTrajectoryId: item.sourceTrajectoryId,
          tags: [...(item.tags ?? [])],
        })),
      trajectories: exportedTrajectories,
      replays: replays.filter((item): item is NonNullable<typeof item> => Boolean(item)),
    };

    await writeJsonAtomic(params.outputFile, manifest);
    return manifest;
  }
}

function getReviewedObservationCount(trajectory: TrajectorySpan): number {
  return trajectory.review?.redactedObservations?.length ?? trajectory.observations.length;
}

function getReviewedActionCount(trajectory: TrajectorySpan): number {
  return trajectory.review?.redactedActions?.length ?? trajectory.actions.length;
}

function getReviewedTrajectory(trajectory: TrajectorySpan): TrajectorySpan {
  return {
    ...trajectory,
    observations: trajectory.review?.redactedObservations
      ? trajectory.review.redactedObservations.map((observation) => ({
          kind: "observation",
          ts: observation.ts,
          source: observation.source,
          summary: observation.summary,
        }))
      : trajectory.observations,
    actions: trajectory.review?.redactedActions
      ? trajectory.review.redactedActions.map((action) => ({
          kind: "action",
          ts: action.ts,
          tool: action.tool,
          summary: action.summary,
        }))
      : trajectory.actions,
  };
}

function getReviewedTranscript(transcript: TranscriptRecord[], trajectory: TrajectorySpan): TranscriptRecord[] {
  if (!trajectory.review?.redactedTranscript) {
    return transcript;
  }
  const header = transcript.find((record) => !("message" in record));
  const redactedMessages = trajectory.review.redactedTranscript.map((message, index) => ({
    id: `reviewed-${trajectory.id}-${index}`,
    message: {
      role: message.role,
      content: message.content,
      timestamp: message.ts,
    },
  }));
  return header ? [header, ...redactedMessages] : redactedMessages;
}
