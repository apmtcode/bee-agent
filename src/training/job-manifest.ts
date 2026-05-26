import type { ReviewedExportManifest, TrainingMode, TrainingTargetPlatform } from "./export-manifest.js";

export type LocalTrainingJobStatus = "planned" | "ready" | "running" | "completed" | "failed";

export type LocalTrainingExecutionStepStatus = "pending" | "running" | "completed" | "failed";

export type LocalTrainingExecutionStep = {
  id: string;
  title: string;
  status: LocalTrainingExecutionStepStatus;
};

export type LocalTrainingExecution = {
  version: 1;
  preparedAt: string;
  startedAt?: string;
  completedAt?: string;
  failedAt?: string;
  lastUpdatedAt: string;
  workingDirectory: string;
  datasetDir: string;
  artifactDir: string;
  logFile: string;
  stateFile: string;
  planFile: string;
  launchScript: string;
  replayEvalFile: string;
  command?: string[];
  environment?: Record<string, string>;
  processId?: number;
  outputModelRef?: string;
  failureReason?: string;
  steps: LocalTrainingExecutionStep[];
};

export type SftTrainingConfig = {
  epochs: number;
  learningRate: number;
  batchSize: number;
};

export type RlTrainingConfig = {
  algorithm: "grpo";
  rewardSource: "replay-manifest";
  rolloutCount: number;
  klPenalty: number;
};

export type LocalTrainingJobManifest = {
  version: 1;
  id: string;
  createdAt: string;
  reviewedExportCreatedAt: string;
  reviewedBy: string;
  purpose: string;
  mode: TrainingMode;
  targetPlatform: TrainingTargetPlatform;
  status: LocalTrainingJobStatus;
  dataset: {
    promotedSkillCount: number;
    memoryCount: number;
    trajectoryCount: number;
    replayCount: number;
  };
  runtime: {
    environment: "local-apple-silicon";
    reviewedExportRequired: true;
    rawCaptureAllowed: false;
  };
  execution?: LocalTrainingExecution;
  config: SftTrainingConfig | RlTrainingConfig;
};

export function createLocalTrainingExecution(params: {
  jobId: string;
  mode: TrainingMode;
  preparedAt?: string;
}): LocalTrainingExecution {
  const preparedAt = params.preparedAt ?? new Date().toISOString();
  const baseDir = `training-jobs/${params.jobId}`;
  return {
    version: 1,
    preparedAt,
    lastUpdatedAt: preparedAt,
    workingDirectory: baseDir,
    datasetDir: `${baseDir}/dataset`,
    artifactDir: `${baseDir}/artifacts`,
    logFile: `${baseDir}/logs/${params.mode}.log`,
    stateFile: `${baseDir}/state.json`,
    planFile: `${baseDir}/plan.json`,
    launchScript: `${baseDir}/run-${params.mode}.sh`,
    replayEvalFile: `${baseDir}/replay-eval.json`,
    steps: [
      { id: "prepare-dataset", title: "Prepare reviewed dataset", status: "pending" },
      { id: "launch-training", title: "Launch local Apple Silicon training", status: "pending" },
      { id: "collect-artifacts", title: "Collect model artifacts", status: "pending" },
      { id: "evaluate-replay", title: "Run replay-backed evaluation", status: "pending" },
    ],
  };
}

export function createLocalTrainingJobManifest(params: {
  id: string;
  exportManifest: ReviewedExportManifest;
  mode: TrainingMode;
}): LocalTrainingJobManifest {
  return {
    version: 1,
    id: params.id,
    createdAt: new Date().toISOString(),
    reviewedExportCreatedAt: params.exportManifest.createdAt,
    reviewedBy: params.exportManifest.reviewedBy,
    purpose: params.exportManifest.purpose,
    mode: params.mode,
    targetPlatform: params.exportManifest.targetPlatform,
    status: "planned",
    dataset: {
      promotedSkillCount: params.exportManifest.promotedSkills.length,
      memoryCount: params.exportManifest.memories.length,
      trajectoryCount: params.exportManifest.trajectories.length,
      replayCount: params.exportManifest.replays.length,
    },
    runtime: {
      environment: "local-apple-silicon",
      reviewedExportRequired: true,
      rawCaptureAllowed: false,
    },
    config: params.mode === "sft"
      ? {
          epochs: 3,
          learningRate: 1e-5,
          batchSize: 1,
        }
      : {
          algorithm: "grpo",
          rewardSource: "replay-manifest",
          rolloutCount: 8,
          klPenalty: 0.02,
        },
  };
}
