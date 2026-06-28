export {
  OperatorCliApp,
  parseSlashCommand,
  type OperatorCliAppOptions,
  type OperatorCliSlashCommand,
} from "./cli/app.js";
export {
  FileOperatorCliTeamStore,
  type OperatorCliTeamRecord,
  type OperatorCliTeammateRecord,
  type OperatorCliTeammateStatus,
} from "./cli/team-store.js";
export {
  OperatorCliConfigLoader,
  deepMergeConfig,
  resolveOperatorCliExecutionConfig,
  resolveOperatorCliStatusLineConfig,
  setProjectPermissionModeConfig,
  setProjectStatusLineConfig,
  type OperatorCliConfigEntry,
  type OperatorCliConfigObject,
  type OperatorCliConfigSource,
  type OperatorCliConfigValue,
  type OperatorCliExecutionConfig,
  type OperatorCliStatusLineConfig,
  type OperatorCliHookEvent,
  type OperatorCliPermissionMode,
  type OperatorCliRuntimeConfig,
} from "./cli/config.js";
export {
  buildOperatorExecutionFingerprint,
  evaluateOperatorExecutionAction,
  runOperatorCommandHooks,
  runOperatorHooks,
  runOperatorPreToolUseHooks,
  type OperatorExecutionAction,
  type OperatorExecutionActionKind,
  type OperatorExecutionPolicyDecision,
  type OperatorHookContext,
  type OperatorHookResult,
  type OperatorPreToolUseDecision,
  type OperatorPreToolUseHookParsedOutput,
  type OperatorPreToolUseHookResult,
  type OperatorPreToolUsePermissionDecision,
} from "./cli/execution-policy.js";
export {
  discoverPromptContext,
  type OperatorCliContextFile,
  type OperatorCliProjectContext,
  type OperatorCliPromptContext,
} from "./cli/prompt.js";
export {
  OperatorEventBus,
  type OperatorEvent,
  type OperatorEventFilter,
  type OperatorEventIteratorOptions,
} from "./kernel/event-bus.js";
export {
  FileBackgroundTaskStore,
  BackgroundTaskExecutionService,
  type BackgroundTaskExecution,
  type BackgroundTaskExecutionState,
  type BackgroundTaskKind,
  type BackgroundTaskRecord,
  type BackgroundTaskRecoveryReason,
  type BackgroundTaskRecoveryResult,
  type BackgroundTaskStatus,
  type BackgroundTaskStoreShape,
  type CompleteBackgroundTaskParams,
  type FailBackgroundTaskParams,
  type MarkBackgroundTaskStartedParams,
  type ReadBackgroundTaskOutputOptions,
  type SpawnBackgroundProcess,
  type StartBackgroundTaskParams,
  type StopBackgroundTaskParams,
} from "./harness/background-tasks.js";
export {
  FileSessionStore,
  type SessionStoreRecord,
  type SessionStoreShape,
} from "./harness/session-store.js";
export {
  JsonlTranscriptStore,
  type TranscriptAppendMessageParams,
  type TranscriptHeader,
  type TranscriptMessageRecord,
  type TranscriptRecord,
} from "./harness/transcript-store.js";
export type {
  OperatorModelSelectionSource,
  OperatorResolvedModelSelection,
  SessionMetadata,
  SessionRecord,
  SessionStatus,
  TranscriptMessageRole,
  WebhookChatReplyTarget,
  WebhookChatSessionMetadata,
} from "./harness/types.js";
export {
  FileApprovalStore,
  type ApprovalStoreShape,
} from "./control-plane/approval-store.js";
export {
  DEFAULT_APPROVAL_TIMEOUT_MS,
  InMemoryApprovalManager,
  type ApprovalDecision,
  type ApprovalRequest,
  type ApprovalRequestInput,
  type ApprovalResolution,
} from "./control-plane/approvals.js";
export {
  OperatorCronService,
} from "./control-plane/cron-service.js";
export {
  resolveAnthropicAuthConfig,
  buildAnthropicMessagesRequest,
  normalizeAnthropicMessagesPayload,
  forwardAnthropicMessagesRequest,
  forwardAnthropicMessagesStreamRequest,
  type AnthropicAuthConfig,
  type AnthropicHttpRequest,
  type AnthropicMessageForwardResponse,
  type AnthropicMessageStreamForwardResponse,
  type EnvReader,
} from "./control-plane/anthropic-client.js";
export {
  OperatorDeliveryService,
  summarizeDeliveryResults,
  type BrowserPushDeliveryTarget,
  type CronDeliveryConfig,
  type CronTerminalDeliveryPayload,
  type PushNotificationDeliveryPayload,
  type DeliveryPayload,
  type DeliveryAttemptResult,
  type DeliveryAttemptStatus,
  type DeliverySummaryStatus,
  type DeliveryTarget,
} from "./control-plane/delivery.js";
export {
  FileCronStore,
  type CronJob,
  type CronJobModelSelection,
  type CronRun,
  type CronStoreShape,
} from "./control-plane/cron-store.js";
export {
  OperatorControlPlaneServer,
  type ControlPlaneFailure,
  type ControlPlaneHttpRequest,
  type ControlPlaneHttpResponse,
  type ControlPlaneRequest,
  type ControlPlaneResponse,
  type ControlPlaneServerOptions,
  type ControlPlaneSuccess,
  type GatewaySessionHealth,
  type PairingTicketCreateResult,
  type PairingTicketResolveResult,
  type SessionBootstrapResult,
  type SessionTaskPlan,
  type SessionTaskPlanEntry,
  type SessionRemoteControlState,
  type SessionRemoteInventoryItem,
  type SessionRemoteInventoryResult,
  type SessionRemoteQuarantine,
  type SessionRemoteRepairResult,
  type SessionRemoteStatusResult,
  type SessionPlatformControlResult,
  type SessionPlatformControlState,
  type SessionPlatformInventoryItem,
  type SessionPlatformInventoryResult,
  type SessionPlatformStatusResult,
  type SpawnSubagentResult,
} from "./control-plane/server.js";
export {
  FilePairingStore,
  PairingTicketError,
  type PairingStoreShape,
  type PairingTicket,
  type PairingTicketStatus,
} from "./control-plane/pairing-store.js";
export {
  FilePlatformBreakerStore,
  type PlatformBreakerRecord,
  type PlatformBreakerStoreShape,
} from "./control-plane/platform-breaker-store.js";
export {
  FilePushStore,
  type PushStoreShape,
  type PushSubscriptionRecord,
  type WebPushSubscriptionKeys,
  type WebPushSubscriptionPayload,
} from "./control-plane/push-store.js";
export {
  OperatorControlPlaneSessionStream,
  type SessionStreamBootstrapParams,
} from "./control-plane/session-stream.js";
export {
  normalizeWebhookChatPayload,
  buildWebhookChatRemoteId,
  buildWebhookChatSessionMetadata,
  hasWebhookChatDeliveryId,
  markWebhookChatReplyDelivered,
  buildWebhookChatReplyRequest,
  type WebhookChatInboundMessage,
  type WebhookChatPayload,
  type NormalizedWebhookChatPayload,
  type WebhookChatReplyRequest,
} from "./control-plane/webhook-chat.js";
export {
  OperatorGatewayTransportConnection,
  type GatewayClientMessage,
  type GatewayEventSubscription,
  type GatewayServerMessage,
  type GatewayTransport,
  type OperatorGatewayTransportConnectionOptions,
} from "./control-plane/gateway-transport.js";
export {
  buildRuntimeEventFilter,
  subscribeRuntimeEvents,
  type RuntimeEventFamily,
  type RuntimeEventSubscriptionOptions,
} from "./control-plane/subscriptions.js";
export {
  FileTrajectoryStore,
  type ReviewTrajectoryParams,
  type TrajectoryStoreShape,
} from "./capture/trajectory-store.js";
export {
  FileCaptureConsentStore,
  type CaptureConsentStoreShape,
} from "./capture/consent-store.js";
export {
  CaptureIngestionService,
  type CaptureIngestionServiceOptions,
  type CreateCaptureConsentParams,
} from "./capture/ingestion.js";
export {
  BrowserCaptureAdapter,
  type BrowserActionKind,
  type BrowserCaptureInput,
} from "./capture/browser-adapter.js";
export {
  DeviceCaptureAdapter,
  type DeviceCaptureInput,
  type DeviceGestureKind,
  type DevicePlatform,
} from "./capture/device-adapter.js";
export {
  OsCaptureObserver,
  type OsEventKind,
  type OsObservationInput,
} from "./capture/os-observer.js";
export {
  ConsentAwareCaptureRecorder,
  type CaptureRecorderInput,
  type CaptureRecorderOptions,
  type CaptureRecordResult,
} from "./capture/recorder.js";
export {
  buildReplayManifest,
  type ReplayManifest,
  type ReplayTimelineEvent,
} from "./capture/replay.js";
export {
  ReplayRuntimeService,
} from "./capture/replay-service.js";
export {
  CAPTURE_TIER_VALUES,
  DEFAULT_SENSITIVE_APP_DENYLIST,
  buildTrajectorySpan,
  defaultCapturePolicy,
  isConsentGrantActive,
  isSensitiveApp,
  normalizeCaptureTier,
  type CaptureConsentGrant,
  type CapturePolicy,
  type CaptureTier,
  type ReviewedTranscriptMessage,
  type ReviewedTrajectoryAction,
  type ReviewedTrajectoryObservation,
  type TrajectoryAction,
  type TrajectoryObservation,
  type TrajectoryReview,
  type TrajectoryReviewStatus,
  type TrajectorySpan,
} from "./capture/trajectory.js";
export {
  FileMemoryStore,
  type MemoryStoreShape,
} from "./memory/memory-store.js";
export type {
  ExecutableSkill,
  ExecutableSkillRun,
  ExecutableSkillStep,
  ExecutableSkillStepKind,
  ExecutableSkillStepResult,
  MemoryItem,
  MemoryItemType,
  MemoryRecallHit,
  PromotedSkill,
  RecallHit,
  RecallHitKind,
  RecallMatchReason,
  RecallReplayPreviewEvent,
  RecallSummary,
  SkillCandidate,
  SkillCandidateStatus,
  SkillRecallHit,
  SkillUsageStats,
  TrajectoryRecallHit,
} from "./memory/types.js";
export {
  buildExecutionGraph,
  type ExecutionGraphNode,
} from "./orchestrator/execution-graph.js";
export type {
  OperatorExecutableSkillStepHandler,
  OperatorPluginCapability,
  OperatorPluginManifest,
  OperatorPluginRuntime,
} from "./plugins/manifest.js";
export {
  FilePluginManifestRegistry,
  type PluginRegistryShape,
  type RegisterPluginParams,
  type RegisteredPluginRecord,
} from "./plugins/manifest-registry.js";
export {
  OperatorPluginRuntimeManager,
} from "./plugins/runtime.js";
export {
  FileRunStore,
  type OperatorRunRecord,
  type OperatorRunStatus,
  type RunStoreShape,
} from "./orchestrator/run-store.js";
export {
  FileMessageStore,
  type MessageStoreShape,
  type OperatorMessageRecord,
} from "./orchestrator/message-store.js";
export {
  FilePlanStore,
  type OperatorPlanApprovalRecord,
  type OperatorPlanRecord,
  type OperatorPlanStatus,
  type OperatorPlanVerificationRecord,
  type OperatorPlanVerificationStatus,
  type PlanStoreShape,
} from "./orchestrator/plan-store.js";
export {
  FileSubagentRegistry,
  type SubagentRunRecord,
  type SubagentRegistryShape,
} from "./orchestrator/subagent-registry.js";
export {
  StandaloneOperatorRuntime,
  type ApprovalPromptParams,
  type CreateCaptureConsentParams as RuntimeCreateCaptureConsentParams,
  type CreateTrainingJobParams as RuntimeCreateTrainingJobParams,
  type ListPluginsParams,
  type OperatorNotificationRecord,
  type OperatorRuntimeEventType,
  type RecoverBackgroundTasksParams,
  type RecordTurnParams,
  type RequestPlanApprovalParams,
  type RespondPlanApprovalParams,
  type ReviewTrajectoryParams as RuntimeReviewTrajectoryParams,
  type SendMessageParams,
  type SendNotificationParams,
  type UpdatePlanVerificationParams,
  type UpsertPlanParams,
  type SendTeammateMessageParams,
  type SendTeammateMessageResult,
  type SpawnSubagentParams,
  type SpawnSubagentResult as RuntimeSpawnSubagentResult,
  type StartSessionParams,
  type StartTeammateParams,
  type StartTeammateResult,
  type SessionLifecycleEventType,
  type StandaloneOperatorOptions,
  type TeammateStatus,
} from "./orchestrator/operator-runtime.js";
export {
  LocalTrainingExporter,
  type CreateReviewedExportParams,
  type LocalTrainingExporterOptions,
} from "./training/exporter.js";
export {
  LocalTrainingExecutionService,
} from "./training/execution-service.js";
export type {
  ReadTrainingJobLogOptions,
  TrainingExecutionState,
} from "./training/execution-service.js";
export {
  LocalAppleSiliconTrainingRunner,
} from "./training/runner.js";
export type {
  LocalTrainingRuntime,
  TrainingJobPlan,
} from "./training/runner.js";
export {
  DEFAULT_MOVEMENT_TRAINING_CONFIG,
  MOVEMENT_END_TOKEN,
  MarkovMovementBackend,
  buildMovementDataset,
  buildMovementDatasetFromReplays,
  evaluateMovementModel,
  tokenizeAction,
  trainMovementModel,
} from "./training/movement-model.js";
export type {
  MovementDataset,
  MovementEvaluation,
  MovementModel,
  MovementModelArtifact,
  MovementModelBackend,
  MovementPrediction,
  MovementPredictionSource,
  MovementStep,
  MovementToken,
  MovementTrainingConfig,
} from "./training/movement-model.js";
export type {
  ExportedMemoryReference,
  ExportedPromotedSkill,
  ExportedReplayManifest,
  ExportedTrajectoryReference,
  ReviewedExportManifest,
  TrainingMode,
  TrainingTargetPlatform,
} from "./training/export-manifest.js";
export {
  createLocalTrainingExecution,
  createLocalTrainingJobManifest,
} from "./training/job-manifest.js";
export type {
  LocalTrainingExecution,
  LocalTrainingExecutionStep,
  LocalTrainingExecutionStepStatus,
  LocalTrainingJobManifest,
  LocalTrainingJobStatus,
  RlTrainingConfig,
  SftTrainingConfig,
} from "./training/job-manifest.js";
export {
  FileTrainingJobStore,
} from "./training/job-store.js";
export type {
  CompleteTrainingJobParams,
  CreateTrainingJobParams,
  FailTrainingJobParams,
  TrainingJobStoreShape,
} from "./training/job-store.js";
