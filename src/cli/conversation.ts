import { resolveOperatorCliExecutionConfig, type OperatorCliRuntimeConfig } from "./config.js";
import { discoverPromptContext, type OperatorCliPromptContext } from "./prompt.js";
import type { OperatorRunProgressPhase, StandaloneOperatorRuntime } from "../orchestrator/operator-runtime.js";
import type { TranscriptMessageRecord, TranscriptRecord } from "../harness/transcript-store.js";

export type OperatorCliConversationTurnParams = {
  runtime: StandaloneOperatorRuntime;
  sessionId: string;
  cwd: string;
  currentDate: string;
  input: string;
  config?: OperatorCliRuntimeConfig;
  approvedCommandFingerprints?: Set<string>;
};

export type OperatorCliConversationTurnResult = {
  assistantText: string;
  trajectorySummary: string;
  promptContext: OperatorCliPromptContext;
  runId: string;
};

type ConversationIntent =
  | { kind: "greeting" }
  | { kind: "help" }
  | { kind: "status" }
  | { kind: "last-user" }
  | { kind: "last-assistant" }
  | { kind: "session-summary" }
  | { kind: "recall"; query: string }
  | { kind: "skills-list" }
  | { kind: "run-skill"; skillId: string }
  | { kind: "watch-run"; runId?: string }
  | { kind: "watch-task"; taskId?: string }
  | { kind: "watch-active" }
  | { kind: "approvals" }
  | { kind: "approve"; approvalId: string }
  | { kind: "deny"; approvalId: string }
  | { kind: "background-list" }
  | { kind: "background-start"; title: string; command: string }
  | { kind: "background-view"; taskId: string; lines?: number }
  | { kind: "background-sync"; taskId: string }
  | { kind: "background-cancel"; taskId: string }
  | { kind: "training" }
  | { kind: "config" }
  | { kind: "prompt" }
  | { kind: "fallback" };

export async function runOperatorCliConversationTurn(
  params: OperatorCliConversationTurnParams,
): Promise<OperatorCliConversationTurnResult> {
  const promptContext = await discoverPromptContext({
    cwd: params.cwd,
    currentDate: params.currentDate,
    config: params.config,
  });
  const transcript = await params.runtime.getTranscript(params.sessionId);
  const history = transcript
    .filter(isTranscriptMessageRecord)
    .map((record) => record.message);
  const intent = parseConversationIntent(params.input);
  const run = await params.runtime.startRun({
    sessionId: params.sessionId,
    title: buildRunTitle(params.input),
    metadata: {
      kind: "cli.freeform",
      input: params.input,
      intent: intent.kind,
    },
  });

  try {
    publishProgress(params.runtime, {
      runId: run.id,
      sessionId: params.sessionId,
      phase: "planning",
      message: `Handling freeform ${intent.kind} request.`,
    });
    const assistantText = await respondToIntent({
      runtime: params.runtime,
      sessionId: params.sessionId,
      runId: run.id,
      cwd: params.cwd,
      input: params.input,
      config: params.config,
      promptContext,
      history,
      intent,
      approvedCommandFingerprints: params.approvedCommandFingerprints,
    });
    const trajectorySummary = buildTrajectorySummary(intent, params.input, assistantText);
    publishProgress(params.runtime, {
      runId: run.id,
      sessionId: params.sessionId,
      phase: "completed",
      message: "Freeform turn completed.",
    });
    await params.runtime.updateRunStatus(run.id, "completed", {
      kind: "cli.freeform",
      input: params.input,
      intent: intent.kind,
      trajectorySummary,
      assistantText,
    });
    return {
      assistantText,
      trajectorySummary,
      promptContext,
      runId: run.id,
    };
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    publishProgress(params.runtime, {
      runId: run.id,
      sessionId: params.sessionId,
      phase: "failed",
      message: `Freeform turn failed: ${message}`,
    });
    await params.runtime.updateRunStatus(run.id, "failed", {
      kind: "cli.freeform",
      input: params.input,
      intent: intent.kind,
      error: message,
    });
    return {
      assistantText: `I hit an error while handling that turn: ${message}`,
      trajectorySummary: `Freeform turn failed: ${params.input}`,
      promptContext,
      runId: run.id,
    };
  }
}

async function respondToIntent(params: {
  runtime: StandaloneOperatorRuntime;
  sessionId: string;
  runId: string;
  cwd: string;
  input: string;
  config?: OperatorCliRuntimeConfig;
  promptContext: OperatorCliPromptContext;
  history: TranscriptMessageRecord[];
  intent: ConversationIntent;
  approvedCommandFingerprints?: Set<string>;
}): Promise<string> {
  switch (params.intent.kind) {
    case "greeting":
      publishProgress(params.runtime, { runId: params.runId, sessionId: params.sessionId, phase: "planning", message: "Preparing greeting." });
      return renderGreeting(params.promptContext, params.history);
    case "help":
      publishProgress(params.runtime, { runId: params.runId, sessionId: params.sessionId, phase: "planning", message: "Preparing help response." });
      return renderHelp();
    case "status":
      publishProgress(params.runtime, { runId: params.runId, sessionId: params.sessionId, phase: "status", message: "Collecting session status." });
      return await renderStatus(params.runtime, params.sessionId, params.promptContext);
    case "last-user":
      publishProgress(params.runtime, { runId: params.runId, sessionId: params.sessionId, phase: "session-summary", message: "Looking up the last user message." });
      return renderPreviousMessage(params.history, "user", "You have not sent an earlier message in this session yet.");
    case "last-assistant":
      publishProgress(params.runtime, { runId: params.runId, sessionId: params.sessionId, phase: "session-summary", message: "Looking up the last assistant message." });
      return renderPreviousMessage(params.history, "assistant", "I have not responded earlier in this session yet.");
    case "session-summary":
      publishProgress(params.runtime, { runId: params.runId, sessionId: params.sessionId, phase: "session-summary", message: "Summarizing the current session." });
      return renderSessionSummary(params.promptContext, params.history);
    case "recall":
      publishProgress(params.runtime, { runId: params.runId, sessionId: params.sessionId, phase: "recall", message: `Searching recall for \"${params.intent.query}\".` });
      return await renderRecall(params.runtime, params.intent.query);
    case "skills-list":
      publishProgress(params.runtime, { runId: params.runId, sessionId: params.sessionId, phase: "executing-skill", message: "Listing available skills." });
      return await renderSkills(params.runtime);
    case "run-skill":
      publishProgress(params.runtime, { runId: params.runId, sessionId: params.sessionId, phase: "executing-skill", message: `Running skill ${params.intent.skillId}.`, relatedSkillId: params.intent.skillId });
      return await renderExecutableSkillRun({
        runtime: params.runtime,
        sessionId: params.sessionId,
        runId: params.runId,
        skillId: params.intent.skillId,
        config: params.config,
        approvedCommandFingerprints: params.approvedCommandFingerprints,
      });
    case "watch-run":
      publishProgress(params.runtime, { runId: params.runId, sessionId: params.sessionId, phase: "planning", message: `Watching run ${params.intent.runId ?? "active"}.` });
      return await renderWatchRun(params.runtime, params.sessionId, params.intent.runId);
    case "watch-task":
      publishProgress(params.runtime, { runId: params.runId, sessionId: params.sessionId, phase: "planning", message: `Watching background task ${params.intent.taskId ?? "active"}.` });
      return await renderWatchTask(params.runtime, params.sessionId, params.intent.taskId);
    case "watch-active":
      publishProgress(params.runtime, { runId: params.runId, sessionId: params.sessionId, phase: "planning", message: "Watching active work." });
      return await renderWatchActive(params.runtime, params.sessionId);
    case "approvals":
      publishProgress(params.runtime, { runId: params.runId, sessionId: params.sessionId, phase: "approvals", message: "Listing pending approvals." });
      return await renderApprovals(params.runtime, params.sessionId);
    case "approve":
      publishProgress(params.runtime, { runId: params.runId, sessionId: params.sessionId, phase: "approval-resolution", message: `Resolving approval ${params.intent.approvalId}.`, relatedApprovalId: params.intent.approvalId });
      return await renderApprovalResolution(
        params.runtime,
        params.sessionId,
        params.runId,
        params.intent.approvalId,
        "approved",
        params.approvedCommandFingerprints,
      );
    case "deny":
      publishProgress(params.runtime, { runId: params.runId, sessionId: params.sessionId, phase: "approval-resolution", message: `Resolving approval ${params.intent.approvalId}.`, relatedApprovalId: params.intent.approvalId });
      return await renderApprovalResolution(
        params.runtime,
        params.sessionId,
        params.runId,
        params.intent.approvalId,
        "denied",
        params.approvedCommandFingerprints,
      );
    case "background-list":
      publishProgress(params.runtime, { runId: params.runId, sessionId: params.sessionId, phase: "starting-background-task", message: "Listing background tasks." });
      return await renderBackgroundTasks(params.runtime, params.sessionId);
    case "background-start":
      publishProgress(params.runtime, { runId: params.runId, sessionId: params.sessionId, phase: "starting-background-task", message: `Starting background task ${params.intent.title}.` });
      return await renderBackgroundStart({
        runtime: params.runtime,
        sessionId: params.sessionId,
        runId: params.runId,
        cwd: params.cwd,
        title: params.intent.title,
        command: params.intent.command,
        config: params.config,
        approvedCommandFingerprints: params.approvedCommandFingerprints,
      });
    case "background-view":
      publishProgress(params.runtime, { runId: params.runId, sessionId: params.sessionId, phase: "syncing-background-task", message: `Reading background task ${params.intent.taskId}.`, relatedTaskId: params.intent.taskId });
      return await renderBackgroundView(params.runtime, params.sessionId, params.runId, params.intent.taskId, params.intent.lines);
    case "background-sync":
      publishProgress(params.runtime, { runId: params.runId, sessionId: params.sessionId, phase: "syncing-background-task", message: `Syncing background task ${params.intent.taskId}.`, relatedTaskId: params.intent.taskId });
      return await renderBackgroundSync(params.runtime, params.sessionId, params.runId, params.intent.taskId);
    case "background-cancel":
      publishProgress(params.runtime, { runId: params.runId, sessionId: params.sessionId, phase: "cancelling-background-task", message: `Cancelling background task ${params.intent.taskId}.`, relatedTaskId: params.intent.taskId });
      return await renderBackgroundCancel(params.runtime, params.sessionId, params.runId, params.intent.taskId);
    case "training":
      publishProgress(params.runtime, { runId: params.runId, sessionId: params.sessionId, phase: "training", message: "Listing training jobs." });
      return await renderTrainingJobs(params.runtime);
    case "config":
      publishProgress(params.runtime, { runId: params.runId, sessionId: params.sessionId, phase: "config", message: "Inspecting CLI config." });
      return renderConfig(params.promptContext.config);
    case "prompt":
      publishProgress(params.runtime, { runId: params.runId, sessionId: params.sessionId, phase: "prompt", message: "Inspecting prompt context." });
      return renderPromptContext(params.promptContext);
    case "fallback":
      publishProgress(params.runtime, { runId: params.runId, sessionId: params.sessionId, phase: "planning", message: "Preparing contextual fallback response." });
      return renderFallback(params.promptContext, params.history, params.input);
  }
}

function parseConversationIntent(input: string): ConversationIntent {
  const trimmed = input.trim();
  if (!trimmed) {
    return { kind: "fallback" };
  }
  const normalized = trimmed.toLowerCase();

  const runSkillMatch = trimmed.match(/^(?:run|execute)\s+skill\s+(.+)$/i);
  if (runSkillMatch?.[1]) {
    return { kind: "run-skill", skillId: runSkillMatch[1].trim() };
  }

  const watchRunMatch = trimmed.match(/^(?:watch|follow)\s+run(?:\s+(.+))?$/i);
  if (watchRunMatch) {
    return watchRunMatch[1]?.trim() ? { kind: "watch-run", runId: watchRunMatch[1].trim() } : { kind: "watch-run" };
  }

  const watchTaskMatch = trimmed.match(/^(?:watch|follow)\s+(?:task|background)(?:\s+(.+))?$/i);
  if (watchTaskMatch) {
    return watchTaskMatch[1]?.trim() ? { kind: "watch-task", taskId: watchTaskMatch[1].trim() } : { kind: "watch-task" };
  }

  if (/^(?:watch|follow)\s+active(?:\s+(?:run|task|background))?$/i.test(trimmed)) {
    return { kind: "watch-active" };
  }

  const approveMatch = trimmed.match(/^approve\s+([\w-]+)$/i);
  if (approveMatch?.[1]) {
    return { kind: "approve", approvalId: approveMatch[1] };
  }

  const denyMatch = trimmed.match(/^deny\s+([\w-]+)$/i);
  if (denyMatch?.[1]) {
    return { kind: "deny", approvalId: denyMatch[1] };
  }

  const recallMatch = trimmed.match(/^(?:recall|remember|search\s+(?:memory|recall)\s+for)\s+(.+)$/i);
  if (recallMatch?.[1]) {
    return { kind: "recall", query: recallMatch[1].trim() };
  }

  const backgroundStartMatch = trimmed.match(/^(?:background\s+start|start\s+background)\s+(.+?)\s+--\s+(.+)$/i);
  if (backgroundStartMatch?.[1] && backgroundStartMatch?.[2]) {
    return {
      kind: "background-start",
      title: backgroundStartMatch[1].trim(),
      command: backgroundStartMatch[2].trim(),
    };
  }

  const backgroundViewMatch = trimmed.match(/^(?:background\s+view|view\s+background)\s+(\S+)(?:\s+(\d+))?$/i);
  if (backgroundViewMatch?.[1]) {
    return {
      kind: "background-view",
      taskId: backgroundViewMatch[1],
      ...(backgroundViewMatch[2] ? { lines: Number.parseInt(backgroundViewMatch[2], 10) } : {}),
    };
  }

  const backgroundSyncMatch = trimmed.match(/^(?:background\s+sync|sync\s+background)\s+(\S+)$/i);
  if (backgroundSyncMatch?.[1]) {
    return { kind: "background-sync", taskId: backgroundSyncMatch[1] };
  }

  const backgroundCancelMatch = trimmed.match(/^(?:background\s+cancel|cancel\s+background)\s+(\S+)$/i);
  if (backgroundCancelMatch?.[1]) {
    return { kind: "background-cancel", taskId: backgroundCancelMatch[1] };
  }

  if (/^(?:background|background\s+list|list\s+background(?:\s+tasks?)?|show\s+background(?:\s+tasks?)?)$/i.test(trimmed)) {
    return { kind: "background-list" };
  }

  if (/^(?:training|list\s+training(?:\s+jobs?)?|show\s+training(?:\s+jobs?)?)$/i.test(trimmed)) {
    return { kind: "training" };
  }

  if (/^(?:skills|list\s+skills|show\s+skills)$/i.test(trimmed)) {
    return { kind: "skills-list" };
  }

  if (/^(?:approvals|list\s+approvals|show\s+approvals)$/i.test(trimmed)) {
    return { kind: "approvals" };
  }

  if (/^(?:config|show\s+config)$/i.test(trimmed)) {
    return { kind: "config" };
  }

  if (/^(?:prompt|show\s+prompt(?:\s+context)?)$/i.test(trimmed)) {
    return { kind: "prompt" };
  }

  if (/^(?:status|session\s+status|where\s+am\s+i|what'?s\s+the\s+status)$/i.test(trimmed)) {
    return { kind: "status" };
  }

  if (/^(?:hello|hi|hey|hello operator|hi operator|hey operator)$/.test(normalized)) {
    return { kind: "greeting" };
  }

  if (
    normalized.includes("help")
    || normalized.includes("what can you do")
    || normalized.includes("how can you help")
  ) {
    return { kind: "help" };
  }

  if (
    normalized.includes("what did i just say")
    || normalized.includes("what did i just ask")
    || normalized.includes("my last message")
  ) {
    return { kind: "last-user" };
  }

  if (
    normalized.includes("what did you just say")
    || normalized.includes("your last response")
    || normalized.includes("repeat your last answer")
  ) {
    return { kind: "last-assistant" };
  }

  if (
    normalized.includes("summarize this session")
    || normalized.includes("summarize our conversation")
    || normalized.includes("what have we talked about")
    || normalized.includes("what are you working on")
  ) {
    return { kind: "session-summary" };
  }

  return { kind: "fallback" };
}

async function renderStatus(
  runtime: StandaloneOperatorRuntime,
  sessionId: string,
  promptContext: OperatorCliPromptContext,
): Promise<string> {
  const session = await runtime.getSession(sessionId);
  const latestAssistant = await runtime.getLatestAssistantText(sessionId);
  return [
    `Current session: ${session?.id ?? sessionId}`,
    `Status: ${session?.status ?? "unknown"}`,
    `Working directory: ${promptContext.project.cwd}`,
    `Loaded config files: ${promptContext.config?.loadedEntries.length ?? 0}`,
    `Loaded instruction files: ${promptContext.project.instructionFiles.length}`,
    `Latest assistant message: ${latestAssistant ?? "<none>"}`,
  ].join("\n");
}

function renderGreeting(promptContext: OperatorCliPromptContext, history: TranscriptMessageRecord[]): string {
  return [
    `Hello. I'm ready in ${promptContext.project.cwd}.`,
    history.length === 0
      ? "This session does not have earlier turns yet."
      : `This session already has ${history.length} stored messages I can use for context.`,
    "You can ask for status, a session summary, recall results, skills, approvals, background tasks, training jobs, config, or prompt context.",
  ].join("\n");
}

function renderHelp(): string {
  return [
    "I can handle freeform local CLI turns for session context, transcript summaries, recall, skills, approvals, background tasks, training jobs, config, and prompt context.",
    "You can also use structured requests like 'status', 'summarize this session', 'recall deploy', 'skills', 'run skill <id>', or 'start background <title> -- <command>'.",
    "Slash commands still work for the full explicit shell surface.",
  ].join("\n");
}

function renderPreviousMessage(
  history: TranscriptMessageRecord[],
  role: TranscriptMessageRecord["role"],
  emptyMessage: string,
): string {
  const previous = [...history].reverse().find((message) => message.role === role);
  return previous ? previous.content : emptyMessage;
}

function renderSessionSummary(promptContext: OperatorCliPromptContext, history: TranscriptMessageRecord[]): string {
  const recent = history.slice(-6);
  return [
    `Session summary for ${promptContext.project.cwd}:`,
    recent.length === 0
      ? "No earlier conversation is stored in this session yet."
      : recent.map((message) => `- ${message.role}: ${truncateSingleLine(message.content, 200)}`).join("\n"),
  ].join("\n");
}

async function renderRecall(runtime: StandaloneOperatorRuntime, query: string): Promise<string> {
  const recall = await runtime.recall(query);
  if (recall.hits.length === 0) {
    return `No recall hits for \"${query}\".`;
  }
  return [
    `Top recall hits for \"${query}\":`,
    ...recall.hits.slice(0, 5).map((hit) => {
      if (hit.kind === "memory") {
        return `- memory ${hit.item.id}: ${hit.item.summary}`;
      }
      if (hit.kind === "skill") {
        return `- skill ${hit.skill.id}: ${hit.skill.title}`;
      }
      return `- trajectory ${hit.trajectoryId}: ${hit.summary}`;
    }),
  ].join("\n");
}

async function renderSkills(runtime: StandaloneOperatorRuntime): Promise<string> {
  const promoted = await runtime.listPromotedSkills();
  const executable = await runtime.listExecutableSkills();
  if (promoted.length === 0 && executable.length === 0) {
    return "No promoted or executable skills.";
  }
  return [
    ...(promoted.length > 0 ? ["Promoted skills:", ...promoted.map((skill) => `- ${skill.id}: ${skill.title}`)] : []),
    ...(executable.length > 0 ? ["Executable skills:", ...executable.map((skill) => `- ${skill.id}: ${skill.title}`)] : []),
  ].join("\n");
}

async function renderExecutableSkillRun(params: {
  runtime: StandaloneOperatorRuntime;
  sessionId: string;
  runId: string;
  skillId: string;
  config?: OperatorCliRuntimeConfig;
  approvedCommandFingerprints?: Set<string>;
}): Promise<string> {
  const run = await params.runtime.runExecutableSkill({
    skillId: params.skillId,
    sessionId: params.sessionId,
    parentRunId: params.runId,
    executionConfig: resolveOperatorCliExecutionConfig(params.config),
    approvedCommandFingerprints: params.approvedCommandFingerprints,
  });
  if (!run) {
    return `Unknown executable skill: ${params.skillId}`;
  }
  return [
    `Completed skill run ${run.id}.`,
    ...run.stepResults.map((step) => `- ${step.stepId}: ${step.output}`),
  ].join("\n");
}

async function renderApprovals(runtime: StandaloneOperatorRuntime, sessionId: string): Promise<string> {
  const approvals = await runtime.listApprovals();
  if (approvals.length === 0) {
    return "No pending approvals.";
  }
  return approvals
    .map((approval) => `${approval.sessionId === sessionId ? "*" : "-"} ${approval.id} ${approval.title} (${approval.summary})`)
    .join("\n");
}

async function renderWatchRun(runtime: StandaloneOperatorRuntime, sessionId: string, runId?: string): Promise<string> {
  const run = runId ? await runtime.getRun(runId) : await runtime.getActiveRun(sessionId);
  if (!run) {
    return runId ? `Unknown run: ${runId}` : `No active run for session ${sessionId}.`;
  }
  return `Watching run ${run.id}: ${run.status} ${run.title}`;
}

async function renderWatchTask(runtime: StandaloneOperatorRuntime, sessionId: string, taskId?: string): Promise<string> {
  const task = taskId ? await runtime.getBackgroundTask(taskId) : await runtime.getActiveBackgroundTask(sessionId);
  if (!task) {
    return taskId ? `Unknown background task: ${taskId}` : `No active background task for session ${sessionId}.`;
  }
  const output = await runtime.getBackgroundTaskOutput(task.id, { lineLimit: 200, offset: 0 });
  return [
    `Watching background task ${task.id}: ${task.status} ${task.title}`,
    output?.output?.trim() ? output.output : "<no output>",
  ].join("\n");
}

async function renderWatchActive(runtime: StandaloneOperatorRuntime, sessionId: string): Promise<string> {
  const run = await runtime.getActiveRun(sessionId);
  if (run) {
    return await renderWatchRun(runtime, sessionId, run.id);
  }
  const task = await runtime.getActiveBackgroundTask(sessionId);
  if (task) {
    return await renderWatchTask(runtime, sessionId, task.id);
  }
  return `No active run or background task for session ${sessionId}.`;
}

async function renderApprovalResolution(
  runtime: StandaloneOperatorRuntime,
  sessionId: string,
  runId: string,
  approvalId: string,
  decision: "approved" | "denied",
  approvedCommandFingerprints?: Set<string>,
): Promise<string> {
  const approvals = await runtime.listApprovals();
  const approval = approvals.find((item) => item.id === approvalId);
  const resolution = await runtime.resolveApproval(approvalId, decision, "operator-cli");
  if (!resolution) {
    return `Unknown approval: ${approvalId}`;
  }
  const fingerprint = typeof approval?.metadata?.fingerprint === "string" ? approval.metadata.fingerprint : undefined;
  if (decision === "approved" && fingerprint) {
    approvedCommandFingerprints?.add(fingerprint);
  }
  publishProgress(runtime, {
    runId,
    sessionId,
    phase: "approval-resolution",
    message: `${decision === "approved" ? "Approved" : "Denied"} ${approvalId}.`,
    relatedApprovalId: approvalId,
  });
  return `${decision === "approved" ? "Approved" : "Denied"} ${approvalId}.`;
}

async function renderBackgroundTasks(runtime: StandaloneOperatorRuntime, sessionId: string): Promise<string> {
  const tasks = await runtime.listBackgroundTasks(sessionId);
  if (tasks.length === 0) {
    return "No background tasks.";
  }
  return tasks.map((task) => `${task.id} ${task.status} ${task.title}`).join("\n");
}

async function renderBackgroundStart(params: {
  runtime: StandaloneOperatorRuntime;
  sessionId: string;
  runId: string;
  cwd: string;
  title: string;
  command: string;
  config?: OperatorCliRuntimeConfig;
  approvedCommandFingerprints?: Set<string>;
}): Promise<string> {
  const executionConfig = resolveOperatorCliExecutionConfig(params.config);
  const authorization = await params.runtime.authorizeCommandExecution({
    sessionId: params.sessionId,
    cwd: params.cwd,
    command: params.command,
    title: params.title,
    kind: "background.start",
    source: "cli",
    executionConfig,
    approvedFingerprints: params.approvedCommandFingerprints,
  });
  if (!authorization.allowed) {
    const pendingApprovalId = authorization.pendingApprovalId;
    publishProgress(params.runtime, {
      runId: params.runId,
      sessionId: params.sessionId,
      phase: "awaiting-approval",
      message: pendingApprovalId
        ? `Awaiting approval ${pendingApprovalId} for background task ${params.title}.`
        : authorization.message,
      ...(pendingApprovalId ? { relatedApprovalId: pendingApprovalId } : {}),
    });
    return authorization.message;
  }
  const task = await params.runtime.startBackgroundTask({
    sessionId: params.sessionId,
    title: authorization.title ?? params.title,
    command: authorization.command,
    cwd: params.cwd,
  });
  publishProgress(params.runtime, {
    runId: params.runId,
    sessionId: params.sessionId,
    phase: "starting-background-task",
    message: `Started background task ${task.id} (${task.status}) ${task.title}`,
    relatedTaskId: task.id,
  });
  const postHookResults = await params.runtime.runPostCommandHooks({
    sessionId: params.sessionId,
    cwd: params.cwd,
    command: authorization.command,
    ...(authorization.title ? { title: authorization.title } : {}),
    kind: "background.start",
    source: "cli",
    executionConfig,
  });
  return [
    `Started background task ${task.id} (${task.status}) ${task.title}`,
    ...postHookResults.map((result) => formatHookResult(result)),
  ]
    .filter(Boolean)
    .join("\n");
}

async function renderBackgroundView(
  runtime: StandaloneOperatorRuntime,
  sessionId: string,
  runId: string,
  taskId: string,
  lines?: number,
): Promise<string> {
  const task = await runtime.syncBackgroundTask(taskId);
  if (!task) {
    return `Unknown background task: ${taskId}`;
  }
  publishProgress(runtime, {
    runId,
    sessionId,
    phase: "syncing-background-task",
    message: `Background task ${task.id} is ${task.status}.`,
    relatedTaskId: task.id,
  });
  const output = await runtime.getBackgroundTaskOutput(taskId, lines ?? 50);
  return [
    `${task.id} ${task.status} ${task.title}`,
    output?.output?.trim() ? output.output : "<no output>",
  ].join("\n");
}

async function renderBackgroundSync(
  runtime: StandaloneOperatorRuntime,
  sessionId: string,
  runId: string,
  taskId: string,
): Promise<string> {
  const task = await runtime.syncBackgroundTask(taskId);
  if (task) {
    publishProgress(runtime, {
      runId,
      sessionId,
      phase: "syncing-background-task",
      message: `Background task ${task.id} is ${task.status}.`,
      relatedTaskId: task.id,
    });
  }
  return task ? `${task.id} ${task.status} ${task.title}` : `Unknown background task: ${taskId}`;
}

async function renderBackgroundCancel(
  runtime: StandaloneOperatorRuntime,
  sessionId: string,
  runId: string,
  taskId: string,
): Promise<string> {
  const task = await runtime.cancelBackgroundTask(taskId);
  if (task) {
    publishProgress(runtime, {
      runId,
      sessionId,
      phase: "cancelling-background-task",
      message: `Cancelled background task ${task.id}.`,
      relatedTaskId: task.id,
    });
  }
  return task ? `Cancelled background task ${task.id}.` : `Unknown background task: ${taskId}`;
}

async function renderTrainingJobs(runtime: StandaloneOperatorRuntime): Promise<string> {
  const jobs = await runtime.listTrainingJobs();
  if (jobs.length === 0) {
    return "No training jobs.";
  }
  return jobs.map((job) => `${job.id} ${job.status} ${job.mode}`).join("\n");
}

function renderConfig(config?: OperatorCliRuntimeConfig): string {
  const executionConfig = resolveOperatorCliExecutionConfig(config);
  return [
    ...(config?.loadedEntries.length
      ? ["Loaded config files:", ...config.loadedEntries.map((entry) => `- ${entry.source}: ${entry.path}`)]
      : ["Loaded config files: <none>"]),
    `permissionMode=${executionConfig.permissionMode}`,
    `hooks.PreToolUse=${executionConfig.hooks.PreToolUse.length}`,
    `hooks.PreCommand=${executionConfig.hooks.PreCommand.length}`,
    `hooks.PostCommand=${executionConfig.hooks.PostCommand.length}`,
    `hooks.SessionStart=${executionConfig.hooks.SessionStart.length}`,
    `hooks.SessionEnd=${executionConfig.hooks.SessionEnd.length}`,
    `hooks.ApprovalRequested=${executionConfig.hooks.ApprovalRequested.length}`,
    `hooks.ApprovalResolved=${executionConfig.hooks.ApprovalResolved.length}`,
    `unsupportedHooks=${executionConfig.unsupportedHookKeys.length === 0 ? "<none>" : executionConfig.unsupportedHookKeys.join(",")}`,
  ].join("\n");
}

function renderPromptContext(promptContext: OperatorCliPromptContext): string {
  return [
    `cwd=${promptContext.project.cwd}`,
    `date=${promptContext.project.currentDate}`,
    `gitStatus=${promptContext.project.gitStatus ?? "<none>"}`,
    ...(promptContext.project.instructionFiles.length === 0
      ? ["instructionFiles=<none>"]
      : ["instructionFiles:", ...promptContext.project.instructionFiles.map((file) => `- ${file.path}`)]),
  ].join("\n");
}

function renderFallback(
  promptContext: OperatorCliPromptContext,
  history: TranscriptMessageRecord[],
  input: string,
): string {
  const recent = history.slice(-4);
  return [
    `I received: ${truncateSingleLine(input, 200)}`,
    `Working directory: ${promptContext.project.cwd}`,
    recent.length === 0
      ? "There is no earlier session history yet."
      : [
          "Recent session context:",
          ...recent.map((message) => `- ${message.role}: ${truncateSingleLine(message.content, 160)}`),
        ].join("\n"),
    "I can help with status, session summaries, recall, approvals, skills, background tasks, training jobs, config, and prompt context.",
  ].join("\n\n");
}

function buildTrajectorySummary(intent: ConversationIntent, input: string, assistantText: string): string {
  return truncateSingleLine(`Freeform ${intent.kind}: ${input} -> ${assistantText}`, 240);
}

function publishProgress(
  runtime: StandaloneOperatorRuntime,
  params: {
    runId?: string;
    sessionId: string;
    phase: OperatorRunProgressPhase;
    message: string;
    relatedApprovalId?: string;
    relatedTaskId?: string;
    relatedSkillId?: string;
  },
): void {
  runtime.publishRunProgress(params);
}

function buildRunTitle(input: string): string {
  return `Freeform turn: ${truncateSingleLine(input, 60)}`;
}

function truncateSingleLine(value: string, limit: number): string {
  const normalized = value.replace(/\s+/g, " ").trim();
  if (normalized.length <= limit) {
    return normalized;
  }
  return `${normalized.slice(0, Math.max(0, limit - 1)).trimEnd()}…`;
}

function formatHookResult(result: { event: string; command: string; stdout: string; stderr: string }): string {
  const parts = [result.stdout, result.stderr].filter(Boolean).join(" ").trim();
  return parts ? `[${result.event}] ${parts}` : `[${result.event}] ${result.command}`;
}

function isTranscriptMessageRecord(
  record: TranscriptRecord,
): record is Extract<TranscriptRecord, { message: TranscriptMessageRecord }> {
  return "message" in record;
}
