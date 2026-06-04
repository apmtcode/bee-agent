import type { OperatorEvent, OperatorEventFilter } from "../kernel/event-bus.js";
import type { OperatorRunProgressEvent, StandaloneOperatorRuntime } from "../orchestrator/operator-runtime.js";

export type RuntimeEventFamily = "run" | "approval" | "background-task" | "monitor" | "skill" | "subagent" | "task" | "plan" | "message" | "notification";

export type RuntimeEventSubscriptionOptions = {
  replay?: boolean;
  filter?: OperatorEventFilter<OperatorEvent>;
  sessionId?: string;
  runId?: string;
  family?: RuntimeEventFamily;
};

export function subscribeRuntimeEvents(
  runtime: StandaloneOperatorRuntime,
  options: RuntimeEventSubscriptionOptions = {},
): AsyncIterable<OperatorEvent> {
  return runtime.events.stream(buildRuntimeEventFilter(options), { replay: options.replay });
}

export function buildRuntimeEventFilter(options: RuntimeEventSubscriptionOptions = {}): OperatorEventFilter<OperatorEvent> | undefined {
  const filters: Array<OperatorEventFilter<OperatorEvent>> = [];
  if (options.filter) {
    filters.push(options.filter);
  }
  if (options.sessionId) {
    filters.push((event) => getEventSessionIds(event).includes(options.sessionId!));
  }
  if (options.runId) {
    filters.push((event) => getEventRunId(event) === options.runId);
  }
  if (options.family) {
    filters.push((event) => matchesFamily(event, options.family ?? "run"));
  }
  if (filters.length === 0) {
    return undefined;
  }
  return (event) => filters.every((filter) => filter(event));
}

function matchesFamily(event: OperatorEvent, family: RuntimeEventFamily): boolean {
  const type = event.type;
  if (family === "run") {
    return type === "run.started" || type === "run.updated" || type === "run.progress";
  }
  if (family === "approval") {
    return type.startsWith("approval.");
  }
  if (family === "background-task") {
    return type.startsWith("background-task.");
  }
  if (family === "monitor") {
    return type.startsWith("background-task.") && getBackgroundTaskKind(event) === "monitor";
  }
  if (family === "skill") {
    return type.startsWith("skill.");
  }
  if (family === "task") {
    return type.startsWith("task.");
  }
  if (family === "plan") {
    return type.startsWith("plan.");
  }
  if (family === "message") {
    return type.startsWith("message.");
  }
  if (family === "notification") {
    return type.startsWith("notification.");
  }
  return type.startsWith("subagent.");
}

function getBackgroundTaskKind(event: OperatorEvent): string | undefined {
  const payload = event.payload;
  if (!payload || typeof payload !== "object") {
    return undefined;
  }
  const kind = (payload as { kind?: unknown }).kind;
  return typeof kind === "string" ? kind : undefined;
}

function getEventSessionIds(event: OperatorEvent): string[] {
  const payload = event.payload;
  if (!payload || typeof payload !== "object") {
    return [];
  }

  const sessionIds = new Set<string>();
  if (typeof (payload as { sessionId?: unknown }).sessionId === "string") {
    sessionIds.add((payload as { sessionId: string }).sessionId);
  }
  if (event.type === "run.progress") {
    sessionIds.add((payload as OperatorRunProgressEvent).sessionId);
  }
  if (typeof (payload as { fromSessionId?: unknown }).fromSessionId === "string") {
    sessionIds.add((payload as { fromSessionId: string }).fromSessionId);
  }
  if (typeof (payload as { toSessionId?: unknown }).toSessionId === "string") {
    sessionIds.add((payload as { toSessionId: string }).toSessionId);
  }
  return [...sessionIds];
}

function getEventRunId(event: OperatorEvent): string | undefined {
  const payload = event.payload;
  if (!payload || typeof payload !== "object") {
    return undefined;
  }
  if (typeof (payload as { id?: unknown }).id === "string" && event.type.startsWith("run.")) {
    return (payload as { id: string }).id;
  }
  if (typeof (payload as { runId?: unknown }).runId === "string") {
    return (payload as { runId: string }).runId;
  }
  return undefined;
}
