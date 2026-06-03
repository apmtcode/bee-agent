import type { OperatorEvent, OperatorEventFilter } from "../kernel/event-bus.js";
import type { OperatorRunProgressEvent, StandaloneOperatorRuntime } from "../orchestrator/operator-runtime.js";

export type RuntimeEventFamily = "run" | "approval" | "background-task" | "skill" | "subagent" | "task";

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
    filters.push((event) => getEventSessionId(event) === options.sessionId);
  }
  if (options.runId) {
    filters.push((event) => getEventRunId(event) === options.runId);
  }
  if (options.family) {
    filters.push((event) => matchesFamily(event.type, options.family ?? "run"));
  }
  if (filters.length === 0) {
    return undefined;
  }
  return (event) => filters.every((filter) => filter(event));
}

function matchesFamily(type: string, family: RuntimeEventFamily): boolean {
  if (family === "run") {
    return type === "run.started" || type === "run.updated" || type === "run.progress";
  }
  if (family === "approval") {
    return type.startsWith("approval.");
  }
  if (family === "background-task") {
    return type.startsWith("background-task.");
  }
  if (family === "skill") {
    return type.startsWith("skill.");
  }
  if (family === "task") {
    return type.startsWith("task.");
  }
  return type.startsWith("subagent.");
}

function getEventSessionId(event: OperatorEvent): string | undefined {
  const payload = event.payload;
  if (!payload || typeof payload !== "object") {
    return undefined;
  }
  if (typeof (payload as { sessionId?: unknown }).sessionId === "string") {
    return (payload as { sessionId: string }).sessionId;
  }
  if (event.type === "run.progress") {
    return (payload as OperatorRunProgressEvent).sessionId;
  }
  return undefined;
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
